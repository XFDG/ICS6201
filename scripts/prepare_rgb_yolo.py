import argparse
import json
import random
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class VocObject:
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass(frozen=True)
class VocAnnotation:
    width: int
    height: int
    objects: List[VocObject]


def parse_voc_xml(xml_bytes: bytes) -> VocAnnotation:
    root = ET.fromstring(xml_bytes)

    size = root.find("size")
    if size is None:
        raise ValueError("VOC xml missing <size>")

    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size in VOC xml: width={width}, height={height}")

    objects: List[VocObject] = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        xmin = int(float(bnd.findtext("xmin", "0")))
        ymin = int(float(bnd.findtext("ymin", "0")))
        xmax = int(float(bnd.findtext("xmax", "0")))
        ymax = int(float(bnd.findtext("ymax", "0")))

        xmin = max(0, min(xmin, width - 1))
        ymin = max(0, min(ymin, height - 1))
        xmax = max(0, min(xmax, width - 1))
        ymax = max(0, min(ymax, height - 1))

        if xmax <= xmin or ymax <= ymin:
            continue

        objects.append(VocObject(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))

    return VocAnnotation(width=width, height=height, objects=objects)


def voc_to_yolo_lines(ann: VocAnnotation, class_id: int = 0) -> List[str]:
    lines: List[str] = []
    w = float(ann.width)
    h = float(ann.height)
    for o in ann.objects:
        x = ((o.xmin + o.xmax) / 2.0) / w
        y = ((o.ymin + o.ymax) / 2.0) / h
        bw = (o.xmax - o.xmin) / w
        bh = (o.ymax - o.ymin) / h
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        bw = min(max(bw, 0.0), 1.0)
        bh = min(max(bh, 0.0), 1.0)
        lines.append(f"{class_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")
    return lines


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def is_ffmpeg_available() -> bool:
    try:
        p = subprocess.run(["ffmpeg", "-version"], capture_output=True, check=False)
        return p.returncode == 0
    except FileNotFoundError:
        return False


def extract_zip_file(z: zipfile.ZipFile, member: str) -> bytes:
    with z.open(member) as f:
        return f.read()


def split_list(items: List[str], ratios: Tuple[float, float, float], seed: int) -> Tuple[List[str], List[str], List[str]]:
    train_r, val_r, test_r = ratios
    s = train_r + val_r + test_r
    train_r, val_r, test_r = train_r / s, val_r / s, test_r / s

    rng = random.Random(seed)
    xs = items[:]
    rng.shuffle(xs)
    n = len(xs)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    train = xs[:n_train]
    val = xs[n_train : n_train + n_val]
    test = xs[n_train + n_val :]
    return train, val, test


def yolo_paths(root: Path, split: str, stem: str, img_ext: str = ".jpg") -> Tuple[Path, Path]:
    img_path = root / "images" / split / f"{stem}{img_ext}"
    lbl_path = root / "labels" / split / f"{stem}.txt"
    return img_path, lbl_path


def process_dut_detection(
    z_train: Path,
    z_val: Path,
    z_test: Path,
    out_root: Path,
    limit_per_split: int = 0,
) -> Dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for split, zp in [("train", z_train), ("val", z_val), ("test", z_test)]:
        with zipfile.ZipFile(zp) as z:
            names = set(z.namelist())
            xml_members = [n for n in z.namelist() if n.startswith(f"{split}/xml/") and n.lower().endswith(".xml")]
            for m in xml_members:
                if limit_per_split > 0 and counts[split] >= limit_per_split:
                    break

                stem = Path(m).stem
                img_member = f"{split}/img/{stem}.jpg"
                if img_member not in names:
                    continue

                xml_bytes = extract_zip_file(z, m)
                img_bytes = extract_zip_file(z, img_member)

                ann = parse_voc_xml(xml_bytes)
                yolo_lines = voc_to_yolo_lines(ann, class_id=0)

                out_stem = f"dut_{stem}"
                img_path, lbl_path = yolo_paths(out_root, split, out_stem, img_ext=".jpg")
                write_bytes(img_path, img_bytes)
                write_text(lbl_path, "\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
                counts[split] += 1
    return counts


def process_drone_detection_dataset(
    z_train: Path,
    z_test: Path,
    out_root: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    limit_total: int = 0,
) -> Dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}

    with zipfile.ZipFile(z_train) as z:
        xml_members = [n for n in z.namelist() if n.startswith("Drone_TrainSet_XMLs/") and n.lower().endswith(".xml")]
        stems = [Path(m).stem for m in xml_members]
        rng = random.Random(seed)
        rng.shuffle(stems)
        n = len(stems)
        n_train = int(n * train_ratio / (train_ratio + val_ratio))
        train_stems = set(stems[:n_train])

        names = set(z.namelist())
        for stem in stems:
            if limit_total > 0 and (counts["train"] + counts["val"]) >= limit_total:
                break

            img_member = f"Drone_TrainSet/{stem}.jpg"
            xml_member = f"Drone_TrainSet_XMLs/{stem}.xml"
            if img_member not in names or xml_member not in names:
                continue

            split = "train" if stem in train_stems else "val"
            xml_bytes = extract_zip_file(z, xml_member)
            img_bytes = extract_zip_file(z, img_member)

            ann = parse_voc_xml(xml_bytes)
            yolo_lines = voc_to_yolo_lines(ann, class_id=0)

            out_stem = f"dronedet_{stem}"
            img_path, lbl_path = yolo_paths(out_root, split, out_stem, img_ext=".jpg")
            write_bytes(img_path, img_bytes)
            write_text(lbl_path, "\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
            counts[split] += 1

    with zipfile.ZipFile(z_test) as z:
        names = set(z.namelist())
        xml_members = [n for n in z.namelist() if n.startswith("Drone_TestSet_XMLs/") and n.lower().endswith(".xml")]
        for xml_member in xml_members:
            if limit_total > 0 and counts["test"] >= limit_total:
                break

            stem = Path(xml_member).stem
            img_member = f"Drone_TestSet/{stem}.jpg"
            if img_member not in names:
                continue

            xml_bytes = extract_zip_file(z, xml_member)
            img_bytes = extract_zip_file(z, img_member)

            ann = parse_voc_xml(xml_bytes)
            yolo_lines = voc_to_yolo_lines(ann, class_id=0)

            out_stem = f"dronedet_{stem}"
            img_path, lbl_path = yolo_paths(out_root, "test", out_stem, img_ext=".jpg")
            write_bytes(img_path, img_bytes)
            write_text(lbl_path, "\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
            counts["test"] += 1

    return counts


def extract_ard_frame(video_path: Path, frame_number_0_based: int, out_img: Path) -> None:
    out_img.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{frame_number_0_based})",
        "-vframes",
        "1",
        str(out_img),
    ]
    subprocess.run(cmd, check=True)


def process_ard_mav(
    z_ard: Path,
    out_root: Path,
    cache_dir: Path,
    seed: int,
    split_ratios: Tuple[float, float, float],
    frame_index_base: int,
    limit_total: int = 0,
) -> Dict[str, int]:
    if not is_ffmpeg_available():
        raise RuntimeError("ffmpeg not found. Please install ffmpeg on the server to extract frames from ARD-MAV videos.")

    counts = {"train": 0, "val": 0, "test": 0}
    cache_videos = cache_dir / "ard_mav" / "videos"
    cache_videos.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(z_ard) as z:
        names = set(z.namelist())
        xml_members = [n for n in z.namelist() if n.startswith("ARD-MAV/Annotations/") and n.lower().endswith(".xml")]
        stems = [Path(m).stem for m in xml_members]

        train_stems, val_stems, test_stems = split_list(stems, split_ratios, seed)
        split_map: Dict[str, str] = {}
        for s in train_stems:
            split_map[s] = "train"
        for s in val_stems:
            split_map[s] = "val"
        for s in test_stems:
            split_map[s] = "test"

        for xml_member in xml_members:
            if limit_total > 0 and sum(counts.values()) >= limit_total:
                break

            stem = Path(xml_member).stem
            split = split_map.get(stem)
            if split is None:
                continue

            parts = stem.split("_")
            if len(parts) < 2:
                continue

            video_name = parts[0]
            frame_idx = int(parts[1])
            frame_0 = frame_idx - frame_index_base
            if frame_0 < 0:
                continue

            video_member = f"ARD-MAV/videos/{video_name}.mp4"
            if video_member not in names:
                continue

            video_path = cache_videos / f"{video_name}.mp4"
            if not video_path.exists():
                write_bytes(video_path, extract_zip_file(z, video_member))

            xml_bytes = extract_zip_file(z, xml_member)
            ann = parse_voc_xml(xml_bytes)
            yolo_lines = voc_to_yolo_lines(ann, class_id=0)

            out_stem = f"ardmav_{stem}"
            img_path, lbl_path = yolo_paths(out_root, split, out_stem, img_ext=".jpg")
            extract_ard_frame(video_path, frame_0, img_path)
            write_text(lbl_path, "\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
            counts[split] += 1

    return counts


def sample_check(out_root: Path, k: int = 30, seed: int = 42) -> List[dict]:
    rng = random.Random(seed)
    problems: List[dict] = []
    label_files: List[Path] = []
    for split in ["train", "val", "test"]:
        label_files.extend((out_root / "labels" / split).glob("*.txt"))

    if not label_files:
        return [{"error": "no labels found"}]

    picks = rng.sample(label_files, min(k, len(label_files)))
    for lbl in picks:
        split = lbl.parent.name
        stem = lbl.stem
        img = out_root / "images" / split / f"{stem}.jpg"
        if not img.exists():
            problems.append({"type": "missing_image", "label": str(lbl), "image": str(img)})
            continue

        for i, line in enumerate(lbl.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                problems.append({"type": "bad_label_format", "label": str(lbl), "line": i, "text": line})
                continue
            try:
                _, x, y, w, h = parts
                x, y, w, h = float(x), float(y), float(w), float(h)
            except Exception:
                problems.append({"type": "bad_label_number", "label": str(lbl), "line": i, "text": line})
                continue
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                problems.append({"type": "out_of_range", "label": str(lbl), "line": i, "text": line})

    return problems


def main(argv: Iterable[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clear", action="store_true")
    p.add_argument("--skip-ard", action="store_true")
    p.add_argument("--limit-dut", type=int, default=0)
    p.add_argument("--limit-dronedet", type=int, default=0)
    p.add_argument("--limit-ard", type=int, default=0)
    args = p.parse_args(list(argv))

    root = Path(args.root).resolve()
    raw = root / "raw_zips"
    out_root = root / "yolo"
    cache_dir = root / "cache"
    cfg_path = root / "configs" / "prepare_rgb.yaml"

    if args.clear:
        ensure_empty_dir(out_root)
        ensure_empty_dir(cache_dir)
    else:
        out_root.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

    cfg: Dict = {}
    if cfg_path.exists():
        try:
            import yaml  # type: ignore

            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:
            cfg = {}

    drone_train_ratio = float(cfg.get("drone_detection_dataset", {}).get("train_val_split", {}).get("train_ratio", 0.9))
    drone_val_ratio = float(cfg.get("drone_detection_dataset", {}).get("train_val_split", {}).get("val_ratio", 0.1))
    ard_split = cfg.get("ard_mav", {}).get("split", {}) or {}
    ard_ratios = (
        float(ard_split.get("train_ratio", 0.8)),
        float(ard_split.get("val_ratio", 0.1)),
        float(ard_split.get("test_ratio", 0.1)),
    )
    frame_index_base = int(cfg.get("ard_mav", {}).get("frame_index_base", 1))

    stats: Dict[str, Dict[str, int]] = {}

    stats["dut_anti_uav_detection"] = process_dut_detection(
        raw / "DUT-Anti-UAV_detection_train.zip",
        raw / "DUT-Anti-UAV_detection_val.zip",
        raw / "DUT-Anti-UAV_detection_test.zip",
        out_root,
        limit_per_split=args.limit_dut,
    )

    stats["drone_detection_dataset"] = process_drone_detection_dataset(
        raw / "DroneDetectionDataset_train.zip",
        raw / "DroneDetectionDataset_test.zip",
        out_root,
        seed=args.seed,
        train_ratio=drone_train_ratio,
        val_ratio=drone_val_ratio,
        limit_total=args.limit_dronedet,
    )

    if not args.skip_ard:
        stats["ard_mav"] = process_ard_mav(
            raw / "ARD-MAV_Glad.zip",
            out_root,
            cache_dir=cache_dir,
            seed=args.seed,
            split_ratios=ard_ratios,
            frame_index_base=frame_index_base,
            limit_total=args.limit_ard,
        )

    out_stats = {
        "root": str(root),
        "seed": args.seed,
        "splits": {"train": 0, "val": 0, "test": 0},
        "sources": stats,
    }
    for src in stats.values():
        for k in list(out_stats["splits"].keys()):
            out_stats["splits"][k] += int(src.get(k, 0))

    write_text(out_root / "stats.json", json.dumps(out_stats, ensure_ascii=False, indent=2) + "\n")

    problems = sample_check(out_root, k=30, seed=args.seed)
    if problems:
        write_text(out_root / "check_problems.json", json.dumps(problems, ensure_ascii=False, indent=2) + "\n")
        print(f"[WARN] sample check found {len(problems)} problems. See: {out_root / 'check_problems.json'}")
    else:
        pth = out_root / "check_problems.json"
        if pth.exists():
            pth.unlink()

    print("[OK] Prepared YOLO dataset")
    print(f"- images: {out_root / 'images'}")
    print(f"- labels: {out_root / 'labels'}")
    print(f"- stats:  {out_root / 'stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

