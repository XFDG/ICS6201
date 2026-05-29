#!/usr/bin/env python3
"""ARD-MAV 快速抽帧 — 每个视频一次性导出全部帧，然后按 XML 标注挑选。

对比原方案：逐帧调用 ffmpeg（107K 次，18 天）
本方案：每个视频一次 ffmpeg 全量导出（60 次，10-20 分钟）
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class VocObject:
    xmin: int; ymin: int; xmax: int; ymax: int


@dataclass(frozen=True)
class VocAnnotation:
    width: int; height: int; objects: List[VocObject]


def parse_voc_xml(xml_bytes: bytes) -> Optional[VocAnnotation]:
    try:
        root = ET.fromstring(xml_bytes)
        size = root.find("size")
        if size is None:
            return None
        width = int(size.findtext("width", "0"))
        height = int(size.findtext("height", "0"))
        if width <= 0 or height <= 0:
            return None
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
    except Exception:
        return None


def voc_to_yolo_lines(ann: VocAnnotation, class_id: int = 0) -> List[str]:
    lines = []
    w, h = float(ann.width), float(ann.height)
    for o in ann.objects:
        x = ((o.xmin + o.xmax) / 2.0) / w
        y = ((o.ymin + o.ymax) / 2.0) / h
        bw = (o.xmax - o.xmin) / w
        bh = (o.ymax - o.ymin) / h
        lines.append(f"{class_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")
    return lines


def split_list(items: List[str], ratios: Tuple[float, float, float], seed: int):
    train_r, val_r, test_r = ratios
    s = train_r + val_r + test_r
    train_r, val_r, test_r = train_r / s, val_r / s, test_r / s
    rng = random.Random(seed)
    xs = items[:]
    rng.shuffle(xs)
    n = len(xs)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    return xs[:n_train], xs[n_train:n_train + n_val], xs[n_train + n_val:]


def yolo_paths(root: Path, split: str, stem: str):
    img_path = root / "images" / split / f"{stem}.jpg"
    lbl_path = root / "labels" / split / f"{stem}.txt"
    return img_path, lbl_path


def extract_video_all_frames(video_path: Path, out_dir: Path) -> int:
    """一次性导出视频全部帧到 out_dir。返回导出的帧数。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error", "-y",
        "-vsync", "0",
        "-i", str(video_path),
        "-q:v", "2",
        "-threads", "0",
        str(out_dir / "frame_%04d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob("frame_*.jpg")))


def process_ard_mav_fast(
    z_ard: Path,
    out_root: Path,
    cache_dir: Path,
    seed: int,
    split_ratios: Tuple[float, float, float],
    frame_index_base: int,
    limit_total: int = 0,
) -> Dict[str, int]:
    """快速 ARD-MAV 处理：批量抽帧而非逐帧。"""

    counts = {"train": 0, "val": 0, "test": 0}
    cache_videos = cache_dir / "ard_mav" / "videos"
    cache_videos.mkdir(parents=True, exist_ok=True)

    # 临时目录用于存放单视频全部帧
    tmp_root = cache_dir / "ard_mav" / "tmp_frames"
    tmp_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(z_ard) as z:
        names = set(z.namelist())
        xml_members = [n for n in z.namelist()
                       if n.startswith("ARD-MAV/Annotations/") and n.lower().endswith(".xml")]
        stems = [Path(m).stem for m in xml_members]

        train_stems, val_stems, test_stems = split_list(stems, split_ratios, seed)
        split_map: Dict[str, str] = {}
        for s in train_stems:
            split_map[s] = "train"
        for s in val_stems:
            split_map[s] = "val"
        for s in test_stems:
            split_map[s] = "test"

        # 按视频分组 XML
        by_video: Dict[str, List[Tuple[str, str]]] = {}
        for xml_member in xml_members:
            stem = Path(xml_member).stem
            split = split_map.get(stem)
            if split is None:
                continue
            parts = stem.split("_")
            if len(parts) < 2:
                continue
            video_name = parts[0]
            by_video.setdefault(video_name, []).append((split, stem, xml_member))

        total_processed = 0
        for vi, (video_name, xml_entries) in enumerate(sorted(by_video.items()), 1):
            if limit_total > 0 and total_processed >= limit_total:
                break

            video_member = f"ARD-MAV/videos/{video_name}.mp4"
            if video_member not in names:
                print(f"  [WARN] video not in zip: {video_member}")
                continue

            video_path = cache_videos / f"{video_name}.mp4"
            if not video_path.exists():
                print(f"  [{vi}/{len(by_video)}] Extracting {video_name}.mp4 from zip...")
                with z.open(video_member) as vf:
                    video_path.write_bytes(vf.read())

            # 一次性导出全部帧
            video_tmp = tmp_root / video_name
            print(f"  [{vi}/{len(by_video)}] Decoding all frames from {video_name}.mp4...")
            try:
                n_frames = extract_video_all_frames(video_path, video_tmp)
            except subprocess.CalledProcessError as e:
                print(f"  [ERROR] ffmpeg failed for {video_name}: {e}")
                continue

            # 按帧号匹配 XML
            matched = 0
            for split, stem, xml_member in xml_entries:
                if limit_total > 0 and total_processed >= limit_total:
                    break
                parts = stem.split("_")
                frame_idx = int(parts[1])
                frame_0 = frame_idx - frame_index_base
                if frame_0 < 0:
                    continue

                # ffmpeg 输出文件名：frame_0001.jpg = 第 1 帧 (frame_index_base=1)
                frame_file = video_tmp / f"frame_{frame_idx:04d}.jpg"
                if not frame_file.exists():
                    continue

                xml_bytes = z.read(xml_member)
                ann = parse_voc_xml(xml_bytes)
                if ann is None:
                    continue
                yolo_lines = voc_to_yolo_lines(ann, class_id=0)

                out_stem = f"ardmav_{stem}"
                img_path, lbl_path = yolo_paths(out_root, split, out_stem)

                # shutil.copy 或直接移动（处理完即删）
                img_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(frame_file), str(img_path))

                lbl_path.parent.mkdir(parents=True, exist_ok=True)
                lbl_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

                counts[split] += 1
                total_processed += 1
                matched += 1

            print(f"    {matched} frames matched, {n_frames} total decoded")

            # 清理临时帧（理论上已 move，但可能有剩余未匹配帧）
            if video_tmp.exists():
                shutil.rmtree(video_tmp, ignore_errors=True)

    # 清理临时目录
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)

    return counts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clear", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    root = Path(args.root).resolve()
    raw = root / "raw_zips"
    out_root = root / "yolo"
    cache_dir = root / "cache"

    import yaml
    cfg_path = root / "configs" / "prepare_rgb.yaml"
    cfg = {}
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    ard_split = cfg.get("ard_mav", {}).get("split", {}) or {}
    ard_ratios = (
        float(ard_split.get("train_ratio", 0.8)),
        float(ard_split.get("val_ratio", 0.1)),
        float(ard_split.get("test_ratio", 0.1)),
    )
    frame_index_base = int(cfg.get("ard_mav", {}).get("frame_index_base", 1))

    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("[ARD-MAV Fast] Processing (batch frame extraction)...")
    import time
    t0 = time.time()
    counts = process_ard_mav_fast(
        raw / "ARD-MAV_Glad.zip",
        out_root,
        cache_dir=cache_dir,
        seed=args.seed,
        split_ratios=ard_ratios,
        frame_index_base=frame_index_base,
        limit_total=args.limit,
    )
    dt = time.time() - t0
    print(f"[ARD-MAV Fast] Done in {dt:.0f}s ({dt/60:.1f} min): train={counts['train']}, val={counts['val']}, test={counts['test']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
