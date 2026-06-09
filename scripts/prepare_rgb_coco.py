import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


@dataclass(frozen=True)
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int


@dataclass(frozen=True)
class CocoAnnotation:
    id: int
    image_id: int
    category_id: int
    bbox: List[float]
    area: float
    iscrowd: int = 0


def read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    rows: List[Tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = parts
        rows.append((int(cls), float(x), float(y), float(w), float(h)))
    return rows


def yolo_to_coco_bbox(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> List[float]:
    cx = x * img_w
    cy = y * img_h
    bw = w * img_w
    bh = h * img_h
    xmin = cx - bw / 2.0
    ymin = cy - bh / 2.0
    xmin = max(0.0, min(float(img_w - 1), xmin))
    ymin = max(0.0, min(float(img_h - 1), ymin))
    bw = max(0.0, min(float(img_w), bw))
    bh = max(0.0, min(float(img_h), bh))
    return [xmin, ymin, bw, bh]


def build_split(yolo_root: Path, split: str, category_id: int, img_id_start: int, ann_id_start: int) -> Tuple[Dict, int, int]:
    images_dir = yolo_root / "images" / split
    labels_dir = yolo_root / "labels" / split

    coco_images: List[Dict] = []
    coco_annotations: List[Dict] = []

    img_id = img_id_start
    ann_id = ann_id_start

    for img_path in sorted(images_dir.glob("*.jpg")):
        with Image.open(img_path) as im:
            w, h = im.size

        coco_images.append(
            {
                "id": img_id,
                "file_name": img_path.name,
                "width": int(w),
                "height": int(h),
            }
        )

        label_path = labels_dir / f"{img_path.stem}.txt"
        for cls, x, y, bw, bh in read_yolo_labels(label_path):
            bbox = yolo_to_coco_bbox(x, y, bw, bh, int(w), int(h))
            area = float(bbox[2] * bbox[3])
            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        img_id += 1

    coco = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": category_id, "name": "drone"}],
    }
    return coco, img_id, ann_id


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--clear", action="store_true")
    args = p.parse_args()

    root = Path(args.root).resolve()
    yolo_root = root / "yolo"
    out_dir = root / "coco" / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clear:
        if (root / "coco").exists():
            shutil.rmtree(root / "coco")
        out_dir.mkdir(parents=True, exist_ok=True)

    img_id = 1
    ann_id = 1
    for split in ["train", "val", "test"]:
        coco, img_id, ann_id = build_split(yolo_root, split, category_id=1, img_id_start=img_id, ann_id_start=ann_id)
        (out_dir / f"instances_{split}.json").write_text(json.dumps(coco, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

