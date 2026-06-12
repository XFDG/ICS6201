# ICS6201 Snapshot Archive — 2026-06-12

This folder is a non-destructive snapshot for temporary Git staging. It copies
the recovery scripts, current reports, manifests, and status snapshots that were
used during the 2026-06 training recovery.

## Current Status

Snapshot time: 2026-06-12 08:58 CST.

- Formal tasks: 15 / 18 complete.
- Primary tasks complete:
  - RT-DETR-L seed1/2/3.
  - Faster R-CNN Detectron2 seed1/2/3.
- Secondary tasks complete:
  - YOLO11 seed1/2/3.
  - YOLOv10 seed1/2/3.
  - YOLOv8 seed1/2/3.
- Remaining:
  - `ddw_yolo_seed1`: running at epoch 168/199.
  - `ddw_yolo_seed2`: running at epoch 167/199.
  - `ddw_yolo_seed3`: interrupted at epoch 142/199 after NaN train loss.

## Folder Contents

- `scripts/`: copies of modified training/recovery scripts.
- `docs/`: copies of ICS6201 summary documents and status figures.
- `log_snapshots/`: copied manifest/state JSON files. These are snapshots only;
  the live training process still uses the original files under `logs/`.
- `reports/`: copied report files present at snapshot time.

## Important Notes

- Do not treat files in this archive as the active training source.
- Do not move live files under `logs/` while DDW-YOLO seed1/seed2 are running.
- Large model artifacts remain under `runs_formal/` and `runs_detectron2/`.
  They are not duplicated into this snapshot archive.
