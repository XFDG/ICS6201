#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export DEVICE="${DEVICE:-0}"
export IMGSZ="${IMGSZ:-640}"
export BATCH="${BATCH:-auto}"
export BATCH_YOLO="${BATCH_YOLO:-64}"
export BATCH_RTDETR="${BATCH_RTDETR:-48}"
export BATCH_DDW="${BATCH_DDW:-64}"
export WORKERS="${WORKERS:-16}"
export D2_IMS_PER_BATCH="${D2_IMS_PER_BATCH:-8}"
export D2_NUM_WORKERS="${D2_NUM_WORKERS:-8}"
export SEEDS="${SEEDS:-1 2 3}"
export YOLO_CACHE="${YOLO_CACHE:-disk}"

export YOLO11_MODEL="${YOLO11_MODEL:-yolo11m.pt}"
export YOLOV8_MODEL="${YOLOV8_MODEL:-yolov8n.pt}"
export YOLOV10_MODEL="${YOLOV10_MODEL:-yolov10n.pt}"
export RTDETR_MODEL="${RTDETR_MODEL:-rtdetr-l.pt}"

export EPOCHS_YOLO="${EPOCHS_YOLO:-200}"
export EPOCHS_RTDETR="${EPOCHS_RTDETR:-120}"
export EPOCHS_FASTER_RCNN="${EPOCHS_FASTER_RCNN:-120}"

python "$SCRIPT_DIR/run_pipeline.py" --mode formal --skip-ard

echo "OK: formal train finished. See $ROOT_DIR/logs/formal and $ROOT_DIR/runs_formal"
