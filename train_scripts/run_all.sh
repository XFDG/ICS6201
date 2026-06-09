#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/train_yolo11n.sh"
bash "$SCRIPT_DIR/train_yolov8n.sh"
bash "$SCRIPT_DIR/train_yolov10n.sh"
bash "$SCRIPT_DIR/train_rtdetr_l.sh"
bash "$SCRIPT_DIR/train_faster_rcnn.sh"
bash "$SCRIPT_DIR/train_ddw_yolo.sh"
