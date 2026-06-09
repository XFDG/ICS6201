#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

python "$ROOT_DIR/scripts/prepare_rgb_yolo.py" --root "$ROOT_DIR" --clear --skip-ard
python "$ROOT_DIR/scripts/write_data_yaml.py" --root "$ROOT_DIR" --out "$ROOT_DIR/configs/drone_rgb_abs.yaml"
python "$ROOT_DIR/scripts/prepare_rgb_coco.py" --root "$ROOT_DIR" --clear

echo "OK: dataset prepared (skip ARD)"
