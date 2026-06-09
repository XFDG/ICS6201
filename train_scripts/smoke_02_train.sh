#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export EPOCHS_SMOKE="${EPOCHS_SMOKE:-1}"
export DEVICE="${DEVICE:-0}"
export IMGSZ="${IMGSZ:-640}"
export BATCH="${BATCH:--1}"
export WORKERS="${WORKERS:-16}"
export SEEDS="${SEEDS:-1 2 3}"
export YOLO_CACHE="${YOLO_CACHE:-disk}"

python "$SCRIPT_DIR/run_pipeline.py" --mode smoke --skip-ard

echo "OK: smoke train finished. See $ROOT_DIR/logs/smoke and $ROOT_DIR/runs_smoke"
