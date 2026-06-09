#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/volume/yzhao04/workspace/miniconda3/envs/fly/bin/python}"
LOG_DIR="$ROOT/logs/recovery_secondary"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/launch_after_yolo11_seed2_${TS}.log"
MANIFEST="$LOG_DIR/manifest_secondary_${TS}.json"
DRY_RUN_LOG="$LOG_DIR/dry_run_secondary_${TS}.log"

{
    echo "[INFO] $(date '+%F %T %Z') waiting for yolo11_seed2 to finish"
    echo "[INFO] root=$ROOT"

    while pgrep -f "run_ultralytics_task.py .*--name yolo11_seed2" >/dev/null; do
        sleep 300
        echo "[INFO] $(date '+%F %T %Z') yolo11_seed2 still running"
    done

    echo "[INFO] $(date '+%F %T %Z') yolo11_seed2 process is gone; waiting 60s for launcher cleanup"
    sleep 60
    echo "[INFO] $(date '+%F %T %Z') building manifest"
    cd "$ROOT"

    "$PYTHON" scripts/recovery_manifest.py \
        --target secondary \
        --include-missing \
        --detectron-target-iter 2062560 \
        --out "$MANIFEST"

    "$PYTHON" - "$MANIFEST" <<'PY'
import json
import sys

manifest_path = sys.argv[1]
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

item = manifest.get("items", {}).get("yolo11_seed2")
if not item or not item.get("complete"):
    print("[ERROR] yolo11_seed2 is not complete; refusing to launch remaining secondary tasks")
    print(json.dumps(item, ensure_ascii=False, indent=2))
    raise SystemExit(2)

selected = manifest.get("selected", [])
print(f"[INFO] manifest selected {len(selected)} task(s): {', '.join(selected)}")
PY

    "$PYTHON" scripts/parallel_launcher.py \
        --mode formal \
        --data configs/drone_rgb_abs.yaml \
        --gpus 1,2,3,4,5,6,7 \
        --target secondary \
        --resume-existing \
        --skip-complete \
        --manifest "$MANIFEST" \
        --dry-run | tee "$DRY_RUN_LOG"

    echo "[INFO] $(date '+%F %T %Z') launching remaining secondary tasks"
    "$PYTHON" scripts/parallel_launcher.py \
        --mode formal \
        --data configs/drone_rgb_abs.yaml \
        --gpus 1,2,3,4,5,6,7 \
        --target secondary \
        --resume-existing \
        --skip-complete \
        --manifest "$MANIFEST" \
        --start-stagger-sec 60

    echo "[INFO] $(date '+%F %T %Z') remaining secondary launcher finished"
} 2>&1 | tee -a "$LOG_FILE"
