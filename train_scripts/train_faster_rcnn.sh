#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
ensure_dataset

D2_IMS_PER_BATCH="${D2_IMS_PER_BATCH:-4}"
D2_NUM_WORKERS="${D2_NUM_WORKERS:-4}"

python "$ROOT_DIR/scripts/prepare_rgb_coco.py" --root "$ROOT_DIR"

for seed in "${SEEDS[@]}"; do
  python "$ROOT_DIR/scripts/train_detectron2_fasterrcnn.py" \
    --root "$ROOT_DIR" \
    --seed "$seed" \
    --epochs "$EPOCHS" \
    --ims-per-batch "$D2_IMS_PER_BATCH" \
    --num-workers "$D2_NUM_WORKERS"
done

