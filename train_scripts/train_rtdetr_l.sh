#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
ensure_dataset

MODEL="${MODEL:-rtdetr-l.pt}"

for seed in "${SEEDS[@]}"; do
  name="rtdetr_l_seed${seed}"
  yolo detect train \
    model="$MODEL" \
    data="$DATA_YAML" \
    epochs="$EPOCHS" \
    imgsz="$IMGSZ" \
    batch="$BATCH" \
    cache="$YOLO_CACHE" \
    lr0="$LR0" \
    weight_decay="$WEIGHT_DECAY" \
    optimizer="$OPTIMIZER" \
    seed="$seed" \
    project="$RUNS_DIR" \
    name="$name"

  yolo detect val model="$RUNS_DIR/$name/weights/best.pt" data="$DATA_YAML" split=val project="$RUNS_DIR" name="${name}_val"
  yolo detect val model="$RUNS_DIR/$name/weights/best.pt" data="$DATA_YAML" split=test project="$RUNS_DIR" name="${name}_test"
done
