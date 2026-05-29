#!/usr/bin/env bash
# ==============================================================================
# ICS6201 无人机检测训练 — 一键执行脚本
#
# 流程：
#   1. 环境检查 (auto_config.py + check_env.py)
#   2. 数据准备 (如果缺失)
#   3. 冒烟验证 (validate_pipeline.py — 极小子集 8 卡并行)
#   4. 正式训练 (parallel_launcher.py — 18 任务 8 卡并行)
#   5. 结果汇总 (summarize_results.py)
#   6. Keep Alive (防止 GPU 闲置被关)
#   7. 输出报告 + 计时
#
# 用法：
#   bash master.sh              # 完整流程
#   bash master.sh --skip-train # 跳过正式训练，仅做验证
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

START_TIME=$(date +%s)
MASTER_LOG="$SCRIPT_DIR/logs/master_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$SCRIPT_DIR/logs"

# --------------- 日志函数 ---------------
log() {
    local msg="[$(date '+%H:%M:%S')] $*"
    echo "$msg" | tee -a "$MASTER_LOG"
}
log_sep() {
    echo "" | tee -a "$MASTER_LOG"
    echo "================================================================" | tee -a "$MASTER_LOG"
    echo "" | tee -a "$MASTER_LOG"
}

# --------------- 计时 ---------------
elapsed() {
    local now=$(date +%s)
    local diff=$((now - START_TIME))
    local h=$((diff / 3600))
    local m=$(((diff % 3600) / 60))
    local s=$((diff % 60))
    printf "%dh%02dm%02ds" $h $m $s
}

# --------------- 激活环境 ---------------
log "Activate fly environment..."
eval "$(/volume/yzhao04/workspace/miniconda3/bin/conda shell.bash hook)"
conda activate fly
export PIP_CONSTRAINT=""

# --------------- 默认环境变量 ---------------
export DEVICE="${DEVICE:-all}"
export IMGSZ="${IMGSZ:-640}"
export WORKERS="${WORKERS:-16}"
export SEEDS="${SEEDS:-1 2 3}"
export YOLO_CACHE="${YOLO_CACHE:-disk}"

export YOLO11_MODEL="${YOLO11_MODEL:-yolo11m.pt}"
export YOLOV8_MODEL="${YOLOV8_MODEL:-yolov8n.pt}"
export YOLOV10_MODEL="${YOLOV10_MODEL:-yolov10n.pt}"
export RTDETR_MODEL="${RTDETR_MODEL:-rtdetr-l.pt}"

export EPOCHS_YOLO="${EPOCHS_YOLO:-200}"
export EPOCHS_RTDETR="${EPOCHS_RTDETR:-120}"
export EPOCHS_FASTER_RCNN="${EPOCHS_FASTER_RCNN:-120}"

export BATCH="${BATCH:--1}"
export D2_IMS_PER_BATCH="${D2_IMS_PER_BATCH:-8}"
export D2_NUM_WORKERS="${D2_NUM_WORKERS:-8}"

SKIP_TRAIN=false
if [[ "${1:-}" == "--skip-train" ]]; then
    SKIP_TRAIN=true
    log "Mode: validate only (--skip-train)"
fi

# ==============================================================================
log_sep
log "ICS6201 Training Pipeline — $(date '+%Y-%m-%d %H:%M:%S')"
log "GPU Device: $DEVICE"
log "Log file: $MASTER_LOG"
log_sep

# ==============================================================================
# STEP 1: 环境检查
# ==============================================================================
log "[STEP 1/6] Environment Check"

log "  Python version..."
python --version | tee -a "$MASTER_LOG"

log "  GPU detection..."
python "$SCRIPT_DIR/scripts/auto_config.py" 2>&1 | tee -a "$MASTER_LOG"
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
log "  Available GPUs: $GPU_COUNT"

log "  Core dependencies..."
python "$SCRIPT_DIR/scripts/check_env.py" 2>&1 | tee -a "$MASTER_LOG"

log "[STEP 1/6] DONE — $(elapsed)"
log_sep

# ==============================================================================
# STEP 2: 数据准备
# ==============================================================================
log "[STEP 2/6] Data Preparation"

if [ ! -d "$SCRIPT_DIR/yolo/images/train" ]; then
    log "  Generating YOLO dataset..."
    python "$SCRIPT_DIR/scripts/prepare_rgb_yolo.py" \
        --root "$SCRIPT_DIR" --clear --skip-ard 2>&1 | tee -a "$MASTER_LOG"
else
    log "  YOLO dataset already exists, skipping."
fi

log "  Generating data YAML..."
python "$SCRIPT_DIR/scripts/write_data_yaml.py" \
    --root "$SCRIPT_DIR" --out "$SCRIPT_DIR/configs/drone_rgb_abs.yaml" 2>&1 | tee -a "$MASTER_LOG"

log "  Generating COCO annotations..."
python "$SCRIPT_DIR/scripts/prepare_rgb_coco.py" \
    --root "$SCRIPT_DIR" 2>&1 | tee -a "$MASTER_LOG"

DATA_YAML="$SCRIPT_DIR/configs/drone_rgb_abs.yaml"
export DATA_YAML

STATS=$(python -c "import json; s=json.load(open('$SCRIPT_DIR/yolo/stats.json','r')); print(s.get('splits',{}))" 2>/dev/null || echo "N/A")
log "  Dataset splits: $STATS"

log "[STEP 2/6] DONE — $(elapsed)"
log_sep

# ==============================================================================
# STEP 3: 冒烟验证
# ==============================================================================
log "[STEP 3/6] Pipeline Validation (tiny subset, multi-GPU)"

VALIDATE_START=$(date +%s)

python "$SCRIPT_DIR/scripts/validate_pipeline.py" 2>&1 | tee -a "$MASTER_LOG"
VALIDATE_RC=${PIPESTATUS[0]}

VALIDATE_ELAPSED=$(python -c "print('$(elapsed)')")

if [ "$VALIDATE_RC" -eq 0 ]; then
    log "  Validation PASSED"
else
    log "  Validation FAILED (exit=$VALIDATE_RC) — check validation_report.md"
fi

log "[STEP 3/6] DONE — $(elapsed)"
log_sep

# ==============================================================================
# STEP 4: 正式训练
# ==============================================================================
if [ "$SKIP_TRAIN" = true ]; then
    log "[STEP 4/6] Formal Training — SKIPPED (--skip-train)"
else
    log "[STEP 4/6] Formal Training (18 tasks, $GPU_COUNT GPUs)"

    TRAIN_START=$(date +%s)

    python "$SCRIPT_DIR/scripts/parallel_launcher.py" --mode formal 2>&1 | tee -a "$MASTER_LOG"
    TRAIN_RC=${PIPESTATUS[0]}

    TRAIN_ELAPSED=$(python -c "print('$(elapsed)')")

    if [ "$TRAIN_RC" -eq 0 ]; then
        log "  Training ALL PASSED"
    else
        log "  Training completed with failures (exit=$TRAIN_RC)"
        log "  Check logs/state_formal.json for details"
        log "  Re-run failed tasks: python scripts/parallel_launcher.py --mode formal --only-failed"
    fi

    log "[STEP 4/6] DONE — $(elapsed)"
fi
log_sep

# ==============================================================================
# STEP 5: 结果汇总
# ==============================================================================
log "[STEP 5/6] Results Summary"

python "$SCRIPT_DIR/scripts/summarize_results.py" --mode all --out "$SCRIPT_DIR/training_report.md" 2>&1 | tee -a "$MASTER_LOG"

log "  Report saved: training_report.md"
log "[STEP 5/6] DONE — $(elapsed)"
log_sep

# ==============================================================================
# STEP 6: Keep Alive
# ==============================================================================
log "[STEP 6/6] Keep Alive"

KEEP_ALIVE="/volume/yzhao04/workspace/gpu-workspace/keep_alive/run.sh"
if [ -f "$KEEP_ALIVE" ]; then
    log "  Executing: $KEEP_ALIVE"
    bash "$KEEP_ALIVE" 2>&1 | tee -a "$MASTER_LOG" || log "  keep_alive returned non-zero (may be normal)"
else
    log "  WARNING: keep_alive script not found: $KEEP_ALIVE"
fi

log "[STEP 6/6] DONE — $(elapsed)"
log_sep

# ==============================================================================
# FINAL: 总计时 + 摘要
# ==============================================================================
TOTAL_ELAPSED=$(python -c "print('$(elapsed)')")

# 快速统计状态
SUMMARY_MD="$SCRIPT_DIR/logs/summary_$(date +%Y%m%d_%H%M%S).md"

cat > "$SUMMARY_MD" << MDEOF
# Master Script Execution Summary

- **Date**: $(date '+%Y-%m-%d %H:%M:%S')
- **Total Duration**: $TOTAL_ELAPSED
- **GPUs Used**: $GPU_COUNT

## Steps

| Step | Status | Duration |
|------|--------|----------|
| 1. Environment Check | DONE | — |
| 2. Data Preparation | DONE | — |
| 3. Pipeline Validation | $([ "$VALIDATE_RC" -eq 0 ] && echo "PASS" || echo "FAIL") | — |
| 4. Formal Training | $([ "$SKIP_TRAIN" = true ] && echo "SKIPPED" || echo "DONE") | — |
| 5. Results Summary | DONE | — |
| 6. Keep Alive | DONE | — |

## Output Files

- \`validation_report.md\` — 冒烟验证报告
- \`training_report.md\` — 训练结果对比
- \`logs/master_*.log\` — 完整执行日志
- \`logs/state_formal.json\` — 训练状态（断点续跑依据）

## Next Steps

- 查看失败任务: \`python scripts/summarize_results.py\`
- 仅重跑失败: \`python scripts/parallel_launcher.py --mode formal --only-failed\`
- 查看训练曲线: \`runs_formal/<run_name>/results.csv\`

MDEOF

# 终端输出
log_sep
log "================================================================"
log "  TRAINING PIPELINE COMPLETE"
log "  Total Duration: $TOTAL_ELAPSED"
log "================================================================"
log ""
log "  Master Log:    $MASTER_LOG"
log "  Validation:    validation_report.md"
log "  Training Rpt:  training_report.md"
log "  Summary:       $SUMMARY_MD"
log ""
log "  State Files:"
ls -lh "$SCRIPT_DIR/logs/state_"*.json 2>/dev/null | while read -r line; do
    log "    $line"
done
log ""
log "================================================================"
log_sep

echo ""
echo "Output files:"
echo "  Master Log:    $MASTER_LOG"
echo "  Validation:    $SCRIPT_DIR/validation_report.md"
echo "  Training Rpt:  $SCRIPT_DIR/training_report.md"
echo "  Summary:       $SUMMARY_MD"
