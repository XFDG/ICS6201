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
#   6. Keep Alive (after all pipeline processes finish; default GPU 0)
#   7. 输出报告 + 计时
#
# 用法：
#   bash master.sh --gpus 1,2,3,4,5,6,7              # 完整流程
#   bash master.sh --gpus 1,2,3,4,5,6,7 --skip-train # 跳过正式训练，仅做验证
#   bash master.sh --gpus 1,2,3,4,5,6,7 --no-keep-alive
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

# --------------- 默认环境变量 ---------------
# --gpus 参数：指定使用的 GPU（如 --gpus 1,2,3,4,5,6,7 留 GPU 0）
SKIP_TRAIN=false
SKIP_VALIDATE=false
START_KEEP_ALIVE="${START_KEEP_ALIVE:-true}"
ABORT_AFTER_VALIDATION=false
TRAIN_RC=0
GPU_IDS=""
RESUME_EXISTING=false
SKIP_COMPLETE=false
NO_SECONDARY=false
DRY_RUN=false
TARGET="${TRAIN_TARGET:-all}"
MANIFEST=""
DETECTRON_TARGET_ITER="${DETECTRON_TARGET_ITER:-0}"
START_STAGGER_SEC="${START_STAGGER_SEC:-60}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-train) SKIP_TRAIN=true; shift ;;
        --skip-validate) SKIP_VALIDATE=true; shift ;;
        --start-keep-alive) START_KEEP_ALIVE=true; shift ;;
        --no-keep-alive) START_KEEP_ALIVE=false; shift ;;
        --gpus) GPU_IDS="$2"; shift 2 ;;
        --resume-existing) RESUME_EXISTING=true; shift ;;
        --skip-complete) SKIP_COMPLETE=true; shift ;;
        --no-secondary) NO_SECONDARY=true; shift ;;
        --target) TARGET="$2"; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --detectron-target-iter) DETECTRON_TARGET_ITER="$2"; shift 2 ;;
        --start-stagger-sec) START_STAGGER_SEC="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Recovery mode auto-disables Phase 3 pipeline validation (it would touch GPU 0
# and the active runs while training is in flight). Explicit --skip-validate
# also disables it.
if [ "$RESUME_EXISTING" = true ] || [ "$SKIP_COMPLETE" = true ] || [ "$NO_SECONDARY" = true ] || [ "$TARGET" != "all" ]; then
    SKIP_VALIDATE=true
fi

# --------------- 激活环境 ---------------
TRAIN_ENV="${TRAIN_ENV:-fly}"
log "Activate ${TRAIN_ENV} environment..."
eval "$(/volume/yzhao04/workspace/miniconda3/bin/conda shell.bash hook)"
conda activate "$TRAIN_ENV"
export PIP_CONSTRAINT=""

if [ -n "$GPU_IDS" ]; then
    export TRAIN_GPU_IDS="$GPU_IDS"
    log "Using physical GPUs: $GPU_IDS"
else
    export TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-}"
fi

export DEVICE="${DEVICE:-0}"
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


# ==============================================================================
log_sep
log "ICS6201 Training Pipeline — $(date '+%Y-%m-%d %H:%M:%S')"
log "Train Env: $TRAIN_ENV"
log "Train GPU IDs: ${TRAIN_GPU_IDS:-auto}"
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
if [ -n "${TRAIN_GPU_IDS:-}" ]; then
    TRAIN_GPU_COUNT=$(python - << PY
ids = [x.strip() for x in "${TRAIN_GPU_IDS}".split(",") if x.strip()]
print(len(ids))
PY
)
else
    TRAIN_GPU_COUNT="$GPU_COUNT"
fi
log "  Training GPUs: $TRAIN_GPU_COUNT"

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
VALIDATE_RC=0

if [ "$SKIP_VALIDATE" = true ]; then
    log "  SKIPPED (--skip-validate / recovery mode)"
else
    set +e  # 验证失败不应终止整个脚本
    VALIDATE_CMD=(python "$SCRIPT_DIR/scripts/validate_pipeline.py" --skip-keep-alive)
    if [ -n "${TRAIN_GPU_IDS:-}" ]; then
        VALIDATE_CMD+=(--gpus "$TRAIN_GPU_IDS")
    fi
    "${VALIDATE_CMD[@]}" 2>&1 | tee -a "$MASTER_LOG"
    VALIDATE_RC=${PIPESTATUS[0]}
    set -e

    VALIDATE_ELAPSED=$(python -c "print('$(elapsed)')")

    if [ "$VALIDATE_RC" -eq 0 ]; then
        log "  Validation PASSED"
    else
        log "  Validation FAILED (exit=$VALIDATE_RC) — check validation_report.md"
        START_KEEP_ALIVE=false
        if [ "$SKIP_TRAIN" != true ]; then
            ABORT_AFTER_VALIDATION=true
            log "  Formal training will be skipped because validation failed."
        fi
    fi
fi

log "[STEP 3/6] DONE — $(elapsed)"
log_sep

# ==============================================================================
# STEP 4: 正式训练
# ==============================================================================
if [ "$SKIP_TRAIN" = true ]; then
    log "[STEP 4/6] Formal Training — SKIPPED (--skip-train)"
elif [ "$ABORT_AFTER_VALIDATION" = true ]; then
    log "[STEP 4/6] Formal Training — SKIPPED (validation failed)"
else
    log "[STEP 4/6] Formal Training (18 tasks, $TRAIN_GPU_COUNT GPUs)"

    TRAIN_START=$(date +%s)

    set +e
    TRAIN_CMD=(python "$SCRIPT_DIR/scripts/parallel_launcher.py" --mode formal)
    if [ -n "${TRAIN_GPU_IDS:-}" ]; then
        TRAIN_CMD+=(--gpus "$TRAIN_GPU_IDS")
    fi
    TRAIN_CMD+=(--target "$TARGET" --start-stagger-sec "$START_STAGGER_SEC")
    if [ "$RESUME_EXISTING" = true ]; then TRAIN_CMD+=(--resume-existing); fi
    if [ "$SKIP_COMPLETE" = true ]; then TRAIN_CMD+=(--skip-complete); fi
    if [ "$NO_SECONDARY" = true ]; then TRAIN_CMD+=(--no-secondary); fi
    if [ "$DRY_RUN" = true ]; then TRAIN_CMD+=(--dry-run); fi
    if [ -n "$MANIFEST" ]; then TRAIN_CMD+=(--manifest "$MANIFEST"); fi
    if [ "${DETECTRON_TARGET_ITER:-0}" != "0" ]; then
        TRAIN_CMD+=(--detectron-target-iter "$DETECTRON_TARGET_ITER")
    fi
    log "  Launching: ${TRAIN_CMD[*]}"
    "${TRAIN_CMD[@]}" 2>&1 | tee -a "$MASTER_LOG"
    TRAIN_RC=${PIPESTATUS[0]}
    set -e

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
if [ "$START_KEEP_ALIVE" != "true" ]; then
    log "  SKIPPED (START_KEEP_ALIVE=false)"
elif [ -f "$KEEP_ALIVE" ]; then
    export KEEP_ALIVE_GPUS="${KEEP_ALIVE_GPUS:-0}"
    log "  Starting daemon on GPU(s) ${KEEP_ALIVE_GPUS}: $KEEP_ALIVE"
    nohup bash "$KEEP_ALIVE" >> "$MASTER_LOG" 2>&1 &
    KEEP_ALIVE_PID=$!
    sleep 3
    if kill -0 "$KEEP_ALIVE_PID" 2>/dev/null; then
        log "  keep_alive daemon started (pid=$KEEP_ALIVE_PID)"
    else
        log "  keep_alive exited early; check $MASTER_LOG"
    fi
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
- **GPUs Used**: $TRAIN_GPU_COUNT

## Steps

| Step | Status | Duration |
|------|--------|----------|
| 1. Environment Check | DONE | — |
| 2. Data Preparation | DONE | — |
| 3. Pipeline Validation | $([ "$VALIDATE_RC" -eq 0 ] && echo "PASS" || echo "FAIL") | — |
| 4. Formal Training | $(if [ "$SKIP_TRAIN" = true ]; then echo "SKIPPED"; elif [ "$ABORT_AFTER_VALIDATION" = true ]; then echo "SKIPPED_VALIDATION_FAILED"; else echo "DONE"; fi) | — |
| 5. Results Summary | DONE | — |
| 6. Keep Alive | $([ "$START_KEEP_ALIVE" = true ] && echo "DONE" || echo "SKIPPED") | — |

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

if [ "$ABORT_AFTER_VALIDATION" = true ]; then
    exit "$VALIDATE_RC"
fi
if [ "$TRAIN_RC" -ne 0 ]; then
    exit "$TRAIN_RC"
fi
exit "$VALIDATE_RC"
