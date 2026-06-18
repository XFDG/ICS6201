#!/usr/bin/env bash
# 激活 fly 环境（无人机检测训练）
# 用法: source /volume/yzhao04/workspace/ICS6201/activate_fly.sh

eval "$(/volume/yzhao04/workspace/miniconda3/bin/conda shell.bash hook)"
conda activate fly

# 覆盖 pip constraint，避免 NVIDIA 内部 constraint 干扰
export PIP_CONSTRAINT=""

echo "=== fly env (drone detection) ==="
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
CUDA_STATUS=$(python -c "import torch; print('YES (' + str(torch.cuda.device_count()) + ' GPUs)' if torch.cuda.is_available() else 'NO (CPU machine)')")
echo "CUDA:    ${CUDA_STATUS}"
echo "Ultralytics: $(python -c 'import ultralytics; print(ultralytics.__version__)')"
