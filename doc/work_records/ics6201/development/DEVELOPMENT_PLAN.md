# ICS6201 无人机可见光检测 — 开发与训练计划

## 环境拓扑

| 项目 | 详情 |
|------|------|
| CPU 机器 | `yzhao04-cpu-shanghai:32196`，有外网，无 GPU |
| GPU 机器 | `yzhao04-gpu-shanghai:32198`，无外网，有 GPU |
| 共享存储 | GPFS `/shared/storage`，两台机器挂载同一路径 |
| 项目路径 | `/path/to/ICS6201` |

**关键约束**：conda 环境和 pip 包必须从 CPU 机器（有外网）安装，安装到共享存储上，GPU 机器直接使用。

---

## 阶段 0：CPU 机器 — 环境搭建

### 0.1 创建 conda 虚拟环境

```bash
# 在 CPU 机器上执行
ssh yzhao04-cpu-shanghai

# 已有 miniconda3 在共享存储
eval "$(/path/to/miniconda3/bin/conda shell.bash hook)"

# 创建专用于无人机检测的环境
conda create -n fly python=3.12 -y
conda activate fly
```

### 0.2 安装 PyTorch（CUDA 12.x）

```bash
# 先确认 GPU 机器的 CUDA 版本，选择匹配的 PyTorch
# 假设 CUDA 12.x，安装对应 torch：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 0.3 安装项目依赖

```bash
cd /path/to/ICS6201

# 核心依赖
pip install ultralytics pillow pyyaml

# Detectron2（Faster R-CNN 需要）
pip install 'setuptools<82'
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 0.4 验证环境（在 CPU 机器上做基础检查）

```bash
python scripts/check_env.py
# 预期：ultralytics/PIL/yaml OK，torch CUDA 不可用（CPU 机器正常）
# ffmpeg 缺失可以用 --skip-ard 规避
```

### 0.5 生成激活脚本

在项目根目录创建 `activate_fly.sh`：

```bash
#!/usr/bin/env bash
eval "$(/path/to/miniconda3/bin/conda shell.bash hook)"
conda activate fly
echo "=== fly env activated ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 阶段 1：数据准备

### 1.1 在 CPU 机器上准备数据（不含 ARD-MAV）

```bash
ssh yzhao04-cpu-shanghai
source /path/to/ICS6201/activate_fly.sh
cd /path/to/ICS6201

# 准备 YOLO + COCO 数据集（跳过 ARD-MAV，因为 CPU 机器无 ffmpeg）
bash train_scripts/smoke_01_prepare.sh
```

输出产物（在共享存储上）：
- `yolo/images/{train,val,test}/` — 图片
- `yolo/labels/{train,val,test}/` — YOLO 标注
- `yolo/stats.json` — 统计
- `coco/annotations/instances_{train,val,test}.json` — COCO 格式
- `configs/drone_rgb_abs.yaml` — 绝对路径数据集配置

### 1.2（可选）在 GPU 机器上准备 ARD-MAV

GPU 机器有 ffmpeg 才能抽帧。如果 GPU 机器也没有 ffmpeg 且有外网限制，可先彻底跳过 ARD-MAV（所有脚本已支持 `--skip-ard`）。

---

## 阶段 2：冒烟训练（GPU 机器，验证全链路）

```bash
ssh yzhao04-gpu-shanghai
source /path/to/ICS6201/activate_fly.sh
cd /path/to/ICS6201

# 确认 CUDA 可用
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE'"

# 所有模型跑 1 epoch
export EPOCHS_SMOKE=1
bash train_scripts/smoke_02_train.sh
```

验证要点：
- 6 个模型每个都能成功开始训练、完成 1 epoch、完成 val/test 评估
- 日志输出到 `logs/smoke/`，状态文件 `logs/state_smoke.json`

---

## 阶段 3：正式训练（GPU 机器，全量）

### 3.1 配置环境变量

参考 `env.example.sh`，在 GPU 机器上设置：

```bash
export DEVICE=0
export IMGSZ=640
export WORKERS=16
export SEEDS="1 2 3"
export YOLO_CACHE=disk

# 模型权重（nano 级别为主，快速出对比结果）
export YOLO11_MODEL=yolo11m.pt
export YOLOV8_MODEL=yolov8n.pt
export YOLOV10_MODEL=yolov10n.pt
export RTDETR_MODEL=rtdetr-l.pt

# 训练轮数
export EPOCHS_YOLO=200
export EPOCHS_RTDETR=120
export EPOCHS_FASTER_RCNN=120

# Batch（按 GPU 显存调整）
export BATCH_YOLO=64
export BATCH_RTDETR=48
export D2_IMS_PER_BATCH=8
export D2_NUM_WORKERS=8
```

### 3.2 启动训练

```bash
bash train_scripts/run_formal.sh
```

训练过程：
- 6 个模型 × 3 个 seed = 18 次训练任务，串行执行
- 状态持久化到 `logs/state_formal.json`，任意中断后可续跑
- 单次训练自动执行 train → val → test 三步

### 3.3 失败重跑

```bash
# 查看失败项
cat logs/state_formal.json | python -c "
import json, sys
s = json.load(sys.stdin)
for k, v in s['items'].items():
    if v.get('status') != 'ok':
        print(k, v.get('error', ''))
"

# 仅重跑失败项
python train_scripts/run_pipeline.py --mode formal --skip-ard --only-failed
```

### 3.4 预估时长

| 模型 | 单次训练（200 epoch） | 3 seeds |
|------|----------------------|---------|
| YOLO11m | 18–50 小时 | 54–150 小时 |
| YOLOv8n | 9–30 小时 | 27–90 小时 |
| YOLOv10n | 12–35 小时 | 36–105 小时 |
| RT-DETR-l | 30–90 小时 | 90–270 小时 |
| Faster R-CNN | 36–120 小时 | 108–360 小时 |
| DDW-YOLO | 18–60 小时 | 54–180 小时 |
| **合计** | — | **约 14–45 天（单卡串行）** |

> 详细分析见 `training_time_estimates_4090_48gb.md`

---

## 阶段 4：结果汇总与分析

### 4.1 收集指标

从 `logs/state_formal.json` 和 `runs_formal/` 中提取每个 run：

- mAP@50, mAP@50-95
- Precision, Recall
- 推理速度（ms/image）
- 模型参数量
- 最佳权重路径

### 4.2 输出对比表

| 模型 | Seed | mAP@50 | mAP@50-95 | Params | Speed |
|------|------|--------|-----------|--------|-------|
| YOLO11m | 1/2/3 | — | — | — | — |
| YOLOv8n | 1/2/3 | — | — | — | — |
| ... | | | | | |

### 4.3 可视化

- 各模型 PR 曲线对比
- 检测效果图（在同一组测试图上跑各模型，拼接对比）
- 训练 loss 曲线

---

## 阶段 5：消融实验与优化（可选）

### 5.1 DDW-YOLO 消融

验证 ECA、BiFPN、P2 检测头各自的贡献：

| 实验 | ECA | BiFPN | P2 | 说明 |
|------|-----|-------|----|------|
| Baseline | — | — | — | 原生 YOLO11m |
| +ECA | ✓ | — | — | 仅加通道注意力 |
| +BiFPN | — | ✓ | — | 仅加权特征融合 |
| +P2 | — | — | ✓ | 仅加小目标检测头 |
| DDW-YOLO | ✓ | ✓ | ✓ | 完整模型 |

### 5.2 超参调优

对最佳模型调整 lr、batch、imgsz，单 seed 验证即可。

### 5.3 大权重冲榜（可选）

如果最终需要冲 SOTA：将关键模型升级到 `l` 或 `x` 权重，减少 seed 到 1 或 2。

---

## 文件速查

| 用途 | 路径 |
|------|------|
| 训练主入口 | `train_scripts/run_pipeline.py` |
| 数据准备 | `scripts/prepare_rgb_yolo.py` |
| 单次训练执行 | `train_scripts/run_ultralytics_task.py` |
| Faster R-CNN 训练 | `scripts/train_detectron2_fasterrcnn.py` |
| DDW-YOLO 模型定义 | `models/ddw_yolo11m_p2_bifpn_eca.yaml` |
| DDW 自定义模块 | `train_scripts/ddw_modules.py` |
| 数据集配置（绝对路径） | `configs/drone_rgb_abs.yaml`（自动生成） |
| 训练状态（断点续跑） | `logs/state_formal.json` |
| 时长预估 | `training_time_estimates_4090_48gb.md` |
| 服务器操作手册 | `SERVER_RUNBOOK.md` |
| 环境激活脚本 | `activate_fly.sh`（待创建） |

---

## 风险与应对

| 风险 | 应对 |
|------|------|
| detectron2 与 CUDA/torch 版本不兼容 | 可跳过 Faster R-CNN（`DDW_MODEL` 为空时自动 skip） |
| GPU 机器无 ffmpeg，ARD-MAV 无法抽帧 | 所有脚本支持 `--skip-ard`，先用 DUT + DroneDetectionDataset 训练 |
| GPU 机器 conda 环境激活失败 | 确保 conda 路径在共享存储上一致（`/path/to/miniconda3`） |
| 单卡训练时间过长 | 先用子集（`--limit-*` 参数）或减少 epoch 数出一版结果 |
| 网络中断导致训练终止 | `run_pipeline.py` 支持 `--only-failed` 断点续跑 |
