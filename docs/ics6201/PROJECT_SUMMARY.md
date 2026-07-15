# ICS6201 无人机可见光检测 — 项目完整总结

## 项目概述

**目标**：用 6 种目标检测模型在 3 套无人机数据集上进行对比实验，找出最适合无人机黑飞检测的方案。

**基本信息**：

| 项目 | 详情 |
|------|------|
| 任务 | 单类（drone）目标检测 |
| 检测类别 | `0: drone` |
| 模型数量 | 6 个（YOLOv8n, YOLOv10n, YOLO11m, RT-DETR-L, Faster R-CNN R50-FPN, DDW-YOLO） |
| 每个模型运行 | 3 个随机种子（seed=1,2,3），共计 18 次训练 |
| 服务器 | CPU: `yzhao04-cpu-shanghai:32196`，GPU: `yzhao04-gpu-shanghai:32198` |
| GPU | 8× NVIDIA H200 (141GB HBM3e) |
| 存储 | GPFS 共享存储 `/volume/yzhao04/`，两台机器相同路径 |
| 项目路径 | `/volume/yzhao04/workspace/ICS6201/` |
| 当前环境 | conda env `fly`（PyTorch 2.5.1+cu121 + Ultralytics + Detectron2） |

---

## 一、数据集

### 数据来源

| 数据集 | 格式 | train | val | test | 合计 |
|--------|------|-------|-----|------|------|
| DUT Anti-UAV Detection | VOC XML | 5,200 | 2,600 | 2,200 | 10,000 |
| DroneDetectionDataset | VOC XML | 46,301 | 5,145 | 2,625 | 54,071 |
| ARD-MAV | 视频+XML | 85,997 | 10,749 | 10,751 | 107,497 |
| **总计** | YOLO | **137,498** | **18,494** | **15,576** | **171,568** |

### 数据准备流程

1. **准备原始 zip** → 按数据集授权方式下载到本地 `raw_zips/`；数据不进入 Git
2. **合并分卷** → `cat ARD-MAV_Glad.zip.part-* > ARD-MAV_Glad.zip`
3. **VOC→YOLO 转换** → `scripts/prepare_rgb_yolo.py`（DUT + DroneDet）
4. **ARD-MAV 快速抽帧** → `scripts/prepare_ard_mav_fast.py`（60 个视频一次性批量导出帧，10-20 分钟）
5. **COCO 格式** → `scripts/prepare_rgb_coco.py`（给 Detectron2 用）

### 关键经验

- ARD-MAV 原始脚本逐帧调用 ffmpeg（107K 次），预估 18 天 → 改为每视频一次性全帧导出，**10-20 分钟完成**
- GPFS 共享存储 I/O 延迟高，大量小文件 `rmtree` 可能超时 → 数据准备避免 `--clear`
- `ls ardmav_*` 匹配 86K 文件会参数溢出返回 0 → 用 Python glob 或 find 统计

---

## 二、环境搭建与踩坑

### Conda 环境安装（CPU 机器有外网，安装到共享存储）

```bash
conda create -n fly python=3.12 -y
conda activate fly
PIP_CONSTRAINT="" pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
PIP_CONSTRAINT="" pip install ultralytics pyyaml
PIP_CONSTRAINT="" pip install opencv-python-headless  # GPU 无 libGL
PIP_CONSTRAINT="" pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
```

激活：`source /volume/yzhao04/workspace/ICS6201/activate_fly.sh`

### 踩坑记录

| 问题 | 原因 | 方案 |
|------|------|------|
| `libGL.so.1` 缺失 | GPU 机器无 GUI 库 | 换 `opencv-python-headless` |
| pip constraint 冲突 | NVIDIA 内部 constraint 锁定 torch 版本 | `PIP_CONSTRAINT=""` 覆盖 |
| detectron2 安装失败 | pip 隔离环境无 torch | `--no-build-isolation` |
| SSH `Permission denied` | 无密钥 | CPU 端生成 ed25519 密钥，通过共享存储加到 GPU `authorized_keys` |
| Faster R-CNN 权重下载失败 | GPU 机器无外网 | CPU 端预下载 `pretrained_weights/faster_rcnn_R_50_FPN_3x.pkl`（160MB） |
| Faster R-CNN 训练 NaN | batch=4 太小梯度不稳定 | batch 改 32 |
| DDW-YOLO 训练报错 | Ultralytics Muon optimizer 要求 2D 参数，DDW 自定义模块 ECA/BiFPN 有 1D/3D 参数 | 改用 `AdamW` 或 `SGD` optimizer |
| `CUDA_VISIBLE_DEVICES` 未生效 | PyTorch 2.5.1 CUDA 12.1 与系统 CUDA 13.1 驱动版本不一致 | 改用 conda env `ai`（PyTorch 2.11 CUDA 13.1） |
| `master.sh` 验证失败后直接退出 | `set -euo pipefail` + 验证非零退出码 | 关键命令前加 `set +e`，后恢复 `set -e` |

---

## 三、脚本体系

### 新建脚本

| 文件 | 功能 |
|------|------|
| `scripts/auto_config.py` | 自动检测可用 GPU 数量、型号、显存 |
| `scripts/parallel_launcher.py` | 多 GPU 并行训练启动器，将任务分配到不同卡 |
| `scripts/validate_pipeline.py` | 极小子集全链路验证（环境/数据/训练/keep_alive） |
| `scripts/summarize_results.py` | 从 state 文件和 runs 目录生成对比报告 |
| `scripts/prepare_ard_mav_fast.py` | ARD-MAV 快速批处理（替代逐帧 ffmpeg） |
| `master.sh` | 一键执行入口 |
| `activate_fly.sh` | conda 环境激活 |

### master.sh 执行流程

```
STEP 1: 环境检查（Python 版本、GPU 检测、依赖导入）
STEP 2: 数据准备（YAML 生成、COCO 标注）
STEP 3: 冒烟验证（极小子集并行训练，输出 validation_report.md）
STEP 4: 正式训练（parallel_launcher.py，18 任务分配到 GPU）
STEP 5: 结果汇总（summarize_results.py，输出 training_report.md）
STEP 6: Keep Alive（启动 GPU 防回收守护进程）
```

### 启动命令

```bash
# 完整流程（留 GPU 0 给其他任务）
./master.sh --gpus 1,2,3,4,5,6,7

# 仅验证，跳过正式训练
./master.sh --gpus 1,2,3,4,5,6,7 --skip-train
```

---

## 四、验证结果

最终快速验证：**14/14 全部通过**，总耗时约 5 分钟。

| 检查项 | 结果 | 耗时 |
|--------|------|------|
| 8 GPU 检测 | PASS | 5s |
| GPU 选择（物理 GPU 1-7） | PASS | 0s |
| 环境导入 | PASS | 0s |
| 数据准备 | PASS | 0s |
| YOLOv8n 训练 | PASS | 39s |
| YOLOv10n 训练 | PASS | 38s |
| YOLO11m 训练 | PASS | 40s |
| DDW-YOLO 训练 | PASS | 44s |
| RT-DETR-L 训练 | PASS | 56s |
| Faster R-CNN 训练 | PASS | 4min05s |
| 多卡并行 | PASS | 0s |
| Keep Alive | PASS | skipped |
| 端到端计时 | PASS | 5min07s |

---

## 五、训练结果与当前状态（2026-06-09 更新）

### 核心任务完成情况

本轮优先保证的核心任务为 **RT-DETR-L** 和 **Faster R-CNN Detectron2**，每个模型 3 个 seed。截止 2026-06-09，6 个核心训练任务已全部完成。2026-07-15 仓库瘦身后不再保存权重和完整逐步日志，只保留结果摘要、RT-DETR `results.csv` 和提取后的轻量指标表。

![ICS6201 primary completion](assets/ics6201_primary_completion_2026-06-09.png)

| 核心任务 | 状态 | 最终进度 | 最终指标 |
|----------|------|----------|----------|
| rtdetr_seed1 | DONE | epoch 120 | mAP50-95 = **0.67942** |
| rtdetr_seed2 | DONE | epoch 120 | mAP50-95 = **0.67529** |
| rtdetr_seed3 | DONE | epoch 120 | mAP50-95 = **0.67576** |
| faster_rcnn_seed1 | DONE | iter = **2062560/2062560** | bbox/AP = **64.6024** |
| faster_rcnn_seed2 | DONE | iter = **2062560/2062560** | bbox/AP = **64.6664** |
| faster_rcnn_seed3 | DONE | iter = **2062560/2062560** | bbox/AP = **64.4290** |

完整轻量指标见 `assets/ics6201_final_metrics_2026-06-07.csv`。RT-DETR-L 的 mAP50-95 均值为 **0.67682**，Faster R-CNN 的原生 Detectron2 bbox/AP 均值为 **64.5660**。

### 非核心任务历史状态

| 任务 | 状态 | 说明 |
|------|------|------|
| yolo11_seed1 | DONE | 曾完成；仓库瘦身后不保留输出和权重 |
| yolo11_seed2 | PARTIAL | 旧调度逻辑下 OOM 后中断；checkpoint 已删除，如需补跑应重新训练 |
| yolo11_seed3 | PENDING | 尚未正式启动 |
| ddw_yolo_seed1-3 | PENDING | 尚未正式启动 |
| yolov10_seed1-3 | PENDING | 尚未正式启动 |
| yolov8_seed1-3 | PENDING | 尚未正式启动 |

### 仓库瘦身后的重跑策略

2026-06-09 的 manifest/checkpoint 恢复方案属于历史执行记录。2026-07-15 清理后，仓库不再包含原始数据和 checkpoint，因此不能直接从旧 `last.pt` 恢复。如需补跑 secondary，应重新准备数据与初始权重，再使用新版 per-GPU worker launcher 从头训练。

数据准备完成后可先执行 dry-run，检查任务与 GPU 分配：

```bash
cd /volume/yzhao04/workspace/ICS6201
source /volume/yzhao04/workspace/miniconda3/etc/profile.d/conda.sh
conda activate fly

python scripts/parallel_launcher.py \
  --mode formal \
  --data configs/drone_rgb_abs.yaml \
  --gpus 1,2,3,4,5,6,7 \
  --target secondary \
  --dry-run
```

确认调度结果无误后，去掉 `--dry-run` 才会开始训练。只有重新产生 checkpoint 后，`recovery_manifest.py` 与 `--resume-existing` 才能再次用于中断恢复。

### GPU 使用说明

```bash
# GPU 0 完全空闲，留给其他任务
CUDA_VISIBLE_DEVICES=0 python your_script.py
```

---

## 六、文件索引

| 文件 | 说明 |
|------|------|
| `master.sh` | 一键执行脚本 |
| `activate_fly.sh` | 环境激活 |
| `DEVELOPMENT_PLAN.md` | 开发计划 |
| `EXECUTION_LOG.md` | 详细执行记录 |
| `validation_report.md` | 验证报告 |
| `training_report.md` | 训练结果（训练完成后生成） |
| `scripts/recovery_manifest.py` | 恢复训练 manifest 生成，识别 complete/partial/pending |
| `scripts/auto_config.py` | GPU 检测 |
| `scripts/parallel_launcher.py` | 多卡并行训练（含进度监控） |
| `scripts/validate_pipeline.py` | 子集验证 |
| `scripts/summarize_results.py` | 结果汇总 |
| `scripts/prepare_ard_mav_fast.py` | ARD-MAV 快速抽帧 |
| `scripts/prepare_rgb_yolo.py` | VOC→YOLO 数据准备 |
| `scripts/prepare_rgb_coco.py` | YOLO→COCO 转换 |
| `scripts/train_detectron2_fasterrcnn.py` | Faster R-CNN 训练 |
| `train_scripts/run_ultralytics_task.py` | 单次 Ultralytics 训练 |
| `train_scripts/ddw_modules.py` | DDW-YOLO 自定义模块（ECA + BiFPN） |
| `models/ddw_yolo11m_p2_bifpn_eca.yaml` | DDW-YOLO 模型定义 |
| `configs/drone_rgb_abs.yaml` | 数据集配置（绝对路径，自动生成） |
| `assets/ics6201_final_metrics_2026-06-07.csv` | 从最终日志提取的 6 个核心任务指标 |
| `pretrained_weights/` | 本地预训练权重目录，不入库 |
| `raw_zips/` | 本地原始数据目录，不入库 |
| `yolo/` | 本地 YOLO 派生数据目录，不入库 |
| `coco/` | 本地 COCO 派生数据目录，不入库 |
| `logs/` | 本地运行日志，不入库 |
| `runs_formal/` | 仅保留已跟踪的小型结果表；权重不入库 |
