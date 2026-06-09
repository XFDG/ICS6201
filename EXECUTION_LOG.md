# 执行记录 — ICS6201 无人机检测训练

## 2026-05-29 阶段 0：环境搭建

### 0.1 创建 conda 环境
- **命令**: `conda create -n fly python=3.12 -y`
- **结果**: 成功，Python 3.12.13

### 0.2 安装 PyTorch
- **命令**: `PIP_CONSTRAINT="" pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
- **结果**: 成功，torch 2.5.1+cu121, torchvision 0.20.1+cu121
- **备注**: 需覆盖 pip constraint（默认 constraint 锁定 NVIDIA 内部 torch build）

### 0.3 安装项目依赖
- **命令**: `PIP_CONSTRAINT="" pip install ultralytics pyyaml`
- **结果**: 成功，ultralytics 8.4.56, pyyaml 6.0.3
- **命令**: `PIP_CONSTRAINT="" pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'`
- **结果**: 成功，detectron2 0.6
- **备注**: detectron2 需 `--no-build-isolation`，否则 build 环境找不到 torch

### 0.4 环境验证
- **命令**: `python scripts/check_env.py`
- **结果**: ultralytics/torch/PIL/yaml 全部 OK，CUDA 不可用（CPU 机器正常），ffmpeg 缺失（可用 --skip-ard 规避）

### 0.5 SSH 配置
- CPU 机器公钥已添加到 GPU 机器 `authorized_keys`
- SSH 连接已通：`ssh root@117.186.102.101 -p 32198`

### 0.6 GPU 机器修复 — opencv/libGL
- **问题**: GPU 机器缺少 libGL.so.1，opencv-python 无法 import
- **解决**: 卸载 opencv-python，安装 opencv-python-headless（无需 GUI 库）
- **命令**: `PIP_CONSTRAINT="" pip uninstall opencv-python -y && pip install opencv-python-headless`
- **结果**: 成功，GPU 机器 Python 环境完全正常

---

## 2026-05-29 阶段 1：数据准备

### 1.1 YOLO 数据集
- **命令**: `python scripts/prepare_rgb_yolo.py --clear --skip-ard`
- **结果**: 成功
- **数据量**: train=51,501 / val=7,745 / test=4,825（总计 64,071 张）
- **数据来源**: DUT Anti-UAV (10,000) + DroneDetectionDataset (54,071)
- **无问题**: check_problems.json 未生成（零问题）

### 1.2 数据集配置
- **命令**: `python scripts/write_data_yaml.py`
- **结果**: `/volume/yzhao04/workspace/ICS6201/configs/drone_rgb_abs.yaml`

### 1.3 COCO 格式（Detectron2 用）
- **命令**: `python scripts/prepare_rgb_coco.py --clear`
- **结果**: 成功
- **标注量**: train=52,649 / val=7,890 / test=5,108

---

## 2026-05-29 阶段 2：GPU 冒烟训练 ✅ 完成

- **机器**: yzhao04-gpu-shanghai (8× NVIDIA H200)
- **命令**: `bash train_scripts/smoke_02_train.sh`
- **配置**: EPOCHS_SMOKE=1, imgsz=640, device=0, workers=16

### 最终结果

| 模型 | 状态 | 备注 |
|------|------|------|
| YOLO11m (3 seeds) | ✅ 全部通过 | |
| YOLOv8n (3 seeds) | ✅ 全部通过 | |
| YOLOv10n (3 seeds) | ✅ 全部通过 | |
| RT-DETR-L (3 seeds) | ✅ 全部通过 | |
| DDW-YOLO (3 seeds) | ✅ 全部通过 | 自定义模型 (BiFPN+ECA+P2) |
| Faster R-CNN (3 seeds) | ❌ 全部失败 | GPU 无外网，无法下载 R-50 预训练权重 |
| **合计** | **15/18 OK, 3 failed** | |

### 待修复：Faster R-CNN 权重下载

- **问题**: detectron2 需要从 URL 下载 R-50-FPN 预训练权重，GPU 机器无外网
- **方案**: 在 CPU 机器上下载权重 → 放到共享存储 → GPU 机器使用本地路径

---

---

## 2026-05-29 阶段 2.5：新脚本创建

### 新建文件
| 文件 | 用途 |
|------|------|
| `scripts/auto_config.py` | 自动检测可用 GPU 数量/型号 |
| `scripts/parallel_launcher.py` | 多 GPU 并行训练启动器（替代串行 run_pipeline） |
| `scripts/validate_pipeline.py` | 极小子集全链路验证（8卡并行 + 计时 + keep_alive） |
| `scripts/summarize_results.py` | 从 state 文件和 runs 目录生成对比报告 |
| `master.sh` | 一键执行脚本（完整流程） |

### 修改文件
| 文件 | 变更 |
|------|------|
| `scripts/train_detectron2_fasterrcnn.py` | 支持本地预训练权重（`FRCNN_WEIGHTS` 环境变量或 `pretrained_weights/` 目录） |
| `pretrained_weights/faster_rcnn_R_50_FPN_3x.pkl` | 预下载的 R-50-FPN 权重（160MB），GPU 离线可用 |

### ARD-MAV 处理
- **状态**: 后台运行中，CPU 端 ffmpeg 逐帧抽帧 ~107K 帧
- **预计耗时**: 6-30 小时（取决于帧间 seek 效率）

## 下一步：验证脚本

在 GPU 机器上运行:
```bash
bash master.sh --skip-train
```
这会执行完整的验证流程（环境检查 + 极小子集 8 卡并行训练 + keep_alive），生成 `validation_report.md`。

验证通过后，运行正式训练:
```bash
bash master.sh
```

---