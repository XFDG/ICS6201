# 服务器训练一键跑通（Python 3.12 / RTX 4090）

本项目的目标：在服务器上把“无人机黑飞可见光检测（单类 drone）”的对比实验**自动连续跑完**，同时能：
- 全程记录完整 stdout/stderr 到日志文件
- 任意中断后可断点续跑
- 失败任务不会卡死整条流水线，最终给出失败清单，你可以只重跑失败项

## 1. 迁移后目录结构（关键）

把整个 `rgb_trainpack/` 拷贝到服务器任意目录，结构保持不变（当前机器的实际路径：`/root/rgb_trainpack`）：

- `rgb_trainpack/scripts/`：数据准备脚本（VOC→YOLO、可选 COCO）
- `rgb_trainpack/configs/`：数据集配置文件（自动生成绝对路径版）
- `rgb_trainpack/train_scripts/`：训练编排脚本（冒烟 + 正式）
- `rgb_trainpack/yolo/`：数据准备后的 YOLO 数据目录（脚本自动生成）
- `rgb_trainpack/logs/`：日志输出（自动生成）
- `rgb_trainpack/runs_smoke/`：冒烟训练 runs（自动生成）
- `rgb_trainpack/runs_formal/`：正式训练 runs（自动生成）

## 2. 安装依赖（Python 3.12）

建议在虚拟环境/conda 环境中安装：

```bash
python --version
pip install -U pip
pip install ultralytics
```

如果你要跑 Faster R-CNN（Detectron2），再安装 detectron2（不同 CUDA/torch 组合命令不同；这里给最常见的源码安装方式）：

```bash
pip install 'setuptools<82'
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## 3. 全局环境变量（统一入口）

在 `rgb_trainpack/` 目录下执行。你可以把下面写到一个 `env_server.sh` 里：

```bash
export DEVICE=0
export IMGSZ=640
export BATCH=-1
export WORKERS=16
export SEEDS="1 2 3"
export YOLO_CACHE=disk
```

可选：DDW‑YOLO（本仓库内置 YAML，推荐直接用这个；不设置也会自动探测并启用）：

```bash
export DDW_MODEL=/root/rgb_trainpack/models/ddw_yolo11m_p2_bifpn_eca.yaml
```

权重选择（默认推荐）：

```bash
export YOLO11_MODEL=yolo11m.pt
export YOLOV8_MODEL=yolov8n.pt
export YOLOV10_MODEL=yolov10n.pt
export RTDETR_MODEL=rtdetr-l.pt
```

训练轮数（按推荐值）：

```bash
export EPOCHS_YOLO=200
export EPOCHS_RTDETR=120
export EPOCHS_FASTER_RCNN=120
```

## 4. 冒烟脚本（两步）

### 4.1 冒烟 1：只做数据准备（不含 ARD-MAV）

```bash
cd /root/rgb_trainpack
bash train_scripts/smoke_01_prepare.sh
```

输出：
- `configs/drone_rgb_abs.yaml`（绝对路径 data.yaml）
- `yolo/images|labels/...`
- `coco/annotations/instances_{train,val,test}.json`（给 detectron2 用）

### 4.2 冒烟 2：所有模型跑 1 epoch（用于验证全链路）

```bash
cd /root/rgb_trainpack
export EPOCHS_SMOKE=1
bash train_scripts/smoke_02_train.sh
```

日志与状态：
- `logs/smoke/*.log`
- `logs/state_smoke.json`
- `runs_smoke/*`

## 5. 正式训练（一键跑到结束）

```bash
cd /root/rgb_trainpack
bash train_scripts/run_formal.sh
```

输出：
- 训练 runs：`runs_formal/`
- 全量日志：`logs/formal/*.log`
- 全局状态：`logs/state_formal.json`

## 6. 失败后如何只重跑未完成任务

正式训练脚本跑完后，如果存在失败项：

1) 打开 `logs/state_formal.json` 查看 `status=failed` 的条目
2) 直接重跑“只跑失败项”：

```bash
cd /root/rgb_trainpack
python train_scripts/run_pipeline.py --mode formal --skip-ard --only-failed
```

它会自动跳过已经 `ok` 的任务，仅重跑失败的 run。

## 7. 日志定位方式（最常用）

- 单个任务日志：`logs/formal/yolo11_seed1.log`（示例）
- 训练输出：`runs_formal/yolo11_seed1/`（Ultralytics）
- detectron2 输出：`runs_detectron2/`（如果你跑了 Faster R-CNN）

## 8. 给下一个 AI 的接手提示（你可以直接把这段贴给它）

1) 项目训练入口是 `rgb_trainpack/train_scripts/run_pipeline.py`  
2) 冒烟脚本：`smoke_01_prepare.sh`、`smoke_02_train.sh`  
3) 正式脚本：`run_formal.sh`  
4) 状态文件：`rgb_trainpack/logs/state_{smoke,formal}.json`（断点续跑依据）  
5) 数据集默认不含全量 ARD-MAV：所有脚本都传了 `--skip-ard`  
6) DDW‑YOLO 默认会尝试使用 `models/ddw_yolo11m_p2_bifpn_eca.yaml`；若你要指定其它模型，设置 `DDW_MODEL=/abs/path/to/your_ddw.yaml`  
