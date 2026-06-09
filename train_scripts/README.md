# 训练脚本（18 次训练任务）

## 含义

- 6 个模型 × 3 个 seed = 18 次训练任务
- 默认 `EPOCHS=1`，用于在服务器上快速跑通流程；后续只需改环境变量即可控制训练轮数

## 使用前准备

1) 进入训练包目录

```bash
cd rgb_trainpack
```

2) 安装依赖（示例）

```bash
pip install ultralytics
```

Faster R-CNN 需要额外安装 detectron2。

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

3) 可选：提前生成 YOLO 数据集（若未生成，训练脚本会自动生成）

```bash
python scripts/prepare_rgb_yolo.py --clear
```

## 全局变量（关键）

```bash
export EPOCHS=1
export SEEDS="1 2 3"
export IMGSZ=640
export BATCH=32
export RUNS_DIR="$(pwd)/runs"
```

DDW-YOLO 需要你提供模型文件路径：

```bash
export MODEL=/path/to/ddw_yolo.py
```

## 运行方式

逐个模型运行：

```bash
bash train_scripts/train_yolo11n.sh
bash train_scripts/train_yolov8n.sh
bash train_scripts/train_yolov10n.sh
bash train_scripts/train_rtdetr_l.sh
bash train_scripts/train_faster_rcnn.sh
bash train_scripts/train_ddw_yolo.sh
```

一键跑完（会依次运行上述脚本）：

```bash
bash train_scripts/run_all.sh
```
