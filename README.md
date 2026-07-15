# RGB 可见光无人机检测训练包

本分支只保留代码、配置、轻量结果和文档。原始数据、派生数据、模型权重、checkpoint 与完整训练日志不进入 Git；获取数据或权重后请放在 `.gitignore` 已覆盖的本地目录中。

## 目录结构

- `configs/`：训练与数据准备配置
- `scripts/`：环境检查、数据转换、调度恢复和结果汇总
- `train_scripts/`：Ultralytics / Detectron2 训练入口
- `models/`：DDW-YOLO 模型配置
- `docs/ics6201/`：项目结果、图表和提取后的轻量指标
- `doc/`：简历与工作总结材料
- `raw_zips/`：本地原始数据目录，需自行创建，不提交
- `yolo/`、`coco/`、`cache/`：脚本生成的本地数据目录，不提交

## 数据集

实验使用以下可见光无人机数据集：

- DUT Anti-UAV Detection
- DroneDetectionDataset
- ARD-MAV

请按各数据集的授权方式获取压缩包，并放入本地 `raw_zips/`。仓库不再通过 Git LFS 分发数据集。

## 数据准备

```bash
python scripts/prepare_rgb_yolo.py --clear
python scripts/prepare_rgb_coco.py --root .
```

ARD-MAV 抽帧依赖 ffmpeg。没有 ffmpeg 时可先跳过：

```bash
python scripts/prepare_rgb_yolo.py --clear --skip-ard
```

## 训练示例

模型权重由框架下载或由使用者在本地准备，不应提交到仓库。

```bash
yolo train model=yolo11n.pt data=configs/drone_rgb.yaml imgsz=640 batch=32 epochs=200
```

完整流水线、恢复调度和历史结果见 [`docs/ics6201/PROJECT_SUMMARY.md`](docs/ics6201/PROJECT_SUMMARY.md)。
