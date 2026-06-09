# RGB 可见光无人机检测训练包

## 目录结构

- `raw_zips/`：原始数据集压缩包（已整理好）
- `configs/`：训练与数据准备配置
- `scripts/`：数据准备脚本（解压 + VOC→YOLO + 合并）
- `yolo/`：生成的 YOLO 数据集输出目录（运行脚本后生成）

## 当前包含的数据集（可见光）

- DUT Anti-UAV（Detection，VOC XML，含 train/val/test）
- DroneDetectionDataset（VOC XML，含 train/test；train 会再切出 val）
- ARD-MAV（视频 + VOC XML；脚本会从视频抽帧生成图片）

## 服务器端使用方式

1) 把整个 `rgb_trainpack/` 上传到服务器（建议保持目录结构不变）

2) 生成 YOLO 格式数据集

```bash
cd rgb_trainpack
python scripts/prepare_rgb_yolo.py --clear
```

如果服务器没有 ffmpeg（ARD-MAV 抽帧需要），可以先跳过 ARD：

```bash
python scripts/prepare_rgb_yolo.py --clear --skip-ard
```

3) 用 Ultralytics 训练（示例）

```bash
yolo train model=yolo11n.pt data=configs/drone_rgb.yaml imgsz=640 batch=32 epochs=200
```

## 输出说明

- `yolo/images/{train,val,test}`：图片
- `yolo/labels/{train,val,test}`：YOLO 标注
- `yolo/stats.json`：各数据源与总量统计
- `yolo/check_problems.json`：抽样校验发现的问题（若存在）

