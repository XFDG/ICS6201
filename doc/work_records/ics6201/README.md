# ICS6201 无人机检测项目记录

> 项目规模：三套无人机可见光数据，共 171,568 张图像；目标比较六种检测器，并建设可恢复的多 GPU 实验流水线。

## 核心成果

| 方向 | 结果 |
|---|---|
| 数据工程 | VOC 转 YOLO/COCO；ARD-MAV 按视频批量抽帧，将预计约 18 天缩短到 10-20 分钟 |
| 训练调度 | 从任务预绑 GPU + 全局线程池改为 per-GPU worker queue，消除错误派卡引发的 OOM |
| 可恢复性 | manifest 驱动 complete/resume/pending 判定，支持断点续跑与结果汇总 |
| 快速验证 | 环境、数据、模型和 launcher 共 14/14 项通过，约 5 分钟 |
| 核心实验 | RT-DETR-L 与 Faster R-CNN 各完成 3 seeds |

## 最终核心指标

| 模型 | 三个 seed | 均值 |
|---|---|---:|
| RT-DETR-L mAP50-95 | 0.67942 / 0.67529 / 0.67576 | 0.67682 |
| Faster R-CNN bbox/AP | 64.6024 / 64.6664 / 64.4290 | 64.5660 |

Faster R-CNN 三组均完成 2,062,560 iterations。不同框架使用各自原生标度，未强行合并成同一指标。

## 文档索引

| 文档 | 内容 |
|---|---|
| [项目总结](./PROJECT_SUMMARY.md) | 数据、模型、环境、运行方式和实验状态 |
| [训练恢复优先级计划](./TRAINING_RECOVERY_PRIORITY_PLAN_2026-06-05.md) | 任务清点、重跑优先级与验收规则 |
| [开发计划](./development/DEVELOPMENT_PLAN.md) | 数据准备、环境、训练和评估阶段 |
| [执行日志](./development/EXECUTION_LOG.md) | 已执行步骤与问题修复 |
| [服务器运行手册](./development/SERVER_RUNBOOK.md) | 多机执行、监控和恢复 |
| [训练时间估算](./development/training_time_estimates_4090_48gb.md) | 不同模型在目标 GPU 上的资源估算 |
| [最终指标 CSV](./assets/ics6201_final_metrics_2026-06-07.csv) | 两个核心模型的三 seed 结果 |

## 结论边界

- 原计划 6 模型 x 3 seeds 共 18 个正式任务，最终完成的是核心 6 个任务，不写成 18/18 完成。
- 原始数据、权重和大体积日志为控制仓库体积已删除；保留代码、开发文档和提取后的指标。
- RT-DETR 的 0-1 mAP 与 Detectron2 的 0-100 bbox/AP 标度不同，展示时必须注明。

