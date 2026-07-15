# H200 Grouped GEMM / SonicMoE 工作记录

> 本目录收录公开安全的实验总结、配置级 tuning 记录、聚合 CSV、绘图脚本和一组四路 benchmark 归档。线上原始 profile、内部模型路径和未脱敏 range 数据未收录。

## 阅读顺序

| 文档 | 内容 |
|---|---|
| [H200 GEMM 性能分析公开版](./H200_GEMM性能分析公开版.md) | 四路 benchmark、profile 去噪、row 口径纠偏、线上回放 |
| [Quack 调优开发记录公开版](./Quack调优开发记录公开版.md) | 14 shapes x 3 distributions 的 config tuning 方法与结果 |
| [SM90 tuning guide](./references/sm90_tuning_guide.md) | SM90 tile、cluster、persistent 与 swizzle 的选择依据 |
| [四路 benchmark 归档](./archive/four_way_benchmark_20260528/README.md) | 早期 CUTLASS / Quack / DeepGEMM / SonicMoE 对照 |

## 核心结论

- Quack pure GEMM 在选定 H200 shape 集上平均约 66.5% MFU。
- SonicMoE e2e forward/backward 约 32.1%/20.8% MFU，瓶颈不只在 GEMM。
- 稳态 profile 中 Quack GEMM 约占 94.98%，token_gather_sum 从受 autotune 污染的表象 69.9% 校正为 3.78%。
- standalone 的 T*top-k 全展开 rows 为 195,976，而线上 DeepEP compact rows 为 29,446；混用口径造成此前 4x-6x 性能误判。
- 三个代表 shape 回放与线上差异为 +0.46% / -1.11% / -5.40%。
- config-level tuning 在目标 shape 上约 +5.3%，14 shapes x 3 分布聚合约 +1.84%。

## 目录说明

| 路径 | 说明 |
|---|---|
| data/ | 已脱敏的聚合 CSV |
| scripts/ | benchmark 汇总、调优与绘图脚本；默认路径已改为仓库相对路径 |
| references/ | SM90 调优方法 |
| archive/ | 历史 benchmark 与原始 JSON |

## 使用边界

- 1.84% 是候选 config 扩展带来的聚合收益，不是 kernel 重写收益。
- 66.5% MFU 是 pure GEMM；不能与 e2e forward/backward 直接等价。
- 部分阶段没有 NCU 硬件计数器权限，结论主要来自 Nsys、latency、FLOPs 和数据形状。
- 未公开的线上原始 ranges 不在本仓库；公开 CSV 足以核对聚合结论，但不能复原内部 workload。

