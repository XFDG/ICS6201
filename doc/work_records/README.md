# 工作记录总索引

> 整理日期：2026-08-11
> 目标：为秋招简历和技术面试保留足够的工程证据，同时让公开仓库保持安全、精简、可读。

## 推荐阅读

1. 先看 [2026 年第三季度公开工作记录](./2026_q3/README.md)，了解最新的 MoE、推理优化、KernelBench 和平台验证工作。
2. 再看 [2026-06-15 至 07-15 月报](./monthly_report/2026-06-15_2026-07-15.md)，了解前序工作。
2. 看 [简历候选表述与取舍](./简历候选表述.md)，比较九坤实习与专业技能的长短版本。
3. 再看 [秋招简历素材库](../旧的简历/秋招简历素材库_AIInfra_2026.md)，按岗位选择 bullet。
4. 需要完整时间线和结论边界时看 [AI Infra 工作总结](../旧的简历/秋招工作总结_AIInfra_2026.md)。
5. 面试前按目标岗位进入 R3、FlashInfer、OE、LLMQRT、GEMM、DeepGEMM 或 ICS6201 专题。
6. 需要追溯阶段过程时再读周报，不把周报中的早期口径当最终结论。

## 专题目录

| 优先级 | 专题 | 可展示能力 | 入口 |
|---|---|---|---|
| P0 | 6.15-7.15 月度总结 | RL、算子开发、故障定位、评测和技术路线 | [monthly_report](./monthly_report/README.md) |
| P0 | R3 / Router Replay | MoE RL、训推一致性、分布式训练、指标设计、CUDA Graph 排障 | [r3](./r3/README.md) |
| P0 | FlashInfer TP2 CUDA Graph | GPU hang 最小复现、FTZ/sentinel 根因、补丁与版本回归 | [flashinfer](./flashinfer/README.md) |
| P0 | OE Async | 异步调度、GPU token history、Triton fused-hash、多卡正确性 | [oe](./oe/README.md) |
| P0 | LLMQRT H200 / AWQ TP=2 | SM90 kernel 排障、W4A16 量化、packed-weight TP、吞吐与显存优化 | [llmqrt](./llmqrt/README.md) |
| P0 | H200 Grouped GEMM / SonicMoE | Nsys、MFU、shape 回放、口径纠偏、SM90 config tuning | [gemm_sonicmoe](./gemm_sonicmoe/README.md) |
| P1 | DeepGEMM 离线交付 | cubin bundle、JIT cache、wheel、SHA 校验、TP=2 环境 | [deep_gemm](./deep_gemm/README.md) |
| P2 | ICS6201 无人机检测 | 数据工程、多 GPU 调度、恢复机制、多 seed 实验 | [ics6201](./ics6201/README.md) |
| P3 | AI Infra 技术调研 | 训练框架、Kernel DSL、通信融合、编译与诊断 | [research](./research/README.md) |
| 过程记录 | 周报 | 决策演进、阶段结果与问题 | [weekly_report](./weekly_report/README.md) |
| 论文讨论 | Native Sparse Attention | 稀疏注意力机制与工程讨论 | [paper_notes](./paper_notes/native_sparse_attention_discussion.md) |

## 简历能力与证据速查

| 能力方向 | 可直接使用的证据 | 主要入口 |
|---|---|---|
| MoE-RL 系统与训推一致性 | route 采集/回放、response-mask、18 项 CPU 测试、8 卡 20-step A/B/C、32 卡 smoke | [R3](./r3/README.md) |
| RL 推理侧算子开发 | OE async decode、GPU token history、Triton fused-hash、TP1/TP2/TP4 正确性与吞吐 | [OE](./oe/README.md) |
| CUDA Graph 与通信 Kernel 排障 | TP/Graph/fusion 控制变量、两卡最小复现、FTZ/sentinel 根因、位级修复和 SASS 回归 | [FlashInfer](./flashinfer/README.md) |
| H200 Kernel 与量化推理 | SM90 GEMV 越界修复、compute-sanitizer、W4A16 AWQ、TP=2 packed-weight 与 NCCL all-reduce | [LLMQRT](./llmqrt/README.md) |
| GPU 性能分析与调优 | Nsys range、MFU/latency、shape replay、compact rows 口径纠偏、SM90 config tuning | [GEMM/SonicMoE](./gemm_sonicmoe/README.md) |
| 性能与能力评测 | MoE/Attention 正交评测、140 组性能用例、25,624 题能力回归、数值等价边界 | [月报](./monthly_report/2026-06-15_2026-07-15.md) |
| AI 协作开发 | Claude Code/Codex 用于检索、实现、测试和文档；人工 review、benchmark 与回归兜底 | [简历候选表述](./简历候选表述.md) |

## 材料筛选规则

### 保留

- 公开安全的开发文档、实验设计、聚合 CSV、图表和绘图脚本。
- 能支持简历数字的环境、口径、对照组和失败边界。
- 仍有阅读价值的周报与仓库分析报告。
- 小体积、可解释、可复用的 benchmark 脚本。

### 不保留

- 模型权重、数据集、checkpoint、原始运行日志和大型 profile。
- 内部仓库 URL、平台 endpoint、节点/IP、镜像、run id 和私有数据路径。
- token、API key、SSH key、授权文件以及任何凭据上下文。
- 无法验证、已被后续实验推翻或可能误导招聘方的结论。

## 指标可信度

| 等级 | 定义 | 示例 |
|---|---|---|
| A | 有同口径对照与聚合数据 | R3 20-step A/B/C、ICS 三 seed |
| B | 有 benchmark/trace 与复现脚本 | Quack config tuning、四路 GEMM |
| C | 短程 smoke 或单次验证 | 32 卡 R3 5-step、单轮平台能力评测 |
| D | 上游论文/README 数据 | 技术调研中的公开性能数字 |

本人确认但尚未保存脱敏公开产物的内部终态记为 U，例如 R3 128 卡 200-step、SwanLab 监控和内部开发仓合入。U 级可用于说明个人实际交付，但不得伪装成公开可复核数据，也不能外推未验证的 graph mode、reward 或收敛结论。

简历主数字优先使用 A/B 级；C 级必须写明 smoke/单次；D 级只能作为背景资料。

## 仓库体积策略

该目录只保存文档和小型证据资产。权重、数据集和不可复用的大型二进制已删除，避免把课程项目仓库变成历史存储盘。
