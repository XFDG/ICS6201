# 面试押题目录

更新日期：2026-09-29。范围为当前简历中的九坤实习条目，以及三组开源贡献；不重复展开量化 Runtime、论文等校内项目。

## 九坤实习

| 条目 | Markdown | PDF |
| --- | --- | --- |
| FA3 确定性 SWA backward | [01_FA3](面试押题_01_FA3确定性SWA反向优化.md) | [PDF](面试押题_01_FA3确定性SWA反向优化.pdf) |
| Sink FC1 计算通信竞争 | [02_SinkFC1](面试押题_02_SinkFC1计算通信竞争.md) | [PDF](面试押题_02_SinkFC1计算通信竞争.pdf) |
| 确定性 Router GEMM | [03_RouterGEMM](面试押题_03_RouterGEMM确定性优化.md) | [PDF](面试押题_03_RouterGEMM确定性优化.pdf) |
| OE 异步状态算子 | [04_OE](面试押题_04_OE异步状态算子.md) | [PDF](面试押题_04_OE异步状态算子.pdf) |
| R3 路由回放 | [05_R3](面试押题_05_R3路由回放.md) | [PDF](面试押题_05_R3路由回放.pdf) |
| FlashInfer rollout hang | [06_FlashInfer](面试押题_06_FlashInferRolloutHang.md) | [PDF](面试押题_06_FlashInferRolloutHang.pdf) |
| MoE Router orth-loss（训练版） | [10_OrthLoss](面试押题_10_MoERouterOrthLoss融合.md) | [PDF](面试押题_10_MoERouterOrthLoss融合.pdf) |

## 开源贡献

| 简历分组 | Markdown | PDF |
| --- | --- | --- |
| Mooncake：7 个已合入 PR | [07_Mooncake](面试押题_07_Mooncake开源贡献.md) | [PDF](面试押题_07_Mooncake开源贡献.pdf) |
| 摩尔线程 TME：10 个已合入 PR | [08_TME](面试押题_08_摩尔线程TME开源贡献.md) | [PDF](面试押题_08_摩尔线程TME开源贡献.pdf) |
| Ray / FlashInfer / Mirage：各 1 个 | [09_其他项目](面试押题_09_RayFlashInferMirage开源贡献.md) | [PDF](面试押题_09_RayFlashInferMirage开源贡献.pdf) |

## 使用顺序

1. 先看开源仓库技术介绍与应用，弄清组件在系统中的位置，再熟悉 STAR 口述、机制、实验和追问。
2. 每个数字都要连同硬件、shape、精度、基线和指标口径一起讲；算子、代理、完整层与训练 step 收益不可互换。
3. R3 回放保证的是执行路由遵循目标，不是训练侧自然路由与推理自然路由自动一致。
4. FlashInfer rollout hang 的贡献为排障、回移上游位级修复及验证；FlashInfer #5171 是另一项测试兼容性贡献。
5. 128 卡 200-step 是本人确认的交付终态，不能拿它替代 8 卡 20-step 对照的详细量化证据；Mirage #755 是 demo 权重 padding 修复，不是通用 argmax mask 保证。

旧押题已归档到 `../旧的简历`；部分旧术语和早期方案与最终代码不一致，复习以本目录新稿及其来源为准。
