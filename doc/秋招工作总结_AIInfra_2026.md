# AI Infra 工作总结（秋招材料）

> 可直接选择的简历 bullet、岗位版本、STAR 故事和数字核对表见 [秋招简历素材库](./秋招简历素材库_AIInfra_2026.md)；精选开发文档与证据见 [工作记录总索引](./work_records/README.md)。

> 整理日期：2026-07-15  
> 记录范围：本机 `docs/` 中 2026-05-18 至 2026-06-27 的可追溯工作，以及 ICS6201 仓库中的实验结果。  
> 核心结论：这段工作的主线不是单点模型训练，而是围绕大模型系统完成“离线交付、GPU 热路径分析、训推一致性验证、分布式故障定位、实验工程化”的闭环。  
> 明确不包含：摩尔线程实习和更早的个人项目；它们已在现有简历中单独描述。本文件也不把公开仓库中的性能数字算作个人实测成果。

## 1. 秋招材料取舍

| 优先级 | 经历 | 建议用途 | 原因 |
|---|---|---|---|
| P0 | R3 / Router Replay 训推一致性 | 实习经历主项目 | 同时覆盖 RL 系统、MoE 路由、分布式训练、指标设计和根因定位，证据最完整 |
| P0 | H200 MoE / Grouped GEMM 性能分析 | 实习经历性能优化项目 | 有真实 profile、口径纠偏、回放验证和 tuning 收益 |
| P1 | DeepGEMM 离线预编译与 wheel 交付 | 实习经历工程化项目 | 能体现部署、构建、缓存系统和多模型验证能力 |
| P2 | ICS6201 无人机检测流水线 | 项目经历备选 | 能体现数据工程、多模型实验和调度恢复，但与 AI Infra 岗位相关度低于前三项 |
| P3 | 开源项目读码与技术报告 | 面试知识储备 | 适合证明技术视野，不宜包装成亲自实现或复现的性能结果 |

建议保持现有简历的 AI Infra 主线：R3 放在最前，Grouped GEMM / DeepGEMM 放在其后；ICS6201 仅在需要补充端到端训练项目时使用。不要把十余份调研报告逐项塞进简历。

## 2. 工作时间线

| 时间 | 工作主题 | 主要产出 |
|---|---|---|
| 2026-05-18 至 05-21 | DeepGEMM 离线预编译 | 359-kernel union bundle、单卡/TP=2 验证、2+1 wheel 交付 |
| 2026-05-23 至 06-05 | H200 Grouped GEMM / SonicMoE / Quack | 72 点四路 benchmark、线上线下口径对齐、1024 条 range 回放、config tuning |
| 2026-06-01 至 06-09 | ICS6201 无人机检测 | 171,568 样本数据流水线、6 模型实验框架、核心 6 个训练任务完成 |
| 2026-06-08 至 06-15 | AI Infra 开源技术调研 | Megatron-LM、Quack、ThunderKittens/mKernel、HPC-Ops、Mirage/MPK 等读码报告 |
| 2026-06-11 至 06-23 | R3 单机验证 | 8xH200 baseline/R2/R3 对照、route observe/replay、logprob drift 指标闭环 |
| 2026-06-24 至 06-27 | R3 集群化与故障定位 | 32 卡运行/恢复 SOP、TP=2 CUDA Graph hang 隔离、checkpoint 与 RLScope 调研 |

## 3. R3：MoE RL 训推路由一致性

### 3.1 问题与职责

在 `veRL + Megatron + vLLM + DeepEP/flex` 的 MoE RL 链路中，rollout/inference engine 与 training engine 可能因数值和执行路径差异选择不同 expert，继而放大 logprob drift。工作重点是把论文中的 Router Replay 转化为可运行、可观测、可对照的工程链路。

个人工作可归纳为四部分：

1. 在 rollout 侧采集 token-layer-topk 维度的 `routed_experts`，并将其传入训练侧。
2. 在 Megatron 训练侧按 response mask 执行 route observe/replay，区分 baseline、R2（记录但不回放）和 R3（回放）。
3. 统一 `route_mismatch_rate`、`topk_overlap`、`f_tau_2`、KL、`k3_kl`、selected logprob absolute difference 等指标口径。
4. 排查训练启动、vLLM rollout hang、checkpoint/恢复和多节点平台资源问题，沉淀运行与恢复 SOP。

### 3.2 已验证结果

8xH200、Qwen3-30B-A3B BF16、GSM8K 的 observe 口径 20-step 对照中：

| 指标 | Baseline / R2 | R3 | 判断 |
|---|---:|---:|---|
| 自然 route mismatch | 约 17%-19% | 0 | replay 机制按预期生效 |
| `f_tau_2` | 基线水平 | 低 36x-145x | 训推概率大幅漂移的 token 比例明显下降 |
| KL | 基线水平 | 低约 4x-7x | 概率分布更一致 |
| selected logprob abs diff | 基线水平 | 低约 2.2x-2.5x | 选中 token 的 logprob 偏差下降 |
| 稳定性 | - | 20 step 内 0 hang、0 `EngineDeadError`、0 `metric_error` | eager 口径链路稳定 |

其中 `f_tau_2` 表示训练与推理概率比超过 2 倍的 token 比例，越低越好。早期 5/25/50-step 对照也得到约 30x-46x 的下降，15-step 独立对照约为 77x；observe 口径的 20-step 结果是更适合对外陈述的主结论。

在 4 节点 x 8 卡的集群 smoke 中，`TP=2 + eager` 的 R2/R3 均完成 5 step：

| 指标 | R2 | R3 | 说明 |
|---|---:|---:|---|
| route mismatch | 0.2579 | 0 | 集群口径下 replay 同样生效 |
| KL | 0.00388 | 0.00089 | R3 约低 4.4x |
| throughput | 113.90 | 104.68 | R3 当前有约 8.1% 吞吐开销 |

### 3.3 故障定位与工程化

- 修复 `plt_num_loops` 错误透传到 Megatron、vLLM 包版本号不兼容等启动阻塞。
- 将 `sample_tokens timed out` 从 sampler/D2H 等下游等待点继续向前隔离，收敛到 `TP=2 + FULL CUDA Graph` 条件下 fused AllReduce + RMSNorm 路径不可靠；关闭 `fuse_allreduce_rms` 的候选 guard 首次通过完整 step。
- 梳理 32 卡 R2/R3 启动参数、checkpoint 保存/停卡恢复 SOP、日志与 rollout dump 边界。
- 对比两个集群配置组的日志，确认 fullgraph + checkpoint 组因 Ray memory pressure OOM，而 eager/no-resume 组能完成 5 step；明确 profiling 是观测工具，不是稳定性修复。

### 3.4 简历可用表述

- 面向 8xH200 的 `veRL + Megatron + vLLM` MoE-RL 链路，实现 rollout route 采集、训练侧 Router Replay 与 response-mask 指标闭环；observe 对照中 baseline/R2 自然路由 mismatch 稳定在 17%-19%，R3 回放后降至 0。
- 统一 `f_tau_2`、KL、route mismatch 与 selected-logprob drift 口径；20-step 对照中 R3 将 `f_tau_2` 降低约 36x-145x、KL 降低约 4x-7x，并实现全程 0 hang / 0 engine error / 0 metric error。
- 定位 vLLM `TP=2 + FULL CUDA Graph` rollout hang，排除 sampler、D2H 和消息队列等下游表象，将问题收敛至 fused AllReduce + RMSNorm 路径；同步沉淀 32 卡启动、checkpoint 和停卡恢复流程。

### 3.5 结论边界

- 当前能证明的是 route 与 logprob 层面的训推一致性改善，不能写成最终 reward 或任务效果已经提升。
- `TP=2 + FULL CUDA Graph` 尚未完成指令级根因闭环；eager 是稳定规避方案，不等于 fullgraph 已修复。
- 32 卡样本显示 R3 有约 8.1% 吞吐开销，简历和面试中不应只说收益、不说代价。

## 4. H200 MoE / GEMM 性能工程

### 4.1 DeepGEMM 离线预编译交付

目标是让 vLLM/DeepGEMM 在离线环境直接加载 cubin，跳过运行时 NVCC JIT。完成内容包括：

- 将预编译 bundle 从 109 扩展到 359 kernels，增幅约 230%。
- 覆盖 8 个 H200 单卡模型和 1 个 TP=2 模型，验证范围内均达到 `cold=0`。
- 首次跑通 Qwen3-8B-FP8 H200 TP=2 的 collect -> merge -> verify 闭环，预编译 load 相比 cold load 加速约 7.7x。
- 交付 baseline vLLM、独立 DeepGEMM 和 combined 三种 wheel；安装后可自动解析包内 bundle。
- 编写 union bundle 自动合并与 SHA-256 一致性校验流程，并修复 TP=2 环境的 NCCL/CUDA 版本冲突。

这里的 7.7x 是加载阶段收益，核心价值是消除 cold compile，不代表稳态 GEMM TFLOPS 提升。

### 4.2 SonicMoE / Quack 线上线下对齐

| 工作 | 结果 |
|---|---|
| 四路 benchmark | CUTLASS 2.x、Quack pure GEMM、SonicMoE e2e、DeepGEMM；18 shapes x 4 路径，共 72 个数据点 |
| 热路径判断 | Quack pure GEMM 平均约 66.5% MFU；SonicMoE e2e FWD/BWD 约 32.1%/20.8%，说明瓶颈不只在 GEMM 本体 |
| profile 去噪 | 识别 Triton autotune/warmup 污染；稳态下 `token_gather_sum` 占比由表象 69.9% 校正为 3.78%，Quack GEMM 约占 94.98% |
| 口径纠偏 | 发现 standalone 使用 `T*topk=195,976` rows，而线上使用 DeepEP compact 后 `TK_valid=29,446` rows；此前 4x-6x 差异并非 H200 算力问题 |
| 本地回放 | 从线上 profile 提取 1024 条 forward ranges，三个代表 shape 的本地/线上差异为 `+0.46%/-1.11%/-5.40%` |
| config tuning | 目标线上 shape 约 `+5.3%`，14 shapes x 3 expert distributions 整体约 `+1.84%` |

### 4.3 简历可用表述

- 面向 H200 MoE 热路径构建 CUTLASS/Quack/SonicMoE/DeepGEMM 四路 benchmark 与 Nsight 分析，识别 `T*topk` 全展开 rows 和 DeepEP compact rows 混用导致的 4x-6x 性能误判；基于 1024 条线上 range 回放，将三个代表 shape 的差异收敛至约 `+0.46%/-1.11%/-5.40%`。
- 扩展 Quack SM90 autotune 候选，目标线上 shape 获得约 `+5.3%` 加速，14 shapes x 3 expert distributions 整体提升约 `+1.84%`；同时确认 pure GEMM 平均 MFU 约 66.5%。
- 设计 DeepGEMM 离线预编译交付，将 bundle 从 109 扩展至 359 kernels，覆盖 8 个 H200 单卡模型及 Qwen3-8B-FP8 TP=2，验证范围内实现 cold compile 清零并完成 wheel-only 交付。

### 4.4 结论边界

- DeepGEMM 的预编译收益与 Quack 热路径 MFU 属于不同指标，不可直接横向比较。
- Quack 改动是 config-level tuning，整体收益约 1%-2%，不应描述成重写 kernel 后的大幅加速。
- 部分阶段因 NCU profiling 权限受限，只能使用 Nsys、时间和理论 FLOPs/IO；不能声称已完成全部硬件计数器分析。

## 5. ICS6201：无人机可见光检测流水线

### 5.1 项目范围

项目目标是比较 YOLOv8n、YOLOv10n、YOLO11m、RT-DETR-L、Faster R-CNN R50-FPN、DDW-YOLO 六种检测器。三套数据集共 171,568 张图像，原计划为 6 模型 x 3 seeds，共 18 个正式任务。

完成的工程工作：

- 将 DUT/DroneDetectionDataset 的 VOC 标注转换为 YOLO，并生成 Detectron2 所需 COCO 标注。
- 将 ARD-MAV 原脚本的逐帧 ffmpeg 调用改为按视频批量抽帧，预计约 18 天的流程缩短至 10-20 分钟。
- 建设环境检查、极小子集 smoke、多 GPU launcher、结果汇总和一键入口，最终 14/14 项快速验证通过，总耗时约 5 分钟。
- 修复“任务预绑定 GPU + 全局线程池”导致空闲 worker 把任务派到忙卡并 OOM 的问题，改为每 GPU worker 队列，并增加 manifest 驱动的 complete/resume/pending 判定。
- 完成核心 RT-DETR-L 与 Faster R-CNN 各 3 seeds；secondary 模型只完成或部分完成一部分，不能写成 18 个任务全部完成。

### 5.2 最终核心结果

不同框架按各自原生标度记录，未强行混成一个指标：

| 模型 | seed 1 | seed 2 | seed 3 | 均值 |
|---|---:|---:|---:|---:|
| RT-DETR-L `mAP50-95` | 0.67942 | 0.67529 | 0.67576 | 0.67682 |
| Faster R-CNN `bbox/AP` | 64.6024 | 64.6664 | 64.4290 | 64.5660 |

Faster R-CNN 三组均完成 2,062,560 iterations；完整原始日志和权重已在仓库瘦身时删除，提取后的指标保存在 `docs/ics6201/assets/ics6201_final_metrics_2026-06-07.csv`。

### 5.3 简历可用表述

- 面向 171,568 张无人机图像构建 VOC -> YOLO/COCO 数据流水线与 8xH200 多模型训练框架；将 ARD-MAV 逐帧 ffmpeg 方案改为视频级批处理，把预计 18 天的数据准备缩短至 10-20 分钟。
- 设计 per-GPU worker 调度与 manifest 恢复机制，解决静态绑卡导致的显存冲突；完成 RT-DETR-L/Faster R-CNN 各 3 seeds 核心实验，RT-DETR-L `mAP50-95` 均值为 0.6768。

## 6. 开源项目读码与技术沉淀

这部分工作的价值是建立 AI Infra 技术地图和后续选型依据。除明确写有本地 benchmark 的项目外，报告中的上游性能数字均来自论文或 README，不作为个人性能成果。

| 主题 | 阅读/分析对象 | 形成的认识或产出 |
|---|---|---|
| 大模型并行训练 | Megatron-LM | 梳理 DP/TP/PP/CP/EP、通信计算 overlap、DeepEP/FA3/GroupedGEMM 集成位置 |
| GPU Kernel DSL | Quack、ThunderKittens | 对比 CuTe-DSL 与 CUDA tile primitive，理解 memory-bound reduction、TMA/WGMMA 和 SM90/SM100 路径 |
| 通信计算融合 | mKernel、HPC-Ops | 梳理 AllGather/GEMM、GEMM/AllReduce、MoE dispatch 与 persistent kernel 的融合方式 |
| 编译与执行模型 | Mirage/MPK | 分析 tensor program superoptimizer、persistent megakernel、in-kernel scheduler 与 task graph runtime |
| 性能评测 | xpu-perf、Frontier | 沉淀从 microbenchmark、模型仿真、trace 到端到端吞吐/TCO 的口径分层 |
| 训练诊断 | ml_toolkit、DeepEye | 评估 capture/align/replay、张量诊断、分布式 trace 和 R3 consistency toolkit 的实现路径 |
| MoE 算法与系统 | SonicMoE 及相关论文 | 梳理 fine-grained MoE 的 activation/HBM IO、Grouped GEMM padding 与 IO-aware fusion |

## 7. 能力归纳

| 能力 | 可验证证据 |
|---|---|
| GPU 性能分析 | Nsight Systems range、MFU/latency 口径、warmup/autotune 去噪、线上 shape 回放 |
| Kernel 与算子优化 | SM90 Grouped GEMM config tuning、CUTLASS/Quack/DeepGEMM 路径比较 |
| 分布式大模型系统 | veRL、Megatron、vLLM、DeepEP，8 卡单机和 32 卡多节点运行与恢复 |
| 实验设计 | baseline/R2/R3 控制变量、route/logprob 指标定义、多 seed、错误边界说明 |
| 部署与交付 | 离线 cubin bundle、wheel-only 安装、SHA-256 校验、无外网环境处理 |
| 故障定位 | CUDA Graph hang、Ray OOM、调度器 OOM、NCCL/CUDA 环境冲突、数据 I/O 瓶颈 |
| 文档与复现 | 周报、技术报告、命令、数据表、文件索引和安全凭据规则 |

## 8. 面试自述短版

我最近的工作集中在 H200 上的大模型系统和 GPU 性能工程。一条主线是 MoE RL 的训推一致性：在 veRL、Megatron 和 vLLM 之间打通 Router Replay，设计 route 与 logprob 指标，并通过 baseline/R2/R3 对照验证 R3 能把自然路由 mismatch 从约 17%-19% 降到 0，同时显著降低概率漂移。另一条主线是 H200 MoE 热路径分析：用 Nsight 和本地回放纠正 compact rows 的测试口径，完成 Quack config tuning；同时做过 DeepGEMM 离线预编译和 wheel 交付。我比较重视结论边界，例如区分 cold-load 加速和稳态 kernel 加速，也会明确 R3 当前改善的是一致性指标，而不是直接宣称最终任务效果提升。

## 9. 精选证据索引

公开仓库只保留经过筛选和脱敏的开发文档、聚合数据与图表：

| 主题 | 公开材料 |
|---|---|
| R3 工程 | [Router Replay 工程实现](./work_records/r3/R3_工程实现公开版.md) |
| R3 结果 | [A/B/C 实验结果](./work_records/r3/R3_实验结果公开版.md) |
| CUDA Graph / 集群恢复 | [CUDA Graph 与集群排障](./work_records/r3/R3_CUDA_Graph与集群排障公开版.md) |
| H200 GEMM | [Grouped GEMM 性能分析](./work_records/gemm_sonicmoe/H200_GEMM性能分析公开版.md) |
| Quack tuning | [配置调优开发记录](./work_records/gemm_sonicmoe/Quack调优开发记录公开版.md) |
| DeepGEMM | [离线预编译与 Wheel 交付](./work_records/deep_gemm/README.md) |
| ICS6201 | [无人机检测项目记录](./work_records/ics6201/README.md) |
| 技术调研 | [AI Infra 仓库读码索引](./work_records/research/README.md) |
| 周报主线 | [周报时间线](./work_records/weekly_report/README.md) |
