# 秋招简历素材库：AI Infra / GPU 性能 / 训练系统

> 整理日期：2026-07-15  
> 使用方法：这是“候选句库”，不是一份应全部放入简历的成稿。每个项目选择 2-3 条最匹配 JD 的 bullet，并确保面试时能讲清口径、对照组、失败过程和结论边界。  
> 证据入口：[工作记录总索引](./work_records/README.md)

## 1. 经历定位

这段工作的主线可以统一表述为：

> 围绕 H200 大模型系统，完成 MoE-RL 训推一致性验证、Grouped GEMM 性能分析与调优、DeepGEMM 离线预编译交付，并将单机实验推进到多节点运行、恢复和故障定位。

建议按岗位选择不同叙事：

| 目标岗位 | 第一项目 | 第二项目 | 第三项目 |
|---|---|---|---|
| AI Infra / 大模型系统 | R3 Router Replay | H200 GEMM | DeepGEMM 离线交付 |
| CUDA / Kernel 性能 | H200 GEMM | DeepGEMM | R3 CUDA Graph 排障 |
| 分布式训练 / RL Infra | R3 Router Replay | R3 集群与恢复 | ICS6201 多 GPU 调度 |
| 算法工程 / 训练工程 | R3 实验设计 | ICS6201 | AI Infra 读码调研 |

一页简历建议最多使用 6 条：R3 三条、GEMM 两条、DeepGEMM 或 ICS 一条。不要把所有数字塞进同一项目。

## 2. 项目命名建议

| 用途 | 推荐名称 |
|---|---|
| 简历主项目 | MoE 强化学习训推一致性与 Router Replay |
| 性能项目 | H200 MoE Grouped GEMM 性能分析与配置调优 |
| 交付项目 | DeepGEMM 离线预编译与 Wheel 交付 |
| 课程项目 | 无人机目标检测数据与多 GPU 训练流水线 |

技术栈可写：

> Python, PyTorch, CUDA, veRL, Megatron-LM, vLLM, DeepEP, CUTLASS/CuTe-DSL, DeepGEMM, Nsight Systems, NCU, Ray, NCCL, Docker, Git

只保留自己确实使用或深入读过的技术；NCU 在 R3 问题上未形成最终证据，面试中应主动说明。

## 3. R3 / Router Replay Bullet 池

### 3.1 系统实现

- 面向 8xH200 的 veRL + Megatron-LM + vLLM MoE-RL 链路，实现 rollout 侧 token-layer-top-k route 采集、训练侧 Router Replay 与 response-mask 指标闭环，打通从推理引擎到训练引擎的数据协议。
- 设计 baseline-observe / R2-observe / R3-replay 三组控制变量实验，保留训练侧自然路由 <code>train_routed_experts</code>，修正“replay target 自比较导致 mismatch 恒为 0”的早期指标口径。
- 为 route tensor 缺失、shape 不一致、空 response mask 等异常增加显式 <code>metric_error</code>，并覆盖 18 个 CPU 测试；GPU 侧按短 smoke、20-step 主实验和多节点验证分层验收。

### 3.2 实验结果

- 在 Qwen3-30B-A3B BF16 + GSM8K 的 20-step observe 对照中，测得 baseline/R2 自然 route mismatch 稳定在约 17%-19%，R3 replay 后全程为 0。
- 统一 f_tau_2、KL、k3-KL 和 selected-logprob drift 口径；20-step 中 R3 将 f_tau_2 降低约 36x-145x、KL 降低约 4x-7x、selected logprob absolute difference 降低约 2.2x-2.5x。
- 在 eager 口径下完成 20-step 稳定性验证，全程 0 hang、0 EngineDeadError、0 metric_error；每步覆盖约 241 万至 269 万有效 token-layer 观测点。
- 将链路扩展到 4 节点 x 8 卡、TP=2 的 R2/R3 5-step smoke，route mismatch 从 0.2579 降至 0、KL 从 0.00388 降至 0.00089，同时量化约 8.1% 吞吐开销。

### 3.3 故障定位与工程化

- 使用 TP、compile、CUDA Graph mode 和 fusion 四维控制变量矩阵排查 vLLM rollout hang，排除 sampler/D2H/MQ 等下游表象，将问题收敛到 TP=2 + FULL CUDA Graph 下 fused AllReduce + RMSNorm 高风险路径。
- 形成多节点 checkpoint 保存/恢复 smoke、停卡恢复和 profile 产物验收流程，区分训练 checkpoint、rollout dump、实验指标与 Nsys/NCU 产物，降低中断后的重跑成本。
- 修复非 Megatron 参数透传导致初始化失败、vLLM 开发版本元数据不兼容等启动阻塞，并以 eager 模式提供稳定规避方案。

### 3.4 推荐组合

AI Infra 简历优先选以下三条：

1. 系统实现第一条；
2. 20-step route + drift 结果合并成一条；
3. CUDA Graph 排障或 32 卡 smoke 二选一。

压缩成三条的可用版本：

- 在 8xH200 的 veRL + Megatron + vLLM MoE-RL 链路中实现 rollout route 采集、训练侧 Router Replay 与 response-mask 指标闭环，并以 baseline/R2/R3 observe 对照验证自然 route divergence。
- Qwen3-30B-A3B BF16 20-step 实验中，将 route mismatch 从约 17%-19% 降至 0，f_tau_2 降低约 36x-145x、KL 降低约 4x-7x，且全程 0 hang / 0 engine error / 0 metric error。
- 通过 TP/compile/graph/fusion 控制变量矩阵定位 vLLM TP=2 FULL CUDA Graph hang，将问题收敛至 fused AllReduce + RMSNorm 路径，并沉淀 eager 规避、checkpoint 与多节点恢复流程。

## 4. H200 Grouped GEMM / SonicMoE Bullet 池

### 4.1 Benchmark 与 Profile

- 构建 CUTLASS 2.x、Quack、DeepGEMM、SonicMoE e2e 四路 H200 benchmark，覆盖 18 shapes x 4 路径共 72 个逻辑数据点；Quack pure GEMM 平均约 66.5% MFU，SonicMoE e2e FWD/BWD 约 32.1%/20.8%。
- 使用 Nsys range 和重复预热剥离 Triton autotune/JIT 污染，将 token_gather_sum 占比从首轮表象 69.9% 校正为稳态 3.78%，确认 Quack GEMM 在稳态区间约占 94.98%。
- 建立 pure GEMM 与 routing/gather/communication/combine 的分层测量口径，避免将 e2e MFU 低直接归因到单个 GEMM kernel。

### 4.2 线上线下口径纠偏

- 识别 standalone 使用 T*top-k=195,976 rows、线上 DeepEP 使用 compact TK_valid=29,446 rows 的口径混用，推翻此前 4x-6x 性能差距判断。
- 从线上 profile 聚合 1,024 条 forward ranges，按 min/target/max shape 构建本地回放，将三个代表 shape 的本地/线上差异收敛至 +0.46% / -1.11% / -5.40%。
- 统一 compact rows、expert-M 分布、padding 和 FLOPs 定义，为后续 autotune 和线上性能回归建立可比较基线。

### 4.3 Config Tuning

- 扩展 Quack SM90 autotune 候选，围绕 tile、cluster、persistent schedule 和 swizzle 对 14 shapes x 3 expert distributions 进行 gated/out 双算子 sweep。
- 目标线上 shape 获得约 5.3% 加速，完整 workload 聚合 latency 提升约 1.84%，weighted MFU 从 60.94% 提升至 62.06%（+1.12 pp）。
- 对非法配置、JIT 进程级失败、warmup 和分布偏斜建立筛选与记录规则；保留完整 CSV、top-config 表和绘图脚本以支持结果复核。

### 4.4 推荐组合

CUDA/性能岗位推荐：

- 面向 H200 MoE 热路径构建四路 benchmark 与 Nsys 分层分析，覆盖 18 shapes x 4 路径；测得 Quack pure GEMM 平均约 66.5% MFU，并定位 e2e FWD/BWD 仅约 32.1%/20.8%。
- 识别 T*top-k 全展开 rows 与 DeepEP compact rows 混用导致的 4x-6x 性能误判；基于 1,024 条线上 range 聚合回放，将三个代表 shape 的本地/线上差异收敛至约正负 5%。
- 扩展 Quack SM90 autotune 候选，目标 shape 提升约 5.3%，14 shapes x 3 expert distributions 聚合提升约 1.84%，weighted MFU 提升 1.12 pp。

## 5. DeepGEMM 离线交付 Bullet 池

- 设计 DeepGEMM cubin 离线预编译流程，将多模型 JIT 产物合并为 union bundle，kernel 覆盖从 109 扩展至 359，增长约 230%。
- 覆盖 8 个 H200 单卡模型和 Qwen3-8B-FP8 TP=2，验证范围内实现 cold compile 清零；TP=2 precompiled load 相比 cold load 约加速 7.7x。
- 交付 baseline vLLM 453MB、独立 DeepGEMM 17MB 和 combined 469MB 三种 wheel 形态，实现安装后自动解析包内 bundle。
- 建立 bundle union、去重和 SHA-256 一致性检查，避免多机离线安装时 cubin 缺失或版本漂移。
- 排查并修复 TP=2 环境的 NCCL/CUDA 包冲突，完成 collect -> merge -> verify 端到端闭环。

推荐压缩成两条：

- 面向隔离网络的 H200 推理环境设计 DeepGEMM cubin 预编译与 wheel-only 交付，将 union bundle 从 109 扩展至 359 kernels，覆盖 8 个单卡模型及 Qwen3-8B-FP8 TP=2，验证范围内 cold compile 清零。
- 建立多模型 bundle 合并、SHA-256 校验和 baseline/独立/combined 三类 wheel 交付；TP=2 precompiled load 相比 cold load 约加速 7.7x，并修复 NCCL/CUDA 依赖冲突。

## 6. ICS6201 项目 Bullet 池

- 面向 171,568 张无人机可见光图像建设 VOC -> YOLO/COCO 数据流水线和六检测器实验框架，统一数据校验、训练入口、评估与结果汇总。
- 将 ARD-MAV 原逐帧 ffmpeg 进程调用改为按视频批量抽帧，把预计约 18 天的数据准备缩短至 10-20 分钟。
- 将“任务预绑定 GPU + 全局线程池”改为 per-GPU worker queue，解决空闲 worker 把任务派到忙卡引发的显存冲突与 OOM。
- 设计 manifest 驱动的 complete/resume/pending 状态机和断点恢复流程，并完成环境、数据、模型、launcher 共 14/14 项快速验证，耗时约 5 分钟。
- 完成 RT-DETR-L 与 Faster R-CNN 各 3 seeds 核心实验；RT-DETR-L mAP50-95 均值 0.67682，Faster R-CNN bbox/AP 均值 64.5660。
- 为跨框架结果保留原生指标标度和 seed 明细，不把 0-1 mAP 与 0-100 AP 直接混合比较。

推荐两条：

- 面向 171,568 张无人机图像构建 VOC -> YOLO/COCO 数据流水线与多 GPU 训练框架；将逐帧 ffmpeg 改为视频级批处理，把预计约 18 天的数据准备缩短至 10-20 分钟。
- 设计 per-GPU worker queue 与 manifest 恢复机制，解决静态绑卡导致的 OOM；完成 RT-DETR-L/Faster R-CNN 各 3 seeds，RT-DETR-L mAP50-95 均值为 0.6768。

## 7. 技术调研 Bullet 池

这部分只适合放一条，或作为面试补充：

- 对 Megatron-LM、Quack、ThunderKittens/mKernel、HPC-Ops、Mirage/MPK 等 AI Infra 项目开展源码级调研，梳理 MoE 并行、通信计算融合、TMA/WGMMA、persistent kernel 和性能诊断方法，形成可追溯技术报告与图表。
- 建立从 microbenchmark、kernel trace、模型仿真到 e2e throughput/TCO 的性能口径分层，用于指导 R3 和 Grouped GEMM 的实验设计。

不能写：

- “优化 Megatron-LM 达到某 MFU”，因为报告中的上游性能数字不是个人实测。
- “实现 ThunderKittens/mKernel”，因为工作内容是读码和技术分析。

## 8. 一页简历组合

### 8.1 AI Infra 版本

R3：

- 在 8xH200 的 veRL + Megatron + vLLM MoE-RL 链路中实现 rollout route 采集、训练侧 Router Replay 与 response-mask 指标闭环。
- 20-step observe 对照中将 route mismatch 从约 17%-19% 降至 0，f_tau_2 降低约 36x-145x、KL 降低约 4x-7x。
- 定位 TP=2 FULL CUDA Graph rollout hang 至 fused AllReduce + RMSNorm 高风险路径，沉淀 eager 规避和多节点 checkpoint/恢复流程。

GEMM / DeepGEMM：

- 纠正 T*top-k 与 DeepEP compact rows 混用造成的 4x-6x 性能误判，基于线上 range 聚合回放将代表 shape 差异收敛至约正负 5%。
- 扩展 Quack SM90 候选，目标 shape 提升约 5.3%、完整 workload 聚合提升约 1.84%。
- 将 DeepGEMM 离线 bundle 从 109 扩展到 359 kernels，覆盖 8 个单卡模型及 TP=2，验证范围内 cold compile 清零。

### 8.2 CUDA / Kernel 性能版本

- 构建 CUTLASS/Quack/DeepGEMM/SonicMoE 四路 H200 benchmark 与 Nsys range，覆盖 18 shapes x 4 路径。
- 通过 warmup/JIT 去噪和 compact-row 口径纠偏，推翻 4x-6x 假差距并建立线上 shape 本地回放。
- 扩展 SM90 tile/cluster/persistent/swizzle 候选，目标 shape 提升 5.3%，聚合提升 1.84%。
- 将 CUDA Graph hang 收敛到 TP=2 FULL graph 下 fused collective + normalization 路径，并明确 NCU 未完成的证据边界。
- 完成 DeepGEMM 359-kernel cubin bundle 和 wheel-only 离线交付。

### 8.3 分布式训练版本

- 在 rollout/trainer 跨引擎链路传递 token-layer-top-k route，设计自然 route observe 与 replay 数据契约。
- 以 8 卡 A/B/C 和 32 卡 R2/R3 smoke 验证一致性收益，并量化 32 卡短程吞吐开销约 8.1%。
- 形成 checkpoint 保存/恢复 smoke、停卡恢复和实验产物分层管理。
- 使用 TP、graph、compile、fusion 控制变量定位多卡 rollout hang。
- 在课程项目中实现 per-GPU worker queue 和 manifest 断点恢复，处理多任务显存冲突。

## 9. STAR 面试故事

### 9.1 R3 指标从“自洽”升级为“真实 observe”

**S**：早期 R3 route mismatch 为 0，看起来机制完全生效，但这个数字可能来自 replay target 与自身比较。  
**T**：证明 rollout route 与训练自然 route 的真实差异，并区分 observe 与 replay。  
**A**：保留 <code>train_routed_experts</code>，让 A/B/C 都采集 rollout route；A/B 只观测，C 才 replay；加入 response mask、有效 token-layer 数和 metric_error。  
**R**：A/B 稳定测得约 17%-19% mismatch，C 为 0；指标从自一致性检查升级为真实对照证据。  
**追问边界**：0 证明 replay route 生效，不证明 reward 改善。

### 9.2 vLLM CUDA Graph Hang

**S**：rollout 最终报 sample_tokens timeout，堆栈容易让人误判 sampler 或消息队列。  
**T**：找到首因条件并恢复实验稳定性。  
**A**：建立 TP、compile、graph mode、fusion 控制变量矩阵；用 eager 证明基本链路；Nsys 对齐上游 kernel 与下游等待点；单独 guard fused AllReduce + RMSNorm。  
**R**：问题收敛到 TP=2 + FULL graph 的 fused 路径，guard 首次通过完整 step，eager 成为稳定规避。  
**追问边界**：没有目标 NCU，不能声称定位到具体指令。

### 9.3 GEMM 4x-6x 假差距

**S**：standalone 比线上慢 4x-6x，初步怀疑 H200 kernel 或 gather。  
**T**：判断差距是真实 kernel 问题还是输入/测量口径问题。  
**A**：拆分 warmup/autotune，比较 T、T*top-k 和 DeepEP compact rows；从线上 range 聚合代表 shape 回放。  
**R**：发现 195,976 与 29,446 rows 混用，统一口径后本地/线上差异收敛到约正负 5%。  
**追问价值**：先修 benchmark，再做优化，避免对错误瓶颈投入。

### 9.4 DeepGEMM 离线交付

**S**：隔离网络环境中 DeepGEMM 首次运行需要 NVCC JIT，部署慢且结果依赖本机编译环境。  
**T**：提供可重复、可校验的预编译交付。  
**A**：收集多模型 cubin，union/去重后打入 wheel，增加 SHA-256 校验，并做单卡与 TP=2 cold=0 验证。  
**R**：bundle 109 -> 359 kernels，覆盖 8 个单卡模型和一个 TP=2 模型；TP=2 load 约加速 7.7x。  
**追问边界**：这是 cold-load，不是热路径 TFLOPS。

### 9.5 多 GPU 调度 OOM

**S**：任务预先绑定 GPU，但由全局线程池拉取，空闲 worker 可能执行绑定到忙卡的任务。  
**T**：消除错误派卡和不可预测 OOM，并支持恢复。  
**A**：改成每张 GPU 独立 worker queue，任务状态写入 manifest，启动时判定 complete/resume/pending。  
**R**：显存冲突消失，核心 6 个正式任务完成，失败任务可以按 manifest 恢复。  
**追问设计**：GPU 是资源所有者，queue 应跟资源绑定，而不是只在任务元数据中写 device id。

### 9.6 数据准备 18 天到分钟级

**S**：ARD-MAV 逐帧启动 ffmpeg，进程创建和重复 seek 使预计耗时达到约 18 天。  
**T**：在不改变帧选择语义的前提下降低 I/O 和进程开销。  
**A**：按视频一次性批量抽帧，再统一映射标注和校验帧数。  
**R**：预计流程缩短到 10-20 分钟。  
**追问验证**：抽样核对帧 id、标注数量和图像可读性，并纳入 14 项 smoke。

## 10. 自我介绍

### 30 秒版

我最近主要做 H200 上的大模型系统和 GPU 性能工程。一条主线是 MoE-RL 训推一致性，在 veRL、Megatron 和 vLLM 间打通 Router Replay，20-step 对照中把自然 route mismatch 从约 17%-19% 降到 0，并显著降低 logprob drift。另一条主线是 H200 Grouped GEMM，通过 Nsys 和线上 shape 回放纠正测试口径，再做 Quack SM90 config tuning；同时完成过 DeepGEMM 离线 cubin 与 wheel 交付。

### 90 秒版

我的工作可以分成系统一致性、GPU 性能和工程交付三部分。系统侧，我在 8xH200 的 veRL + Megatron + vLLM MoE-RL 链路中实现 rollout route 采集和训练侧 Router Replay，并把指标从早期自一致性升级为真实 observe 口径。Qwen3-30B-A3B 的 20-step A/B/C 对照中，baseline/R2 的自然 route mismatch 约 17%-19%，R3 为 0，f_tau_2 下降约 36x-145x。我还把多卡 rollout hang 从 sampler timeout 的表象收敛到 TP=2 FULL CUDA Graph 下 fused AllReduce + RMSNorm 路径。

性能侧，我做了 CUTLASS、Quack、DeepGEMM 和 SonicMoE 四路 H200 benchmark。最关键的工作不是直接改 kernel，而是发现 standalone 用 T*top-k rows、线上用 DeepEP compact rows，导致 4x-6x 假差距；统一口径后代表 shape 回放差异缩小到约正负 5%，再做 SM90 config tuning取得目标 shape 5.3%、聚合 1.84% 的提升。交付侧，我将 DeepGEMM bundle 从 109 扩展到 359 kernels，并做成离线 wheel。我在表达结果时会区分 cold-load 与 hot kernel，也会明确 R3 目前证明一致性改善，而不是直接宣称 reward 提升。

## 11. 技能与证据矩阵

| 能力关键词 | 直接证据 | 面试可展开问题 |
|---|---|---|
| PyTorch / 分布式 | 8 卡与 32 卡 R3、per-GPU worker | rank、collective、资源所有权 |
| Megatron / vLLM / veRL | rollout route 到 trainer replay | 数据契约、并行与权重同步 |
| MoE | route mismatch、DeepEP compact rows | top-k、expert-M、dispatch/combine |
| CUDA 性能 | Nsys range、MFU、warmup 去噪 | latency 口径、FLOPs、同步 |
| CUTLASS / CuTe-DSL | Quack SM90 config tuning | tile、cluster、swizzle、persistent |
| CUDA Graph | TP=2 FULL graph hang | capture/replay、rank 对称性、fusion |
| 实验设计 | A/B/C、多 seed、smoke 分层 | 控制变量、统计口径、外推边界 |
| 部署交付 | cubin bundle、wheel、SHA-256 | JIT、ABI/依赖、离线安装 |
| 故障恢复 | checkpoint smoke、manifest resume | 状态完整性、幂等、失败恢复 |
| 技术写作 | 专题报告、周报、图表和索引 | 如何保证数字可追溯 |

## 12. 数字核对表

| 数字 | 正确含义 | 不可写成 |
|---|---|---|
| 17%-19% -> 0 | 20-step observe 的 route mismatch | reward 提升 17%-19% |
| 36x-145x | R3 相对 A/B 的 f_tau_2 下降范围 | 36%-145% |
| 4x-7x | KL 下降范围 | 所有模型通用收益 |
| 8.1% | 32 卡 5-step smoke 中当前吞吐开销 | 长跑最终开销 |
| 66.5% MFU | Quack pure GEMM 选定 shape 平均 | SonicMoE e2e MFU |
| 32.1% / 20.8% | SonicMoE e2e FWD/BWD | 单 kernel 峰值 |
| 1.84% | 14 shapes x 3 分布 config tuning 聚合收益 | kernel 重写收益 |
| 5.3% | 目标 shape 的配置级收益 | 全 workload 平均 |
| 7.7x | DeepGEMM TP=2 precompiled load vs cold load | 稳态 GEMM 加速 |
| 109 -> 359 | union bundle kernel 数 | 模型数 |
| 171,568 | 三套无人机数据总图像数 | 训练样本均被所有模型使用 |
| 18 天 -> 10-20 分钟 | 数据准备流程估算与优化后范围 | 实测严格加速倍数 |
| 6 个任务 | 两核心模型各 3 seeds | 原计划 18 个任务全部完成 |

## 13. 明确不要写

- 不写“R3 提高了最终 reward/准确率”，因为没有完整任务效果对照。
- 不写“修复 vLLM FULL CUDA Graph bug”，因为只有路径收敛和候选 guard，缺指令级闭环。
- 不写“个人 PR 已合并”，整理时目标分支已有等价能力，当前没有可验证的合并状态。
- 不写“重写 Quack kernel 获得 5.3%”，实际是 config-level candidate tuning。
- 不写“DeepGEMM 推理加速 7.7x”，实际是 cold/load 阶段。
- 不写“完成 18 个检测任务”，核心完成量为 6 个。
- 不把调研报告中的上游 README/论文性能数据写成个人实测。
- 不在简历或公开仓库放内部 URL、节点、镜像、模型路径、run id 或任何凭据。

## 14. 证据入口

| 项目 | 公开材料 |
|---|---|
| R3 实现 | [工程实现公开版](./work_records/r3/R3_工程实现公开版.md) |
| R3 结果 | [实验结果公开版](./work_records/r3/R3_实验结果公开版.md) |
| R3 排障 | [CUDA Graph 与集群排障](./work_records/r3/R3_CUDA_Graph与集群排障公开版.md) |
| H200 GEMM | [性能分析公开版](./work_records/gemm_sonicmoe/H200_GEMM性能分析公开版.md) |
| Quack tuning | [调优开发记录](./work_records/gemm_sonicmoe/Quack调优开发记录公开版.md) |
| DeepGEMM | [离线交付索引](./work_records/deep_gemm/README.md) |
| ICS6201 | [项目记录索引](./work_records/ics6201/README.md) |
| 技术调研 | [仓库读码索引](./work_records/research/README.md) |
| 时间线 | [周报索引](./work_records/weekly_report/README.md) |
