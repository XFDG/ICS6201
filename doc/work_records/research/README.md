# AI Infra 仓库读码与技术调研

> 这些报告用于建立技术地图和支持工程选型。除文档明确标注的本地实验外，论文、README 或上游仓库中的性能数字均不是个人实测成果。

## 主题索引

| 主题 | 报告 | 重点 |
|---|---|---|
| 大模型并行训练 | [Megatron-LM](./megatron_lm_analysis_2026-06-08.md) | DP/TP/PP/CP/EP、MoE、overlap、软件栈 |
| GPU Kernel DSL | [Quack](./quack_analysis_2026-06-08.md) | CuTe-DSL、memory-bound kernel、JIT/autotune |
| Tile primitive 与通信融合 | [ThunderKittens / mKernel](./thunderkittens_mkernel_analysis_2026-06-08.md) | TMA/WGMMA、persistent kernel、AG-GEMM |
| 通信计算融合算子 | [HPC-Ops](./hpc_ops_analysis_2026-06-12.md) | AllGather/GEMM、GEMM/AllReduce、MoE dispatch |
| 编译与 persistent kernel | [Mirage / MPK](./mirage_analysis_2026-06-14.md) | superoptimizer、megakernel、task graph runtime |
| 性能建模 | [Frontier](./frontier_analysis_2026-06-10.md) | microbenchmark、模型仿真、trace、TCO |
| 训练诊断 | [ml_toolkit](./ml_toolkit_analysis_2026-06-11.md) | capture/align/replay、张量和路由诊断 |
| Agent 可观测性 | [DeepEye](./deepeye_analysis_2026-06-15.md) | workflow trace、安全模型、诊断入口 |
| Agent 框架 | [OpenManus](./openmanus_analysis_2026-06-15.md) | tool loop、sandbox、配置与扩展点 |
| MoE 算法与系统 | [从 MoE 到 SonicMoE](./从MoE到SonicMoE_公开资料版.md) | activation/HBM IO、padding、IO-aware fusion |

## 形成的能力地图

- 训练系统：并行维度、MoE dispatch、通信计算 overlap、checkpoint。
- GPU 性能：shape、latency、MFU、memory traffic、warmup/autotune 去噪。
- Kernel 路径：CUTLASS、CuTe-DSL、Triton、TMA/WGMMA、persistent kernel。
- 分布式算子：AllGather/GEMM、GEMM/AllReduce、DeepEP、Grouped GEMM。
- 诊断工具：Nsys/NCU、trace、capture/align/replay、统一指标口径。

## 使用方式

- 简历中只写“完成多仓库读码与技术选型报告”，不要把上游 benchmark 写成个人优化结果。
- 面试中围绕一个真实工程问题引用调研，例如用 Megatron MoE 数据流解释 R3 route 注入点，用 SM90 调优指南解释 Quack 候选选择。
- 报告中的本地路径是当时读码索引，不是公开复现环境；公开仓库地址优先参考报告列出的上游链接。

## 图表

配套架构图与数据位于 [assets](./assets/)；绘图脚本一并保留，便于核对图中聚合项。

