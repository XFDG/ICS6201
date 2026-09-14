# Open Doc：可分享技术文档索引

> 更新日期：2026-09-11  
> 组织原则：一项技术一篇独立 Markdown/PDF；旧综合稿保留，不以换标题方式重复已有专题。

## 本轮新增独立技术专题

| 序号 | 技术专题 | Markdown | PDF | 结论性质 |
|---:|---|---|---|---|
| 01 | 确定性 FA3 反向中的 dQ 依赖链调度 | [MD](topics/01_确定性_FA3_反向中的_dQ_依赖链调度_脱敏版.md) | [PDF](topics/01_确定性_FA3_反向中的_dQ_依赖链调度_脱敏版.pdf) | 已完成算子闭环，训练框架 E2E 待补 |
| 02 | MoE Router 正交损失的等价变换与子图融合 | [MD](topics/02_MoE_Router正交损失的等价变换与子图融合_脱敏版.md) | [PDF](topics/02_MoE_Router正交损失的等价变换与子图融合_脱敏版.pdf) | 四卡 MoE Layer proxy 已验证，默认关闭 |
| 03 | Sink-aware Top-K 融合中的顺序语义门禁 | [MD](topics/03_Sink-aware_TopK融合中的顺序语义门禁_脱敏版.md) | [PDF](topics/03_Sink-aware_TopK融合中的顺序语义门禁_脱敏版.pdf) | 性能通过、严格顺序未通过，NO-GO |
| 04 | NCCL 与 GEMM 跨 Stream 资源竞争的分层归因 | [MD](topics/04_NCCL与GEMM跨Stream资源竞争的分层归因_脱敏版.md) | [PDF](topics/04_NCCL与GEMM跨Stream资源竞争的分层归因_脱敏版.pdf) | 原 trace 归因 + 四卡语义 proxy，不外推 full step |
| 05 | CUDA Graph 下确定性 Router GEMM 多后端自动选核 | [MD](topics/05_CUDA_Graph下确定性Router_GEMM的多后端自动选核_脱敏版.md) | [PDF](topics/05_CUDA_Graph下确定性Router_GEMM的多后端自动选核_脱敏版.pdf) | Kernel/模型与 worker-engine 分层验证，显式 opt-in |
| 06 | 尾 Warp 归约中的静默错误与 Active Mask 修复 | [MD](topics/06_尾Warp归约中的静默错误与Active_Mask修复_脱敏版.md) | [PDF](topics/06_尾Warp归约中的静默错误与Active_Mask修复_脱敏版.pdf) | 18/18 边界用例逐位通过 |
| 07 | Blackwell 多精度训练的系统化验证方法 | [MD](topics/07_Blackwell多精度训练的系统化验证方法_脱敏版.md) | [PDF](topics/07_Blackwell多精度训练的系统化验证方法_脱敏版.pdf) | dense 系统/数值链路验证，不证明收敛 |
| 08 | 为什么单机 Prefill/Decode 分离可能更慢 | [MD](topics/08_为什么单机Prefill_Decode分离可能更慢_脱敏版.md) | [PDF](topics/08_为什么单机Prefill_Decode分离可能更慢_脱敏版.pdf) | 通路触发已验证，payload 未独立闭环 |

## 保留的旧综合稿

- [《大模型训推算子优化的证据闭环》Markdown](大模型训推算子优化的证据闭环_脱敏版_20260911.md)
- [《大模型训推算子优化的证据闭环》PDF](大模型训推算子优化的证据闭环_脱敏版_20260911.pdf)

综合稿仅作为全景索引，本轮没有撤回或覆盖。

## 已有内容：本轮不重复生成

### 已发布文章

- [一次 CUDA Graph Hang 的位级追踪：从 RPC 超时定位到 FTZ 哨兵误判](https://blog.csdn.net/XFDG01/article/details/163074432)
- [从 MoE 到 SonicMoE：为什么细粒度 MoE 需要 IO-aware 算子](https://blog.csdn.net/XFDG01/article/details/161791508)
- [AI 模型中的 Slot、Compact、Permute 等操作到底是什么意思？](https://blog.csdn.net/XFDG01/article/details/161649773)
- [Hopper TMA 学习笔记：从 GPU 内部 Tensor 版 DMA 理解 TMA](https://blog.csdn.net/XFDG01/article/details/160625925)

### 仓库中已有的公开/专题材料

- LLM 推理优化评测的四层证据链；
- MoE 推理优化的正交评测方法；
- 异步历史依赖算子与 CPU—GPU 同步消除；
- 模型权重盘点；
- R3 Router Replay；
- H200 Grouped GEMM / Quack / SonicMoE；
- DeepGEMM 离线预编译与 wheel；
- 自研量化推理 Runtime 的架构、H200 迁移与 Tensor Parallel。

以上材料继续保留在 `doc/work_records/`，本轮没有改名重写。

## 脱密口径

新增专题均删除或泛化了项目关联的公司、团队与人员、内部模型/任务代号、私有仓库与提交、主机/IP/端口、绝对路径、镜像、运行编号、原始日志/trace/profile 和内部绝对业务指标。保留作者署名、开源技术名称、通用机制、相对结果和明确的证据边界。对外发布前仍应执行所在组织的正式内容审批。
