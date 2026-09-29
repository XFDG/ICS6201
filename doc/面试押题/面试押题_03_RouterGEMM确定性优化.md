# 面试押题 03｜确定性 MoE Router GEMM

更新：2026-09-29。对应九坤实习中的“确定性 MoE Router GEMM”。本文是个人面试准备材料，内部路径仅供自查，不用于对外分发。

## 0. 开源仓库技术介绍与应用

[vLLM](https://github.com/vllm-project/vllm) 是大模型推理与服务引擎，围绕连续批处理、PagedAttention/KV cache 管理及并行执行提高服务吞吐；[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) 则是底层矩阵乘库。两者的关系是“服务执行框架调用计算内核”，不是同一层级的产品。

**典型应用：**在线文本生成、离线批量生成和 RL rollout。本题在 vLLM 的 Router 调用处选择、预检并缓存适合 shape/dtype 的内核，既要局部更快，也要守住跨 batch 确定性和 CUDA Graph 约束。

## 1. 先讲清系统、问题和我的职责

vLLM 是推理执行框架；MoE Router 根据 hidden states 计算每个 expert 的分数，再选 top-k expert。这里优化的是 Router gate 的小矩阵乘，不是整个专家 FFN GEMM，也不是重新设计 MoE 路由算法。实测主模型是 48 层 M0-Math，形状为 `A[M,2048] @ W[128,2048].T`，top-8 路由；它是 IQuest 模型研发过程中的验证模型，不能把全部数字说成 IQuest-Q1（320B-A15B）整模测得。

我的工作包括：定位 batch-invariant（BI）模式下的 Router 热点，实现 Triton full-K 候选与 dtype/shape 选路，将 DeepGEMM 接入 vLLM，补充 capture 前预检、缓存、失败回退和真实模型验收。DeepGEMM 的基础 GEMM 实现不是我的原创；团队 HPC MoE 的 caller-owned workspace 也不能归到我个人的 Router kernel 实现中。

## 2. STAR 口述版（约 90 秒）

**S｜背景。** 训推一致性要求同一个请求不因 batch 组成变化而改变结果，因此推理开启了 BI；但 decode 中 Router 的小 M GEMM 退化明显，48 层重复调用把固定开销放大。

**T｜任务。** 我负责在不放松确定性和路由正确性的前提下，降低 Router 开销，并保证优化能安全进入 vLLM 的编译与 CUDA Graph 路径。

**A｜行动。** 我先用 profiler 还原真实 Router 调用，用 NCU 确认原 persistent 的大 tile 在小 M 下只有一个 CTA，SM 利用率约 0.41%。随后沿 M/N 切小 tile、保留完整 K 归约，按 dtype 建立 DeepGEMM、Triton full-K、persistent 分层选择；所有 JIT、数值和 Graph 预检在 capture 前完成，正式 replay 只读已缓存决定。验证上既看矩阵误差，也看跨 batch needle row、top-8 路由、逐 token 输出和 TP rank 一致性。

**R｜结果。** BF16 的历史 48 层 Router 聚合耗时中位下降 73.39%，135/135 kernel 与 20/20 整模型场景逐位一致。另一组 prefix-cache 命中请求实验中，在 async OE 已开启时增加 Router 优化，TP1/TP2 整请求吞吐分别提升 3.81%/4.36%。这些是不同测试口径，不相加，也不把 kernel 降时说成整模型加速 73%。

## 3. 为什么慢：不是“确定性必须单 CTA”

旧 persistent 的代表配置是 `BLOCK_M=128`、`BLOCK_N=128`。对于 `M=1～12,N=128`，输出 grid 只有一个 CTA；H200 有 132 个 SM，大多数 SM 无任务。tile 内的无效行虽然不写回，仍消耗计算和资源。原生微基准约 13.8 μs，对照 cuBLAS 约 4.6 μs；NCU 还记录 occupancy 12.5%、Waves/SM 0.01。

确定性约束的核心是**同一输出元素的计算路径和归约顺序稳定**，不是全矩阵只能由一个 CTA 计算。我的 full-K 只拆 M/N，每个输出元素仍由唯一 CTA 拥有，CTA 内按固定顺序遍历 K，不做跨 CTA partial sum 合并，也不使用浮点 atomic accumulation。

注意：“full-K”不表示把 K=2048 一次性装进寄存器；实现仍以 `BLOCK_K=64` 分段加载，只是完整 K 都由同一个输出 CTA 按固定顺序处理。

## 4. 具体实现与取舍

| 路径 | 已验证的选择策略 | 核心约束 |
|---|---|---|
| BF16 | DeepGEMM → Triton full-K → persistent | 目标 shape/架构 guard；数值与 Graph 预检 |
| FP16 | Triton full-K → persistent | full-K 代表参数 `BM16/BN16/BK64`、2 warps、4 stages |
| FP32 | per-row full-K，`M<=128`；否则 persistent | `BM1/BN32/BK64`、4 warps、3 stages；FP64 严格精度门槛 |
| 非目标场景 | 原 persistent 或原平台路径 | 不改变 dense/QKV/O projection、量化 Router、非 CUDA 和 BI 关闭路径 |

接入点是 `UnquantizedLinearMethod.apply` 中的 Router 专用 dispatch；通过 custom op 建立编译边界，避免 Python import、JIT 与选路逻辑进入捕获图。selector key 包含 requested mode、device、dtype、shape、stride、bias 等签名；记录 `requested/selected/reason`，避免把“配置了 auto”误认成“确实运行了 DeepGEMM”。

**回退只在 capture 前发生。** 依赖不存在、guard 不满足或预检失败时选择下一后端；capture/replay 内出现 cache miss 或运行错误必须显式失败，不能捕获 CUDA 异步错误后偷偷换核。多个 live graph 的地址稳定性与缓冲区生命周期也是集成检查项，但不能说 full-K 使用了一个实际不存在的跨 CTA reduction workspace。

**默认仍可保持保守。** 7 月验收时 DeepGEMM 相对 persistent 为 2.949×，但相对优化后的 full-K 只有 1.127×，未达到预设 1.2× 默认晋级门槛，因此 auto 作为 opt-in，persistent 保留默认与回滚入口。后续集成状态和这个历史门槛不能混为一谈。

## 5. 实验数字应该怎么解释

| 数字 | 对应试验 | 不能说成什么 |
|---|---|---|
| Router 聚合降时 73.39% | 7 月 23 日 BF16、20 场景、48 层 Router device trace，中位约 789.23 → 210.01 μs/step | 整模型吞吐提升 73.39% |
| 135/135、20/20 | 该轮 DeepGEMM needle/batch/repeat kernel 与整模型 parity | 所有模型和 shape 天然逐位相同 |
| FP32 降时 38.65%/38.20% | TP1/TP2 真实 gate、15 个 Graph case/rank、每 case 100 次 replay | 完整 FP32 模型部署结果 |
| TP1 +3.81%、TP2 +4.36% | cached-request 四组实验 D/C：async OE 下增加 Router 优化 | sync OE 下 Router 独立收益 |
| TP1 +4.21%、TP2 +4.26% | 同实验 B/A：sync OE 下增加 Router 优化 | 与 D/C 可相加的第二份收益 |

cached-request 指相同前缀已 warm 且 KV cache 命中后，一条新 128-token 请求的完整 wall time；包含末块重算、调度和输出处理，不等于严格 pure decode 或 cold prefill。每轮 28 场景取几何平均，再跨 3 轮取几何平均。

后续 mixed 动态注入测试曾失败，不能用静态/cached 测试 PASS 抹掉这个边界。真实输入补审计的 5,242,880 行 Router 输出逐位相同，将问题范围缩到 Graph/编译交互，而不是证明 DeepGEMM 点值不准。未见对应更新产物时，不宣称该历史问题已全面消失。

## 6. 高频问题与直接回答

### Q1：batch invariance 和“同输入重复确定性”有什么区别？

重复确定性是相同输入布局多次执行不变；batch invariance 还要求把同一逻辑行放进不同 batch size、不同位置或与不同请求拼 batch，结果仍相同。因此我用 needle row 检查首/中/末位置，不能只重复跑同一个张量。

### Q2：浮点相加不满足结合律，那不做 split-K 就绝对确定吗？

不做 split-K 消除了一个重要不稳定来源，但不是跨硬件、编译器和所有 tile 的万能证明。还要检查具体 MMA/累加顺序、dtype、TF32 等计算策略。我的结论来自固定实现的设计约束和已覆盖签名的逐位测试，超出范围重新验收。

### Q3：为什么不直接用 cuBLAS？

目标 shape 上曾观察到 cuBLAS 也可跨 batch 逐位一致，所以不能泛称 cuBLAS 一定不确定。但我需要可审计、可约束的生产契约，不把某个版本某组 shape 的偶然行为外推为所有输入保证；后端选择必须由 guard 和预检限制。

### Q4：为什么误差很小还要检查 top-k？

Router 的离散选择对第 k/k+1 名边界敏感，小 logit 误差可能切换专家。矩阵 allclose 不是路由语义等价的充分条件，因此额外检查 expert ID、routing weight bits 和最终 token。也不能由“边界较近”推断实际有相同比例 token 会翻转。

### Q5：FP32 为什么不要求和 persistent 逐位一致？

最终 per-row full-K 与 persistent 的 FP32 算术实现不同。要求的是自身跨 batch/repeat/Graph 稳定，以及相对 FP64 `rtol=1e-5, atol=1e-6` 通过；不能把 BF16/FP16 的 backend parity 门槛机械套到 FP32。

### Q6：为什么 guard 还要包含 stride？

数学 shape 相同不代表内存布局相同，转置权重、非连续输入或 bias 会改变访存假设和可用 kernel。签名缺项可能复用不适用的选择结果。7 月联合报告还提示 weight identity/version 未进入 key 的结构性风险，不能把现有缓存说成对任意权重更新都安全。

### Q7：CUDA Graph 为什么不能发现慢了就自动换核？

capture 固定的是具体操作、依赖和地址；JIT、分配、控制流切换都应在图外做完。运行期错误又可能异步上报，此时继续执行不能保证上下文健康，所以采用 capture 前静态回退、replay 内 fail-fast。

### Q8：怎么证明收益不是 fallback 或 profiler 误分类？

核对 selector 实际选择，并在 trace 中把 `gatherTopK` 同 stream 紧邻前驱归为 Router GEMM，断言每 step 48 次；同时记录 kernel 名。microbenchmark 随机化后端顺序，冷 JIT 与稳态分开，模型测试独立加载并重复多轮。

### Q9：GPU busy 为什么不等于 end-to-end wall？

报告中的 GPU busy 是 device kernel duration 的和，不等于硬件利用率，也不包含全部 CPU 调度、输出处理和等待；并发时 kernel duration 还可能重叠。端到端收益必须从同步 wall 或请求级吞吐单独计算。

### Q10：你自己的贡献与团队贡献怎么划分？

我负责 Router 小 M 瓶颈归因、full-K、selector/preflight/fallback、vLLM 接入及验证；DeepGEMM 是依赖项目。后续多算子集成保留了四位作者，HPC BF16 MoE、caller-owned workspace 和 FA3 sink attention 分别有对应团队作者，不把整个算子栈包装成个人原创。

## 7. 被追问时主动交代的边界

- 已发布模型名称可以更新为 IQuest-Q1，但历史测试模型、dtype、TP 和数值不能“改名迁移”。
- 73.39% 是 Router 聚合 kernel 降时；3.81%/4.36% 是另一请求级矩阵的增量收益。
- 一个局部 kernel 的数值正确性不等于整个动态 serving workload 已验收。
- Opt-in、默认启用、内部合入与上游贡献是不同状态；按实际版本回答。

## 8. 开发记录与证据索引

- 瓶颈与 NCU：`/volume/pt-train/users/zhaoye/doc/bi/reports/bi_cost_baseline_20260723.md`；`/volume/pt-train/users/zhaoye/doc/bi/assets/native/README_native.md`。
- 73.39%、135/135、20/20：`/volume/pt-train/users/zhaoye/doc/bi/reports/deepgemm_router_eval_20260723.md`。
- 最终分层策略：`/volume/pt-train/users/zhaoye/doc/bi/pr/DEVELOPMENT.md`；`/volume/pt-train/users/zhaoye/doc/bi/reports/router_gemm_auto_20260724.md`。
- FP32 最终结果：`/volume/pt-train/users/zhaoye/doc/bi/reports/router_gemm_fp32_full_k_tp_20260724.md`。
- cached-request 四组：`/volume/pt-train/users/zhaoye/doc/oe/BI_RouterGEMM_OE联合验收_20260728.md`。
- mixed 边界：`/volume/pt-train/users/zhaoye/doc/oe/BI_RouterGEMM_OE全面性能测试执行记录_20260729.md`；`/volume/pt-train/users/zhaoye/doc/oe/BI_RouterGEMM_OE数值正确性复验_20260729.md`。
- 团队集成归属：`/volume/pt-train/users/zhaoye/doc/vllm_kernel_stack_mr_description_20260826.md`。
