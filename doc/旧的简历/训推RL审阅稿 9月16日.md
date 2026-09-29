# 冯浩然｜AI Infra 中文简历（内容审阅稿）

> 版本：2026-09-15  
> 用途：仅审阅内容与重点；**尚未修改任何 Word 或 PDF**。确认后将直接覆盖现有 Word/PDF，沿用 `冯浩然_AiInfra香港中文大学_15024999885.docx` 的字体、段落、页边距与两页版式。

冯浩然  
电话：(+86) 15024999885　邮箱：225010160@link.cuhk.edu.cn　个人网站

## 教育经历

| 时间 | 学校 | 专业 / 学位 |
|---|---|---|
| 2025.09–2027.06（预计） | 香港中文大学（深圳） | 集成电路与系统 硕士 |
| 2021.09–2025.06 | 山东大学（985） | 电子科学与技术 / 计算机科学与技术 本科 |

获得荣誉：山东大学学业奖学金（前 20%）、山东大学特长奖学金（竞赛创新）

## 实习经历

### 2026.04–至今　九坤投资－AI 研究院　大模型与高性能计算实习生

- **训练算子支持**：
  1. **FA3 deterministic SWA backward 的 dQ 依赖链调度优化**：针对长序列、变长滑窗 Attention 的 deterministic backward 回退，负责问题归因、调度实现与验证；通过 Nsys/NCU 和 dQ/dK/dV 因子隔离，确认 dQ semaphore 依赖链占确定性增量 **99.67%**。在原 fused main kernel 中将绝对 ticket 改为 contributor-relative ticket，并使 reverse scheduler 与归约顺序对齐；复用既有长度元数据进行长短序列分流，短序列保留原生 fallback，不新增 kernel、workspace 或 D2H。代表 packed shape 完整 backward 由 **6.755 ms 降至 1.699 ms（3.98×）**，2K–12K 加速 **1.59–5.37×**；1000 次自身 bitwise、3601/3601 数值与通用回归通过，完成 wheel 交付。
  2. **Sink FC1 GEMM 与通信竞争控核探究**：训练 trace 显示跨 stream 的 SendRecv／AllGatherV 会使 FC1 额外退化 **15.48%/28.73%**；先纠正 FC1 实际 BF16 shape 为 `8192×3072×3072`，确认约 200 μs 的单算子基线正常，瓶颈在通信—计算资源竞争而非 GEMM 异常。搭建 4×H200 双 stream 代理，以 NCCL CTA 预算控制通信并行度、以 DeepGEMM SM budget 控制计算资源，结合 Nsys 校验实际 kernel grid 与真实重叠；完成 121 组粗扫、89 组细扫及候选复验。AllGatherV 代理中，**12 CTA + DG120** 将联合 span 降低 **12.80%**，其中同通信配置下 DeepGEMM 控核独立贡献 **7.58%**；AllToAllV 的主要收益来自提高通信并行度，未将其归因为 GEMM 优化。

- **推理算子支持**：
  1. **CUDA Graph 下的确定性 MoE Router GEMM**：针对 batch-invariant MoE decode 小 `M` 下 persistent tile 无效计算及 CUDA Graph 确定性约束，负责在 vLLM 中工程化接入 DeepGEMM／Triton Full-K／persistent 三级后端；设计 tensor-signature selector、capture 前数值与 Graph preflight、cache/fallback、workspace 生命周期和路径回归，正式 replay 固化后端、不在图内动态选核。48 层 Router GEMM 中位耗时下降 **73.39%**，135/135 kernel、20/20 模型场景逐位一致；prefix-cache hit 下 TP1/TP2 整请求处理性能提升 **3.81%/4.36%**。
  2. **OE 异步状态算子**：将 recent-token history 与 `oe_input_ids` 构造保留在 GPU，以 Triton fused-hash 消除 TP1 的同步与回传，并修复 prefill、decode、mixed batch、请求恢复、slot reuse/reorder 的跨 step 状态；TP1 严格 28/28 case 通过、吞吐较 sync **+4.99%**。TP>1 采用 async-unfused + batch-invariant 安全路径后，TP2/TP4 均达 **84/84、0 mismatch**，decode 吞吐 **+4.6%/+3.0%**。

- **RL 训推一致性与 Rollout 稳定性**：
  1. **R3 Router Replay 与多卡可观测性**：针对 rollout 侧 vLLM 与训练侧 Megatron 的 MoE 路由偏差，打通 `[token, MoE-layer, top-k]` route 采集、传输与训练侧回放，建立 response-mask 对齐、异常检测及 `route mismatch / fτ² / KL` 指标闭环；8×H200、Qwen3-30B-A3B BF16 的 20-step 对照中，route mismatch 由 **17%–19% 降至 0**，fτ²/KL 分别降低 **36–145×/4–7×**。进一步扩展至 128 卡集群，完成内部大模型竞赛题单轮 200-step rollout 并接入 SwanLab 监控，关键路由与漂移指标与 8 卡对照一致。
  2. **FlashInfer CUDA Graph hang 排查与修复**：针对 H200、vLLM TP=2 FULL CUDA Graph rollout 卡死，构建两卡最小复现，经 TP1/TP2、eager/FULL Graph、fused/unfused 等控制变量将根因收敛至 fused AllReduce + RMSNorm 中 Lamport `-0.0` sentinel 的 FTZ 误判；回移上游 `0x80000000` 位级判断修复并完成模型级 Graph on/off 回归，重复 Graph replay 未再出现 hang 或 timeout。

### 2026.01–2026.04　摩尔线程　算子与编译器优化实习生

- **TensorFlow MUSA Extension 算子、图优化与稳定性**：负责 muDNN GELU 接入、GELU fusion 链路修复、benchmark 和热点算子优化，推动整网 11 个 GELU 全部融合，真实 shape testcase 耗时降低约 **36.6%**；独立定位 `StridedSlice<int32>/Pack<int32>` 将 shape tensor 错误送入 device path 的根因并重构 HostMemory 链路，使 inference 500 轮成功率约 30% 提升至 1000 轮 **100%**，4 万/40 万/80 万轮长跑稳定。同时将 Logical_Or 由 **21.2 μs 降至 10.7 μs**，整网吞吐由 **8187.48 提升至 8284.65**。

## 开源贡献

- **GitHub 已合入 5 个外部项目 PR（账号：XFDG）**：Mooncake 已合入 **4 个 PR**（[#3601](https://github.com/kvcache-ai/Mooncake/pull/3601)、[#3604](https://github.com/kvcache-ai/Mooncake/pull/3604)、[#3660](https://github.com/kvcache-ai/Mooncake/pull/3660)、[#3726](https://github.com/kvcache-ai/Mooncake/pull/3726)），覆盖存储元数据持久化、RDMA 端口恢复、内存注销生命周期及队列失败传播；Mirage 已合入 [#755](https://github.com/mirage-project/mirage/pull/755)，修复 `argmax` 中 padding rows 的错误取值。

> 注：FlashInfer #3304 是上游已合入修复；简历仅表述本人完成问题定位、回移与回归，不将其列为个人 GitHub 合入贡献。

## 项目与科研

### 2025.12–至今　基于 PyTorch Extension 的高性能 LLM 量化推理 Runtime 开发

- 在 W4A16 AWQ、W8A8 SmoothQuant、FP8 Runtime 中完成 FlashAttention-2/4 可配置接入、GQA KV-head 展开、softcap mask、SDPA 回退及跨 GPU 动态构建；以 Compute Sanitizer 定位 `gemv_kernel_g128` 尾组越界，补充边界保护并实现 Qwen2 packed-AWQ TP=2 列/行切分与 NCCL all-reduce。代表 shape 的 memcheck/initcheck/synccheck/racecheck 均为 0 error/0 hazard；Qwen2.5-Coder-32B W4A16 双卡部署使 checkpoint 显存 **-70.5%**、峰值显存 **-66.8%**、端到端输出吞吐 **+76.8%**，并通过数值、跨 rank token 与交互验收。

### 2026.01–至今　关键 Token 加权的思维链蒸馏｜AAAI 2027 已投稿｜[项目代码](https://github.com/jokerhan01/cot-main)

- **方法与职责**：参与“结构恢复—关键 Token 加权监督—偏好优化”三阶段框架；负责模型训练与调优，通过逐 Token 扰动教师推理、以参考答案生成似然下降量估计 Token 重要性，并用于加权 SFT 与高重要性辅助损失。

- **实验结果**：完成 LoRA/DPO 训练和 vLLM TP=4 评测；Qwen2.5-7B-Instruct 在 GSM8K/SVAMP 上取得 **94.01%/94.00%** accuracy，较最强基线分别提升 **5.51/10.10** 个百分点；论文已投稿 AAAI 2027。

## 综合素质

- **技术方向**：关注 GPU 高性能算子开发与大模型训练/推理系统优化，具备 Attention、MoE Router、量化 GEMM、异步状态管理、Tensor Parallel 与 CUDA Graph 工程实践。
- **开发与性能工程**：熟练使用 C/C++、Python、CUDA、Triton、PyTorch Extension；能够使用 Nsys、NCU、Compute Sanitizer 完成算子性能分析与正确性验证。
- **语言与证书**：IELTS **6.5**、CET-6，持有华为 HCIA-AI 认证；具备英文技术文档阅读、检索与跨仓源码分析能力。

## 本版相对上一版的调整

1. 九坤保留“训练算子支持—推理算子支持—RL 训推一致性与 Rollout 稳定性”三条主线，分别按 **2 项、2 项、2 项**编号展开。正文自然遵循“问题—个人动作—结果证据—边界”，不显示 STAR 标签。
2. Router GEMM、FA3 backward 分别位于推理/训练第一成果位；OE 异步状态算子放入推理算子支持，R3 与 FlashInfer hang 留在 RL 主线，避免叙事混杂。
3. 摩尔线程实习与量化 Runtime 项目均合并为一段；论文标题补充公开项目代码链接。
4. 开源贡献只保留 GitHub 已核验的 **5 个**外部项目已合入 PR（Mooncake 4 个、Mirage 1 个）；不计入 MUSA PR，也不把上游 FlashInfer PR 写成个人贡献。
5. 通过内容确认后，只更新现有 `冯浩然_AiInfra香港中文大学_15024999885.docx/.pdf`，继承其 Word 字体与版式；不会另起一份 Word/PDF。
