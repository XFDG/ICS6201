# 开源修复、回归验证与审阅跟进

记录范围：2026-10-12 至 10-15；整理日期：2026-10-19。

本文依据本地开发记录整理。PR 状态为当时快照，本次未重新查询社区状态；“已发布”不等于“已合入”。

详细的复现、根因、补丁迭代与测试过程见 [GitHub 修复开发过程](../github_repairs/README.md)。本页保留简要结果。

## Triton：不支持的 FP8 格式转换应清晰报错

E4M3 与 E5M2 位宽相同，但直接转换仍涉及舍入与后端支持。原问题在编译下沉阶段产生断言。最初的隐式 FP16 中转方案通过数值测试，但维护者不接受自动插入转换的设计。

10 月 15 日修订为前端明确拒绝不支持的直接转换，同时保留用户显式 FP16 转换链和同格式转换。这一修订把 API 行为与维护者要求对齐，数值正确性没有被当作设计获批的替代品。

记录中的验证：12 项编译检查、14 项 H200 新回归和 48 项既有浮点转换测试通过。SM89 只做编译，不能写成对应硬件运行通过；未覆盖解释器、AMD 硬件或完整 CI。

公开入口：[Triton PR #12290](https://github.com/triton-lang/triton/pull/12290)。10 月 15 日已更新原 PR，等待复审，未声称合入。

## Quack：隔离 autotuning warmup 的随机状态

warmup 使用默认 CUDA generator，导致应用后续随机序列受冷启动或调优缓存命中影响。修复使用局部初始化的独立 generator，并保持原 warmup 工作内容。

回归同时检查 CPU、所有可见 CUDA RNG 状态及后续随机值，覆盖零/非零 warmup 时长与默认/非默认设备。原版新增 4 项回归全部失败，修复后全部通过；完整 autotuner 测试文件 7 项通过。这里验证的是随机性隔离，没有测性能收益。

公开入口：[Quack PR #219](https://github.com/Dao-AILab/quack/pull/219)。10 月 15 日记录显示 upstream checkout 在受信任工作流上下文拒绝 fork 代码，后续检查未运行；此类权限门禁没有通过放宽安全配置来绕过。

## DeepSpeed：限制 BF16 overflow opt-out 的生效范围

早期补丁对无效梯度范数的放行范围过宽，引入 FP32 的 Inf/NaN 参数污染回归；另一个 overlap optimizer 路径也需要同等约束。修订将 opt-out 限制为 BF16 且显式关闭 overflow 检查，保留 FP32/FP16 默认保护。

| 验证 | 记录中的最终结果 | 限制 |
|---|---|---|
| 新回归文件 | 20 项通过 | 含原生分布式测试和分支单测，不能全部当作集成测试 |
| FP32 CPU 分布式 | 每 rank 12/12 步检查通过 | 指定 ZeRO 阶段与 clipping 组合 |
| 原生 CPU optimizer-process | 每 rank 24/24 步检查通过 | 只覆盖全量更新 warmup，不含 selective/top-k 阶段 |
| H200 / NCCL | 每 rank 36/36 步检查通过 | 不含 GPU optimizer offload、GPU ZenFlow 或 ZeRO3 |
| 原有 overflow suite | 36 项通过 | 不等于完整 CI 矩阵 |

公开入口：[DeepSpeed PR #8818](https://github.com/deepspeedai/DeepSpeed/pull/8818)。截至 10 月 15 日记录已获批准、仍未合入；被跳过的 CI 项不作为已执行证据。

## Mooncake、CUTLASS 与 vLLM 的后续验证

| 项目 | 修复或验证内容 | 证据与边界 |
|---|---|---|
| [Mooncake #3661](https://github.com/kvcache-ai/Mooncake/pull/3661) | metadata lifetime 测试 fixture 补齐 worker 生命周期状态 | 原断言复现，修复后通过；相关 RDMA 测试 105 通过、1 跳过，队列测试 13/13；没有证明真实 RNIC 传输或完整 CI |
| [CUTLASS #3614](https://github.com/NVIDIA/cutlass/pull/3614) | 大 workspace 的动态 descriptor 溢出改用静态 byte descriptor | 3 项边界回归及小形状 reference 通过；大形状仅 launch smoke，不能称大形状数值验证通过 |
| [vLLM #58143](https://github.com/vllm-project/vllm/pull/58143) | 补齐真实 BERT checkpoint 的 main/branch/HF 对照 | 12 组评分完成，unpadded 对照一致，padded 对匹配 mask 的误差显著下降；复用 native artifacts，不声称完整源码构建通过 |

以上三项依据 10 月 12–13 日的修复记录。部分工作是补足测试或修复 fixture，不应统一描述成新增生产功能。

## 可复用经验

每次返修应重新验证被改动的条件范围，特别是精度、optimizer override 和边界尺寸。测试应包含会失败的原版负对照，并区分单元测试、真实硬件集成、编译 smoke 与数值 reference。提交发布、维护者审阅、CI 和合入分别记账，防止把本地闭环写成社区交付完成。
