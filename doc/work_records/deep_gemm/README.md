# DeepGEMM 离线预编译与 Wheel 交付

> 目标：让隔离网络环境中的 vLLM/DeepGEMM 直接加载预编译 cubin，跳过运行时 NVCC JIT，并以可校验 wheel 交付。

## 核心结果

| 项目 | 结果 |
|---|---:|
| union bundle | 109 -> 359 kernels，增加约 230% |
| H200 单卡覆盖 | 8 个模型，验证范围内 cold=0 |
| TP=2 覆盖 | Qwen3-8B-FP8，cold=0 |
| TP=2 load | precompiled load 相比 cold load 约 7.7x |
| wheel | baseline 453MB、DeepGEMM 17MB、combined 469MB |

## 工程链路

~~~text
模型 shape 收集
  -> 运行时 JIT 产物归档
  -> 多模型 bundle union / 去重
  -> SHA-256 一致性检查
  -> 单卡与 TP=2 cold=0 验证
  -> baseline + DeepGEMM / combined wheel
  -> 离线安装与包内 bundle 自动解析
~~~

关键工作还包括修复 TP=2 环境的 NCCL/CUDA 包冲突，并区分“bundle 命中”“加载耗时”和“稳态 GEMM latency”三个指标。

## 文档

| 文档 | 说明 |
|---|---|
| [Stage3 最终简报](./report_stage3_final_brief.md) | 模型覆盖、kernel 数、wheel 大小和未完成项 |
| [Wheel-only 交付说明](./deep_gemm_wheel_only_delivery_20260521.md) | two-wheel/combined 安装、校验和交付边界 |
| [Offline cubin 设计计划](./deepgemm-offline-cubin-plan.md) | config cache 与预编译目录设计背景 |

## 结论边界

- 7.7x 是 cold/load 启动阶段收益，不是 hot GEMM TFLOPS 提升。
- 8 个模型表示验证覆盖，不表示所有可能 shape 已预编译。
- 其中一个 BF16 dense 小模型不会触发 DeepGEMM，保留在清单中是为了验证“不误触发”。
- 超大模型和 H100 因当时硬件条件未覆盖，不能写成全平台支持。

## 简历建议

面向离线 H200 推理环境设计 DeepGEMM cubin 预编译与 wheel-only 交付，将 union bundle 从 109 扩展到 359 kernels，覆盖 8 个单卡模型及 Qwen3-8B-FP8 TP=2，验证范围内实现 cold compile 清零，并通过 SHA-256 保证 bundle 合并一致性。

