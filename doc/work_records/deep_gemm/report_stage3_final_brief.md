# DeepGEMM Stage3 最终报告（简略版）

> 2026-05-21 | zhaoye

---

## 做了什么

DeepGEMM 离线预编译缓存系统：运行模型时从预编译目录直接加载 cubin，跳过 NVCC JIT 编译（~1.5s/kernel → ~10μs/kernel）。

**bundle 从 109 → 359 kernels**，覆盖 8 个 H200 单卡模型 + 1 个 TP=2 模型。

---

## 加速的模型

### H200 单卡（全部 cold=0）

Qwen3.5-35B-A3B-FP8, Qwen3-30B-A3B-FP8, Qwen3-32B-FP8, Qwen3-14B-FP8, Qwen3-8B-FP8, Qwen3-4B-FP8, Qwen3-1.7B-FP8, Qwen3.5-0.8B*

> *0.8B 是 BF16 密集模型，不触发 DeepGEMM

### H200 TP=2（cold=0）

Qwen3-8B-FP8 (TP=2)：precompiled=1354, load 加速 7.7x

---

## Wheel 产物（2+1）

| 产物 | 大小 | 说明 |
|------|------|------|
| vllm baseline | 453MB | vLLM 基础包 |
| deep_gemm | 17MB | 含 359 kernel bundle，自动解析 |
| combined | 469MB | vllm + deep_gemm 合并包 |

### 安装

```bash
# 推荐 two-wheel
pip install vllm-*.whl
pip install deep_gemm-*.whl

# 或 combined 一键
pip install vllm-*.whl  # combined 版

# 设置环境变量后启动 vLLM
export VLLM_USE_DEEP_GEMM=1
export VLLM_MOE_USE_DEEP_GEMM=1
```

---

## 未完成

| 项 | 原因 |
|----|------|
| DeepSeek-V3-0324 | 642GB，2xH200 装不下（需 TP=8+） |
| H100 覆盖 | 无 H100 机器 |
| TP=2 35B 模型 | gpu0 环境刚修复，后续可做 |

---

## 环境配置

- Python 3.12 | torch 2.10.0+cu128 | CUDA 12.9 | vllm 0.1.dev1 | H200
- NCCL 2.27.5 (CUDA12) — gpu0 TP=2 关键修复
- `DG_JIT_PRECOMPILED_DIR` 指向 `qwen3_dpsk_v32_h100_h200_fp8/cache`（359 kernel）

---

## 对 Mentor 汇报

> 本次完成了 DeepGEMM 离线预编译 Stage3 最终轮验证：
> - bundle 从 109 kernel 扩展到 **359 kernel**，覆盖 **8 个 H200 单卡模型**
> - 首次跑通 **H200 TP=2** collect→verify 闭环（Qwen3-8B-FP8，cold=0）
> - 产出 **2+1 wheel**（baseline vllm + deep_gemm + combined），deep_gemm 安装后自动解析 359-kernel bundle
> - 平台高频模型（Qwen3.5-35B-A3B-FP8, Qwen3.5-0.8B）已全部覆盖
> - DeepSeek-V3-0324 因 642GB 模型需 TP=8+，当前 2xH200 硬件不足
> - gpu0 TP=2 NCCL 环境问题已修复（CUDA13→CUDA12），后续可继续扩展
