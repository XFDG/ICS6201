# LLMQRT H200 / AWQ / Tensor Parallel 工作记录

> 快照日期：2026-07-16
> 源工程：`/volume/pt-train/users/zhaoye/LLMQRT`

## 当前进展

| 方向 | 已验证结果 |
|---|---|
| H200 迁移 | LLMQRT W4A16 AWQ 已从 RTX 5060 Laptop 迁移至 2×NVIDIA H200（SM90/CUDA 13.0） |
| SM90 kernel | 使用 compute-sanitizer 定位并修复 GEMV 尾部量化 group 越界；补充 current stream、launch check 与输入断言 |
| Tensor Parallel | 实现 Qwen2 packed-AWQ TP=2：列/行分片、group-aligned metadata 与 NCCL all-reduce |
| 大模型接入 | Dense 32B/72B 原生 TP=2 已通过；72B 每 rank 峰值显存 70.09 GiB |
| 32B 量化 | Qwen2.5-Coder-32B-Instruct 已完成本地 128×512 校准的 W4A16 AWQ 量化与双卡交互验收 |

## Coder-32B 量化前后对照

对照使用同一模型、tokenizer、prompt、固定 64 个输出 token、8-token warmup、
greedy decoding 和 3 次重复；TP 延迟取两个 rank 中较慢值，再报告中位数。

| 指标 | Dense FP16 TP=2 | AWQ W4A16 TP=2 | 变化 |
|---|---:|---:|---:|
| 64-token 中位耗时 | 25.2858 s | 14.3022 s | -43.4% |
| E2E output 吞吐 | 2.5311 tok/s | 4.4748 tok/s | **+76.8% / 1.77×** |
| 运行期峰值显存/rank | 32.02 GiB | 10.63 GiB | **-66.8%** |
| 加载峰值显存/rank | 33.15 GiB | 18.12 GiB | -45.3% |
| checkpoint 目录 | 61.04 GiB | 18.02 GiB | **-70.5% / 3.39× 更小** |

`e2e_output_tok_s` 包含 prefill、首 token 和 decode，不表述为纯 decode 吞吐。

## 验证证据

- 正式量化完成 64/64 层：量化 3191.99 秒，峰值显存 72.72 GiB，保存 10.05 秒。
- checkpoint schema：4 个 safetensors、1667 tensors，448 组 `qweight/qzeros/scales` 完整配对。
- Coder-32B TP=2 每 rank 的 448 个量化 Linear metadata 全部通过检查。
- 首层五类投影的 decode GEMV / prefill GEMM 共 10 组 CUDA-vs-PyTorch 对照通过，最大绝对误差 0.007812。
- 代表 shape 通过 memcheck、initcheck、synccheck 和 racecheck，均为 0 error/0 hazard。
- 两个 rank 与三次 benchmark 重复的 token IDs 一致；中文交互生成与正常退出已验证。

## 文档索引

| 文档 | 内容 |
|---|---|
| [H200 / TP 开发日志](./h200_development_log.md) | 完整时间线、根因、修复、量化结果、benchmark、命令和简历表述 |
| [H200 TP=1 验证指南](./h200_tp1.md) | 环境、SM90 构建、单卡运行和交互命令 |
| [LLMQRT 架构图](./architecture.md) | 加载流程、Attention 后端、量化方式与数据流 |

## 当前推荐交互入口

```bash
cd /volume/pt-train/users/zhaoye/LLMQRT
bash scripts/chat_h200_tp2_awq.sh \
  /volume/pt-train/users/zhaoye/models/Qwen2.5-Coder-32B-Instruct-AWQ \
  --max-new-tokens 256
```

看到 `[TP=2 AWQ READY]` 后输入问题；`q` 退出，`clear` 清空历史，`mem` 查看
两个 rank 的显存。完整复现命令见开发日志的“固化命令”章节。

## 快照边界

- 本目录只复制 Markdown 开发材料，不复制模型权重、CUDA 二进制和原始运行日志。
- 这是 2026-07-16 的文档快照；后续继续开发时，以 LLMQRT 源工程中的文档为准，再同步到此处。
- 文档包含集群绝对路径；若公开发布，应先做路径和环境信息脱敏。
