# FlashInfer TP2 CUDA Graph Hang 根因与修复（公开版）

> 时间：2026-06-23 至 2026-07-15
> 核心结论：vLLM TP=2 + FULL CUDA Graph rollout hang 的 GPU 首因是 FlashInfer 0.6.4 MNNVL fused allreduce + RMSNorm 中的 Lamport sentinel 判断受 FTZ 影响；回移官方修复或升级到 0.6.12 后，两卡最小复现与模型级 graph replay 均通过。
> 公开边界：不包含内部仓库、节点、模型路径、原始 trace、运行 ID 和构建产物。

## 1. 现象与隔离

外层最初表现为 `sample_tokens` RPC timeout 和 engine-dead 类错误，但 sampler、D2H copy 与消息队列都是等待上游 GPU 工作完成的下游位置。通过 TP1/TP2、eager/FULL graph、fused/unfused 和 PDL 开关的控制变量实验，问题被缩小到 TP=2 FULL graph replay 中的 fused allreduce + RMSNorm 路径。

## 2. 根因

旧实现使用浮点比较识别 Lamport `-0.0` sentinel。在 fast-math/FTZ 语义下，合法的 FP32 负次正规数可能在比较时被当成 `-0.0`，worker 因而进入永久轮询。

修复采用精确 bit-pattern 比较识别 `0x80000000`，避免浮点比较被 FTZ 改写。该修复对应 FlashInfer 官方 [PR #3304](https://github.com/flashinfer-ai/flashinfer/pull/3304)。

## 3. 验证结果

| 验证 | 修复前 | 修复后 |
|---|---:|---:|
| 目标 16×16 FULL graph | 0/3 通过 | 2/2 通过 |
| 精确 prompt-9 用例 | 0/1 通过 | 3/3 通过 |
| 模型级 graph off | - | 2/2 通过 |
| 模型级 graph on | - | 2/2 通过 |
| 目标 SASS `FSETP.*.FTZ` | 8 处 | 0 处 |
| FlashInfer 0.6.12 隔离回归 | - | eager、5 次 graph replay、prompt-9、16×16 全部通过 |

## 4. 当前建议与边界

- 需要保持旧依赖时，使用 FlashInfer 0.6.4 + PR #3304 回移，并清理旧 JIT cache 后重新构建。
- 长期方案是升级到已经包含等价修复的新版 FlashInfer，并重新完成依赖和模型回归。
- 当前已闭环两卡最小复现和模型级 AsyncLLM 回归；完整大规模 RL `main_ppo` 仍需最终验收。
- CUDA Graph off/on 的性能差异属于 graph 模式收益，不能归因成两行 sentinel 修复带来的性能提升。

## 5. 月报一句话

定位并修复 FlashInfer 0.6.4 在 vLLM TP=2 + FULL CUDA Graph 下因 FTZ 误判 Lamport sentinel 导致的 rollout hang，回移补丁和 FlashInfer 0.6.12 均通过两卡最小复现与 graph replay 回归。
