# DeepSpeed：BF16 overflow opt-out 的修复与返修

开发记录：2026-10-14，审阅状态快照截至 10 月 15 日。[issue #8805](https://github.com/deepspeedai/DeepSpeed/issues/8805) · [PR #8818](https://github.com/deepspeedai/DeepSpeed/pull/8818)。

## 1. 原问题与期望行为

BF16 配置显式关闭 overflow 检查后，ZeRO-1/2 的无效全局范数分支仍可能跳过更新。修复需要尊重已有 opt-out 契约，同时保持默认保护与其他精度的行为。

“允许 opt-out”并不是建议在训练中关闭保护；只是在用户明确选择该模式时，更新逻辑应与配置语义一致。

## 2. 初版为什么没有发现问题

初版修改范数检查 gate，并通过 BF16 正负对照、既有 FP16 测试和 H200 的 BF16 验证。但初版 gate 的条件范围过宽，未覆盖 FP32 的保护行为。

后续 review 发现两个 P1：FP32 在 Inf/NaN 时错误继续更新、参数被污染；ZenFlow overlap override 还保留另一个不一致的无效范数分支。前一轮绿色测试不能覆盖这两个新发现。

## 3. 把失败样本变成返修矩阵

FP32 使用原生 CPU/gloo 两 rank 复现，分别覆盖 ZeRO 阶段、clipping 开关、有限梯度、Inf 和 NaN。初版公开补丁每 rank 为 4 pass / 8 fail；失败表现是本应 skip 的异常更新污染参数。

返修把 opt-out 收窄到“BF16 且显式关闭检查”，并在普通路径与 override 中同步实现。契约如下，示意不等同于完整生产代码：

```text
发现无效全局范数
  ├─ BF16 且显式 opt-out：遵循 opt-out 更新语义
  └─ 其他情况：保留原保护/跳步行为
```

没有改默认值、范数算法或梯度裁剪公式，也没有用放宽误差容限隐藏问题。

## 4. 为什么要测 override

父类路径正确并不能保证覆写 `step()` 的 optimizer 路径正确。除了分支级测试，还运行真实 CPU optimizer-process，以检查实际 worker 流程中的行为。mock dispatch/cleanup 的单测与原生进程集成分别记录。

| 验证 | 负对照 | 返修后 |
|---|---|---|
| 新回归文件 | 原基线 3 fail / 17 pass | 20 pass：4 个原生分布式 FP32 case，16 个分支单测 |
| FP32 CPU/gloo | 初版公开补丁每 rank 4 pass / 8 fail | 每 rank 12/12，异常输入 skip、参数 finite |
| CPU optimizer-process | 原 override 每 rank 16 pass / 8 fail | 每 rank 24/24，8 个配置 |
| H200/NCCL | 原始未修复源码每 rank 28/36 步检查通过 | 每 rank 36/36，12 个配置 |
| 既有 overflow suite | 不以旧结果替代返修回归 | 36 pass |

不同负对照有意使用原基线或初版公开补丁：前者复现原 bug，后者复现修复引入的回归。它们不能混成一个“修复前”数字。

有限梯度更新还与独立 FP32 Torch SGD 参考在转换到目标 dtype 后逐位比较。这样可以同时确认异常 gate 与正常更新没有相互干扰。

## 5. 发布和范围

返修以普通新提交更新原 PR，并回复两条 P1、更新测试说明。10 月 15 日快照为已批准但未合入；跳过的 CI 不作为已执行测试，本次不重新查询状态。

原生 CPU worker 仅覆盖 full-update warmup，不含 selective/top-k；GPU 测试不含 optimizer offload、GPU ZenFlow 或 ZeRO3。没有完整模型收敛、吞吐或跨厂商验证。

这次返修说明，条件开关需要按 dtype、默认值和 override 路径建立测试矩阵。一个局部条件只有一行，也可能扩大实际生效范围。
