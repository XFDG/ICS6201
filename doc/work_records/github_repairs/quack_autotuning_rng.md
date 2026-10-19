# Quack：autotuning warmup 污染应用随机序列

开发日期：2026-10-14；review 状态记录截至 10 月 15 日。[issue #218](https://github.com/Dao-AILab/quack/issues/218) · [PR #219](https://github.com/Dao-AILab/quack/pull/219)。

## 1. 现象

调优 warmup 通过 `torch.randn` 生成矩阵，使用默认 CUDA generator。应用即使设定相同 seed，后续随机结果仍可能因本次调优是否冷启动而变化。问题不是矩阵乘算错，而是工具层消耗了应用的随机状态。

## 2. 原版最小复现

先对原版执行零时长 warmup，再比较 CPU 和 CUDA RNG 状态，并用独立克隆的 generator 生成后续参考样本。原版表现为 CPU 状态不变、默认 CUDA 状态改变、下一批随机值不一致。

零时长仍能暴露问题，说明污染发生在 warmup 准备阶段，不要求运行长时间调优循环。这个对照帮助把问题缩到随机矩阵初始化。

## 3. 补丁与不变量

生产改动位于 `quack/autotuner.py`：给现有随机矩阵生成传入独立、局部初始化的 CUDA generator。warmup 矩阵形状、BF16 类型、时长和乘法循环保持原逻辑。

示意契约：

```text
应用默认 RNG ──应用随机操作──> 下一应用样本
独立 warmup RNG ──warmup 矩阵──> 调优预热
```

两个随机流分离，应用的下一样本不应受是否执行 warmup 影响。独立 generator 必须绑定目标设备，不能只在默认设备上验证。

## 4. 测试为什么同时看状态和值

新增测试放在已有 `tests/test_autotuner.py`，覆盖零/非零时长和默认/非默认 CUDA 设备。每次保存 CPU 及所有可见 CUDA RNG 状态；warmup 后先检查状态，再比较后续 64 个随机值与独立 generator 的输出。

状态检查证明调用没有推进共享 RNG，数值检查直接证明应用可观察结果保持一致。若只重复设置全局 seed 再比较，会掩盖 warmup 对状态的污染。

| 对照 | 原版 | 修复版 |
|---|---|---|
| 原 issue 的 warmup | CUDA 状态改变，后续样本不匹配 | 状态保留，样本逐位一致 |
| 新增 4 项回归 | 4/4 失败，每项 64/64 样本不匹配 | 4/4 通过 |
| 完整 autotuner 测试文件 | 未运行整文件 | 7/7 通过，无跳过 |
| 定向格式与 lint | 不适用 | Ruff check/format 通过 |

两侧使用相同源码基线与隔离 SDK 环境。SDK 依赖问题单独处理，不把依赖升级混入生产修复。

## 5. 发布后遇到的 CI 边界

10 月 14 日完成普通提交、推送、PR 和 issue 跟进。10 月 15 日记录显示，upstream checkout 在受信任的 `pull_request_target` 上下文拒绝 fork 代码，因此 Ruff 和后续 GPU 检查没有执行完成。

这属于工作流信任边界，不构成 RNG 修复失败的证据，也不能宣称 CI 通过。已向维护者说明，未添加不安全 checkout 绕过项。本次文档不重新查询当前 CI。

## 6. 结论范围

已证明受测 warmup 不污染应用随机状态。没有运行完整 CuTe kernel 套件，没有测性能收益，也没有据此承诺整个应用训练可完全确定。局部工具的无副作用契约应独立测试。
