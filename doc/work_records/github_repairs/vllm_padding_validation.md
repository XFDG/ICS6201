# vLLM：Cross-encoder padding 修复的模型级验证

开发记录：2026-10-08、10-12 至 10-13。[PR #58143](https://github.com/vllm-project/vllm/pull/58143)。

## 1. 问题与修复范围

Cross-encoder 评分中的 segment/token type 与 padding 边界需要与 tokenizer 语义一致。已有补丁采用文档区间的半开区间表示，保留 tokenizer 的 padding type，并增加非法 token type 回归。

本轮主要工作是主线同步和补齐 reviewer 要求的真实模型证据，不能把全部过程描述成再次新增生产实现。

## 2. 先保留主线和修复双方的测试

10 月 8 日合入主线时，手动冲突集中在 scoring 测试文件。保留本 PR 的区间/padding 回归，也保留上游新增 reranker、RequestError 等测试，未用整体 ours/theirs 覆盖。

实际源文件 CPU 单测 10 项通过，变更文件的定向检查通过。但 CPU 输入处理验证不能替代模型评分，因此继续准备 main/branch/HF 三方对照。

## 3. 环境失败不能算作模型验证

模型运行曾被 native attention 扩展缺失、依赖版本及辅助通信路径问题阻塞。这些失败先于目标评分逻辑，不构成修复正负对照。后续补齐隔离依赖并使用正式支持的 Attention 后端配置后才运行完整评分矩阵。

最终仍复用了已有 native artifacts，而不是完整构建当时主线的全部原生依赖。报告保留这一限制，没有将 source branch 对照等同于全新 native build 验收。

## 4. 关键控制变量是 mask

选择固定公开 BERT cross-encoder checkpoint，使用 H200、FP16、eager、单卡和未激活 raw logits。main/branch 使用相同 input IDs，分别比较无 padding 和右侧 padding。

受测 vLLM 路径将追加 PAD 作为可见位置。匹配的 HF padded 参考因此使用 all-ones attention mask，并使用 tokenizer 的 padding token type。另设 HF 默认 masked-padding 对照，避免混淆。

| 对照 | 目的 |
|---|---|
| main vs branch，无 padding | 确认正常路径不变 |
| main vs branch，右侧 padding | 观察修复触发路径 |
| branch vs HF，匹配 mask | 检查对应语义下的评分一致性 |
| HF 默认 masked padding | 证明 mask 的不同本身会改变参照，不作为 visible-padding 的等价目标 |

这项 PR 没有修改生产 attention-mask 语义，不能声称修复后自动等价于所有 HF padding 设置。

## 5. 实际结果

main/branch 共完成 12 组评分，另有 HF 参考与 mask control。无 padding 的 main/branch 最大绝对差为 0；branch 与 HF 的最大绝对差约为 0.001401。

右侧 padding 时，相对匹配 mask 的 HF 参考，main 最大绝对差约 5.286148，branch 约 0.001151。这里是实际观测值，没有为得到通过结果放宽已有断言；也没有把不同 mask 下的 HF logits 混入同一比较。

## 6. 发布与经验

10 月 13 日更新原 PR 的验证说明和 reviewer 回复，本轮没有新增 vLLM 生产代码提交。记录为等待复审，本次不查询实时合入状态。

本例说明“同样 padding 长度”不代表输入语义一致。必须同时核对 token IDs、token types、attention mask、精度与输出激活方式，模型级证据才可解释。
