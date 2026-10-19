# GitHub 修复开发过程（脱密版）

整理日期：2026-10-19。以 10 月 1–15 日的开发记录为依据，参考此前 FlashInfer 文档的“现象—隔离—机制—补丁—验证—边界”结构。

## Word 与 PDF

完整合集：[Word](GitHub修复开发记录_脱密合集.docx) · [PDF](GitHub修复开发记录_脱密合集.pdf)。本目录每篇 Markdown 也提供同名 `.docx` 与 `.pdf` 分篇版。

导出版保留正文、表格与公开链接；流程图转为等义步骤列表。Word 可编辑，PDF 嵌入中文字体。可用仓库 `tools/documents/export_github_repairs.py` 重新生成，依赖 `python-docx`、`reportlab`、`markdown`、`beautifulsoup4` 及中文 TrueType 字体（通过 `--font` 指定）。

## 详细案例

| 案例 | 开发过程的重点 | 入口 |
|---|---|---|
| Triton FP8 转换 | 舍入错误与不支持转换的分层定位；根据 review 改写 API 行为 | [详细记录](triton_fp8_conversion.md) |
| Quack autotuning | 从全局 RNG 污染到独立 generator；状态和数值双重负对照 | [详细记录](quack_autotuning_rng.md) |
| DeepSpeed overflow | BF16 opt-out 修复引入 FP32 回归后的收窄与 sibling 路径验证 | [详细记录](deepspeed_overflow.md) |
| Mooncake 队列 | 取消 reservation、唤醒、quota 与 FIFO 的状态机闭环 | [详细记录](mooncake_queue_lifecycle.md) |
| CUTLASS workspace | 从 Python 整数猜测收敛到 descriptor 的 Int32 extent 溢出 | [详细记录](cutlass_workspace_descriptor.md) |
| vLLM padding | CPU 单测、主线合并、真实模型和 attention mask 对照 | [详细记录](vllm_padding_validation.md) |

先读[修复流程与证据组织](workflow.md)，再看具体案例。此前的[FlashInfer 根因详解](../flashinfer/flashinfer_mnnvl_lamport_ftz_hang_explained_20260717.md)展示了源码、SASS、控制组和补丁 A/B 如何形成互相支持的证据。

## 文档范围

本次只整理开发过程，没有重新运行 GPU 测试或查询上游 PR 实时状态。测试数量和结果均注明所属阶段；公开链接用于定位原 PR，状态只代表原始记录日期。示意伪代码用于解释契约，不是可直接应用的补丁。

原始材料包括 10 月 1 日 Mooncake 取消排空记录、10 月 8 日 CPU review 回归、10 月 12–13 日三项目验证、10 月 14 日 Triton/Quack/DeepSpeed 修复以及 10 月 15 日 review 更新。这里移除了个人信息、内部地址、工作树绝对路径、运行标识、凭据上下文和部署拓扑；保留公开项目、源码相对位置、PR 链接、测试方法和已知限制。

归档与公开摘要之间的对应关系见[近期工作总览](../2026_q4/README.md)。这组文档补充开发过程，不改变既有摘要所述的验收范围。
