# 简历与面试材料入口

更新日期：2026-09-29。当前版本统一放在本目录；不再使用 `ICS6201/output`。

## 当前简历

| 版本 | 可编辑文本 | Word | PDF |
| --- | --- | --- | --- |
| 中文综合版 | [Markdown](冯浩然_AiInfra香港中文大学_15024999885.md) | [Word](冯浩然_AiInfra香港中文大学_15024999885.docx) | [PDF](冯浩然_AiInfra香港中文大学_15024999885.pdf) |
| 推理优化版 | — | [Word](冯浩然_AIInfra中文简历_推理优化版.docx) | [PDF](冯浩然_AIInfra中文简历_推理优化版.pdf) |
| 训练优化版 | — | [Word](冯浩然_AIInfra中文简历_训练优化版.docx) | [PDF](冯浩然_AIInfra中文简历_训练优化版.pdf) |
| 英文版 | — | [Word](Haoran_Feng_AIInfra_Resume.docx) | [PDF](Haoran_Feng_AIInfra_Resume.pdf) |
| 面试介绍 | — | [Word](面试介绍精简版.docx) | [PDF](面试介绍精简版.pdf) |

四版简历均为两页；面试介绍一页。推理版突出 Router GEMM、OE、rollout；训练版突出 FA3、Sink FC1、orth-loss、R3，并保留 Blackwell 验证项目。

## 面试准备

见 [面试押题目录](面试押题/README.md)：七项九坤工作及三组开源贡献，专题各有 Markdown 和 PDF，包含开源仓库技术介绍与应用、STAR 口述、实现细节、实验口径、追问与证据来源。README 仅保留 Markdown。

## 归档与恢复

- 旧简历、旧面试介绍、旧押题及原 `output` 下文件，全部保存在 [旧的简历](旧的简历)。
- 文件名为 `简短主题 M月D日.扩展名`，不带个人信息。日期交叉核对 Word/PDF 元数据与 Git 内容历史，纯复制或移动不算更新；详见 [归档说明](旧的简历/README.md)。
- [归档清单](旧的简历/归档清单_20260929.json) 保留 52 个旧文件的原位置、命名依据、原修改时间和 SHA-256；原始历史文档保留，Word 解包附件已移出仓库。
- `open_doc`、`work_records`、`cot-main` 等开发与研究材料保持原位，不当作旧简历移动。

## 重建

在 `ICS6201` 根目录运行：

```bash
python3 tools/documents/build_release_resume_word.py doc/冯浩然_AiInfra香港中文大学_15024999885.md
python3 tools/documents/render_interview_notes.py
python3 tools/documents/qa_resume_refresh.py --render
```

其他版本直接维护 Word 并导出 PDF，不再保留独立 Markdown。归档脚本为一次性迁移脚本，避免把新版本误归档。
