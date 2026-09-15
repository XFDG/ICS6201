#!/usr/bin/env python3
"""Update the Chinese/English resumes and the concise interview script in place.

The two resume DOCX files are edited from their existing table-based templates so
the original page size, photo, section rules, and alignment are retained.  PDF
files are first produced in a temporary directory and are installed only after
the two-page checks succeed.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.shared import Inches, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from pypdf.annotations import Link
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


ROOT = Path("/volume/pt-train/users/zhaoye/ICS6201")
DOC_DIR = ROOT / "doc"
TMP_ROOT = ROOT / "tmp" / "pdfs" / "resume_update_20260915"

CH_DOCX = DOC_DIR / "冯浩然_AiInfra香港中文大学_15024999885.docx"
CH_PDF = DOC_DIR / "冯浩然_AiInfra香港中文大学_15024999885.pdf"
TRAIN_DOCX = DOC_DIR / "冯浩然_AIInfra中文简历_训练优化版.docx"
TRAIN_PDF = DOC_DIR / "冯浩然_AIInfra中文简历_训练优化版.pdf"
INFER_DOCX = DOC_DIR / "冯浩然_AIInfra中文简历_推理优化版.docx"
INFER_PDF = DOC_DIR / "冯浩然_AIInfra中文简历_推理优化版.pdf"
EN_DOCX = DOC_DIR / "Haoran_Feng_AIInfra_Resume.docx"
EN_PDF = DOC_DIR / "Haoran_Feng_AIInfra_Resume.pdf"
ANON_DOCX = DOC_DIR / "AIInfra中文简历_匿名版_小红书水印.docx"
ANON_PDF = DOC_DIR / "AIInfra中文简历_匿名版_小红书水印.pdf"
INTERVIEW_DOCX = DOC_DIR / "面试介绍精简版.docx"
WATERMARK_DIR = ROOT / "小红书商品" / "面试"
WATERMARK_TEXT = "小红书小冯别放弃"
WATERMARK_VERSION = "raster-v2"
WATERMARK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
PERSONAL_WEBSITE = "https://xfdg.github.io/"
MOONCAKE_PRS_URL = "https://github.com/kvcache-ai/Mooncake/pulls?q=is%3Apr+is%3Amerged"
MIRAGE_PR_URL = "https://github.com/mirage-project/mirage/pull/755"

CN_BODY = "宋体"
CN_HEADING = "黑体"
EN_FONT = "Times New Roman"
BLUE = RGBColor(5, 99, 193)


def set_run_font(run, *, latin: str, east_asia: str, size: float, bold: bool = False, color=None, underline=False):
    run.font.name = latin
    run._element.rPr.rFonts.set(qn("w:eastAsia"), east_asia)
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = color
    run.font.underline = underline


def reset_cell(cell):
    cell.text = ""
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return p


def set_cell(cell, text: str, *, latin: str, east_asia: str, size: float, bold=False, alignment=None):
    p = reset_cell(cell)
    if alignment is not None:
        p.alignment = alignment
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    run = p.add_run(text)
    set_run_font(run, latin=latin, east_asia=east_asia, size=size, bold=bold)
    return p


def set_contact_cell(cell, prefix: str, link_label: str, *, latin: str, east_asia: str, size: float):
    """Set the centered contact line and retain an actual Word hyperlink."""
    p = reset_cell(cell)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    add_run(p, prefix, latin=latin, east_asia=east_asia, size=size)
    add_hyperlink(p, link_label, PERSONAL_WEBSITE, latin=latin, east_asia=east_asia, size=size)
    return p


def set_section_cell(cell, title: str, *, english=False):
    return set_cell(
        cell,
        title,
        latin=EN_FONT if english else CN_HEADING,
        east_asia=CN_HEADING,
        size=12,
        bold=True,
    )


def compact_paragraph(p, *, space_after=0.4, line_spacing=1.0, left=0, first=0):
    fmt = p.paragraph_format
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(space_after)
    fmt.line_spacing = line_spacing
    fmt.left_indent = Pt(left)
    fmt.first_line_indent = Pt(first)
    fmt.keep_together = False


def add_run(p, text: str, *, latin: str, east_asia: str, size: float, bold=False, color=None, underline=False):
    run = p.add_run(text)
    set_run_font(run, latin=latin, east_asia=east_asia, size=size, bold=bold, color=color, underline=underline)
    return run


def add_hyperlink(p, text: str, url: str, *, latin: str, east_asia: str, size: float):
    """Append a real external hyperlink with compact blue underlined styling."""
    run = p.add_run(text)
    set_run_font(run, latin=latin, east_asia=east_asia, size=size, color=BLUE, underline=True)
    rid = p.part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), rid)
    p._p.remove(run._r)
    hyperlink.append(run._r)
    p._p.append(hyperlink)


def next_content_paragraph(cell):
    """Use the cell's required first paragraph before adding new ones."""
    if len(cell.paragraphs) == 1 and not cell.paragraphs[0].text:
        return cell.paragraphs[0]
    return cell.add_paragraph()


def add_group(cell, title: str, *, latin: str, east_asia: str, body_size: float, space_after=0.25, line_spacing=1.0):
    p = next_content_paragraph(cell)
    compact_paragraph(p, space_after=space_after, line_spacing=line_spacing, left=7, first=-7)
    add_run(p, "• ", latin=latin, east_asia=east_asia, size=body_size, bold=True)
    add_run(p, title, latin=latin, east_asia=east_asia, size=body_size, bold=True)


def add_numbered(cell, number: int, title: str, text: str, *, latin: str, east_asia: str, body_size: float, space_after=0.65, line_spacing=1.0):
    p = next_content_paragraph(cell)
    compact_paragraph(p, space_after=space_after, line_spacing=line_spacing, left=13, first=-11)
    add_run(p, f"{number}. ", latin=latin, east_asia=east_asia, size=body_size)
    add_run(p, title, latin=latin, east_asia=east_asia, size=body_size, bold=True)
    add_run(p, text, latin=latin, east_asia=east_asia, size=body_size)


def add_bullet(cell, title: str, text: str, *, latin: str, east_asia: str, body_size: float, space_after=0.65, line_spacing=1.0):
    p = next_content_paragraph(cell)
    compact_paragraph(p, space_after=space_after, line_spacing=line_spacing, left=7, first=-7)
    add_run(p, "• ", latin=latin, east_asia=east_asia, size=body_size)
    add_run(p, title, latin=latin, east_asia=east_asia, size=body_size, bold=True)
    add_run(p, text, latin=latin, east_asia=east_asia, size=body_size)
    return p


def body_cell(document: Document, cell, *, english=False):
    reset_cell(cell)
    # The empty first paragraph is retained by Word, but not used for content.
    first = cell.paragraphs[0]
    compact_paragraph(first, space_after=0, line_spacing=1.0)
    return cell


def move_tail_rows_to_next_page(table, first_tail_row: int):
    """Split one template table and place its tail after a hard page break.

    LibreOffice does not reliably honor page-break-before inside a table cell.
    Splitting the existing table preserves its exact grid/borders while keeping
    the concise projects block intact at the top of page two.
    """
    tail_table = deepcopy(table._tbl)
    for tr in list(tail_table.tr_lst)[:first_tail_row]:
        tail_table.remove(tr)
    for tr in list(table._tbl.tr_lst)[first_tail_row:]:
        table._tbl.remove(tr)

    page_break = OxmlElement("w:p")
    run = OxmlElement("w:r")
    br = OxmlElement("w:br")
    br.set(qn("w:type"), "page")
    run.append(br)
    page_break.append(run)
    table._tbl.addnext(page_break)
    page_break.addnext(tail_table)


def clear_row_height(row):
    """Remove a legacy fixed table-row height so content determines spacing."""
    tr_pr = row._tr.trPr
    if tr_pr is None:
        return
    for height in list(tr_pr.findall(qn("w:trHeight"))):
        tr_pr.remove(height)


def resume_template_tables(document: Document):
    """Return header/experience/skills tables and normalize a prior split run."""
    tables = document.tables
    if len(tables) == 3:
        return tables
    if len(tables) != 4:
        raise RuntimeError(f"Unexpected resume template table count: {len(tables)}")

    header, experience, projects, skills = tables
    for tr in list(projects._tbl.tr_lst):
        experience._tbl.append(deepcopy(tr))
    parent = projects._tbl.getparent()
    page_break = projects._tbl.getprevious()
    parent.remove(projects._tbl)
    if page_break is not None and page_break.tag == qn("w:p"):
        parent.remove(page_break)
    return header, experience, skills


def configure_resume_metadata(document: Document, *, title: str):
    props = document.core_properties
    props.title = title
    props.author = "冯浩然 / Haoran Feng"
    props.subject = "AI Infrastructure Resume"
    props.comments = "Updated 2026-09-15"


def update_chinese_resume(source: Path, output: Path):
    shutil.copy2(source, output)
    d = Document(output)
    configure_resume_metadata(d, title="冯浩然 - AI Infra 中文简历")
    t0, t1, t2 = resume_template_tables(d)

    set_cell(t0.cell(0, 0), "冯浩然", latin=EN_FONT, east_asia=CN_BODY, size=26, alignment=WD_ALIGN_PARAGRAPH.CENTER)
    set_contact_cell(
        t0.cell(1, 0),
        "电话：(+86)15024999885   邮箱：225010160@link.cuhk.edu.cn   个人网站：",
        "xfdg.github.io",
        latin=CN_BODY,
        east_asia=CN_BODY,
        size=10,
    )
    set_section_cell(t0.cell(3, 0), "教育经历")
    set_cell(t0.cell(4, 0), "2025.09-2027.06(预计)", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    set_cell(t0.cell(4, 1), "香港中文大学（深圳）", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    set_cell(t0.cell(4, 2), "集成电路与系统 硕士", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
    set_cell(t0.cell(5, 0), "2021.09-2025.06", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    set_cell(t0.cell(5, 1), "山东大学（985）", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    set_cell(t0.cell(5, 2), "电子科学与技术/计算机科学与技术 本科", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
    set_cell(t0.cell(6, 0), "获得荣誉：山东大学学业奖学金（前20%） 山东大学特长奖学金（竞赛创新）", latin=EN_FONT, east_asia=CN_BODY, size=10)

    # The template has merged cells in this table.  Use row cells consistently:
    # Table.cell() indexes the stale five-column grid and misaddresses later rows.
    internship_rows = t1.rows
    set_section_cell(internship_rows[0].cells[0], "实习经历")
    set_cell(internship_rows[1].cells[0], "2026.04-至今", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    set_cell(internship_rows[1].cells[1], "九坤投资-AI研究院", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    set_cell(internship_rows[1].cells[3], "大模型与高性能计算实习生", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)

    cell = body_cell(d, internship_rows[2].cells[0])
    body_size = 8.45
    add_bullet(cell, "300B 级 MoE 基座模型开发与优化：", "参与内部300B级MoE基座模型的训练、推理与RL rollout基础设施开发，围绕训练算子、推理算子和训推一致性进行性能优化、稳定性排障与多卡验证。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=6, line_spacing=1.1)
    add_group(cell, "训练算子支持", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=2.5, line_spacing=1.1)
    add_numbered(cell, 1, "FA3 deterministic SWA backward 的 dQ 依赖链调度优化：", "针对长序列、变长滑窗 Attention 的 deterministic backward 回退，负责问题归因、调度实现与验证；通过 Nsys/NCU 和 dQ/dK/dV 因子隔离，确认 dQ semaphore 依赖链占确定性增量99.67%。在原 fused main kernel 中将绝对 ticket 改为 contributor-relative ticket，并使 reverse scheduler 与归约顺序对齐；复用既有长度元数据分流，短序列保留 fallback，不新增 kernel/workspace/D2H。代表 packed shape 完整 backward 由6.755 ms降至1.699 ms（3.98×），2K-12K加速1.59-5.37×；1000次自身bitwise、3601/3601回归通过并交付 wheel。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=5.5, line_spacing=1.1)
    add_numbered(cell, 2, "Sink FC1 GEMM 与通信竞争控核探究：", "训练 trace 显示跨 stream SendRecv/AllGatherV 使 FC1 额外退化15.48%/28.73%；纠正实际 BF16 shape 为8192×3072×3072，确认约200 μs基线正常。搭建4×H200双 stream 代理，以 NCCL CTA 预算控制通信并行度、DeepGEMM SM budget 控制计算资源，并用 Nsys 校验实际 grid 与重叠；完成121组粗扫、89组细扫及复验。AllGatherV代理中12 CTA + DG120将联合 span 降低12.80%，同通信设置下控核独立贡献7.58%；AllToAllV主要收益来自通信并行度，不归因为 GEMM。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=5.5, line_spacing=1.1)
    add_group(cell, "推理算子支持", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=2.5, line_spacing=1.1)
    add_numbered(cell, 1, "CUDA Graph 下的确定性 MoE Router GEMM：", "针对 batch-invariant decode 小M下 persistent tile 无效计算及 CUDA Graph 确定性约束，负责在 vLLM 工程化接入 DeepGEMM/Triton Full-K/persistent三级后端；设计 tensor-signature selector、capture前数值与Graph preflight、cache/fallback、workspace生命周期和路径回归，replay固化后端、不在图内动态选核。48层 Router GEMM中位耗时下降73.39%，135/135 kernel、20/20模型场景逐位一致；prefix-cache hit下TP1/TP2整请求处理性能提升3.81%/4.36%。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=5.5, line_spacing=1.1)
    add_numbered(cell, 2, "OE 异步状态算子：", "将 recent-token history 与oe_input_ids构造保留在GPU，以Triton fused-hash消除TP1同步与回传，并修复prefill/decode/mixed batch/请求恢复/slot reuse/reorder的跨step状态；TP1严格28/28 case通过、吞吐较sync +4.99%。TP>1采用async-unfused + batch-invariant安全路径后，TP2/TP4均达84/84、0 mismatch，decode吞吐+4.6%/+3.0%。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=5.5, line_spacing=1.1)
    add_group(cell, "RL 训推一致性与 Rollout 稳定性", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=2.5, line_spacing=1.1)
    add_numbered(cell, 1, "R3 Router Replay 与多卡可观测性：", "针对rollout侧vLLM与训练侧Megatron的MoE路由偏差，打通[token, MoE-layer, top-k] route采集、传输与训练侧回放，建立response-mask对齐、异常检测及route mismatch/fτ²/KL闭环；8×H200、Qwen3-30B-A3B BF16的20-step对照中，route mismatch由17%-19%降至0，fτ²/KL分别降低36-145×/4-7×。进一步扩展至128卡集群，在内部300B级MoE基座模型上完成竞赛题单轮200-step rollout并接入SwanLab监控，关键指标与8卡对照一致。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=5.5, line_spacing=1.1)
    add_numbered(cell, 2, "FlashInfer CUDA Graph hang 排查与修复：", "针对H200、vLLM TP=2 FULL CUDA Graph rollout卡死，构建两卡最小复现，经TP1/TP2、eager/FULL Graph、fused/unfused等控制变量将根因收敛至fused AllReduce + RMSNorm中Lamport -0.0 sentinel的FTZ误判；回移上游0x80000000位级判断修复并完成模型级Graph on/off回归，重复Graph replay未再出现hang或timeout。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=5.5, line_spacing=1.1)
    p = cell.add_paragraph()
    compact_paragraph(p, space_after=5.5, line_spacing=1.1, left=0, first=0)
    add_run(p, "2026.01-2026.04", latin=CN_HEADING, east_asia=CN_HEADING, size=8.75, bold=True)
    add_run(p, "                         摩尔线程                         算子与编译器优化实习生", latin=CN_HEADING, east_asia=CN_HEADING, size=8.75, bold=True)
    add_bullet(cell, "TensorFlow MUSA Extension 算子、图优化与稳定性：", "负责muDNN GELU接入、GELU fusion链路修复、benchmark和热点算子优化，推动整网11个GELU全部融合，真实shape耗时降低36.6%；独立定位shape tensor误入device path并重构HostMemory，使inference 500轮成功率约30%提升至1000轮100%，4万/40万/80万轮长跑稳定；Logical_Or由21.2 μs降至10.7 μs，整网吞吐8187.48提升至8284.65。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=5.5, line_spacing=1.1)

    set_section_cell(internship_rows[3].cells[0], "开源贡献")
    set_cell(internship_rows[4].cells[0], "2026.08-09", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    set_cell(internship_rows[4].cells[2], "AI Infra 上游开源贡献", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
    cell = body_cell(d, internship_rows[5].cells[0])
    p = cell.paragraphs[0]
    compact_paragraph(p, space_after=0, line_spacing=1.0, left=7, first=-7)
    add_run(p, "• GitHub 已合入5个外部项目PR（XFDG）：Mooncake ", latin=EN_FONT, east_asia=CN_BODY, size=8.4, bold=True)
    for i, pr in enumerate(("#3601", "#3604", "#3660", "#3726")):
        add_hyperlink(p, pr, f"https://github.com/kvcache-ai/Mooncake/pull/{pr[1:]}", latin=EN_FONT, east_asia=CN_BODY, size=8.4)
        add_run(p, "/" if i < 3 else "（Merged），", latin=EN_FONT, east_asia=CN_BODY, size=8.4)
    add_run(p, "覆盖存储持久化、RDMA端口恢复、内存注销生命周期与队列失败传播；Mirage ", latin=EN_FONT, east_asia=CN_BODY, size=8.4)
    add_hyperlink(p, "#755", "https://github.com/mirage-project/mirage/pull/755", latin=EN_FONT, east_asia=CN_BODY, size=8.4)
    add_run(p, "（Merged）修复argmax中padding rows错误取值。", latin=EN_FONT, east_asia=CN_BODY, size=8.4)

    set_section_cell(internship_rows[6].cells[0], "项目与科研")
    set_cell(internship_rows[7].cells[0], "2025.12-至今", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    set_cell(internship_rows[7].cells[1], "", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    set_cell(internship_rows[7].cells[2], "基于 PyTorch Extension 的高性能 LLM 量化推理 Runtime 开发", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
    cell = body_cell(d, internship_rows[8].cells[0])
    add_bullet(cell, "H200 量化 Runtime 部署与TP2扩展：", "为在H200双卡交付32B量化模型推理，负责W4A16/W8A8/FP8后端适配、FlashAttention-2/4与GQA/softcap/SDPA链路接入，并以Compute Sanitizer定位gemv_kernel_g128尾组越界、补充边界保护；进一步实现Qwen2 packed-AWQ TP=2列/行切分与NCCL all-reduce。代表shape四类Sanitizer检查均为0 error/0 hazard；Qwen2.5-Coder-32B W4A16使checkpoint/峰值显存降低70.5%/66.8%，端到端输出吞吐提升76.8%，通过数值、跨rank token与交互验收。", latin=EN_FONT, east_asia=CN_BODY, body_size=8.55, space_after=5.5, line_spacing=1.1)
    clear_row_height(internship_rows[8])
    set_cell(internship_rows[9].cells[0], "2026.01-至今", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    set_cell(internship_rows[9].cells[1], "", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    set_cell(internship_rows[9].cells[2], "关键 Token 加权的思维链蒸馏｜AAAI 2027 已投稿", latin=CN_HEADING, east_asia=CN_HEADING, size=8.7, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
    cell = body_cell(d, internship_rows[10].cells[0])
    add_bullet(cell, "关键Token加权 CoT 蒸馏：", "针对通用CoT蒸馏对关键推理token区分不足的问题，参与“结构恢复—关键Token加权监督—偏好优化”三阶段框架，负责模型训练、超参调优与vLLM TP=4评测；以逐token扰动教师推理后参考答案生成似然的下降量估计重要性，并将其用于加权SFT与辅助损失。完成LoRA/DPO训练后，Qwen2.5-7B-Instruct在GSM8K/SVAMP取得94.01%/94.00% accuracy，较最强基线提升5.51/10.10个百分点；论文已投稿AAAI 2027。", latin=EN_FONT, east_asia=CN_BODY, body_size=8.55, space_after=5.5, line_spacing=1.1)

    set_section_cell(t2.cell(0, 0), "综合素质")
    cell = body_cell(d, t2.cell(1, 0))
    add_bullet(cell, "技术方向：", "关注GPU高性能算子开发与大模型训练/推理系统优化，具备Attention、MoE Router、量化GEMM、异步状态管理、Tensor Parallel与CUDA Graph工程实践。", latin=EN_FONT, east_asia=CN_BODY, body_size=8.7, space_after=4, line_spacing=1.1)
    add_bullet(cell, "开发与性能工程：", "熟练使用C/C++、Python、CUDA、Triton、PyTorch Extension；能够使用Nsys、NCU、Compute Sanitizer完成算子性能分析与正确性验证。", latin=EN_FONT, east_asia=CN_BODY, body_size=8.7, space_after=4, line_spacing=1.1)
    add_bullet(cell, "语言与证书：", "IELTS 6.5、CET-6，持有华为HCIA-AI认证；具备英文技术文档阅读、检索与跨仓源码分析能力。", latin=EN_FONT, east_asia=CN_BODY, body_size=8.7, space_after=0, line_spacing=1.1)

    move_tail_rows_to_next_page(t1, 3)
    d.save(output)


def update_chinese_resume_variant(source: Path, output: Path, *, focus: str):
    """Create a two-page Chinese resume whose evidence is truly focus-specific.

    These are deliberately not the general resume with a different heading.  The
    internship body, project order, and skills all change with the target role:
    the training version is built around backward/MoE training work, while the
    inference version is built around serving kernels and runtime execution.
    """
    update_chinese_resume(source, output)
    d = Document(output)
    if len(d.tables) != 4:
        raise RuntimeError(f"Expected four tables in focused resume, got {len(d.tables)}")
    _, internship, projects, skills = d.tables
    body_size = 8.45

    cell = body_cell(d, internship.rows[2].cells[0])
    if focus == "training":
        configure_resume_metadata(d, title="冯浩然 - AI Infra 训练优化简历")
        add_bullet(cell, "300B 级 MoE 基座模型训练算子与 RL 训练基础设施：", "面向内部300B级MoE基座模型，负责训练侧热点算子与rollout训练闭环；以trace归因、最小可控实验和数值/确定性回归推进优化，核心聚焦Attention backward及MoE计算—通信竞争。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=6, line_spacing=1.1)
        add_group(cell, "训练算子优化", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=2.5, line_spacing=1.1)
        add_numbered(cell, 1, "FA3 deterministic SWA backward 的 dQ 依赖链调度：", "针对长序列、变长滑窗Attention的deterministic backward回退，先以Nsys/NCU和dQ/dK/dV因子隔离将99.67%的确定性增量收敛到dQ semaphore链；随后在原fused main kernel中将绝对ticket改为contributor-relative ticket，并对齐reverse scheduler与归约顺序，不新增kernel/workspace/D2H。代表packed trace完整backward由6.755 ms降至1.699 ms（3.98×），2K-12K加速1.59-5.37×；1000次自身bitwise与3601/3601回归通过。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=5.5, line_spacing=1.1)
        add_numbered(cell, 2, "MoE Router orth-loss 的 compiled 子图融合：", "针对每层orth-loss重复构造identity、Gram GEMM之外仍有normalize、margin、pointwise与reduction碎片化的问题，推导identity-free/margin等价式，并以torch.compile(fullgraph=True)融合周边子图；识别V1 in-place触发CopySlices backward回退后，最终采用默认关闭的V3 compiled路径。steady forward GPU operation由17降至7，目标函数forward+backward提升18.77%、峰值显存降低11.11%；4×H200完整MoE Layer forward+backward提升3.97%，value/gradient 9/9通过。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=5.5, line_spacing=1.1)
        add_numbered(cell, 3, "Sink FC1 GEMM—通信竞争的控核代理实验：", "针对训练trace中跨stream SendRecv/AllGatherV使FC1额外退化15.48%/28.73%的现象，先校正真实BF16 shape为8192×3072×3072并排除“慢GEMM”误判；构建4×H200双stream的all_gather(list)代理，以NCCL CTA预算控制通信并发、DeepGEMM SM budget控核，完成121组粗扫、89组细扫及Nsys grid/重叠复验。12 CTA + DG120使代理joint span降低12.80%，同通信设置下控核独立贡献7.58%；结果用于下一步训练trace验证，不外推为full-step收益。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=5.5, line_spacing=1.1)
        add_group(cell, "RL 训练侧路由闭环", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=2.5, line_spacing=1.1)
        add_numbered(cell, 1, "R3 Router Replay 的训练侧回放与观测：", "为消除rollout侧vLLM与训练侧Megatron的MoE路由偏差，打通[token, MoE-layer, top-k]采集、传输、response-mask对齐与训练回放，建立route mismatch/fτ²/KL异常闭环；8×H200、20-step对照中mismatch由17%-19%降至0，fτ²/KL降低36-145×/4-7×，并完成128卡内部300B级模型单轮200-step监控运行。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=5.5, line_spacing=1.1)

        set_cell(projects.rows[4].cells[0], "2026.01-至今", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
        set_cell(projects.rows[4].cells[2], "关键 Token 加权的思维链蒸馏｜AAAI 2027 已投稿", latin=CN_HEADING, east_asia=CN_HEADING, size=8.7, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
        cell = body_cell(d, projects.rows[5].cells[0])
        add_bullet(cell, "关键Token加权 CoT 蒸馏：", "针对通用CoT蒸馏无法区分关键推理token的问题，参与“结构恢复—关键Token加权监督—偏好优化”三阶段训练框架，负责训练、超参调优和vLLM TP=4评测；以逐token扰动教师推理后参考答案的生成似然下降量估计重要性，并用于加权SFT与辅助损失。Qwen2.5-7B-Instruct在GSM8K/SVAMP达到94.01%/94.00%，较最强基线提升5.51/10.10个百分点；论文已投稿AAAI 2027。", latin=EN_FONT, east_asia=CN_BODY, body_size=8.55, space_after=5.5, line_spacing=1.1)
        set_cell(projects.rows[6].cells[0], "2026.08-09", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
        set_cell(projects.rows[6].cells[2], "Blackwell 多精度训练验证（B200）", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
        cell = body_cell(d, projects.rows[7].cells[0])
        add_bullet(cell, "Blackwell 多精度训练验证：", "为验证低精度训练链路的数值、重启与强扩展稳定性，以random-init + mock data重建dense decoder工作负载，建立BF16/FP8/MXFP8/NVFP4、1-8 GPU与80个Transformer Engine forward+backward shape的矩阵，并覆盖checkpoint save/load/restart。BF16/NVFP4强扩展24/24正式运行、720 optimizer steps的step-time CV<1%；8卡NVFP4相对BF16吞吐1.265×、板卡tokens/J 1.434×，80/80有限值且checkpoint恢复loss相对差异1.38e-6（不外推真实数据收敛）。", latin=EN_FONT, east_asia=CN_BODY, body_size=8.55, space_after=5.5, line_spacing=1.1)
        skill_focus = "聚焦GPU训练算子与大模型训练优化，具备Attention backward、MoE GEMM—通信竞争、RL训练回放和多卡确定性/数值验证经验。"
    elif focus == "inference":
        configure_resume_metadata(d, title="冯浩然 - AI Infra 推理优化简历")
        add_bullet(cell, "300B 级 MoE 基座模型推理内核与 Runtime 优化：", "面向内部300B级MoE基座模型的serving与RL rollout执行链路，负责从shape感知选核、CUDA Graph确定性到跨step状态管理和Tensor Parallel验证的推理侧优化。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=6, line_spacing=1.1)
        add_group(cell, "推理算子与执行路径优化", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=2.5, line_spacing=1.1)
        add_numbered(cell, 1, "CUDA Graph 下的确定性 MoE Router GEMM：", "针对batch-invariant decode小M下persistent tile无效计算和Graph replay不能动态选核的约束，在vLLM工程化接入DeepGEMM/Triton Full-K/persistent三级后端；负责tensor-signature selector、capture前数值/Graph preflight、cache/fallback、workspace生命周期与路径回归，replay中固化后端决策。48层Router GEMM中位耗时下降73.39%，135/135 kernel、20/20模型场景逐位一致；prefix-cache hit下TP1/TP2整请求性能提升3.81%/4.36%。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=5.5, line_spacing=1.1)
        add_numbered(cell, 2, "OE 异步状态算子与多卡安全路径：", "将recent-token history和oe_input_ids构造留在GPU，以Triton fused-hash移除TP1的GPU同步与H→D回传bubble；同时覆盖prefill/decode/mixed batch/恢复/slot reuse/reorder的跨step状态语义。TP1严格28/28 case通过、吞吐较sync +4.99%；TP>1采用async-unfused + batch-invariant安全路径后，TP2/TP4均84/84、0 mismatch，decode吞吐+4.6%/+3.0%。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=5.5, line_spacing=1.1)
        add_numbered(cell, 3, "CUDA Graph Rollout hang 的 Kernel 级排障：", "针对H200、vLLM TP=2 FULL CUDA Graph下rollout卡死，构建两卡最小复现并以TP/Graph/fused控制变量定位fused AllReduce + RMSNorm中Lamport -0.0 sentinel的FTZ误判；采用0x80000000位级判断后，模型级Graph on/off反复回归未再出现hang或timeout。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=5.5, line_spacing=1.1)
        add_group(cell, "Rollout 一致性验证", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=2.5, line_spacing=1.1)
        add_numbered(cell, 1, "R3 Router Replay：", "为验证rollout推理与训练侧路由对齐，完成route采集、response-mask对齐、训练回放及mismatch/fτ²/KL闭环；8×H200、20-step中route mismatch由17%-19%降至0，并扩展至128卡内部300B级模型的单轮200-step观测。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=5.5, line_spacing=1.1)

        set_cell(projects.rows[4].cells[0], "2025.12-至今", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
        set_cell(projects.rows[4].cells[2], "基于 PyTorch Extension 的高性能 LLM 量化推理 Runtime 开发", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
        cell = body_cell(d, projects.rows[5].cells[0])
        add_bullet(cell, "H200 量化 Runtime 部署与 TP2 扩展：", "为在双H200交付32B量化模型推理，负责W4A16/W8A8/FP8后端、FlashAttention-2/4与GQA/softcap/SDPA链路适配；以Compute Sanitizer定位gemv_kernel_g128尾组越界并补齐边界保护，继而实现Qwen2 packed-AWQ TP=2的Q/K/V、gate/up列切分及O/down行切分与NCCL all-reduce。代表shape四类Sanitizer均0 error/0 hazard；32B W4A16使checkpoint/峰值显存降低70.5%/66.8%，端到端输出吞吐提升76.8%。", latin=EN_FONT, east_asia=CN_BODY, body_size=8.55, space_after=5.5, line_spacing=1.1)
        set_cell(projects.rows[6].cells[0], "2026.01-至今", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
        set_cell(projects.rows[6].cells[2], "关键 Token 加权的思维链蒸馏｜AAAI 2027 已投稿", latin=CN_HEADING, east_asia=CN_HEADING, size=8.7, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
        cell = body_cell(d, projects.rows[7].cells[0])
        add_bullet(cell, "关键Token加权 CoT 蒸馏：", "负责训练、调优与vLLM TP=4评测；Qwen2.5-7B-Instruct在GSM8K/SVAMP达到94.01%/94.00%，较最强基线提升5.51/10.10个百分点。", latin=EN_FONT, east_asia=CN_BODY, body_size=8.55, space_after=5.5, line_spacing=1.1)
        skill_focus = "聚焦GPU推理算子与大模型Serving/Rollout优化，具备CUDA Graph、MoE Router GEMM、异步状态管理、量化Runtime、Tensor Parallel和多卡确定性验证经验。"
    else:
        raise ValueError(f"Unsupported resume focus: {focus}")

    p = internship.rows[2].cells[0].add_paragraph()
    compact_paragraph(p, space_after=5.5, line_spacing=1.1, left=0, first=0)
    add_run(p, "2026.01-2026.04", latin=CN_HEADING, east_asia=CN_HEADING, size=8.75, bold=True)
    add_run(p, "                         摩尔线程                         算子与编译器优化实习生", latin=CN_HEADING, east_asia=CN_HEADING, size=8.75, bold=True)
    add_bullet(internship.rows[2].cells[0], "TensorFlow MUSA Extension 算子与编译器优化：", "负责TensorFlow on MUSA的算子支持、图优化与稳定性治理：完成muDNN GELU接入、GELU fusion链路修复及benchmark，推动整网11个GELU全部融合、真实shape耗时降低36.6%；独立定位StridedSlice<int32>/Pack<int32>的shape tensor误入device path根因并重构HostMemory路径，使inference 500轮成功率约30%提升至1000轮100%，4万/40万/80万轮长跑稳定；优化Logical_Or scalar broadcast将21.2 μs降至10.7 μs，整网吞吐由8187.48提升至8284.65。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=5.5, line_spacing=1.1)

    cell = body_cell(d, skills.cell(1, 0))
    add_bullet(cell, "技术方向：", skill_focus, latin=EN_FONT, east_asia=CN_BODY, body_size=8.7, space_after=4, line_spacing=1.1)
    add_bullet(cell, "开发与性能工程：", "熟练使用C/C++、Python、CUDA、Triton、PyTorch Extension；能够使用Nsys、NCU、Compute Sanitizer完成算子性能分析与正确性验证。", latin=EN_FONT, east_asia=CN_BODY, body_size=8.7, space_after=4, line_spacing=1.1)
    add_bullet(cell, "语言与证书：", "IELTS 6.5、CET-6，持有华为HCIA-AI认证；具备英文技术文档阅读、检索与跨仓源码分析能力。", latin=EN_FONT, east_asia=CN_BODY, body_size=8.7, space_after=0, line_spacing=1.1)
    d.save(output)


def add_word_watermark(document: Document, text: str):
    """Add a standard Word/Office VML watermark without changing page flow."""
    watermark_xml = f'''<w:pict xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        xmlns:v="urn:schemas-microsoft-com:vml"
        xmlns:o="urn:schemas-microsoft-com:office:office">
      <v:shape id="XiaohongshuWatermark" o:spid="_x0000_s1026" type="#_x0000_t136"
        style="position:absolute;margin-left:0;margin-top:0;width:430pt;height:58pt;rotation:315;z-index:-251654144;mso-position-horizontal:center;mso-position-horizontal-relative:margin;mso-position-vertical:center;mso-position-vertical-relative:margin"
        fillcolor="#B22222" stroked="f">
        <v:fill opacity="0.16"/>
        <v:textpath style="font-family:&quot;宋体&quot;;font-size:1pt" string="{text}"/>
      </v:shape>
    </w:pict>'''
    for section in document.sections:
        header = section.header
        p = header.paragraphs[0]
        p.clear()
        p._p.append(parse_xml(watermark_xml))


def update_anonymous_chinese_resume(source: Path, output: Path, *, word_watermark=True):
    """Build a de-identified Chinese resume without retaining personal links."""
    update_chinese_resume(source, output)
    d = Document(output)
    if len(d.tables) != 4:
        raise RuntimeError(f"Expected four tables in anonymized resume, got {len(d.tables)}")
    t0, t1, t2, t3 = d.tables

    d.core_properties.title = "AI Infra Resume - Anonymous"
    d.core_properties.author = "Anonymous Candidate"
    d.core_properties.subject = "AI Infrastructure Resume"

    set_cell(t0.cell(0, 0), "匿名候选人", latin=EN_FONT, east_asia=CN_BODY, size=26, alignment=WD_ALIGN_PARAGRAPH.CENTER)
    set_cell(t0.cell(1, 0), "联系方式：可在面试阶段提供", latin=CN_BODY, east_asia=CN_BODY, size=10, alignment=WD_ALIGN_PARAGRAPH.CENTER)
    set_section_cell(t0.cell(3, 0), "教育经历")
    set_cell(t0.cell(4, 0), "硕士在读", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    set_cell(t0.cell(4, 1), "某港校", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    set_cell(t0.cell(4, 2), "集成电路与系统", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
    set_cell(t0.cell(5, 0), "本科", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    set_cell(t0.cell(5, 1), "某985高校", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    set_cell(t0.cell(5, 2), "电子/计算机相关专业", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
    set_cell(t0.cell(6, 0), "获得荣誉：校级学业奖学金（前20%）及竞赛创新奖学金", latin=EN_FONT, east_asia=CN_BODY, size=10)

    rows = t1.rows
    set_section_cell(rows[0].cells[0], "实习经历")
    set_cell(rows[1].cells[0], "近期", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    set_cell(rows[1].cells[1], "某量化机构 AI 研究院", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    set_cell(rows[1].cells[3], "大模型与高性能计算实习生", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
    cell = body_cell(d, rows[2].cells[0])
    body_size = 8.45
    add_bullet(cell, "百亿级 MoE 基座模型开发与优化：", "参与训练、推理与RL rollout基础设施开发，围绕训练算子、推理算子和训推一致性完成性能优化、稳定性排障与多卡验证。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=6, line_spacing=1.1)
    add_group(cell, "训练算子支持", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=2.5, line_spacing=1.1)
    add_numbered(cell, 1, "FA3 deterministic SWA backward：", "定位长序列变长滑窗Attention deterministic backward回退的dQ依赖链，并在原fused kernel内完成ticket与scheduler对齐；代表shape完整backward由6.755 ms降至1.699 ms（3.98×），3601/3601回归和1000次逐位重复通过。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=5.5, line_spacing=1.1)
    add_numbered(cell, 2, "Sink FC1 GEMM 与通信竞争控核：", "构建4×H200双stream代理，以NCCL CTA预算和DeepGEMM SM budget控制通信、计算资源，完成121组粗扫和89组细扫；AllGatherV代理中联合span降低12.80%，其中控核独立贡献7.58%。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=5.5, line_spacing=1.1)
    add_group(cell, "推理算子支持", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=2.5, line_spacing=1.1)
    add_numbered(cell, 1, "CUDA Graph 确定性 MoE Router GEMM：", "在推理引擎中接入多后端选择、capture前preflight、cache/fallback与workspace生命周期；48层Router GEMM中位耗时降低73.39%，135/135 kernel和20/20模型场景逐位一致。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=5.5, line_spacing=1.1)
    add_numbered(cell, 2, "OE 异步状态算子：", "将历史token构造保留在GPU，并修复prefill/decode/mixed batch/恢复/slot reuse/reorder的跨step状态；TP1 28/28 case通过、吞吐+4.99%，TP2/TP4均84/84、0 mismatch。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=5.5, line_spacing=1.1)
    add_group(cell, "RL 训推一致性与 Rollout 稳定性", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=2.5, line_spacing=1.1)
    add_numbered(cell, 1, "Router Replay 与多卡可观测性：", "打通路由采集、传输、训练侧回放与指标闭环；代表性MoE模型20-step对照中，route mismatch由17%-19%降至0，fτ²/KL分别降低36-145×/4-7×，并扩展至128卡监控运行。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=5.5, line_spacing=1.1)
    add_numbered(cell, 2, "CUDA Graph hang 排查与修复：", "以两卡最小复现和TP/Graph/fused控制变量定位FTZ导致的Lamport sentinel误判，采用0x80000000位级判断修复；模型级Graph on/off回归中未再出现hang或timeout。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=5.5, line_spacing=1.1)
    p = cell.add_paragraph()
    compact_paragraph(p, space_after=5.5, line_spacing=1.1, left=0, first=0)
    add_run(p, "此前", latin=CN_HEADING, east_asia=CN_HEADING, size=8.75, bold=True)
    add_run(p, "                               某GPU厂商                               算子与编译器优化实习生", latin=CN_HEADING, east_asia=CN_HEADING, size=8.75, bold=True)
    add_bullet(cell, "TensorFlow设备后端、图优化与稳定性：", "完成GELU接入、融合修复与热点路径优化，推动整网11个GELU融合、真实shape耗时降低36.6%；重构HostMemory路径使1000轮成功率达到100%，并完成长跑验证。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=5.5, line_spacing=1.1)

    rows = t2.rows
    set_section_cell(rows[0].cells[0], "开源贡献")
    set_cell(rows[1].cells[0], "近年", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    set_cell(rows[1].cells[2], "外部开源贡献", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
    cell = body_cell(d, rows[2].cells[0])
    add_bullet(cell, "已合入多个外部项目PR：", "涉及存储持久化、RDMA端口恢复、内存注销生命周期、队列失败传播及算子边界条件修复。", latin=EN_FONT, east_asia=CN_BODY, body_size=8.55, space_after=4, line_spacing=1.1)
    set_section_cell(rows[3].cells[0], "项目与科研")
    set_cell(rows[4].cells[0], "近期", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    set_cell(rows[4].cells[1], "", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    set_cell(rows[4].cells[2], "高性能大模型量化推理 Runtime", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
    cell = body_cell(d, rows[5].cells[0])
    add_bullet(cell, "量化 Runtime 部署与TP扩展：", "完成多精度后端、Attention与多卡构建适配，定位并修复量化GEMV尾组越界；实现TP=2切分与all-reduce，代表性双卡模型显存显著降低、输出吞吐提升76.8%。", latin=EN_FONT, east_asia=CN_BODY, body_size=8.55, space_after=5.5, line_spacing=1.1)
    set_cell(rows[6].cells[0], "近期", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    set_cell(rows[6].cells[1], "", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    set_cell(rows[6].cells[2], "关键Token加权 CoT 蒸馏研究", latin=CN_HEADING, east_asia=CN_HEADING, size=8.7, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
    cell = body_cell(d, rows[7].cells[0])
    add_bullet(cell, "训练与评测：", "负责训练、调优与多卡评测，将token重要性用于加权监督；7B模型在两个数学推理基准上分别达到94.01%/94.00%，较最强基线提升5.51/10.10个百分点。", latin=EN_FONT, east_asia=CN_BODY, body_size=8.55, space_after=5.5, line_spacing=1.1)

    set_section_cell(t3.cell(0, 0), "综合素质")
    cell = body_cell(d, t3.cell(1, 0))
    add_bullet(cell, "技术方向：", "GPU高性能算子开发与大模型训练/推理系统优化。", latin=EN_FONT, east_asia=CN_BODY, body_size=8.7, space_after=4, line_spacing=1.1)
    add_bullet(cell, "开发与性能工程：", "C/C++、Python、CUDA、Triton、PyTorch Extension；熟悉Nsys、NCU与Compute Sanitizer。", latin=EN_FONT, east_asia=CN_BODY, body_size=8.7, space_after=4, line_spacing=1.1)
    add_bullet(cell, "语言：", "IELTS 6.5、CET-6，具备英文技术文档阅读与跨仓源码分析能力。", latin=EN_FONT, east_asia=CN_BODY, body_size=8.7, space_after=0, line_spacing=1.1)
    if word_watermark:
        add_word_watermark(d, WATERMARK_TEXT)
    d.save(output)


def update_english_resume(source: Path, output: Path):
    shutil.copy2(source, output)
    d = Document(output)
    configure_resume_metadata(d, title="Haoran Feng - AI Infrastructure Resume")
    t0, t1, t2 = resume_template_tables(d)

    set_cell(t0.cell(0, 0), "Haoran Feng", latin=EN_FONT, east_asia=CN_BODY, size=26, alignment=WD_ALIGN_PARAGRAPH.CENTER)
    set_contact_cell(
        t0.cell(1, 0),
        "Phone: (+86) 150 2499 9885    Email: 225010160@link.cuhk.edu.cn    Website: ",
        "xfdg.github.io",
        latin=EN_FONT,
        east_asia=CN_BODY,
        size=10,
    )
    set_section_cell(t0.cell(3, 0), "EDUCATION", english=True)
    set_cell(t0.cell(4, 0), "Sep 2025 - Jun 2027 (Expected)", latin=EN_FONT, east_asia=CN_BODY, size=9, bold=True)
    set_cell(t0.cell(4, 1), "CUHK-Shenzhen", latin=EN_FONT, east_asia=CN_BODY, size=9, bold=True)
    set_cell(t0.cell(4, 2), "M.S., Integrated Circuits and Systems", latin=EN_FONT, east_asia=CN_BODY, size=9, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
    set_cell(t0.cell(5, 0), "Sep 2021 - Jun 2025", latin=EN_FONT, east_asia=CN_BODY, size=9, bold=True)
    set_cell(t0.cell(5, 1), "Shandong University (985)", latin=EN_FONT, east_asia=CN_BODY, size=9, bold=True)
    set_cell(t0.cell(5, 2), "B.Eng., Electronic Science / Computer Science", latin=EN_FONT, east_asia=CN_BODY, size=9, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
    set_cell(t0.cell(6, 0), "Honors: Academic Scholarship (Top 20%); Special Talent Scholarship in Innovation, Shandong University", latin=EN_FONT, east_asia=CN_BODY, size=10)

    internship_rows = t1.rows
    set_section_cell(internship_rows[0].cells[0], "INTERNSHIP EXPERIENCE", english=True)
    set_cell(internship_rows[1].cells[0], "Apr 2026 - Present", latin=EN_FONT, east_asia=CN_BODY, size=9, bold=True)
    set_cell(internship_rows[1].cells[1], "Ubiquant Investment - AI Lab", latin=EN_FONT, east_asia=CN_BODY, size=9, bold=True)
    set_cell(internship_rows[1].cells[3], "LLM and HPC Intern", latin=EN_FONT, east_asia=CN_BODY, size=9, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)

    cell = body_cell(d, internship_rows[2].cells[0])
    body_size = 8.0
    add_bullet(cell, "300B-class MoE foundation model development and optimization: ", "Developing training, inference, and RL-rollout infrastructure for an internal 300B-class MoE foundation model, with ownership spanning kernel performance, runtime stability, and multi-GPU validation.", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=0.45)
    add_group(cell, "Training Kernel Support", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    add_numbered(cell, 1, "FA3 deterministic SWA backward: ", "Used Nsys/NCU factorization to attribute 99.67% of deterministic overhead to the dQ semaphore chain. Implemented contributor-relative tickets aligned with the reverse scheduler in the fused kernel, with no new kernel/workspace/D2H and a short-sequence fallback. Cut packed backward from 6.755 to 1.699 ms (3.98x), achieved 1.59-5.37x at 2K-12K, and passed 3,601/3,601 regressions plus 1,000 bitwise repeats.", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    add_numbered(cell, 2, "Sink FC1 GEMM and communication contention: ", "Corrected FC1 to BF16 8192x3072x3072 and isolated cross-stream communication contention rather than a slow GEMM. Built a 4xH200 dual-stream proxy, sweeping NCCL CTA budgets and DeepGEMM SM budgets (121 coarse, 89 fine) with Nsys grid/overlap checks. In the AllGather proxy, 12 CTA + DG120 reduced joint span by 12.80%, including 7.58% independent gain from compute throttling; AllToAll gains were attributed to communication parallelism.", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    add_group(cell, "Inference Kernel Support", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    add_numbered(cell, 1, "Deterministic MoE Router GEMM under CUDA Graph: ", "Integrated DeepGEMM, Triton Full-K, and persistent backends in vLLM; owned the tensor-signature selector, pre-capture numeric/graph preflight, cache/fallback, workspace lifecycle, and path tests. Replay freezes the backend decision. Reduced median latency of 48 router GEMMs by 73.39%, passed 135/135 kernel and 20/20 model bitwise checks, and improved prefix-cache-hit request throughput by 3.81%/4.36% on TP1/TP2.", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    add_numbered(cell, 2, "Asynchronous OE state operator: ", "Kept recent-token history and oe_input_ids construction on GPU with Triton fused hashing, fixing cross-step state for prefill/decode/mixed batches/recovery/slot reuse/reorder. Passed 28/28 TP1 cases with +4.99% throughput; the TP>1 async-unfused batch-invariant path reached 84/84 with zero mismatches and +4.6%/+3.0% decode throughput on TP2/TP4.", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    add_group(cell, "RL Training-Inference Consistency and Rollout Reliability", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    add_numbered(cell, 1, "R3 Router Replay and multi-GPU observability: ", "Built token/layer/top-k route capture, transport, training-side replay, response-mask alignment, and mismatch/f-tau2/KL diagnostics between vLLM and Megatron. On 8xH200/Qwen3-30B-A3B, reduced route mismatch from 17-19% to 0, f-tau2 by 36-145x, and KL by 4-7x; extended to a monitored 128-GPU, 200-step rollout on an internal 300B-class MoE foundation model with aligned key metrics.", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    add_numbered(cell, 2, "FlashInfer CUDA Graph hang: ", "Reproduced a vLLM TP=2 FULL CUDA Graph rollout hang on two H200 GPUs. TP/eager/graph/fused controls traced it to FTZ misclassifying a Lamport -0.0 sentinel in fused AllReduce + RMSNorm. Backported the upstream 0x80000000 bitwise fix; repeated model graph on/off replays had no hang or timeout.", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    p = cell.add_paragraph()
    compact_paragraph(p, space_after=0.55, line_spacing=1.0)
    add_run(p, "Jan 2026 - Apr 2026", latin=EN_FONT, east_asia=CN_BODY, size=8.5, bold=True)
    add_run(p, "                    Moore Threads                    Operator & Compiler Optimization Intern", latin=EN_FONT, east_asia=CN_BODY, size=8.5, bold=True)
    add_bullet(cell, "TensorFlow MUSA Extension, graph optimization, and reliability: ", "Integrated muDNN GELU, repaired the fusion pipeline, and optimized hot paths; all 11 GELUs fused and representative-shape latency fell 36.6%. Reworked HostMemory after tracing shape tensors entering the device path, lifting success from about 30% at 500 iterations to 100% at 1,000 and passing 40K/400K/800K stress runs. Reduced Logical_Or from 21.2 to 10.7 us and raised end-to-end throughput from 8187.48 to 8284.65.", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=0.1)

    set_section_cell(internship_rows[3].cells[0], "OPEN-SOURCE CONTRIBUTIONS", english=True)
    set_cell(internship_rows[4].cells[0], "Aug - Sep 2026", latin=EN_FONT, east_asia=CN_BODY, size=9, bold=True)
    set_cell(internship_rows[4].cells[2], "Merged External Contributions", latin=EN_FONT, east_asia=CN_BODY, size=9, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
    cell = body_cell(d, internship_rows[5].cells[0])
    p = cell.paragraphs[0]
    compact_paragraph(p, space_after=0, line_spacing=1.0, left=7, first=-7)
    add_run(p, "• 5 merged external PRs (XFDG): Mooncake ", latin=EN_FONT, east_asia=CN_BODY, size=8.15, bold=True)
    for i, pr in enumerate(("#3601", "#3604", "#3660", "#3726")):
        add_hyperlink(p, pr, f"https://github.com/kvcache-ai/Mooncake/pull/{pr[1:]}", latin=EN_FONT, east_asia=CN_BODY, size=8.15)
        add_run(p, "/" if i < 3 else " (Merged), ", latin=EN_FONT, east_asia=CN_BODY, size=8.15)
    add_run(p, "covering storage durability, RDMA recovery, memory-unregister lifecycle, and queue error propagation; Mirage ", latin=EN_FONT, east_asia=CN_BODY, size=8.15)
    add_hyperlink(p, "#755", "https://github.com/mirage-project/mirage/pull/755", latin=EN_FONT, east_asia=CN_BODY, size=8.15)
    add_run(p, " (Merged) fixes argmax padding rows.", latin=EN_FONT, east_asia=CN_BODY, size=8.15)

    set_section_cell(internship_rows[6].cells[0], "PROJECTS AND RESEARCH", english=True)
    set_cell(internship_rows[7].cells[0], "Dec 2025 - Present", latin=EN_FONT, east_asia=CN_BODY, size=9, bold=True)
    set_cell(internship_rows[7].cells[1], "", latin=EN_FONT, east_asia=CN_BODY, size=9, bold=True)
    set_cell(internship_rows[7].cells[2], "High-Performance Quantized LLM Runtime with PyTorch Extensions", latin=EN_FONT, east_asia=CN_BODY, size=8.8, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
    cell = body_cell(d, internship_rows[8].cells[0])
    add_bullet(cell, "H200 quantized-runtime deployment and TP2 extension: ", "To deliver 32B quantized inference on two H200 GPUs, owned W4A16/W8A8/FP8 backend adaptation, FlashAttention-2/4 and GQA/soft-cap/SDPA integration, and kernel debugging. Used Compute Sanitizer to fix a tail-group out-of-bounds access, then implemented packed-AWQ TP=2 sharding with NCCL all-reduce. All four Sanitizer modes passed representative shapes; two-GPU Qwen2.5-Coder-32B W4A16 reduced checkpoint/peak memory by 70.5%/66.8% and increased end-to-end output throughput by 76.8%.", latin=EN_FONT, east_asia=CN_BODY, body_size=8.0, space_after=0.2)
    clear_row_height(internship_rows[8])
    set_cell(internship_rows[9].cells[0], "Jan 2026 - Present", latin=EN_FONT, east_asia=CN_BODY, size=9, bold=True)
    set_cell(internship_rows[9].cells[1], "", latin=EN_FONT, east_asia=CN_BODY, size=9, bold=True)
    set_cell(internship_rows[9].cells[2], "Token-Weighted CoT Distillation | AAAI 2027 Submission", latin=EN_FONT, east_asia=CN_BODY, size=8.5, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
    cell = body_cell(d, internship_rows[10].cells[0])
    add_bullet(cell, "Token-weighted CoT distillation: ", "To address the lack of token-level importance awareness in generic CoT distillation, contributed to a three-stage structure-recovery, token-weighted supervision, and preference-optimization pipeline. Owned training, hyperparameter tuning, and vLLM TP=4 evaluation; estimated importance from reference-answer likelihood drops after teacher-rationale perturbations and applied it to weighted SFT plus an auxiliary loss. Qwen2.5-7B-Instruct reached 94.01%/94.00% on GSM8K/SVAMP, +5.51/+10.10 points over the strongest baselines; manuscript submitted to AAAI 2027.", latin=EN_FONT, east_asia=CN_BODY, body_size=8.0, space_after=0.1)

    set_section_cell(t2.cell(0, 0), "SKILLS", english=True)
    cell = body_cell(d, t2.cell(1, 0))
    add_bullet(cell, "Focus: ", "GPU high-performance kernel development and LLM training/inference optimization; experience in Attention, MoE routing, quantized GEMM, async state, tensor parallelism, and CUDA Graphs.", latin=EN_FONT, east_asia=CN_BODY, body_size=8.15, space_after=0.2)
    add_bullet(cell, "Development and performance engineering: ", "C/C++, Python, CUDA, Triton, and PyTorch Extensions; profiling and correctness validation with Nsight Systems, Nsight Compute, and Compute Sanitizer.", latin=EN_FONT, east_asia=CN_BODY, body_size=8.15, space_after=0.2)
    add_bullet(cell, "Languages and credentials: ", "IELTS 6.5; CET-6; Huawei HCIA-AI certified. Experienced in reading technical documentation, source analysis, and cross-repository debugging.", latin=EN_FONT, east_asia=CN_BODY, body_size=8.15, space_after=0)

    move_tail_rows_to_next_page(t1, 6)
    d.save(output)


def update_interview_intro(source: Path, output: Path):
    shutil.copy2(source, output)
    d = Document(output)
    configure_resume_metadata(d, title="冯浩然 - 面试介绍精简版")
    for p in d.paragraphs:
        p.clear()
    content = [
        "面试官您好，我叫冯浩然，目前在香港中文大学（深圳）攻读集成电路与系统硕士，本科毕业于山东大学。我主要关注 GPU 高性能算子开发和大模型训练/推理系统优化。",
        "目前我在九坤投资 AI 研究院实习，工作分为训练算子、推理算子，以及 RL 训推一致性与 rollout 稳定性三条主线。",
        "训练侧，我定位到长序列 deterministic Attention backward 的主要回退来自 dQ semaphore 依赖链，并在原 fused kernel 内对齐 contributor-relative ticket 与 reverse scheduler；代表 shape 完整 backward 从6.755毫秒降至1.699毫秒，提升3.98倍，并通过3601项回归和1000次逐位确定性测试。",
        "推理侧，我在 vLLM 中工程化接入确定性 Router GEMM 的多后端选择和 CUDA Graph preflight，使48层 Router GEMM中位耗时下降73.39%；同时完成 OE 异步状态算子，将历史 token 构造保留在GPU，TP1吞吐较同步路径提升4.99%，TP2/TP4均达到0 mismatch。",
        "RL 侧，我打通 R3 Router Replay 的采集、回放和指标闭环，在8张H200上将 route mismatch 从17%-19%降到0，并扩展到128卡运行；另针对 TP=2 FULL CUDA Graph rollout hang 构建两卡最小复现，定位 FlashInfer 中 FTZ 导致 Lamport sentinel 误判，回移位级修复后 Graph replay 不再出现 hang 或 timeout。",
        "此前在摩尔线程，我负责 TensorFlow MUSA Extension 的算子、图优化和稳定性工作：实现并修复 GELU 融合，使整网11个 GELU 全部融合、真实 shape 耗时降低36.6%；也修复了 shape tensor HostMemory 路径导致的长跑随机崩溃。",
        "这些经历让我形成了从最小复现、源码和 Kernel 定位，到正确性回归、性能剖析和多卡验证的工作方法。我希望继续从事 AI Infra、GPU 算子和大模型训练推理系统相关工作。谢谢。",
    ]
    # Reuse the original first seven paragraphs and remove the remaining blank paragraphs.
    for idx, text in enumerate(content):
        p = d.paragraphs[idx]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        compact_paragraph(p, space_after=7, line_spacing=1.2)
        run = p.add_run(text)
        set_run_font(run, latin=EN_FONT, east_asia=CN_BODY, size=11)
    for p in d.paragraphs[len(content):]:
        p._element.getparent().remove(p._element)
    d.save(output)


def convert_to_pdf(docx_path: Path, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # `svp` is the LibreOffice headless VCL backend available on this host;
    # `gen` attempts to open an X display even with --headless.
    env.setdefault("SAL_USE_VCLPLUGIN", "svp")
    pdf_path = outdir / f"{docx_path.stem}.pdf"
    pdf_path.unlink(missing_ok=True)
    proc = subprocess.run(
        ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", str(outdir), str(docx_path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not pdf_path.exists():
        raise RuntimeError(f"PDF conversion failed for {docx_path.name}: {proc.stdout}\n{proc.stderr}")
    return pdf_path


def assert_two_pages(pdf_path: Path):
    reader = PdfReader(pdf_path)
    if len(reader.pages) != 2:
        raise RuntimeError(f"{pdf_path.name} has {len(reader.pages)} pages, expected exactly 2")


def assert_focus_specific_resume(pdf_path: Path, *, focus: str):
    """Guard against a focused version silently drifting back to a general resume."""
    text = "\n".join(page.extract_text() or "" for page in PdfReader(pdf_path).pages)
    if focus == "training":
        required = ("FA3 deterministic", "orth-loss", "Blackwell 多精度训练验证")
        excluded = ("OE 异步状态算子", "H200 量化 Runtime", "Router GEMM")
    elif focus == "inference":
        required = ("Router GEMM", "OE 异步状态算子", "H200 量化 Runtime")
        excluded = ("FA3 deterministic", "orth-loss", "Blackwell 多精度训练验证")
    else:
        raise ValueError(f"Unsupported resume focus: {focus}")
    missing = [token for token in required if token not in text]
    unexpected = [token for token in excluded if token in text]
    if missing or unexpected:
        raise RuntimeError(
            f"{pdf_path.name} is not a true {focus}-focused resume; "
            f"missing={missing}, unexpected={unexpected}"
        )


def docx_external_links(docx_path: Path) -> set[str]:
    """Return external targets recorded in Word relationship parts."""
    targets: set[str] = set()
    with ZipFile(docx_path) as archive:
        for name in archive.namelist():
            if not name.startswith("word/") or not name.endswith(".rels"):
                continue
            root = ElementTree.fromstring(archive.read(name))
            for relation in root:
                if relation.attrib.get("TargetMode") == "External" and relation.attrib.get("Target"):
                    targets.add(relation.attrib["Target"])
    return targets


def pdf_external_links(pdf_path: Path) -> set[str]:
    """Return URI actions embedded in a PDF, including table-cell hyperlinks."""
    targets: set[str] = set()
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        for reference in page.get("/Annots", []):
            annotation = reference.get_object()
            action = annotation.get("/A")
            if action and action.get("/URI"):
                targets.add(str(action["/URI"]))
    return targets


def assert_required_link(docx_path: Path, pdf_path: Path, url: str):
    normalized = url.rstrip("/")
    for artifact, targets in ((docx_path, docx_external_links(docx_path)), (pdf_path, pdf_external_links(pdf_path))):
        if normalized not in {target.rstrip("/") for target in targets}:
            raise RuntimeError(f"Missing hyperlink {url} in {artifact.name}")


def pdf_word_boxes(pdf_path: Path):
    """Read Poppler word boxes as (page index, page height, text, box)."""
    proc = subprocess.run(
        ["pdftotext", "-bbox", str(pdf_path), "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Could not extract text boxes from {pdf_path.name}: {proc.stderr}")
    root = ElementTree.fromstring(proc.stdout)
    results = []
    pages = [element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "page"]
    for page_index, page in enumerate(pages):
        page_height = float(page.attrib["height"])
        for word in (element for element in page.iter() if element.tag.rsplit("}", 1)[-1] == "word"):
            text = word.text or ""
            results.append(
                (
                    page_index,
                    page_height,
                    text,
                    (
                        float(word.attrib["xMin"]),
                        float(word.attrib["yMin"]),
                        float(word.attrib["xMax"]),
                        float(word.attrib["yMax"]),
                    ),
                )
            )
    return results


def add_pdf_website_link(pdf_path: Path):
    """Add transparent contact and portfolio links to a generated PDF.

    LibreOffice 7.3 omits table-cell hyperlink annotations during DOCX-to-PDF
    conversion. The first rectangle covers the centered Website field. The
    remaining rectangles recover the visible Mooncake PR group and Mirage PR
    by locating their Poppler text boxes after conversion.
    """
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    writer.append_pages_from_reader(reader)
    writer.add_annotation(0, Link(rect=(390, 663, 560, 683), border=[0, 0, 0], url=PERSONAL_WEBSITE))
    anchors = (
        ("#3601/#3604/#3660/#3726", MOONCAKE_PRS_URL),
        ("#755", MIRAGE_PR_URL),
    )
    found = {needle: False for needle, _ in anchors}
    for page_index, page_height, text, (x_min, y_min, x_max, y_max) in pdf_word_boxes(pdf_path):
        for needle, url in anchors:
            if needle in text and not found[needle]:
                # Poppler uses a top-left origin; PDF annotations use bottom-left.
                rect = (x_min - 1, page_height - y_max - 1, x_max + 1, page_height - y_min + 1)
                writer.add_annotation(page_index, Link(rect=rect, border=[0, 0, 0], url=url))
                found[needle] = True
    if not found["#3601/#3604/#3660/#3726"] or not found["#755"]:
        raise RuntimeError(f"Open-source link anchors missing in {pdf_path.name}")
    metadata = {str(k): str(v) for k, v in (reader.metadata or {}).items() if k and v}
    writer.add_metadata(metadata)
    temporary = pdf_path.with_suffix(".link.tmp.pdf")
    with temporary.open("wb") as handle:
        writer.write(handle)
    os.replace(temporary, pdf_path)


def watermark_pdf_in_place(pdf_path: Path, text: str) -> bool:
    reader = PdfReader(pdf_path)
    if reader.metadata and reader.metadata.get("/XiaohongshuWatermarkVersion") == WATERMARK_VERSION:
        return False

    # A rasterized CJK overlay avoids PDF CMap compatibility problems when a
    # CID font is merged into third-party PDFs.  The embedded alpha PNG renders
    # consistently in Chrome, Acrobat, and Poppler.
    font = ImageFont.truetype(WATERMARK_FONT, 72, index=0)
    bbox = font.getbbox(text)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    image = Image.new("RGBA", (text_width + 80, text_height + 80), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    draw.text((40 - bbox[0], 40 - bbox[1]), text, font=font, fill=(166, 28, 28, 62))
    image = image.rotate(32, expand=True, resample=Image.Resampling.BICUBIC)
    png = io.BytesIO()
    image.save(png, format="PNG")
    png.seek(0)
    overlay_image = ImageReader(png)

    writer = PdfWriter()
    for source_page in reader.pages:
        width = float(source_page.mediabox.width)
        height = float(source_page.mediabox.height)
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=(width, height))
        display_width = image.width / 2
        display_height = image.height / 2
        c.drawImage(
            overlay_image,
            width / 2 - display_width / 2,
            height / 2 - display_height / 2,
            width=display_width,
            height=display_height,
            mask="auto",
        )
        c.save()
        overlay = PdfReader(io.BytesIO(buf.getvalue())).pages[0]
        source_page.merge_page(overlay)
        writer.add_page(source_page)
    metadata = {str(k): str(v) for k, v in (reader.metadata or {}).items() if k and v}
    metadata["/XiaohongshuWatermark"] = text
    metadata["/XiaohongshuWatermarkVersion"] = WATERMARK_VERSION
    writer.add_metadata(metadata)
    temp_path = pdf_path.with_suffix(".watermark.tmp.pdf")
    with temp_path.open("wb") as handle:
        writer.write(handle)
    os.replace(temp_path, pdf_path)
    return True


def main():
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    temp_chinese_docx = TMP_ROOT / CH_DOCX.name
    temp_training_docx = TMP_ROOT / TRAIN_DOCX.name
    temp_inference_docx = TMP_ROOT / INFER_DOCX.name
    temp_anonymous_docx = TMP_ROOT / ANON_DOCX.name
    temp_english_docx = TMP_ROOT / EN_DOCX.name
    update_chinese_resume(CH_DOCX, temp_chinese_docx)
    update_chinese_resume_variant(CH_DOCX, temp_training_docx, focus="training")
    update_chinese_resume_variant(CH_DOCX, temp_inference_docx, focus="inference")
    update_anonymous_chinese_resume(CH_DOCX, temp_anonymous_docx, word_watermark=False)
    update_english_resume(EN_DOCX, temp_english_docx)

    temp_chinese_pdf = convert_to_pdf(temp_chinese_docx, TMP_ROOT)
    temp_training_pdf = convert_to_pdf(temp_training_docx, TMP_ROOT)
    temp_inference_pdf = convert_to_pdf(temp_inference_docx, TMP_ROOT)
    temp_anonymous_pdf = convert_to_pdf(temp_anonymous_docx, TMP_ROOT)
    temp_english_pdf = convert_to_pdf(temp_english_docx, TMP_ROOT)
    for pdf_path in (temp_chinese_pdf, temp_training_pdf, temp_inference_pdf, temp_english_pdf):
        add_pdf_website_link(pdf_path)
    for pdf_path in (temp_chinese_pdf, temp_training_pdf, temp_inference_pdf, temp_anonymous_pdf, temp_english_pdf):
        assert_two_pages(pdf_path)
    assert_focus_specific_resume(temp_training_pdf, focus="training")
    assert_focus_specific_resume(temp_inference_pdf, focus="inference")
    for docx_path, pdf_path in (
        (temp_chinese_docx, temp_chinese_pdf),
        (temp_training_docx, temp_training_pdf),
        (temp_inference_docx, temp_inference_pdf),
        (temp_english_docx, temp_english_pdf),
    ):
        assert_required_link(docx_path, pdf_path, PERSONAL_WEBSITE)

    # The DOCX watermark is inserted after conversion; the PDF uses a raster
    # overlay for reliable CJK rendering in all common viewers.
    anonymous_doc = Document(temp_anonymous_docx)
    add_word_watermark(anonymous_doc, WATERMARK_TEXT)
    anonymous_doc.save(temp_anonymous_docx)
    watermark_pdf_in_place(temp_anonymous_pdf, WATERMARK_TEXT)

    for source, destination in (
        (temp_chinese_docx, CH_DOCX),
        (temp_chinese_pdf, CH_PDF),
        (temp_training_docx, TRAIN_DOCX),
        (temp_training_pdf, TRAIN_PDF),
        (temp_inference_docx, INFER_DOCX),
        (temp_inference_pdf, INFER_PDF),
        (temp_anonymous_docx, ANON_DOCX),
        (temp_anonymous_pdf, ANON_PDF),
        (temp_english_docx, EN_DOCX),
        (temp_english_pdf, EN_PDF),
    ):
        os.replace(source, destination)
    print("updated linked Chinese/English resumes and focused Chinese variants")


if __name__ == "__main__":
    main()
