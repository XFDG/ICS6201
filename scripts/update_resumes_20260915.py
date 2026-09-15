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

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.shared import Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


ROOT = Path("/volume/pt-train/users/zhaoye/ICS6201")
DOC_DIR = ROOT / "doc"
TMP_ROOT = ROOT / "tmp" / "pdfs" / "resume_update_20260915"

CH_DOCX = DOC_DIR / "冯浩然_AiInfra香港中文大学_15024999885.docx"
CH_PDF = DOC_DIR / "冯浩然_AiInfra香港中文大学_15024999885.pdf"
EN_DOCX = DOC_DIR / "Haoran_Feng_AIInfra_Resume.docx"
EN_PDF = DOC_DIR / "Haoran_Feng_AIInfra_Resume.pdf"
INTERVIEW_DOCX = DOC_DIR / "面试介绍精简版.docx"
WATERMARK_DIR = ROOT / "小红书商品" / "面试"
WATERMARK_TEXT = "小红书小冯别放弃"
WATERMARK_VERSION = "raster-v2"
WATERMARK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

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


def add_group(cell, title: str, *, latin: str, east_asia: str, body_size: float):
    p = next_content_paragraph(cell)
    compact_paragraph(p, space_after=0.25, line_spacing=1.0, left=7, first=-7)
    add_run(p, "• ", latin=latin, east_asia=east_asia, size=body_size, bold=True)
    add_run(p, title, latin=latin, east_asia=east_asia, size=body_size, bold=True)


def add_numbered(cell, number: int, title: str, text: str, *, latin: str, east_asia: str, body_size: float):
    p = next_content_paragraph(cell)
    compact_paragraph(p, space_after=0.65, line_spacing=1.0, left=13, first=-11)
    add_run(p, f"{number}. ", latin=latin, east_asia=east_asia, size=body_size)
    add_run(p, title, latin=latin, east_asia=east_asia, size=body_size, bold=True)
    add_run(p, text, latin=latin, east_asia=east_asia, size=body_size)


def add_bullet(cell, title: str, text: str, *, latin: str, east_asia: str, body_size: float, space_after=0.65):
    p = next_content_paragraph(cell)
    compact_paragraph(p, space_after=space_after, line_spacing=1.0, left=7, first=-7)
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
    set_cell(t0.cell(1, 0), "电话：(+86)15024999885   邮箱：225010160@link.cuhk.edu.cn   个人网站", latin=CN_BODY, east_asia=CN_BODY, size=10, alignment=WD_ALIGN_PARAGRAPH.CENTER)
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
    body_size = 8.25
    add_group(cell, "训练算子支持", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    add_numbered(cell, 1, "FA3 deterministic SWA backward 的 dQ 依赖链调度优化：", "针对长序列、变长滑窗 Attention 的 deterministic backward 回退，负责问题归因、调度实现与验证；通过 Nsys/NCU 和 dQ/dK/dV 因子隔离，确认 dQ semaphore 依赖链占确定性增量99.67%。在原 fused main kernel 中将绝对 ticket 改为 contributor-relative ticket，并使 reverse scheduler 与归约顺序对齐；复用既有长度元数据分流，短序列保留 fallback，不新增 kernel/workspace/D2H。代表 packed shape 完整 backward 由6.755 ms降至1.699 ms（3.98×），2K-12K加速1.59-5.37×；1000次自身bitwise、3601/3601回归通过并交付 wheel。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    add_numbered(cell, 2, "Sink FC1 GEMM 与通信竞争控核探究：", "训练 trace 显示跨 stream SendRecv/AllGatherV 使 FC1 额外退化15.48%/28.73%；纠正实际 BF16 shape 为8192×3072×3072，确认约200 μs基线正常。搭建4×H200双 stream 代理，以 NCCL CTA 预算控制通信并行度、DeepGEMM SM budget 控制计算资源，并用 Nsys 校验实际 grid 与重叠；完成121组粗扫、89组细扫及复验。AllGatherV代理中12 CTA + DG120将联合 span 降低12.80%，同通信设置下控核独立贡献7.58%；AllToAllV主要收益来自通信并行度，不归因为 GEMM。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    add_group(cell, "推理算子支持", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    add_numbered(cell, 1, "CUDA Graph 下的确定性 MoE Router GEMM：", "针对 batch-invariant decode 小M下 persistent tile 无效计算及 CUDA Graph 确定性约束，负责在 vLLM 工程化接入 DeepGEMM/Triton Full-K/persistent三级后端；设计 tensor-signature selector、capture前数值与Graph preflight、cache/fallback、workspace生命周期和路径回归，replay固化后端、不在图内动态选核。48层 Router GEMM中位耗时下降73.39%，135/135 kernel、20/20模型场景逐位一致；prefix-cache hit下TP1/TP2整请求处理性能提升3.81%/4.36%。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    add_numbered(cell, 2, "OE 异步状态算子：", "将 recent-token history 与oe_input_ids构造保留在GPU，以Triton fused-hash消除TP1同步与回传，并修复prefill/decode/mixed batch/请求恢复/slot reuse/reorder的跨step状态；TP1严格28/28 case通过、吞吐较sync +4.99%。TP>1采用async-unfused + batch-invariant安全路径后，TP2/TP4均达84/84、0 mismatch，decode吞吐+4.6%/+3.0%。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    add_group(cell, "RL 训推一致性与 Rollout 稳定性", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    add_numbered(cell, 1, "R3 Router Replay 与多卡可观测性：", "针对rollout侧vLLM与训练侧Megatron的MoE路由偏差，打通[token, MoE-layer, top-k] route采集、传输与训练侧回放，建立response-mask对齐、异常检测及route mismatch/fτ²/KL闭环；8×H200、Qwen3-30B-A3B BF16的20-step对照中，route mismatch由17%-19%降至0，fτ²/KL分别降低36-145×/4-7×。进一步扩展至128卡集群完成内部大模型竞赛题单轮200-step rollout并接入SwanLab监控，关键指标与8卡对照一致。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    add_numbered(cell, 2, "FlashInfer CUDA Graph hang 排查与修复：", "针对H200、vLLM TP=2 FULL CUDA Graph rollout卡死，构建两卡最小复现，经TP1/TP2、eager/FULL Graph、fused/unfused等控制变量将根因收敛至fused AllReduce + RMSNorm中Lamport -0.0 sentinel的FTZ误判；回移上游0x80000000位级判断修复并完成模型级Graph on/off回归，重复Graph replay未再出现hang或timeout。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    p = cell.add_paragraph()
    compact_paragraph(p, space_after=0.55, line_spacing=1.0, left=0, first=0)
    add_run(p, "2026.01-2026.04", latin=CN_HEADING, east_asia=CN_HEADING, size=8.75, bold=True)
    add_run(p, "                         摩尔线程                         算子与编译器优化实习生", latin=CN_HEADING, east_asia=CN_HEADING, size=8.75, bold=True)
    add_bullet(cell, "TensorFlow MUSA Extension 算子、图优化与稳定性：", "负责muDNN GELU接入、GELU fusion链路修复、benchmark和热点算子优化，推动整网11个GELU全部融合，真实shape耗时降低36.6%；独立定位shape tensor误入device path并重构HostMemory，使inference 500轮成功率约30%提升至1000轮100%，4万/40万/80万轮长跑稳定；Logical_Or由21.2 μs降至10.7 μs，整网吞吐8187.48提升至8284.65。", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size, space_after=0.1)

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
    add_bullet(cell, "量化 Runtime、Kernel 排障与TP2部署：", "在W4A16 AWQ、W8A8 SmoothQuant、FP8 Runtime中完成FlashAttention-2/4可配置接入、GQA KV-head展开、softcap mask、SDPA回退及跨GPU动态构建；以Compute Sanitizer定位gemv_kernel_g128尾组越界，补充边界保护并实现Qwen2 packed-AWQ TP=2列/行切分与NCCL all-reduce。代表shape四类Sanitizer检查均为0 error/0 hazard；Qwen2.5-Coder-32B W4A16双卡部署使checkpoint显存-70.5%、峰值显存-66.8%、端到端输出吞吐+76.8%，并通过数值、跨rank token与交互验收。", latin=EN_FONT, east_asia=CN_BODY, body_size=8.35, space_after=0.1)
    set_cell(internship_rows[9].cells[0], "2026.01-至今", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    set_cell(internship_rows[9].cells[1], "", latin=CN_HEADING, east_asia=CN_HEADING, size=9, bold=True)
    p = set_cell(internship_rows[9].cells[2], "关键 Token 加权的思维链蒸馏｜AAAI 2027 已投稿｜", latin=CN_HEADING, east_asia=CN_HEADING, size=8.7, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
    add_hyperlink(p, "项目代码", "https://github.com/jokerhan01/cot-main", latin=CN_HEADING, east_asia=CN_HEADING, size=8.7)
    cell = body_cell(d, internship_rows[10].cells[0])
    add_bullet(cell, "方法与结果：", "参与“结构恢复—关键Token加权监督—偏好优化”三阶段框架，负责模型训练与调优；以逐Token扰动教师推理导致的参考答案生成似然下降量估计重要性，并用于加权SFT与辅助损失。完成LoRA/DPO训练和vLLM TP=4评测；Qwen2.5-7B-Instruct在GSM8K/SVAMP上取得94.01%/94.00% accuracy，较最强基线提升5.51/10.10个百分点，论文已投稿AAAI 2027。", latin=EN_FONT, east_asia=CN_BODY, body_size=8.35, space_after=0.1)

    set_section_cell(t2.cell(0, 0), "综合素质")
    cell = body_cell(d, t2.cell(1, 0))
    add_bullet(cell, "技术方向：", "关注GPU高性能算子开发与大模型训练/推理系统优化，具备Attention、MoE Router、量化GEMM、异步状态管理、Tensor Parallel与CUDA Graph工程实践。", latin=EN_FONT, east_asia=CN_BODY, body_size=8.5, space_after=0.25)
    add_bullet(cell, "开发与性能工程：", "熟练使用C/C++、Python、CUDA、Triton、PyTorch Extension；能够使用Nsys、NCU、Compute Sanitizer完成算子性能分析与正确性验证。", latin=EN_FONT, east_asia=CN_BODY, body_size=8.5, space_after=0.25)
    add_bullet(cell, "语言与证书：", "IELTS 6.5、CET-6，持有华为HCIA-AI认证；具备英文技术文档阅读、检索与跨仓源码分析能力。", latin=EN_FONT, east_asia=CN_BODY, body_size=8.5, space_after=0)

    move_tail_rows_to_next_page(t1, 6)
    d.save(output)


def update_english_resume(source: Path, output: Path):
    shutil.copy2(source, output)
    d = Document(output)
    configure_resume_metadata(d, title="Haoran Feng - AI Infrastructure Resume")
    t0, t1, t2 = resume_template_tables(d)

    set_cell(t0.cell(0, 0), "Haoran Feng", latin=EN_FONT, east_asia=CN_BODY, size=26, alignment=WD_ALIGN_PARAGRAPH.CENTER)
    set_cell(t0.cell(1, 0), "Phone: (+86) 150 2499 9885    Email: 225010160@link.cuhk.edu.cn    Portfolio", latin=EN_FONT, east_asia=CN_BODY, size=10, alignment=WD_ALIGN_PARAGRAPH.CENTER)
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
    add_group(cell, "Training Kernel Support", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    add_numbered(cell, 1, "FA3 deterministic SWA backward: ", "Used Nsys/NCU factorization to attribute 99.67% of deterministic overhead to the dQ semaphore chain. Implemented contributor-relative tickets aligned with the reverse scheduler in the fused kernel, with no new kernel/workspace/D2H and a short-sequence fallback. Cut packed backward from 6.755 to 1.699 ms (3.98x), achieved 1.59-5.37x at 2K-12K, and passed 3,601/3,601 regressions plus 1,000 bitwise repeats.", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    add_numbered(cell, 2, "Sink FC1 GEMM and communication contention: ", "Corrected FC1 to BF16 8192x3072x3072 and isolated cross-stream communication contention rather than a slow GEMM. Built a 4xH200 dual-stream proxy, sweeping NCCL CTA budgets and DeepGEMM SM budgets (121 coarse, 89 fine) with Nsys grid/overlap checks. In the AllGather proxy, 12 CTA + DG120 reduced joint span by 12.80%, including 7.58% independent gain from compute throttling; AllToAll gains were attributed to communication parallelism.", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    add_group(cell, "Inference Kernel Support", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    add_numbered(cell, 1, "Deterministic MoE Router GEMM under CUDA Graph: ", "Integrated DeepGEMM, Triton Full-K, and persistent backends in vLLM; owned the tensor-signature selector, pre-capture numeric/graph preflight, cache/fallback, workspace lifecycle, and path tests. Replay freezes the backend decision. Reduced median latency of 48 router GEMMs by 73.39%, passed 135/135 kernel and 20/20 model bitwise checks, and improved prefix-cache-hit request throughput by 3.81%/4.36% on TP1/TP2.", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    add_numbered(cell, 2, "Asynchronous OE state operator: ", "Kept recent-token history and oe_input_ids construction on GPU with Triton fused hashing, fixing cross-step state for prefill/decode/mixed batches/recovery/slot reuse/reorder. Passed 28/28 TP1 cases with +4.99% throughput; the TP>1 async-unfused batch-invariant path reached 84/84 with zero mismatches and +4.6%/+3.0% decode throughput on TP2/TP4.", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    add_group(cell, "RL Training-Inference Consistency and Rollout Reliability", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
    add_numbered(cell, 1, "R3 Router Replay and multi-GPU observability: ", "Built token/layer/top-k route capture, transport, training-side replay, response-mask alignment, and mismatch/f-tau2/KL diagnostics between vLLM and Megatron. On 8xH200/Qwen3-30B-A3B, reduced route mismatch from 17-19% to 0, f-tau2 by 36-145x, and KL by 4-7x; extended to a monitored 128-GPU, 200-step rollout with aligned key metrics.", latin=EN_FONT, east_asia=CN_BODY, body_size=body_size)
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
    add_bullet(cell, "Quantized runtime, kernel debugging, and TP2 deployment: ", "Added configurable FlashAttention-2/4, GQA KV-head expansion, soft-cap masking, SDPA fallback, and multi-GPU builds to a W4A16/W8A8/FP8 runtime. Used Compute Sanitizer to fix a tail-group out-of-bounds access and implemented packed-AWQ TP=2 sharding with NCCL all-reduce. All four Sanitizer modes passed representative shapes; two-GPU Qwen2.5-Coder-32B W4A16 reduced checkpoint/peak memory by 70.5%/66.8% and increased end-to-end output throughput by 76.8%.", latin=EN_FONT, east_asia=CN_BODY, body_size=8.0, space_after=0.1)
    set_cell(internship_rows[9].cells[0], "Jan 2026 - Present", latin=EN_FONT, east_asia=CN_BODY, size=9, bold=True)
    set_cell(internship_rows[9].cells[1], "", latin=EN_FONT, east_asia=CN_BODY, size=9, bold=True)
    p = set_cell(internship_rows[9].cells[2], "Token-Weighted CoT Distillation | AAAI 2027 Submission | ", latin=EN_FONT, east_asia=CN_BODY, size=8.5, bold=True, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
    add_hyperlink(p, "Code", "https://github.com/jokerhan01/cot-main", latin=EN_FONT, east_asia=CN_BODY, size=8.5)
    cell = body_cell(d, internship_rows[10].cells[0])
    add_bullet(cell, "Method and results: ", "Trained and tuned a three-stage structure-recovery, token-weighted supervision, and preference-optimization pipeline; token importance was estimated from reference-answer likelihood drops after teacher-rationale perturbations. Completed LoRA/DPO training and vLLM TP=4 evaluation. Qwen2.5-7B-Instruct reached 94.01%/94.00% on GSM8K/SVAMP, +5.51/+10.10 points over the strongest baselines; manuscript submitted to AAAI 2027.", latin=EN_FONT, east_asia=CN_BODY, body_size=8.0, space_after=0.1)

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
    proc = subprocess.run(
        ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", str(outdir), str(docx_path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    pdf_path = outdir / f"{docx_path.stem}.pdf"
    if proc.returncode != 0 or not pdf_path.exists():
        raise RuntimeError(f"PDF conversion failed for {docx_path.name}: {proc.stdout}\n{proc.stderr}")
    return pdf_path


def assert_two_pages(pdf_path: Path):
    reader = PdfReader(pdf_path)
    if len(reader.pages) != 2:
        raise RuntimeError(f"{pdf_path.name} has {len(reader.pages)} pages, expected exactly 2")


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
    temp_ch_docx = TMP_ROOT / CH_DOCX.name
    temp_en_docx = TMP_ROOT / EN_DOCX.name
    temp_intro_docx = TMP_ROOT / INTERVIEW_DOCX.name
    update_chinese_resume(CH_DOCX, temp_ch_docx)
    update_english_resume(EN_DOCX, temp_en_docx)
    update_interview_intro(INTERVIEW_DOCX, temp_intro_docx)
    temp_ch_pdf = convert_to_pdf(temp_ch_docx, TMP_ROOT)
    temp_en_pdf = convert_to_pdf(temp_en_docx, TMP_ROOT)
    assert_two_pages(temp_ch_pdf)
    assert_two_pages(temp_en_pdf)

    # Install only after both document/PDF pairs pass the page-count gate.
    os.replace(temp_ch_docx, CH_DOCX)
    os.replace(temp_en_docx, EN_DOCX)
    os.replace(temp_intro_docx, INTERVIEW_DOCX)
    os.replace(temp_ch_pdf, CH_PDF)
    os.replace(temp_en_pdf, EN_PDF)

    changed = 0
    for pdf_path in sorted(WATERMARK_DIR.glob("*.pdf")):
        changed += int(watermark_pdf_in_place(pdf_path, WATERMARK_TEXT))
    print(f"updated resumes; watermarked {changed} PDF files")


if __name__ == "__main__":
    main()
