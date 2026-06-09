"""Generate ICS6201 project environment report PPT."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

# ── Colors ──
C_BG       = RGBColor(0x0D, 0x1B, 0x2A)   # dark navy
C_ACCENT   = RGBColor(0x1B, 0x99, 0x8B)   # teal
C_WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
C_LIGHT    = RGBColor(0xE0, 0xE0, 0xE0)
C_GRAY     = RGBColor(0xA0, 0xA0, 0xA0)
C_ORANGE   = RGBColor(0xF4, 0x84, 0x3F)
C_BLUE     = RGBColor(0x4D, 0xA8, 0xDA)
C_GREEN    = RGBColor(0x4E, 0xC9, 0x8B)
C_YELLOW   = RGBColor(0xF0, 0xC7, 0x41)
C_RED      = RGBColor(0xE8, 0x5D, 0x5D)

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
W = prs.slide_width
H = prs.slide_height

# ── Helpers ──
def add_bg(slide, color=C_BG):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_rect(slide, left, top, width, height, color, alpha=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    if alpha is not None:
        from lxml import etree
        spPr = shape._element.spPr
        solidFill = spPr.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}solidFill')
        if solidFill is not None:
            srgb = solidFill.find('{http://schemas.openxmlformats.org/drawingml/2006/main}srgbClr')
            if srgb is not None:
                alpha_el = etree.SubElement(srgb, '{http://schemas.openxmlformats.org/drawingml/2006/main}alpha')
                alpha_el.set('val', str(int(alpha * 1000)))
    return shape

def add_text_box(slide, left, top, width, height, text, font_size=18,
                 color=C_WHITE, bold=False, alignment=PP_ALIGN.LEFT, font_name='Microsoft YaHei'):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox

def add_accent_line(slide, left, top, width, color=C_ACCENT):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, Pt(4))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()

def add_table(slide, left, top, width, rows, cols, data, col_widths=None):
    table_shape = slide.shapes.add_table(rows, cols, left, top, width, Inches(0.4 * rows))
    table = table_shape.table
    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = w
    for r in range(rows):
        for c in range(cols):
            cell = table.cell(r, c)
            cell.text = str(data[r][c])
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(13)
                p.font.name = 'Microsoft YaHei'
                if r == 0:
                    p.font.bold = True
                    p.font.color.rgb = C_WHITE
                else:
                    p.font.color.rgb = C_LIGHT
                p.alignment = PP_ALIGN.CENTER
            # cell fill
            from lxml import etree
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            solidFill = etree.SubElement(tcPr, '{http://schemas.openxmlformats.org/drawingml/2006/main}solidFill')
            if r == 0:
                etree.SubElement(solidFill, '{http://schemas.openxmlformats.org/drawingml/2006/main}srgbClr', val='1B998B')
            else:
                etree.SubElement(solidFill, '{http://schemas.openxmlformats.org/drawingml/2006/main}srgbClr', val='152030')
            # borders
            for edge in ['lnL', 'lnR', 'lnT', 'lnB']:
                ln = etree.SubElement(tcPr, '{http://schemas.openxmlformats.org/drawingml/2006/main}' + edge, w='6350')
                sf = etree.SubElement(ln, '{http://schemas.openxmlformats.org/drawingml/2006/main}solidFill')
                etree.SubElement(sf, '{http://schemas.openxmlformats.org/drawingml/2006/main}srgbClr', val='2A3A4A')
    return table_shape

def add_bullet_list(slide, left, top, width, height, items, font_size=15, color=C_LIGHT):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = 'Microsoft YaHei'
        p.space_after = Pt(6)
        p.level = 0
    return txBox

# ============================================================
# SLIDE 1 — Title
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_rect(slide, Inches(0), Inches(0), W, H, RGBColor(0x0A, 0x14, 0x22))
# accent stripe
add_rect(slide, Inches(0), Inches(2.8), Inches(0.15), Inches(1.8), C_ACCENT)

add_text_box(slide, Inches(1), Inches(1.8), Inches(11), Inches(1),
             'ICS6201', font_size=28, color=C_ACCENT, bold=True)
add_text_box(slide, Inches(1), Inches(2.6), Inches(11), Inches(1.2),
             'UAV Visual Detection', font_size=48, color=C_WHITE, bold=True)
add_text_box(slide, Inches(1), Inches(3.6), Inches(11), Inches(0.8),
             'Environment Setup & Training Progress Report', font_size=24, color=C_LIGHT)
add_accent_line(slide, Inches(1), Inches(4.6), Inches(3))
add_text_box(slide, Inches(1), Inches(5.0), Inches(6), Inches(0.5),
             'Date: 2026-06-09  |  Server: 8x NVIDIA H200 (141GB)', font_size=14, color=C_GRAY)

# ============================================================
# SLIDE 2 — Project Overview
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(6), Inches(0.7),
             'Project Overview', font_size=36, color=C_WHITE, bold=True)
add_accent_line(slide, Inches(0.8), Inches(1.05), Inches(2.5))

# Left: objective
add_rect(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(5.2), RGBColor(0x12, 0x22, 0x35))
add_text_box(slide, Inches(1.1), Inches(1.7), Inches(5), Inches(0.5),
             'Objective', font_size=20, color=C_ACCENT, bold=True)
add_bullet_list(slide, Inches(1.1), Inches(2.2), Inches(5), Inches(4), [
    'Task: Single-class (drone) object detection',
    'Compare 6 detection models across 3 UAV datasets',
    'Each model x 3 random seeds = 18 training runs',
    'Find the best solution for UAV illegal flight detection',
], font_size=16)

# Right: quick stats
add_rect(slide, Inches(7), Inches(1.5), Inches(5.5), Inches(5.2), RGBColor(0x12, 0x22, 0x35))
stats = [
    ('171,568', 'Total Images'),
    ('6', 'Models'),
    ('18', 'Training Runs'),
    ('8x H200', 'GPU Cluster'),
]
for i, (val, label) in enumerate(stats):
    y = Inches(1.8 + i * 1.15)
    add_text_box(slide, Inches(7.4), y, Inches(2.5), Inches(0.6),
                 val, font_size=32, color=C_ACCENT, bold=True)
    add_text_box(slide, Inches(10), y + Inches(0.05), Inches(2.2), Inches(0.5),
                 label, font_size=16, color=C_LIGHT)

# ============================================================
# SLIDE 3 — Datasets
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(6), Inches(0.7),
             'Datasets', font_size=36, color=C_WHITE, bold=True)
add_accent_line(slide, Inches(0.8), Inches(1.05), Inches(2.5))

data = [
    ['Dataset', 'Format', 'Train', 'Val', 'Test', 'Total'],
    ['DUT Anti-UAV', 'VOC XML', '5,200', '2,600', '2,200', '10,000'],
    ['DroneDetection', 'VOC XML', '46,301', '5,145', '2,625', '54,071'],
    ['ARD-MAV', 'Video+XML', '85,997', '10,749', '10,751', '107,497'],
    ['Total', 'YOLO', '137,498', '18,494', '15,576', '171,568'],
]
add_table(slide, Inches(0.8), Inches(1.5), Inches(11.7), 5, 6, data)

add_text_box(slide, Inches(0.8), Inches(4.2), Inches(11), Inches(0.5),
             'Data Pipeline', font_size=22, color=C_ACCENT, bold=True)
add_bullet_list(slide, Inches(0.8), Inches(4.8), Inches(11), Inches(2.5), [
    'VOC XML -> YOLO format via prepare_rgb_yolo.py',
    'ARD-MAV video fast frame extraction (107K frames in ~15 min, vs 18 days naive)',
    'YOLO -> COCO format for Detectron2 via prepare_rgb_coco.py',
], font_size=15)

# ============================================================
# SLIDE 4 — Models
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(6), Inches(0.7),
             'Models Under Comparison', font_size=36, color=C_WHITE, bold=True)
add_accent_line(slide, Inches(0.8), Inches(1.05), Inches(2.5))

models_data = [
    ['#', 'Model', 'Framework', 'Category', 'Status'],
    ['1', 'YOLOv8n', 'Ultralytics', 'YOLO family', 'Pending'],
    ['2', 'YOLOv10n', 'Ultralytics', 'YOLO family', 'Pending'],
    ['3', 'YOLO11m', 'Ultralytics', 'YOLO family', 'Partial'],
    ['4', 'RT-DETR-L', 'Ultralytics', 'Transformer', 'DONE'],
    ['5', 'Faster R-CNN R50-FPN', 'Detectron2', 'Two-stage', 'DONE'],
    ['6', 'DDW-YOLO (ECA+BiFPN)', 'Ultralytics', 'Custom YOLO', 'Pending'],
]
add_table(slide, Inches(0.8), Inches(1.5), Inches(11.7), 7, 5, models_data)

add_text_box(slide, Inches(0.8), Inches(5.2), Inches(11), Inches(0.5),
             'Each model trained with 3 random seeds (1, 2, 3) for reproducibility',
             font_size=15, color=C_GRAY)

# ============================================================
# SLIDE 5 — Environment Setup
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(8), Inches(0.7),
             'Environment Setup & Pitfalls', font_size=36, color=C_WHITE, bold=True)
add_accent_line(slide, Inches(0.8), Inches(1.05), Inches(2.5))

# Left: env
add_rect(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(5.2), RGBColor(0x12, 0x22, 0x35))
add_text_box(slide, Inches(1.1), Inches(1.7), Inches(5), Inches(0.5),
             'Conda Environment: fly', font_size=18, color=C_ACCENT, bold=True)
add_bullet_list(slide, Inches(1.1), Inches(2.3), Inches(5), Inches(4), [
    'Python 3.12 + PyTorch 2.5.1+cu121',
    'Ultralytics 8.4.56',
    'Detectron2 0.6',
    'opencv-python-headless (no libGL on GPU node)',
    'Installed on shared GPFS: /volume/yzhao04/',
], font_size=14)

# Right: pitfalls
add_rect(slide, Inches(7), Inches(1.5), Inches(5.5), Inches(5.2), RGBColor(0x12, 0x22, 0x35))
add_text_box(slide, Inches(7.3), Inches(1.7), Inches(5), Inches(0.5),
             'Key Pitfalls Solved', font_size=18, color=C_ORANGE, bold=True)
pitfalls = [
    ('libGL missing', ' -> opencv-python-headless'),
    ('pip constraint conflict', ' -> PIP_CONSTRAINT=""'),
    ('detectron2 build fail', ' -> --no-build-isolation'),
    ('Faster RCNN NaN loss', ' -> batch 4->32'),
    ('DDW-YOLO optimizer', ' -> AdamW (no Muon)'),
    ('SSH denied', ' -> ed25519 key via GPFS'),
]
for i, (prob, sol) in enumerate(pitfalls):
    y = Inches(2.3 + i * 0.65)
    add_text_box(slide, Inches(7.3), y, Inches(2.5), Inches(0.4),
                 prob, font_size=13, color=C_RED, bold=True)
    add_text_box(slide, Inches(9.8), y, Inches(2.5), Inches(0.4),
                 sol, font_size=13, color=C_GREEN)

# ============================================================
# SLIDE 6 — GPU Cluster Comparison
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(8), Inches(0.7),
             'GPU Cluster Comparison', font_size=36, color=C_WHITE, bold=True)
add_accent_line(slide, Inches(0.8), Inches(1.05), Inches(2.5))

gpu_data = [
    ['Cluster', 'GPU', 'Count', 'VRAM/GB', 'Total/GB', 'BW/TB/s', 'FP8'],
    ['Shanghai (ours)', 'H200', '8', '141', '1128', '38.4', 'Yes'],
    ['Beijing H100', 'H100', '4', '80', '320', '13.4', 'Yes'],
    ['A100 cluster', 'A100', '8', '80', '640', '16.0', 'No'],
    ['RTX 4090', '4090', '4', '24', '96', '4.0', 'No'],
]
add_table(slide, Inches(0.8), Inches(1.5), Inches(11.7), 5, 7, gpu_data)

add_text_box(slide, Inches(0.8), Inches(4.5), Inches(11.5), Inches(0.5),
             'Our platform: Shanghai 8x H200 = best fit for ICS6201 training',
             font_size=18, color=C_GREEN, bold=True)
add_bullet_list(slide, Inches(0.8), Inches(5.1), Inches(11), Inches(2), [
    '1128GB total VRAM — largest capacity, best for large batch & long context',
    '38.4 TB/s aggregate HBM bandwidth — ideal for memory-bound workloads',
    'Hopper FP8 Transformer Engine — supports next-gen quantized inference',
    'CPU node handles downloads, builds, docs; GPU node handles training only',
], font_size=14)

# ============================================================
# SLIDE 7 — Automation Scripts
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(8), Inches(0.7),
             'Automation & Script System', font_size=36, color=C_WHITE, bold=True)
add_accent_line(slide, Inches(0.8), Inches(1.05), Inches(2.5))

# master.sh flow
add_rect(slide, Inches(0.8), Inches(1.5), Inches(11.7), Inches(2.2), RGBColor(0x12, 0x22, 0x35))
add_text_box(slide, Inches(1.1), Inches(1.65), Inches(11), Inches(0.5),
             'master.sh Pipeline', font_size=20, color=C_ACCENT, bold=True)

steps = [
    ('1. Env Check', C_BLUE),
    ('2. Data Prep', C_GREEN),
    ('3. Smoke Test', C_YELLOW),
    ('4. Train (7 GPU)', C_ORANGE),
    ('5. Summary', C_ACCENT),
    ('6. Keep Alive', C_RED),
]
for i, (label, color) in enumerate(steps):
    x = Inches(1.2 + i * 1.9)
    add_rect(slide, x, Inches(2.3), Inches(1.6), Inches(0.7), color, alpha=80)
    add_text_box(slide, x, Inches(2.35), Inches(1.6), Inches(0.6),
                 label, font_size=12, color=C_WHITE, bold=True, alignment=PP_ALIGN.CENTER)

# Key scripts table
scripts_data = [
    ['Script', 'Function'],
    ['auto_config.py', 'Auto-detect GPU count, model, VRAM'],
    ['parallel_launcher.py', 'Multi-GPU parallel training scheduler'],
    ['validate_pipeline.py', 'Subset smoke test (env/data/train)'],
    ['summarize_results.py', 'Generate comparison report from logs'],
    ['prepare_ard_mav_fast.py', 'Fast batch frame extraction for ARD-MAV'],
    ['recovery_manifest.py', 'Identify complete/partial/pending tasks'],
]
add_table(slide, Inches(0.8), Inches(4.2), Inches(11.7), 7, 2, scripts_data)

# ============================================================
# SLIDE 8 — Validation Results
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(8), Inches(0.7),
             'Validation: 14/14 PASSED', font_size=36, color=C_GREEN, bold=True)
add_accent_line(slide, Inches(0.8), Inches(1.05), Inches(2.5))

val_data = [
    ['Check Item', 'Result', 'Time'],
    ['8 GPU detection', 'PASS', '5s'],
    ['GPU selection (1-7)', 'PASS', '0s'],
    ['Environment imports', 'PASS', '0s'],
    ['Data preparation', 'PASS', '0s'],
    ['YOLOv8n training', 'PASS', '39s'],
    ['YOLOv10n training', 'PASS', '38s'],
    ['YOLO11m training', 'PASS', '40s'],
    ['DDW-YOLO training', 'PASS', '44s'],
    ['RT-DETR-L training', 'PASS', '56s'],
    ['Faster R-CNN training', 'PASS', '4m05s'],
    ['Multi-GPU parallel', 'PASS', '0s'],
    ['Keep Alive daemon', 'PASS', '-'],
    ['End-to-end timing', 'PASS', '5m07s'],
]
add_table(slide, Inches(0.8), Inches(1.5), Inches(11.7), 14, 3, val_data)

# ============================================================
# SLIDE 9 — Training Results
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(8), Inches(0.7),
             'Training Results (2026-06-09)', font_size=36, color=C_WHITE, bold=True)
add_accent_line(slide, Inches(0.8), Inches(1.05), Inches(2.5))

# Core tasks
add_rect(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(5.2), RGBColor(0x12, 0x22, 0x35))
add_text_box(slide, Inches(1.1), Inches(1.65), Inches(5), Inches(0.5),
             'Core Tasks (All DONE)', font_size=20, color=C_GREEN, bold=True)
core_data = [
    ['Task', 'mAP50-95', 'Status'],
    ['RT-DETR-L seed1', '0.6794', 'DONE'],
    ['RT-DETR-L seed2', '0.6753', 'DONE'],
    ['RT-DETR-L seed3', '0.6758', 'DONE'],
    ['Faster RCNN seed1', '-', 'DONE'],
    ['Faster RCNN seed2', '-', 'DONE'],
    ['Faster RCNN seed3', '-', 'DONE'],
]
add_table(slide, Inches(1.1), Inches(2.3), Inches(5), 7, 3, core_data)

# Secondary tasks
add_rect(slide, Inches(7), Inches(1.5), Inches(5.5), Inches(5.2), RGBColor(0x12, 0x22, 0x35))
add_text_box(slide, Inches(7.3), Inches(1.65), Inches(5), Inches(0.5),
             'Secondary Tasks', font_size=20, color=C_YELLOW, bold=True)
sec_data = [
    ['Task', 'Status'],
    ['YOLO11m seed1', 'DONE'],
    ['YOLO11m seed2', 'PARTIAL (OOM)'],
    ['YOLO11m seed3', 'PENDING'],
    ['DDW-YOLO seed1-3', 'PENDING'],
    ['YOLOv10 seed1-3', 'PENDING'],
    ['YOLOv8 seed1-3', 'PENDING'],
]
add_table(slide, Inches(7.3), Inches(2.3), Inches(5), 7, 2, sec_data)

# ============================================================
# SLIDE 10 — Summary & Next Steps
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)
add_text_box(slide, Inches(0.8), Inches(0.4), Inches(8), Inches(0.7),
             'Summary & Next Steps', font_size=36, color=C_WHITE, bold=True)
add_accent_line(slide, Inches(0.8), Inches(1.05), Inches(2.5))

# Summary
add_rect(slide, Inches(0.8), Inches(1.5), Inches(5.5), Inches(5.2), RGBColor(0x12, 0x22, 0x35))
add_text_box(slide, Inches(1.1), Inches(1.65), Inches(5), Inches(0.5),
             'Achievements', font_size=22, color=C_GREEN, bold=True)
add_bullet_list(slide, Inches(1.1), Inches(2.3), Inches(5), Inches(4), [
    'Environment fully operational on 8x H200',
    '171K images across 3 datasets prepared',
    'All 6 models validated (14/14 checks pass)',
    'Core tasks done: RT-DETR-L mAP50-95 ~0.677',
    'Core tasks done: Faster R-CNN x3 seeds',
    'Automation pipeline: master.sh one-click',
], font_size=15, color=C_GREEN)

# Next steps
add_rect(slide, Inches(7), Inches(1.5), Inches(5.5), Inches(5.2), RGBColor(0x12, 0x22, 0x35))
add_text_box(slide, Inches(7.3), Inches(1.65), Inches(5), Inches(0.5),
             'Next Steps', font_size=22, color=C_ORANGE, bold=True)
add_bullet_list(slide, Inches(7.3), Inches(2.3), Inches(5), Inches(4), [
    'Recover YOLO11m seed2 from last.pt',
    'Run remaining secondary tasks (YOLOv8, YOLOv10, DDW-YOLO)',
    'Generate full comparison report (18 runs)',
    'Analyze mAP50-95 across all models & seeds',
    'Write final experiment analysis paper',
], font_size=15, color=C_ORANGE)

# ── Save ──
out_path = '/home/ai/workspace/ICS6201/docs/ics6201/ICS6201_Environment_Report.pptx'
prs.save(out_path)
print(f'PPT saved to: {out_path}')
