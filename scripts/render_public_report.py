#!/usr/bin/env python3
"""Render the sanitized Markdown technical report to a styled PDF."""

from pathlib import Path
import html
import re

import markdown
from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "doc" / "open_doc" / "大模型训推算子优化的证据闭环_脱敏版_20260911.md"
OUTPUT = ROOT / "doc" / "open_doc" / "大模型训推算子优化的证据闭环_脱敏版_20260911.pdf"
HTML_DEBUG = ROOT / "tmp" / "pdfs" / "大模型训推算子优化的证据闭环_脱敏版_20260911.html"


CSS = r"""
@font-face {
  font-family: "Noto Sans CJK SC";
  src: url("file:///usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc");
  font-weight: 400;
}
@font-face {
  font-family: "Noto Serif CJK SC";
  src: url("file:///usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc");
  font-weight: 400;
}
@page {
  size: A4;
  margin: 17mm 17mm 19mm 18mm;
  @top-left {
    content: "大模型训推算子优化的证据闭环";
    font-family: "Noto Sans CJK SC", sans-serif;
    font-size: 8.5pt;
    color: #64748b;
  }
  @top-right {
    content: "脱敏分享版";
    font-family: "Noto Sans CJK SC", sans-serif;
    font-size: 8.5pt;
    color: #64748b;
  }
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font-family: "Noto Sans CJK SC", sans-serif;
    font-size: 8pt;
    color: #64748b;
  }
}
@page:first {
  @top-left { content: none; }
  @top-right { content: none; }
}
html { color: #172033; background: white; }
body {
  font-family: "Noto Serif CJK SC", "Songti SC", serif;
  font-size: 10.2pt;
  line-height: 1.58;
  letter-spacing: 0.01em;
}
h1, h2, h3, h4 {
  font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
  color: #102a43;
  break-after: avoid;
  page-break-after: avoid;
}
h1 {
  font-size: 25pt;
  line-height: 1.25;
  text-align: center;
  margin: 13mm 0 5mm;
  padding-bottom: 5mm;
  border-bottom: 2.3pt solid #0b6e99;
}
h2 {
  font-size: 16pt;
  margin: 8mm 0 3mm;
  padding-left: 3mm;
  border-left: 4pt solid #0b6e99;
}
h3 { font-size: 12.5pt; margin: 5mm 0 2mm; color: #0b5676; }
h4 { font-size: 11pt; margin: 4mm 0 1.5mm; }
p { margin: 0 0 2.6mm; text-align: justify; }
blockquote {
  margin: 3mm 0 7mm;
  padding: 3mm 5mm;
  background: #eff7fb;
  border-left: 3pt solid #0b6e99;
  color: #38566b;
  font-family: "Noto Sans CJK SC", sans-serif;
  font-size: 9.5pt;
}
blockquote p { margin: 0; text-align: left; }
ul, ol { margin: 1.5mm 0 3mm 6mm; padding-left: 4mm; }
li { margin: 1mm 0; }
strong { color: #0b4f6c; }
code {
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 8.7pt;
  color: #8b2942;
  background: #f5f7f9;
  padding: 0.1em 0.28em;
  border-radius: 2px;
  overflow-wrap: anywhere;
}
pre {
  margin: 2.5mm 0 4mm;
  padding: 3mm 4mm;
  background: #f4f7fa;
  border: 0.5pt solid #cbd5e1;
  border-radius: 3px;
  white-space: pre-wrap;
  break-inside: avoid;
}
pre code { background: transparent; padding: 0; color: #29384a; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 3mm 0 5mm;
  font-family: "Noto Sans CJK SC", sans-serif;
  font-size: 8.7pt;
  line-height: 1.42;
}
thead { display: table-header-group; }
tr { break-inside: avoid; }
th {
  background: #0b6e99;
  color: white;
  font-weight: 400;
  padding: 1.8mm 2mm;
  border: 0.5pt solid #8fb8ca;
  text-align: left;
}
td {
  padding: 1.55mm 2mm;
  border: 0.5pt solid #cbd5e1;
  vertical-align: top;
}
tbody tr:nth-child(even) { background: #f6f9fb; }
img {
  display: block;
  max-width: 100%;
  max-height: 168mm;
  margin: 4mm auto 5mm;
  break-inside: avoid;
}
.task-list-item { list-style: none; margin-left: -4mm; }
.task-list-item input { margin-right: 1.5mm; }
.cover-note {
  text-align: center;
  color: #587187;
  font-family: "Noto Sans CJK SC", sans-serif;
  margin-bottom: 9mm;
}
"""


def build_html(markdown_text: str) -> str:
    body = markdown.markdown(
        markdown_text,
        extensions=[
            "tables",
            "fenced_code",
            "sane_lists",
            "toc",
        ],
        output_format="html5",
    )
    # Markdown leaves checklist markers as text; render them consistently without
    # interactive form fields in the PDF.
    body = body.replace("[ ]", "□").replace("[x]", "■").replace("[X]", "■")
    body = re.sub(
        r"(<blockquote>.*?</blockquote>)",
        r"\1<div class=\"cover-note\">公开分享材料 · 可独立阅读 · 不含内部复现入口</div>",
        body,
        count=1,
        flags=re.S,
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="author" content="{html.escape('冯浩然')}">
  <meta name="description" content="脱敏的大模型训练、推理与 RL 算子优化技术报告">
  <title>大模型训推算子优化的证据闭环</title>
  <style>{CSS}</style>
</head>
<body>{body}</body>
</html>"""


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    markdown_text = SOURCE.read_text(encoding="utf-8")
    rendered = build_html(markdown_text)
    HTML_DEBUG.parent.mkdir(parents=True, exist_ok=True)
    HTML_DEBUG.write_text(rendered, encoding="utf-8")
    HTML(string=rendered, base_url=str(SOURCE.parent)).write_pdf(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
