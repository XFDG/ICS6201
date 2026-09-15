#!/usr/bin/env python3
"""Render every standalone Markdown article in doc/open_doc/topics to PDF."""

from pathlib import Path
import html
import re

import markdown
from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[1]
TOPICS = ROOT / "doc" / "open_doc" / "topics"
DEBUG_DIR = ROOT / "tmp" / "pdfs" / "topic_html"


BASE_CSS = r"""
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
  margin: 15.5mm 16mm 17mm 16mm;
  @top-left {
    content: "__RUNNING_TITLE__";
    font-family: "Noto Sans CJK SC", sans-serif;
    font-size: 7.8pt;
    color: #64748b;
  }
  @top-right {
    content: "脱敏技术专题";
    font-family: "Noto Sans CJK SC", sans-serif;
    font-size: 7.8pt;
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
html { color: #172033; background: #fff; }
body {
  font-family: "Noto Serif CJK SC", serif;
  font-size: 9.8pt;
  line-height: 1.52;
  letter-spacing: 0.008em;
}
h1, h2, h3, h4 {
  font-family: "Noto Sans CJK SC", sans-serif;
  color: #102a43;
  break-after: avoid;
  page-break-after: avoid;
}
h1 {
  font-size: 21pt;
  line-height: 1.28;
  text-align: left;
  margin: 6mm 0 3.5mm;
  padding: 0 0 3.5mm 4mm;
  border-left: 5pt solid #0b6e99;
  border-bottom: 1pt solid #8fb8ca;
}
h2 {
  font-size: 14.3pt;
  margin: 5.2mm 0 2.2mm;
  padding-left: 3mm;
  border-left: 3.5pt solid #0b6e99;
}
h3 { font-size: 11.8pt; margin: 3.8mm 0 1.6mm; color: #0b5676; }
h4 { font-size: 10.5pt; margin: 3mm 0 1.3mm; }
p { margin: 0 0 2.2mm; text-align: justify; orphans: 3; widows: 3; }
blockquote {
  margin: 2mm 0 4.5mm;
  padding: 3mm 4.5mm;
  background: #eff7fb;
  border-left: 3pt solid #0b6e99;
  color: #38566b;
  font-family: "Noto Sans CJK SC", sans-serif;
  font-size: 9.3pt;
}
blockquote p { margin: 0; text-align: left; }
ul, ol { margin: 1.2mm 0 2.5mm 6mm; padding-left: 4mm; }
li { margin: 0.7mm 0; }
strong { color: #0b4f6c; }
code {
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 8.55pt;
  color: #8b2942;
  background: #f5f7f9;
  padding: 0.1em 0.28em;
  border-radius: 2px;
  overflow-wrap: anywhere;
}
pre {
  margin: 2mm 0 3.2mm;
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
  margin: 2.5mm 0 4mm;
  font-family: "Noto Sans CJK SC", sans-serif;
  font-size: 8.35pt;
  line-height: 1.42;
  break-inside: avoid;
  page-break-inside: avoid;
}
thead { display: table-header-group; }
tr { break-inside: avoid; }
th {
  background: #0b6e99;
  color: white;
  font-weight: 400;
  padding: 1.7mm 2mm;
  border: 0.5pt solid #8fb8ca;
  text-align: left;
}
td {
  padding: 1.5mm 2mm;
  border: 0.5pt solid #cbd5e1;
  vertical-align: top;
}
tbody tr:nth-child(even) { background: #f6f9fb; }
img { display: block; max-width: 100%; max-height: 165mm; margin: 4mm auto 5mm; break-inside: avoid; }
"""


LONG_ARTICLE_CSS = r"""
body { font-size: 9.55pt; line-height: 1.47; }
h2 { margin-top: 4.6mm; margin-bottom: 1.9mm; }
h3 { margin-top: 3.3mm; margin-bottom: 1.4mm; }
p { margin-bottom: 1.9mm; }
ul, ol { margin-top: 1mm; margin-bottom: 2.1mm; }
li { margin-top: 0.55mm; margin-bottom: 0.55mm; }
table { margin-top: 2.2mm; margin-bottom: 3.4mm; }
"""


EXTRA_LONG_ARTICLE_CSS = r"""
body { font-size: 9.25pt; line-height: 1.42; }
h1 { margin-top: 4.5mm; margin-bottom: 3mm; padding-bottom: 3mm; }
h2 { margin-top: 4mm; margin-bottom: 1.7mm; }
h3 { margin-top: 2.8mm; margin-bottom: 1.2mm; }
p { margin-bottom: 1.65mm; }
blockquote { margin-bottom: 3.5mm; padding-top: 2.5mm; padding-bottom: 2.5mm; }
ul, ol { margin-top: 0.8mm; margin-bottom: 1.8mm; }
li { margin-top: 0.35mm; margin-bottom: 0.35mm; }
table { font-size: 8.05pt; line-height: 1.34; margin-top: 1.8mm; margin-bottom: 2.8mm; }
th { padding-top: 1.4mm; padding-bottom: 1.4mm; }
td { padding-top: 1.25mm; padding-bottom: 1.25mm; }
pre { margin-top: 1.6mm; margin-bottom: 2.5mm; padding-top: 2.5mm; padding-bottom: 2.5mm; }
"""


def title_from_markdown(source: str) -> str:
    for line in source.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    raise ValueError("Markdown article has no H1 title")


def make_html(markdown_text: str, title: str) -> str:
    body = markdown.markdown(
        markdown_text,
        extensions=["tables", "fenced_code", "sane_lists", "toc"],
        output_format="html5",
    )
    body = body.replace("[ ]", "□").replace("[x]", "■").replace("[X]", "■")
    running = title if len(title) <= 28 else title[:27] + "…"
    css = BASE_CSS.replace("__RUNNING_TITLE__", running.replace('"', "”"))
    compact_titles = (
        "MoE Router 正交损失",
        "CUDA Graph 下确定性 Router GEMM",
        "Blackwell 多精度训练",
        "为什么单机 Prefill/Decode",
    )
    if title.startswith(compact_titles):
        css += LONG_ARTICLE_CSS
    if title.startswith(("Blackwell 多精度训练", "为什么单机 Prefill/Decode")):
        css += EXTRA_LONG_ARTICLE_CSS
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="author" content="冯浩然">
  <meta name="description" content="{html.escape(title)}：脱敏技术专题">
  <title>{html.escape(title)}</title>
  <style>{css}</style>
</head>
<body>{body}</body>
</html>"""


def render(source: Path) -> Path:
    markdown_text = source.read_text(encoding="utf-8")
    title = title_from_markdown(markdown_text)
    rendered = make_html(markdown_text, title)
    output = source.with_suffix(".pdf")
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    (DEBUG_DIR / f"{source.stem}.html").write_text(rendered, encoding="utf-8")
    HTML(string=rendered, base_url=str(source.parent)).write_pdf(output)
    return output


def main() -> None:
    sources = sorted(TOPICS.glob("*.md"))
    if not sources:
        raise RuntimeError(f"No Markdown topics found in {TOPICS}")
    for source in sources:
        print(render(source))


if __name__ == "__main__":
    main()
