#!/usr/bin/env python3
"""Render the reviewed, de-identified sale notes; never ingest raw transcripts.

Usage: python tools/documents/render_sale_interviews.py [reviewed.md ...]
"""
from pathlib import Path
import argparse
import html
import importlib.util

import markdown
from weasyprint import HTML

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "sale_topic_style", Path(__file__).with_name("render_open_doc_topics.py")
)
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)


def render(source: Path) -> Path:
    source = source.resolve()
    text = source.read_text(encoding="utf-8")
    title = style.title_from_markdown(text)
    css = style.BASE_CSS.replace("__RUNNING_TITLE__", title[:28].replace('"', "”"))
    css = css.replace("脱敏技术专题", "闲鱼 · 草原小可汗")
    css += """
    body { font-size: 10.2pt; line-height: 1.6; }
    h1 { font-size: 20pt; }
    h2 { font-size: 13.5pt; }
    h3 { font-size: 11.2pt; }
    table { break-inside: auto; page-break-inside: auto; table-layout: fixed; }
    th, td, p, a { overflow-wrap: anywhere; }
    a { color: #12628a; text-decoration: none; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; }
    """
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    page = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title><style>{css}</style></head>"
        f"<body>{body}</body></html>"
    )
    output = source.with_suffix(".pdf")
    HTML(string=page, base_url=str(source.parent)).write_pdf(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", type=Path)
    args = parser.parse_args()
    sources = args.sources or sorted((ROOT / "小红书商品/面试补充").glob("[0-9][0-9]_*.md"))
    for source in sources:
        print(render(source), flush=True)
