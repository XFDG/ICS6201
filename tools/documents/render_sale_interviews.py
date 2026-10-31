#!/usr/bin/env python3
"""Render the reviewed, de-identified sale notes; never ingest raw transcripts.

Defaults: read 商品归档/02_面试源稿/Markdown and render maintenance PDFs to
商品归档/02_面试源稿/排版中间稿. Single-round Markdown files are maintenance
sources; only the merged company PDFs belong in the customer delivery.

Usage: python tools/documents/render_sale_interviews.py [reviewed.md ...]
       python tools/documents/render_sale_interviews.py --output-dir /tmp/previews
       python tools/documents/render_sale_interviews.py --dry-run
"""
from pathlib import Path
import argparse
import html
import importlib.util

import markdown
from weasyprint import HTML

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = ROOT / "商品归档/02_面试源稿/Markdown"
DEFAULT_OUTPUT_DIR = ROOT / "商品归档/02_面试源稿/排版中间稿"
spec = importlib.util.spec_from_file_location(
    "sale_topic_style", Path(__file__).with_name("render_open_doc_topics.py")
)
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)


def render(source: Path, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    source = source.resolve()
    if source.suffix.lower() != ".md":
        raise ValueError(f"Only reviewed Markdown sources are accepted: {source}")
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
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{source.stem}.pdf"
    HTML(string=page, base_url=str(source.parent)).write_pdf(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", type=Path)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR,
                        help="Directory to search when explicit sources are omitted")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Maintenance preview destination; defaults to the archive's 排版中间稿 folder")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print source-to-output mappings without writing PDFs")
    args = parser.parse_args()
    sources = args.sources or sorted(args.source_dir.glob("[0-9][0-9]_*.md"))
    if not sources:
        parser.error(f"No reviewed Markdown sources found in {args.source_dir}")
    if any(not source.is_file() or source.suffix.lower() != ".md" for source in sources):
        parser.error("Every source must be an existing reviewed Markdown file")
    outputs = [args.output_dir.resolve() / f"{source.stem}.pdf" for source in sources]
    if len(set(outputs)) != len(outputs):
        parser.error("Sources have duplicate filenames that would overwrite one output")
    for source in sources:
        if args.dry_run:
            print(f"{source.resolve()} -> {args.output_dir.resolve() / (source.stem + '.pdf')}")
        else:
            print(render(source, args.output_dir), flush=True)
