"""Render interview notes with the existing technical-article stylesheet."""
from pathlib import Path
import argparse
import importlib.util
import html
import markdown
from weasyprint import HTML

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('topic_style',Path(__file__).with_name('render_open_doc_topics.py'))
topic=importlib.util.module_from_spec(spec)
spec.loader.exec_module(topic)
parser=argparse.ArgumentParser()
parser.add_argument('sources',nargs='*',type=Path)
args=parser.parse_args()
sources=args.sources or sorted((ROOT/'doc/面试押题').glob('面试押题*.md'))
for source in sources:
    text=source.read_text()
    title=topic.title_from_markdown(text)
    css=topic.BASE_CSS.replace('__RUNNING_TITLE__',title[:25].replace('"',''))
    css=css.replace('脱敏技术专题','个人面试准备')
    css+='''
      table { break-inside:auto; page-break-inside:auto; table-layout:fixed; }
      th,td { overflow-wrap:anywhere; word-break:normal; }
      a { color:#12628a; overflow-wrap:anywhere; text-decoration:none; }
      p { overflow-wrap:anywhere; }
      pre { overflow-wrap:anywhere; white-space:pre-wrap; }
      pre code { font-family:"Noto Sans CJK SC", monospace; }
      h1 { font-size:19pt; } h2 { font-size:13.5pt; }
    '''
    body=markdown.markdown(text,extensions=['tables','fenced_code','sane_lists','toc'])
    page=f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>{css}</style></head><body>{body}</body></html>'
    HTML(string=page,base_url=str(source.parent.resolve())).write_pdf(source.with_suffix('.pdf'))
    print(source.with_suffix('.pdf'),flush=True)
