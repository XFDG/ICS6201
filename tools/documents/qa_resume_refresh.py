"""Check delivered PDFs and archive integrity; render previews for visual review."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import subprocess
from pypdf import PdfReader
import pdfplumber

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--render',action='store_true')
args=parser.parse_args()
spec=importlib.util.spec_from_file_location('sheets',Path(__file__).with_name('make_pdf_contact_sheets.py'))
sheets=importlib.util.module_from_spec(spec);spec.loader.exec_module(sheets)
pdfs=sorted((ROOT/'doc').glob('*.pdf'))+sorted((ROOT/'doc/面试押题').glob('*.pdf'))
for file in pdfs:
    reader=PdfReader(file)
    if file.parent==ROOT/'doc':
        assert len(reader.pages)<=(1 if '面试介绍' in file.name else 2),file
    with pdfplumber.open(file) as doc:
        for i,p in enumerate(doc.pages):
            assert len(p.chars)>15,(file,i,'empty page')
            outside=[c for c in p.chars if c['x0'] < 3 or c['x1'] > p.width-3 or c['top'] < 0 or c['bottom'] > p.height]
            assert not outside,(file,i,'overflow',outside[:1])
    print(file.name,'pages=',len(reader.pages),'links=',sum(len(p.get('/Annots',[])) for p in reader.pages),flush=True)
    if args.render:
        target=ROOT/'tmp/resume_refresh_20260929/qa_current'/file.stem
        target.mkdir(parents=True,exist_ok=True)
        subprocess.run(['pdftoppm','-scale-to','1200','-png',str(file),str(target/'page')],check=True,stdout=subprocess.DEVNULL)
        print(sheets.make_sheet(target),flush=True)
manifest=json.loads((ROOT/'doc/旧的简历/归档清单_20260929.json').read_text())
for record in manifest['files']:
    file=ROOT/record['archived']
    assert file.stat().st_mtime_ns==record['mtime_ns'],file
    assert hashlib.sha256(file.read_bytes()).hexdigest()==record['sha256'],file
for record in manifest.get('directories',[]):
    for item in record['files']:
        file=ROOT/record['archived']/item['relative']
        assert file.stat().st_mtime_ns==item['mtime_ns'],file
        assert hashlib.sha256(file.read_bytes()).hexdigest()==item['sha256'],file
assert {p.name for p in (ROOT/'doc').glob('*.md')} == {
    'README.md','冯浩然_AiInfra香港中文大学_15024999885.md'
}
assert not (ROOT/'doc/面试押题/README.pdf').exists()
assert not (ROOT/'output').exists()
print('ARCHIVE VERIFIED:',len(manifest['files']),'files; output directory absent.')
