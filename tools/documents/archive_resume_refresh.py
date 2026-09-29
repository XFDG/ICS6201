"""Archive only enumerated resume/interview artifacts, preserving mtime and hashes."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'doc'
ARCHIVE = DOC / '旧的简历'
MANIFEST = ARCHIVE / '归档清单_20260929.json'
parser=argparse.ArgumentParser()
parser.add_argument('--apply',action='store_true')
args=parser.parse_args()
if MANIFEST.exists():
    raise SystemExit('This archive pass already has a manifest; refusing to repeat.')
suffixes={'.md','.docx','.pdf'}
targets=sorted([p for p in DOC.iterdir() if p.is_file() and p.suffix in suffixes])
targets+=sorted(p for p in (DOC/'面试押题').iterdir() if p.is_file() and p.suffix in suffixes)
targets+=sorted(p for p in ARCHIVE.iterdir() if p.is_file() and p.suffix in suffixes)
targets+=sorted(p for p in (ROOT/'output').rglob('*') if p.is_file())
plans=[]
reserved=set()
for p in targets:
    stat=p.stat()
    timestamp=datetime.fromtimestamp(stat.st_mtime,ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    name=f'{timestamp}__{p.name}'
    dest=ARCHIVE/name
    n=1
    while dest in reserved or (dest.exists() and dest!=p):
        dest=ARCHIVE/f'{timestamp}__{p.stem}__{n}{p.suffix}'
        n+=1
    reserved.add(dest)
    plans.append({'original':str(p.relative_to(ROOT)),'archived':str(dest.relative_to(ROOT)),
                  'mtime_ns':stat.st_mtime_ns,'size':stat.st_size,
                  'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
for item in plans: print(item['original'],'->',item['archived'])
print('Files:',len(plans),'mode:','APPLY' if args.apply else 'DRY RUN')
if args.apply:
    ARCHIVE.mkdir(exist_ok=True)
    for item in plans:
        p=ROOT/item['original']; dest=ROOT/item['archived']
        if p!=dest: p.rename(dest)
        assert hashlib.sha256(dest.read_bytes()).hexdigest()==item['sha256']
        assert dest.stat().st_mtime_ns==item['mtime_ns']
    MANIFEST.write_text(json.dumps({'timestamp_timezone':'Asia/Shanghai', 'files':plans},ensure_ascii=False,indent=2)+'\n')
    for directory in sorted((ROOT/'output').rglob('*'), key=lambda p:len(p.parts), reverse=True):
        if directory.is_dir(): directory.rmdir()
    (ROOT/'output').rmdir()
