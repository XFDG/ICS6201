"""One-time archive relabeling; inspect the dry run before passing --apply.

Preserves bytes and filesystem timestamps. Dates are evidence-based estimates,
not a claim to recover edit events that neither Word nor Git recorded.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import re
import subprocess
import zipfile
import xml.etree.ElementTree as ET
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'doc/旧的简历'
MANIFEST = ARCHIVE / '归档清单_20260929.json'
TZ = ZoneInfo('Asia/Shanghai')

def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()

def metadata(path):
    try:
        if path.suffix == '.docx':
            with zipfile.ZipFile(path) as z:
                root = ET.fromstring(z.read('docProps/core.xml'))
            raw = root.findtext('{http://purl.org/dc/terms/}modified')
            date = datetime.fromisoformat(raw.replace('Z', '+00:00'))
            source = 'Word core.xml modified'
        elif path.suffix == '.pdf':
            meta = PdfReader(path).metadata
            date = meta.modification_date or meta.creation_date
            source = 'PDF ModDate/CreationDate'
        else:
            return None
        if date and date.year == 2026:
            return {'date': date.astimezone(TZ).isoformat(), 'source': source}
    except Exception:
        pass
    return None

def history(record, path):
    blob = git('hash-object', str(path))
    log = git('log', '--follow', '--raw', '--no-abbrev',
              '--format=COMMIT|%H|%cI|%s', '--', record['original'])
    commit = None
    for line in log.splitlines():
        if line.startswith('COMMIT|'):
            _, sha, date, subject = line.split('|', 3)
            commit = {'commit': sha, 'date': date, 'subject': subject}
        elif line.startswith(':') and commit:
            parts = line.split()
            if parts[3] == blob and parts[2] != parts[3]:
                return {**commit, 'status': parts[4], 'blob': blob}
    return None

def short_name(original):
    name = Path(original).stem
    rules = [
        ('匿名版_小红书水印', '匿名水印简历'),
        ('F同学', '匿名水印简历'), ('小冯别放弃', '匿名简历'),
        ('Haoran', '英文简历'), ('推理优化版', '推理简历'),
        ('训练优化版', '训练简历'), ('模型发布后', '模型发布简历'),
        ('RL训练推理算子版_内容', '训推RL审阅稿'),
        ('RL训练推理算子版', '训推RL简历'),
        ('开源贡献版', '开源贡献简历'), ('项目与科研版', '科研简历'),
        ('面试自我介绍', '面试介绍'), ('面试介绍', '面试介绍'),
        ('TE_RDMA', 'RDMA押题'), ('R3与FlashInfer', 'R3复盘'),
        ('FlashInfer与R3', 'FlashInfer与R3押题'), ('Mooncake', 'Mooncake押题'),
        ('量化Runtime', 'Runtime与Router押题'), ('工作总结', '工作总结'),
        ('素材库', '简历素材'), ('秋招简历', '秋招简历'),
    ]
    for key, label in rules:
        if key in name:
            return label
    if name.startswith('最新'):
        return '简历草稿' + name.removeprefix('最新')
    return '综合简历'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get('naming_version') == 2:
        raise SystemExit('Already reorganized; refusing a second migration.')
    records = manifest['files']
    by_original = {r['original']: r for r in records}
    plan = []
    used = set()
    for record in records:
        path = ROOT / record['archived']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        evidence = []
        meta = metadata(path)
        if meta:
            evidence.append(meta)
        hist = history(record, path)
        if hist:
            evidence.append({'source': 'Git content change (rename/copy excluded)', **hist})
        paired = None
        # A same-stem PDF can recover a date for python-docx's 2013 template.
        if path.suffix == '.docx' and not meta:
            pair = by_original.get(str(Path(record['original']).with_suffix('.pdf')))
            if pair:
                paired = metadata(ROOT / pair['archived'])
                if paired:
                    paired = {**paired, 'source': '同版 ' + paired['source']}
                    evidence.append(paired)
        chosen = meta or paired
        tracked = bool(git('ls-files', '--', record['original']))
        if not hist and tracked:
            chosen = {'date': datetime.fromtimestamp(record['mtime_ns']/1e9, TZ).isoformat(),
                      'source': '本地内容不同于 Git 已记录版本；以原 mtime 回退，内部时间可能沿用模板'}
        # A later actual modification proves that old template metadata is stale.
        if hist and hist['status'].startswith('M') and (
            not chosen or hist['date'][:10] > chosen['date'][:10]
        ):
            chosen = {'date': hist['date'], 'source': 'Git 实际内容修改；Word/PDF 时间较旧或缺失'}
        if not chosen:
            if hist:
                chosen = {'date': hist['date'], 'source': 'Git 内容记录（无有效文档时间）'}
            else:
                chosen = {'date': datetime.fromtimestamp(record['mtime_ns']/1e9, TZ).isoformat(),
                          'source': '原文件 mtime 回退（无匹配内容提交或有效元数据）'}
        day = datetime.fromisoformat(chosen['date']).astimezone(TZ)
        label = short_name(record['original'])
        base = f'{label} {day.month}月{day.day}日'
        target = ARCHIVE / (base + path.suffix)
        n = 2
        while target in used or (target.exists() and target != path):
            target = ARCHIVE / f'{label}{n} {day.month}月{day.day}日{path.suffix}'
            n += 1
        used.add(target)
        plan.append((path, target, record, evidence, chosen))
        print(path.name, '->', target.name, '|', chosen['source'])
    if not args.apply:
        return
    # Capture unpacked Word directories as independent preserved artifacts.
    directories = []
    for path in sorted(ARCHIVE.iterdir()):
        if not path.is_dir():
            continue
        match = next((dst for src, dst, *_ in plan if src.suffix == '.docx' and src.stem == path.name), None)
        if match is None:
            raise RuntimeError(f'Unrecognized directory: {path}')
        target = ARCHIVE / 'Word解包附件' / match.stem
        assert not target.exists()
        files = [{'relative': str(p.relative_to(path)),
                  'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                  'mtime_ns': p.stat().st_mtime_ns}
                 for p in path.rglob('*') if p.is_file()]
        directories.append((path, target, files))
    for src, dst, record, evidence, chosen in plan:
        src.rename(dst)
        record['previous_archived'] = record['archived']
        record['archived'] = str(dst.relative_to(ROOT))
        record['date_evidence'] = evidence
        record['naming_date'] = chosen
    manifest['directories'] = []
    for src, dst, files in directories:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        manifest['directories'].append({'original': str(src.relative_to(ROOT)),
            'archived': str(dst.relative_to(ROOT)), 'files': files})
    manifest['naming_version'] = 2
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    for record in records:
        p = ROOT / record['archived']
        assert p.stat().st_mtime_ns == record['mtime_ns']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == record['sha256']
    print('Verified preserved files:', len(records))

if __name__ == '__main__':
    main()
