#!/usr/bin/env python3
"""Build the AI Infra delivery directly from its two canonical PDF directories.

Requires pypdf. Existing PDFs with different content are never overwritten.
Generated metadata and ZIP revisions are backed up in the archive directory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "小红书商品"
ARCHIVE_ROOT = ROOT / "商品归档"
PUBLIC_NOTE_NAME = "00_说明与来源.md"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pdf_pages(data: bytes, label: str) -> int:
    if not data.startswith(b"%PDF-"):
        raise ValueError(f"Invalid PDF header: {label}")
    reader = PdfReader(io.BytesIO(data), strict=True)
    if reader.is_encrypted:
        raise ValueError(f"Encrypted PDF cannot be delivered: {label}")
    pages = len(reader.pages)
    if pages < 1:
        raise ValueError(f"Empty PDF: {label}")
    for page in reader.pages:
        _ = page.mediabox
    return pages


def backup_if_different(path: Path, digest: str, backup_dir: Path) -> None:
    """Keep older generated content outside both customer-facing directories."""
    if not path.exists():
        return
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Generated output must be a regular file: {path}")
    previous = sha256(path.read_bytes())
    if previous == digest:
        return
    path_key = sha256(str(path.resolve()).encode("utf-8"))[:16]
    backup = backup_dir / f"{path.name}.{path_key}.{previous}.bak"
    backup_dir.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        if backup.is_symlink() or not backup.is_file() or sha256(backup.read_bytes()) != previous:
            raise ValueError(f"Conflicting generated-output backup: {backup}")
    else:
        shutil.copyfile(path, backup)
        if sha256(backup.read_bytes()) != previous:
            raise ValueError(f"Backup checksum mismatch: {backup}")


def write_generated(path: Path, data: bytes, backup_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_if_different(path, sha256(data), backup_dir)
    with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def role_for(path: str) -> str:
    if path.startswith("技术PDF/"):
        return "技术学习"
    return "公司面试复盘"


def group_for(path: str) -> str:
    parts = Path(path).parts
    if parts[0] == "技术PDF":
        return "技术资料"
    if len(parts) != 2 or parts[0] != "面试" or Path(parts[1]).suffix.lower() != ".pdf":
        raise ValueError(f"Interview PDFs must be flat company files: {path}")
    return "公司面试复盘"


def flat_interview_pdfs(directory: Path, *, required: bool = True) -> list[Path]:
    """Reject obsolete round folders and non-PDF files without deleting them."""
    if not directory.exists() and not required:
        return []
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Interview directory must be a regular directory: {directory}")
    files = sorted(directory.iterdir())
    for path in files:
        if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".pdf":
            raise ValueError(f"Interview directory may contain only flat company PDFs; entry preserved: {path}")
    return files


def count_unique(rows: list[dict]) -> tuple[int, int]:
    by_hash = {row["sha256"]: row["pages"] for row in rows}
    return len(by_hash), sum(by_hash.values())


def markdown_link(path: str) -> str:
    label = path.replace("|", "\\|")
    return f"[{label}](<{path}>)"


def reading_notes(rows: list[dict], build_date: str, title: str) -> str:
    pdfs = [row for row in rows if row["kind"] == "pdf"]
    unique_count, unique_pages = count_unique(pdfs)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in pdfs:
        groups[group_for(row["path"])].append(row)
    lines = [
        f"# {title}",
        "",
        f"整理日期：{build_date}",
        "",
        f"本次交付共 {len(pdfs)} 个 PDF 文件；按文件 SHA-256 去重后为 **{unique_count} 份 PDF、{unique_pages} 页**。统计不包含阅读与来源说明，不把同一文件的多份副本重复计数。",
        "",
        "## 包含什么",
        "",
        "| 分类 | PDF 份数（按 SHA-256 去重） | 页数 |",
        "| --- | ---: | ---: |",
    ]
    for group in ("技术资料", "公司面试复盘"):
        number, pages = count_unique(groups[group])
        lines.append(f"| {group} | {number} | {pages} |")
    lines += [
        "",
        "面试目录按公司平铺，每家公司一份 PDF；同一公司的相关岗位与已确认轮次集中在公司合册中。技术资料按主题分类。",
        "",
        "## 先读什么",
        "",
        "1. 先读 AI Infra、推理、训练、RL 与算子优化八股文，建立知识框架。",
        "2. 按目标公司阅读面试合册，结合其中的岗位与轮次复习问题。",
        "3. 区分记录中确认的问题和整理补充的复习提纲，用自己的项目准备回答与证据。",
        "4. 对需要深入的主题，再阅读 vLLM、DeepInfer、MoE、GPU 架构与论文解读。",
        "",
        "## 阅读口径",
        "",
        "- 公司合册可能包含多个岗位或轮次，文件份数不代表独立面试场次；不同技术资料也可能覆盖相近知识点。文件哈希去重无法识别主题重合。",
        "- 面试资料是去噪、脱敏后的复盘与学习讲义；实际记录中的问题和整理者补充的复习内容应按各册标注区分。内容不构成公司的官方题库或标准答案。",
        "- 部分 PDF 保留了原有的小红书账号水印。",
        "- 本包仅交付整理后的 PDF 与阅读/来源说明，不含面试原始转写、录音、个人 Offer 图片或商品营销文档。",
        "- 技术资料反映各文档标注版本与整理时间；框架行为、硬件参数和性能判断应结合实际版本及测量条件理解。",
        "",
        "## 阅读说明文件",
        "",
        "- 本说明：阅读路线、份数与完整 PDF 目录。",
    ]
    for row in rows:
        if row["kind"] == "reading_note":
            lines.append(f"- {markdown_link(row['path'])}：对应资料的使用方式与来源/基础校准。")
    lines += [
        "",
        "## PDF 文件目录",
        "",
        "| 文件 | 页数 | 用途 |",
        "| --- | ---: | --- |",
    ]
    for row in pdfs:
        lines.append(f"| {markdown_link(row['path'])} | {row['pages']} | {row['content_role']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE, help="整理后的商品目录，内含面试与技术PDF")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "AI INFRA 面经+知识总结", help="对客资料目录")
    parser.add_argument("--archive", type=Path, help="输出 ZIP 路径（默认与资料目录同级，含整理日期）")
    parser.add_argument("--manifest", type=Path, default=ARCHIVE_ROOT / "03_整理记录/交付清单.json", help="机器清单路径")
    parser.add_argument("--backup-dir", type=Path, default=ARCHIVE_ROOT / "04_历史附件/历史备份", help="历史生成物备份目录")
    parser.add_argument("--date", default=date.today().isoformat(), help="整理日期 YYYY-MM-DD")
    args = parser.parse_args()
    build_date = date.fromisoformat(args.date)
    if build_date.year < 1980:
        raise ValueError("ZIP build date must be 1980 or later")
    source, output = args.source.resolve(), args.output.resolve()
    archive = (args.archive or output.with_name(f"{output.name}_{args.date}.zip")).resolve()
    manifest_path, backup_dir = args.manifest.resolve(), args.backup_dir.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate directory trees")
    for directory in (source, output):
        if any(path == directory or directory in path.parents for path in (archive, manifest_path, backup_dir)):
            raise ValueError("Archive, manifest and backups must be outside both product directories")
    if archive.suffix.lower() != ".zip" or manifest_path.suffix.lower() != ".json":
        raise ValueError("--archive must name a .zip and --manifest a .json; source PDFs/Markdown cannot be outputs")

    planned: dict[str, Path] = {}
    for path in flat_interview_pdfs(source / "面试"):
        planned[path.relative_to(source).as_posix()] = path
    technical = source / "技术PDF"
    if technical.is_symlink() or not technical.is_dir():
        raise ValueError(f"Technical directory must be a regular directory: {technical}")
    for path in sorted(technical.rglob("*")):
        if path.is_file() and (path.suffix.lower() == ".pdf" or path.name == PUBLIC_NOTE_NAME):
            planned[path.relative_to(source).as_posix()] = path

    # Validate all sources and PDF conflicts before copying any delivery files.
    rows: list[dict] = []
    for relative, path in sorted(planned.items()):
        if path.is_symlink():
            raise ValueError(f"Symlink source is not supported: {path}")
        data = path.read_bytes()
        is_pdf = path.suffix.lower() == ".pdf"
        destination = output / relative
        digest = sha256(data)
        if destination.is_symlink():
            raise ValueError(f"Symlink delivery file is not supported: {destination}")
        if is_pdf and destination.exists() and (not destination.is_file() or sha256(destination.read_bytes()) != digest):
            raise ValueError(f"Existing PDF differs; preserved without overwriting: {destination}")
        rows.append({
            "path": relative,
            "source_path": f"{source.name}/{relative}",
            "sha256": digest,
            "bytes": len(data),
            "kind": "pdf" if is_pdf else "reading_note",
            "pages": pdf_pages(data, relative) if is_pdf else 0,
            "content_role": role_for(relative) if is_pdf else "来源与阅读说明",
        })
    flat_interview_pdfs(output / "面试", required=False)
    for name in ("面试", "技术PDF"):
        for path in (output / name).rglob("*"):
            if not path.is_file() or path.suffix.lower() != ".pdf":
                continue
            if path.relative_to(output).as_posix() not in planned:
                raise ValueError(f"Unmapped delivery PDF preserved; organize it before rebuilding: {path}")

    for row in rows:
        destination = output / row["path"]
        data = planned[row["path"]].read_bytes()
        if row["kind"] == "reading_note":
            write_generated(destination, data, backup_dir)
        elif not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        if sha256(destination.read_bytes()) != row["sha256"]:
            raise ValueError(f"Delivery checksum mismatch: {row['path']}")

    notes = reading_notes(rows, args.date, output.name).encode("utf-8")
    source_notes = notes
    if (source / "上架素材/闲鱼").is_dir():
        listing_links = (
            "## 商品图片与宣传文案\n\n"
            "- [闲鱼商品主图](上架素材/闲鱼/商品图/01_闲鱼主图_AI_INFRA.png)\n"
            "- [字节录用意向书水印图](上架素材/闲鱼/商品图/02_字节录用意向书_闲鱼草原小可汗水印.png)\n"
            "- [闲鱼商品宣传文案](上架素材/闲鱼/闲鱼商品文案.md)\n"
            "- [小红书商品宣传文案](上架素材/小红书/商品宣传文案.md)\n"
            "- [全部上架素材与说明](上架素材/闲鱼/README.md)\n\n"
        )
        source_notes = notes.decode("utf-8").replace("\n\n", "\n\n" + listing_links, 1).encode("utf-8")
    write_generated(source / "README.md", source_notes, backup_dir)
    notes_path = output / "00_资料目录.md"
    write_generated(notes_path, notes, backup_dir)
    pdfs = [row for row in rows if row["kind"] == "pdf"]
    unique_count, unique_pages = count_unique(pdfs)
    grouped = defaultdict(list)
    for row in pdfs:
        grouped[group_for(row["path"])].append(row)
    summary = {
        "pdf_files": len(pdfs),
        "unique_pdf_files": unique_count,
        "unique_pdf_pages": unique_pages,
        "duplicate_pdf_files": len(pdfs) - unique_count,
        "groups": {name: {"pdf_files": len(items), "unique_pdf_files": count_unique(items)[0], "unique_pdf_pages": count_unique(items)[1]} for name, items in sorted(grouped.items())},
    }

    archive.parent.mkdir(parents=True, exist_ok=True)
    archive_files = [notes_path, *(output / row["path"] for row in rows)]
    with tempfile.NamedTemporaryFile(prefix=archive.name + ".", suffix=".tmp", dir=archive.parent, delete=False) as handle:
        temporary_archive = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for path in archive_files:
                info = zipfile.ZipInfo(path.relative_to(output).as_posix(), (*build_date.timetuple()[:3], 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                bundle.writestr(info, path.read_bytes())
        with zipfile.ZipFile(temporary_archive) as bundle:
            if bundle.testzip() is not None or set(bundle.namelist()) != {path.relative_to(output).as_posix() for path in archive_files}:
                raise ValueError("Archive membership or CRC verification failed")
            for row in rows:
                if sha256(bundle.read(row["path"])) != row["sha256"]:
                    raise ValueError(f"Archive checksum mismatch: {row['path']}")
        backup_if_different(archive, sha256(temporary_archive.read_bytes()), backup_dir)
        temporary_archive.replace(archive)
    finally:
        temporary_archive.unlink(missing_ok=True)

    manifest = {
        "schema_version": 3,
        "date": args.date,
        "source_root": source.name,
        "delivery_directory": output.name,
        "archive": archive.name,
        "archive_sha256": sha256(archive.read_bytes()),
        "statistics_method": "Identical SHA-256 counts once; unique pages summed once per hash. Similar topics remain separate files.",
        "summary": summary,
        "reading_note": {"path": "00_资料目录.md", "source_path": f"{source.name}/README.md", "sha256": sha256(notes), "source_sha256": sha256(source_notes)},
        "files": rows,
    }
    write_generated(manifest_path, (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"), backup_dir)
    print(json.dumps({"output": str(output), "archive": str(archive), "manifest": str(manifest_path), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, zipfile.BadZipFile) as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
