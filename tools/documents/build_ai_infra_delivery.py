#!/usr/bin/env python3
"""Build the customer-facing AI Infra PDF delivery without copying raw records.

Requires pypdf. Existing delivery files are preserved; conflicting names stop the
build. The ZIP contains only the two documented PDF trees and reading notes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import zipfile

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "小红书商品"
ROOT_PDFS = {
    "DeepInfer_仓库深度解析.pdf": "05_LLM 推理与面试复习/DeepInfer_仓库深度解析.pdf",
    "Megatron_Core_MoE_论文深度解析.pdf": "04_论文深度解读/Megatron_Core_MoE_论文深度解析.pdf",
    "RL_Interview_Handbook_2026-10-31.pdf": "06_训练与强化学习/RL_Interview_Handbook_2026-10-31.pdf",
    "Training_Interview_Handbook_2026-10-31.pdf": "06_训练与强化学习/Training_Interview_Handbook_2026-10-31.pdf",
}
PAPER_ARCHIVE = "AI_Infra_7篇论文与技术_深度解析PDF合集.zip"
PAPER_DIRECTORY = "04_论文深度解读/7篇论文解析"
PUBLIC_NOTES = ("00_说明与来源.md", "阅读说明与基础校准.md")


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
    # Access every page so malformed page trees fail before publication.
    for page in reader.pages:
        _ = page.mediabox
    return pages


def write_if_missing(path: Path, data: bytes) -> bool:
    """Preserve identical existing files and refuse to overwrite different ones."""
    if path.exists():
        if not path.is_file() or sha256(path.read_bytes()) != sha256(data):
            raise ValueError(f"Existing file differs; preserved without overwriting: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def backup_if_different(path: Path, digest: str) -> None:
    """Preserve previous generated metadata before replacing changed content."""
    if not path.exists():
        return
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Generated output must be a regular file: {path}")
    previous = sha256(path.read_bytes())
    if previous == digest:
        return
    backup = path.with_name(f"{path.name}.{previous}.bak")
    if backup.exists():
        if not backup.is_file() or sha256(backup.read_bytes()) != previous:
            raise ValueError(f"Conflicting generated-output backup: {backup}")
    else:
        shutil.copyfile(path, backup)
        if sha256(backup.read_bytes()) != previous:
            raise ValueError(f"Backup checksum mismatch: {backup}")


def write_generated(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_if_different(path, sha256(data))
    with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def decoded_member(info: zipfile.ZipInfo) -> PurePosixPath:
    name = info.filename
    if not info.flag_bits & 0x800:
        # Legacy Windows ZIPs often store GBK names without the UTF-8 flag.
        try:
            raw = name.encode("cp437")
            for encoding in ("utf-8", "gb18030"):
                try:
                    name = raw.decode(encoding)
                    break
                except UnicodeDecodeError:
                    pass
        except UnicodeEncodeError:
            pass
    if "\\" in name or "\x00" in name:
        raise ValueError(f"Unsafe ZIP filename: {name!r}")
    member = PurePosixPath(name)
    if member.is_absolute() or any(p in ("..", ".") for p in member.parts):
        raise ValueError(f"Unsafe ZIP path: {name!r}")
    if not member.parts or ":" in member.parts[0]:
        raise ValueError(f"Unsafe ZIP path: {name!r}")
    return member


def normalize_sources(source: Path) -> dict[str, str]:
    """Add categorized PDF copies and safely unpack the seven-paper archive."""
    origins: dict[str, str] = {}
    technical = source / "技术PDF"
    for filename, relative in ROOT_PDFS.items():
        original = source / filename
        target = technical / relative
        if not original.is_file() and not target.is_file():
            raise FileNotFoundError(f"Neither original nor categorized PDF exists: {filename}")
        data = (original if original.is_file() else target).read_bytes()
        pdf_pages(data, filename)
        write_if_missing(target, data)
        if sha256(target.read_bytes()) != sha256(data):
            raise ValueError(f"Copy checksum mismatch: {target}")
        origins[target.relative_to(source).as_posix()] = filename

    archive = source / PAPER_ARCHIVE
    if not archive.is_file():
        # Git stores the categorized PDFs, while the original ZIP and root PDF
        # copies may intentionally remain local. A clean checkout is sufficient.
        paper_directory = technical / PAPER_DIRECTORY
        papers = sorted(paper_directory.rglob("*.pdf"))
        if len(papers) != 7:
            raise FileNotFoundError(f"Original paper ZIP is absent and categorized set is incomplete: {paper_directory}")
        for paper in papers:
            pdf_pages(paper.read_bytes(), str(paper))
            member = paper.relative_to(paper_directory).as_posix()
            origins[paper.relative_to(source).as_posix()] = f"{PAPER_ARCHIVE}!{member}"
        return origins
    extracted_names: set[str] = set()
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError(f"Corrupt ZIP: {archive}")
        for info in bundle.infolist():
            if info.is_dir():
                continue
            relative = decoded_member(info)
            if relative.suffix.lower() != ".pdf":
                raise ValueError(f"Unexpected non-PDF archive member: {relative}")
            if relative.as_posix() in extracted_names:
                raise ValueError(f"Duplicate archive filename: {relative}")
            extracted_names.add(relative.as_posix())
            data = bundle.read(info)
            pdf_pages(data, f"{PAPER_ARCHIVE}!{relative}")
            target = technical / PAPER_DIRECTORY / relative
            write_if_missing(target, data)
            if sha256(target.read_bytes()) != sha256(data):
                raise ValueError(f"Extraction checksum mismatch: {target}")
            origins[target.relative_to(source).as_posix()] = f"{PAPER_ARCHIVE}!{relative}"
    if len(extracted_names) != 7:
        raise ValueError(f"Expected 7 paper PDFs, found {len(extracted_names)}")
    return origins


def role_for(path: str) -> str:
    if path.startswith("技术PDF/"):
        return "技术学习"
    if "HR" in path.upper():
        return "HR沟通准备"
    if any(word in path for word in ("三七互娱", "跨域业务")):
        return "跨域赠阅"
    return "AI Infra技术面试复盘"


def group_for(path: str) -> str:
    if path.startswith("技术PDF/"):
        return "技术资料"
    if path.startswith("面试/03_补充复盘/"):
        return "新增按轮次复盘"
    if path.startswith("面试/02_专题复盘/"):
        return "原有专题复盘"
    return "原有按公司复盘"


def count_unique(rows: list[dict]) -> tuple[int, int]:
    by_hash = {row["sha256"]: row["pages"] for row in rows}
    return len(by_hash), sum(by_hash.values())


def reading_notes(rows: list[dict], build_date: str, output_name: str) -> str:
    pdfs = [row for row in rows if row["kind"] == "pdf"]
    unique_count, unique_pages = count_unique(pdfs)
    counts: dict[str, list[dict]] = defaultdict(list)
    for row in pdfs:
        counts[group_for(row["path"])].append(row)
    lines = [
        f"# {output_name}",
        "",
        f"整理日期：{build_date}",
        "",
        f"本次交付共 {len(pdfs)} 个 PDF 文件；按文件 SHA-256 去重后为 **{unique_count} 份 PDF、{unique_pages} 页**。统计不包含本说明及来源说明，不把同一文件的多份副本重复计数。",
        "",
        "## 包含什么",
        "",
        "| 分类 | PDF 份数（按 SHA-256 去重） | 页数 |",
        "| --- | ---: | ---: |",
    ]
    for group in ("技术资料", "原有按公司复盘", "原有专题复盘", "新增按轮次复盘"):
        number, pages = count_unique(counts[group])
        lines.append(f"| {group} | {number} | {pages} |")
    role_counts = Counter(row["content_role"] for row in {r["sha256"]: r for r in pdfs}.values())
    lines += [
        "",
        f"面试资料中，AI Infra 技术面试复盘 {role_counts['AI Infra技术面试复盘']} 份，HR 沟通准备 {role_counts['HR沟通准备']} 份，跨域赠阅 {role_counts['跨域赠阅']} 份。HR 资料侧重求职沟通；三七互娱海外游戏运营与跨域业务分析作为赠阅，不计为 AI Infra 技术面试场次。",
        "",
        "## 先读什么",
        "",
        "1. 先读 AI Infra、推理、训练、RL 与算子优化八股文，建立知识框架。",
        "2. 按目标岗位阅读按公司复盘，再用专题复盘练习从现象、证据、机制到验证的讲述。",
        "3. 阅读补充复盘中的真实问题整理与单独标明的复习提纲；根据自己的项目准备回答。",
        "4. 对需要深入的主题，再阅读 vLLM、DeepInfer、MoE、GPU 架构与论文解读。",
        "",
        "## 阅读口径",
        "",
        "- 按公司整理、专题整理与新增按轮次整理采用不同组织方式，可能覆盖同一场面试或相近知识点；份数代表不同文件，不代表互不重合的面试场次。文件哈希去重无法识别主题重合。",
        "- 面试资料是去噪、脱敏后的复盘与学习讲义；实际记录中的问题和整理者补充的复习内容应按各册标注区分。内容不构成公司的官方题库或标准答案。",
        "- 原有 PDF 中保留了已有的小红书账号水印。部分原有专题 PDF 为图像型，可能不能搜索或复制文字。",
        "- 本包仅交付整理后的 PDF 与阅读/来源说明，不含面试原始转写、录音、个人 Offer 图片或商品营销文档。",
        "- 技术资料反映各文档标注版本与整理时间；框架行为、硬件参数和性能判断应结合实际版本及测量条件理解。",
        "",
        "## 阅读说明文件",
        "",
        "- `00_资料目录.md`：本说明、阅读路线与完整 PDF 目录。",
    ]
    for row in rows:
        if row["kind"] == "reading_note":
            lines.append(f"- `{row['path']}`：对应资料的使用方式与来源/基础校准。")
    lines += [
        "",
        "## PDF 文件目录",
        "",
        "| 文件 | 页数 | 用途 |",
        "| --- | ---: | --- |",
    ]
    for row in sorted(pdfs, key=lambda row: row["path"]):
        label = row["path"].replace("|", "\\|")
        lines.append(f"| {label} | {row['pages']} | {row['content_role']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE, help="资料源目录（默认仓库中的小红书商品）")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "AI INFRA 面经+知识总结", help="对客资料目录；保留其中已有文件")
    parser.add_argument("--archive", type=Path, help="输出 ZIP 路径（默认与资料目录同级，含整理日期）")
    parser.add_argument("--manifest", type=Path, help="机器清单路径（默认源目录/交付清单.json）")
    parser.add_argument("--date", default=date.today().isoformat(), help="整理日期 YYYY-MM-DD")
    parser.add_argument("--expected-supplement-count", type=int, default=11, help="要求已完成的新增复盘 PDF 数量，防止误打包未完成版本")
    args = parser.parse_args()
    date.fromisoformat(args.date)
    source = args.source.resolve()
    output = args.output.resolve()
    archive = (args.archive or output.with_name(f"{output.name}_{args.date}.zip")).resolve()
    manifest_path = (args.manifest or source / "交付清单.json").resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate directory trees")
    if output == archive or output in archive.parents:
        raise ValueError("Archive must be outside the delivery directory")
    if archive.suffix.lower() != ".zip":
        raise ValueError("--archive must name a .zip file, never a source PDF or reading note")
    if manifest_path.suffix.lower() != ".json":
        raise ValueError("--manifest must name a .json file, never a source PDF or reading note")
    if archive == manifest_path or archive == source / PAPER_ARCHIVE:
        raise ValueError("Generated outputs must not overwrite each other or the original paper ZIP")
    for generated in (archive, manifest_path):
        if generated.suffix.lower() in (".pdf", ".md"):
            raise ValueError(f"Generated output cannot overwrite a PDF or Markdown input: {generated}")

    supplements = source / "面试补充"
    supplement_pdfs = sorted(supplements.rglob("*.pdf")) if supplements.exists() else []
    if len(supplement_pdfs) != args.expected_supplement_count:
        raise ValueError(f"Expected {args.expected_supplement_count} completed supplemental PDFs in {supplements}; found {len(supplement_pdfs)}")
    provenance = normalize_sources(source)

    planned: dict[str, Path] = {}
    for directory, destination in [
        (source / "技术PDF", Path("技术PDF")),
        (source / "面试", Path("面试")),
        (source / "面试new", Path("面试/02_专题复盘")),
        (supplements, Path("面试/03_补充复盘")),
    ]:
        for path in sorted(directory.rglob("*.pdf")):
            if path.is_symlink():
                raise ValueError(f"Symlink source is not supported: {path}")
            planned[(destination / path.relative_to(directory)).as_posix()] = path
    for name in PUBLIC_NOTES:
        for path in sorted((source / "技术PDF").rglob(name)):
            planned[(Path("技术PDF") / path.relative_to(source / "技术PDF")).as_posix()] = path
    supplement_note = supplements / "阅读说明与基础校准.md"
    if supplement_pdfs:
        if not supplement_note.is_file():
            raise FileNotFoundError(supplement_note)
        planned["面试/03_补充复盘/阅读说明与基础校准.md"] = supplement_note

    output.mkdir(parents=True, exist_ok=True)
    existing_by_hash: dict[str, str] = {}
    for directory in (output / "技术PDF", output / "面试"):
        for path in sorted(directory.rglob("*.pdf")):
            existing_by_hash.setdefault(sha256(path.read_bytes()), path.relative_to(output).as_posix())

    source_by_hash: dict[str, list[str]] = defaultdict(list)
    for destination, path in planned.items():
        data = path.read_bytes()
        digest = sha256(data)
        if path.suffix.lower() == ".pdf":
            pdf_pages(data, str(path))
        source_path = path.relative_to(source).as_posix()
        source_by_hash[digest].append(source_path)
        # Identical PDFs already present elsewhere are retained once, unless an
        # existing different-path duplicate already predates this build.
        if path.suffix.lower() == ".pdf" and digest in existing_by_hash and not (output / destination).exists():
            continue
        write_if_missing(output / destination, data)
        if sha256((output / destination).read_bytes()) != digest:
            raise ValueError(f"Delivery checksum mismatch: {destination}")
        if path.suffix.lower() == ".pdf":
            existing_by_hash.setdefault(digest, destination)

    rows: list[dict] = []
    for directory in (output / "技术PDF", output / "面试"):
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(output).as_posix()
            allowed_note = path.name in PUBLIC_NOTES and path.suffix == ".md"
            if path.suffix.lower() != ".pdf" and not allowed_note:
                # Existing unrelated files stay on disk, but never enter the ZIP.
                continue
            if path.is_symlink():
                raise ValueError(f"Symlink delivery file is not supported: {path}")
            data = path.read_bytes()
            digest = sha256(data)
            sources = source_by_hash.get(digest, [])
            if not sources:
                raise ValueError(f"Delivery file has no verified source in this build: {relative}")
            row = {
                "path": relative,
                "source_path": "小红书商品/" + sources[0],
                "source_paths": ["小红书商品/" + name for name in sources],
                "sha256": digest,
                "bytes": len(data),
                "kind": "pdf" if path.suffix.lower() == ".pdf" else "reading_note",
                "pages": pdf_pages(data, relative) if path.suffix.lower() == ".pdf" else 0,
                "content_role": role_for(relative) if path.suffix.lower() == ".pdf" else "来源说明",
            }
            if sources[0] in provenance:
                row["original_source"] = "小红书商品/" + provenance[sources[0]]
            rows.append(row)
    rows.sort(key=lambda row: row["path"])
    notes = reading_notes(rows, args.date, output.name)
    notes_path = output / "00_资料目录.md"
    write_generated(notes_path, notes.encode("utf-8"))
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
                # Fixed timestamps and relative names make identical builds
                # reproducible and allow extraction into the chosen directory.
                info = zipfile.ZipInfo(path.relative_to(output).as_posix(), (*date.fromisoformat(args.date).timetuple()[:3], 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                bundle.writestr(info, path.read_bytes())
        with zipfile.ZipFile(temporary_archive) as bundle:
            expected = {path.relative_to(output).as_posix() for path in archive_files}
            if bundle.testzip() is not None or set(bundle.namelist()) != expected:
                raise ValueError("Archive membership or CRC verification failed")
            for row in rows:
                if sha256(bundle.read(row["path"])) != row["sha256"]:
                    raise ValueError(f"Archive checksum mismatch: {row['path']}")
        backup_if_different(archive, sha256(temporary_archive.read_bytes()))
        temporary_archive.replace(archive)
    finally:
        temporary_archive.unlink(missing_ok=True)

    manifest = {
        "schema_version": 1,
        "date": args.date,
        "source_root": "小红书商品",
        "delivery_directory": output.name,
        "archive": archive.name,
        "archive_sha256": sha256(archive.read_bytes()),
        "statistics_method": "PDF files grouped by identical SHA-256; unique pages summed once per hash. Similar topics are not treated as identical files.",
        "summary": summary,
        "reading_note": {"path": "00_资料目录.md", "sha256": sha256(notes.encode("utf-8"))},
        "files": rows,
    }
    write_generated(manifest_path, (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"output": str(output), "archive": str(archive), "manifest": str(manifest_path), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, zipfile.BadZipFile) as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
