#!/usr/bin/env python3
"""Generate a recovery manifest for ICS6201 training runs.

Scans Ultralytics runs (``runs_formal/*``) and Detectron2 runs
(``runs_detectron2/*``), picks a canonical directory per logical task, and
records its progress, checkpoint paths, and complete/resume status. The
manifest is consumed by ``parallel_launcher.py`` to drive recovery.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


PRIMARY_FAMILIES = {"rtdetr", "faster_rcnn"}
SECONDARY_FAMILIES = {"yolo11", "ddw_yolo", "yolov10", "yolov8"}

DEFAULT_TARGET_EPOCHS = {
    "rtdetr": 120,
    "yolo11": 200,
    "ddw_yolo": 200,
    "yolov10": 200,
    "yolov8": 200,
}


# matches "rtdetr_seed1", "rtdetr_seed1-2", "yolo11_seed2-3"
ULTRA_DIR_RE = re.compile(r"^(?P<family>[a-zA-Z0-9]+(?:_[a-zA-Z]+)?)_seed(?P<seed>\d+)(?:-(?P<suffix>\d+))?$")
# matches "faster_rcnn_seed1", "faster_rcnn_seed1-2"
D2_DIR_RE = re.compile(r"^faster_rcnn_seed(?P<seed>\d+)(?:-(?P<suffix>\d+))?$")


def _read_last_epoch(results_csv: Path) -> Optional[int]:
    if not results_csv.exists():
        return None
    try:
        with open(results_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            last: Optional[Dict[str, str]] = None
            for row in reader:
                last = row
        if not last:
            return None
        for k in ("epoch", "Epoch", "EPOCH"):
            if k in last:
                try:
                    return int(float(str(last[k]).strip()))
                except Exception:
                    return None
        return None
    except Exception:
        return None


def _last_metrics_iter(metrics_json: Path) -> Optional[int]:
    if not metrics_json.exists():
        return None
    last_iter: Optional[int] = None
    try:
        with open(metrics_json, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if "iteration" in obj:
                    try:
                        last_iter = int(obj["iteration"])
                    except Exception:
                        continue
        return last_iter
    except Exception:
        return None


def _last_checkpoint(d: Path) -> Optional[Path]:
    pointer = d / "last_checkpoint"
    if pointer.exists():
        try:
            name = pointer.read_text(encoding="utf-8").strip()
            candidate = d / name
            if candidate.exists():
                return candidate
        except Exception:
            pass
    # Fall back to the newest model_*.pth
    candidates = sorted(d.glob("model_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _scan_ultralytics(runs_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    if not runs_dir.exists():
        return grouped
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        m = ULTRA_DIR_RE.match(d.name)
        if not m:
            continue
        family = m.group("family")
        seed = int(m.group("seed"))
        run_key = f"{family}_seed{seed}"
        results_csv = d / "results.csv"
        last_pt = d / "weights" / "last.pt"
        best_pt = d / "weights" / "best.pt"
        last_epoch = _read_last_epoch(results_csv)
        target_epoch = DEFAULT_TARGET_EPOCHS.get(family, 0)
        entry = {
            "dir": str(d),
            "dir_name": d.name,
            "suffix": int(m.group("suffix")) if m.group("suffix") else 0,
            "mtime": d.stat().st_mtime,
            "results_csv": str(results_csv) if results_csv.exists() else None,
            "last_epoch": last_epoch,
            "target_epoch": target_epoch,
            "last_pt": str(last_pt) if last_pt.exists() else None,
            "best_pt": str(best_pt) if best_pt.exists() else None,
            "family": family,
            "seed": seed,
            "kind": "ultralytics",
        }
        grouped.setdefault(run_key, []).append(entry)
    return grouped


def _scan_detectron2(runs_dir: Path, target_iter: Optional[int]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    if not runs_dir.exists():
        return grouped
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        m = D2_DIR_RE.match(d.name)
        if not m:
            continue
        seed = int(m.group("seed"))
        run_key = f"faster_rcnn_seed{seed}"
        metrics_json = d / "metrics.json"
        last_iter = _last_metrics_iter(metrics_json)
        last_ckpt = _last_checkpoint(d)
        model_final = d / "model_final.pth"
        entry = {
            "dir": str(d),
            "dir_name": d.name,
            "suffix": int(m.group("suffix")) if m.group("suffix") else 0,
            "mtime": d.stat().st_mtime,
            "metrics_json": str(metrics_json) if metrics_json.exists() else None,
            "last_iter": last_iter,
            "target_iter": target_iter,
            "last_checkpoint": str(last_ckpt) if last_ckpt else None,
            "model_final": str(model_final) if model_final.exists() else None,
            "family": "faster_rcnn",
            "seed": seed,
            "kind": "detectron2",
        }
        grouped.setdefault(run_key, []).append(entry)
    return grouped


def _pick_canonical(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Highest progress wins; ties broken by most recent mtime."""

    def progress(e: Dict[str, Any]) -> float:
        if e["kind"] == "ultralytics":
            return float(e.get("last_epoch") or -1)
        return float(e.get("last_iter") or -1)

    sortable = sorted(entries, key=lambda e: (progress(e), e["mtime"]), reverse=True)
    canonical = sortable[0]
    canonical = dict(canonical)
    canonical["_alternatives"] = [
        {"dir": e["dir"], "last_epoch": e.get("last_epoch"), "last_iter": e.get("last_iter"), "mtime": e["mtime"]}
        for e in sortable[1:]
    ]
    return canonical


def _classify_priority(family: str) -> str:
    if family in PRIMARY_FAMILIES:
        return "primary"
    if family in SECONDARY_FAMILIES:
        return "secondary"
    return "unknown"


def _evaluate_ultralytics(canonical: Dict[str, Any]) -> Dict[str, Any]:
    last_epoch = canonical.get("last_epoch")
    target_epoch = canonical.get("target_epoch") or 0
    last_pt = canonical.get("last_pt")
    best_pt = canonical.get("best_pt")
    complete = bool(last_epoch and target_epoch and last_epoch >= target_epoch - 1)
    resume_allowed = bool(last_pt) and not complete
    reason = "complete" if complete else ("has_last_pt" if last_pt else "no_checkpoint")
    return {
        "last_progress": last_epoch,
        "target_progress": target_epoch,
        "progress_kind": "epoch",
        "last_checkpoint": last_pt,
        "best_checkpoint": best_pt,
        "complete": complete,
        "resume_allowed": resume_allowed,
        "reason": reason,
    }


def _evaluate_detectron2(canonical: Dict[str, Any]) -> Dict[str, Any]:
    last_iter = canonical.get("last_iter")
    target_iter = canonical.get("target_iter")
    last_ckpt = canonical.get("last_checkpoint")
    model_final = canonical.get("model_final")
    if model_final:
        complete = True
        reason = "model_final"
    elif target_iter and last_iter and last_iter >= target_iter - 1:
        complete = True
        reason = "max_iter_reached"
    else:
        complete = False
        reason = "has_last_checkpoint" if last_ckpt else "no_checkpoint"
    resume_allowed = bool(last_ckpt) and not complete
    return {
        "last_progress": last_iter,
        "target_progress": target_iter,
        "progress_kind": "iter",
        "last_checkpoint": last_ckpt,
        "best_checkpoint": model_final,
        "complete": complete,
        "resume_allowed": resume_allowed,
        "reason": reason,
    }


def build_manifest(
    runs_dir: Path,
    detectron_dir: Path,
    target: str,
    target_iter: Optional[int],
) -> Dict[str, Any]:
    ultra = _scan_ultralytics(runs_dir)
    d2 = _scan_detectron2(detectron_dir, target_iter)
    items: Dict[str, Any] = {}

    for run_key, entries in {**ultra, **d2}.items():
        canonical = _pick_canonical(entries)
        family = canonical["family"]
        priority = _classify_priority(family)
        if canonical["kind"] == "ultralytics":
            verdict = _evaluate_ultralytics(canonical)
        else:
            verdict = _evaluate_detectron2(canonical)

        items[run_key] = {
            "run_key": run_key,
            "family": family,
            "seed": canonical["seed"],
            "priority": priority,
            "kind": canonical["kind"],
            "canonical_dir": canonical["dir"],
            "canonical_dir_name": canonical["dir_name"],
            "candidate_dirs": [e["dir"] for e in entries],
            "alternatives": canonical.get("_alternatives", []),
            **verdict,
        }

    # Filter by target if requested. We always emit a manifest containing all
    # discovered runs, but mark whether each should be scheduled.
    selected: List[str] = []
    for run_key, item in items.items():
        if target == "all":
            scheduled = not item["complete"]
        elif target == "primary":
            scheduled = item["priority"] == "primary" and not item["complete"]
        elif target == "secondary":
            scheduled = item["priority"] == "secondary" and not item["complete"]
        else:
            scheduled = False
        item["scheduled"] = scheduled
        if scheduled:
            selected.append(run_key)

    return {
        "runs_dir": str(runs_dir),
        "detectron_dir": str(detectron_dir),
        "target": target,
        "selected": sorted(selected),
        "items": items,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs_formal")
    p.add_argument("--detectron-dir", default="runs_detectron2")
    p.add_argument("--target", choices=["primary", "secondary", "all"], default="primary")
    p.add_argument("--detectron-target-iter", type=int, default=0,
                   help="Optional override for Detectron2 target max_iter; 0 means unknown")
    p.add_argument("--out", default="")
    args = p.parse_args()

    target_iter = args.detectron_target_iter or None
    manifest = build_manifest(
        Path(args.runs_dir).resolve(),
        Path(args.detectron_dir).resolve(),
        args.target,
        target_iter,
    )

    blob = json.dumps(manifest, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(blob + "\n", encoding="utf-8")
        print(f"manifest written: {out_path}")
    else:
        print(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
