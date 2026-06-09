#!/usr/bin/env python3
"""结果汇总 — 从 state 文件和 runs 目录提取指标，生成对比报告。"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]


def read_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"items": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def read_results_csv_last_row(path: Path) -> Dict[str, float]:
    import csv
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            last = None
            for row in reader:
                last = row
        if not last:
            return {}
        out: Dict[str, float] = {}
        for k, v in last.items():
            try:
                out[k.strip()] = float(v)
            except (ValueError, TypeError):
                pass
        return out
    except Exception:
        return {}


def find_runs_dirs(runs_root: Path, prefix: str = "") -> List[Path]:
    if not runs_root.exists():
        return []
    return sorted([d for d in runs_root.iterdir() if d.is_dir() and d.name.startswith(prefix)])


def merge_metrics(runs_dirs: List[Path]) -> Dict[str, Dict[str, Any]]:
    """从 runs 目录读取每个 run 的 args.yaml 和 results.csv 提取关键指标"""
    import yaml
    out: Dict[str, Dict[str, Any]] = {}
    for d in runs_dirs:
        name = d.name
        info: Dict[str, Any] = {"run_dir": str(d)}

        args_yaml = d / "args.yaml"
        if args_yaml.exists():
            try:
                args = yaml.safe_load(args_yaml.read_text(encoding="utf-8"))
                if isinstance(args, dict):
                    info["epochs"] = args.get("epochs", "?")
                    info["model"] = args.get("model", name)
                    info["batch"] = args.get("batch", "?")
                    info["imgsz"] = args.get("imgsz", "?")
            except Exception:
                pass

        csv_data = read_results_csv_last_row(d / "results.csv")
        for metric in ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"]:
            if metric in csv_data:
                info[metric.replace("metrics/", "").replace("(B)", "")] = round(csv_data[metric], 4)

        best_pt = d / "weights" / "best.pt"
        if best_pt.exists():
            info["best_pt_size_mb"] = round(best_pt.stat().st_size / (1024 * 1024), 1)

        out[name] = info
    return out


def build_comparison_table(
    state: Dict[str, Any],
    runs_info: Dict[str, Dict[str, Any]],
) -> str:
    items = state.get("items", {})
    summary = state.get("summary", {})

    lines = [
        "# Training Results Summary",
        "",
        f"**Date**: {summary.get('updated_at', summary.get('completed_at', 'N/A'))}",
        f"**Mode**: {state.get('mode', 'N/A')}",
        f"**Result**: {summary.get('ok', 0)} OK / {summary.get('failed', 0)} Failed / {summary.get('skipped', 0)} Skipped",
        "",
    ]

    # 如果有 runs 目录，生成详细表
    model_results: Dict[str, Dict[str, List[float]]] = {}
    for key, item in sorted(items.items()):
        task = item.get("task", key)
        status = item.get("status", "?")
        if status != "ok":
            continue

        model_name = task.split("_seed")[0] if "_seed" in task else task
        seed = item.get("seed", "?")

        if model_name not in model_results:
            model_results[model_name] = {"seeds": [], "mAP50": [], "mAP50-95": []}

        model_results[model_name]["seeds"].append(str(seed))

        ri = runs_info.get(key) or runs_info.get(f"validate_{key}")
        if ri:
            m50 = ri.get("mAP50(B)")
            m5095 = ri.get("mAP50-95(B)")
            if m50 is not None:
                model_results[model_name]["mAP50"].append(m50)
            if m5095 is not None:
                model_results[model_name]["mAP50-95"].append(m5095)

    if any(v["mAP50"] for v in model_results.values()):
        lines.extend([
            "## Model Comparison",
            "",
            "| Model | Seeds | mAP@50 | mAP@50-95 | Params | Speed |",
            "|-------|-------|--------|-----------|--------|-------|",
        ])
        for model_name, data in sorted(model_results.items()):
            seeds = ",".join(data["seeds"])
            m50s = data["mAP50"]
            m5095s = data["mAP50-95"]

            def fmt(vals):
                if not vals:
                    return "—"
                mu = sum(vals) / len(vals)
                if len(vals) > 1:
                    import math
                    std = math.sqrt(sum((x - mu) ** 2 for x in vals) / (len(vals) - 1))
                    return f"{mu:.4f} ± {std:.4f}"
                return f"{mu:.4f}"

            lines.append(f"| {model_name} | {seeds} | {fmt(m50s)} | {fmt(m5095s)} | — | — |")

    # 详细列表
    lines.extend([
        "",
        "## Per-Run Details",
        "",
        "| Run | Status | GPU | Elapsed | Epochs | mAP@50 | mAP@50-95 | Best PT |",
        "|-----|--------|-----|---------|--------|--------|-----------|---------|",
    ])

    for key, item in sorted(items.items()):
        status = item.get("status", "?")
        gpu = item.get("gpu", "?")
        elapsed = item.get("elapsed_sec", "?")
        epochs = item.get("epochs", "?")

        ri = runs_info.get(key, {})
        m50 = ri.get("mAP50(B)", "—")
        m5095 = ri.get("mAP50-95(B)", "—")
        best_size = ri.get("best_pt_size_mb", "—")

        elapsed_str = f"{elapsed}s" if isinstance(elapsed, (int, float)) else str(elapsed)
        lines.append(f"| {key} | {status} | {gpu} | {elapsed_str} | {epochs} | {m50} | {m5095} | {best_size}MB |")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "formal", "all"], default="all")
    p.add_argument("--out", default=str(ROOT / "training_report.md"))
    args = p.parse_args()

    output: List[str] = []
    output.append("# ICS6201 Training Report\n")

    for mode in ["smoke", "formal"]:
        if args.mode not in (mode, "all"):
            continue
        state_path = ROOT / "logs" / f"state_{mode}.json"
        if not state_path.exists():
            continue

        state = read_state(state_path)
        runs_root = ROOT / f"runs_{mode}"
        runs_dirs = find_runs_dirs(runs_root)
        runs_info = merge_metrics(runs_dirs)

        section = build_comparison_table(state, runs_info)
        output.append(section)

    if len(output) == 1:
        output.append("No training results found.\n")

    report = "\n---\n".join(output)
    out_path = Path(args.out)
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
