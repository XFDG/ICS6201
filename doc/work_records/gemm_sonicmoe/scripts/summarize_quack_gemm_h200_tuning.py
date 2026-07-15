"""Summarize quack GEMM H200 tuning CSV into compact Markdown tables."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(DATA_DIR / "quack_gemm_h200_tuning_extended_target_2026-06-04.csv"),
    )
    return parser.parse_args()


def to_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.inf


def main() -> None:
    args = parse_args()
    rows = []
    with Path(args.csv).open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["ms"] = to_float(row["ms"])
            row["mfu_percent"] = to_float(row["mfu_percent"])
            rows.append(row)

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["case"], row["op"])].append(row)

    print("| case | op | valid / total | best ms | best MFU | best config | second best ms | gap |")
    print("|---|---:|---:|---:|---:|---|---:|---:|")
    for key in sorted(groups):
        group = groups[key]
        valid = sorted([row for row in group if row["status"] == "ok"], key=lambda r: r["ms"])
        if not valid:
            print(f"| {key[0]} | {key[1]} | 0 / {len(group)} | - | - | - | - | - |")
            continue
        best = valid[0]
        second = valid[1] if len(valid) > 1 else None
        second_ms = "-" if second is None else f"{second['ms']:.4f}"
        gap = "-" if second is None else f"{(second['ms'] / best['ms'] - 1) * 100:.2f}%"
        print(
            f"| {key[0]} | {key[1]} | {len(valid)} / {len(group)} | "
            f"{best['ms']:.4f} | {best['mfu_percent']:.1f}% | `{best['config_label']}` | "
            f"{second_ms} | {gap} |"
        )

    print("\nFailed configs:")
    failed = [row for row in rows if row["status"] != "ok"]
    if not failed:
        print("- none")
    else:
        for row in failed[:20]:
            print(f"- {row['case']} {row['op']} `{row['config_label']}`: {row['error']}")
        if len(failed) > 20:
            print(f"- ... {len(failed) - 20} more")


if __name__ == "__main__":
    main()
