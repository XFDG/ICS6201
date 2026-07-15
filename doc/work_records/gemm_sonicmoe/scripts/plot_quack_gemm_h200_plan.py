"""Plot the 2026-06-05 quack GEMM H200 plan results."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


H = 3072
I = 1536
H200_BF16_PEAK_TFLOPS = 989.0
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SET_ORDER = ("before", "current_after", "expanded_after")
SET_LABELS = {
    "before": "before",
    "current_after": "current",
    "expanded_after": "expanded",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-csv",
        default=str(DATA_DIR / "quack_gemm_h200_plan_summary_2026-06-05.csv"),
    )
    parser.add_argument(
        "--out-png",
        default=str(Path(__file__).resolve().parent / "quack_gemm_h200_plan_2026-06-05_summary.png"),
    )
    parser.add_argument(
        "--out-aggregate-csv",
        default=str(DATA_DIR / "quack_gemm_h200_plan_aggregate_2026-06-05.csv"),
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            for key in ("tokens", "routed_m", "skew_ratio", "best_ms", "mfu_percent"):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (row["case"], row["distribution"], row["candidate_set"])
        grouped[key][row["op"]] = row

    agg: list[dict] = []
    for (case, distribution, candidate_set), ops in grouped.items():
        gated = ops["gated"]
        out = ops["out"]
        total_ms = gated["best_ms"] + out["best_ms"]
        routed_m = gated["routed_m"]
        flops = 6.0 * routed_m * H * I
        mfu = flops / (total_ms / 1000.0) / (H200_BF16_PEAK_TFLOPS * 1e12) * 100.0
        agg.append(
            {
                "case": case,
                "distribution": distribution,
                "candidate_set": candidate_set,
                "tokens": gated["tokens"],
                "routed_m": routed_m,
                "skew_ratio": gated["skew_ratio"],
                "total_ms": total_ms,
                "weighted_mfu_percent": mfu,
                "gated_ms": gated["best_ms"],
                "out_ms": out["best_ms"],
                "gated_config": gated["best_config"],
                "out_config": out["best_config"],
            }
        )
    agg.sort(key=lambda item: (item["distribution"], item["routed_m"], SET_ORDER.index(item["candidate_set"])))
    return agg


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def by_key(agg: list[dict]) -> dict[tuple, dict]:
    return {(row["case"], row["distribution"], row["candidate_set"]): row for row in agg}


def style_axes(ax) -> None:
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot(agg: list[dict], out_path: Path) -> None:
    lookup = by_key(agg)
    random_rows = [row for row in agg if row["distribution"] == "random" and row["candidate_set"] == "before"]
    cases = [row["case"] for row in sorted(random_rows, key=lambda row: row["routed_m"])]
    x = np.arange(len(cases))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)

    ax = axes[0, 0]
    for set_name in SET_ORDER:
        values = [lookup[(case, "random", set_name)]["total_ms"] for case in cases]
        ax.plot(x, values, marker="o", linewidth=1.8, label=SET_LABELS[set_name])
    ax.set_title("Random Distribution: Two-GEMM Latency")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=45, ha="right", fontsize=8)
    ax.legend(frameon=False)
    style_axes(ax)

    ax = axes[0, 1]
    before = [lookup[(case, "random", "before")]["total_ms"] for case in cases]
    for set_name in ("current_after", "expanded_after"):
        values = [lookup[(case, "random", set_name)]["total_ms"] for case in cases]
        speedups = [(b / v - 1.0) * 100.0 for b, v in zip(before, values)]
        ax.plot(x, speedups, marker="o", linewidth=1.8, label=SET_LABELS[set_name])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Random Distribution: Speedup vs Before")
    ax.set_ylabel("Speedup (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=45, ha="right", fontsize=8)
    ax.legend(frameon=False)
    style_axes(ax)

    ax = axes[1, 0]
    distributions = sorted({row["distribution"] for row in agg})
    width = 0.24
    dist_x = np.arange(len(distributions))
    for offset, set_name in zip((-width, 0, width), SET_ORDER):
        values = []
        for dist in distributions:
            values.append(sum(row["total_ms"] for row in agg if row["distribution"] == dist and row["candidate_set"] == set_name))
        ax.bar(dist_x + offset, values, width, label=SET_LABELS[set_name])
    ax.set_title("All 14 Cases: Total Latency by Expert Distribution")
    ax.set_ylabel("Latency sum (ms)")
    ax.set_xticks(dist_x)
    ax.set_xticklabels(distributions)
    ax.legend(frameon=False)
    style_axes(ax)

    ax = axes[1, 1]
    for set_name in SET_ORDER:
        values = []
        for dist in distributions:
            rows = [row for row in agg if row["distribution"] == dist and row["candidate_set"] == set_name]
            total_flops = sum(6.0 * row["routed_m"] * H * I for row in rows)
            total_ms = sum(row["total_ms"] for row in rows)
            values.append(total_flops / (total_ms / 1000.0) / (H200_BF16_PEAK_TFLOPS * 1e12) * 100.0)
        ax.plot(dist_x, values, marker="o", linewidth=1.8, label=SET_LABELS[set_name])
    ax.set_title("All 14 Cases: Weighted MFU by Distribution")
    ax.set_ylabel("MFU (%)")
    ax.set_xticks(dist_x)
    ax.set_xticklabels(distributions)
    ax.legend(frameon=False)
    style_axes(ax)

    fig.suptitle("Quack GEMM H200 Follow-up Plan Results", fontsize=15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.summary_csv))
    agg = aggregate(rows)
    write_csv(Path(args.out_aggregate_csv), agg)
    plot(agg, Path(args.out_png))
    print(args.out_aggregate_csv)
    print(args.out_png)

    config_counts = Counter(
        (row["candidate_set"], row["gated_config"])
        for row in agg
        if row["candidate_set"] != "before"
    )
    print("Top gated configs:")
    for (set_name, config), count in config_counts.most_common(10):
        print(f"{set_name:14s} {count:3d} {config}")


if __name__ == "__main__":
    main()
