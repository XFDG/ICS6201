"""Plot quack GEMM H200 before/after benchmark results.

The chart layout follows the local report convention: use 1x2 combined PNGs
instead of many standalone single-chart images.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(DATA_DIR / "quack_gemm_h200_before_after_more_inputs_2026-06-04.csv"),
    )
    parser.add_argument(
        "--out-prefix",
        default=str(Path(__file__).resolve().parent / "quack_gemm_h200_tuning"),
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            for key in (
                "tokens",
                "routed_m",
                "before_ms",
                "after_ms",
                "improvement_percent",
                "before_mfu_percent",
                "after_mfu_percent",
            ):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    by_case: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_case[row["case"]][row["op"]] = row

    agg = []
    for case, ops in by_case.items():
        gated = ops["gated"]
        out = ops["out"]
        before_sum = gated["before_ms"] + out["before_ms"]
        after_sum = gated["after_ms"] + out["after_ms"]
        speedup = before_sum / after_sum
        # Both rows carry the same T/TK.
        agg.append(
            {
                "case": case,
                "tokens": gated["tokens"],
                "routed_m": gated["routed_m"],
                "before_sum_ms": before_sum,
                "after_sum_ms": after_sum,
                "sum_improvement_percent": (speedup - 1.0) * 100.0,
                "gated_before_ms": gated["before_ms"],
                "gated_after_ms": gated["after_ms"],
                "out_before_ms": out["before_ms"],
                "out_after_ms": out["after_ms"],
                "gated_gain_percent": gated["improvement_percent"],
                "out_gain_percent": out["improvement_percent"],
                "before_mfu_percent": weighted_mfu(gated, out, "before"),
                "after_mfu_percent": weighted_mfu(gated, out, "after"),
            }
        )
    agg.sort(key=lambda item: item["routed_m"])
    return agg


def weighted_mfu(gated: dict, out: dict, prefix: str) -> float:
    # For these two GEMMs, gated FLOPs are 2x out FLOPs, so weighted MFU can be
    # reconstructed from per-op MFUs and timings via total work / total time.
    gated_ms = gated[f"{prefix}_ms"]
    out_ms = out[f"{prefix}_ms"]
    # Use normalized work units: gated=2, out=1 at the same TK/H/I.
    effective_peak_ms = 2.0 / (gated[f"{prefix}_mfu_percent"] / 100.0) + 1.0 / (
        out[f"{prefix}_mfu_percent"] / 100.0
    )
    total_work = 3.0
    # The expression above is the time in normalized peak units; this converts
    # back to percent. It is equivalent to total FLOPs / total elapsed / peak.
    return total_work / effective_peak_ms * 100.0


def style_axes(ax) -> None:
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_latency_speedup(agg: list[dict], out_path: Path) -> None:
    labels = [item["case"] for item in agg]
    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    ax = axes[0]
    ax.bar(x - width / 2, [item["before_sum_ms"] for item in agg], width, label="before")
    ax.bar(x + width / 2, [item["after_sum_ms"] for item in agg], width, label="after")
    ax.set_title("Two-GEMM Total Latency")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend(frameon=False)
    style_axes(ax)

    ax = axes[1]
    gains = [item["sum_improvement_percent"] for item in agg]
    colors = ["#2a9d8f" if gain >= 0 else "#d62828" for gain in gains]
    ax.bar(x, gains, color=colors)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("After vs Before Speedup")
    ax.set_ylabel("Speedup (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    style_axes(ax)

    fig.suptitle("Quack GEMM H200 Before / After", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_mfu_breakdown(agg: list[dict], out_path: Path) -> None:
    labels = [item["case"] for item in agg]
    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    ax = axes[0]
    ax.plot(x, [item["before_mfu_percent"] for item in agg], marker="o", label="before")
    ax.plot(x, [item["after_mfu_percent"] for item in agg], marker="o", label="after")
    ax.set_title("Weighted MFU of Two GEMMs")
    ax.set_ylabel("MFU (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend(frameon=False)
    style_axes(ax)

    ax = axes[1]
    gated_gain = [item["gated_gain_percent"] for item in agg]
    out_gain = [item["out_gain_percent"] for item in agg]
    ax.bar(x - width / 2, gated_gain, width, label="gated")
    ax.bar(x + width / 2, out_gain, width, label="out")
    # Zero-height bars are visually invisible. Add small baseline markers so
    # readers can tell "0% speedup" apart from missing data.
    gated_zero_x = [x[i] - width / 2 for i, gain in enumerate(gated_gain) if abs(gain) < 1e-9]
    out_zero_x = [x[i] + width / 2 for i, gain in enumerate(out_gain) if abs(gain) < 1e-9]
    ax.scatter(gated_zero_x, [0] * len(gated_zero_x), marker="o", s=18, color="#1f77b4", zorder=4)
    ax.scatter(out_zero_x, [0] * len(out_zero_x), marker="o", s=18, color="#ff7f0e", zorder=4)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Where the Speedup Comes From")
    ax.set_ylabel("Op speedup (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend(frameon=False)
    style_axes(ax)

    fig.suptitle("Quack GEMM H200 MFU / Op Breakdown", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.csv))
    agg = aggregate(rows)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    latency_path = prefix.with_name(prefix.name + "_latency_speedup.png")
    mfu_path = prefix.with_name(prefix.name + "_mfu_breakdown.png")
    plot_latency_speedup(agg, latency_path)
    plot_mfu_breakdown(agg, mfu_path)
    print(latency_path)
    print(mfu_path)


if __name__ == "__main__":
    main()
