#!/usr/bin/env python3
"""Generate the public one-step evidence chart for the TP=2 graph hang."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ASSET_DIR = Path(__file__).resolve().parent
DATA_PATH = ASSET_DIR / "vllm_tp2_rootcause_matrix_2026-06-25.csv"
OUTPUT_PATH = ASSET_DIR / "vllm_tp2_rootcause_matrix_2026-06-25.png"

COLORS = {
    "PASS": "#2A9D8F",
    "HANG": "#E76F51",
}


def load_rows() -> list[dict[str, str]]:
    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = load_rows()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
        }
    )

    fig, (ax_matrix, ax_latency) = plt.subplots(
        1,
        2,
        figsize=(15.2, 7.2),
        gridspec_kw={"width_ratios": [1.7, 1]},
    )
    fig.suptitle(
        "vLLM TP=2 FULL CUDA Graph Candidate-Path Isolation",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    ordered = list(reversed(rows))
    y_positions = list(range(len(ordered)))
    for y, row in zip(y_positions, ordered):
        color = COLORS[row["result"]]
        ax_matrix.barh(y, 1, color=color, alpha=0.9, height=0.66)
        ax_matrix.text(
            0.03,
            y,
            row["result"],
            va="center",
            ha="left",
            color="white",
            fontweight="bold",
        )
        ax_matrix.text(
            1.04,
            y,
            row["configuration"],
            va="center",
            ha="left",
            fontsize=9.2,
        )

    ax_matrix.set_yticks(y_positions, [row["id"] for row in ordered])
    ax_matrix.set_xlim(0, 2.75)
    ax_matrix.set_xticks([])
    ax_matrix.set_title("A. One-step evidence matrix", loc="left")
    ax_matrix.set_xlabel(
        "P1 excludes the sampler as first cause; F0 points to the fused path.",
        labelpad=12,
    )
    for spine in ax_matrix.spines.values():
        spine.set_visible(False)
    ax_matrix.legend(
        handles=[
            Patch(facecolor=COLORS["PASS"], label="PASS"),
            Patch(facecolor=COLORS["HANG"], label="HANG"),
        ],
        loc="lower right",
        frameon=False,
    )

    pass_rows = [row for row in rows if row["gen_seconds"]]
    labels = [row["id"] for row in pass_rows]
    values = [float(row["gen_seconds"]) for row in pass_rows]
    bars = ax_latency.bar(
        labels,
        values,
        color=["#457B9D", "#2A9D8F", "#264653"],
        width=0.62,
    )
    ax_latency.bar_label(
        bars,
        labels=[f"{value:.2f}s" for value in values],
        padding=4,
    )
    ax_latency.set_ylim(0, max(values) * 1.22)
    ax_latency.set_ylabel("Generation time (seconds)")
    ax_latency.set_title("B. Passing-run generation time", loc="left")
    ax_latency.grid(axis="y", alpha=0.25)
    ax_latency.spines["top"].set_visible(False)
    ax_latency.spines["right"].set_visible(False)
    ax_latency.text(
        0.5,
        -0.13,
        "F0 and FIX passed once; repeat regression and focused NCU remain required.",
        transform=ax_latency.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
    )

    fig.text(
        0.01,
        0.012,
        "Scope: Qwen3-30B-A3B, single-node 8xH200, TP=2, one training step.",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
