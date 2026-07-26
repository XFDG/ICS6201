#!/usr/bin/env python3
"""Render the repository pipeline diagram from the accompanying CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["key"]: row for row in rows}


def box(ax, x, y, text, *, width=0.13, height=0.16, color="#e8f1fb"):
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=color, edgecolor="#315a7d", linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=9, wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops={"arrowstyle": "->", "color": "#315a7d", "lw": 1.5},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.csv)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    ax.set_title("Training pipeline: GSM8K train only", fontsize=14, fontweight="bold")
    box(ax, 0.075, 0.52, f"GSM8K train\n{rows['GSM8K train']['value']} problems\n+ token-weight asset", width=0.14, color="#fef3d6")
    stages = [
        (0.25, "Stage 1\nStructured CoT SFT"),
        (0.43, "Stage 2\nToken-weighted SFT"),
        (0.61, "DPO data\n5 samples / question\ncorrect vs wrong"),
        (0.79, "Stage 3\nDPO"),
        (0.925, "Final\nLoRA adapter"),
    ]
    for x, label in stages:
        box(ax, x, 0.52, label)
    for left, right in zip([0.075] + [x for x, _ in stages[:-1]], [x for x, _ in stages]):
        arrow(ax, left + 0.07, 0.52, right - 0.06, 0.52)
    ax.text(0.5, 0.18, "Safety checks: finite loss -> finite gradients -> finite trainable parameters -> finite evaluation loss",
            ha="center", va="center", fontsize=10, color="#6b2f2f")

    ax = axes[1]
    ax.set_title("Held-out evaluation and metric meaning", fontsize=14, fontweight="bold")
    box(ax, 0.09, 0.68, f"GSM8K test\n{rows['GSM8K test']['value']} problems", color="#e9f8ec")
    box(ax, 0.09, 0.30, f"SVAMP test\n{rows['SVAMP test']['value']} problems", color="#e9f8ec")
    box(ax, 0.28, 0.50, "Four-section\nreasoning prompt", color="#fef3d6")
    box(ax, 0.48, 0.68, "Greedy x1\nT=0", color="#e8f1fb")
    box(ax, 0.48, 0.30, "Sampled x5\nT=0.8, top-p=0.95", color="#e8f1fb")
    box(ax, 0.68, 0.50, "Parse final\ninteger answer", color="#f5eafa")
    box(ax, 0.88, 0.68, "greedy\npass@1")
    box(ax, 0.88, 0.30, "sampled pass@1\nand pass@5")
    arrow(ax, 0.16, 0.65, 0.21, 0.54)
    arrow(ax, 0.16, 0.34, 0.21, 0.46)
    arrow(ax, 0.355, 0.55, 0.405, 0.66)
    arrow(ax, 0.355, 0.45, 0.405, 0.32)
    arrow(ax, 0.555, 0.65, 0.60, 0.53)
    arrow(ax, 0.555, 0.35, 0.60, 0.47)
    arrow(ax, 0.755, 0.53, 0.805, 0.65)
    arrow(ax, 0.755, 0.47, 0.805, 0.33)
    ax.text(0.5, 0.08, "No test questions enter Stage 1, Stage 2, DPO construction, or model selection.",
            ha="center", va="center", fontsize=10, color="#216b3b")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
