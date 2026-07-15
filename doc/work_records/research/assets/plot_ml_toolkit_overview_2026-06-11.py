#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ml_toolkit_overview_data_2026-06-11.csv"
OUT = ROOT / "ml_toolkit_overview_2026-06-11.png"


def read_rows():
    with DATA.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    rows = read_rows()
    repo = [r for r in rows if r["section"] == "repo"]
    r3 = [r for r in rows if r["section"] == "r3_readiness"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    ax = axes[0]
    labels = [r["item"].replace("_", "\n") for r in repo]
    values = [float(r["value"]) for r in repo]
    colors = ["#386641", "#6A994E", "#A7C957", "#BC4749", "#4B6CB7"]
    ax.bar(range(len(values)), values, color=colors[: len(values)], width=0.65)
    ax.set_title("Repository Footprint")
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.03, f"{int(v)}", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    labels = [r["item"].replace("_", "\n") for r in r3]
    values = [float(r["value"]) for r in r3]
    colors = ["#2E7D5B" if v >= 3 else "#DDA15E" if v >= 1 else "#C44536" for v in values]
    ax.barh(range(len(values)), values, color=colors, height=0.58)
    ax.set_title("R3 Utility Readiness")
    ax.set_xlabel("0 missing  |  1 planned  |  2 partial  |  3 ready")
    ax.set_xlim(0, 3.2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    for i, v in enumerate(values):
        ax.text(v + 0.05, i, f"{v:.0f}", va="center", fontsize=9)
    ax.invert_yaxis()

    fig.suptitle("ml_toolkit: analysis/aligner base with pending R3-specific layer", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT, dpi=180)


if __name__ == "__main__":
    main()
