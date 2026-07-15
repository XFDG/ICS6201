from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


DATA_PATH = Path(__file__).with_name("openmanus_overview_data_2026-06-15.csv")
OUT_PATH = Path(__file__).with_name("openmanus_overview_2026-06-15.png")


def load_rows() -> list[dict[str, str]]:
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    rows = load_rows()
    panels = ["Code Surface", "Agent and Tools", "Runtime Config", "Execution Model"]
    colors = ["#0f766e", "#2563eb", "#9333ea", "#ea580c"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("OpenManus Local Repository Overview", fontsize=16, fontweight="bold")

    for ax, panel, color in zip(axes.ravel(), panels, colors):
        panel_rows = [row for row in rows if row["panel"] == panel]
        metrics = [row["metric"] for row in panel_rows]
        values = [int(row["value"]) for row in panel_rows]
        y_positions = range(len(metrics))

        ax.barh(list(y_positions), values, color=color, alpha=0.88)
        ax.set_yticks(list(y_positions), labels=metrics)
        ax.invert_yaxis()
        ax.set_title(panel)
        ax.set_xlabel("Count")
        ax.grid(axis="x", linestyle="--", alpha=0.28)

        for y, value in zip(y_positions, values):
            ax.text(value + max(values) * 0.02, y, str(value), va="center", fontsize=9)

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_PATH, dpi=180)


if __name__ == "__main__":
    main()
