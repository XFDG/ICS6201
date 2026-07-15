from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "frontier_overview_data_2026-06-10.csv"
OUT = ROOT / "frontier_overview_2026-06-10.png"


df = pd.read_csv(DATA)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Frontier overview: code surface, assets, examples, and reported fidelity", fontsize=15)

coverage = df[df["section"].eq("coverage")]
axes[0, 0].barh(coverage["item"], coverage["value"], color="#3b82f6")
axes[0, 0].set_title("Code Surface")
axes[0, 0].set_xlabel("file count")
axes[0, 0].invert_yaxis()

assets = df[df["section"].eq("assets")]
axes[0, 1].barh(assets["item"], assets["value"], color="#10b981")
axes[0, 1].set_title("Checked-in Config/Profile Assets")
axes[0, 1].set_xlabel("file count")
axes[0, 1].invert_yaxis()

examples = df[df["section"].eq("examples")]
axes[1, 0].barh(examples["item"], examples["value"], color="#f59e0b")
axes[1, 0].set_title("Runnable Surface")
axes[1, 0].set_xlabel("file count")
axes[1, 0].invert_yaxis()

paper_items = [
    "co-location baseline error",
    "co-location latency error",
    "disaggregation baseline error",
    "disaggregation latency error",
    "throughput mean error",
]
paper = df[df["section"].eq("paper")].set_index("item").loc[paper_items].reset_index()
colors = ["#ef4444", "#22c55e", "#ef4444", "#22c55e", "#6366f1"]
axes[1, 1].barh(paper["item"], paper["value"], color=colors)
axes[1, 1].set_title("Paper-reported Error Metrics")
axes[1, 1].set_xlabel("error (%)")
axes[1, 1].invert_yaxis()
for ax in axes.flat:
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT, dpi=180)
print(OUT)
