from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "mirage_overview_data_2026-06-14.csv"
OUT = ROOT / "mirage_overview_2026-06-14.png"

df = pd.read_csv(DATA)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Mirage / MPK overview: code surface, task coverage, runnable surface, and reported speedups", fontsize=14)

code = df[df["section"].eq("code")]
axes[0, 0].barh(code["item"], code["value"], color="#3b82f6")
axes[0, 0].set_title("Code Surface")
axes[0, 0].set_xlabel("file count")
axes[0, 0].invert_yaxis()

tasks = df[df["section"].eq("mpk_tasks")]
axes[0, 1].barh(tasks["item"], tasks["value"], color="#10b981")
axes[0, 1].set_title("MPK Task Header Coverage")
axes[0, 1].set_xlabel("file count")
axes[0, 1].invert_yaxis()

surface = df[df["section"].eq("surface")]
axes[1, 0].barh(surface["item"], surface["value"], color="#f59e0b")
axes[1, 0].set_title("Demos, Benchmarks, Tests")
axes[1, 0].set_xlabel("file count")
axes[1, 0].invert_yaxis()

perf_order = [
    "MPK README max latency reduction",
    "MPK README min latency reduction",
    "MPK arXiv max speedup",
    "Mirage superoptimizer max speedup",
    "Mirage superoptimizer min speedup",
    "LoRA tutorial speedup",
    "local HEAD MLA gather speedup",
]
perf = df[df["section"].eq("performance")].set_index("item").loc[perf_order].reset_index()
colors = ["#ef4444", "#ef4444", "#6366f1", "#8b5cf6", "#8b5cf6", "#22c55e", "#14b8a6"]
axes[1, 1].barh(perf["item"], perf["value"], color=colors)
axes[1, 1].set_title("Reported Speedups / Latency Reductions")
axes[1, 1].set_xlabel("x")
axes[1, 1].invert_yaxis()

for ax in axes.flat:
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT, dpi=180)
print(OUT)
