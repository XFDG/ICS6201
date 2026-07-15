#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "tk_mkernel_overview_data.csv"
OUT = HERE / "tk_mkernel_overview_2026-06-08.png"


def load_rows():
    with DATA.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    rows = load_rows()
    tk_rows = [r for r in rows if r["section"] == "ThunderKittens"]
    mk_rows = [r for r in rows if r["section"] == "mKernel"]

    tk_labels = [
        "L01",
        "L02",
        "L03",
        "L04",
        "L05",
        "L06",
        "L07",
        "L08",
        "L09",
        "Full",
    ]
    tk_vals = [float(r["tk_tflops"]) for r in tk_rows]

    mk_labels = [
        "AG+GEMM",
        "GEMM+AR",
        "Dispatch\n+GEMM",
        "Ring\nAttn",
        "GEMM+RS",
        "MoE FFN\n+Combine",
    ]
    base = np.array([float(r["baseline_tflops"]) for r in mk_rows])
    ours = np.array([float(r["mkernel_tflops"]) for r in mk_rows])
    speedup = ours / base

    xpu_rows = [r for r in rows if r["section"] == "xpu-perf"]
    xpu_labels = ["Basic\nops", "LLM\nops", "Basic\nworkloads", "LLM\nworkloads", "XCCL\nworkloads"]
    xpu_vals = [float(r["xpu_count"]) for r in xpu_rows]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.2))

    ax = axes[0, 0]
    x = np.arange(len(tk_vals))
    ax.plot(x, tk_vals, marker="o", color="#2F6B5F", linewidth=2.2)
    ax.fill_between(x, tk_vals, color="#2F6B5F", alpha=0.12)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(tk_labels)
    ax.set_ylabel("TFLOPS")
    ax.set_title("ThunderKittens B200 GEMM Progression")
    ax.grid(axis="y", alpha=0.25, which="both")
    for xi, yi in zip(x, tk_vals):
        ax.annotate(f"{yi:.0f}", (xi, yi), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.text(0.02, 0.04, "Source: educational_b200 README, M=N=K=4096",
            transform=ax.transAxes, fontsize=8, color="#555555")

    ax = axes[0, 1]
    x = np.arange(len(mk_labels))
    w = 0.36
    ax.bar(x - w / 2, base, w, label="Best published baseline", color="#4C72B0")
    ax.bar(x + w / 2, ours, w, label="mKernel", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(mk_labels)
    ax.set_ylabel("TFLOPS per GPU (bf16)")
    ax.set_title("mKernel EFA Largest-Shape Throughput")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, max(ours) * 1.16)
    for xi, yi in zip(x - w / 2, base):
        ax.annotate(f"{yi:.0f}", (xi, yi), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8)
    for xi, yi, sp in zip(x + w / 2, ours, speedup):
        ax.annotate(f"{yi:.0f}\n{sp:.2f}x", (xi, yi), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8,
                    linespacing=0.95)
    ax.text(0.02, 0.04, "Source: UCCL mKernel blog + local GLU plot script",
            transform=ax.transAxes, fontsize=8, color="#555555")

    ax = axes[1, 0]
    x = np.arange(len(xpu_labels))
    bars = ax.bar(x, xpu_vals, color=["#6D9DC5", "#6D9DC5", "#A7C957", "#A7C957", "#F2C14E"])
    ax.set_xticks(x)
    ax.set_xticklabels(xpu_labels)
    ax.set_ylabel("Count")
    ax.set_title("xpu-perf Local Coverage Snapshot")
    ax.grid(axis="y", alpha=0.25)
    for b, v in zip(bars, xpu_vals):
        ax.annotate(f"{int(v)}", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.text(0.02, 0.04, "Source: local xpu-perf op_defs/workloads tree",
            transform=ax.transAxes, fontsize=8, color="#555555")

    ax = axes[1, 1]
    layer_labels = ["Kernel\nprimitives", "Fused distributed\noperators", "Benchmark &\nmethodology"]
    layer_scores = [1, 2, 3]
    colors = ["#2F6B5F", "#F58518", "#6D9DC5"]
    ax.barh(np.arange(3), layer_scores, color=colors)
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(layer_labels)
    ax.set_xlim(0, 3.7)
    ax.set_xlabel("Stack level")
    ax.set_title("Project Positioning")
    ax.grid(axis="x", alpha=0.2)
    for i, (name, val) in enumerate(zip(["ThunderKittens", "mKernel", "xpu-perf"], layer_scores)):
        ax.annotate(name, (val, i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.text(0.02, 0.04, "From implementation substrate to evaluation loop",
            transform=ax.transAxes, fontsize=8, color="#555555")

    fig.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
