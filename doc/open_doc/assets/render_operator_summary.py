#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "operator_optimization_summary_20260911.png"


def configure_font() -> None:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    ]
    for item in candidates:
        path = Path(item)
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def label_bars(ax, bars, fmt):
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + ax.get_xlim()[1] * 0.018,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(width),
            va="center",
            fontsize=9,
            color="#263238",
        )


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    configure_font()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.3), constrained_layout=True)
    fig.suptitle("代表性算子优化与验证结果（不同层级不可直接横向外推）", fontsize=16, fontweight="bold")

    colors = ["#0B6E99", "#14866D", "#D9822B", "#7B61A8"]

    ax = axes[0, 0]
    names = ["Attention\n完整反向", "Sink Top-K\nForward", "Sink Top-K\nFwd+Bwd", "Router GEMM\nMicrokernel"]
    vals = [3.98, 3.15, 2.09, 2.95]
    bars = ax.barh(np.arange(len(names)), vals, color=colors)
    ax.set_yticks(np.arange(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 4.6)
    ax.set_xlabel("加速倍数（×）")
    ax.set_title("A. Kernel / Operator 层")
    label_bars(ax, bars, "{:.2f}×")
    ax.text(0.02, -0.24, "Sink Top-K 为默认关闭的原型结果", transform=ax.transAxes, fontsize=8, color="#5F6B73")

    ax = axes[0, 1]
    names = ["Orth-loss\n目标函数", "完整 MoE\nLayer", "Router 请求\nPrefill/Decode", "Router 请求\n长 Decode"]
    vals = [18.77, 3.97, 3.30, 4.95]
    bars = ax.barh(np.arange(len(names)), vals, color=colors)
    ax.set_yticks(np.arange(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 22)
    ax.set_xlabel("提升（%）")
    ax.set_title("B. Layer / Request 层")
    label_bars(ax, bars, "{:.2f}%")
    ax.text(0.02, -0.24, "长 Decode 为 4.66%–5.24% 区间中值", transform=ax.transAxes, fontsize=8, color="#5F6B73")

    ax = axes[1, 0]
    names = ["TP1 GPU 历史算子", "TP2 Decode 安全路径", "TP4 Decode 安全路径"]
    vals = [4.99, 4.60, 3.00]
    bars = ax.barh(np.arange(len(names)), vals, color=["#0B6E99", "#14866D", "#D9822B"])
    ax.set_yticks(np.arange(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 6.1)
    ax.set_xlabel("吞吐提升（%）")
    ax.set_title("C. 有状态异步 Decode")
    label_bars(ax, bars, "{:.2f}%")
    ax.text(0.02, -0.24, "TP2/TP4 为 async-unfused + batch-invariant", transform=ax.transAxes, fontsize=8, color="#5F6B73")

    ax = axes[1, 1]
    names = ["Legacy", "Compiled"]
    vals = [17, 7]
    bars = ax.bar(names, vals, color=["#8A96A3", "#0B6E99"], width=0.58)
    ax.set_ylim(0, 20)
    ax.set_ylabel("GPU operation 数")
    ax.set_title("D. Router orth-loss 碎算子收敛")
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, str(value), ha="center", fontsize=11, fontweight="bold")
    ax.text(0.5, 0.52, "17 → 7", transform=ax.transAxes, ha="center", fontsize=18, color="#0B6E99", fontweight="bold")

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
