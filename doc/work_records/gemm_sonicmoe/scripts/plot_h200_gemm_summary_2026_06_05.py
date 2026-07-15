"""Generate summary charts for the H200 GEMM comparison report."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "data"
OUT_DIR = ROOT / "scripts"
H200_BF16_PEAK_TFLOPS = 989.0
ONLINE_H = 3072
ONLINE_I = 1536


def save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def chart_overview() -> None:
    cats = [
        "CUTLASS 2.x\nraw grouped",
        "quack pure\nGEMM",
        "sonic-moe\ne2e FWD",
        "sonic-moe\ne2e BWD",
    ]
    mfu = [35.9, 66.5, 32.1, 20.8]
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759"]

    avg_m = np.array([512, 1024, 2048, 4096])
    quack_pure = np.array([59.4, 65.5, 71.5, 74.9])
    e2e_fwd = np.array([30.7, 32.6, 32.8, 32.2])
    e2e_bwd = np.array([19.8, 21.2, 21.1, 21.2])

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8))
    ax = axes[0]
    bars = ax.bar(cats, mfu, color=colors, width=0.62)
    ax.set_ylabel("MFU (%)")
    ax.set_title("H200 GEMM Path MFU")
    ax.set_ylim(0, 80)
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, mfu):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.2, f"{val:.1f}%", ha="center", fontsize=9)

    ax = axes[1]
    ax.plot(avg_m, quack_pure, "o-", color="#59A14F", lw=2.4, label="quack pure GEMM")
    ax.plot(avg_m, e2e_fwd, "s-", color="#F28E2B", lw=2.4, label="sonic-moe e2e FWD")
    ax.plot(avg_m, e2e_bwd, "^-", color="#E15759", lw=2.4, label="sonic-moe e2e BWD")
    ax.set_xscale("log", base=2)
    ax.set_xticks(avg_m)
    ax.set_xticklabels([str(x) for x in avg_m])
    ax.set_xlabel("avg tokens per expert")
    ax.set_ylabel("MFU (%)")
    ax.set_title("MFU vs Expert M")
    ax.set_ylim(0, 82)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("H200 MoE GEMM Performance Overview", fontsize=14)
    save(fig, "h200_gemm_summary_2026-06-05_overview.png")


def chart_quack_tuning() -> None:
    aggregate = read_csv(PROFILES / "quack_gemm_h200_plan_aggregate_2026-06-05.csv")
    rows_by_dist_set: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in aggregate:
        rows_by_dist_set[(row["distribution"], row["candidate_set"])].append(row)

    dists = ["balanced", "random", "skewed"]
    labels = ["balanced", "random", "skewed"]

    def total_ms(dist: str, candidate_set: str) -> float:
        return sum(float(r["total_ms"]) for r in rows_by_dist_set[(dist, candidate_set)])

    def weighted_mfu(dist: str, candidate_set: str) -> float:
        rows = rows_by_dist_set[(dist, candidate_set)]
        total_flops = sum(6.0 * float(r["routed_m"]) * ONLINE_H * ONLINE_I for r in rows)
        total_seconds = sum(float(r["total_ms"]) for r in rows) / 1000.0
        return total_flops / total_seconds / (H200_BF16_PEAK_TFLOPS * 1e12) * 100.0

    before = np.array([total_ms(d, "before") for d in dists])
    current = np.array([total_ms(d, "current_after") for d in dists])
    expanded = np.array([total_ms(d, "expanded_after") for d in dists])
    before_mfu = np.array([weighted_mfu(d, "before") for d in dists])
    expanded_mfu = np.array([weighted_mfu(d, "expanded_after") for d in dists])
    speedup = (before / expanded - 1.0) * 100.0

    target = [r for r in aggregate if r["case"] == "online_target"]
    target_by_dist: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in target:
        target_by_dist[row["distribution"]][row["candidate_set"]] = row
    target_before = np.array([float(target_by_dist[d]["before"]["total_ms"]) for d in dists])
    target_expanded = np.array([float(target_by_dist[d]["expanded_after"]["total_ms"]) for d in dists])
    target_speedup = (target_before / target_expanded - 1.0) * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.0))
    x = np.arange(len(dists))
    w = 0.25

    ax = axes[0, 0]
    ax.bar(x - w, before, w, label="before", color="#4C78A8")
    ax.bar(x, current, w, label="current_after", color="#F28E2B")
    ax.bar(x + w, expanded, w, label="expanded_after", color="#59A14F")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total latency (ms)")
    ax.set_title("14 Cases: Total Two-GEMM Latency")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.bar(labels, speedup, color="#59A14F", width=0.55)
    ax.set_ylabel("Speedup vs before (%)")
    ax.set_title("Expanded Candidate Speedup")
    ax.grid(axis="y", alpha=0.25)
    for i, val in enumerate(speedup):
        ax.text(i, val + 0.06, f"{val:.2f}%", ha="center", fontsize=9)

    ax = axes[1, 0]
    ax.bar(x - w / 2, before_mfu, w, color="#4C78A8", label="before")
    ax.bar(x + w / 2, expanded_mfu, w, color="#59A14F", label="expanded_after")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Weighted MFU (%)")
    ax.set_title("Weighted MFU by Expert Distribution")
    ax.set_ylim(56, 65)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.bar(labels, target_speedup, color="#B07AA1", width=0.55)
    ax.set_ylabel("Speedup vs before (%)")
    ax.set_title("online_target Speedup")
    ax.set_ylim(0, 6.4)
    ax.grid(axis="y", alpha=0.25)
    for i, val in enumerate(target_speedup):
        ax.text(i, val + 0.10, f"{val:.2f}%", ha="center", fontsize=9)

    fig.suptitle("Quack GEMM H200 Config-Level Tuning", fontsize=14)
    save(fig, "h200_gemm_summary_2026-06-05_quack_tuning.png")


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.array(values, dtype=float), q))


def chart_online_and_deepgemm() -> None:
    ranges = read_csv(PROFILES / "cshi_opt_rank64_fused_projection_ranges.csv")
    tk = np.array([float(r["TK_valid"]) for r in ranges])
    gated = np.array([float(r["gemm_gated_us"]) for r in ranges])
    out = np.array([float(r["gemm_out_us"]) for r in ranges])
    gather = np.array([float(r["token_gather_us"]) for r in ranges])
    kernel_sum = np.array([float(r["kernel_sum_us"]) for r in ranges])

    buckets = [
        ("0-20k", tk < 20000),
        ("20-24k", (tk >= 20000) & (tk < 24000)),
        ("24-28k", (tk >= 24000) & (tk < 28000)),
        ("28-32k", (tk >= 28000) & (tk < 32000)),
        ("32k+", tk >= 32000),
    ]
    bucket_labels = [b[0] for b in buckets]
    gated_avg = [float(gated[mask].mean()) for _, mask in buckets]
    out_avg = [float(out[mask].mean()) for _, mask in buckets]
    gather_avg = [float(gather[mask].mean()) for _, mask in buckets]

    local_replay = np.array([707.800, 1298.570, 1600.330])
    online_replay = np.array([704.544, 1313.120, 1691.681])
    replay_labels = ["min", "target", "max"]

    deep_labels = ["gpu0", "gpu1"]
    jit_off = np.array([10.387571, 10.587196])
    jit_on = np.array([0.357475, 0.439432])
    jit_speedup = jit_off / jit_on

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.0))

    ax = axes[0, 0]
    x = np.arange(len(bucket_labels))
    ax.bar(x, gated_avg, label="gemm_gated", color="#4C78A8")
    ax.bar(x, out_avg, bottom=gated_avg, label="gemm_out", color="#F28E2B")
    ax.bar(x, gather_avg, bottom=np.array(gated_avg) + np.array(out_avg), label="token_gather", color="#59A14F")
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_ylabel("Avg latency (us)")
    ax.set_title("Online 5-op Latency by TK_valid")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.hist(kernel_sum, bins=34, color="#4C78A8", alpha=0.85)
    ax.axvline(percentile(kernel_sum.tolist(), 50), color="#F28E2B", lw=2, label="P50")
    ax.axvline(percentile(kernel_sum.tolist(), 90), color="#E15759", lw=2, label="P90")
    ax.set_xlabel("5-op sum latency (us)")
    ax.set_ylabel("Count")
    ax.set_title("Online _FusedMoEProjection Distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    w = 0.35
    x = np.arange(len(replay_labels))
    ax.bar(x - w / 2, local_replay, w, label="local H200 replay", color="#4C78A8")
    ax.bar(x + w / 2, online_replay, w, label="online profile", color="#F28E2B")
    ax.set_xticks(x)
    ax.set_xticklabels(replay_labels)
    ax.set_ylabel("5-op sum latency (us)")
    ax.set_title("Local H200 Replay vs Online Profile")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    x = np.arange(len(deep_labels))
    ax.bar(x - w / 2, jit_off, w, label="JIT cold", color="#E15759")
    ax.bar(x + w / 2, jit_on, w, label="precompiled", color="#59A14F")
    ax.set_xticks(x)
    ax.set_xticklabels(deep_labels)
    ax.set_ylabel("First latency (s)")
    ax.set_title("DeepGEMM Precompiled First-Latency")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    for i, val in enumerate(jit_speedup):
        ax.text(i, max(jit_off[i], jit_on[i]) + 0.35, f"{val:.1f}x", ha="center", fontsize=9)

    fig.suptitle("Online Alignment and DeepGEMM JIT Performance", fontsize=14)
    save(fig, "h200_gemm_summary_2026-06-05_online_deepgemm.png")


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    chart_overview()
    chart_quack_tuning()
    chart_online_and_deepgemm()


if __name__ == "__main__":
    main()
