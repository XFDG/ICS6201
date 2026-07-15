"""Execute the H200 quack GEMM follow-up plan without modifying source code.

The script keeps the same 14-shape dataset used in the 2026-06-04 report and
adds two checks:

1. current_after: the small candidate set that already showed gains.
2. expanded_after: a focused, more aggressive candidate set for H200/SM90.

It also repeats the comparison under different expert-M distributions while
keeping each case's T/TK fixed. This tests whether best configs are driven only
by total compact rows or also by per-expert imbalance.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.1")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-13.1/bin/ptxas")
import torch

from quack.gemm_config import GemmConfig

from compare_quack_gemm_before_after_h200 import cfg, proposed_gated_configs, proposed_out_configs
from tune_quack_gemm_h200 import (
    H200_BF16_PEAK_TFLOPS,
    Case,
    config_label,
    flops_for,
    gated_call,
    make_gated_configs,
    make_out_configs,
    out_call,
    unique_configs,
    valid_sm90_gated_config,
    valid_sm90_out_config,
    warm_gpu,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


CASES = (
    Case("m4k_tk5k", 4096, 4915),
    Case("m4k_tk7k", 4096, 7373),
    Case("m8k_tk10k", 8192, 9830),
    Case("m8k_tk12k", 8192, 12288),
    Case("m12k_tk18k", 12288, 18432),
    Case("online_min", 12176, 14395),
    Case("m16k_tk20k", 16384, 19661),
    Case("m20k_tk24k", 20000, 24000),
    Case("online_target", 24497, 29446),
    Case("m24k_tk36k", 24576, 36864),
    Case("online_max", 28940, 36560),
    Case("m32k_tk42k", 32768, 42598),
    Case("m40k_tk52k", 40960, 53248),
    Case("m48k_tk64k", 49152, 65536),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=3072)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--distributions",
        default="random,balanced,skewed",
        help="Comma-separated expert-M distributions: random,balanced,skewed.",
    )
    parser.add_argument(
        "--out-summary-csv",
        default=str(DATA_DIR / "quack_gemm_h200_plan_summary_2026-06-05.csv"),
    )
    parser.add_argument(
        "--out-top-csv",
        default=str(DATA_DIR / "quack_gemm_h200_plan_top_configs_2026-06-05.csv"),
    )
    return parser.parse_args()


def expanded_gated_configs() -> list[GemmConfig]:
    # Use the same safe extension rule as tune_quack_gemm_h200.py: mutate
    # existing quack SM90 configs into cluster_m=1 and swizzle=4 variants.
    # This is wider than current_after while avoiding unsupported tile shapes
    # that can abort the JIT process instead of raising a Python exception.
    return unique_configs(proposed_gated_configs() + make_gated_configs("extended"))


def expanded_out_configs() -> list[GemmConfig]:
    return unique_configs(proposed_out_configs() + make_out_configs("extended"))


def config_key(conf: GemmConfig) -> tuple:
    data = asdict(conf)
    return tuple((key, data[key]) for key in sorted(data))


def source_rows(tokens: int, routed_m: int, topk: int) -> torch.Tensor:
    extra = routed_m - tokens
    if not (0 <= extra <= (topk - 1) * tokens):
        raise ValueError(f"invalid routed_m={routed_m} for tokens={tokens}, topk={topk}")

    token_counts = torch.ones(tokens, dtype=torch.int32)
    remaining = extra
    while remaining > 0:
        take = min(tokens, remaining)
        token_counts[:take] += 1
        remaining -= take

    chunks: list[torch.Tensor] = []
    for slot in range(topk):
        rows = torch.nonzero(token_counts > slot, as_tuple=False).flatten()
        if rows.numel() > 0:
            chunks.append(rows)
    src = torch.cat(chunks).to(torch.int64)
    if src.numel() != routed_m:
        raise RuntimeError(f"source rows mismatch: {src.numel()} vs {routed_m}")
    return src


def counts_for_distribution(
    *, routed_m: int, experts: int, distribution: str, seed: int
) -> torch.Tensor:
    if distribution == "balanced":
        base = routed_m // experts
        rem = routed_m % experts
        counts = torch.full((experts,), base, dtype=torch.int64)
        counts[:rem] += 1
        return counts

    if distribution == "skewed":
        weights = torch.tensor(
            [0.32, 0.18, 0.11, 0.08, 0.06, 0.05, 0.04, 0.035, 0.03, 0.025, 0.022, 0.02, 0.017, 0.015, 0.013, 0.008],
            dtype=torch.float64,
        )
        weights = weights[:experts]
        weights = weights / weights.sum()
        raw = weights * routed_m
        counts = torch.floor(raw).to(torch.int64)
        counts += 1
        overflow = int(counts.sum().item() - routed_m)
        if overflow > 0:
            order = torch.argsort(counts, descending=True)
            i = 0
            while overflow > 0:
                idx = int(order[i % experts])
                if counts[idx] > 1:
                    counts[idx] -= 1
                    overflow -= 1
                i += 1
        elif overflow < 0:
            frac = raw - torch.floor(raw)
            order = torch.argsort(frac, descending=True)
            for i in range(-overflow):
                counts[int(order[i % experts])] += 1
        return counts

    if distribution == "random":
        gen = torch.Generator()
        gen.manual_seed(seed + routed_m + experts)
        expert_ids = torch.randint(0, experts, (routed_m,), dtype=torch.int64, generator=gen)
        return torch.bincount(expert_ids, minlength=experts).to(torch.int64)

    raise ValueError(f"unknown distribution: {distribution}")


def make_metadata(
    *,
    tokens: int,
    topk: int,
    experts: int,
    routed_m: int,
    distribution: str,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src = source_rows(tokens=tokens, routed_m=routed_m, topk=topk)
    counts = counts_for_distribution(
        routed_m=routed_m, experts=experts, distribution=distribution, seed=seed
    )
    expert_ids = torch.repeat_interleave(torch.arange(experts, dtype=torch.int64), counts)
    if expert_ids.numel() != routed_m:
        raise RuntimeError(f"expert ids mismatch: {expert_ids.numel()} vs {routed_m}")

    gen = torch.Generator()
    gen.manual_seed(seed + routed_m * 17 + tokens)
    shuffled = expert_ids[torch.randperm(routed_m, generator=gen)]
    order = torch.argsort(shuffled, stable=True)
    sorted_counts = torch.bincount(shuffled[order], minlength=experts).to(torch.int32)
    cu = torch.empty(experts + 1, dtype=torch.int32)
    cu[0] = 0
    cu[1:] = torch.cumsum(sorted_counts, dim=0)
    a_idx = src[order].to(torch.int32)
    return cu.to(device), a_idx.to(device), sorted_counts.to(device)


def make_inputs(
    *,
    case: Case,
    hidden_size: int,
    intermediate_size: int,
    experts: int,
    topk: int,
    distribution: str,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cu, a_idx, tokens_per_expert = make_metadata(
        tokens=case.tokens,
        topk=topk,
        experts=experts,
        routed_m=case.routed_m,
        distribution=distribution,
        device=device,
        seed=seed,
    )
    x = torch.randn(case.tokens, hidden_size, device=device, dtype=torch.bfloat16)
    w1 = torch.randn(experts, hidden_size, 2 * intermediate_size, device=device, dtype=torch.bfloat16)
    w2 = torch.randn(experts, intermediate_size, hidden_size, device=device, dtype=torch.bfloat16)
    preact = torch.empty(case.routed_m, 2 * intermediate_size, device=device, dtype=torch.bfloat16)
    postact = torch.empty(case.routed_m, intermediate_size, device=device, dtype=torch.bfloat16)
    down_out = torch.empty(case.routed_m, hidden_size, device=device, dtype=torch.bfloat16)
    return {
        "x": x,
        "w1": w1,
        "w2": w2,
        "cu": cu,
        "a_idx": a_idx,
        "tokens_per_expert": tokens_per_expert,
        "preact": preact,
        "postact": postact,
        "down_out": down_out,
    }


def measure_ms(fn, warmup: int, iters: int, repeats: int) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    return statistics.median(samples)


def bench_config(
    *,
    op: str,
    inp: dict[str, torch.Tensor],
    conf: GemmConfig,
    warmup: int,
    iters: int,
    repeats: int,
) -> tuple[float, str]:
    call = (lambda: gated_call(inp, conf)) if op == "gated" else (lambda: out_call(inp, conf))
    try:
        call()
        torch.cuda.synchronize()
        return measure_ms(call, warmup=warmup, iters=iters, repeats=repeats), ""
    except Exception as exc:  # noqa: BLE001 - keep sweeping when a probe is invalid.
        torch.cuda.synchronize()
        return math.inf, repr(exc).replace("\n", " ")[:300]


def best_from_timings(
    configs: list[GemmConfig], timings: dict[tuple, float]
) -> tuple[float, GemmConfig]:
    best_ms = math.inf
    best_conf = configs[0]
    for conf in configs:
        ms = timings.get(config_key(conf), math.inf)
        if ms < best_ms:
            best_ms = ms
            best_conf = conf
    return best_ms, best_conf


def mfu_percent(op: str, routed_m: int, ms: float, hidden_size: int, intermediate_size: int) -> float:
    flops = flops_for(op, routed_m, hidden_size, intermediate_size)
    return flops / (ms / 1000.0) / (H200_BF16_PEAK_TFLOPS * 1e12) * 100.0


def skew_ratio(tokens_per_expert: torch.Tensor) -> float:
    counts = tokens_per_expert.detach().cpu().to(torch.float64)
    mean = counts.mean().item()
    return counts.max().item() / mean if mean else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    distributions = [item.strip() for item in args.distributions.split(",") if item.strip()]
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    print(
        f"execute H200 plan: device={args.device}, warmup={args.warmup}, "
        f"iters={args.iters}, repeats={args.repeats}, distributions={distributions}"
    )
    print(f"GPU: {torch.cuda.get_device_name(device)} {torch.cuda.get_device_capability(device)}")

    config_sets = {
        "before": {
            "gated": make_gated_configs("builtin"),
            "out": make_out_configs("builtin"),
        },
        "current_after": {
            "gated": proposed_gated_configs(),
            "out": proposed_out_configs(),
        },
        "expanded_after": {
            "gated": expanded_gated_configs(),
            "out": expanded_out_configs(),
        },
    }
    print(
        "configs: "
        + ", ".join(
            f"{name}(gated={len(sets['gated'])},out={len(sets['out'])})"
            for name, sets in config_sets.items()
        )
    )
    warm_gpu()

    summary_rows: list[dict] = []
    top_rows: list[dict] = []
    for distribution in distributions:
        for case in CASES:
            inp = make_inputs(
                case=case,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                experts=args.experts,
                topk=args.topk,
                distribution=distribution,
                device=device,
                seed=args.seed,
            )
            skew = skew_ratio(inp["tokens_per_expert"])
            print(f"\ncase={case.name} dist={distribution} T={case.tokens} TK={case.routed_m} skew={skew:.2f}")

            for op in ("gated", "out"):
                union = unique_configs(
                    config_sets["before"][op]
                    + config_sets["current_after"][op]
                    + config_sets["expanded_after"][op]
                )
                timings: dict[tuple, float] = {}
                errors: dict[tuple, str] = {}
                for conf in union:
                    ms, err = bench_config(
                        op=op,
                        inp=inp,
                        conf=conf,
                        warmup=args.warmup,
                        iters=args.iters,
                        repeats=args.repeats,
                    )
                    timings[config_key(conf)] = ms
                    errors[config_key(conf)] = err

                sorted_configs = sorted(
                    union,
                    key=lambda conf: timings.get(config_key(conf), math.inf),
                )
                for rank, conf in enumerate(sorted_configs[:5], start=1):
                    ms = timings[config_key(conf)]
                    top_rows.append(
                        {
                            "case": case.name,
                            "distribution": distribution,
                            "tokens": case.tokens,
                            "routed_m": case.routed_m,
                            "skew_ratio": skew,
                            "op": op,
                            "rank": rank,
                            "ms": ms,
                            "mfu_percent": mfu_percent(
                                op,
                                case.routed_m,
                                ms,
                                args.hidden_size,
                                args.intermediate_size,
                            )
                            if math.isfinite(ms)
                            else 0.0,
                            "config": config_label(conf),
                            **asdict(conf),
                            "error": errors[config_key(conf)],
                        }
                    )

                for set_name, sets in config_sets.items():
                    best_ms, best_conf = best_from_timings(sets[op], timings)
                    summary_rows.append(
                        {
                            "case": case.name,
                            "distribution": distribution,
                            "tokens": case.tokens,
                            "routed_m": case.routed_m,
                            "skew_ratio": skew,
                            "op": op,
                            "candidate_set": set_name,
                            "best_ms": best_ms,
                            "mfu_percent": mfu_percent(
                                op,
                                case.routed_m,
                                best_ms,
                                args.hidden_size,
                                args.intermediate_size,
                            )
                            if math.isfinite(best_ms)
                            else 0.0,
                            "best_config": config_label(best_conf),
                            **asdict(best_conf),
                        }
                    )
                    print(
                        f"  {op:5s} {set_name:14s} {best_ms:.4f} ms "
                        f"{summary_rows[-1]['mfu_percent']:.1f}% {config_label(best_conf)}"
                    )

    write_csv(Path(args.out_summary_csv), summary_rows)
    write_csv(Path(args.out_top_csv), top_rows)
    print(f"\nwrote {args.out_summary_csv}")
    print(f"wrote {args.out_top_csv}")


if __name__ == "__main__":
    main()
