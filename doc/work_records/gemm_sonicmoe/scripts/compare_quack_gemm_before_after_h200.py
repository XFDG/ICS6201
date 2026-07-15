"""Compare current quack SM90 configs against a proposed expanded config set.

This is a quack-only benchmark. "before" means the current built-in SM90
autotune search space. "after" means the same search space plus a small number
of H200/MoE-oriented candidates. No quack source file is modified here.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import asdict
from pathlib import Path

import torch

from quack.gemm_config import GemmConfig

from tune_quack_gemm_h200 import (
    H200_BF16_PEAK_TFLOPS,
    Case,
    config_label,
    flops_for,
    gated_call,
    make_gated_configs,
    make_inputs,
    make_out_configs,
    out_call,
    unique_configs,
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


def cfg(
    tile_m: int,
    tile_n: int,
    cluster_m: int,
    cluster_n: int,
    *,
    pingpong: bool,
    dynamic: bool = False,
    swizzle: int = 8,
) -> GemmConfig:
    return GemmConfig(
        tile_m=tile_m,
        tile_n=tile_n,
        pingpong=pingpong,
        is_dynamic_persistent=dynamic,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        max_swizzle_size=swizzle,
        device_capacity=9,
        use_tma_gather=False,
        swap_ab=False,
    )


def proposed_gated_configs() -> list[GemmConfig]:
    before = make_gated_configs("builtin")
    extras = [
        cfg(128, 192, 1, 1, pingpong=True),
        cfg(128, 224, 1, 1, pingpong=False),
        cfg(128, 256, 1, 1, pingpong=False),
        cfg(256, 192, 1, 1, pingpong=False),
        cfg(192, 128, 1, 1, pingpong=True),
    ]
    return unique_configs(before + extras)


def proposed_out_configs() -> list[GemmConfig]:
    before = make_out_configs("builtin")
    extras = [
        cfg(128, 192, 1, 2, pingpong=True, swizzle=4),
        cfg(192, 128, 1, 2, pingpong=True, swizzle=4),
        cfg(192, 128, 1, 1, pingpong=True),
        cfg(128, 128, 1, 2, pingpong=True, swizzle=4),
        cfg(256, 192, 2, 1, pingpong=False, dynamic=True),
        cfg(128, 256, 1, 1, pingpong=False),
    ]
    return unique_configs(before + extras)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=3072)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--out-csv",
        default=str(DATA_DIR / "quack_gemm_h200_before_after_more_inputs_2026-06-04.csv"),
    )
    return parser.parse_args()


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


def bench_config(op: str, inp: dict[str, torch.Tensor], conf: GemmConfig, args: argparse.Namespace) -> float:
    call = (lambda: gated_call(inp, conf)) if op == "gated" else (lambda: out_call(inp, conf))
    call()
    torch.cuda.synchronize()
    return measure_ms(call, args.warmup, args.iters, args.repeats)


def best_of_configs(
    *,
    op: str,
    inp: dict[str, torch.Tensor],
    configs: list[GemmConfig],
    args: argparse.Namespace,
) -> tuple[float, GemmConfig]:
    best_ms = math.inf
    best_config = configs[0]
    for conf in configs:
        ms = bench_config(op, inp, conf, args)
        if ms < best_ms:
            best_ms = ms
            best_config = conf
    return best_ms, best_config


def config_key(conf: GemmConfig) -> tuple:
    data = asdict(conf)
    return tuple((key, data[key]) for key in sorted(data))


def measure_union_configs(
    *,
    op: str,
    inp: dict[str, torch.Tensor],
    configs: list[GemmConfig],
    args: argparse.Namespace,
) -> dict[tuple, float]:
    timings: dict[tuple, float] = {}
    for conf in unique_configs(configs):
        timings[config_key(conf)] = bench_config(op, inp, conf, args)
    return timings


def best_from_timings(
    *,
    configs: list[GemmConfig],
    timings: dict[tuple, float],
) -> tuple[float, GemmConfig]:
    best_ms = math.inf
    best_config = configs[0]
    for conf in configs:
        ms = timings[config_key(conf)]
        if ms < best_ms:
            best_ms = ms
            best_config = conf
    return best_ms, best_config


def mfu_percent(op: str, routed_m: int, ms: float, hidden_size: int, intermediate_size: int) -> float:
    flops = flops_for(op, routed_m, hidden_size, intermediate_size)
    return flops / (ms / 1000.0) / (H200_BF16_PEAK_TFLOPS * 1e12) * 100.0


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    print(
        f"before/after quack GEMM compare: device={args.device}, "
        f"warmup={args.warmup}, iters={args.iters}, repeats={args.repeats}"
    )
    print(f"GPU: {torch.cuda.get_device_name(device)} {torch.cuda.get_device_capability(device)}")

    before_gated = make_gated_configs("builtin")
    before_out = make_out_configs("builtin")
    after_gated = proposed_gated_configs()
    after_out = proposed_out_configs()
    print(
        f"configs: gated before={len(before_gated)} after={len(after_gated)}, "
        f"out before={len(before_out)} after={len(after_out)}"
    )
    warm_gpu()

    rows: list[dict] = []
    for case in CASES:
        inp = make_inputs(
            case=case,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            experts=args.experts,
            topk=args.topk,
            device=device,
            seed=args.seed,
        )
        print(f"\ncase={case.name} T={case.tokens} TK={case.routed_m}")
        for op, before_configs, after_configs in (
            ("gated", before_gated, after_gated),
            ("out", before_out, after_out),
        ):
            timings = measure_union_configs(
                op=op,
                inp=inp,
                configs=unique_configs(before_configs + after_configs),
                args=args,
            )
            before_ms, before_conf = best_from_timings(configs=before_configs, timings=timings)
            after_ms, after_conf = best_from_timings(configs=after_configs, timings=timings)
            before_mfu = mfu_percent(
                op, case.routed_m, before_ms, args.hidden_size, args.intermediate_size
            )
            after_mfu = mfu_percent(
                op, case.routed_m, after_ms, args.hidden_size, args.intermediate_size
            )
            speedup = before_ms / after_ms
            row = {
                "case": case.name,
                "tokens": case.tokens,
                "routed_m": case.routed_m,
                "op": op,
                "before_ms": before_ms,
                "after_ms": after_ms,
                "speedup": speedup,
                "improvement_percent": (speedup - 1.0) * 100.0,
                "before_mfu_percent": before_mfu,
                "after_mfu_percent": after_mfu,
                "before_config": config_label(before_conf),
                "after_config": config_label(after_conf),
                **{f"before_{k}": v for k, v in asdict(before_conf).items()},
                **{f"after_{k}": v for k, v in asdict(after_conf).items()},
            }
            rows.append(row)
            print(
                f"  {op:5s} before={before_ms:.4f}ms {before_mfu:.1f}% "
                f"after={after_ms:.4f}ms {after_mfu:.1f}% "
                f"speedup={(speedup - 1) * 100:.2f}% "
                f"after={config_label(after_conf)}"
            )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
