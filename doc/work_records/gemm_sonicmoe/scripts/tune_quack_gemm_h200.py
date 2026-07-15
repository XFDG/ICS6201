"""Sweep quack GEMM configs for H200/SM90 MoE shapes.

This script benchmarks only quack GEMM kernels:

  1. gated up-projection:
       x[T,H] gathered by A_idx -> w1[E,H,2I] -> preact[TK,2I], postact[TK,I]
  2. down-projection:
       postact[TK,I] -> w2[E,I,H] -> down_out[TK,H]

It intentionally excludes DeepEP, token routing, combine, and communication.
No project source code is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.1")
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-13.1/bin/ptxas")
import torch

from quack.gemm_config import GemmConfig, get_all_configs
from quack.gemm_interface import default_config, gemm_gated_tuned, gemm_tuned


H200_BF16_PEAK_TFLOPS = 989.0
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class Case:
    name: str
    tokens: int
    routed_m: int


CASES = (
    Case("online_min", 12176, 14395),
    Case("online_target", 24497, 29446),
    Case("online_max", 28940, 36560),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--cases", default="online_min,online_target,online_max")
    parser.add_argument("--hidden-size", type=int, default=3072)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--candidate-set",
        choices=("builtin", "extended"),
        default="extended",
        help="builtin uses quack's current SM90 autotune space; extended adds a small set of H200 schedule variants.",
    )
    parser.add_argument(
        "--limit-configs-per-op",
        type=int,
        default=0,
        help="For quick smoke tests. 0 means no limit.",
    )
    parser.add_argument(
        "--out-csv",
        default=str(DATA_DIR / "quack_gemm_h200_tuning_2026-06-04.csv"),
    )
    parser.add_argument(
        "--out-jsonl",
        default=str(DATA_DIR / "quack_gemm_h200_tuning_2026-06-04.jsonl"),
    )
    return parser.parse_args()


def selected_cases(names_csv: str) -> list[Case]:
    names = {name.strip() for name in names_csv.split(",") if name.strip()}
    cases = [case for case in CASES if case.name in names]
    missing = names - {case.name for case in cases}
    if missing:
        raise ValueError(f"unknown cases: {sorted(missing)}")
    return cases


def stable_config_key(conf: GemmConfig) -> tuple:
    data = asdict(conf)
    return tuple((key, data[key]) for key in sorted(data))


def unique_configs(configs: list[GemmConfig]) -> list[GemmConfig]:
    seen: set[tuple] = set()
    result: list[GemmConfig] = []
    for conf in configs:
        key = stable_config_key(conf)
        if key in seen:
            continue
        seen.add(key)
        result.append(conf)
    return result


def valid_sm90_gated_config(conf: GemmConfig) -> bool:
    return (
        conf.device_capacity == 9
        and not conf.swap_ab
        and conf.cluster_n == 1
        and conf.tile_n != 208
        and not conf.is_dynamic_persistent
        and not conf.use_tma_gather
    )


def valid_sm90_out_config(conf: GemmConfig) -> bool:
    return (
        conf.device_capacity == 9
        and not conf.swap_ab
        and not conf.use_tma_gather
    )


def make_gated_configs(candidate_set: str) -> list[GemmConfig]:
    configs = [conf for conf in get_all_configs("gated") if valid_sm90_gated_config(conf)]
    if candidate_set == "extended":
        # H200-specific probes around quack's current SM90 space:
        # - cluster_m=1 to reduce tail cost for uneven expert M
        # - smaller swizzle to test whether the default 8-way swizzle hurts locality
        extras: list[GemmConfig] = []
        for conf in configs:
            extras.append(GemmConfig(**{**asdict(conf), "cluster_m": 1, "cluster_n": 1}))
            extras.append(GemmConfig(**{**asdict(conf), "max_swizzle_size": 4}))
        configs += [conf for conf in extras if valid_sm90_gated_config(conf)]
    return unique_configs(configs)


def make_out_configs(candidate_set: str) -> list[GemmConfig]:
    configs = [conf for conf in get_all_configs(None) if valid_sm90_out_config(conf)]
    if candidate_set == "extended":
        extras: list[GemmConfig] = []
        for conf in configs:
            # cluster 1x1 can win for small or imbalanced grouped-M cases.
            extras.append(GemmConfig(**{**asdict(conf), "cluster_m": 1, "cluster_n": 1}))
            # Dynamic persistent scheduling is off by default on SM90, but it is a
            # useful targeted probe for grouped-M tail effects in the down GEMM.
            if conf.tile_m in (128, 256) and conf.tile_n in (128, 160, 192):
                extras.append(GemmConfig(**{**asdict(conf), "is_dynamic_persistent": True}))
            if conf.max_swizzle_size == 8:
                extras.append(GemmConfig(**{**asdict(conf), "max_swizzle_size": 4}))
        configs += [conf for conf in extras if valid_sm90_out_config(conf)]
    return unique_configs(configs)


def make_deepep_like_metadata(
    *,
    tokens: int,
    topk: int,
    experts: int,
    routed_m: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cu_seqlens_m, A_idx, tokens_per_expert for a compact local MoE case.

    This emulates the online compact row contract for quack only. It preserves:
      - full token count T
      - compact local routed row count TK
      - sorted expert-major layout via cu_seqlens_m
      - gather_A through A_idx
    """

    extra = routed_m - tokens
    if not (0 <= extra <= (topk - 1) * tokens):
        raise ValueError(f"invalid routed_m={routed_m} for tokens={tokens}, topk={topk}")

    token_counts = torch.ones(tokens, dtype=torch.int32)
    remaining = extra
    while remaining > 0:
        take = min(tokens, remaining)
        token_counts[:take] += 1
        remaining -= take

    source_rows: list[torch.Tensor] = []
    for slot in range(topk):
        rows = torch.nonzero(token_counts > slot, as_tuple=False).flatten()
        if rows.numel() > 0:
            source_rows.append(rows)
    src = torch.cat(source_rows).to(torch.int64)
    assert src.numel() == routed_m

    gen = torch.Generator()
    gen.manual_seed(seed + tokens + routed_m)
    compact_experts = torch.randint(0, experts, (routed_m,), dtype=torch.int64, generator=gen)
    order = torch.argsort(compact_experts, stable=True)
    tokens_per_expert = torch.bincount(compact_experts[order], minlength=experts).to(torch.int32)

    cu = torch.empty(experts + 1, dtype=torch.int32)
    cu[0] = 0
    cu[1:] = torch.cumsum(tokens_per_expert, dim=0)
    a_idx = src[order].to(torch.int32)
    return cu.to(device), a_idx.to(device), tokens_per_expert


def make_inputs(
    *,
    case: Case,
    hidden_size: int,
    intermediate_size: int,
    experts: int,
    topk: int,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cu, a_idx, tokens_per_expert = make_deepep_like_metadata(
        tokens=case.tokens,
        topk=topk,
        experts=experts,
        routed_m=case.routed_m,
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


def warm_gpu(duration_s: float = 0.25) -> None:
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    start = time.perf_counter()
    while time.perf_counter() - start < duration_s:
        a = a @ a
        torch.cuda.synchronize()


def measure_ms(call: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def gated_call(inp: dict[str, torch.Tensor], conf: GemmConfig) -> None:
    gemm_gated_tuned.fn(
        inp["x"],
        inp["w1"],
        inp["preact"],
        inp["postact"],
        None,
        None,
        "swiglu",
        inp["cu"],
        inp["a_idx"],
        False,
        conf,
        None,
    )


def out_call(inp: dict[str, torch.Tensor], conf: GemmConfig) -> None:
    gemm_tuned.fn(
        inp["postact"],
        inp["w2"],
        inp["down_out"],
        None,
        None,
        1.0,
        1.0,
        inp["cu"],
        None,
        None,
        None,
        False,
        False,
        conf,
    )


def flops_for(op: str, routed_m: int, hidden_size: int, intermediate_size: int) -> int:
    if op == "gated":
        return 2 * routed_m * hidden_size * (2 * intermediate_size)
    if op == "out":
        return 2 * routed_m * intermediate_size * hidden_size
    raise ValueError(op)


def mfu_percent(ms: float, flops: int) -> float:
    return flops / (ms / 1000.0) / (H200_BF16_PEAK_TFLOPS * 1e12) * 100.0


def config_label(conf: GemmConfig) -> str:
    return (
        f"m{conf.tile_m}_n{conf.tile_n}_cm{conf.cluster_m}_cn{conf.cluster_n}_"
        f"pp{int(conf.pingpong)}_dyn{int(conf.is_dynamic_persistent)}_sw{conf.max_swizzle_size}"
    )


def run_one_config(
    *,
    op: str,
    case: Case,
    inp: dict[str, torch.Tensor],
    conf: GemmConfig,
    warmup: int,
    iters: int,
    hidden_size: int,
    intermediate_size: int,
) -> dict:
    call = (lambda: gated_call(inp, conf)) if op == "gated" else (lambda: out_call(inp, conf))
    try:
        # First call compiles or loads the CuTe/CUTLASS JIT artifact.
        call()
        torch.cuda.synchronize()
        ms = measure_ms(call, warmup=warmup, iters=iters)
        status = "ok"
        error = ""
    except Exception as exc:  # noqa: BLE001 - benchmark should keep sweeping.
        torch.cuda.synchronize()
        ms = math.inf
        status = "failed"
        error = repr(exc).replace("\n", " ")[:500]
    flops = flops_for(op, case.routed_m, hidden_size, intermediate_size)
    return {
        "case": case.name,
        "op": op,
        "tokens": case.tokens,
        "routed_m": case.routed_m,
        "ms": ms,
        "mfu_percent": 0.0 if not math.isfinite(ms) else mfu_percent(ms, flops),
        "status": status,
        "error": error,
        "config_label": config_label(conf),
        **{f"config_{k}": v for k, v in asdict(conf).items()},
    }


def write_rows(rows: list[dict], csv_path: Path, jsonl_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    print(
        "quack GEMM H200 tuning: "
        f"device={args.device}, visible={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}, "
        f"candidate_set={args.candidate_set}, warmup={args.warmup}, iters={args.iters}"
    )
    print(f"GPU: {torch.cuda.get_device_name(device)} capability={torch.cuda.get_device_capability(device)}")

    gated_configs = make_gated_configs(args.candidate_set)
    out_configs = make_out_configs(args.candidate_set)
    if args.limit_configs_per_op > 0:
        gated_configs = gated_configs[: args.limit_configs_per_op]
        out_configs = out_configs[: args.limit_configs_per_op]
    print(f"gated configs={len(gated_configs)}, out configs={len(out_configs)}")
    print(f"default config={default_config(device)}")

    warm_gpu()

    rows: list[dict] = []
    for case in selected_cases(args.cases):
        inp = make_inputs(
            case=case,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            experts=args.experts,
            topk=args.topk,
            device=device,
            seed=args.seed,
        )
        print(
            f"\ncase={case.name} T={case.tokens} TK={case.routed_m} "
            f"tokens_per_expert={inp['tokens_per_expert'].tolist()}"
        )
        for op, configs in (("gated", gated_configs), ("out", out_configs)):
            best: dict | None = None
            for i, conf in enumerate(configs, start=1):
                row = run_one_config(
                    op=op,
                    case=case,
                    inp=inp,
                    conf=conf,
                    warmup=args.warmup,
                    iters=args.iters,
                    hidden_size=args.hidden_size,
                    intermediate_size=args.intermediate_size,
                )
                rows.append(row)
                if row["status"] == "ok" and (best is None or row["ms"] < best["ms"]):
                    best = row
                print(
                    f"  {op:5s} {i:02d}/{len(configs):02d} {row['status']:6s} "
                    f"{row['ms']:.4f} ms {row['mfu_percent']:.1f}% {row['config_label']}"
                )
            if best is not None:
                print(
                    f"  BEST {op}: {best['ms']:.4f} ms {best['mfu_percent']:.1f}% "
                    f"{best['config_label']}"
                )

    write_rows(rows, Path(args.out_csv), Path(args.out_jsonl))
    print(f"\nwrote {args.out_csv}")
    print(f"wrote {args.out_jsonl}")


if __name__ == "__main__":
    main()
