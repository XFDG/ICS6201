#!/usr/bin/env python3
"""
Four-way Grouped GEMM Benchmark
Same input, four paths: ① CUTLASS 2.x ② quack (CUTLASS 3.x JIT) ③ deep_gemm ④ sonic-moe e2e

Usage:
  PYTHONPATH=/path/to/sonic-moe:$PYTHONPATH python compare_four_way.py
  PYTHONPATH=/path/to/sonic-moe:$PYTHONPATH python compare_four_way.py --quick
  PYTHONPATH=/path/to/sonic-moe:$PYTHONPATH python compare_four_way.py --config my_shapes.txt
"""
import os, sys, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
PEAK_TFLOPS = 989.0

WARMUP = 5
REPEAT = 30
QUICK_REPEAT = 10


# ── helpers ──────────────────────────────────────────────

def dirichlet_dist(T, E, alpha=0.5, seed=42):
    import torch
    gen = torch.Generator()
    gen.manual_seed(seed)
    props = torch.distributions.Dirichlet(torch.full((E,), alpha, device="cpu")).sample()
    counts = (props * T).long().clamp(min=1)
    d = T - counts.sum().item()
    if d > 0:
        counts[0] += d
    elif d < 0:
        for _ in range(-d):
            idx = counts.argmax().item()
            if counts[idx] > 1:
                counts[idx] -= 1
    return counts


def build_inputs(E, T, K, N, seed=42):
    import torch
    torch.cuda.set_device(0)
    batch_sizes_cpu = dirichlet_dist(T, E, alpha=0.5, seed=seed)
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    a = torch.randn(T, K, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    b = torch.randn(E, K, N, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    return a, b, batch_sizes_cpu


def tf_mfu(flops, ms, peak=PEAK_TFLOPS):
    tflops = flops / (ms / 1000) / 1e12
    return tflops, tflops / peak * 100


def parse_config(path):
    """Parse custom shape file. Format: E,T,K,N one per line, # for comments."""
    shapes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 4:
                shapes.append(tuple(int(x) for x in parts))
    return shapes


# ── path ①: CUTLASS 2.x ─────────────────────────────────

def bench_cutlass2x(a, b, batch_sizes_cpu, K, N, quick=False):
    from ubiq_gemm.ops import ubiq_gmm
    import torch
    repeat = QUICK_REPEAT if quick else REPEAT
    for _ in range(WARMUP):
        ubiq_gmm(a, b, batch_sizes_cpu, False)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        ubiq_gmm(a, b, batch_sizes_cpu, False)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / repeat * 1000
    total_flops = sum(2 * int(bs.item()) * K * N for bs in batch_sizes_cpu)
    tflops, mfu = tf_mfu(total_flops, ms)
    return {"ms": round(ms, 4), "TFLOPS": round(tflops, 1), "MFU%": round(mfu, 1)}


# ── path ②: quack CUTLASS 3.x JIT (pure GEMM) ───────────

def bench_quack_pure(a, b, batch_sizes_cpu, K, N, quick=False):
    from quack.gemm_interface import gemm
    import torch
    repeat = QUICK_REPEAT if quick else REPEAT
    cu = torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"),
                     batch_sizes_cpu.cuda().cumsum(0).int()])
    out = torch.empty(a.shape[0], N, device="cuda", dtype=torch.bfloat16)
    for _ in range(WARMUP):
        gemm(a, b, out=out, cu_seqlens_m=cu)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        gemm(a, b, out=out, cu_seqlens_m=cu)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / repeat * 1000
    total_flops = sum(2 * int(bs.item()) * K * N for bs in batch_sizes_cpu)
    tflops, mfu = tf_mfu(total_flops, ms)
    return {"ms": round(ms, 4), "TFLOPS": round(tflops, 1), "MFU%": round(mfu, 1)}


# ── path ③: deep_gemm (pure GEMM, masked mode) ───────────

def bench_deep_gemm(a, b, batch_sizes_cpu, K, N, quick=False):
    """Use masked variant which supports custom shapes more flexibly."""
    import deep_gemm
    import torch
    repeat = QUICK_REPEAT if quick else REPEAT
    deep_gemm.set_ignore_compile_dims(True)

    E = b.shape[0]
    total_m = a.shape[0]
    expected_m = max(int(bs.item()) for bs in batch_sizes_cpu)

    # masked variant: mask per-expert valid M
    masked_m = batch_sizes_cpu.cuda().int()
    out = torch.empty(total_m, N, device="cuda", dtype=torch.bfloat16)

    for _ in range(WARMUP):
        deep_gemm.m_grouped_bf16_gemm_nt_masked(a, b, out, masked_m, expected_m)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        deep_gemm.m_grouped_bf16_gemm_nt_masked(a, b, out, masked_m, expected_m)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / repeat * 1000

    total_flops = sum(2 * int(bs.item()) * K * N for bs in batch_sizes_cpu)
    tflops, mfu = tf_mfu(total_flops, ms)
    return {"ms": round(ms, 4), "TFLOPS": round(tflops, 1), "MFU%": round(mfu, 1)}


# ── path ④: sonic-moe e2e ────────────────────────────────

def bench_sonicmoe_e2e(E, T, K, N, seed=42, quick=False):
    import torch
    from sonicmoe import MoE
    from sonicmoe.enums import ActivationType, KernelBackendMoE
    repeat = QUICK_REPEAT if quick else REPEAT
    K_topk = min(8, E)
    torch.manual_seed(seed)
    torch.cuda.set_device(0)
    moe = MoE(
        num_experts=E, num_experts_per_tok=K_topk,
        hidden_size=K, intermediate_size=N,
        activation_function=ActivationType.SWIGLU,
        add_bias=False, std=0.02,
    ).to(dtype=torch.bfloat16, device="cuda")
    x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)

    for _ in range(WARMUP):
        y, _ = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        y, _ = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)
    torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - start) / repeat * 1000

    # BWD
    for _ in range(WARMUP):
        y, _ = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)
        y.backward(dy, retain_graph=True)
        x.grad = None
        for p in moe.parameters():
            p.grad = None
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        y, _ = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)
        y.backward(dy, retain_graph=True)
        x.grad = None
        for p in moe.parameters():
            p.grad = None
    torch.cuda.synchronize()
    bwd_ms = (time.perf_counter() - start) / repeat * 1000

    fwd_flops = 3 * K_topk * T * K * N
    bwd_flops = 2 * fwd_flops
    fwd_tf, fwd_mfu = tf_mfu(fwd_flops, fwd_ms)
    bwd_tf, bwd_mfu = tf_mfu(bwd_flops, bwd_ms)
    return {"fwd_ms": round(fwd_ms, 4), "fwd_TFLOPS": round(fwd_tf, 1), "fwd_MFU%": round(fwd_mfu, 1),
            "bwd_ms": round(bwd_ms, 4), "bwd_TFLOPS": round(bwd_tf, 1), "bwd_MFU%": round(bwd_mfu, 1)}


# ── main ──────────────────────────────────────────────────

DEFAULT_CONFIG = [
    (8, 4096, 4096, 6144),
    (8, 4096, 6144, 4096),
    (8, 8192, 4096, 6144),
    (8, 8192, 6144, 4096),
    (4, 2048, 4096, 6144),
    (4, 2048, 6144, 4096),
    (4, 4096, 4096, 6144),
    (4, 4096, 6144, 4096),
    (4, 8192, 4096, 6144),
    (4, 8192, 6144, 4096),
    (2, 1024, 4096, 6144),
    (2, 1024, 6144, 4096),
    (2, 2048, 4096, 6144),
    (2, 2048, 6144, 4096),
    (2, 4096, 4096, 6144),
    (2, 4096, 6144, 4096),
    (2, 8192, 4096, 6144),
    (2, 8192, 6144, 4096),
]

if __name__ == "__main__":
    # Ensure CUDA toolkit is available (for ptxas needed by deep_gemm + Triton)
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.1")
    os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-13.1/bin/ptxas")
    cuda_bin = "/usr/local/cuda-13.1/bin"
    if cuda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{cuda_bin}:{os.environ.get('PATH', '')}"

    import torch

    quick = "--quick" in sys.argv
    config_path = None
    for i, a in enumerate(sys.argv):
        if a == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]

    if config_path:
        config = parse_config(config_path)
        print(f"Using custom config: {config_path} ({len(config)} shapes)")
    else:
        config = DEFAULT_CONFIG

    out_json = os.path.join(HERE, "four_way_results.json")
    out_md = os.path.join(HERE, "four_way_results.md")
    results = []
    n = len(config)

    print(f"{'='*85}")
    print(f"Four-way Grouped GEMM Benchmark — {n} shapes")
    print(f"  PEAK: {PEAK_TFLOPS} TFLOPS (H200 BF16), repeat={QUICK_REPEAT if quick else REPEAT}")
    print(f"{'='*85}")

    for i, (E, T, K, N) in enumerate(config):
        avg_m = round(T / E, 1)
        proj = "UP" if K == 4096 else "DOWN"
        seed = 42 + i

        print(f"\n[{i+1}/{n}] E={E} T={T} avg_M={avg_m} K={K} N={N} ({proj})")

        a, b, batch_sizes_cpu = build_inputs(E, T, K, N, seed=seed)
        row = {"E": E, "T": T, "avg_M": avg_m, "K": K, "N": N, "proj": proj}

        # ①
        r1 = bench_cutlass2x(a, b, batch_sizes_cpu, K, N, quick)
        print(f"  ① CUTLASS 2.x:  {r1['ms']:>8.4f} ms  {r1['TFLOPS']:>7.1f} TF  {r1['MFU%']:>5.1f}% MFU")
        row["c2x_ms"] = r1["ms"]; row["c2x_TFLOPS"] = r1["TFLOPS"]; row["c2x_MFU%"] = r1["MFU%"]

        # ② quack (CUTLASS 3.x JIT)
        r2 = bench_quack_pure(a, b, batch_sizes_cpu, K, N, quick)
        print(f"  ② quack 3.x:    {r2['ms']:>8.4f} ms  {r2['TFLOPS']:>7.1f} TF  {r2['MFU%']:>5.1f}% MFU")
        row["qk_ms"] = r2["ms"]; row["qk_TFLOPS"] = r2["TFLOPS"]; row["qk_MFU%"] = r2["MFU%"]

        # ③ deep_gemm
        try:
            r3 = bench_deep_gemm(a, b, batch_sizes_cpu, K, N, quick)
            print(f"  ③ deep_gemm:    {r3['ms']:>8.4f} ms  {r3['TFLOPS']:>7.1f} TF  {r3['MFU%']:>5.1f}% MFU")
        except Exception as e:
            r3 = {"ms": 0, "TFLOPS": 0, "MFU%": 0}
            print(f"  ③ deep_gemm:    SKIP ({str(e)[:80]})")
        row["dg_ms"] = r3["ms"]; row["dg_TFLOPS"] = r3["TFLOPS"]; row["dg_MFU%"] = r3["MFU%"]

        # ④ sonic-moe e2e
        try:
            r4 = bench_sonicmoe_e2e(E, T, K, N, seed=seed, quick=quick)
            print(f"  ④ sonic-moe e2e: FWD {r4['fwd_ms']:>8.4f} ms  {r4['fwd_TFLOPS']:>7.1f} TF  {r4['fwd_MFU%']:>5.1f}% MFU")
            print(f"                    BWD {r4['bwd_ms']:>8.4f} ms  {r4['bwd_TFLOPS']:>7.1f} TF  {r4['bwd_MFU%']:>5.1f}% MFU")
        except Exception as e:
            r4 = {"fwd_ms": 0, "fwd_TFLOPS": 0, "fwd_MFU%": 0,
                   "bwd_ms": 0, "bwd_TFLOPS": 0, "bwd_MFU%": 0}
            print(f"  ④ sonic-moe e2e: SKIP ({str(e)[:80]})")
        row["moe_fwd_ms"] = r4["fwd_ms"]; row["moe_fwd_TFLOPS"] = r4["fwd_TFLOPS"]; row["moe_fwd_MFU%"] = r4["fwd_MFU%"]
        row["moe_bwd_ms"] = r4["bwd_ms"]; row["moe_bwd_TFLOPS"] = r4["bwd_TFLOPS"]; row["moe_bwd_MFU%"] = r4["bwd_MFU%"]

        # gaps
        if r1["TFLOPS"] > 0 and r2["TFLOPS"] > 0:
            row["gap_2x_3x"] = round((r2["TFLOPS"] / r1["TFLOPS"] - 1) * 100, 1)
            print(f"     gap(2.x→3.x): {r2['TFLOPS']/r1['TFLOPS']:.2f}x ({row['gap_2x_3x']:+.1f}%)", end="")
        else:
            row["gap_2x_3x"] = 0
        if r2["TFLOPS"] > 0 and r3["TFLOPS"] > 0:
            row["gap_qk_dg"] = round((r3["TFLOPS"] / r2["TFLOPS"] - 1) * 100, 1)
            print(f"  gap(qk→dg): {r3['TFLOPS']/r2['TFLOPS']:.2f}x ({row['gap_qk_dg']:+.1f}%)", end="")
        else:
            row["gap_qk_dg"] = 0
        print()

        results.append(row)

    # ── save JSON ──
    with open(out_json, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "peak_TFLOPS": PEAK_TFLOPS, "results": results}, f, indent=2)

    # ── markdown ──
    cols = [k for k in results[0].keys() if k not in ("proj",) and not k.startswith("gap")]
    lines = [
        "# Four-way Grouped GEMM Benchmark",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')} | H200 BF16 peak: {PEAK_TFLOPS} TFLOPS",
        f"Token dist: Dirichlet(α=0.5)",
        "",
        "| E | T | avg_M | K | N | ① C2.x | ② quack | ③ dg | ④ MoE FWD | ④ MoE BWD | 2x→3x |",
        "|---|---|-------|---|---|--------|---------|------|----------|----------|-------|",
    ]
    for r in results:
        lines.append(
            f"| {r['E']} | {r['T']} | {r['avg_M']} | {r['K']} | {r['N']} | "
            f"{r['c2x_TFLOPS']}TF({r['c2x_MFU%']}%) | "
            f"{r['qk_TFLOPS']}TF({r['qk_MFU%']}%) | "
            f"{r['dg_TFLOPS']}TF({r['dg_MFU%']}%) | "
            f"{r['moe_fwd_TFLOPS']}TF({r['moe_fwd_MFU%']}%) | "
            f"{r['moe_bwd_TFLOPS']}TF({r['moe_bwd_MFU%']}%) | "
            f"{r['gap_2x_3x']:+.1f}% |"
        )

    # averages (only for non-zero values)
    def safe_avg(key):
        vals = [r[key] for r in results if r[key] > 0]
        return sum(vals) / len(vals) if vals else 0

    avg_c2x = safe_avg("c2x_MFU%")
    avg_qk = safe_avg("qk_MFU%")
    avg_dg = safe_avg("dg_MFU%")
    avg_moe_fwd = safe_avg("moe_fwd_MFU%")
    avg_moe_bwd = safe_avg("moe_bwd_MFU%")

    lines.extend([
        "",
        "## Summary",
        f"- ① CUTLASS 2.x (ubiq_gmm): **{avg_c2x:.1f}% MFU**",
        f"- ② quack CUTLASS 3.x (JIT): **{avg_qk:.1f}% MFU**",
        f"- ③ deep_gemm: **{avg_dg:.1f}% MFU**" if avg_dg > 0 else "- ③ deep_gemm: **SKIP (shape not compiled)**",
        f"- ④ sonic-moe e2e: FWD **{avg_moe_fwd:.1f}% MFU** / BWD **{avg_moe_bwd:.1f}% MFU**" if avg_moe_fwd > 0 else "- ④ sonic-moe e2e: **SKIP**",
        "",
    ])
    if avg_c2x > 0 and avg_qk > 0:
        lines.append(f"2.x → 3.x (quack JIT): **{avg_qk/avg_c2x*100-100:+.1f}%** MFU change")

    with open(out_md, "w") as f:
        f.write("\n".join(lines))

    print(f"\n{'='*85}")
    print(f"Results: {out_json} / {out_md}")
    print(f"  ① {avg_c2x:.1f}% → ② {avg_qk:.1f}% → ③ {avg_dg:.1f}% → ④ FWD {avg_moe_fwd:.1f}% / BWD {avg_moe_bwd:.1f}% MFU")
    print(f"{'='*85}")
