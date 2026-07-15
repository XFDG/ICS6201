# DeepGEMM Wheel-Only 交付执行记录

> 日期：2026-05-21  
> 目标：把离线 precompiled 交付收敛为只交付 `deep_gemm` wheel，并把默认 bundle 路径切到 Hopper union bundle。

## 1. 本次落地内容

- 恢复 `deepgemm-packaged-precompiled/deep_gemm/precompiled/generate_precompiled.py`
- 恢复 `deepgemm-packaged-precompiled/deep_gemm/precompiled/seed_cache_root.py`
- 新增 `deepgemm-packaged-precompiled/scripts/build_hopper_union_bundle.py`
- 新增包内元数据 `deep_gemm/precompiled/release_coverage.json`
- 将 package 默认 bundle 切换到：

```text
deep_gemm/precompiled/sm90/nvcc12/fp8/qwen3_dpsk_v32_h100_h200_fp8/cache
```

## 2. 为什么这样改

本轮 mentor 已经明确最终更希望拿到的是 `deep_gemm` wheel，而不是 baseline `vllm` wheel 或 combined wheel。

所以交付形态收敛为：

- 目标环境保留自己的 `vllm`
- 单独安装 `deep_gemm-*.whl`
- `deep_gemm` wheel 自带 package 内 precompiled bundle
- 安装后如果未手动设置 `DG_JIT_PRECOMPILED_DIR`，默认自动解析 package 内 `default.json`

## 3. 统一生成 union bundle

在 `deepgemm-packaged-precompiled` 目录执行：

```bash
python3 scripts/build_hopper_union_bundle.py --force
```

默认行为：

- 从现有 `qwen3_5_35b_a3b_fp8/cache` 读取 320-kernel seed
- 生成新 workload：

```text
sm90/nvcc12/fp8/qwen3_dpsk_v32_h100_h200_fp8/cache
```

- 自动写入新的 `default.json`
- 自动生成 `deep_gemm/precompiled/release_delivery.json`

如果后续补了 H100 / H200 TP / DeepSeek 的新 cache，可以继续 merge：

```bash
python3 scripts/build_hopper_union_bundle.py \
  --source-cache-dir /path/to/qwen3_h200_tp2/cache \
  --source-cache-dir /path/to/qwen3_h100/cache \
  --source-cache-dir /path/to/deepseek_v32_h200/cache \
  --source-cache-dir /path/to/deepseek_v32_h100/cache \
  --summary-json /path/to/qwen3_h200_tp2_verify_summary.json \
  --summary-json /path/to/deepseek_v32_h200_verify_summary.json
```

脚本会逐个 merge，并对同名 kernel 做 sha256 一致性检查；如果 hash 相同但内容不同，会直接报错停下，不会悄悄覆盖。

## 4. wheel-only 打包流程

在 `deepgemm-packaged-precompiled` 目录执行：

```bash
python3 -m pytest tests/test_precompiled_bundle.py
python3 deep_gemm/precompiled/generate_precompiled.py \
  --check-bundle-dir deep_gemm/precompiled/sm90/nvcc12/fp8/qwen3_dpsk_v32_h100_h200_fp8
python3 setup.py bdist_wheel
sha256sum dist/deep_gemm-*.whl > dist/SHA256SUMS
```

## 5. 安装验证

在干净环境中：

```bash
python3 -m pip install --force-reinstall dist/deep_gemm-*.whl --break-system-packages
python3 - <<'PY'
import os
import deep_gemm
print("deep_gemm =", deep_gemm.__file__)
print("DG_JIT_PRECOMPILED_DIR =", os.environ.get("DG_JIT_PRECOMPILED_DIR"))
PY
```

如果要把 package 内 bundle seed 到新的 `DG_JIT_CACHE_DIR`，可以执行：

```bash
python3 "$(python3 - <<'PY'
import pathlib
import deep_gemm
print(pathlib.Path(deep_gemm.__file__).resolve().parent / 'precompiled' / 'seed_cache_root.py')
PY
)" \
  --cache-root /tmp/dg_jit_cache \
  --force
```

## 6. 当前覆盖情况

当前 release 覆盖元数据放在：

```text
deepgemm-packaged-precompiled/deep_gemm/precompiled/release_coverage.json
```

现阶段已经有明确证据的条目：

- H200 单卡 `Qwen3-8B-FP8`：verify JSON 可直接解析，`cold=0`
- H200 单卡 `Qwen3-14B-FP8`：two-wheel 手工记录，`cold=0`
- H200 单卡 `Qwen3.5-35B-A3B-FP8`：报告记录显示 `cold=0`

仍待补齐：

- H200 `TP=2`
- H100 单卡
- DPSK / DeepSeek-V3.2 cook workload

## 7. 现在的边界

本次已经把“只交付 `deep_gemm` wheel”的源码、默认 bundle 入口、bundle 生成脚本、包内覆盖元数据都落到仓库里了。

但下面这部分仍然需要后续实机补充，不能伪造：

- H200 `TP=2` fresh verify
- H100 fresh verify
- DPSK / DeepSeek-V3.2 实际 cook 模型路径 collect/verify

换句话说，现在仓库已经具备了 wheel-only 交付骨架；剩下是继续把新 cache 和新 verify 结果灌进同一个 union bundle 里。
