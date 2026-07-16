# LLMQRT H200 / Tensor Parallel 开发日志

> 持续更新文档。只记录已经观察到的事实；计划项与已验证结果分开标注。

## 目标

1. 将原先 RTX 5060 Laptop 环境中的 LLMQRT 迁移到 2×NVIDIA H200（SM90）。
2. 先验证单卡正确性，再实现并验证真正的 TP=2；不将 `device_map` 分层加载表述为 Tensor Parallel。
3. 修复或替换在 SM90 上非法访存的手写 AWQ CUDA kernel。
4. 接入 `/volume/pt-train/models` 中的另一款模型，验证可读的对话输出。

## 环境基线

| 项目 | 值 |
|---|---|
| GPU | 2×NVIDIA H200，SM90 |
| Driver | 570.86.15 |
| CUDA toolkit | 13.0 |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| Transformers | 4.54.0 |
| 持久环境 | `/volume/pt-train/users/zhaoye/envs/gpu-cu130` |

## 2026-07-16：TP=1 基线

状态：**成功**。

模型：`/volume/pt-train/users/zhaoye/models/Qwen2.5-0.5B-Instruct-AWQ`

用户复跑结果：

```text
GPU: NVIDIA H200
CUDA: 13.0
generated_tokens=64
decode_tok_s=10.27
peak_gib=0.68
```

成功判据均满足：模型类型识别、24 层量化 Linear 替换、权重加载、GPU forward、64-token 自回归生成、统计输出。回答后半段质量欠佳是 0.5B 模型能力/生成策略问题，不代表运行失败。

注：旧脚本直接 tokenize 普通字符串，没有套模型的 chat template，容易让 Instruct 模型续写出 `Human:` 等训练格式；现已改为 `apply_chat_template(..., add_generation_prompt=True)`。旧字段 `decode_tok_s` 实际包含 prefill 和首 token，已改名为更准确的 `generation_tok_s`，在拆分 TTFT/steady-state decode 前不把它作为纯 decode 指标。

修复后的默认路径为 `LLMQRT_AWQ_BACKEND=cuda`。H200 端到端回归中，模型生成 31 个 token 后主动 EOS，`generation_tok_s=11.93`，峰值显存 0.68 GiB；输出为可读中文。`LLMQRT_AWQ_BACKEND=torch` 仍保留为数值排查和新 shape 的安全回退。

## SM90 CUDA kernel 状态

状态：**已修复并验证**。

- 扩展已用 `-gencode=arch=compute_90,code=sm_90` 编译并可加载。
- AWQ 激活必须为 FP16；用 BF16 会得到明确的 dtype 错误。
- 原手写 AWQ CUDA 路径在真实生成中触发 `cudaErrorIllegalAddress`。
- `compute-sanitizer` 将错误定位到 decode 阶段的 `gemv_kernel_g128`：Qwen MLP `IC=896, OC=4864` 时，warp tile 按 8 个 group 读取，而实际只有 7 个 group；最后四个 lane 在读取 scale/zero metadata 时越界。
- 该问题不是 SM90 指令不兼容，而是原 kernel 缺少尾部 group 边界保护；旧 GPU 上也存在潜在越界，只是 H200 的内存布局使它稳定暴露。
- 已增加 `group_idx >= IC / group_size` 保护，使用调用 tensor 的当前 device、PyTorch 当前 CUDA stream，并在 launch 后立即检查错误。
- 独立 memcheck 显示现有 `gemv_coalesced_v2`、普通 GEMM 和 double-buffer GEMM 在目标模型四类 Linear shape 上均无越界；默认 decode 路径切换为 coalesced v2，原 GEMV 边界修复仍保留作回归。
- 原故障 shape `M=1, K=896, N=4864` 在修复后的原 GEMV 上运行两轮，`compute-sanitizer` 报告 `ERROR SUMMARY: 0 errors`。
- 同一 shape 的原 GEMV 与默认 coalesced v2 对 PyTorch 解量化参考实现均通过数值比较；随机压力输入最大绝对误差为 0.003906（FP16，`rtol=atol=0.02`）。
- 修复后的默认 CUDA 路径完成真实模型自回归生成，`RESULT tp=1 backend=llmqrt-awq-cuda status=PASS`。
- 自定义 CUDA kernel 仅实现 `group_size=128`；其他合法 AWQ group size 会自动转入解量化加 `torch.matmul`，避免返回未初始化输出。
- 为避免 GPU 环境中已安装的旧 `runtime-0.1` egg 抢占本地源码，补充了 `runtime_refact/__init__.py`；启动脚本始终将仓库根目录置于 `PYTHONPATH` 首位。

## TP=2 状态

状态：**Transformers BF16 与 LLMQRT AWQ 两条真 TP=2 路径均已验证**。

验收标准：

- 两个独立进程各绑定一张 H200；
- 每张卡只持有目标 Linear 的一个 tensor shard；
- 前向中存在必要的 collective（例如 all-reduce/all-gather）；
- TP=2 输出与 TP=1 基线在允许误差内一致；
- 日志能证明 `world_size=2`、rank/GPU 映射和每 rank 显存。

首次真 TP=2 验证：`/volume/pt-train/models/Qwen2.5-0.5B-Instruct`，通过 Transformers 4.54 内置 `tp_plan="auto"` 和 `torchrun --nproc_per_node=2` 运行：

```text
rank=0 gate_global=(4864,896) gate_local=(2432,896) q_global=(896,896) q_local=(448,896)
rank=1 gate_global=(4864,896) gate_local=(2432,896) q_global=(896,896) q_local=(448,896)
RESULT tp=2 backend=transformers-native status=PASS
```

这证明 q/gate 等 Linear 权重在两个 rank 上真实按 tensor 维度切分，而不是按层放到不同 GPU。该路径当前是项目内新增的 BF16/Hugging Face baseline；LLMQRT 自定义 AWQ packed weight 尚不能直接套用 Transformers 的 TP plan。

LLMQRT packed-AWQ TP=2 已实现并通过：

- Q/K/V、gate/up 按输出维切分；O/down 按输入维切分后做 NCCL all-reduce。
- rank 1 的 row shard 通过全局 `input_offset` 选择量化 group，处理 `448 % 128 != 0` 的跨 group 边界。
- 两个 rank 的 token IDs 完全一致，TP1/TP2 相同 prompt 的前 16 个 greedy token 一致。
- 目标模型切分后运行期单 rank 峰值显存约从 TP1 0.68 GiB 降到 TP2 0.39 GiB。每个 rank 当前仍先加载完整 checkpoint 再切 shard，因此该数字不代表加载峰值下降，也不代表已经能加载单卡放不下的模型。

```text
rank=0 qweight=(896,56) kweight=(896,8) oweight=(448,112) gate=(896,304) down=(2432,112)
rank=1 qweight=(896,56) kweight=(896,8) oweight=(448,112) gate=(896,304) down=(2432,112)
TOKEN_MATCH True
RESULT tp=2 backend=llmqrt-awq-torch status=PASS
```

该 AWQ TP=2 第一版使用安全的 PyTorch 解量化计算；由于 q/k/v shard 的输出维分别为 448/64，不满足现有手写 GEMM 的 `OC % 128 == 0`，CUDA-kernel TP2 还需要 padding 或新 kernel，不能直接开启。

当前 packed-AWQ 分片器面向已验证的 Qwen2.5 配置，并显式检查 attention head、KV head 和 packed tensor 维度能否被 TP=2 整除；不能据此宣称任意 Qwen2 checkpoint 均已适配。

性能边界：0.5B 小模型上通信开销大于分片收益；相同 prompt、16 个 greedy token 的 TP=1/TP=2 分别约为 8.00/7.17 token/s，因此这里只主张正确性、真分片和单 rank 运行期显存下降，不主张 TP=2 加速。更大模型需要另做同口径 benchmark。

## 其他模型接入

状态：**跨架构模型已验证**。

优先选择 runtime 已支持的 `qwen2`、`qwen3` 或 `llama`，并检查模型是否带有 LLMQRT 可解析的量化配置。普通 BF16/FP16 checkpoint 不能直接冒充 AWQ checkpoint。

- `/volume/pt-train/models` 内没有可直接供 LLMQRT 使用的 AWQ checkpoint。
- 已有中央 FP8 模型采用 `compressed-tensors` 或 block-FP8 配置，也与当前 LLMQRT checkpoint schema 不兼容。
- 选择 `/volume/pt-train/models/Llama-3.2-3B-Instruct` 作为第二模型：不同于原 Qwen 架构、带 chat template、BF16 权重完整，适合验证 H200 TP=2 对话。

Llama-3.2-3B-Instruct 真 TP=2 结果：

```text
gate_global=(8192,3072) gate_local=(4096,3072)
q_global=(3072,3072) q_local=(1536,3072)
load_s≈13.05, generated_tokens=32, generation_tok_s≈3.60
peak_gib=3.40 / rank
TOKEN_MATCH True
RESULT tp=2 backend=transformers-native status=PASS
```

最终交付命令用 `--max-new-tokens 64` 复跑时，LLMQRT AWQ TP=2 在生成 39 tokens 后主动 EOS，两个 rank 均为 9.05 token/s、运行期峰值 0.39 GiB，`TOKEN_MATCH True`。Llama TP=2 的最终态 16-token 复跑为 3.54 token/s、峰值 3.40 GiB/rank。TP=1 交互脚本也已用 `你好`、`q` 的输入序列验证，能够回答后正常退出。

### 32B 大模型验证

`/volume/pt-train/models/Qwen2.5-32B-Instruct` 是完整 BF16 checkpoint（17 shards，权重 61.03 GiB），已通过同一原生 TP=2 脚本：

```text
rank=0 gate_global=(27648,5120) gate_local=(13824,5120) q_global=(5120,5120) q_local=(2560,5120)
rank=1 gate_global=(27648,5120) gate_local=(13824,5120) q_global=(5120,5120) q_local=(2560,5120)
load_s=102.07, generated_tokens=32, generation_tok_s=2.12
peak_gib=32.02 / rank
TOKEN_MATCH True
RESULT tp=2 backend=transformers-native status=PASS
```

这条结果证明 32B 级完整模型可在 2×H200 上真实 tensor parallel 推理，不是仅加载小模型或用 `device_map` 按层切分。

### 72B 大模型验证

`/volume/pt-train/models/Qwen2.5-72B-Instruct` 是完整 72.706B 参数 BF16 checkpoint（37 shards，权重 135.43 GiB），也已通过 TP=2：

```text
rank=0 gate_global=(29568,8192) gate_local=(14784,8192) q_global=(8192,8192) q_local=(4096,8192)
rank=1 gate_global=(29568,8192) gate_local=(14784,8192) q_global=(8192,8192) q_local=(4096,8192)
load_s=234.95, generated_tokens=16, generation_tok_s=1.61
peak_gib=70.09 / rank
TOKEN_MATCH True
RESULT tp=2 backend=transformers-native status=PASS
```

32B 双卡交互入口也已验证：加载后输入中文问题得到可读回答，随后输入 `q` 正常退出；同一入口可用于 72B。

## 2026-07-16：Coder-32B 量化前后对照实验

状态：**成功**。目标模型固定为
`/volume/pt-train/models/Qwen2.5-Coder-32B-Instruct`，量化目标为
W4A16 AWQ、group size 128、asymmetric zero point、GEMM packing；不再用 0.5B
模型代表大模型性能。

公平对照协议已固化到 `scripts/benchmark_h200_tp2.py`：Dense FP16 与 AWQ
使用同一个源 tokenizer、同一个 chat-template prompt、batch size 1、8-token
warmup、64 个强制输出 token、greedy decoding、3 次重复。每次测量取两个 TP
rank 中较慢的 wall time，再报告 3 次中位数；`e2e_output_tok_s` 包含 prefill、
首 token 和 decode，因此不称为纯 decode 吞吐。脚本同时校验输入 ID 哈希、跨
rank token 一致性和三次重复 token 一致性，并分开记录加载/运行期的 allocated
与 reserved 显存。

最终 Dense FP16 量化前基线已经跑通：

```text
model=Qwen2.5-Coder-32B-Instruct, TP=2
prompt_tokens=54, new_tokens=64, repeats=3
elapsed_s=27.0101,25.2858,25.2277
median_s=25.2858, e2e_output_tok_s=2.5311
load_peak_allocated_gib_max=33.15, runtime_peak_allocated_gib_max=32.02
input_ids_sha256=a76729bf9134ce029a18c5c968462f3af987cf8adee6f5eac9d2f870fa121d3d
rank_token_match=True, repeat_token_match=True
```

量化后使用 LLMQRT 自定义 SM90 CUDA kernel 和同一份 TP=2 benchmark，最终
结果为：

```text
model=Qwen2.5-Coder-32B-Instruct-AWQ, TP=2, backend=cuda
prompt_tokens=54, new_tokens=64, repeats=3
elapsed_s=15.8681,14.3022,14.3015
median_s=14.3022, e2e_output_tok_s=4.4748
load_peak_allocated_gib_max=18.12, runtime_peak_allocated_gib_max=10.63
input_ids_sha256=a76729bf9134ce029a18c5c968462f3af987cf8adee6f5eac9d2f870fa121d3d
rank_token_match=True, repeat_token_match=True
```

| 指标 | Dense FP16 TP=2 | AWQ W4A16 TP=2 | 变化 |
|---|---:|---:|---:|
| 64-token 中位耗时 | 25.2858 s | 14.3022 s | -43.4% |
| E2E output 吞吐 | 2.5311 tok/s | 4.4748 tok/s | **+76.8% / 1.77×** |
| 运行期峰值显存/rank | 32.02 GiB | 10.63 GiB | **-66.8%** |
| 加载峰值显存/rank | 33.15 GiB | 18.12 GiB | -45.3% |
| checkpoint 目录 | 61.04 GiB | 18.02 GiB | **-70.5% / 3.39× 更小** |

这里的速度对比是相同模型、tokenizer、prompt、固定输出长度、warmup 和重复次数
下的实测；`e2e_output_tok_s` 仍包含 prefill 和首 token，不等同于纯 decode
吞吐。加载时间受 page cache 影响，不作为主要加速结论。

离线校准集从 LLMQRT 与本地 vLLM 源码中确定性抽取，避免 GPU 节点依赖外网。
当前正式配置为 128×512 tokens。正式量化是在上述代码与脚本完成后重新抽取，
实际使用 22 个本地源码文件，校准 token 哈希为：

```text
a3b2db5b454a553bec4da83f7d1987fc058fb48dab75020c41fe9bb89065669b
```

正式量化完成 64/64 层并正常退出：

```text
load_s=27.94
quant_s=3191.99（53 分 11.99 秒）
save_s=10.05
quant_peak_gib=72.72
RESULT quantize status=PASS
```

输出位于
`/volume/pt-train/users/zhaoye/models/Qwen2.5-Coder-32B-Instruct-AWQ`。
schema 审计结果：4 个 safetensors、18.0015 GiB 权重、1667 tensors；其中
FP16=771、INT32=896，448 组 `qweight/qzeros/scales` 完整配对，且量化
projection 中没有残留 dense `.weight`。配置确认为 AWQ GEMM、4 bit、group
128、asymmetric zero point。

SM90/TP=2 适配检查已经完成：Coder-32B 的 q、k/v、gate/up、o、down 局部
维度分别为 2560、512、13824、2560、13824，均是 128 的整数倍。对于这种
group-aligned row shard，分片器现在同时切分 `qweight/qzeros/scales`；对于
0.5B 的非对齐 shape，则继续保留全局 metadata 和 PyTorch fallback。新增的
3 个 CPU 单测覆盖 aligned、non-aligned 与 Coder-32B 精确 shape，均已通过。

量化工具使用项目隔离的 AutoAWQ 0.2.9，不安装其预编译 inference kernels；
后续推理仍使用本项目为 SM90 编译和修复的 CUDA 扩展。AutoAWQ 已停止维护，
因此先用 0.5B checkpoint 做 Transformers 4.54 兼容冒烟，确认通过后再启动
32B 正式量化。

0.5B 冒烟已经完整通过：24/24 层量化、标准 GEMM checkpoint 保存、schema
检查、LLMQRT 自定义 CUDA 加载和 8-token 真实生成均成功。为兼容当前
Transformers 4.54，隔离 overlay 补了两处最小修复：Catcher 代理 Qwen2 layer
的 `attention_type`；分块校准时 hidden states 与带 batch 维度的 causal mask
同步切片。

正式 32B 作业保持完整 128×512 校准规模。实测 `parallel-samples=16` 会让
旧 AutoAWQ 实现的大 batch 变慢，因此正式参数采用更快且显存稳定的 4；这只
改变校准 forward 的分批方式，不改变校准 token、AWQ 搜索算法或输出格式。

量化进行期间又加固了 SM90 推理扩展：普通 GEMM 与 double-buffer GEMM 现在
都在 PyTorch current CUDA stream 上启动，launch 后立即检查错误；C++ 入口会
检查 device、FP16/INT32 dtype、contiguous、packed shape、K32 与 split-K。
CUDA 13.0/SM90 重编译成功后，Coder-32B TP=2 的代表 shape 已通过：

```text
memcheck:  DB-GEMM M=8,  K=13824, N=5120  -> 0 errors
memcheck:  GEMV    M=1,  K=13824, N=5120  -> 0 errors
memcheck:  DB-GEMM M=54, K=2560,  N=5120  -> 0 errors
initcheck: DB-GEMM M=54, K=2560,  N=5120  -> 0 errors
synccheck: DB-GEMM M=54, K=2560,  N=5120  -> 0 errors
racecheck: DB-GEMM M=54, K=2560,  N=5120  -> 0 hazards
```

其中 K=13824 覆盖 down projection 的最大 row shard 和最后四个量化 group，
M=54 覆盖正式 benchmark prompt 的非完整 M16 尾 tile。复现入口为
`scripts/sanitize_awq_coder32b_sm90.sh`。

加载正式 checkpoint 并完成 TP=2 分片后，每个 rank 的 64×7=448 个量化
Linear 均通过 metadata 检查。首层 q/k/o/gate/down 的 decode GEMV 和 prefill
GEMM 共 10 组 CUDA-vs-PyTorch 参考对照全部通过；最大全局绝对误差 0.007812，
多数 shape 不超过 0.001953。row-parallel 的本地 partial output 与 NCCL
all-reduce 后结果分别比较，避免跨 rank 误差相消造成假通过。

交互入口也完成了真实 stdin 验收：打印 `[TP=2 AWQ READY]` 后输入中文问题，
生成 96 个可读中文 token，`e2e_output_tok_s=3.81`，再输入 `q` 正常退出。

## 固化命令

所有命令均在 GPU 节点执行。

构建 SM90 扩展：

```bash
cd /volume/pt-train/users/zhaoye/LLMQRT
source /volume/pt-train/users/zhaoye/envs/gpu-cu130/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=9.0
python setup_runtime.py build_ext --inplace
```

Coder-32B 量化前 Dense FP16 TP=2 benchmark：

```bash
bash scripts/benchmark_h200_tp2.sh \
  hf-fp16 \
  /volume/pt-train/models/Qwen2.5-Coder-32B-Instruct \
  --tokenizer /volume/pt-train/models/Qwen2.5-Coder-32B-Instruct \
  --warmup-tokens 8 --new-tokens 64 --repeats 3 --seed 20260716
```

重新执行正式量化（约 53 分钟；若输出目录非空，脚本会拒绝覆盖）：

```bash
bash scripts/quantize_coder32b_awq.sh \
  /volume/pt-train/models/Qwen2.5-Coder-32B-Instruct \
  /volume/pt-train/users/zhaoye/models/Qwen2.5-Coder-32B-Instruct-AWQ \
  --calib-samples 128 --calib-seq-len 512 \
  --parallel-samples 4 --chunk-memory-mib 512
```

checkpoint、SM90 sanitizer 与 TP=2 数值验证：

```bash
python scripts/inspect_awq_checkpoint.py \
  /volume/pt-train/users/zhaoye/models/Qwen2.5-Coder-32B-Instruct-AWQ

bash scripts/sanitize_awq_coder32b_sm90.sh

bash scripts/validate_awq_tp2_cuda.sh \
  /volume/pt-train/users/zhaoye/models/Qwen2.5-Coder-32B-Instruct-AWQ
```

Coder-32B 量化后 LLMQRT CUDA TP=2 benchmark：

```bash
bash scripts/benchmark_h200_tp2.sh \
  llmqrt-awq \
  /volume/pt-train/users/zhaoye/models/Qwen2.5-Coder-32B-Instruct-AWQ \
  --tokenizer /volume/pt-train/models/Qwen2.5-Coder-32B-Instruct \
  --awq-backend cuda \
  --warmup-tokens 8 --new-tokens 64 --repeats 3 --seed 20260716
```

Coder-32B AWQ 双卡交互对话（本轮最终推荐入口）：

```bash
bash scripts/chat_h200_tp2_awq.sh \
  /volume/pt-train/users/zhaoye/models/Qwen2.5-Coder-32B-Instruct-AWQ \
  --max-new-tokens 256
```

看到 `[TP=2 AWQ READY]` 后直接输入问题；输入 `q`/`quit`/`exit` 退出，
`clear` 清空历史，`mem` 查看两个 rank 的显存。

LLMQRT AWQ TP=1 / 交互对话：

```bash
bash scripts/run_h200_tp1.sh /volume/pt-train/users/zhaoye/models/Qwen2.5-0.5B-Instruct-AWQ --max-new-tokens 64
bash scripts/chat_h200_tp1.sh /volume/pt-train/users/zhaoye/models/Qwen2.5-0.5B-Instruct-AWQ --max_new_tokens 256
```

LLMQRT packed-AWQ TP=2：

```bash
bash scripts/run_h200_tp2_awq.sh /volume/pt-train/users/zhaoye/models/Qwen2.5-0.5B-Instruct-AWQ --max-new-tokens 64
```

第二模型 Llama-3.2-3B-Instruct 原生 BF16 TP=2：

```bash
bash scripts/run_h200_tp2_hf.sh /volume/pt-train/models/Llama-3.2-3B-Instruct --max-new-tokens 64
```

32B 大模型原生 BF16 TP=2：

```bash
bash scripts/run_h200_tp2_hf.sh /volume/pt-train/models/Qwen2.5-32B-Instruct --max-new-tokens 64
```

72B 大模型原生 BF16 TP=2：

```bash
bash scripts/run_h200_tp2_hf.sh /volume/pt-train/models/Qwen2.5-72B-Instruct --max-new-tokens 32
```

32B/72B 双卡交互对话：

```bash
bash scripts/chat_h200_tp2_hf.sh /volume/pt-train/models/Qwen2.5-32B-Instruct --max-new-tokens 256
# 或替换为：/volume/pt-train/models/Qwen2.5-72B-Instruct
```

## 简历建议表述

- **量化 Runtime 与后端适配**｜在支持 W4A16 AWQ、W8A8 SmoothQuant 与 FP8 的 PyTorch Extension Runtime 中，完成 FlashAttention-2/4 可配置接入，补齐 GQA 的 KV heads 展开、softcap mask 及 SDPA→Torch 回退；将 CUDA arch、ccache 与模型路径改为动态/可选配置，使工程可在本地 GPU 与 H200 环境构建部署。
- **H200 Kernel 与 Tensor Parallel**｜在公司 2×H200（SM90/CUDA 13.0）环境完成 AWQ Runtime 迁移；使用 compute-sanitizer 将 decode 非法访存定位到 `gemv_kernel_g128`：当 `K/group_size=7` 时 warp 仍按 8 组读取，导致末尾 lane 越界访问 scale/zero；增加 group 边界保护、current-stream launch 与错误检查后，代表 shape 的 memcheck/initcheck/synccheck/racecheck 均为 0 error/0 hazard；进一步实现 Qwen2 packed-AWQ TP=2，完成 Q/K/V、gate/up 列切分及 O/down 行切分与 NCCL all-reduce。
- **32B 量化部署与验证**｜基于 128×512 本地代码校准完成 Qwen2.5-Coder-32B W4A16 量化与双卡交付；在同模型、prompt、64-token、3 次重复的公平对照下，checkpoint 由 61.04 降至 18.02 GiB（-70.5%），E2E output 吞吐由 2.53 提升至 4.47 tok/s（+76.8%），运行期峰值显存由 32.02 降至 10.63 GiB/rank（-66.8%）；通过 10 组 CUDA/PyTorch 数值对照（最大绝对误差 0.007812）、跨 rank token 一致性及中文交互验收。

## 待办

- [x] SM90 kernel 最小复现和 compute-sanitizer 定位
- [x] 修复 AWQ CUDA GEMV 路径并做 sanitizer、数值和端到端回归
- [x] TP=2 最小实现和双卡验证
- [x] 第二模型端到端对话验证
- [x] 固化安装、构建、TP=1、TP=2 命令
- [x] Coder-32B Dense FP16 TP=2 量化前公平基线
- [x] Coder-32B 本地 128×512 校准与 W4A16 AWQ checkpoint
- [x] Coder-32B SM90 CUDA TP=2 数值、sanitizer、吞吐与交互验收
