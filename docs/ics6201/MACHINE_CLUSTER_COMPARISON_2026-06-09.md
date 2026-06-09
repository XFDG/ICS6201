# ICS6201 / 算子开发机器与 GPU 集群对比

> 更新时间：2026-06-09  
> 口径：当前上海 CPU/GPU 机器为实测；历史 H200 环境来自本地文档；北京 4x H100 来自用户描述，版本待补查；4090x4 与 A100x8 为常见集群参考规格。

## 1. 总结

当前最适合继续承载 ICS6201 正式训练、H200 MoE/GEMM profiling、vLLM/DeepGEMM 验证的是 **上海 8x H200 GPU 机**。它有 8 张 141GB H200，总显存约 1128GB，总 HBM 带宽约 38.4TB/s；相比 4x H100，它主要强在显存容量和带宽，相比 A100x8 又强在 Hopper FP8/Transformer Engine 与 HBM3e。

CPU 机负责下载、构建、文档、数据准备和远程调度；GPU 机负责真实训练、推理、profiling。4090x4 更适合作为低成本开发/小模型调试集群，不适合大模型多卡训练基线；A100x8 是成熟稳定的训练平台，但缺 FP8，长上下文和大 batch 场景不如 H100/H200。

![Cluster capability comparison](assets/machine_cluster_comparison_2026-06-09.png)

## 2. 当前上海机器

### 2.1 CPU 机器：`zhaoye-cpu-shanghai-0`

| 项目 | 配置 |
|------|------|
| 主机 | `zhaoye-cpu-shanghai-0` |
| CPU | 2x Intel Xeon Platinum 8480+ |
| CPU 线程 | 224 logical CPUs |
| 内存 | 2.0TiB |
| 共享存储 | `/volume/yzhao04`，约 50T |
| GPU | 无 |
| Conda 环境 | `/volume/yzhao04/workspace/miniconda3/envs/{fly,ai}` |
| 主要角色 | 有外网下载、pip/conda 安装、wheel 构建、文档整理、数据准备、SSH 调度 GPU |

适合任务：

- 下载模型、数据集、Python wheel。
- 构建不需要 GPU 的包，或准备源码/patch。
- 生成 YOLO/COCO 配置、统计数据集、写报告。
- 通过 SSH 控制 GPU 训练，不直接承担 CUDA 计算。

不适合任务：

- 真实 GPU 训练、CUDA kernel benchmark、vLLM GPU 推理。

### 2.2 GPU 机器：`zhaoye-gpu-shanghai-0`

| 项目 | 配置 |
|------|------|
| 主机 | `zhaoye-gpu-shanghai-0` |
| CPU | 2x Intel Xeon Platinum 8558 |
| CPU 线程 | 192 logical CPUs |
| 内存 | 2.0TiB |
| GPU | 8x NVIDIA H200 |
| 单卡显存 | 143771MiB，约 141GB |
| 总显存 | 约 1128GB |
| Driver | 580.105.08 |
| 共享存储 | `/volume/yzhao04`，约 50T |
| 实际训练环境 | conda `fly`: Python 3.12.13, PyTorch 2.5.1+cu121, Ultralytics 8.4.56, Detectron2 0.6 |
| 备用/算子环境 | conda `ai`: Python 3.12.13, PyTorch 2.11.0a0 CUDA 13.1, Ultralytics 8.4.58, Detectron2 0.6 |

适合任务：

- ICS6201 目标检测训练：YOLO 系列、RT-DETR-L、Faster R-CNN Detectron2。
- H200 MoE/GEMM profiling：quack、sonic-moe、DeepGEMM、CUTLASS 对比。
- vLLM / DeepGEMM / FP8 Qwen 模型验证。
- 需要大显存、长上下文、大 batch、TP=2/4/8 的模型验证。
- 100B 级模型推理/训练实验，尤其是能利用 FP8 或大 KV cache 的场景。

限制与注意：

- GPU 机无外网或外网不稳定时，依赖安装应优先在 CPU 机完成。
- GPFS 共享存储小文件 I/O 会影响多卡 DataLoader，训练时要控制 workers、错峰启动、避免重复生成 COCO 标注。
- H200 与 H100 计算核心同属 Hopper，纯 compute-bound 小算子不一定显著快于 H100；优势主要来自 141GB HBM3e 和 4.8TB/s 带宽。

## 3. 历史北京 4x H100 机器

| 项目 | 配置 |
|------|------|
| 地点 | 北京旧 CPU/GPU 资源 |
| GPU | 4x NVIDIA H100，按用户描述 |
| 单卡显存 | 通常 80GB HBM3，具体形态待确认 |
| 总显存 | 约 320GB |
| 带宽参考 | H100 SXM 约 3.35TB/s/卡 |
| FP8 | 支持 Hopper FP8 Transformer Engine |
| 当前版本记录 | 本地文档未找到完整 driver / CUDA / PyTorch 记录，需补查旧机器日志 |

适合任务：

- Hopper FP8/Transformer Engine 训练和推理验证。
- 7B/13B/34B/70B 模型的 LoRA、SFT、推理、部分 TP 训练。
- Megatron-LM、vLLM、DeepSpeed、LLaMA-Factory 等标准多卡任务。
- 单机 4 卡 kernel / NCCL / TP 调试。

相对上海 8x H200 的差距：

- 总显存约 320GB，不适合直接承载 600GB+ 权重或极长上下文。
- 卡数只有 4，TP/PP/DP 组合空间小于 8x H200。
- H100 与 H200 compute 接近，但 H200 的显存容量和带宽更适合 memory-bound 任务。

## 4. 参考集群：4x RTX 4090

| 项目 | 配置 |
|------|------|
| GPU | 4x NVIDIA RTX 4090 |
| 单卡显存 | 24GB GDDR6X |
| 总显存 | 96GB |
| 单卡带宽 | 约 1.008TB/s |
| 互联 | 通常 PCIe，无 NVLink |
| FP8 Transformer Engine | 不作为数据中心 Hopper FP8 训练平台使用 |
| ECC/MIG | 无数据中心级 ECC/MIG |

适合任务：

- 小模型开发、CV 训练、YOLO/Detectron 小批量实验。
- 7B/13B QLoRA、LoRA、推理服务、算法原型验证。
- CUDA/Triton kernel 功能正确性开发，小规模 profiling。
- 不需要 NVLink 的 embarrassingly parallel 任务。

不适合任务：

- 70B+ BF16 全量训练。
- 需要大显存 KV cache 的长上下文推理。
- 严肃多卡通信优化、NCCL/NVLink 性能基线。
- 需要数据中心稳定性、ECC、MIG 隔离的任务。

## 5. 参考集群：8x A100 80GB

| 项目 | 配置 |
|------|------|
| GPU | 8x NVIDIA A100 80GB |
| 单卡显存 | 80GB HBM2e |
| 总显存 | 640GB |
| 单卡带宽 | 约 2.0TB/s |
| FP8 Transformer Engine | 不支持 |
| 优势 | 生态成熟、稳定、标准训练脚本兼容性好 |

适合任务：

- 传统 BF16/FP16/TF32 大模型训练和推理。
- 70B 级模型 TP/PP/ZeRO 训练与评估。
- CV 大模型、Diffusion、RLHF、embedding/reranker 批量任务。
- 需要 8 卡但不依赖 FP8 的稳定生产训练。

相对 H200 的差距：

- 无 Hopper FP8 Transformer Engine。
- 单卡显存 80GB，长上下文 KV cache 和大 batch 容量低于 H200。
- 显存带宽约 2.0TB/s/卡，低于 H100/H200。

## 6. 横向对比

| 资源 | 状态 | GPU | 总显存 | 主要版本 | 最适合任务 | 不建议任务 |
|------|------|-----|--------|----------|------------|------------|
| 上海 CPU | 实测可用 | 无 | 无 | conda `fly` / `ai` 存在共享盘 | 下载、安装、构建、文档、数据准备 | GPU 训练/推理 |
| 上海 GPU | 实测可用 | 8x H200 141GB | 1128GB | Driver 580.105.08；`fly` torch 2.5.1+cu121；`ai` torch 2.11 CUDA13.1 | ICS6201、H200 MoE/GEMM、vLLM、长上下文、大模型 TP | 过多小文件并发 I/O、不控 workers 的多任务训练 |
| 北京旧 GPU | 用户描述，版本待补 | 4x H100 80GB | 320GB | 待确认 | Hopper FP8、4卡 TP、70B 内任务 | 600GB+ 权重、超长上下文 |
| 4090x4 | 参考集群 | 4x RTX 4090 24GB | 96GB | 常见 CUDA 12.x/PyTorch cu12x | 小模型、LoRA/QLoRA、CV、原型开发 | 大模型全参训练、NVLink/NCCL 基线 |
| A100x8 | 参考集群 | 8x A100 80GB | 640GB | 常见 CUDA 11.8/12.x/PyTorch 稳定栈 | BF16/FP16 大模型训练、成熟生产任务 | FP8 Hopper 专项、H200 级长上下文 |

## 7. 模型/任务选择建议

| 任务/模型 | 首选资源 | 备选资源 | 原因 |
|-----------|----------|----------|------|
| ICS6201 目标检测全量训练 | 上海 8x H200 | A100x8 / H100x4 | H200 显存富余，7 卡并行训练可保留 GPU0 |
| RT-DETR-L / Faster R-CNN Detectron2 | 上海 8x H200 | A100x8 | 数据读取和验证较重，H200 显存更安全 |
| YOLO 系列 / DDW-YOLO | 上海 8x H200 / 4090x4 | A100x8 | 小模型可在 4090 上开发，正式对比放 H200 |
| Qwen 7B/14B LoRA | 4090x4 | H100/H200/A100 | 低成本开发即可 |
| Qwen 32B/72B 推理或 SFT | H200/H100/A100 | 4090x4 仅量化/分片尝试 | 需要 80GB 级显存和更稳定互联 |
| Qwen3.5-35B-A3B-FP8 / DeepGEMM | 上海 H200 | 北京 H100 | Hopper FP8 + HBM，且本地已有 H200 经验 |
| 100B-180B 长上下文推理 | 上海 8x H200 | A100x8/H100x4 视量化而定 | H200 141GB 对 KV cache 和 batch 更友好 |
| DeepSeek-V3 级 600GB+ 权重实验 | 上海 8x H200 | A100x8 勉强、H100x4 不适合 | 8x H200 总显存最大，通信/overhead 仍需谨慎 |
| 通信融合 / mKernel / NCCL profiling | 上海 8x H200 | H100x4 | H200/H100 都是 Hopper；8 卡拓扑更接近实际场景 |
| 低成本 CUDA/Triton 功能开发 | 4090x4 | CPU 编译 + H200 验证 | 4090 迭代便宜，但最终性能必须上数据中心卡验证 |

## 8. 已知版本与环境清单

| 环境 | Python | PyTorch | CUDA runtime | 其他 |
|------|--------|---------|--------------|------|
| 上海 GPU `fly` | 3.12.13 | 2.5.1+cu121 | 12.1 | Ultralytics 8.4.56, Detectron2 0.6 |
| 上海 GPU `ai` | 3.12.13 | 2.11.0a0+eb65b36914.nv26.02 | 13.1 | Ultralytics 8.4.58, Detectron2 0.6 |
| 旧 4x H200 文档环境 | 3.12 | 2.10.0+cu128 / 2.9.1+cu128 口径 | 12.8/12.9 | 见历史 H200 文档 |
| 北京 4x H100 | 待确认 | 待确认 | 待确认 | 需要补旧机器日志 |
| 4090x4 | 参考 | 常见 PyTorch cu12x | 常见 CUDA 12.x | 具体集群需现场确认 |
| A100x8 | 参考 | 常见 PyTorch cu118/cu12x | 常见 CUDA 11.8/12.x | 具体集群需现场确认 |

## 9. 结论

1. **当前上海 8x H200 是主力平台**：显存、带宽、Hopper FP8 和 8 卡规模都最适合课程训练与算子/推理实验。
2. **CPU 机继续承担“外网 + 构建 + 文档 + 调度”角色**：不要在 GPU 机上重复做网络安装和重数据准备。
3. **北京 4x H100 可作为 Hopper 备选平台**：适合 4 卡 FP8/TP 实验，但版本信息需要补查。
4. **4090x4 适合低成本开发，不适合作为最终性能结论平台**。
5. **A100x8 适合成熟 BF16/FP16 训练，但缺 FP8 和 H200 级显存/带宽**。

## 10. 参考来源

- 本地文档：`docs/ics6201/PROJECT_SUMMARY.md`
- 本地文档：`docs/base_environment/machine_env_check_2026-05-18.md`
- 本地文档：`docs/groued_gemm&sonic_moe/references/h200_gemm_performance_summary_2026-06-05.md`
- 本地文档：`docs/ohter/h200-recommendations.md`
- NVIDIA A100 product page: https://www.nvidia.com/en-us/data-center/a100/
- NVIDIA A100 datasheet: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf
- H100/H200 reference specs used for comparison: https://www.runpod.io/articles/guides/nvidia-h100 , https://www.runpod.io/articles/guides/nvidia-h200-gpu
- RTX 4090 reference specs: https://www.techspot.com/specs/gpu/252744-nvidia-geforce-rtx-4090.html

