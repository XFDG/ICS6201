# Blackwell 多精度训练的系统化验证方法

> 脱敏分享版｜本文讨论 Blackwell 级 GPU 上 BF16、FP8、MXFP8、NVFP4 训练验证的方法与边界。工作负载使用随机初始化与合成数据，不对应任何真实模型权重或训练数据。

## 结论先行

验证新一代 GPU 的低精度训练能力，不能只跑一个 GEMM 或看到 step 能结束。一个可交付的验证矩阵至少要同时覆盖：

- 多种精度，以及与研究问题匹配的 GPU 规模；
- 真实训练拓扑中出现的算子 shape；
- 不同序列长度；
- forward、backward、optimizer step；
- output、dX、dW 的 finite 与数值检查；
- checkpoint save/load/restart；
- 吞吐、扩展效率和 GPU 板卡能效；
- 冷启动、正式稳态与长序列 pilot 的分离口径。

本次正式结果来自两套独立矩阵：四精度主性能矩阵只覆盖 1 GPU 与 8 GPU，24/24 个 30-step run 完成，共 720 optimizer steps；固定 global batch 的强扩展矩阵只比较 BF16 与 NVFP4，覆盖 1/2/4/8 GPU，另有 24/24 个 30-step run、720 optimizer steps 完成，两套矩阵的 run 间 step-time 变异系数均小于 1%。在代表性中等序列长度下，NVFP4 相对 BF16 的单卡/8-GPU 吞吐分别为 1.249/1.265 倍，8-GPU 板卡口径 tokens/J 为 1.434 倍；80/80 个 Transformer Engine forward+backward case 输出与梯度均为 finite；仅 BF16、8-GPU checkpoint 恢复对照中，下一 step 的 loss 相对差为 `1.38e-6`。

这些结果证明运行稳定性、相对性能和恢复链路，不证明真实数据上的收敛与最终质量。

## 1. 为什么只跑 Microbenchmark 不够

低精度 Tensor Core 的峰值通常很漂亮，但真实训练还有：

- 不同 M/N/K 的 shape 分布；
- attention 与 MLP projection 的不同算术强度；
- scaling、amax、量化/反量化开销；
- 梯度与权重更新；
- 通信和并行效率；
- checkpoint 中额外状态；
- 长序列下的显存和稳定性。

一个大 M GEMM 获得 2× 峰值，不意味着端到端训练也有 2×。反过来，小 M projection 可能因量化准备成本而比 BF16 更慢。

## 2. 用同一拓扑做公平对照

为了隔离硬件与精度路径，可用随机初始化和 mock data 重建同一训练拓扑。这样可以固定：

- layer 数与 hidden/FFN 关系；
- attention、MLP 与 normalization 结构；
- global/micro batch；
- sequence length；
- 并行策略；
- optimizer 与 checkpoint 状态。

随机权重的优势是可分享、可重复且不依赖真实数据；局限是无法回答真实 loss 曲线与质量问题。因此报告必须把“系统可运行”与“模型可收敛”分开。

## 3. 验证矩阵设计

### 3.1 精度维度

| 精度 | 主要目的 | 需要关注 |
|---|---|---|
| BF16 | 稳定基线 | 吞吐、显存、恢复 |
| FP8 | 成熟低精度对照 | scaling、amax、梯度有限性 |
| MXFP8 | block/microscaling 路径 | shape 约束、额外元数据 |
| NVFP4 | 更低位训练路径 | 数值误差、量化开销、能效 |

### 3.2 规模维度

本次使用的是约 7B 参数的 dense decoder 随机初始化 workload，不包含 MoE/专家并行。GPU 规模也不是所有精度的笛卡尔积，而是按研究问题拆成两套矩阵：

| 矩阵 | 精度 | GPU 规模 | 主要问题 |
|---|---|---|---|
| 主性能矩阵 | BF16、FP8、MXFP8、NVFP4 | 1、8 GPU | 同规模低精度吞吐与稳定性 |
| 固定 global batch 强扩展 | BF16、NVFP4 | 1、2、4、8 GPU | 1→8 GPU speedup 与并行效率 |

两套矩阵分别区分：

- 单卡 kernel/算子效率；
- 多卡强扩展；
- 固定 global batch 的扩展效率；
- 通信增加后的收益保留率。

### 3.3 序列长度

短、中、长序列应分别报告。本次 512/1024/2048/4096/8192 单卡序列扫描统一为 5-step capacity/performance pilot；其中 4096 另有主性能矩阵的 30-step×3 正式结果。更长序列也只做 5-step pilot。这些 pilot 只能说明显存、基础数值和短程性能没有立即失败，不能代替稳态或收敛验证。

### 3.4 Shape 回放

从训练拓扑提取 Transformer Engine 相关的 forward+backward shape，覆盖 output、dX、dW。受测集合包含 80 个代表性 case；每个 case 至少检查：

- shape 与 dtype；
- finite；
- reference 差异；
- 重复运行；
- forward/backward 都能完成。

## 4. 三种运行阶段必须分开

```text
环境 smoke
   -> 确认 import、设备、精度能力

短步 pilot
   -> 发现 OOM、NaN、shape 不支持、checkpoint 基础问题

正式 steady run
   -> warmup 后统计多 step 的吞吐、CV、功耗
```

把 pilot 与正式运行混在一起，会让编译、cache、首次分配和功耗爬升污染结果。

## 5. 性能口径

### 5.1 Throughput

主性能矩阵使用相同 topology、global batch、sequence length 与 optimizer 设置，只改变精度路径；四种精度只在 1 GPU 与 8 GPU 比较。报告相对值而非内部绝对 tok/s：

| 配置 | 相对 BF16 吞吐 |
|---|---:|
| 1-GPU NVFP4 | 1.249× |
| 8-GPU NVFP4 | 1.265× |

这些数字是端到端受测训练拓扑结果，不是某个 GEMM 的峰值。

### 5.2 Sequence crossover

在单卡 matched 条件下，低精度收益随序列长度发生 crossover。下表统一来自 5-step capacity/performance pilot；4096 另有正式 30-step×3 主矩阵结果，但表内仍按 pilot 口径理解：

| Sequence length | NVFP4 / BF16 throughput |
|---:|---:|
| 512 | 0.864× |
| 1024 | 0.903× |
| 2048 | 1.032× |
| 4096 | 1.249× |
| 8192 | 1.361× |

短序列时量化、scaling 和额外元数据成本无法摊薄；序列增长后，Tensor Core 吞吐优势才逐步占主导。这比单独报告一个“最高加速”更能说明低精度的适用区间。

### 5.3 Scaling efficiency

对固定工作量，可计算：

```text
speedup(n)    = throughput(n) / throughput(1)
efficiency(n) = speedup(n) / n
```

低精度单卡更快，不一定拥有更高多卡效率；如果通信比例随计算缩短而上升，扩展效率可能更低。

在独立的固定 global batch 强扩展矩阵中，只比较 BF16 与 NVFP4 的 1/2/4/8 GPU。1→8 GPU 对照中，BF16 speedup 为 5.569×、效率 69.6%；NVFP4 speedup 为 4.826×、效率 60.3%。这并不与 NVFP4 的 8-GPU 绝对吞吐优势矛盾：低精度缩短了计算，使通信和固定开销占比更高。

### 5.4 Stability

四精度 1/8-GPU 主性能矩阵为 24/24 个 run、720 optimizer steps；BF16/NVFP4 1/2/4/8-GPU 强扩展矩阵另有 24/24 个 run、720 optimizer steps。两套正式矩阵的 run 间 step-time CV 均小于 1%。CV 用于说明各自口径下的稳态波动，不代表跨环境可重复性；仍需保留软件版本、时钟、功耗策略和 warmup 口径。

## 6. 能效：必须注明统计边界

能效可写为：

```text
tokens/J = processed_tokens / GPU_board_energy
```

8-GPU NVFP4 相对 BF16 的 GPU 板卡 tokens/J 为 1.434×。这里的分母只含 GPU board energy，不含：

- CPU 与主机内存；
- 网络交换；
- 存储；
- 风扇与制冷；
- 集群基础设施。

因此它是板卡级能效，不是数据中心 PUE 口径，更不能直接换算成总成本。

## 7. 数值门禁：Finite 只是第一层

### 7.1 Operator 层

80/80 个 forward+backward shape 的 output、dX、dW 均为 finite。这可以发现 NaN/Inf 和不支持路径，但不能证明误差足够小。

下一层应记录：

- max absolute error；
- relative L2；
- cosine similarity；
- 不同尺度与异常值输入；
- 累积多层后的误差。

受测 NVFP4 局部算子的 relative-L2 在部分 shape 上可达到约 17.7%。这再次说明 finite 只是最低门槛，不能替代误差预算和收敛实验。

### 7.2 Step 层

短步训练检查 loss finite、gradient finite、无 skipped update。随机数据上的 loss 数值只用于比较恢复连续性，不能代表真实任务质量。

### 7.3 训练层

真正的 convergence parity 需要真实数据、更长训练、相同 seed/数据顺序和质量评测。本轮未覆盖，因此明确不做收敛声明。

## 8. Checkpoint 必须测试“下一步一致”

仅能保存和加载文件不够。完整恢复测试应为：

```text
Run A: step 0 -> 1 -> 2 -> 3
Run B: step 0 -> 1 -> 2 -> save
       reload -> step 3
compare(A.step3, B.step3)
```

完整方案应恢复模型、optimizer、scheduler、随机状态，以及低精度运行所需的元数据。本次实际执行的恢复对照仅为 BF16、8 GPU：下一 step loss 相对差为 `1.38e-6`，说明该 BF16 链路的状态连续性良好；它不证明 FP8、MXFP8 或 NVFP4 checkpoint 已验证，也不等同于长周期 restart 完全验证。

## 9. Shape-dependent 收益

低精度收益并不均匀：

- 小 M attention projection 可能被量化准备成本主导；
- 较大 M 的 gate/up/down projection 更容易摊薄开销；
- 通信占比高时，算力提升无法线性转化为 step 加速；
- 不同 precision 的支持矩阵和 fallback 也不同。

因此，从这些结果出发，更值得后续评估的方案是按 shape 和算子族选择精度，而不是全局“一键换成最低位”。这是一项实验启发的策略建议，并非本次已经实现或验证的 selective-precision runtime。

## 10. 报告结果时的边界

可以说：

- 四精度已覆盖 1/8 GPU 主性能矩阵，BF16/NVFP4 已覆盖 1/2/4/8 GPU 强扩展；
- 512～8192 序列扫描和更长序列属于 5-step pilot，checkpoint 仅验证 BF16、8 GPU；
- 正式运行稳定，给定拓扑下 NVFP4 有明确相对吞吐和板卡能效收益；
- 代表性 TE shape 的 output/dX/dW 均 finite。

不能说：

- NVFP4 已证明与 BF16 收敛等价；
- 1.265× 可外推所有模型或序列长度；
- 四种精度都已完成 1/2/4/8 GPU 强扩展或低精度 checkpoint；
- board-level tokens/J 等于整机或集群能效；
- 5-step 超长序列 pilot 等于长跑稳定。

## 11. 可复用检查表

- [ ] 是否在同一 topology、batch、sequence 和 optimizer 下比较？
- [ ] 是否覆盖 BF16/FP8/MXFP8/NVFP4 的 forward 与 backward？
- [ ] 是否区分 smoke、pilot 和正式稳态？
- [ ] 是否从真实拓扑抽取 shape，而不是只测理想 GEMM？
- [ ] output、dX、dW 是否检查 finite 与误差？
- [ ] 是否明确哪些精度覆盖 1/2/4/8 GPU，而不是暗示全组合已测？
- [ ] checkpoint 是否比较恢复后的下一 step，并注明实际覆盖的精度？
- [ ] 能效是否明确只含 GPU board？
- [ ] 是否明确随机数据不能证明收敛？

## 12. 总结

低精度平台验证是一项系统工程：峰值算力只是起点，真正的交付需要贯通 shape、forward/backward、并行扩展、稳定性、checkpoint 和能效。本次 dense decoder 结果必须按矩阵解读：四精度正式对照限于 1/8 GPU，1/2/4/8 GPU 强扩展限于 BF16/NVFP4，序列扫描是 5-step pilot，checkpoint 仅覆盖 BF16。随机初始化与合成数据适合建立可重复的基础矩阵，但必须诚实地停在“系统与数值链路验证”这一层；按 shape 选择精度仍是后续建议，不能跨越到已实现策略、真实收敛或模型质量结论。
