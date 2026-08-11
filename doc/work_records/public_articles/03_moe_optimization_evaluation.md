# MoE 推理优化怎么测才可信：正交实验、全矩阵扫描与数值验收

> 本文由真实工程实践抽象而来。模型、硬件、软件环境、实验规模和结果均已匿名化；文中的归一化数字只用于说明方法，不对应任何具体生产系统。

## 结论先行

评测 MoE 推理优化，难点不是跑出更快的数字，而是证明收益来自目标改动，覆盖真实 shape，且没有以数值错误或局部退化为代价。

一套可信评测至少应做到：

1. 用 `baseline / 仅 A / 仅 B / A+B` 的 2×2 因子实验拆开贡献。
2. 四组共享相同的确定性、编译、缓存和测量前提。
3. 分开观察 prefill 与 decode，并扫描 batch、上下文和 TP 规模。
4. 同时报告收益区、持平区和退化区，不只展示峰值。
5. 先完成算子、模型数值和生成轨迹的分层验收，再讨论性能。

下面以两类通用优化为例：A 是 MoE 专家计算后端，B 是长上下文 decode 的 Attention 调度。

## 1. 为什么要做 2×2 正交实验

最小实验设计如下：

| 组别 | MoE 优化 A | Attention 优化 B | 回答的问题 |
|---|---:|---:|---|
| Baseline | 关 | 关 | 统一基线是什么 |
| A only | 开 | 关 | A 的独立贡献 |
| B only | 关 | 开 | B 的独立贡献 |
| A+B | 开 | 开 | 两者能否叠加 |

```text
                    B 关闭              B 开启
               ┌──────────────┬──────────────┐
A 关闭         │   Baseline   │    B only    │
               ├──────────────┼──────────────┤
A 开启         │    A only    │     A+B      │
               └──────────────┴──────────────┘

A 主效应 = A only - Baseline
B 主效应 = B only - Baseline
交互项   = (A+B - A only) - (B only - Baseline)
```

组合收益不严格等于两个独立收益之和，并不自动说明实现有错。A 加速后，B 可能成为新瓶颈；两者也可能争用带宽。2×2 设计让这些交互可见。

## 2. 控制变量：最容易被忽略的实验基础

最危险的干扰项通常是全局开关。例如确定性模式可能限制矩阵乘算法、改变归约顺序或切换 Attention 路径。如果只有部分组启用，它造成的整模型开销就会被错误归因给某项优化。

四组必须固定以下条件：

| 类别 | 必须一致的内容 |
|---|---|
| 模型 | 权重、精度、并行切分、最大序列长度 |
| 输入 | token、batch、输入/输出长度、采样参数 |
| 执行 | 编译模式、CUDA Graph、确定性模式、Attention 后端 |
| 资源 | GPU 数量、卡绑定、并发隔离 |
| 测量 | warmup、计时窗口、重复次数、统计方法 |
| 缓存 | 编译缓存、模型缓存、进程启动方式 |

推荐让运行器只接受 A、B 两个真正的实验变量：

```python
# 伪代码
for moe_opt in [False, True]:
    for attn_opt in [False, True]:
        env = common_environment.copy()
        env["MOE_OPT"] = moe_opt
        env["ATTN_OPT"] = attn_opt

        for case in case_matrix:
            warmup(case, env)
            samples = [benchmark(case, env) for _ in range(REPEATS)]
            save(case, moe_opt, attn_opt, median(samples))
```

还要记录目标 kernel 是否真正执行。开关为“开”不代表后端已加载；日志、算子计数或短 profile 至少应提供一种旁证。

## 3. 全矩阵应该怎样设计

单点 benchmark 只能说明一个点。要给出适用边界，至少要覆盖四个维度：

| 维度 | 推荐分桶 | 主要观察 |
|---|---|---|
| Phase | Prefill / Decode | 优化作用于哪段热路径 |
| Batch | Small / Medium / Large | 并行度与固定开销 |
| Context | Short / Medium / Long | token 数与 KV 深度 |
| TP | 单卡 / 多卡 | 通信是否稀释收益 |

矩阵可以按“冒烟—规则网格—拐点补测”三步展开。Prefill 记录输入吞吐或首 token 前计算时间；decode 应先 warmup，再统计稳定窗口，不能把 prefill 混入。端到端指标可以保留，但不能替代阶段归因。

TP 也要分两种口径：一是同一 TP 内优化相对 baseline 的收益，二是不同 TP 的绝对性能。若模型单卡可容纳，多卡张量并行可能因 All-Reduce 降低吞吐；此时 TP 的主要价值是扩展容量，而不是加速。

## 4. 用归一化趋势，而不是峰值讲故事

下面是完全匿名化的**示意数据**，只展示如何读结果：

| 场景 | Baseline | A only | B only | A+B |
|---|---:|---:|---:|---:|
| Prefill / Short | 1.00 | 1.16 | 1.00 | 1.15 |
| Prefill / Long | 1.00 | 1.11 | 0.99 | 1.09 |
| Decode / Short | 1.00 | 1.04 | 1.02 | 1.05 |
| Decode / Long | 1.00 | 1.05 | 1.21 | 1.27 |

这组数字均为**示意**，不能作为真实性能声明。它表达的分析方式是：

- A 更偏向 prefill，说明大 token 矩阵更利于专家计算优化发挥。
- B 的收益随上下文变长而扩大，符合长 KV decode 的特征。
- A+B 在部分场景接近叠加，但并非处处如此。
- B 在 prefill 持平或略负，因此不能宣传成“全阶段加速”。

最终结果适合画成归一化热力图，并保留负收益点，因为退化区正是选择策略和 fallback 的依据。

关键 case 应重复多轮并取中位数，同时保留波动范围。单次异常值只能触发复测，不能直接成为最好或最差结论。

## 5. 性能与数值要分层验收

推理正确性不是一个单一布尔值。推荐建立四层证据：

| 层级 | 比较对象 | 回答的问题 | 局限 |
|---|---|---|---|
| 算子级 | 固定输入的输出张量 | Kernel 是否接近参考实现 | 不覆盖整模型传播 |
| Prefill 数值级 | 全部目标 token 的 logprob | 单次前向是否漂移 | 不覆盖自回归反馈 |
| 分布级 | CE、relative-L2、容差内比例 | 漂移整体有多大 | 阈值需预先定义 |
| Decode 轨迹级 | token 与 logprob 序列 | 端到端是否稳定 | 首次分叉后会放大 |

常用指标为：

```text
relative_L2 = ||y_opt - y_ref||₂ / max(||y_ref||₂, ε)

element_relative_error[i]
  = |y_opt[i] - y_ref[i]| / max(|y_ref[i]|, ε)

within_ratio
  = count(element_relative_error < τ) / element_count
```

阈值 `τ` 必须在看结果前确定，不能为某次实验临时放宽。验收顺序可写成：

```python
# 伪代码
op_ok = compare_operator(reference, optimized, tau_op)
model_ok = relative_l2(prompt_logprobs) < tau_model
ce_ok = abs(ce_opt - ce_ref) < tau_ce
trajectory = compare_decode_trajectory(ref_run, opt_run)

if not op_ok:
    reject("operator gate failed")
elif not model_ok:
    hold("model-level drift needs analysis")
else:
    review_performance(trajectory, ce_ok)
```

Decode 的 greedy 轨迹在首个 token 分叉后，上下文已经不同。因此序列不一致是重要告警，却不能直接证明 MoE kernel 算错；应回到算子边界捕获同一输入，再与高精度参考比较。反过来，CE 几乎不变也不能抵消已经失败的数值阈值。

## 6. 八个常见误区

1. 把全局模式当作局部优化变量，导致四组不可比。
2. 只测一个峰值 case，不知道收益边界。
3. 混用不同模型、版本或计时口径的绝对吞吐。
4. 每组只跑一次，让缓存和调度抖动决定结论。
5. 正确性失败后仍宣传速度。
6. 看到 token 分叉就直接归因于 MoE。
7. 默认 TP 越大越快，忽略通信占比。
8. 删除负收益点，失去 fallback 设计依据。

发布结论前再检查：四组是否只有 A、B 不同；prefill/decode 是否分开；关键点是否复测；是否区分同 TP 相对收益与跨 TP 绝对性能；是否先过数值门禁；所有估算和归一化数字是否明确标为示意。

## 结语

可信的 MoE 评测是一条证据链：用正交实验确认“是谁带来收益”，用全矩阵说明“在哪里有效”，用分层数值验收回答“快的结果是否仍可信”。峰值数字适合做标题，控制变量、退化区和未通过项才决定一项优化能否落地。
