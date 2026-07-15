# Native Sparse Attention: 稀疏记忆机制讨论

日期：2026-06-14

## 参考论文

| arXiv ID | 论文 | 公开链接 |
|---|---|---|
| 2510.11370 | Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers | [arXiv](https://arxiv.org/abs/2510.11370) |
| 2605.21312 | Frontier: Towards Comprehensive and Accurate LLM Inference Simulation | [arXiv](https://arxiv.org/abs/2605.21312) |
| 2512.02556 | DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models | [arXiv](https://arxiv.org/abs/2512.02556) |
| 2502.11089 | Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention | [arXiv](https://arxiv.org/abs/2502.11089) |
| 1701.06538 | Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer | [arXiv](https://arxiv.org/abs/1701.06538) |
| 2512.14080 | SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations | [arXiv](https://arxiv.org/abs/2512.14080) |
| 2603.05451 | FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling | [arXiv](https://arxiv.org/abs/2603.05451) |

## 问题：它的“稀疏记忆力”到底是什么？

我的理解：NSA 的“稀疏记忆”不是把历史 KV cache 简单删掉，也不是只保留最近窗口。它更像是给每个 query 动态构造一个小型工作记忆：

1. 全局粗粒度记忆：把长历史按块压缩成少量 summary token，让 query 至少能扫到全局轮廓。
2. 重要细节记忆：根据 query 对压缩块的注意力分数，选出 top-n 个重要连续块，再回到原始 KV cache 里取这些块的细粒度 token。
3. 近期局部记忆：永远保留最近 w 个 token，避免局部语法、短程依赖被压缩破坏。

最后，三个分支分别做 attention，再用可学习 gate 加权融合。也就是说，NSA 的核心不是“少看一点历史”，而是“先用便宜的全局索引找到该细看的历史，再加上最近窗口”。

## 机制拆开看

### 1. Token Compression：先给长历史做目录

历史 key/value 会被分成连续块。论文里用块长 `l`、stride `d`，并用带块内位置编码的可学习 MLP `phi` 把每个 key/value 块压成一个 compressed key/value。

这一步像给一本长书建立目录：不是把每个字都拿出来比对，而是先得到每一段的大意。这样 query 可以用很少的 attention 计算获得全局信号。

### 2. Token Selection：用目录分数找原文细节

query 先和 compressed keys 做 attention，得到每个压缩块的重要性分数。NSA 再把这些分数映射到 selection blocks 上，选择分数最高的 top-n 个连续块，并从原始 KV cache 里取出这些块的完整 key/value。

这个设计有两个关键点：

- 选择是按连续块取，不是随机 token 取。这样 GPU 可以做连续内存访问，Tensor Core 利用率更好。
- 在 GQA/MQA 场景里，同一个 KV 组内的 query heads 会共享 block selection，避免每个 head 选一套不同 KV，导致实际内存读取量又膨胀回去。

### 3. Sliding Window：最近内容不交给压缩分支赌运气

NSA 还固定加入最近 `w` 个 token。原因很朴素：局部模式学得快，而且对语言建模很重要。如果没有单独窗口，模型可能让压缩/选择分支偷懒地只学局部模式；单独窗口把局部依赖接走，压缩和选择分支就更容易专注长程信息。

### 4. Gated Output：三个记忆源按需混合

三个分支的输出不是简单拼接后丢给下一层，而是分别算 attention output，再通过可学习 gate 加权：

```text
output = gate_cmp * attention(query, compressed KV)
       + gate_slc * attention(query, selected raw KV)
       + gate_win * attention(query, recent window KV)
```

这让模型能对不同 query 调整“粗看全局 / 细看关键段 / 看最近上下文”的比例。

## 一个小例子：先用玩具参数看懂

假设历史上下文有 32 个 token，现在第 33 个 query 在问：

```text
retry_limit 最后被设置成多少？
```

我们用一个很小的 NSA 配置帮助理解：

| 参数 | 含义 | 例子值 |
|---|---|---|
| `l` | 压缩块长度 | 4 |
| `d` | 压缩 stride | 4 |
| `l'` | 选择块长度 | 4 |
| `n` | 选择 top-n 块 | 2 |
| `w` | 最近窗口 | 4 |

历史被压成 8 个 compressed tokens：

```text
B1=[1..4], B2=[5..8], B3=[9..12], B4=[13..16],
B5=[17..20], B6=[21..24], B7=[25..28], B8=[29..32]
```

query 先和这 8 个 compressed tokens 算 attention。假设分数最高的是：

```text
B3=[9..12]   // 初始配置里写过 retry_limit = 3
B6=[21..24]  // 后面覆盖成 retry_limit = 5
```

那么 selected branch 会回到原始 KV cache，取出 `[9..12]` 和 `[21..24]` 的完整 token。sliding branch 再固定取最近 `[29..32]`。compressed branch 仍然保留所有 8 个块摘要。

所以第 33 个 query 实际看的不是完整 32 个原始 token，而是：

```text
8 个压缩摘要 + 8 个被选中的原始细节 token + 4 个最近 token
```

## 论文参数例子：`l=32, d=16, l'=64, n=16, w=512`

现在把玩具参数换成论文效率分析里的真实参数。假设 decoding 时 KV cache 里已经有 `s=65536` 个历史 token，当前 query 要生成下一个 token。

### 1. Compression branch：用重叠块做全局目录

`l=32` 表示每个压缩块看 32 个连续 token，`d=16` 表示相邻压缩块每次向右滑 16 个 token。所以压缩块是重叠的：

```text
C0 = token[1..32]
C1 = token[17..48]
C2 = token[33..64]
C3 = token[49..80]
...
```

每个 `C_i` 会被一个可学习压缩函数 `phi` 压成 1 个 compressed KV。因为 stride 只有 block size 的一半，每个中间 token 大致会被两个压缩摘要覆盖。这个重叠设计的目的，是减少边界切块带来的信息断裂：一个关键信息如果刚好横跨 `[1..32]` 和 `[33..64]` 的边界，仍然可能被 `C1=[17..48]` 捕获到。

64k 历史下，压缩摘要数量按论文表格口径近似为：

```text
s / d = 65536 / 16 = 4096
```

严格按完整块计数是 `floor((s-l)/d)+1 = floor((65536-32)/16)+1 = 4095`，但论文 Table 4 用 `s/d` 作为 memory-access token-equivalent 的估算口径，所以讨论效率时看 `4096` 更贴近论文表。

这一步的直觉是：full attention 要直接翻 65536 页原文；compression branch 先生成 4096 条目录项，每条目录项摘要 32 个 token，并且目录项之间有重叠。

### 2. Selection branch：从目录分数映射回原文块

`l'=64` 表示原始 KV cache 会按 64-token 的 selection block 切块：

```text
S0 = token[1..64]
S1 = token[65..128]
S2 = token[129..192]
...
S1023 = token[65473..65536]
```

总共有：

```text
65536 / 64 = 1024 个 selection blocks
```

当前 query 先和 4096 个 compressed keys 做 attention，得到 4096 个压缩块分数。NSA 不想再额外做一次昂贵的“query 对 1024 个原始块打分”，于是复用 compression attention 的中间分数，把它们按空间重叠关系聚合成 selection block 的重要性分数。

用 `l=32,d=16,l'=64` 来直觉理解：一个 64-token selection block 覆盖 4 个 stride 位置，每个位置附近又有重叠的 32-token compression block。因此 `S_j` 的分数不是凭空算的，而是来自那些和 `S_j` 空间上重叠、相邻的 compressed attention 分数。query 如果强烈关注某几个目录项，和这些目录项覆盖同一区域的原文块就会被提高优先级。

然后 `n=16` 表示只保留分数最高的 16 个 selection blocks。每个 block 64 个 token，所以 selected raw KV 的访问量是：

```text
n * l' = 16 * 64 = 1024
```

注意这里选的是原始 KV，不是压缩摘要。也就是说 selected branch 会拿到 16 个最相关区域的完整细节。它像是先看目录，再翻到 16 个最相关页段读原文。

### 3. Sliding window branch：固定保留最近 512 token

`w=512` 表示不管 selection 选了哪里，NSA 都会额外保留最近 512 个 token：

```text
W = token[65025..65536]
```

这部分主要负责局部连续性：当前句子的语法、上一轮对话、刚刚定义的变量、刚出现的函数参数。它也避免 compression/selection 分支被短程模式“带偏”，让那两个分支更专注长程检索。

### 4. 合起来：为什么是 5632？

论文 Table 4 的 64k decoding memory access 是：

```text
compression tokens + selected raw tokens + window tokens
= s/d + n*l' + w
= 65536/16 + 16*64 + 512
= 4096 + 1024 + 512
= 5632
```

Full Attention 每步需要读取全部历史 KV：

```text
65536
```

因此只看 KV cache 读取量，理论比例是：

```text
65536 / 5632 = 11.64
```

这就是论文表里 64k decoding 预期 `11.6x` speedup 的来源。这里的 `5632` 是 token-equivalent 访问量，不等于“最终只记住 5632 个原始 token”。更准确地说，它由三种记忆拼出来：

| 分支 | 访问量 | 看到的信息 |
|---|---:|---|
| compression | `4096` | 整个 64k 历史的粗粒度目录 |
| selection | `1024` | top-16 个相关原文块，每块 64 token |
| window | `512` | 最近局部上下文 |
| 总计 | `5632` | 全局可扫、重点可细读、局部不断片 |

### 5. 具体到一个 64k 文档问答场景

假设 64k 上下文是一整个代码仓库和运行日志，当前 query 是：

```text
retry_limit 最后到底是多少？为什么测试还是重试 5 次？
```

历史里可能有这些位置：

```text
token[4200..4220]    README 里写默认 retry_limit = 3
token[18880..18910]  config/base.yaml 里写 retry_limit = 3
token[47360..47395]  config/prod.yaml 覆盖 retry_limit = 5
token[64120..64180]  最近日志里出现 "retry attempt 5/5"
```

NSA 的三分支大概会这样工作：

1. compression branch 扫 4096 个摘要。它不一定能保留每个数字的精确语义，但能让 query 发现“有几个区域和 retry_limit/config/retry attempt 相关”。
2. selection branch 根据这些摘要分数，把对应 64-token 原文块选出来。例如可能选中覆盖 `4200`、`18880`、`47360` 附近的几个 selection blocks。这样模型能读到精确配置值和覆盖关系。
3. sliding window branch 固定读最近 512 token。如果日志就在最近窗口里，模型不用赌 selection 是否选中日志区域，可以直接看到 `attempt 5/5`。
4. gate 决定三路输出的权重。问“最后到底是多少”时，selected branch 可能更重要；问“刚才日志为什么这样显示”时，window branch 权重可能更高；需要判断全局上下文有哪些配置文件时，compression branch 也会参与。

所以 NSA 的“稀疏记忆力”不是把 64k 历史粗暴裁成 5632 token，而是把 64k 组织成一种可检索的层级记忆：`4096` 个目录项保证全局可达，`16*64` 个原文块保证关键细节可读，`512` 个最近 token 保证局部连续。这个组合刚好对 LLM decoding 的 KV-cache 瓶颈很友好，因为真正昂贵的是每生成一个 token 都从 HBM 里读大量 KV。

## 为什么它能训练，而不是只做推理 patch？

很多稀疏注意力方法是在 full-attention 模型训练完之后，推理时再裁剪 KV 或选择块。问题是模型训练时没学过这种信息缺失方式，推理时突然变稀疏，容易损失能力。

NSA 的“native”主要体现在：预训练时就使用这个三分支稀疏 attention 结构。压缩 MLP、分支 KV 投影和 gate 都是可训练的，forward/backward 也有对应稀疏 kernel。虽然 top-n block selection 本身仍是离散选择，但模型从一开始就在这种稀疏访问模式下优化，因此压缩摘要会学成可检索目录，selected branch 会学会补关键细节，window branch 则稳定短程信息。

## 直觉总结

NSA 的稀疏记忆可以看成三层记忆系统：

| 记忆层 | 作用 | 类比 |
|---|---|---|
| compressed attention | 全局粗扫，找到可能相关区域 | 目录/摘要 |
| selected attention | 对相关历史块做细读 | 翻到对应页读原文 |
| sliding attention | 保留最近局部上下文 | 手边刚读过的几行 |

它真正高明的地方在于没有只从算法角度说“少算 attention”，而是把稀疏模式设计成 GPU 友好的连续块访问，并且照顾 GQA/MQA 下共享 KV cache 的现实瓶颈。换句话说，它省的是实际 decoding 中最贵的 KV cache 读取，而不是只在纸面 FLOPs 上变稀疏。

## 参考

- arXiv: [2502.11089 Native Sparse Attention](https://arxiv.org/abs/2502.11089)
- PDF: [https://arxiv.org/pdf/2502.11089](https://arxiv.org/pdf/2502.11089)
