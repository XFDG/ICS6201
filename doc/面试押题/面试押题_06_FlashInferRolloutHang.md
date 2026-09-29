# 面试押题 06｜FlashInfer CUDA Graph Rollout Hang

更新：2026-09-29。对应九坤实习中的“FlashInfer CUDA Graph hang”。核心贡献是最小复现、系统到 kernel 的根因定位、**回移上游修复**、内部接入和验证，不是原创上游 PR #3304。

## 0. 开源仓库技术介绍与应用

[FlashInfer](https://github.com/flashinfer-ai/flashinfer) 是大模型 serving 的 GPU 算子库，覆盖 attention、采样、矩阵乘及通信等计算组件；[vLLM](https://github.com/vllm-project/vllm) 是其上层的一类推理执行框架。FlashInfer 不单独承担请求排队和完整 RL 训练。

**典型应用：**推理框架调用高效 attention 或融合算子，减少访存和启动开销。本题的跨卡 AllReduce、residual 与 RMSNorm 融合发生在这个底层；因此一处同步协议错误可以向上表现为整个 rollout 超时。

## 1. 系统在干什么，问题在哪里

FlashInfer 提供推理算子与通信融合能力。这里的问题在 H200 TP=2 上的 MNNVL fused AllReduce + residual + RMSNorm：vLLM 的编译 fusion pass 将模型图中的对应模式替换为该算子，FULL CUDA Graph 捕获并重放它。RL 框架通过 vLLM 生成 rollout，因此底层 kernel 不返回，外层最终表现为 `sample_tokens` timeout 或 `EngineDeadError`。

最初业务是 8×H200，最小复现缩到 2×H200、Qwen3-30B-A3B BF16、vLLM 0.16 调试路径、FlashInfer 0.6.4。两卡不是因为模型只能这样放，而是 TP=1 不会走同一跨 rank fused collective 路径。

## 2. STAR 口述版（约 90 秒）

**S｜背景。** RL rollout 在 TP=2、FULL CUDA Graph 下卡死，外层只看到 RPC 和 EngineCore 队列超时，容易误以为 sampler 或通信框架出了问题。

**T｜任务。** 我负责把问题从完整 RL 作业缩成可复现的两卡用例，判断真正的 GPU 阻塞点，并在已有 FlashInfer 版本中交付小范围修复。

**A｜行动。** 我做 eager/Graph、fused/unfused、PDL 开关等控制组，确认目标是 MNNVL fused kernel；结合源码、定向 payload 和 SASS，定位到 Lamport `-0.0` sentinel 使用浮点比较，被 fast-math/FTZ 将合法负次正规 payload 误判为“数据未到”。随后回移上游 PR #3304 的两行位级判断，重建 JIT 通信库，并验证所有 worker 实际加载了 patched header 和新 `.so`。

**R｜结果。** 原 16 请求×16 token 用例修复前 0/3 通过，修复后 2/2 通过；固定 prompt-9 用例 fresh cache 3/3 通过；目标 kernel 中相关 `FSETP.*.FTZ` 指令由 8 处变为 0。内部 FlashInfer MR !1 已合入，但上游位级修法的作者不是我，且两卡验收不能替代任意大规模 RL 长跑验收。

## 3. 排障链路：从观察点追到不返回的 kernel

```text
sample_tokens timeout / EngineDeadError
  → sampler / token D2H 等待 GPU 工作完成
  → FULL Graph 中 fused AllReduce + RMSNorm 未结束
  → 少数线程在 Lamport readiness polling 永久等待
  → 合法 packed payload 被旧 FTZ predicate 误当成 sentinel
```

| 控制组 | 观测 | 能证明什么 |
|---|---|---|
| compile-only + fused，自然模型输入 | PASS | 编译和该路径并非每次都会失败 |
| FULL Graph + unfused | PASS | 绕开目标 fused kernel 可以恢复活性 |
| FULL Graph + fused，旧实现 | FAIL/HANG | 目标稳定暴露条件 |
| sampler 前同步 | 仍 hang | sampler 不是最早阻塞源 |
| PDL ON/OFF，旧目标用例 | 都不能消除失败 | 不是切换 PDL 就能修复的故障 |
| 原配置 + bitwise backport + fresh JIT | PASS | 修复与活性恢复形成 A/B 证据 |

没有独立证据就不说“我直接抓到了原始 RL hang 的某层某 lane 的 payload”。现有结论由控制组、源代码、上游定向回归、SASS 和本地补丁 A/B 共同支持；这是比假称完整现场 dump 更准确的贡献叙述。

## 4. 位级机制：旧实现到底错在哪

MNNVL oneshot 使用一个特殊 payload 位模式表示“此 lane 尚未由 producer 写入”：

```text
FP32 -0.0 sentinel = 0x80000000
```

poller 读入 packed payload，若仍有 sentinel 就继续等待。旧判断是：

```cpp
return val == 0.F && signbit(val);
```

不是 `val < 0`，也不是只检查符号位。严格浮点语义下，这个表达式本来可以区分 ±0；但目标 JIT 编译使用 `-use_fast_math`，其 FTZ 比较把负 FP32 次正规数按带符号零处理，扩大了 sentinel 的匹配集合。

| 原始位模式 | 真实含义 | 旧 FTZ 浮点判断 | 精确 bit 判断 |
|---|---|---:|---:|
| `0x80000000` | 精确 -0 sentinel | true | true |
| `0x80000001` | 合法负次正规数 | true，误判 | false |
| `0x80010000` | 合法 packed payload，可解释成负 FP32 次正规 | true，误判 | false |
| `0x80800000` | 负 FP32 正规数 | false | false |

**关键纠错：FTZ 影响这里的浮点 predicate，不等于 global memory 中 payload 被永久改写成 -0。** 如果原始 bits 真的已不可逆地变成 sentinel，仅改位比较无法恢复；正是因为原始 bits 仍然存在，整数 equality 才能正确区别合法 payload 与 sentinel。

为什么 BF16 也受 FP32 判断影响？此 kernel 用 `float4` 作为 128-bit 搬运载体，一个 32-bit lane 装两个 BF16。小端下，低半字 `0x0000`、高半字 `0x8001` 组成 `0x80010000`；这是一对合法 BF16，但临时按 FP32 layout 看是负次正规。polling 应检查传输协议位模式，不能把 packed lane 当 FP32 数学量近似比较。

producer 已写入合法数据，之后不会再写第二次；consumer 却永远认为“还没到”，于是无限 polling。其余线程可能继续在 warp shuffle、block barrier 或 cluster barrier 等待，最终整个 fused kernel 无法完成。这里不是“所有 barrier 都被 Lamport 去掉”，也不能把设计泛称为“完全无同步”。

## 5. 修改与发布为什么只有两行却仍有工程难度

上游 PR #3304 的核心修改为：

```cpp
constexpr uint32_t kNEGZERO_FP32 = 0x80000000U;
return __float_as_uint(val) == kNEGZERO_FP32;
```

修复用整数位比较表达 readiness 契约，不改变 AllReduce/RMSNorm 数学、buffer layout、PDL 或 CUDA Graph 算法，也无需为了这一个 predicate 全局关闭 fast-math。内部保留 FlashInfer 0.6.4 API/package，回移补丁，而不是必须升级到最新版本。

部署至少检查四件事：

1. 每个 worker 导入的 FlashInfer package/header 确实含补丁。
2. 从 patched header 重新生成 `trtllm_mnnvl_comm.so`，不加载旧 JIT cache。
3. trace/日志确认仍是 MNNVL + fused + FULL capture/replay，不是 guard 自动回退到 unfused 后“看似修好”。
4. old/fixed 使用隔离环境和独立 JIT workspace，保存构建身份、SASS 与 fresh-process 重复结果。

`flashinfer.__version__` 仍为 0.6.4 是预期行为；版本字符串不能证明补丁是否生效。只改 vLLM 接入点、checkout 仓库或替换 Python 文件，也不意味着每个 worker 的已编译通信库已刷新。

内部 MR !1 于 2026-07-23 合入 FlashInfer main，记录的 squash commit 为 `013538cf19773ebf21707d18f0a34d5449bbf640`。后续 vLLM 依赖/接入变更是单独的交付层，不能与 FlashInfer 内核源代码改动混为一条“原创修复”。

## 6. 实验结果与边界

| 验证项 | 原始 0.6.4 | backport 后 | 结果用途 |
|---|---:|---:|---|
| 16 requests × 16 tokens，FULL+fused | 0/3 PASS | 2/2 PASS | 目标模型级失败用例 A/B |
| 精确 prompt-9，fresh/release cache | 0/1 PASS | 3/3 PASS | 稳定触发用例复验 |
| 目标 kernel 的 `FSETP.*.FTZ` | 8 | 0 | 编译产物符合位级修复预期 |
| 内部 0.6.4 wheel，TP2 standalone Graph replay | 不作为旧版该行口径 | 两 rank 各 5/5，数值 PASS，exit 0 | collective、capture/replay、退出验证 |
| FULL+unfused / compile-only+fused | PASS | PASS | 控制组未被修复破坏 |

SASS 数量是当前编译器与目标 specialization 的审计结果，不是跨 CUDA 版本不变的认证标准。根因修复的证据是 source、实际 binary 身份和 old/fixed 行为共同成立。

当前押题不把 7 月两卡结果表补写成“128 卡 FULL Graph 已通过”，不把稳定性修复写成吞吐提升，也不把后续其他 FlashInfer barrier/configuration hang 归为同一问题。

## 7. 高频问题与直接回答

### Q1：为什么先查 GPU 而不是把 RPC timeout 调大？

timeout 只表示上层没有收到响应。sampler 前同步也阻塞，说明 GPU 工作尚未完成；无限 polling 不会因延长 RPC 超时恢复。先判断可完成的慢请求还是活性问题，再决定是否调整超时。

### Q2：CUDA Graph 是根因吗？

不是。根因是受 FTZ 影响的 sentinel predicate；Graph 在自然 workload 中稳定暴露了目标路径和数据/缓冲区复用条件。随机 eager 通过不能证明 predicate 正确，定向负次正规 payload 在 eager fused 也可触发旧实现。

### Q3：为什么 `-0.0 == 0.0` 为真却还用 -0 当 sentinel？

协议使用的是独特位模式，不是数值不相等性；所以读取端应做精确位比较。原实现通过“数值等于零且 signbit 为真”间接表达，忽略了 FTZ 对次正规数比较的影响。

### Q4：合法 payload 真的可以是精确 -0 吗？

实现会在写入前处理与 sentinel 冲突的精确负零模式，例如相应 BF16 -0 清洗为 +0；合法负次正规数并不是精确负零，不应被一起清洗。问题在消费者把更大一组位模式误识别为 sentinel，而非缺少“所有负数都改零”的操作。

### Q5：为什么不直接换 NaN 或额外 ready flag？

那会改通信协议、布局和同步开销，涉及更大验证范围；这次协议需要的是识别唯一位模式，局部 bitwise 修复即可。不能说 NaN/ready flag 永远不可行，也不需要为小范围修复重做整个 collective。

### Q6：关闭 fast-math 能不能修？

可能避开这个 predicate 的 FTZ 行为，但同时影响除法、sqrt、FMA 等优化，改动面更大。既然判断的是精确 bits，直接 bitcast 后整数比较更准确，也保留其他数学快路径。

### Q7：为什么两行改动还要看 SASS？

header 会被 JIT 编译，旧 `.so` 缓存可能继续运行；源码看似修好不代表 GPU 执行了新逻辑。目标 specialization 的 SASS 结合 `.so` 身份能确认危险 predicate 已改变，功能回归再确认活性和数值没有被破坏。

### Q8：如果修完后跑通，怎么排除只是自动 fallback？

核验实际 backend 为 mnnvl、fusion pass 命中、effective fusion=true，并确认两 rank 都完成 capture/replay。调试分支有过自动关闭 fusion 的 guard，必须识别这个分支行为，不能只看配置文件写了 True。

### Q9：为什么关 PDL 没有解决问题？

PDL 改变依赖 launch 的执行机制，但旧 sentinel 的判断表达式仍然错误。ON/OFF 控制组只能排除当前故障由 PDL 开关直接造成，不能由此对所有 PDL 问题做普遍结论。

### Q10：你到底改了什么，哪些不是你的成果？

我做两卡复现、定位矩阵、机制归因、上游 #3304 回移、JIT 交付与模型回归；位级修复本身来自上游。内部 MR 合入可以如实写，不能把上游 PR #3304 算进自己的 GitHub 已合入 PR 数量。

### Q11：为什么不升级最新 FlashInfer？

当时目标环境需要保留 0.6.4 的依赖和 API 组合，直接整体升级会引入额外兼容变量。最小 backport 有清晰 diff 和回归范围；同时对包含修复的新版本做过补充验证，但“验证过升级”不等于“生产采用了最新版本”。

### Q12：上线和回滚的要点是什么？

所有 worker 使用同一已记录的 package/header 与新 JIT workspace，逐 rank 验证 replay 和退出；遇到未知部署问题可关闭 fusion 绕过目标路径。若回滚到有缺陷旧内核，必须同时关掉该融合，不能恢复旧补丁却继续声称问题已解决。

## 8. 面试前必查的四个纠错点

- 旧 predicate 是 `val == 0.F && signbit(val)`，不是 `val < 0`。
- FTZ 的错误发生在比较语义，不是把原始 payload 永久改写成 sentinel。
- 目标核心文件为 `trtllm_mnnvl_allreduce.cuh`；不要拿另一份 AllReduce kernel 的 barrier 代码冒充这次根因。
- 修复是上游回移；自己的贡献是定位、适配、验证、内部交付。

## 9. 开发记录与证据索引

- 位级根因与贡献边界：`/volume/pt-train/users/zhaoye/doc/flash_infer_debug/reports/flashinfer_mnnvl_lamport_ftz_hang_explained_20260717.md`。
- 调用链、两行修复与内部合入：`/volume/pt-train/users/zhaoye/doc/flash_infer_debug/reports/r3_vllm_flashinfer_mnnvl_brief_20260724.md`。
- 部署/JIT/SASS/A/B：`/volume/pt-train/users/zhaoye/doc/flash_infer_debug/guides/vllm016_flashinfer064_pr3304_backport_guide_20260715.md`。
- 统一实验报告：`/volume/pt-train/users/zhaoye/doc/flash_infer_debug/reports/tp2_full_cudagraph_master_report_20260715.md`。
- 上游修复（归属为社区）：[FlashInfer PR #3304](https://github.com/flashinfer-ai/flashinfer/pull/3304)；[issue #3053](https://github.com/flashinfer-ai/flashinfer/issues/3053)。
- 内部合入记录：[FlashInfer MR !1](https://gitlab-cn-beijing.siflow.cn/inference/flashinfer/-/merge_requests/1)。

以上外链为本地开发报告保存的追溯入口，本篇未把“链接存在”当成额外的新实验验证。
