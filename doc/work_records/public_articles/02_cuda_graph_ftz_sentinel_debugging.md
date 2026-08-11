# 一次 CUDA Graph Hang 的位级追踪：从 RPC 超时定位到 FTZ 哨兵误判

> 说明：本文案例已经匿名化，只讨论开源组件中可公开验证的故障机制与排障方法，不对应任何特定业务、模型或生产环境。文中的代码、位模式和问题关联均可在 FlashInfer、vLLM 与 NVIDIA 的公开资料中核对。

## 结论先行

这次故障最初表现为一次普通的 RPC 超时：推理引擎迟迟收不到张量并行 worker 的采样结果，最终在消息队列等待处报错。但 RPC 并不是根因，它只是整条等待链的最后一环。

真正没有完成的是 FlashInfer MNNVL fused AllReduce + RMSNorm kernel。旧实现用浮点比较判断通信缓冲区中的 `-0.0f` Lamport sentinel：

```cpp
return val == 0.F && signbit(val);
```

目标 kernel 使用 fast-math 编译，比较指令带有 FTZ（Flush To Zero）语义。当一个合法的 packed BF16 payload，其 32-bit 搬运位模式恰好落入 FP32 负次正规数范围时，FTZ 会让 `val == 0` 成立，而最高符号位又让 `signbit(val)` 成立。合法数据于是被误判成“远端尚未写入”的 sentinel，polling loop 永远不退出。

正确修复不是关闭整个 fast-math，而是把协议判断恢复成精确位比较：

```cpp
constexpr uint32_t kNEGZERO_FP32 = 0x80000000U;
return __float_as_uint(val) == kNEGZERO_FP32;
```

这是一个活性修复：它消除错误分类和永久轮询，不改变 AllReduce 或 RMSNorm 的数学公式。公开修复见 FlashInfer [PR #3304](https://github.com/flashinfer-ai/flashinfer/pull/3304)，关联的现场记录见 FlashInfer [Issue #3053](https://github.com/flashinfer-ai/flashinfer/issues/3053) 与 vLLM [Issue #35772](https://github.com/vllm-project/vllm/issues/35772)。

## 1. 为什么 RPC 超时只是表象

分布式推理中的一次采样请求，大致会经过下面这条链路：

```text
CUDA Graph replay
  -> fused AllReduce + RMSNorm kernel
  -> sampler
  -> sampled token 的 GPU-to-CPU copy
  -> TP worker 返回结果
  -> EngineCore 收到 RPC response
```

只要前面的 GPU kernel 永远不结束，后面的 D2H copy 就无法完成；worker 不能构造 response，EngineCore 最终只能在 RPC 或消息队列上超时。因此，“最后一个可见栈帧在哪里”和“第一个没有完成的工作在哪里”是两个不同问题。

排查这类故障时，与其反复增加 RPC timeout，不如给关键异步阶段增加成对事件：

```text
kernel replay begin / queued
sampled-token copy begin / end
RPC enqueue begin / end
```

如果两个 rank 都出现 copy begin，却始终没有 copy end，而此前又确认目标 graph replay 已经提交，那么调查方向应转向 GPU producer，而不是继续围绕 CPU 队列打转。

## 2. Lamport sentinel 到底在等什么

MNNVL one-shot AllReduce 需要判断远端 rank 的 multicast 数据是否已经覆盖通信 buffer。一个常见做法是先把待写区域填成“不应与正常数据混淆”的 sentinel；消费者不断 volatile load，直到所有 lane 都不再是 sentinel，才进入后续规约与归一化。

这里使用的 FP32 sentinel 是负零：

```text
-0.0f 的精确位模式 = 0x80000000
```

轮询逻辑可以抽象为：

```cpp
while (true) {
    float4 packed = loadPackedVolatile<float4>(ptr);

    bool ready = true;
    for (int i = 0; i < 4; ++i) {
        ready &= !isNegZero(packed[i]);
    }

    if (ready) {
        break;
    }
}
```

关键点是：`float4` 在这里首先是一个 128-bit 搬运载体，并不意味着其中每个 32-bit lane 都是需要按 FP32 数值解释的业务数据。对于 BF16 输入，一个 `float4` 可以承载八个 BF16 元素，每个 32-bit lane 实际装着两个相邻 BF16。

sentinel 是一个协议位模式。既然协议问的是“这 32 bit 是否精确等于 `0x80000000`”，判断就不应引入浮点近似、舍入或次正规数处理语义。

## 3. packed BF16 如何撞上 FP32 负次正规模式

考虑小端布局下相邻的两个 BF16：

```text
低 16 bit：0x0000  // BF16 +0
高 16 bit：0x8001  // BF16 负次正规数
```

拼成一个 32-bit lane 后得到：

```text
0x80010000
```

从 BF16 payload 看，这是两个合法元素；但如果把同一组 bit 临时解释成 FP32，它又恰好属于负次正规数。注意：

```text
合法 packed payload：0x80010000
真正的 -0 sentinel：0x80000000
```

两者显然不相等。问题出在旧判断没有比较 bit，而是先做了浮点相等比较。

NVIDIA 的 [NVCC 文档](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math) 明确说明，`--use_fast_math` 隐含 `--ftz=true`。根据 [PTX ISA 对 `.ftz` 的定义](https://docs.nvidia.com/cuda/parallel-thread-execution/#comparison-and-selection-instructions-setp)，单精度次正规输入会按保留符号的零处理。于是旧 predicate 对 `0x80010000` 的解释变成：

```cpp
val == 0.F     // true：FTZ 比较把负次正规数视作 -0
signbit(val)   // true：原始最高位仍为 1
```

两个条件同时成立，合法 payload 被错误识别成 sentinel。

这里还要区分“读取”和“解释”：volatile load 把原始 32 bit 读入寄存器，FTZ 发生在浮点 predicate 对它进行解释时。它不等于 global memory 中的数据被永久改写成零；同一寄存器做整数位比较，仍能区分 `0x80010000` 与 `0x80000000`。

## 4. 一个 lane 的误判为何会卡住整个 kernel

producer 已经写入真实 payload，之后不会为了满足错误的 poller 再写一次。命中该模式的线程会不断经历：

```text
读取合法 payload
  -> FTZ predicate 误判为 sentinel
  -> 继续等待
  -> 再次读取同一 payload
  -> 再次误判
```

少数线程永久轮询后，阻塞会逐层扩散：

```text
部分 lane 卡在 Lamport spin-wait
  -> 同 warp 其他 lane 卡在 full-mask shuffle
  -> 其他 warp 卡在 block barrier
  -> 相关 CTA 卡在 cluster barrier
  -> fused kernel 不结束
  -> D2H copy 不结束
  -> worker 无法返回
  -> RPC timeout
```

FlashInfer Issue #3053 公布的 cuda-gdb 现场正是这种形态：部分线程停在 `loadPackedVolatile` 或 `isNegZero`，其他线程分别等待 `__shfl_xor_sync`、`__syncthreads()` 与 cluster barrier。它提供了从位级错误到上层超时之间缺失的一环。

## 5. CUDA Graph 是暴露条件，不是 FTZ 的来源

一个容易形成的错误结论是：“CUDA Graph 把数值变成了 `-0.0`。”实际上，FTZ 来自 kernel 的编译语义和浮点比较；Graph 本身没有把 payload 改写成 sentinel。

Graph replay 可能让问题更稳定地出现，是因为它会重复走到同一个 fused kernel，复用已捕获的地址、buffer 与状态，并让某些 shape 和 payload 组合更容易重现。类似地，关闭 fusion 后问题消失，只能说明绕开了包含 polling 的目标 kernel，不能说明 sentinel 协议已经被修复。

因此：

- eager 通过，不代表 Graph 中的 sentinel 实现正确；
- 随机 standalone 输入通过，不代表定向负次正规位模式安全；
- unfused 通过，是重要控制组，但不是根因修复；
- Graph 是触发路径的一部分，FTZ predicate 才是错误分类发生的位置。

公开 Issue 的复现环境集中在特定 GPU 与软件组合。本文只解释上游代码和 ISA 能支持的机制，不将“能够触发”泛化成“所有架构、所有输入都必现”。

## 6. 为什么 bitwise fix 是更合适的修复

PR #3304 把判断改为：

```diff
 constexpr uint16_t kNEGZERO_FP16 = 0x8000U;
+constexpr uint32_t kNEGZERO_FP32 = 0x80000000U;

 if constexpr (std::is_same_v<T, float>) {
-    return val == 0.F && signbit(val);
+    return __float_as_uint(val) == kNEGZERO_FP32;
 }
```

新代码只回答一个问题：输入的 32 bit 是否精确等于 `0x80000000`。因此：

```text
0x80000000 -> sentinel
0x80000001 -> 普通负次正规位模式
0x80010000 -> 普通 packed payload
```

直接关闭 `-use_fast_math` 不是理想的第一选择，因为它还会改变除法、平方根、FMA 以及数学库路径，影响面远大于一个 sentinel predicate。位比较既保留其余 fast-math 路径，又准确表达了通信协议。

同样需要强调：这是活性修复，不是“补丁让数值更精确”的性能或精度优化。它不自动证明补丁前后所有可完成调用都逐元素 bitwise 相同；数值回归、活性回归和性能回归仍应分别进行。

## 7. 一套可复用的排障证据链

只凭一次 hang 或一段 SASS 都不足以闭环。更可靠的方法是建立最小控制矩阵：

| 变量组合 | 目的 |
|---|---|
| Graph OFF + fused | 判断 fusion 在非 Graph 路径是否可完成 |
| Graph ON + unfused | 判断 Graph 本身是否可完成 |
| Graph ON + fused + old source | 目标失败路径 |
| Graph ON + fused + fixed source | 补丁后同路径回归 |
| 定向负次正规输入 + old/fixed | 直接覆盖 sentinel 碰撞边界 |

每一格都要确认实际执行路径，而不能只看配置文件。至少需要证明：目标 fusion 确实命中、backend 正确、两个 rank 都完成 capture 并提交 replay，没有悄悄退化成 unfused fallback。

定向输入也不应只有一个“会挂”的样本。PR #3304 的公开回归思路可以概括为三组区域：第一组把 `0x0000` 与 `0x8001` 等负次正规 BF16 位模式交替排布，让一次 4-byte poll load 形成 `0x80010000` 一类 FP32 负次正规模式；第二组放入精确 BF16/FP16 `-0.0`，验证写入侧既有的 negative-zero 处理；第三组放入正次正规模式，作为绝不能命中 sentinel 的控制组。这样不仅能证明修复后的调用“不再挂”，还能回答它为何不挂：新的 predicate 只接受精确 sentinel，而不是碰巧绕开了某个 shape。

随机张量仍然有价值，但它回答的是常规数值路径是否工作，无法高概率覆盖狭窄的协议碰撞区间。协议边界测试应直接从 bit pattern 出发，再叠加 dtype、fusion 模式和 shape 变化。若测试只保存最终 PASS/FAIL，却没有记录实际 backend、capture/replay 和输入构造，它仍不足以区分真正修复、fallback 与未命中触发数据三种情况。

证据可以分为四层：

1. **行为层**：相同输入和配置下，old/fixed 的目标路径结果不同，控制组保持可用。
2. **trace 层**：找到第一个缺失的 end 事件，并证明 RPC timeout 位于它之后。
3. **源码层**：确认运行时实际加载的 header 包含 bitwise predicate，而非旧浮点判断。
4. **二进制层**：检查目标 JIT `.so` 的来源与反汇编，确认没有继续执行旧 predicate。

通用的源码和 binary 审计可以写成：

```bash
rg -n 'kNEGZERO_FP32|__float_as_uint|val == 0\.F && signbit' \
  <FLASHINFER_INCLUDE>/flashinfer/comm/trtllm_mnnvl_allreduce.cuh

cuobjdump --dump-sass <JIT_SO> \
  | rg 'oneshotAllreduceFusionKernel|FSETP.*FTZ'
```

SASS 会随架构、CUDA 版本和编译选项变化，不能把某个固定指令数量当作跨环境唯一标准。源码、构建来源、控制矩阵和运行行为必须相互印证。

此外，hang 用例必须带外部 watchdog，并使用 fresh process 重复。否则一次异常可能留下 orphan worker、污染后续 GPU 状态，或者复用错误的编译 cache，制造假阳性和假阴性。

## 8. 最容易被忽略的 JIT cache 交付陷阱

`trtllm_mnnvl_allreduce.cuh` 不是运行时解释文件，它会被编译进 FlashInfer JIT 通信库。只修改 header，却继续加载旧 cache 中的 `.so`，GPU 执行的仍然是旧 predicate。

一次可靠交付至少要核对四个 identity：

| 身份 | 要确认什么 |
|---|---|
| package | worker 从哪个 FlashInfer 安装导入 |
| header | 该安装中的源码是否包含修复 |
| workspace | JIT cache 是否为本次构建独立创建 |
| binary | 实际加载的 `.so` 是否由修复后 header 生成 |

推荐使用隔离的 old/fixed package 与版本化 JIT workspace，而不是覆盖共享环境。多节点、多容器或多用户部署时，每个 worker 都可能拥有不同 cache；只在 launcher 节点检查版本字符串远远不够。backport 后版本号甚至可能保持不变，因此“版本看起来对”也不能替代 source 与 binary 审计。

FlashInfer [v0.6.12 release](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.12) 已公开列出该 bitwise sentinel 修复。长期维护通常应优先使用包含修复的正式版本；但升级仍需单独验证依赖、接口、JIT 重建和目标工作负载，不能把“上游已修复”等同于“当前环境已完成生产验收”。

## 9. 最终验收清单

在宣布问题解决前，建议逐项确认：

- 目标 worker 实际加载修复后的 package/header；
- 使用新的、可追踪的 JIT workspace 重新生成通信 `.so`；
- trace 证明 fused path、目标 backend 和 Graph replay 均真实发生；
- Graph OFF + fused、Graph ON + unfused 等控制组可完成；
- 定向 negative-subnormal、negative-zero、positive-subnormal 输入均有覆盖；
- old/fixed 使用相同输入、相同编译配置和隔离 cache；
- 每轮使用 fresh process，失败由外部 watchdog 有界终止；
- 数值正确性、活性和性能分别验收；
- 进程退出后没有残留 worker 或被污染的共享 cache；
- 最终业务规模回归单独完成，不用最小复现替代生产验收。

## 10. 可迁移到其他 GPU 问题的经验

这次故障最有价值的地方，不是某两行补丁本身，而是几条可复用的方法：

1. **沿等待链逆向追踪。** 从 RPC timeout 回到第一个缺失的 GPU/CPU 完成事件。
2. **把协议值和数学值分开。** sentinel、tag、epoch 等控制字段若定义为 bit pattern，就按 bit 判断。
3. **不要被载体类型误导。** `float4` 可能只是搬运格式，不能据此给 packed payload 强加 FP32 数值语义。
4. **用控制矩阵隔离变量。** Graph、compile、fusion、backend、cache 与输入位模式要分别控制。
5. **同时审计 source 与 binary。** 对 JIT kernel 来说，源码正确但 cache 错误，等价于没有修复。
6. **为罕见位模式写定向测试。** 随机输入适合覆盖常规数值，不适合证明协议边界不会碰撞。
7. **谨慎陈述边界。** 一次最小复现闭环可以证明根因和修复语义，但不能自动外推到所有硬件、所有模型或生产负载。

当一个 GPU hang 最终落在 `-0.0` 与 `0x80000000` 的差别上时，它提醒我们：在高性能内核里，数值、位模式、编译语义和并发活性往往不是四个独立问题。只有把它们放进同一条证据链，才能从“某处超时”走到真正可验证、可交付的修复。

## 参考资料

- [FlashInfer Issue #3053：Hang in oneshotAllreduceFusionKernel During Piecewise CUDA Graph Replay](https://github.com/flashinfer-ai/flashinfer/issues/3053)
- [FlashInfer PR #3304：MNNVL Allreduce uses bitwise sentinel checking](https://github.com/flashinfer-ai/flashinfer/pull/3304)
- [vLLM Issue #35772：FusedARRMS Hang during CUDA Graph capture](https://github.com/vllm-project/vllm/issues/35772)
- [FlashInfer v0.6.12 Release](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.12)
- [NVIDIA NVCC：`--use_fast_math`](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math)
- [NVIDIA PTX ISA：comparison and `.ftz`](https://docs.nvidia.com/cuda/parallel-thread-execution/#comparison-and-selection-instructions-setp)
