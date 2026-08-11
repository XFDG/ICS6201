# 异步调度遇上历史依赖：如何消除 CPU-GPU 同步并守住生成一致性

> 本文由真实工程实践抽象而来。模型、硬件、软件环境、实验规模和结果均已匿名化；模块名称、流程与伪代码只用于讲解通用方法，不对应任何具体生产实现。

## 结论先行

异步推理最怕一种隐藏依赖：GPU 已生成上一步 token，下一步却要等它回到 CPU，计算历史特征，再传回 GPU。即使 CPU 计算很短，这条往返依赖也会在逐 token decode 中形成稳定气泡。

本文用“历史增强嵌入（History-Enhanced Embedding，HEE）”作为通用示例。HEE 根据最近若干 token 生成辅助索引或特征，并参与下一步 embedding。可靠的异步化需要同时做到：历史常驻 GPU；以 request identity 管理状态；分别处理 pure decode 与 mixed；提供安全 fallback；用确定性门禁控制多卡误差。还要准确描述融合边界：融合“历史到辅助索引”，不等于整个 HEE forward 已经安全融合。

## 1. 同步气泡从哪里来

一类 HEE 可以抽象为：

```text
recent_tokens = [token(t-k+1), ..., token(t)]
history_ids   = HistoryTransform(recent_tokens)
history_embed = HistoryEmbedding(history_ids)
hidden(t+1)   = BaseEmbedding(token(t)) + Project(history_embed)
```

同步路径存在两次跨设备依赖：

```text
GPU model forward -> sample token(t)
          │
          ▼
GPU -> CPU：等待 token 回传
          │
CPU：更新历史并计算 history_ids
          │
          ▼
CPU -> GPU：拷贝 history_ids
          │
GPU：HEE forward -> step t+1
```

异步方案应把闭环留在 GPU：

```text
GPU sample token
  -> 写入 current-token buffer
  -> 读取 per-request recent history
  -> fused history transform
  -> history_ids
  -> HEE forward

CPU 只准备调度元数据，不等待真实 token 参与热路径计算
```

因此，这类优化首先是在消除依赖边，其次才是加速计算。

## 2. GPU 常驻历史必须绑定请求身份

最小状态包括每个活跃请求的短历史、请求到当前 GPU 行的映射、本步调度位置、采样结果和历史输出 buffer。

```python
# Pure decode 伪代码
for req in schedule.decode_requests:
    if req.scheduled_token_count == 0:
        continue
    row = request_to_row[req.id]
    token = current_token_buffer[row]
    previous = recent_token_buffer[row]
    history_output_buffer[row] = history_transform(previous, token)
    recent_token_buffer[row] = shift_and_append(previous, token)
```

融合实现必须保持三项不变量：读取当前请求的历史；未调度请求不误更新；sample 写入、历史读取和 HEE forward 之间有明确流顺序。

不能只按 slot 保存历史。Slot 是临时位置，请求身份才是稳定语义；请求可能离开 batch、被抢占或恢复到新位置。安全生命周期应是：

```text
创建请求 -> 从 prompt 尾部初始化历史 -> 分配临时 GPU row
         -> decode 按 request id 更新
         -> 暂离 batch 时保留状态
         -> 恢复时映射新 row
         -> 完成后清理
```

Prefill 后第一个 decode step 尤其关键：GPU 历史必须来自 prompt suffix，而不是零值、占位 token 或旧 slot 内容。

## 3. Pure decode 与 mixed 要分开设计

Pure decode 中，本步真实 token 全由 GPU 采样产生，可以完全走 GPU 闭环：

```text
sampled token -> recent history -> fused transform
              -> history_ids -> HEE forward
```

这是异步 HEE 最自然的收益场景。融合 kernel 还能把多个小操作合成一次 launch，减少固定调度成本。

Mixed prefill+decode 则同时包含两类数据：prompt token 已在 CPU 完整可见，decode token 的真实值刚在 GPU 产生。稳定方案通常是 hybrid：CPU 先生成已知部分，GPU 再覆盖 decode 位置。

```python
# Mixed 伪代码
history_ids = cpu_fill_for_known_tokens(batch)
copy_to_gpu(history_ids)

gpu_overwrite_history_ids(
    rows=scheduled_decode_rows,
    sampled_tokens=current_token_buffer,
    recent_history=recent_token_buffer,
    output=history_ids,
)
run_single_hee_forward(history_ids)
```

Mixed 的首要目标是正确合并两种来源，而不是强求全 GPU。若让 prefill 也绕远路，可能增加 kernel、拷贝或调度步骤，反而退化。

## 4. Fused、unfused 与安全 fallback

CPU baseline 适合作为正确性参考；GPU unfused 保留 GPU 闭环，可作为通用 fallback；GPU fused 则是支持 shape 下的性能路径。融合实现可能受布局、并行规模或编译模式限制，因此必须显式选择：

```python
# 伪代码
mode = GPU_FUSED if fused.supports(case) else GPU_UNFUSED
if not gpu_path.supports(case):
    mode = CPU_SAFE_PATH
log_selected_mode_once(mode)
```

测试要验证 fallback 确实发生，并让当前路径可观测。

## 5. 为什么微小误差会变成轨迹分叉

动态 batch 或多卡推理中，sync 与 async 的 batch 组成、调度顺序可能不同，Attention、矩阵乘和跨卡归约因而选择不同分块或累加顺序。浮点低位出现差异并不罕见，HEE 又会把差异反馈到后续输入：

```text
batch / scheduling / reduction path 不同
  -> logits 出现微小差异
  -> top-1 与 top-2 接近时选出不同 token
  -> recent history 不同
  -> history transform 与 HEE 输入不同
  -> 后续 hidden states、logits 继续分叉
```

最终 token 不一致不能直接证明历史 transform 算错。应依次对齐 input token、position、request mapping、history IDs 和 HEE 出口，再逐层寻找首次数值差异。若差异最先出现在 Attention 或归约之后，继续修改历史算法通常没有意义。

若验收要求同一请求在不同 batch 轨迹下保持一致，就需要 batch-invariant 或等价的确定性路径。最重要的原则是 baseline 和 optimized 同时使用相同前提：

| Baseline | Optimized | 是否可比 |
|---|---|---|
| 默认路径 | 确定性路径 | 否，数值前提不同 |
| 确定性路径 | 默认路径 | 否，数值前提不同 |
| 默认路径 | 默认路径 | 可观察非确定性风险 |
| 确定性路径 | 确定性路径 | 可做严格一致性对比 |

确定性通常有性能代价，报告应同时说明风险、门禁结果、相对成本和未覆盖范围。小矩阵通过不代表所有 phase、shape 和并行规模已全面正确。

## 6. 分层验证与融合边界

验证应由局部到整体推进：固定输入对齐 CPU/GPU 与 fused/unfused transform；用重排、抢占、slot 复用测试状态机；比较 HEE 出口张量；分别覆盖 Prefill、Decode、Mixed；最后在相同确定性前提下比较多卡 logprob 和生成轨迹。只有数值门禁通过后才运行性能矩阵，否则先定位首次差异。

HEE 还可以拆成两段：

```text
阶段 A：token/history -> history_ids
阶段 B：history_ids + base embedding -> hidden states
```

先融合 A 通常更稳，因为模型 forward 仍接收相同的 `history_ids`，三个 phase 可以共用一个入口。直接融合 A+B 会跨越调度器和模型图边界；即使算子单测和 eager 正确，在编译或 CUDA Graph 中仍可能因分支特化、动态 buffer 生命周期和图重放契约而失败。

验收必须逐级进行：算子单测、整模型 eager、编译无图、CUDA Graph replay。任一级失败，都不能用性能数字证明成功。

上线前应确认：状态按 request identity 管理；pure decode 和 mixed 数据流明确；执行路径可观测、可回退；两组使用相同确定性前提；性能只在数值门禁后统计；文档没有夸大融合范围。

## 结语

历史依赖算子与异步调度的冲突，本质上是“上一 token 在哪里、下一步依赖在哪里”。让短历史常驻 GPU、按请求身份维护状态，并融合历史生成，可以拿掉 decode 热路径的 CPU-GPU 气泡。但可靠方案还必须处理生命周期、mixed 合并、多卡数值路径和编译图契约，做到可加速、可验证、可回退。
