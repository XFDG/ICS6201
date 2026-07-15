# OE Async 工作记录（公开版）

> 时间：2026-06-29 至 2026-07-13
> 核心结论：完成 OE 在异步调度下的 GPU 数据链路与 token-history 正确性闭环；TP1 fused-hash 获得稳定收益，TP2/TP4 通过 batch-invariant 安全路径完成多阶段正确性和吞吐验证。
> 公开边界：不包含内部仓库、分支地址、模型路径、节点信息、镜像、运行 ID 和原始日志。

## 1. 问题与实现

异步 decode 中，新采样 token 仍位于 GPU，若先回传 CPU 再构造 OE 输入，会在热路径引入额外同步；同时，prefill、decode、mixed batch、请求驱逐/恢复、slot 复用和 reorder 会共同影响跨 step 的 token history，单次请求通过不能证明状态管理正确。

本阶段完成：

- 在 GPU 侧使用 Triton fused-hash，由 token history 直接生成 `oe_input_ids`，减少 GPU→CPU 等待和 CPU→GPU 回传。
- 修复首次 decode、mixed prefill/decode、请求恢复、slot 复用和 reorder 下的 recent-token history 更新。
- 为 TP>1 增加 batch-invariant 正确性门槛；TP1 默认使用 fused-hash，TP>1 当前使用 async-unfused + batch-invariant 安全路径。
- 将验证拆成 pure decode、prefill 和 mixed 三个阶段，并同时保留同步、async-unfused 和 fused-hash 对照。

## 2. 验证结果

| 范围 | 结果 | 判断 |
|---|---:|---|
| TP1 fused-hash vs sync，严格 28-case 矩阵 | 吞吐提升 5.0% | GPU 侧构造能够减少同步热路径开销 |
| TP1 fused-hash vs async-unfused，严格 28-case 矩阵 | 吞吐提升约 1.9% | 融合本身仍有增量收益 |
| TP2，28 shapes × 3 rounds | 0 mismatch；decode 吞吐约 +4.6% | batch-invariant 安全路径正确且有性能收益 |
| TP4，28 shapes × 3 rounds | 0 mismatch；decode 吞吐约 +3.0% | 多卡正确性通过，收益略低于 TP2 |

28 个 shape 的 decode 矩阵覆盖 7 个 batch size、4 个输入长度，并各重复 3 轮；prefill 和 mixed 另做全量正确性覆盖，不计入上述 decode 吞吐主数字。

## 3. 当前边界

- TP2/TP4 的结果不能写成“fused-hash 已在多卡默认交付”；当前多卡路径是 async-unfused + batch-invariant。
- mixed 短插入窗口在 TP2/TP4 下观察到约 6%/10% 回退，因此 mixed 只作为正确性覆盖，不宣传性能收益。
- batch-invariant 是正确性门槛，本身存在约 5% 吞吐代价，不能写成加速开关。
- 性能数字只适用于本次模型、输入矩阵和测量口径，不能直接外推到所有并行度或负载。
- 正确性需要覆盖跨 request 状态和 mixed batch，不能只用单 step 输出一致判断。

## 4. 月报一句话

完成 OE 算子 async scheduling 支持和 token history 正确性修复，TP1 fused-hash 在严格正确性矩阵下较同步路径提升 5.0%，TP2/TP4 的 28 组 decode 用例各重复 3 轮均 0 mismatch，吞吐分别提升约 4.6%/3.0%。

## 5. 聚合数据

- [OE async 正确性与吞吐汇总](./assets/oe_async_summary.csv)

公开仓库只保留聚合数字；原始日志、内部路径和含运行标识的旧图表未收录。
