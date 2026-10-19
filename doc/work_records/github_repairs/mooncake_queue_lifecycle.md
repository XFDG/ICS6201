# Mooncake：取消 reservation 的排空与 worker 生命周期

开发记录：2026-10-01、10-08、10-12 至 10-13。[PR #3661](https://github.com/kvcache-ai/Mooncake/pull/3661)。

## 1. 现象：没有在途请求，队列仍会被耗尽

批量 admission 失败时，已经预留的位置不能直接回退共享 tail，否则可能破坏其他生产者的进度。实现将这些位置发布为空取消条目，让单消费者按 FIFO 回收。

取消条目不增加真实 `inflight_slices`。idle worker 只依据 inflight 工作时可能继续睡眠，取消槽位得不到回收，连续拒绝最终耗尽队列。这是资源回收与唤醒状态机的问题。

## 2. 第一层修复仍不足：唤醒不等于排空

最初补丁加入 publication readiness、取消唤醒和等待 predicate。但发送额度检查位于消费之前时，quota 不足仍会跳过取消条目。消费者即使已经醒来，也不能回收槽位。

最终顺序是：发布空条目 → 唤醒 idle worker → readiness 允许进入消费路径 → 在发送 quota 检查前清理队首取消条目 → 真实请求继续接受原 quota/priority 控制。

## 3. 并发契约

| 部分 | 必须保持的约束 |
|---|---|
| Readiness | 检查当前 head 的 sequence；后面的已发布位置不能越过未发布队首 |
| Publication | release 发布与 acquire 读取配对，先确认就绪再读取 payload |
| 空条目清理 | 仅单消费者执行，复用已有 pop 回收；遇到真实或未发布 head 立即停止 |
| Quota | 空条目回收不需要发送额度，真实请求和 overflow 仍受控制 |
| 唤醒 | 发布后在对应 mutex 下协调 suspend/notify，等待 predicate 使用同一协议 |
| Accounting | 不把取消条目伪装成真实 inflight 请求 |

head 的 relaxed 读取依赖 MPSC 单消费者条件，不能把这一方案直接用于多消费者队列。

## 4. 定向验证

新增测试覆盖队首发布顺序、连续取消超过容量、零 inflight 唤醒、无 quota 排空、真实队首停止、未发布队首停止和槽位全部返还。

| 验证 | 结果 | 边界 |
|---|---|---|
| 真实队列测试 | 13 项通过 | 6 既有、7 新增 |
| 重复运行 | 100 轮、1300 次通过 | 重复成功不等于形式化并发证明 |
| 旧 quota 顺序模型 | 8 次取消后槽满 | 控制流模型，不是完整旧 worker 集成 |
| 新顺序模型 | 32 次取消完成，真实发送为 0 | 证明取消回收没有借机绕过真实发送 quota |

这一阶段没有验证真实网卡、多节点 RDMA、TSAN 或完整 transport 套件。

## 5. 后续 CI 归因：区分构建环境与队列逻辑

后续还处理了 wheel 与安装 Torch 版本不匹配的问题。缺少目标版本的扩展模块不能直接解释为队列代码回归。10 月 8 日的 selector 单测 8 项通过，但没有把较早的 wheel/collective 测试算作新提交重跑结果。

主线合并时只处理实际冲突，保留原修复与新增上游逻辑；没有借环境修复扩展修改队列算法。

## 6. metadata lifetime fixture 修复

后续原生测试在 admission 处失败，尚未进入预期 metadata lifetime 检查。原因是测试只创建合成 worker contexts，没有建立生产 guard 所要求的 running 状态。

补丁仅在 fixture setup/teardown 设置对应生命周期状态，没有放松生产 admission guard 或删除断言。原失败方法在相同条件下通过；相关 RDMA 测试为 105 pass、1 skip，队列测试 13/13。

该 fixture 修复于 10 月 13 日记录为已推送原 PR；仍不表示真实 RNIC 或完整 CI 通过。本次归档不重新查询 review/merge 状态。

## 7. 经验

并发故障需要追踪预留、发布、可见性、唤醒、额度和回收的整个状态转换。修复后测试失败也应先确认失败层级：环境不匹配、fixture 缺状态与生产算法错误是不同问题，需要不同证据。
