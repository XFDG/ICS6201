# R3 / Router Replay 工作记录

> 本目录是可公开版本，只保留工程机制、实验口径、聚合指标和排障方法。内部仓库地址、集群入口、模型与数据路径、run id、凭据和未脱敏日志均未收录。

## 阅读顺序

| 文档 | 内容 | 适合用途 |
|---|---|---|
| [R3 工程实现公开版](./R3_工程实现公开版.md) | rollout route 采集、训练侧 replay、指标契约和测试 | 简历项目展开、技术面试 |
| [R3 实验结果公开版](./R3_实验结果公开版.md) | A/B/C 设计、8 卡主结果、32 卡 smoke 和结论边界 | 数据核对、结果陈述 |
| [R3 CUDA Graph 与集群排障公开版](./R3_CUDA_Graph与集群排障公开版.md) | vLLM hang 收敛、checkpoint、恢复和 profiling | 故障定位面试 |

## 可验证结论

| 结论 | 证据 | 边界 |
|---|---|---|
| Router Replay 在 observe 口径下生效 | baseline/R2 自然 route mismatch 约 17%-19%，R3 为 0 | 证明路由一致性，不等同于 reward 提升 |
| logprob 训推漂移显著下降 | 20-step 中 f_tau_2 低约 36x-145x，KL 低约 4x-7x | 只适用于当前模型、精度与实验设置 |
| eager 口径可稳定运行 | 20 step 内 0 hang、0 engine error、0 metric error | FULL graph 的 FlashInfer 根因已在两卡闭环，大规模 RL 仍待验收 |
| 多节点链路可完成短程对照 | 4 节点 x 8 卡、TP=2、eager 的 R2/R3 5-step smoke 完成 | 短程 smoke 不能代替长跑结论 |

## 配套资产

- [R3 工程进度图](./assets/r3_engineering_progress_abc_validation_2026-06-15.png)
- [R3 周状态图](./assets/r3_weekly_status_2026-06-15.png)
- [CUDA Graph 排障流程](./assets/r3_vllm_tp2_cuda_graph_hang_flow_2026-06-25.png)
- [根因矩阵](./assets/vllm_tp2_rootcause_matrix_2026-06-25.png)
- [Checkpoint 恢复流程](./assets/r3_checkpoint_resume_flow_2026-06-24.png)
- [根因矩阵原始数据](./assets/vllm_tp2_rootcause_matrix_2026-06-25.csv)
- [FlashInfer TP2 CUDA Graph 最终根因](../flashinfer/README.md)

## 未收录内容

- 原始运行日志、checkpoint、rollout dump 和模型权重。
- 内部平台命令、镜像、节点地址、对象存储配置和访问凭据。
- 尚未完成对照的 100-step R2/R3 结果。
- 原始 SASS、Nsys/NCU profile 和未脱敏构建产物；最终聚合证据见 FlashInfer 专题。
