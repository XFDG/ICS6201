# 周报时间线

> 周报保留当时的阶段判断，因此后续报告可能修正早期口径。最终对外数字以各专题“公开版”文档为准。

| 周期 | 主线 | 最终归档 |
|---|---|---|
| [5 月 18 日至 22 日](<./5月18日-5月22日 周报.md>) | DeepGEMM 离线预编译与 wheel | [DeepGEMM](../deep_gemm/README.md) |
| [5 月 23 日至 25 日](<./5月23日-5月25日 周报.md>) | Grouped GEMM 初始 profile | [H200 GEMM](../gemm_sonicmoe/H200_GEMM性能分析公开版.md) |
| [5 月 25 日至 29 日](<./5月25日-5月29日 周报.md>) | 四路 benchmark 与 SonicMoE | [四路归档](../gemm_sonicmoe/archive/four_way_benchmark_20260528/README.md) |
| [6 月 1 日至 5 日](<./6月1日-6月5日 周报.md>) | Quack 线上回放与 config tuning | [Quack 调优](../gemm_sonicmoe/Quack调优开发记录公开版.md) |
| [6 月 11 日至 18 日](./6月11日-6月18日周报.md) | R3 单机链路、早期 A/B/C | [R3 实现](../r3/R3_工程实现公开版.md) |
| [6 月 22 日至 26 日](<./6月22日-6月26日 周报.md>) | R3 observe、32 卡、CUDA Graph 排障 | [R3 结果](../r3/R3_实验结果公开版.md) |

## 阅读注意

- 6 月 11 日至 18 日周报中的 route 自一致性不是自然 route divergence 主证据。
- R3 主结果使用 6 月 23 日的 20-step observe 口径。
- GEMM 的早期四路平均值与后续扩展汇总来自不同 shape 集，均在专题文档中注明。
- 周报已移除内部链接和凭据上下文，保留技术决策与聚合数据。

