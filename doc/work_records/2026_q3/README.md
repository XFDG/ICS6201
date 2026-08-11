# 2026 年第三季度公开工作记录

> 更新时间：2026-08-11
> 版本：脱敏整理版

本目录汇总本季度新增的工程记录。它保留问题定义、技术路线、验证方法、结论边界和可迁移的经验；删除或泛化了人员信息、内部项目/模型代号、仓库与分支、主机和网络信息、存储路径、镜像、运行标识、原始日志、凭据及未公开的绝对业务指标。

## 阅读入口

| 主题 | 公开记录 | 重点 |
|---|---|---|
| 推理优化与 PDL | [inference_optimization.md](./inference_optimization.md) | Attention、MoE、Router GEMM、异步调度与 Programmatic Dependent Launch 的受控验证 |
| MoE 架构与评测 | [moe.md](./moe.md) | feature gate、语义 oracle、正交实验、性能归因与停止条件 |
| KernelBench 与算子设计 | [kernelbench.md](./kernelbench.md) | 评测平台、生成候选评估、修复闭环、算子接入前置检查 |
| 平台评测与系统验证 | [platform_and_systems.md](./platform_and_systems.md) | 评测配置、故障定位、CE 数值验证与 Router/OE 验收 |
| 覆盖范围与脱敏规则 | [coverage.md](./coverage.md) | 本次汇总的材料范围、保留/删除规则和证据等级 |

## 使用边界

- 文中性能描述以相对趋势、区间和适用条件为主；不应外推到未测模型、硬件、并行配置或负载。
- “数值一致”只代表所述输入、精度和容差下通过的测试，不能替代完整训练或线上验收。
- 本目录适合展示工程方法与问题拆解；如需复现，应以公开上游项目、公开数据和本地环境重新建立基线。
