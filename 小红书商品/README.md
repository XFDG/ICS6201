# AI INFRA 面经+知识总结

## 商品图片与宣传文案

- [闲鱼商品主图](上架素材/闲鱼/商品图/01_闲鱼主图_AI_INFRA.png)
- [字节录用意向书水印图](上架素材/闲鱼/商品图/02_字节录用意向书_闲鱼草原小可汗水印.png)
- [闲鱼商品宣传文案](上架素材/闲鱼/闲鱼商品文案.md)
- [小红书商品宣传文案](上架素材/小红书/商品宣传文案.md)
- [全部上架素材与说明](上架素材/闲鱼/README.md)

整理日期：2026-10-31

本次交付共 39 个 PDF 文件；按文件 SHA-256 去重后为 **39 份 PDF、481 页**。统计不包含阅读与来源说明，不把同一文件的多份副本重复计数。

## 包含什么

| 分类 | PDF 份数（按 SHA-256 去重） | 页数 |
| --- | ---: | ---: |
| 技术资料 | 29 | 435 |
| 公司面试复盘 | 10 | 46 |

面试目录按公司平铺，每家公司一份 PDF；同一公司的相关岗位与已确认轮次集中在公司合册中。技术资料按主题分类。

## 先读什么

1. 先读 AI Infra、推理、训练、RL 与算子优化八股文，建立知识框架。
2. 按目标公司阅读面试合册，结合其中的岗位与轮次复习问题。
3. 区分记录中确认的问题和整理补充的复习提纲，用自己的项目准备回答与证据。
4. 对需要深入的主题，再阅读 vLLM、DeepInfer、MoE、GPU 架构与论文解读。

## 阅读口径

- 公司合册可能包含多个岗位或轮次，文件份数不代表独立面试场次；不同技术资料也可能覆盖相近知识点。文件哈希去重无法识别主题重合。
- 面试资料是去噪、脱敏后的复盘与学习讲义；实际记录中的问题和整理者补充的复习内容应按各册标注区分。内容不构成公司的官方题库或标准答案。
- 部分 PDF 保留了原有的小红书账号水印。
- 本包仅交付整理后的 PDF 与阅读/来源说明，不含面试原始转写、录音、个人 Offer 图片或商品营销文档。
- 技术资料反映各文档标注版本与整理时间；框架行为、硬件参数和性能判断应结合实际版本及测量条件理解。

## 阅读说明文件

- 本说明：阅读路线、份数与完整 PDF 目录。
- [技术PDF/01_AI Infra基础与算子/00_说明与来源.md](<技术PDF/01_AI Infra基础与算子/00_说明与来源.md>)：对应资料的使用方式与来源/基础校准。

## PDF 文件目录

| 文件 | 页数 | 用途 |
| --- | ---: | --- |
| [技术PDF/01_AI Infra基础与算子/01_AIInfra_系统基础与面试问答.pdf](<技术PDF/01_AI Infra基础与算子/01_AIInfra_系统基础与面试问答.pdf>) | 8 | 技术学习 |
| [技术PDF/01_AI Infra基础与算子/02_CUDA_算子优化与性能分析问答.pdf](<技术PDF/01_AI Infra基础与算子/02_CUDA_算子优化与性能分析问答.pdf>) | 8 | 技术学习 |
| [技术PDF/02_GPU架构与迁移/01_Hopper_Blackwell架构迁移与性能研究.pdf](<技术PDF/02_GPU架构与迁移/01_Hopper_Blackwell架构迁移与性能研究.pdf>) | 21 | 技术学习 |
| [技术PDF/02_GPU架构与迁移/02_Blackwell_产业落地与公开性能基线.pdf](<技术PDF/02_GPU架构与迁移/02_Blackwell_产业落地与公开性能基线.pdf>) | 9 | 技术学习 |
| [技术PDF/03_LLM推理/01_LLMServing_推理系统与指标快速问答.pdf](<技术PDF/03_LLM推理/01_LLMServing_推理系统与指标快速问答.pdf>) | 8 | 技术学习 |
| [技术PDF/03_LLM推理/02_vLLM_并行量化与面试详解.pdf](<技术PDF/03_LLM推理/02_vLLM_并行量化与面试详解.pdf>) | 36 | 技术学习 |
| [技术PDF/03_LLM推理/03_vLLM_执行调度与框架机制.pdf](<技术PDF/03_LLM推理/03_vLLM_执行调度与框架机制.pdf>) | 33 | 技术学习 |
| [技术PDF/03_LLM推理/04_DeepInfer_推理执行引擎与源码解析.pdf](<技术PDF/03_LLM推理/04_DeepInfer_推理执行引擎与源码解析.pdf>) | 21 | 技术学习 |
| [技术PDF/04_MoE系统/01_MoK_计算通信融合与源码核验.pdf](<技术PDF/04_MoE系统/01_MoK_计算通信融合与源码核验.pdf>) | 18 | 技术学习 |
| [技术PDF/04_MoE系统/02_MegatronCore_MoE分布式训练系统解析.pdf](<技术PDF/04_MoE系统/02_MegatronCore_MoE分布式训练系统解析.pdf>) | 20 | 技术学习 |
| [技术PDF/04_MoE系统/03_MoonEP_动态冗余专家与负载均衡.pdf](<技术PDF/04_MoE系统/03_MoonEP_动态冗余专家与负载均衡.pdf>) | 8 | 技术学习 |
| [技术PDF/05_训练与RL/01_Training_大模型训练原理与系统面试详解.pdf](<技术PDF/05_训练与RL/01_Training_大模型训练原理与系统面试详解.pdf>) | 59 | 技术学习 |
| [技术PDF/05_训练与RL/02_RL_强化学习算法与Infra面试详解.pdf](<技术PDF/05_训练与RL/02_RL_强化学习算法与Infra面试详解.pdf>) | 55 | 技术学习 |
| [技术PDF/06_论文与前沿技术/01_HiLS_分层稀疏注意力与长上下文.pdf](<技术PDF/06_论文与前沿技术/01_HiLS_分层稀疏注意力与长上下文.pdf>) | 9 | 技术学习 |
| [技术PDF/06_论文与前沿技术/02_HiSparse_分层KVCache与显存管理.pdf](<技术PDF/06_论文与前沿技术/02_HiSparse_分层KVCache与显存管理.pdf>) | 8 | 技术学习 |
| [技术PDF/06_论文与前沿技术/03_KDA_线性注意力机制与算子设计.pdf](<技术PDF/06_论文与前沿技术/03_KDA_线性注意力机制与算子设计.pdf>) | 9 | 技术学习 |
| [技术PDF/06_论文与前沿技术/04_KimiK3_大模型架构与训练系统协同.pdf](<技术PDF/06_论文与前沿技术/04_KimiK3_大模型架构与训练系统协同.pdf>) | 8 | 技术学习 |
| [技术PDF/06_论文与前沿技术/05_TileSight_从GPU核心到集群的性能建模.pdf](<技术PDF/06_论文与前沿技术/05_TileSight_从GPU核心到集群的性能建模.pdf>) | 17 | 技术学习 |
| [技术PDF/07_AI Agent与Harness/01_Harness_Agent执行系统与DeepSeek技术图解.pdf](<技术PDF/07_AI Agent与Harness/01_Harness_Agent执行系统与DeepSeek技术图解.pdf>) | 16 | 技术学习 |
| [技术PDF/07_AI Agent与Harness/02_Harness_轻量支架与模型协同演进.pdf](<技术PDF/07_AI Agent与Harness/02_Harness_轻量支架与模型协同演进.pdf>) | 6 | 技术学习 |
| [技术PDF/07_AI Agent与Harness/03_Sandbox_Agent运行环境与权限设计.pdf](<技术PDF/07_AI Agent与Harness/03_Sandbox_Agent运行环境与权限设计.pdf>) | 6 | 技术学习 |
| [技术PDF/07_AI Agent与Harness/04_Codex_从代码助手到长期任务Agent.pdf](<技术PDF/07_AI Agent与Harness/04_Codex_从代码助手到长期任务Agent.pdf>) | 6 | 技术学习 |
| [技术PDF/07_AI Agent与Harness/05_Context_Memory与大众化Agent工作流.pdf](<技术PDF/07_AI Agent与Harness/05_Context_Memory与大众化Agent工作流.pdf>) | 6 | 技术学习 |
| [技术PDF/07_AI Agent与Harness/06_Agent_多智能体编排与企业软件形态.pdf](<技术PDF/07_AI Agent与Harness/06_Agent_多智能体编排与企业软件形态.pdf>) | 6 | 技术学习 |
| [技术PDF/07_AI Agent与Harness/07_MultiAgent_并行推理与实时交互.pdf](<技术PDF/07_AI Agent与Harness/07_MultiAgent_并行推理与实时交互.pdf>) | 6 | 技术学习 |
| [技术PDF/07_AI Agent与Harness/08_Agent_最小产品界面与智能成本.pdf](<技术PDF/07_AI Agent与Harness/08_Agent_最小产品界面与智能成本.pdf>) | 6 | 技术学习 |
| [技术PDF/07_AI Agent与Harness/09_Codex_Rust架构与开源工程实践.pdf](<技术PDF/07_AI Agent与Harness/09_Codex_Rust架构与开源工程实践.pdf>) | 6 | 技术学习 |
| [技术PDF/07_AI Agent与Harness/10_Codex_云端个人Agent与Infra演进.pdf](<技术PDF/07_AI Agent与Harness/10_Codex_云端个人Agent与Infra演进.pdf>) | 8 | 技术学习 |
| [技术PDF/07_AI Agent与Harness/11_JevMem_Agent长期记忆与预算控制.pdf](<技术PDF/07_AI Agent与Harness/11_JevMem_Agent长期记忆与预算控制.pdf>) | 8 | 技术学习 |
| [面试/MiniMax.pdf](<面试/MiniMax.pdf>) | 4 | 公司面试复盘 |
| [面试/OPPO.pdf](<面试/OPPO.pdf>) | 4 | 公司面试复盘 |
| [面试/字节跳动.pdf](<面试/字节跳动.pdf>) | 10 | 公司面试复盘 |
| [面试/小米.pdf](<面试/小米.pdf>) | 4 | 公司面试复盘 |
| [面试/小马智行.pdf](<面试/小马智行.pdf>) | 6 | 公司面试复盘 |
| [面试/智元机器人.pdf](<面试/智元机器人.pdf>) | 2 | 公司面试复盘 |
| [面试/曦望.pdf](<面试/曦望.pdf>) | 6 | 公司面试复盘 |
| [面试/理想.pdf](<面试/理想.pdf>) | 2 | 公司面试复盘 |
| [面试/百度.pdf](<面试/百度.pdf>) | 4 | 公司面试复盘 |
| [面试/阿里.pdf](<面试/阿里.pdf>) | 4 | 公司面试复盘 |
