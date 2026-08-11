# 共享存储里的模型权重怎么盘点：从目录扫描到可运行性分级

> 本文抽象自真实工程实践，涉及的存储结构、模型、人员、规模和统计结果均已匿名化。所有路径和示例数据均为通用占位，不对应任何特定组织或生产环境。

## 结论先行

模型权重盘点不是一次 `find`，而是一条可重复的资产治理流水线。实用方案至少要解决五件事：

1. 识别真正可加载的模型目录，而不是把缓存、adapter 和训练状态都算成模型。
2. 在大型共享存储上可控扫描，支持参数化根目录、并发、超时和部分结果保留。
3. 从异构 `config.json` 中提取统一元数据，兼容 MoE 的多种字段名。
4. 同时保留配置级视图与路径级记录，因为配置相同不代表权重相同。
5. 把“可运行性分级”明确标成容量粗估，不能冒充真实加载结果。

最终建议生成两种产物：Markdown 给人浏览，CSV 或数据库给工具消费。扫描器负责陈述事实，推荐、归档和删除策略应放在下游。

## 一、先定义什么才算一个有效模型目录

共享盘里的 `config.json` 可能来自完整模型、LoRA adapter、Tokenizer 缓存、测试夹具或未完成下载。直接按配置文件计数，结果通常会严重失真。

本文把“有效 safetensors 模型目录”定义为：

- `config.json` 可以解析；
- 同目录存在非 adapter 的 `.safetensors`；
- 若存在 safetensors index，索引引用的分片全部存在；
- 权重总大小超过可配置的最小阈值；
- 路径不属于缓存、虚拟环境、测试和构建目录；
- 权重文件不是优化器、调度器或随机状态等训练产物。

这个定义有意聚焦“可直接加载的 safetensors 模型”，不会覆盖只使用旧式 `.bin` 的模型，也不能证明某个推理引擎一定能成功启动。盘点报告必须明确范围，避免把“未收录”误解为“不存在”。

## 二、参数化扫描根，用 `rg` 查找配置

在文件数量很大的共享存储上，Python 逐层递归通常慢且难以中断。`rg --files` 更适合快速枚举，并能直接筛选 `config.json`。根目录、并发和超时必须由参数传入，不能硬编码组织路径或用户名。

```python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def find_configs(root: Path, timeout_s: int) -> tuple[list[Path], str | None]:
    try:
        result = subprocess.run(
            ["rg", "--files", str(root), "-g", "config.json"],
            capture_output=True, text=True, check=False, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode(errors="replace")
        paths = [Path(x) for x in partial.splitlines() if x]
        return paths, f"timeout; retained {len(paths)} partial entries"

    if result.returncode not in (0, 1):
        return [], f"scan failed: rc={result.returncode}"
    return [Path(x) for x in result.stdout.splitlines() if x], None


parser = argparse.ArgumentParser()
parser.add_argument("roots", nargs="+", type=Path)
parser.add_argument("--timeout", type=int, default=60)
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()
```

返回码 1 通常只表示没有匹配文件，不应当作程序崩溃。单个扫描根超时时，已经产生的结果仍应保留，同时把 warning 写入报告。资产清单可以声明“本轮部分覆盖”，但不能静默丢数据。

多个独立扫描根可用线程池并发执行，但并发数不是越高越好。共享文件系统的元数据服务可能成为瓶颈，过高并发会影响训练与推理任务。生产实现应支持限速、低峰调度和按根分配预算，并记录扫描耗时、超时根数与覆盖率。

## 三、验证完整性并统一提取元数据

找到候选目录后，还要进行第二阶段检查：解析配置、枚举有效分片、累计精确字节数；若存在 `model.safetensors.index.json`，则解析 `weight_map` 并验证全部引用文件。只检查“有一个 safetensors 文件”可能把残缺下载当成模型。

adapter 也不能只靠文件名前缀识别。更稳妥的做法是同时检查 adapter 配置、PEFT 元数据和权重键；无法确定时标记为 `adapter_or_unknown`，不要直接并入完整模型。

不同架构对相同概念使用不同字段名。例如专家总数可能写成 `num_experts`、`num_local_experts` 或 `n_routed_experts`，top-k 也有多种命名，而且字段可能嵌套在子配置中。可以递归收集候选键：

```python
from typing import Any

EXPERT_KEYS = {"num_experts", "num_local_experts", "n_routed_experts"}
TOPK_KEYS = {"num_experts_per_tok", "num_selected_experts", "moe_top_k"}


def values_for(obj: Any, keys: set[str]) -> list[Any]:
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in keys:
                found.append(value)
            found.extend(values_for(value, keys))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(values_for(value, keys))
    return found
```

建议统一输出 architecture、model type、dtype、量化配置、层数、hidden size、上下文长度、精确权重字节数、分片数、experts、top-k、配置哈希、索引状态和扫描 warning。

MoE 可先按“专家数大于 1”判断，再用公开架构标识兜底，但要保留置信度。多模态或辅助子配置也可能包含专家字段，简单取最大值存在误判风险。高准确率场景应为常见架构增加适配器，把未知架构送入人工复核。

## 四、配置分组不等于权重去重

大量训练 step 和复制目录会让逐路径列表难以阅读。可以按以下字段建立“同配置组”：

```text
Dense/MoE 类型、architecture、model type
dtype、量化配置、层数、hidden size、上下文长度
experts、top-k、精确权重字节数
```

Markdown 中每组只展示一个代表记录与路径数量，CSV 中保留每条路径及其组 ID。代表记录只是导航入口，不代表该权重更正确。

必须强调：**同配置组不等于权重 bitwise 相同。** 两个训练 step 可以有完全相同的结构和大小，却包含不同参数。配置分组适合压缩展示，不能直接驱动删除。内容去重至少需要 manifest，包括配置哈希、索引哈希、分片文件名和精确大小；准备回收空间时，再对疑似重复项计算分块或全文件哈希。

## 五、容量分级只是粗估

按权重大小和单设备可用预算，可以估计最低设备数：

```text
estimated_devices = ceil(weight_bytes / usable_weight_budget_per_device)
```

这里应使用“可供权重使用的预算”，不是设备标称显存。真实运行还要容纳 KV Cache、激活、CUDA Graph、通信 buffer 和引擎 workspace；量化格式、多模态组件及并行切分也会改变结果。

因此字段应命名为 `capacity_estimate` 或 `estimated_min_devices`，不要直接写“单卡可运行”。真正的可运行性需要 smoke test：加载 tokenizer、解析自定义代码、初始化模型、执行最小请求，并记录峰值显存。静态容量估算用于筛选候选，加载测试才提供运行事实。

## 六、Markdown + CSV 双输出

Markdown 面向读者，适合展示扫描日期、规则版本、覆盖率、Dense/MoE 汇总、配置组和 warning。CSV 或数据库面向自动化，应保留完整字段：

```text
group_id, group_size, representative_id
model_class, architecture, dtype, quantization
weight_bytes, shards, layers, hidden_size
experts, top_k, context_length
path_id, manifest_id, scan_status, warning
```

如果报告可能离开受控环境，对外导出层必须把真实路径替换为稳定匿名 ID。路径通常包含用户名、项目名和训练阶段，本身就是敏感信息。普通读者只看汇总，具备授权的工具才可解析回真实位置。

## 七、从一次性脚本升级为持续治理

完整重扫会随资产增长而越来越贵，可以逐步加入：

1. **增量索引**：保存目录 mtime、inode、配置哈希和上次状态，只重扫变化目录。
2. **标准 manifest**：用配置、索引、分片清单与可选强哈希生成稳定 ID。
3. **原子输出**：写临时文件并校验后再替换正式清单，避免产生半份结果。
4. **软链接治理**：同时记录逻辑路径和规范化路径，防止重复计数。
5. **权限审计**：把无权限目录计入 coverage，不以提升权限绕过边界。
6. **来源与许可证**：记录来源、使用范围、再分发限制和自定义代码要求。
7. **加载验证**：定期为高价值模型执行最小加载与推理测试。

删除、归档和推荐应是独立决策层。扫描器不应因为“配置相同”或“长期未访问”就自动删除任何权重。

## 发布或上线前检查表

- [ ] 扫描根、并发、超时和阈值均由参数传入。
- [ ] 超时与权限错误会进入报告，部分结果不会静默丢失。
- [ ] safetensors index 引用的分片已验证完整。
- [ ] adapter、缓存、测试目录和训练状态已排除或单独标记。
- [ ] MoE 字段支持嵌套配置，并保留未知或低置信度状态。
- [ ] 配置分组与内容去重在概念和实现上分离。
- [ ] 容量结果明确标为粗估，不冒充加载实测。
- [ ] Markdown 与 CSV 从同一份结构化数据生成。
- [ ] 对外导出不包含真实路径、用户名或内部模型名。

## 结语

可靠的权重清单不只回答“盘里有什么”，还要说明它是否完整、是什么结构、可能需要多少资源，以及结论如何追溯。把参数化扫描、完整性验证、统一元数据、双层分组和权限治理串起来，散落目录才能变成可搜索、可审计、可持续维护的模型资产。
