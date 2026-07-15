# Four-Way Grouped GEMM Benchmark Archive

> 归档时间: 2026-06-15

这组文件原本散落在 `grouped_gemm/test/`，属于 2026-05-28 的四路对比实验，不是 `grouped_gemm` 工程测试的一部分。为了保持工程仓库纯净，移动到 workspace docs archive。

## 文件

| 文件 | 说明 |
|------|------|
| `compare_four_way.py` | 四路对比脚本：CUTLASS 2.x / quack pure GEMM / deep_gemm / sonic-moe e2e |
| `four_way_results.json` | 实验原始结果 |
| `four_way_results.md` | 实验 Markdown 结果表 |

## 结论摘要

原始结果显示：

```text
CUTLASS 2.x (ubiq_gmm): 35.2% MFU
quack CUTLASS 3.x JIT: 64.9% MFU
deep_gemm: SKIP (shape not compiled)
sonic-moe e2e FWD: 31.0% MFU
```

这些结论已被后续 sonic-moe / quack 分析文档吸收。
