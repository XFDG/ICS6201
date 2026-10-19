# CUTLASS：大 workspace 的 descriptor 溢出定位

开发记录：2026-10-12 至 10-13。[PR #3614](https://github.com/NVIDIA/cutlass/pull/3614)。

## 1. 现象与修正后的根因

FMHA backward 在大 workspace 场景失败，最初关注 Python 侧字节数算术。但 host `int()` 转换不足以修复问题：真正失败的是动态 DLPack descriptor 的 Int32 extent，在 byte workspace 达到 `2**31` 时溢出。

因此需要区分三步：Python 计算大小、CUDA storage 分配、descriptor 提取传参。大尺寸失败本身不能证明显存不足或 kernel 索引错误。

## 2. 缩小复现范围

在实际 DSL 环境中单独创建 workspace descriptor，并调用 `.__c_pointers__()`。动态描述在边界及以上失败，静态描述可工作，进而将原因缩到 descriptor 表示而非整个 Attention 数学计算。

测试三个字节长度：`2147483647`、`2147483648`、`5697536000`。前两项贴住有符号 32 位边界，第三项覆盖原大 workspace 级别。这里的数字是公开库边界测试参数，不是私有模型配置。

## 3. 补丁选择

修改位于 `examples/python/CuTeDSL/cute/blackwell/kernel/attention/fmha/fmha_bwd.py`，并新增对应 workspace 回归文件。新的 factory 使用静态 byte workspace descriptor，同时保持底层 CUDA uint8 storage 生命周期。

kernel 只需要 byte pointer，原 workspace offset 逻辑保持不变。保留 host 整数算术处理，但不再把它单独描述为完整修复。descriptor 与 storage 生命周期一起检查，避免“参数可转换”却留下悬空存储。

## 4. 验证分三层

| 层级 | 实际结果 | 不能据此声称 |
|---|---|---|
| Descriptor 边界回归 | B200 / DSL 4.8.0 下 3/3 通过 | 全部 Attention 形状正确 |
| 小形状完整 backward | 数值 reference 通过 | 大形状同样经过 reference |
| 大形状 workspace | 约 5.70 GB launch smoke 通过 | 大形状数值已验证；该次跳过 reference |

10 月 13 日补丁发布沿用前一日完成的 B200 证据，没有声称当天重新跑了 GPU。静态检查使用定向 fatal-error 项，新测试格式检查通过；没有把 kernel 全文件格式检查写成通过，也没有为了格式化重排无关实现。

## 5. Review 与经验

使用普通新提交更新原 PR，并更正根因与测试范围，请求 reviewer 复审。旧 diff 的批准不覆盖新 factory 和测试，本次文档不宣称当前已合入。

该案例的关键是把数据表示边界从计算边界中分离。Python 大整数、GPU storage 和外部 descriptor 的宽度规则不同，边界测试应落在发生窄化的实际层级。
