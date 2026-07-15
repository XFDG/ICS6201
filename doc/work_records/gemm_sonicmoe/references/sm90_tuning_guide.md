# SM90 Hopper Grouped GEMM 调优操作说明

> 2026-05-25

---

## 一、背景

CUTLASS 3.x Hopper Grouped GEMM (参考 `57_hopper_grouped_gemm.cu`) 是目前 CUTLASS 对 SM90 分组 GEMM 的最佳实现。使用 TMA (Tensor Memory Access) 硬件加速数据搬运，GMMA (wgmma) 指令加速矩阵乘法，单 kernel 处理所有 expert groups。

### 可调参数

| 参数 | 含义 | 候选值 | 适用场景 |
|------|------|--------|----------|
| **TileShape<M,N,K>** | 每 threadblock 处理的 tile 大小 | M: 64/128/256, N: 64/128/256, K: 32/64 | 大 M → 大 M tile; 小 M → 小 M tile |
| **ClusterShape<M,N>** | 多少个 SM 协作一个 tile | M: 1/2, N: 1/2 | 大 tile → 开 cluster; 小 tile → C<1,1> |
| **Schedule** | 线程调度策略 | Cooperative / Pingpong | 大 M → Cooperative; 小 M → Pingpong |
| **NumStages** | 流水线深度 | auto (StageCountAutoCarveout) | auto 通常最优 |

### 已知的经验规律 (来自 DeepGEMM 等):

- **小 M (<128)**: Pingpong schedule + TileM=64, Cluster<1,1>
- **中 M (128-1024)**: Cooperative + TileM=128, Cluster<1,2>
- **大 M (>1024)**: Cooperative + TileM=256, Cluster<2,1>
- **N 接近 6144**: TileN=128 对齐度好; N=4096: TileN=128 或 64
- **K=4096**: TileK=32; K=6144: TileK=32 或 64

### MoE per-GPU 场景映射:

| EP | E_gpu | avg M (T=4096) | 推荐配置 |
|----|-------|-----------------|----------|
| 8 | 8 | ~512 | Cooperative <128,128,32> C<1,2> |
| 16 | 4 | ~1024 | Cooperative <256,128,64> C<2,1> |
| 32 | 2 | ~2048 | Cooperative <256,128,64> C<2,2> |

---

## 二、调优参考

> 以下 tile/cluster 配置来自早期 grouped GEMM 调研。旧内部计划未纳入公开仓库；这里仅保留通用 SM90 调参经验，作为阅读 quack/CUTLASS kernel 配置时的参考。

### 自定义配置模板

```cpp
template <int TM, int TN, int TK, int CM, int CN>
struct CustomCoopConfig {
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using TileShape = Shape<Int<TM>, Int<TN>, Int<TK>>;
  using ClusterShape = Shape<Int<CM>, Int<CN>, Int<1>>;
};

template <int TM, int TN, int TK, int CM, int CN>
struct CustomPPConfig {
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = Shape<Int<TM>, Int<TN>, Int<TK>>;
  using ClusterShape = Shape<Int<CM>, Int<CN>, Int<1>>;
};
```
