# DeepGEMM Config Cache 方案

## 1. 背景与目标

### 当前行为

```
调用 fp8_gemm_nt(m, n, k, ...) 
  → 构造 GemmDesc
  → get_best_config<ArchSpec>(desc)   ← 每次重新算，无缓存
      → 遍历所有 layout candidates
      → 对每个 candidate 调用 get_layout_info() 计算 wave/cycle 估算
      → 选出最优 layout
      → 推导 storage/pipeline/launch config
  → compiler->build(config)           ← 已有 cubin 磁盘缓存
```

### 问题
`get_best_config` 每次都重新遍历 layout candidates 计算，相同 shape 多次调用时完全重复劳动。

### 目标
为 `get_best_config` 加一层持久化缓存：

```
get_best_config(desc)
  → 先查缓存（内存 map）
    → 命中：直接返回 GemmConfig ✓
    → 未命中：走原有逻辑算出 config → 写入缓存 → 返回
```

---

## 2. 数据结构分析

### 缓存 Key：GemmDesc
`GemmDesc` 中决定 config 结果的字段（与 shape 和运行环境相关）：
```
gemm_type, kernel_type, m, n, k, num_groups
a_dtype, b_dtype, cd_dtype
major_a, major_b
with_accumulation
num_sms, tc_util
compiled_dims
expected_m, expected_n, expected_k, expected_num_groups
```
已有 `operator<<` 可直接序列化为唯一字符串，用作 key。

### 缓存 Value：GemmConfig
```
Layout:         swap_ab, block_m, block_n, block_k, cluster_m, cluster_n   (6 int)
StorageConfig:  load_block_m, load_block_n, store_block_m, store_block_n,
                swizzle_a_mode, swizzle_b_mode, swizzle_cd_mode             (7 int)
PipelineConfig: smem_size, num_stages                                        (2 int)
LaunchConfig:   num_sms, num_sms_per_cluster, num_threads,
                num_tma_threads, num_math_threads,
                num_non_epilogue_threads, num_epilogue_threads               (7 int)
```
总共 22 个 int，序列化为空格分隔的字符串即可，无需外部依赖。

---

## 3. 实现方案

### 3.1 新增文件：`csrc/jit/config_cache.hpp`

```cpp
class ConfigCache {
    // 内存缓存
    std::unordered_map<std::string, GemmConfig> cache;
    // 磁盘缓存路径
    std::filesystem::path cache_file;
    std::mutex mtx;  // 多线程安全

public:
    ConfigCache();                                        // 读取磁盘缓存到内存
    std::optional<GemmConfig> get(const std::string& key);
    void put(const std::string& key, const GemmConfig& config);  // 同时写磁盘

private:
    static std::string serialize(const GemmConfig& config);
    static GemmConfig deserialize(const std::string& str);
};
```

**磁盘缓存文件**：`$DG_JIT_CACHE_DIR/config_cache.txt`（默认 `~/.deep_gemm/config_cache.txt`）

**文件格式**（每行一条记录，`|` 分隔 key 和 value）：
```
GemmDesc(gemm_type=0,...,expected_k=0)|0 128 128 64 1 1 0 128 0 0 128 0 1 2 49152 4 ...
```

### 3.2 修改文件：`csrc/jit_kernels/heuristics/common.hpp`

在 `get_best_config()` 开头加缓存查询，末尾加缓存写入：

```cpp
template <typename ArchSpec>
static GemmConfig get_best_config(const GemmDesc& desc) {
    desc.check_validity();

    // ── 新增：查缓存 ──────────────────────────────
    std::ostringstream oss;
    oss << desc;
    const auto key = oss.str();
    if (const auto cached = config_cache->get(key); cached.has_value())
        return cached.value();
    // ─────────────────────────────────────────────

    // 原有逻辑（不动）
    const auto layout_candidates = ArchSpec::get_layout_candidates(desc);
    ...
    const auto gemm_config = GemmConfig { ... };

    // ── 新增：存缓存 ──────────────────────────────
    config_cache->put(key, gemm_config);
    // ─────────────────────────────────────────────

    return gemm_config;
}
```

### 3.3 暴露给 Python（可选）：`csrc/apis/runtime.hpp`

```cpp
m.def("clear_config_cache", []() { config_cache->clear(); });
m.def("get_config_cache_size", []() { return config_cache->size(); });
m.def("set_config_cache_dir", [](const std::string& path) {
    config_cache->set_cache_dir(path);
});
```

---

## 4. 涉及改动的文件

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `csrc/jit/config_cache.hpp` | **新增** | ConfigCache 完整实现 |
| `csrc/jit_kernels/heuristics/common.hpp` | 修改 | 加缓存查询/写入（~10 行） |
| `csrc/apis/runtime.hpp` | 修改（可选） | 暴露 Python API |

---

## 5. 工作流

### 仓库结构
```
origin   → personal fork（地址省略）
upstream → team repository（地址省略）
```

### 开发循环
```
CPU 机器（CC 辅助写代码）
  → git push origin feature/config-cache
GPU 机器
  → git pull
  → pip install -e .  / python setup.py develop
  → python tests/test_fp8.py  验证正确性
  → 验证缓存是否命中（DG_PRINT_CONFIGS=1）
```

---

## 6. 待验证

- [ ] GPU 机器能否正常编译安装 deepgemm（`pip install -e .`）
- [ ] 并发场景下缓存写入的线程安全
- [ ] 缓存文件在多机分布式下的共享策略（NFS / 各机独立）
- [ ] `num_sms` 是运行时动态值，是否应纳入 key（不同机器 SM 数不同）
