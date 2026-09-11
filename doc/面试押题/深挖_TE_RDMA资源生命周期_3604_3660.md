# 深挖 · Transfer Engine RDMA 资源生命周期（#3604 + #3660）

> 日期：2026-08-31
> 配套：`面试押题_Mooncake_阿里_20260827.md` 的 §3 深挖④ / §4 速览
> 材料来源：本地仓 `/volume/pt-train/users/zhaoye/Mooncake`（origin=kvcache-ai，fork=XFDG）。#3604 的 `evictAll`/`resume` 改动在 XFDG fork 的 feature 分支上，主干 HEAD `160b963` grep 不到——面试讲时说清楚「这是我提的 PR，改动在我 fork 分支」。
> 一句话：这两条是 Transfer Engine 层「RDMA 资源不随生命周期回收」的两个侧面——一个泄漏**连接态**（死 QP），一个泄漏**地址空间**（VMA 碎片）。

---

## 0. 术语表（缩写第一次出现处也会再解释一遍）

| 缩写 | 全称 | 一句话解释 |
|---|---|---|
| **TE** | Transfer Engine | Mooncake 的高性能数据传输库，负责把 prefill 算出的 KVCache 搬到 decode 节点。 |
| **TENT** | 内部 RDMA 传输框架 | TE 内部的一套 RDMA 传输实现（管 context / endpoint_store / workers / QP）。 |
| **KVCache** | Key-Value Cache | Transformer 推理时缓存的注意力键值对，避免每步重算整段 prompt。 |
| **RDMA** | Remote Direct Memory Access（远程直接内存访问） | 网卡直接读写远程主机内存、绕过 CPU 和内核的传输技术，低延迟高吞吐。 |
| **QP** | Queue Pair（队列对） | RDMA 的连接单元，一收一发两个队列组成，是 RDMA 通信的最小连接实体。 |
| **WR** | Work Request（工作请求） | 投递给 QP 的一次收发操作，硬件异步执行。 |
| **CQ** | Completion Queue（完成队列） | 硬件把 WR 完成结果（成功或错误）写进这里，上层轮询它来感知 IO 是否完成。 |
| **endpoint** | 端点 | TENT 里对一个 peer（对端节点）连接的封装，内部持有若干 QP。 |
| **EndpointStore** | 端点缓存 | 缓存已建立的 endpoint，按淘汰策略（FIFO / SIEVE）管理，避免每次重建连接。 |
| **FIFO / SIEVE** | First-In-First-Out / SIEVE 淘汰算法 | 两种缓存淘汰策略，`EndpointStore` 有这两个实现。 |
| **MR** | Memory Region（内存区域） | 向网卡注册过、允许 RDMA 直接访问的一段内存。 |
| **VMA** | Virtual Memory Area（虚拟内存区域） | Linux 内核里描述进程一段连续虚拟地址映射的结构，`/proc/<pid>/maps` 里的每一行就是一个 VMA。 |
| **COW** | Copy-On-Write（写时复制） | fork 出子进程后父子共享物理页、谁写谁才复制的机制。 |
| **SLO** | Service Level Objective（服务级目标） | 服务承诺的性能指标，如首 token 延迟、token 间延迟的上限。 |

---

## 1. #3604 — RDMA 端口恢复不清 stale endpoint（TE 最对口）

### 1.1 现象

RDMA（远程直接内存访问）端口 down 再 up 后，传输**卡住**，而且反直觉地**比断网期间还慢 5×**：
- reporter 7-rail 实测：恢复期 **5.48s** vs 断网期 **1.03s**；
- 持续到 **20s 超时**才放。

### 1.2 根因链（面试官最想听的因果）

```
① 端口 down ──→ ② QP 进入 ERR 态（从未销毁/重建）
   PORT_ERR                    │
                               │
   ③ 端口恢复事件 PORT_ACTIVE ──→ ④ EndpointStore 继续返回 stale endpoint
      只调 context->resume()        （指向死 QP）
      翻转 DeviceStatus 标志          │
              ╎                       ▼
              ╎                  ⑤ WR 灌进死 QP，永不 complete
              └╌╌ 缺这条边          → 传输停滞 → 卡到 20s 超时
                  （应调 evictEndpoints）
```

逐步拆：

1. **端口 down** 触发 `IBV_EVENT_PORT_ERR`，处理它只调 `context->pause()`。
2. down 期间，那些 **QP（Queue Pair，队列对，RDMA 连接的最小单元）** 已经进入 `IBV_QPS_ERR` 状态——**从未被销毁或重建**。
3. 端口恢复触发 `IBV_EVENT_PORT_ACTIVE` 事件。处理它的代码（核过主干 `tent/src/transport/rdma/workers.cpp:852`）**只调了 `context->resume()`**——而 `resume()` 只翻转一个 `DeviceStatus` 标志回 `DEVICE_ENABLED`。
4. **EndpointStore（端点缓存）** 恢复后继续把这些 **stale endpoint（陈旧端点，仍指向死 QP）** 返回出去。
5. **WR（Work Request，工作请求）** 被 post 进死 QP，永远 **complete（在 CQ / Completion Queue 完成队列里出现完成结果）** 不了 → 传输停滞。

**为什么恢复比断网还慢**（证明你真跑过 reporter 的数、不是背概念）：
- 断网期间：请求快速失败、快速重试，所以 1.03s。
- 恢复后：链路"看起来"好了，请求往死 QP 里灌，既不成功也不报错，一直卡到 20s 超时才被上层放弃。**恢复反而更慢，是这条 bug 最反直觉的证据。**

### 1.3 我的修（4 文件，+60/-1）

```
resume() ──→ evictEndpoints() ──→ evictAll()  ──惰性──→ getOrInsert()
新增调用                          active map           建全新 endpoint
                                  → waiting list       + 新 QP → 满吞吐
                                       │
                                       └→ beginDestroy(): QP → ERR 态，
                                          硬件把 inflight WR flush 到 CQ
```

- 给 `EndpointStore` 加一个纯虚 `evictAll()`，在 **FIFO** 和 **SIEVE**（两种缓存淘汰算法，First-In-First-Out 与 SIEVE）两个实现里各自实现：把 active map 里每个 entry 挪到 waiting list 并调 `beginDestroy()`——让 QP 进 ERR 态、硬件把 **inflight WR（在飞的工作请求）** flush 到 **CQ（Completion Queue，完成队列）**。核过 `tent/src/transport/rdma/endpoint.cpp:334-346`，`beginDestroyNoLock` 确实是「把每个 QP `ibv_modify_qp` 到 `IBV_QPS_ERR`」这一步。
- `RdmaContext::resume()` 翻转状态回 `DEVICE_ENABLED` 后，调 `evictEndpoints()`。
- 下一次 `getOrInsert()` 给每个 peer 建**全新 endpoint + 新 QP**，恢复满吞吐。

### 1.4 关联 bug（体现 review 深度，这条最出彩）

这个 PR 的 CI 一开始就红。我去读 job log（`gh run view <id> --job <jobid> --log-failed`），发现是**真编译错**，不是 flaky——`SIEVEEndpointStore` 因为基类 `evictAll()=0` 是纯虚、而它**头文件里漏了 `evictAll()` 声明**（只有 `.cpp` 实现）→ `invalid new-expression of abstract class type`（抽象类不能实例化）。补上 `void evictAll() override;` 声明（commit `bd2e9d3`）。

> 这就是「主干绿 ≠ 我的 PR 没引入真 bug」的活例子：主干绿是因为没走到这条抽象类实例化路径，一旦我的改动引用它就暴露编译错。

### 1.5 预判追问 & 答法

- **「为什么 evictAll 而不是只重建坏的那几个 QP？」** → 端口 flap（抖动式 down→up）后无法可靠区分哪些 QP 还活着；全清 + 惰性重建（`getOrInsert` 按需建新的）最简单且正确，代价是恢复后每个 peer 第一次访问重建一次，可接受。
- **「beginDestroy 之后 inflight 的请求怎么办？」** → 转 ERR 态会让硬件把 inflight WR 以 error 形式 flush 到 CQ，上层能感知失败并重试到新 QP，而不是永远等一个死 QP。
- **「FIFO 和 SIEVE 两个 store 是什么？」** → endpoint 的两种缓存淘汰策略；`evictAll` 要在两个实现里都提供，否则用 SIEVE 的路径没法全清。
- **「CI 红你怎么判断是真 bug？」** → `gh run view --log-failed` 读日志，看到抽象类实例化错误，不是超时 / 环境问题；主干绿不代表我的 PR 没引入真编译错。

---

## 2. #3660 — unregister 不还原 MADV_DOFORK → VMA 耗尽（TE 速览）

### 2.1 现象

高频 register / unregister（多模态每请求都注册一段 buffer）后，`ibv_reg_mr`（注册 **MR / Memory Region，内存区域**）和 `ibv_create_qp` 全部返回 `ENOMEM`（内存不足）。

### 2.2 根因

```
ibv_fork_init() ──→ ibv_reg_mr 对注册区调 madvise(MADV_DONTFORK)
                          │
       unregister 路径 ──╳ 不还原 MADV_DOFORK
                          │
                          ▼
       每次注册切出一段永久 DONTFORK 的 VMA，分裂永不愈合
                          │
                          ▼
       高频注册/反注册 → VMA 数量爆炸 → 撞 vm.max_map_count 上限 → ENOMEM
```

- `ibv_fork_init()` 之后，`ibv_reg_mr` 会对注册区调 `madvise(addr, len, MADV_DONTFORK)`。`MADV_DONTFORK` 是告诉内核「fork 出的子进程不要继承这段映射」。
- 但 **unregister 路径不还原 `MADV_DOFORK`**（DOFORK 是 DONTFORK 的反操作，恢复「子进程可继承」）→ 每次注册都会在地址空间里切出一段永久标记 DONTFORK 的 **VMA（Virtual Memory Area，虚拟内存区域，内核里描述一段连续虚拟地址映射的结构）**，**分裂永不愈合**。
- 高频注册 / 反注册 → VMA 数量爆炸 → 撞上 `vm.max_map_count`（进程最大 VMA 数上限）→ 后续注册全 `ENOMEM`。

### 2.3 我的修

两条 unregister 路径（`rdma_transport` + TENT）在 `ibv_dereg_mr` 成功后补 `madvise(addr, len, MADV_DOFORK)`；还原失败只 `warn`——因为 dereg 本身已经成功了，内存已解注册，还原失败不影响正确性。

### 2.4 预判追问 & 答法

- **「MADV_DONTFORK 干嘛的？」** → RDMA（远程直接内存访问）注册内存的物理页不能被 fork 出的子进程 **COW（Copy-On-Write，写时复制）** 破坏映射（否则网卡 DMA 会写到错误的物理页），所以标 DONTFORK 让子进程不继承这段；但 **MR（内存区域）** 销毁后该段普通内存应恢复可 fork / 可合并，否则 VMA 一直碎、永不合并 → 耗尽 `max_map_count`。

---

## 3. 两条的共性（面 Mooncake 组时收尾用）

| PR | 泄漏的东西 | 触发场景 | 层次 |
|---|---|---|---|
| **#3604** | 死 QP / endpoint（**连接态**） | 端口 flap（网络抖动恢复） | 连接生命周期 |
| **#3660** | VMA 映射碎片（**地址空间**） | 高频 register/unregister（多模态） | 地址空间生命周期 |

> **一句话：** 都是「RDMA 资源不随生命周期回收」——#3604 是 QP/endpoint 不随端口 flap 回收（连接态泄漏），#3660 是 VMA 不随 unregister 回收（地址空间泄漏）。都在真实 churn（端口抖动 / 多模态高频注册）下才暴露，且都做了最小复现 + 定位。这套归纳能体现系统 sense。
