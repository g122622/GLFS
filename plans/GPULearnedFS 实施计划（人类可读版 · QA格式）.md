# GAMA 实施计划（人类可读版 · QA 格式）

> **目标**：把 `gpufs` 重构成 **CPU thin shim + GPU control plane** 架构，并让 usable mode 通过 `backing_root` 真实代理到 ext4 路径。

---

## Q1: 当前这一步最重要的工程目标是什么？

**A1:**
把现有原型从“内存态文件系统”改造成“GPU 元数据控制面 + CPU 入口层 + ext4 backing root”的结构，具体包括：
- CPU 只保留 libfuse3 入口、请求封包、错误返回、trace。
- GPU control plane 负责元数据查找、写入决策、统计与批处理。
- usable mode 下，miss 或数据路径操作可以代理到 `backing_root`。
- strict mode 下，accelerator miss 直接返回 `-ENOENT`。

---

## Q2: 为什么要这样拆？

**A2:**
因为现在代码里，`gpufs_ops.cpp` 仍然把 runtime map、children map、索引训练、FUSE 回调全混在一起。这样短期能跑，但和新 SPEC 不一致。拆成两层后：
- 语义更清楚
- 后续接真实 CUDA 更容易
- usable mode 的 ext4 delegation 更容易做
- strict mode 的 benchmark 更可信

---

## Q3: 这次代码重构要改哪些核心文件？

**A3:**
优先级如下：
1. `src/fuse/gpufs_ops.h`
2. `src/fuse/gpufs_ops.cpp`
3. `src/fuse/client_api.cpp` / `.h`
4. `src/config/config_manager.h` / `.cpp`
5. `src/main.cpp`
6. `src/core/gpu_index_adapter.h` / `.cpp`
7. `CMakeLists.txt`

---

## Q4: 新架构里 `GPULearnedFS` 应该长什么样？

**A4:**
建议拆成两层状态：

### CPU shim 层
- `mount_point`
- `backing_root`
- `strict_mode`
- `usable_mode`
- FUSE 运行时锁
- 活跃请求指针

### GPU control plane 层
- `IGPUControlPlane` 或等价接口
- 路径编码器
- metadata table
- request queue
- stats

在当前阶段，如果 GPU 还没接真实实现，可以先把 control plane 做成一个**显式接口 + 本地 stub**，但接口必须先定下来。

---

## Q5: usable mode 的真实 ext4 fallback 怎么做？

**A5:**
采用配置指定的 `fs.backing_root`。规则是：
- 所有 backing 路径通过 `backing_root + relative_path` 映射。
- `getattr/readdir/open/read/write/create/mkdir/rename/unlink/rmdir/truncate` 在 usable mode 下都可以按规则代理。
- strict mode 绝不走这个路径。

---

## Q6: 这一步是不是要直接接真实 CUDA？

**A6:**
是的，这次按你的要求，**直接把 CUDA 依赖接进构建**，但实现可以分两层：
- 构建层：允许 CUDA 编译/链接
- 运行层：先把 GPU control plane 做成可替换的后端接口

也就是说，先把“GPU 侧接口”钉死，再逐步把 stub 换成真实 kernel / cuda runtime 实现。

---

## Q7: 最小可交付版本要包含什么？

**A7:**
最小交付应当同时满足：
- 能编译
- 能以 FUSE 挂载
- 能在 strict / usable 两种模式下运行
- usable mode 能读写 `backing_root`
- `getattr/readdir/create/mkdir/unlink/rmdir/rename/truncate/read/write` 语义可用
- benchmark 仍可跑 `random_stat / seq_readdir / mixed_rw`

---

## Q8: GPU control plane 接口要怎么定义？

**A8:**
建议至少包含这些概念：
- `submit_lookup`
- `submit_mutation`
- `flush`
- `get_stats`
- `train`
- `batch_lookup`

如果你愿意，我下一步会把这些接口直接写进 `gpu_index_adapter`，让它从“learned index”演进成“control plane facade”。

---

## Q9: 我需要你先确认的唯一风险点是什么？

**A9:**
当前最大的风险是：**strict / usable 的语义和真实 ext4 代理一旦接进来，现有的内存态节点表会和 backing root 双写或失真。**

所以必须先定一个原则：
- 是“内存元数据为主，ext4 只是数据后端”
- 还是“ext4 为真源，GPU 只做缓存/加速”

这个问题需要你明确确认。

---

## Q10: 现在我建议的实施顺序是什么？

**A10:**
### Step 1
重构配置：补 `backing_root` / `strict_mode` / `usable_mode`。

### Step 2
拆分 FUSE 状态：把 CPU shim 和 GPU control plane 接口分开。

### Step 3
把 `gpufs_getattr/readdir/...` 改成先走 control plane，再决定是否 fallback。

### Step 4
接入 `backing_root` 的真实文件系统代理。

### Step 5
更新 benchmark 和 README，保证文档与代码一致。

---

## Q11: 这次重构后，代码风格要注意什么？

**A11:**
- 接口先行
- 语义先行
- GPU control plane 的边界不能再模糊
- 任何 fallback 都必须显式
- 代码要能支撑后续真实 CUDA 实现，而不是把 stub 当最终形态

---

## Q12: 什么时候可以开始动手改代码？

**A12:**
我现在就可以开始，但在动手前你必须先回答上一条风险问题：

> **内存元数据为主，还是 ext4 为真源？**

这个答案会决定我怎么改 `gpufs_ops.cpp` 里的状态模型。

---

## Q13: 当前建议的代码重构结论是什么？

**A13:**
把 `gpufs` 改成：
- **CPU thin shim**：libfuse3 入口 + request routing
- **GPU control plane**：元数据控制接口 + stats + future CUDA backend
- **usable mode**：通过 `backing_root` 显式代理 ext4
- **strict mode**：accelerator miss 直接失败

---

## Q14: 最后一句话总结？

**A14:**
这次改造的本质不是“再做一个文件系统”，而是把 `gpufs` 变成一个**GPU-assisted metadata control plane for ext4-like filesystems**。
