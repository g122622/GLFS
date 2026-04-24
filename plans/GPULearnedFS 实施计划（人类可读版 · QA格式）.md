# 📘 GPULearnedFS 实施计划（人类可读版 · QA格式）

> **一句话目标**：在WSL2上构建一个**原型文件系统**，用**5种可切换的GPU Learned Index**替代传统元数据查找，验证其在多模态训练元数据访问场景下的**吞吐加速收益**。  
> **核心读者**：你（开发者）、协作工程师、未来维护者  
> **文档性质**：实施指南 + 决策记录 + 风险清单

---

## 🎯 第一部分：项目概览

### Q: 这个项目到底要做什么？
**A**: 实现一个FUSE文件系统，当应用调用`stat()/readdir()`时：
1. 将文件路径通过**Trie前缀编码**转为`uint64`整数键
2. 用**GPU上的Learned Index模型**（5种可选）预测该路径对应的inode
3. 直接通过inode访问ext4后端，跳过传统目录树遍历
4. 全程记录性能指标，与"纯FUSE"和"CPU Learned Index"基线对比

### Q: 为什么选WSL2而不是原生Linux或Windows驱动？
**A**: 权衡结果：
| 方案 | 优点 | 缺点 | 我们的选择 |
|-----|------|------|-----------|
| Windows Minifilter | 原生集成 | 需WHQL签名、调试复杂、无CUDA直接支持 | ❌ |
| 原生Linux + FUSE | 生态成熟 | 需双系统/虚拟机，割裂开发环境 | ❌ |
| **WSL2 + FUSE + CUDA** | ✅ 单环境开发 ✅ NVIDIA官方支持CUDA on WSL2 ✅ 接近原生ext4性能 | ⚠️ 需注意`/mnt/c`性能陷阱 | ✅ **首选** |

### Q: 验证成功的标准是什么？
**A**: 满足以下全部条件即视为理论验证成功：
- ✅ **功能正确**：`stat`查询结果与ext4原生一致（严格模式，miss返回`-ENOENT`）
- ✅ **性能达标**：在**混合负载**（70% stat + 20% open + 10% create）下，GPU variant 的 **throughput_qps ≥ 2.5× CPU variant**
- ✅ **资源合规**：索引模型+缓存 **显存占用 ≤ 1GB**（RTX 5060的7.7GB中预留充足余量）
- ✅ **可复现**：提供一键脚本，新人30分钟内可跑通benchmark对比

---

## ❓ 第二部分：核心决策问答（按模块分类）

### 🔹 功能设计 (Functionality)

#### Q: 路径怎么转成Learned Index能处理的整数键？(F1)
**A**: 采用 **Trie前缀编码**（`path_encoder`模块）：
```
示例: /data/img/cat/001.jpg
→ 拆分: ["data", "img", "cat", "001.jpg"]
→ 每层哈希: hash("data")=0x3A, hash("img")=0x7F, ...
→ 拼接: key = (0x3A<<48) | (0x7F<<32) | (0x12<<16) | 0x8B
→ 优势: 保留层级语义，readdir时可批量预测子节点
```
- 编码参数（`max_depth=32`, `bits_per_level=8`）通过config配置
- 编码过程**无状态、线程安全**，适合FUSE多线程回调

#### Q: 查询一个不存在的路径时，系统怎么响应？(F2)
**A**: **严格模式**（Strict Mode）：
- GPU index返回`INVALID_INODE` → 立即返回`-ENOENT`给应用
- **不fallback到ext4原生lookup**（避免污染benchmark结果）
- 同时记录miss事件到Perfetto，供后续分析index覆盖度

#### Q: 系统启动时要做什么？(F3)
**A**: **同步阻塞式初始化**（简单可靠，适合原型）：
```
1. 扫描数据集目录，收集所有 (path, inode) 对
2. 用path_encoder批量编码为 (key, value)
3. 调用选定index后端的train()方法（CPU侧）
4. 将训练好的模型参数cudaMemcpy到GPU显存
5. 完成mount，开始响应FUSE请求
```
- 大数据集（1000万文件）预计初始化时间：3~8分钟
- 后续支持**离线预训练**（save/load .index文件）作为优化项

#### Q: 显存只有1GB预算，怎么分配？(F4)
**A**: **全量常驻策略**（静态索引 + 可控规模）：
```
预算分配:
• 模型参数: ≤ 600 MB (选用轻量模型如PGM-Index)
• 键值映射表: ≤ 200 MB (1000万 × (8B key + 8B value + 8B meta))
• CUDA workspace: ≤ 150 MB (batch lookup临时缓冲)
• 预留: ≥ 50 MB (驱动开销 + 安全余量)
```
- 通过config的`index.resource.max_vram_bytes`硬约束
- 启动时检查：若预估占用 > 预算，直接报错退出

#### Q: 需要模拟哪些POSIX错误码？(F5)
**A**: **最小必要集合**（保证benchmark公平性）：
- `-ENOENT`: 路径不在索引中（核心语义）
- `-EACCES`: ext4后端返回权限错误（透传）
- 其他错误（如`-ENOMEM`）在原型阶段统一映射为`-EIO` + 日志

---

### 🔹 模块设计 (Modularity)

#### Q: 5种GPU Index怎么统一接口？(M1)
**A**: 定义**抽象基类 `IGPUIndex`**，每个后端实现一个派生类：
```cpp
class IGPUIndex {
public:
  virtual void train(keys, values, config) = 0;
  virtual vector<uint64_t> batch_lookup(keys, cudaStream_t) = 0;
  virtual bool save/load(filepath) = 0;
  virtual IndexStats get_stats() = 0;  // p50/p99/throughput/gpu_util
  virtual void enable_profiling(bool) = 0;
  virtual size_t get_vram_usage() = 0;
};
```
- **编译时选择**：通过`-DINDEX_TYPE=RMI`等macro链接具体后端（避免运行时依赖冲突）
- **性能分析**：`get_stats()`返回标准化指标，benchmark脚本统一采集

#### Q: 各index仓库的CUDA版本/依赖冲突怎么解决？(M2)
**A**: **运行时单后端策略**（简单务实）：
- 每次编译只链接**一个**index后端（如`-DINDEX_TYPE=GIDX`）
- 对比不同index时：**重新编译**（CMake切换macro + `make clean`）
- 优势：避免符号冲突、依赖地狱；缺点：切换需重编（原型阶段可接受）
- 未来优化：用`dlopen`动态加载.so（需统一ABI，留作V2规划）

#### Q: 配置文件用什么格式？(M3)
**A**: **分层JSON**（无需热重载，结构清晰）：
```json
{
  "fs": {
    "mount_point": "/mnt/gpufs",
    "fuse_opts": ["-o", "auto_cache"]
  },
  "index": {
    "type": "g-index",
    "training": { "sample_ratio": 1.0, "key_encoding": "trie_prefix" },
    "inference": { "batch_size": 256, "fallback_on_miss": false },
    "resource": { "max_vram_bytes": 1073741824 }
  },
  "benchmark": {
    "warmup_iters": 100,
    "metrics": ["p50", "p99", "throughput"]
  }
}
```
- 用`nlohmann/json`（header-only）解析，启动时校验必填字段
- 校验失败时：打印友好错误 + 退出码1，方便CI集成

#### Q: 性能数据怎么收集和可视化？(M4)
**A**: **双通道埋点**：
1. **FUSE内置埋点**：每个op前后记录`clock_gettime(CLOCK_MONOTONIC)`，输出CSV
2. **Perfetto C++ SDK**：在关键路径插入`PERFETTO_TRACK_EVENT`：
   - `fuse.getattr.entry` / `.exit`
   - `index.lookup.start` / `.end`
   - `ext4.fallback`（虽不用，但预留）
- 输出：`trace_*.perfetto.json` + `metrics_*.csv`
- 可视化：Perfetto UI + 自定义Python绘图脚本

---

### 🔹 并发设计 (Concurrency)

#### Q: FUSE多线程下怎么保证index查询安全？(C1)
**A**: **全局互斥锁**（简单可靠，静态索引无写竞争）：
```cpp
// 在gpufs_getattr等回调中:
pthread_mutex_lock(&fs->global_lock);  //  entry
// 1. path encoding (stateless)
// 2. index->batch_lookup() (read-only, but config access needs sync)
// 3. ext4 delegation
pthread_mutex_unlock(&fs->global_lock);  // exit
```
- 理由：静态索引只读，无内部状态修改；全局锁开销远低于FUSE+ext4本身
- 未来优化：若验证锁成为瓶颈，可升级为`pthread_rwlock`或lock-free读

#### Q: GPU kernel调用怎么异步化？(C2)
**A**: **CUDA Stream + Callback**（平衡灵活性与复杂度）：
```cpp
// batch_lookup实现:
cudaStream_t stream;
cudaStreamCreate(&stream);

// 1. 分配pinned host buffer for results
cudaHostRegister(results_host, size, cudaHostRegisterDefault);

// 2. 启动后端kernel
launch_backend_kernel(keys_d, results_d, stream);

// 3. 注册回调：kernel完成后填充host结果
cudaStreamAddCallback(stream, [](cudaStream_t, cudaError_t, void* ctx) {
  auto* req = static_cast<LookupRequest*>(ctx);
  memcpy(req->out_vec, req->results_host, size);  // 非阻塞copy
  req->promise.set_value();  // 通知调用方
}, req, 0);

// 4. 立即返回，调用方可wait promise或继续其他工作
return future;
```
- 优势：支持batch overlap，适合DataLoader预取场景
- 注意：WSL2下测试`cudaStreamAddCallback`实际延迟，必要时fallback同步

#### Q: 静态索引的并发不变量是什么？(C3)
**A**: **强只读不变量**（加载后永不修改）：
- 训练/加载完成后，索引的GPU显存内容**禁止任何写操作**
- 所有查询路径**无需加锁**（原子读+无副作用）
- 实现保障：`IGPUIndex`派生类构造函数中标记`is_trained=true`，析构前拒绝train()

#### Q: Benchmark程序怎么避免干扰测量？(C4)
**A**: **串行执行 + 资源隔离**：
- 每次只运行**一个**benchmark场景（随机stat / 顺序readdir / 混合 / 突发）
- 运行前：`sync; echo 3 > /proc/sys/vm/drop_caches` 清空page cache
- 运行中：用`nvidia-smi dmon`监控GPU util，确保无其他CUDA进程
- 运行后：自动收集`dmesg` + `perf stat`辅助分析

---

### 🔹 环境与部署 (WSL2 Specific)

#### Q: 数据存在哪？性能有坑吗？(W1)
**A**: **坚持使用 `/home/user/data`（WSL2原生ext4）**：
```
✅ 推荐: /home/yourname/datasets/multimodal/
   - 位于WSL2虚拟磁盘，原生ext4语义
   - 小文件元数据操作性能接近原生Linux

❌ 避免: /mnt/c/Users/.../data
   - 通过9P协议桥接，小文件stat延迟高3-10倍
   - 不适合元数据密集型benchmark
```
- 验证命令：`time find /home/user/data -name "*.jpg" | wc -l` vs `/mnt/c/...`

#### Q: CUDA on WSL2有什么限制？(W2)
**A**: **先按标准CUDA开发，遇到问题再patch**（快速迭代优先）：
- 已知注意点：
  - 避免`cudaMallocManaged`（统一内存），用显式`cudaMemcpy`
  - 单GPU场景，无需处理`cudaDevicePeerAccess`
  - `cudaHostRegister` pinned memory在WSL2下有效，可加速传输
- 验证清单：
  ```bash
  nvidia-smi                    # 应显示 RTX 5060
  cuda-install-samples-12.x.sh ~
  cd ~/NVIDIA_CUDA-12.x_Samples/1_Utilities/deviceQuery
  make && ./deviceQuery        # Result = PASS
  ```

---

### 🔹 Benchmark设计

#### Q: 测试数据集怎么生成？(B1)
**A**: **可配置合成生成器**（`scripts/generate_dataset.py`）：
```python
# 核心参数:
scale_factor: float  # 1.0 = 100万文件, 10.0 = 1000万文件
depth_dist: "lognormal(μ=3, σ=1)"  # 目录深度分布
files_per_dir: "zipf(α=1.2)"       # 每目录文件数分布
file_exts: [".jpg", ".mp4", ".txt"] # 多模态扩展名

# 输出:
/home/user/data/
├── train/
│   ├── img/
│   │   ├── cat/000001.jpg ... (60%)
│   │   └── dog/...
│   ├── video/... (30%)
│   └── text/... (10%)
└── val/ (相同结构，用于验证泛化)
```
- 生成同时输出`training_set.jsonl`: `{"path": "...", "inode": 12345}`

#### Q: 负载怎么模拟真实训练场景？(B2)
**A**: **合成负载 + 参数可调**（可控 + 可复现）：
| 场景 | 请求分布 | 模拟目标 |
|-----|---------|---------|
| `random_stat` | 均匀随机采样路径 | `find /data -name "*.jpg"`类扫描 |
| `seq_readdir` | 按目录树BFS顺序 | DataLoader遍历dataset |
| `mixed_rw` | 70% stat + 20% open + 10% create | 训练step真实workload |
| `burst_access` | Poisson到达 + 批量突发 | multi-node训练元数据风暴 |

- 每个场景支持`--iters N` + `--batch-size M`参数

#### Q: 对比基线怎么保证公平？(B3)
**A**: **三层隔离设计**：
```
[相同物理文件]
     ↓
[相同FUSE客户端API] ← 所有方案通过同一套fuse_client_*调用
     ↓
[不同实现层]
├─ Baseline1: FUSE passthrough → ext4 (无index)
├─ Baseline2: FUSE + CPU Learned Index → ext4 (同模型，无CUDA)
└─ Ours:     FUSE + GPU Learned Index → ext4 (目标方案)
```
- 关键控制：
  - 所有方案使用**同一份**`/home/user/data`
  - Benchmark脚本**自动切换mount点**，无需手动干预
  - 预热策略统一：前100次迭代不计入统计

#### Q: 最关注哪个性能指标？(B4)
**A**: **吞吐 (throughput_qps)** 为主，延迟为辅：
- **Primary**: `total_valid_ops / (end_time - start_time - warmup_time)`
- **Secondary**: p50/p99/p999 latency（用`std::nth_element`高效计算）
- **Diagnostic**: GPU util%, VRAM usage, index miss rate（仅debug用）
- 输出格式：
  ```csv
  scenario,index_type,throughput_qps,p50_us,p99_us,vram_mb,gpu_util
  mixed_rw,g-index,15234,8.2,24.1,892,78.3
  mixed_rw,cpu-rmi,5821,22.1,67.4,-,-
  mixed_rw,fuse-passthrough,2103,45.6,120.3,-,-
  ```

---

## 🗓️ 第三部分：分阶段实施计划

### 📅 Phase 0: 环境准备（1天）
```bash
# 交付物: 可运行CUDA的WSL2 + 依赖安装脚本
- [ ] wsl --update + 安装NVIDIA驱动
- [ ] 验证: nvidia-smi + deviceQuery PASS
- [ ] 安装: libfuse3-dev, cmake, nlohmann-json, perfetto-sdk
- [ ] 创建项目骨架: `mkdir -p gpufs/{src,scripts,specs}`
```

### 📅 Phase 1: 核心模块原型（5天）
```bash
# 交付物: 可mount的FUSE FS，支持stat查询（单index后端）
- [ ] path_encoder: Trie前缀编码 + 单元测试
- [ ] config_manager: JSON加载 + 校验
- [ ] gpu_index_adapter: IGPUIndex抽象类 + RMI-CUDA后端实现
- [ ] fuse_ops: getattr/readdir最小实现 + global lock
- [ ] 验证: `mount -t fuse gpufs /mnt/gpufs && stat /mnt/gpufs/data/img/cat/001.jpg`
```

### 📅 Phase 2: 多后端 + Benchmark（4天）
```bash
# 交付物: 5种index可切换 + 完整benchmark脚本
- [ ] 集成G-Index/cuLearnedIndex后端（按M2-D策略，编译时切换）
- [ ] benchmark_runner: 4种负载生成 + Perfetto埋点 + CSV输出
- [ ] Baseline实现: FUSE passthrough + CPU RMI
- [ ] 验证: `./scripts/run_benchmarks.sh --index g-index --scale 0.1`
```

### 📅 Phase 3: 优化与文档（2天）
```bash
# 交付物: 性能报告 + 用户指南 + 风险清单
- [ ] 调优: batch_size / pinned memory / stream并发
- [ ] 编写: README.md（含30分钟快速开始）
- [ ] 输出: `report_v1.pdf`（含对比表格 + Perfetto截图）
- [ ] 归档: specs/ 目录（SYSSPEC规范）+ src/ 代码
```

> 🎯 **总周期**: ~12人日（单人全职约2.5周），可并行任务已标注

---

## 🧰 第四部分：开发准备清单

### ✅ 软件依赖
```bash
# 系统包 (Ubuntu 24.04)
sudo apt install -y \
  build-essential cmake git libfuse3-dev \
  nvidia-cuda-toolkit nvidia-container-runtime \
  python3 python3-pip perf

# Python依赖 (benchmark脚本)
pip install numpy pandas matplotlib seaborn

# Header-only库 (vendored in src/vendor/)
# - nlohmann/json (v3.11.3)
# - perfetto-sdk (v47.0, C++17 compatible)
```

### ✅ 项目结构
```
gpufs/
├── CMakeLists.txt          # 顶层构建，支持 -DINDEX_TYPE=xxx
├── configs/
│   ├── default.json        # 默认配置模板
│   └── schema.json         # JSON Schema for validation
├── src/
│   ├── main.cpp            # fuse_main entry
│   ├── config/
│   │   ├── config_manager.h/cpp
│   ├── core/
│   │   ├── path_encoder.h/cpp
│   │   ├── gpu_index_adapter.h  # IGPUIndex abstract
│   │   └── backends/          # 每个后端一个子目录
│   │       ├── rmi_cuda/
│   │       ├── g_index/
│   │       └── ...
│   ├── fuse/
│   │   ├── gpufs_ops.h/cpp
│   │   └── client_api.h/cpp   # benchmark调用的统一接口
│   ├── benchmark/
│   │   ├── runner.h/cpp
│   │   ├── workloads/         # random_stat.cpp, mixed_rw.cpp...
│   │   └── perfetto_integration.h
│   └── utils/
│       ├── timer.h, error_mapper.h, ...
├── scripts/
│   ├── generate_dataset.py   # B1-D 数据生成
│   ├── run_benchmarks.sh     # B3 一键对比
│   ├── plot_results.py       # 生成对比图表
│   └── wsl2_setup.sh         # W1/W2 环境检查
├── specs/                    # SYSSPEC规范（机器可读）
│   ├── path_encoder.spec
│   ├── gpu_index_adapter.spec
│   └── ...
└── docs/
    ├── README.md            # 人类指南（本文件精简版）
    ├── ARCHITECTURE.md      # 架构图 + 数据流
    └── TROUBLESHOOTING.md   # WSL2+CUDA常见问题
```

### ✅ 关键命令速查
```bash
# 编译 (选择index后端)
cmake -B build -DINDEX_TYPE=GIDX -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j$(nproc)

# 运行 (开发模式)
./build/src/gpufs -f configs/default.json -s  # -s = foreground + debug log

# Benchmark (对比GPU vs CPU)
./scripts/run_benchmarks.sh \
  --scale 0.1 \          # 100万文件
  --iters 10000 \        # 每场景1万次请求
  --backends g-index,cpu-rmi,fuse-passthrough

# 分析结果
python scripts/plot_results.py --input results/mixed_rw_20240601.csv
```

---

## ⚠️ 第五部分：风险与应对

| 风险 | 概率 | 影响 | 应对方案 |
|-----|------|------|---------|
| **CUDA on WSL2兼容性问题** | 中 | 高 | ✅ 优先用标准CUDA API；❌ 避免实验性特性；🔧 预留`#ifdef __WSL__`补丁点 |
| **5种index依赖冲突** | 高 | 中 | ✅ 编译时单后端策略（M2-D）；📦 用Docker隔离构建环境（可选） |
| **1GB显存不够用** | 低 | 高 | ✅ 启动时预估+硬校验；🗜️ 优先选用PGM-Index等轻量模型；📉 支持`sample_ratio<1.0`降规模 |
| **FUSE多线程+GPU上下文冲突** | 中 | 中 | ✅ 全局锁保守方案（C1-B）；🔍 用`cudaGetLastError`+Perfetto定位；🚀 后续可升级lock-free |
| **Benchmark结果波动大** | 高 | 低 | ✅ 固定随机种子 + drop_caches；📊 报告95%置信区间；🔁 自动重试3次取中位数 |
| **时间超期** | 中 | 中 | ✅ Phase 1交付最小可行原型（仅RMI+stat）；🎯 优先保证throughput指标，延迟优化留V2 |

---
