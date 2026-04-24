# GPULearnedFS

一个面向 WSL2 + libfuse3 + CUDA 的原型文件系统项目。

## 快速启动

### 1. 构建

```bash
cmake -S . -B build -DGLFS_BUILD_TESTS=ON -DGLFS_BUILD_BENCHMARKS=ON
cmake --build build -j$(nproc)
```

### 2. 挂载

```bash
./scripts/mount.sh
```

默认挂载点：`/home/user/data`

### 3. 卸载

```bash
./scripts/umount.sh /home/user/data
```

## 运行参数

- `--config FILE`：配置文件路径
- `--mount DIR`：挂载目录
- `--background`：不强制前台模式
- 其余参数会原样传给 FUSE

## 当前支持的原型语义

已实现（原型级，内存态）：

- `stat/getattr`
- `readdir`
- `mkdir/rmdir`
- `create/unlink`
- `rename`
- `read/write/truncate`
- `utimens`

说明：当前版本不接入真实 ext4 后端，文件内容与元数据都在进程内存中，重启/卸载后不会持久化。

## 关于“把 ext4 目录挂载到 GLFS”

如果你的目标是“把已有 ext4 目录作为 GLFS 的后端数据源”，这是合理的下一步：

- 不建议用 Linux 的 `mount --bind` 直接把 ext4 目录叠到同一个挂载点（会和 FUSE 挂载冲突）
- 正确做法一般是：GLFS 自己维护一个 `backing_root`（例如 `/home/user/data_real`），在 FUSE 回调中把路径映射到该目录并用 `openat/statat/readdir` 等系统调用代理

是否与计划/spec 冲突取决于你想要的 benchmark 严格模式（比如 miss 是否允许 fallback），这需要对照 `plans/GLFS SPEC.md` 和 `plans/GPULearnedFS 实施计划（人类可读版 · QA格式）.md` 再定。

## WSL2 注意事项

- 避免把数据放在 `/mnt/c`
- 需要安装 `libfuse3-dev`
- 挂载时建议先确认 `fusermount3` 可用
