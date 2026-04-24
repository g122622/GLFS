#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <ctime>

#ifndef FUSE_USE_VERSION
#define FUSE_USE_VERSION 31
#endif

#include <fuse3/fuse.h>

#include "config/config_manager.h"
#include "core/gpu_index_adapter.h"
#include "core/path_encoder.h"
#include "fuse/backing_root_proxy.h"

namespace glfs {

struct GPULearnedFS {
    struct NodeEntry {
        std::uint64_t inode = 0;
        bool is_dir = false;
        mode_t mode = 0;
        uid_t uid = 0;
        gid_t gid = 0;
        std::uint64_t size = 0;
        std::timespec atime{};
        std::timespec mtime{};
        std::timespec ctime{};
        std::vector<std::uint8_t> data;  // files only
    };

    IGPUControlPlane* control_plane = nullptr;
    BackingRootProxy backing_root;
    PathConfig path_cfg;
    std::string mount_point = "/home/user/data";
    bool strict_mode = false;
    bool usable_mode = true;
    mutable std::mutex global_lock;
    std::map<std::string, NodeEntry> nodes;
    std::map<std::string, std::vector<std::string>> children;
    std::uint64_t next_inode = 1;
    bool verbose = false;
};

void gpufs_init(GPULearnedFS& fs,
                IGPUControlPlane* control_plane,
                const FSConfig& cfg);

int gpufs_getattr(const char* path, struct stat* stbuf, struct fuse_file_info* fi);
int gpufs_readdir(const char* path, void* buf, fuse_fill_dir_t filler, off_t offset, struct fuse_file_info* fi, enum fuse_readdir_flags flags);
int gpufs_open(const char* path, struct fuse_file_info* fi);
int gpufs_create(const char* path, mode_t mode, struct fuse_file_info* fi);
int gpufs_unlink(const char* path);
int gpufs_mkdir(const char* path, mode_t mode);
int gpufs_rmdir(const char* path);
int gpufs_rename(const char* from, const char* to, unsigned int flags);
int gpufs_truncate(const char* path, off_t size, struct fuse_file_info* fi);
int gpufs_read(const char* path, char* buf, size_t size, off_t offset, struct fuse_file_info* fi);
int gpufs_write(const char* path, const char* buf, size_t size, off_t offset, struct fuse_file_info* fi);
int gpufs_utimens(const char* path, const struct timespec tv[2], struct fuse_file_info* fi);

GPULearnedFS* active_fs();
void set_active_fs(GPULearnedFS* fs);
struct fuse_operations build_fuse_operations();

}  // namespace glfs
