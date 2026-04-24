#include "fuse/client_api.h"

#include <cerrno>

namespace glfs {

namespace {

int noop_filler(void*, const char*, const struct stat*, off_t, enum fuse_fill_dir_flags) {
    return 0;
}

}  // namespace

int fuse_client_stat(const char* mount_point, const char* rel_path, struct stat* out) {
    if (!mount_point || !rel_path || !out) {
        return -EINVAL;
    }
    GPULearnedFS* fs = glfs::active_fs();
    if (!fs) {
        return -EIO;
    }
    (void)mount_point;
    return gpufs_getattr(rel_path, out, nullptr);
}

int fuse_client_readdir(const char* mount_point, const char* rel_path, void* buf, ::fuse_fill_dir_t filler) {
    if (!mount_point || !rel_path || !filler) {
        return -EINVAL;
    }
    GPULearnedFS* fs = glfs::active_fs();
    if (!fs) {
        return -EIO;
    }
    return gpufs_readdir(rel_path, buf, filler ? filler : noop_filler, 0, nullptr, static_cast<enum fuse_readdir_flags>(0));
}

}  // namespace glfs
