#pragma once

#include <sys/stat.h>

#include "fuse/gpufs_ops.h"

namespace glfs {

int fuse_client_stat(const char* mount_point, const char* rel_path, struct stat* out);
int fuse_client_readdir(const char* mount_point, const char* rel_path, void* buf, fuse_fill_dir_t filler);

}  // namespace glfs
