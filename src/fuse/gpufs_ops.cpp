#include "fuse/gpufs_ops.h"

#include <cerrno>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <ctime>
#include <unistd.h>

#include "utils/perfetto_integration.h"
#include "utils/timer.h"

namespace glfs {

namespace {

GPULearnedFS*& active_fs_storage() {
    static GPULearnedFS* fs = nullptr;
    return fs;
}

std::string join_mount_path(const std::string& mount_point, const char* path) {
    const std::string rel = (path && path[0] == '/') ? std::string(path) : (path ? std::string(path) : std::string());
    if (rel.empty()) {
        return mount_point;
    }
    if (rel.front() == '/') {
        if (mount_point.empty() || mount_point == "/") {
            return normalize_path(rel);
        }
        if (rel.rfind(mount_point, 0) == 0) {
            return normalize_path(rel);
        }
        if (mount_point.back() == '/') {
            return normalize_path(mount_point + rel.substr(1));
        }
        return normalize_path(mount_point + rel);
    }
    if (mount_point.back() == '/') {
        return normalize_path(mount_point + rel);
    }
    return normalize_path(mount_point + "/" + rel);
}

std::string parent_of(const std::string& abs) {
    auto parent = std::filesystem::path(abs).parent_path().string();
    if (parent.empty()) {
        return "/";
    }
    return parent;
}

std::string basename_of(const std::string& abs) {
    return std::filesystem::path(abs).filename().string();
}

void fill_stat_from_inode(std::uint64_t inode, struct stat* stbuf) {
    std::memset(stbuf, 0, sizeof(*stbuf));
    stbuf->st_ino = inode;
    stbuf->st_mode = S_IFREG | 0444;
    stbuf->st_nlink = 1;
    stbuf->st_size = static_cast<off_t>(inode % 4096ULL + 128ULL);
    stbuf->st_blksize = 4096;
    stbuf->st_blocks = (stbuf->st_size + 511) / 512;
}

std::timespec now_ts() {
    std::timespec ts{};
    ::clock_gettime(CLOCK_REALTIME, &ts);
    return ts;
}

void fill_stat_from_node(const GPULearnedFS::NodeEntry& n, struct stat* st) {
    std::memset(st, 0, sizeof(*st));
    st->st_ino = n.inode;
    st->st_mode = n.mode;
    st->st_nlink = n.is_dir ? 2 : 1;
    st->st_uid = n.uid;
    st->st_gid = n.gid;
    st->st_size = static_cast<off_t>(n.is_dir ? 4096 : n.size);
    st->st_blksize = 4096;
    st->st_blocks = (st->st_size + 511) / 512;
    st->st_atim = n.atime;
    st->st_mtim = n.mtime;
    st->st_ctim = n.ctime;
}

}  // namespace

GPULearnedFS* active_fs() {
    return active_fs_storage();
}

void set_active_fs(GPULearnedFS* fs) {
    active_fs_storage() = fs;
}

void gpufs_init(GPULearnedFS& fs, IGPUIndex* index, const PathConfig& cfg, const std::vector<std::pair<std::string, std::uint64_t>>& training_set) {
    fs.index = index;
    fs.path_cfg = cfg;
    fs.mount_point = cfg.mount_point;
    fs.nodes.clear();
    fs.children.clear();
    fs.next_inode = 2;

    {
        GPULearnedFS::NodeEntry root;
        root.inode = 1;
        root.is_dir = true;
        root.mode = static_cast<mode_t>(S_IFDIR | 0755);
        root.uid = static_cast<uid_t>(::getuid());
        root.gid = static_cast<gid_t>(::getgid());
        root.size = 0;
        root.atime = root.mtime = root.ctime = now_ts();
        fs.nodes[fs.mount_point] = std::move(root);
    }
    fs.children[fs.mount_point] = {};

    std::vector<std::uint64_t> keys;
    std::vector<std::uint64_t> values;
    for (const auto& [path, inode] : training_set) {
        const auto normalized = normalize_path(path);
        const bool is_dir = std::filesystem::path(normalized).extension().empty();
        GPULearnedFS::NodeEntry entry;
        entry.inode = inode;
        entry.is_dir = is_dir;
        entry.mode = static_cast<mode_t>(is_dir ? (S_IFDIR | 0755) : (S_IFREG | 0644));
        entry.uid = static_cast<uid_t>(::getuid());
        entry.gid = static_cast<gid_t>(::getgid());
        entry.size = 0;
        entry.atime = entry.mtime = entry.ctime = now_ts();
        entry.data.clear();
        fs.nodes[normalized] = std::move(entry);
        const auto parent = parent_of(normalized);
        fs.children[parent].push_back(basename_of(normalized));
        if (is_dir && fs.children.find(normalized) == fs.children.end()) {
            fs.children[normalized] = {};
        }
        const auto key = encode_path(path, cfg);
        if (key.value != EncodedKey::INVALID_KEY) {
            keys.push_back(key.value);
            values.push_back(inode);
        }
    }
    for (auto& [_, child_list] : fs.children) {
        std::sort(child_list.begin(), child_list.end());
        child_list.erase(std::unique(child_list.begin(), child_list.end()), child_list.end());
    }
    if (fs.index) {
        TrainingConfig tcfg;
        tcfg.index_type = "g-index";
        tcfg.sample_ratio = 1.0f;
        fs.index->train(keys, values, tcfg);
    }
    active_fs_storage() = &fs;
}

void gpufs_seed_demo_tree(GPULearnedFS& fs) {
    std::vector<std::pair<std::string, std::uint64_t>> training_set = {
        {fs.mount_point + "/train", 2},
        {fs.mount_point + "/train/img", 3},
        {fs.mount_point + "/train/img/cat.jpg", 4},
        {fs.mount_point + "/train/img/dog.jpg", 5},
        {fs.mount_point + "/train/text", 6},
        {fs.mount_point + "/train/text/readme.txt", 7},
    };
    gpufs_init(fs, fs.index, fs.path_cfg, training_set);
}

int gpufs_getattr(const char* path, struct stat* stbuf, ::fuse_file_info*) {
    const auto start = now_ns();
    if (!path || !stbuf) {
        return -EINVAL;
    }
    GPULearnedFS* fs = active_fs_storage();
    if (!fs || !fs->index) {
        return -EIO;
    }
    std::lock_guard<std::mutex> lock(fs->global_lock);
    const std::string abs = join_mount_path(fs->mount_point, path);

    // Usability rule for prototype:
    // - if we have a node entry, treat it as authoritative (even if the learned index misses)
    auto it = fs->nodes.find(abs);
    if (it != fs->nodes.end()) {
        fill_stat_from_node(it->second, stbuf);
        it->second.atime = now_ts();
        perfetto_track_event("fuse.getattr", start, now_ns() - start);
        return 0;
    }

    auto key = encode_path(abs, fs->path_cfg);
    if (key.value == EncodedKey::INVALID_KEY) {
        return -EINVAL;
    }
    auto inode_vec = fs->index->batch_lookup({key.value});
    if (inode_vec.empty() || inode_vec[0] == INVALID_INODE) {
        return -ENOENT;
    }
    fill_stat_from_inode(inode_vec[0], stbuf);
    perfetto_track_event("fuse.getattr", start, now_ns() - start);
    return 0;
}

int gpufs_readdir(const char* path, void* buf, ::fuse_fill_dir_t filler, off_t, ::fuse_file_info*, enum ::fuse_readdir_flags) {
    const auto start = now_ns();
    GPULearnedFS* fs = active_fs_storage();
    if (!fs || !filler) {
        return -EIO;
    }
    std::lock_guard<std::mutex> lock(fs->global_lock);
    const std::string abs = join_mount_path(fs->mount_point, path);

    auto node_it = fs->nodes.find(abs);
    if (node_it == fs->nodes.end()) {
        return -ENOENT;
    }
    if (!node_it->second.is_dir) {
        return -ENOTDIR;
    }

    auto it = fs->children.find(abs);
    const std::vector<std::string>* kids = nullptr;
    if (it != fs->children.end()) {
        kids = &it->second;
    }
    struct stat dir_stub{};
    fill_stat_from_node(node_it->second, &dir_stub);
    filler(buf, ".", &dir_stub, 0, static_cast<fuse_fill_dir_flags>(0));
    filler(buf, "..", &dir_stub, 0, static_cast<fuse_fill_dir_flags>(0));
    if (kids) {
        for (const auto& name : *kids) {
            std::string child_path = abs == "/" ? "/" + name : abs + "/" + name;
            auto node = fs->nodes.find(child_path);
            struct stat st{};
            if (node != fs->nodes.end()) {
                fill_stat_from_node(node->second, &st);
            }
            filler(buf, name.c_str(), &st, 0, static_cast<fuse_fill_dir_flags>(0));
        }
    }
    node_it->second.atime = now_ts();
    perfetto_track_event("fuse.readdir", start, now_ns() - start);
    return 0;
}

int gpufs_open(const char* path, struct fuse_file_info*) {
    GPULearnedFS* fs = active_fs_storage();
    if (!fs) {
        return -EIO;
    }
    const std::string abs = join_mount_path(fs->mount_point, path);
    auto it = fs->nodes.find(abs);
    if (it == fs->nodes.end()) {
        return -ENOENT;
    }
    return it->second.is_dir ? -EISDIR : 0;
}

int gpufs_create(const char* path, mode_t mode, struct fuse_file_info* fi) {
    GPULearnedFS* fs = active_fs_storage();
    if (!fs || !path) {
        return -EIO;
    }
    std::lock_guard<std::mutex> lock(fs->global_lock);
    const std::string abs = join_mount_path(fs->mount_point, path);
    if (fs->nodes.find(abs) != fs->nodes.end()) {
        return -EEXIST;
    }
    const std::string parent = parent_of(abs);
    auto parent_it = fs->nodes.find(parent);
    if (parent_it == fs->nodes.end() || !parent_it->second.is_dir) {
        return -ENOENT;
    }
    const std::uint64_t inode = fs->next_inode++;
    GPULearnedFS::NodeEntry entry;
    entry.inode = inode;
    entry.is_dir = false;
    entry.mode = static_cast<mode_t>(S_IFREG | (mode & 0777));
    entry.uid = static_cast<uid_t>(::getuid());
    entry.gid = static_cast<gid_t>(::getgid());
    entry.size = 0;
    entry.atime = entry.mtime = entry.ctime = now_ts();
    entry.data.clear();
    fs->nodes[abs] = std::move(entry);
    fs->children[parent].push_back(basename_of(abs));
    std::sort(fs->children[parent].begin(), fs->children[parent].end());
    if (fi) {
        fi->fh = inode;
    }
    return 0;
}

int gpufs_unlink(const char* path) {
    GPULearnedFS* fs = active_fs_storage();
    if (!fs || !path) {
        return -EIO;
    }
    std::lock_guard<std::mutex> lock(fs->global_lock);
    const std::string abs = join_mount_path(fs->mount_point, path);
    auto it = fs->nodes.find(abs);
    if (it == fs->nodes.end() || it->second.is_dir) {
        return -ENOENT;
    }
    fs->nodes.erase(it);
    const auto parent = parent_of(abs);
    auto kids_it = fs->children.find(parent);
    if (kids_it != fs->children.end()) {
        auto& kids = kids_it->second;
        kids.erase(std::remove(kids.begin(), kids.end(), basename_of(abs)), kids.end());
    }
    return 0;
}

int gpufs_mkdir(const char* path, mode_t mode) {
    GPULearnedFS* fs = active_fs_storage();
    if (!fs || !path) {
        return -EIO;
    }
    std::lock_guard<std::mutex> lock(fs->global_lock);
    const std::string abs = join_mount_path(fs->mount_point, path);
    if (fs->nodes.find(abs) != fs->nodes.end()) {
        return -EEXIST;
    }
    const std::string parent = parent_of(abs);
    auto parent_it = fs->nodes.find(parent);
    if (parent_it == fs->nodes.end() || !parent_it->second.is_dir) {
        return -ENOENT;
    }
    const std::uint64_t inode = fs->next_inode++;
    GPULearnedFS::NodeEntry entry;
    entry.inode = inode;
    entry.is_dir = true;
    entry.mode = static_cast<mode_t>(S_IFDIR | (mode & 0777));
    entry.uid = static_cast<uid_t>(::getuid());
    entry.gid = static_cast<gid_t>(::getgid());
    entry.size = 0;
    entry.atime = entry.mtime = entry.ctime = now_ts();
    entry.data.clear();
    fs->nodes[abs] = std::move(entry);
    fs->children[parent].push_back(basename_of(abs));
    std::sort(fs->children[parent].begin(), fs->children[parent].end());
    fs->children[abs] = {};
    return 0;
}

int gpufs_rename(const char* from, const char* to, unsigned int flags) {
    GPULearnedFS* fs = active_fs_storage();
    if (!fs || !from || !to) {
        return -EIO;
    }
    if (flags != 0) {
        return -EINVAL;
    }
    std::lock_guard<std::mutex> lock(fs->global_lock);
    const std::string abs_from = join_mount_path(fs->mount_point, from);
    const std::string abs_to = join_mount_path(fs->mount_point, to);
    auto it = fs->nodes.find(abs_from);
    if (it == fs->nodes.end()) {
        return -ENOENT;
    }
    if (fs->nodes.find(abs_to) != fs->nodes.end()) {
        return -EEXIST;
    }

    const std::string from_parent = parent_of(abs_from);
    const std::string to_parent = parent_of(abs_to);
    auto to_parent_it = fs->nodes.find(to_parent);
    if (to_parent_it == fs->nodes.end() || !to_parent_it->second.is_dir) {
        return -ENOENT;
    }

    auto moved = std::move(it->second);
    fs->nodes.erase(it);
    moved.ctime = now_ts();
    fs->nodes[abs_to] = std::move(moved);

    // update parent child lists
    auto from_kids_it = fs->children.find(from_parent);
    if (from_kids_it != fs->children.end()) {
        auto& kids = from_kids_it->second;
        kids.erase(std::remove(kids.begin(), kids.end(), basename_of(abs_from)), kids.end());
    }
    fs->children[to_parent].push_back(basename_of(abs_to));
    std::sort(fs->children[to_parent].begin(), fs->children[to_parent].end());

    // directory children mapping key move (non-recursive)
    auto ch_it = fs->children.find(abs_from);
    if (ch_it != fs->children.end()) {
        fs->children[abs_to] = std::move(ch_it->second);
        fs->children.erase(ch_it);
    }

    return 0;
}

int gpufs_truncate(const char* path, off_t size, struct fuse_file_info*) {
    if (size < 0) {
        return -EINVAL;
    }
    GPULearnedFS* fs = active_fs_storage();
    if (!fs || !path) {
        return -EIO;
    }
    std::lock_guard<std::mutex> lock(fs->global_lock);
    const std::string abs = join_mount_path(fs->mount_point, path);
    auto it = fs->nodes.find(abs);
    if (it == fs->nodes.end()) {
        return -ENOENT;
    }
    if (it->second.is_dir) {
        return -EISDIR;
    }
    it->second.data.resize(static_cast<std::size_t>(size), 0);
    it->second.size = static_cast<std::uint64_t>(size);
    it->second.mtime = it->second.ctime = now_ts();
    return 0;
}

int gpufs_read(const char* path, char* buf, size_t size, off_t offset, struct fuse_file_info*) {
    if (offset < 0) {
        return -EINVAL;
    }
    GPULearnedFS* fs = active_fs_storage();
    if (!fs || !path || !buf) {
        return -EIO;
    }
    std::lock_guard<std::mutex> lock(fs->global_lock);
    const std::string abs = join_mount_path(fs->mount_point, path);
    auto it = fs->nodes.find(abs);
    if (it == fs->nodes.end()) {
        return -ENOENT;
    }
    if (it->second.is_dir) {
        return -EISDIR;
    }
    const std::size_t off = static_cast<std::size_t>(offset);
    if (off >= it->second.data.size()) {
        it->second.atime = now_ts();
        return 0;
    }
    const std::size_t n = std::min<std::size_t>(size, it->second.data.size() - off);
    std::memcpy(buf, it->second.data.data() + off, n);
    it->second.atime = now_ts();
    return static_cast<int>(n);
}

int gpufs_write(const char* path, const char* buf, size_t size, off_t offset, struct fuse_file_info*) {
    if (offset < 0) {
        return -EINVAL;
    }
    GPULearnedFS* fs = active_fs_storage();
    if (!fs || !path || !buf) {
        return -EIO;
    }
    std::lock_guard<std::mutex> lock(fs->global_lock);
    const std::string abs = join_mount_path(fs->mount_point, path);
    auto it = fs->nodes.find(abs);
    if (it == fs->nodes.end()) {
        return -ENOENT;
    }
    if (it->second.is_dir) {
        return -EISDIR;
    }
    const std::size_t off = static_cast<std::size_t>(offset);
    const std::size_t end = off + size;
    if (end > it->second.data.size()) {
        it->second.data.resize(end, 0);
    }
    std::memcpy(it->second.data.data() + off, buf, size);
    it->second.size = static_cast<std::uint64_t>(it->second.data.size());
    it->second.mtime = it->second.ctime = now_ts();
    return static_cast<int>(size);
}

int gpufs_utimens(const char* path, const struct timespec tv[2], struct fuse_file_info*) {
    GPULearnedFS* fs = active_fs_storage();
    if (!fs || !path || !tv) {
        return -EIO;
    }
    std::lock_guard<std::mutex> lock(fs->global_lock);
    const std::string abs = join_mount_path(fs->mount_point, path);
    auto it = fs->nodes.find(abs);
    if (it == fs->nodes.end()) {
        return -ENOENT;
    }
    it->second.atime = tv[0];
    it->second.mtime = tv[1];
    it->second.ctime = now_ts();
    return 0;
}

int gpufs_rmdir(const char* path) {
    GPULearnedFS* fs = active_fs_storage();
    if (!fs || !path) {
        return -EIO;
    }
    std::lock_guard<std::mutex> lock(fs->global_lock);
    const std::string abs = join_mount_path(fs->mount_point, path);
    auto it = fs->nodes.find(abs);
    if (it == fs->nodes.end() || !it->second.is_dir) {
        return -ENOENT;
    }
    auto child_it = fs->children.find(abs);
    if (child_it != fs->children.end() && !child_it->second.empty()) {
        return -ENOTEMPTY;
    }
    fs->nodes.erase(it);
    fs->children.erase(abs);
    const auto parent = parent_of(abs);
    auto kids_it = fs->children.find(parent);
    if (kids_it != fs->children.end()) {
        auto& kids = kids_it->second;
        kids.erase(std::remove(kids.begin(), kids.end(), basename_of(abs)), kids.end());
    }
    return 0;
}

namespace {

static int fuse_getattr_thunk(const char* path, struct stat* stbuf, struct fuse_file_info* fi) {
    return gpufs_getattr(path, stbuf, fi);
}

static int fuse_readdir_thunk(const char* path, void* buf, fuse_fill_dir_t filler, off_t offset, struct fuse_file_info* fi, enum fuse_readdir_flags flags) {
    return gpufs_readdir(path, buf, filler, offset, fi, flags);
}

static int fuse_open_thunk(const char* path, struct fuse_file_info* fi) {
    return gpufs_open(path, fi);
}

static int fuse_create_thunk(const char* path, mode_t mode, struct fuse_file_info* fi) {
    return gpufs_create(path, mode, fi);
}

}  // namespace

struct fuse_operations build_fuse_operations() {
    struct fuse_operations ops{};
    ops.getattr = fuse_getattr_thunk;
    ops.readdir = fuse_readdir_thunk;
    ops.open = fuse_open_thunk;
    ops.create = fuse_create_thunk;
    ops.unlink = [](const char* path) -> int { return gpufs_unlink(path); };
    ops.mkdir = [](const char* path, mode_t mode) -> int { return gpufs_mkdir(path, mode); };
    ops.rmdir = [](const char* path) -> int { return gpufs_rmdir(path); };
    ops.rename = [](const char* from, const char* to, unsigned int flags) -> int { return gpufs_rename(from, to, flags); };
    ops.truncate = [](const char* path, off_t size, struct fuse_file_info* fi) -> int { return gpufs_truncate(path, size, fi); };
    ops.read = [](const char* path, char* buf, size_t size, off_t off, struct fuse_file_info* fi) -> int { return gpufs_read(path, buf, size, off, fi); };
    ops.write = [](const char* path, const char* buf, size_t size, off_t off, struct fuse_file_info* fi) -> int { return gpufs_write(path, buf, size, off, fi); };
    ops.utimens = [](const char* path, const struct timespec tv[2], struct fuse_file_info* fi) -> int { return gpufs_utimens(path, tv, fi); };
    return ops;
}

}  // namespace glfs
