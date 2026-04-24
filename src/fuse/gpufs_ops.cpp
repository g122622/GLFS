#include "fuse/gpufs_ops.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iostream>
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

void insert_cached_entry(GPULearnedFS& fs,
                         const std::string& abs_mount_path,
                         std::uint64_t inode,
                         bool is_dir,
                         mode_t mode,
                         uid_t uid,
                         gid_t gid,
                         std::uint64_t size) {
    GPULearnedFS::NodeEntry entry;
    entry.inode = inode;
    entry.is_dir = is_dir;
    entry.mode = mode;
    entry.uid = uid;
    entry.gid = gid;
    entry.size = size;
    entry.atime = entry.mtime = entry.ctime = now_ts();
    fs.nodes[abs_mount_path] = std::move(entry);

    const auto parent = parent_of(abs_mount_path);
    if (abs_mount_path != fs.mount_point) {
        fs.children[parent].push_back(basename_of(abs_mount_path));
        if (is_dir && fs.children.find(abs_mount_path) == fs.children.end()) {
            fs.children[abs_mount_path] = {};
        }
    }
}

void load_cache_from_backing_root(GPULearnedFS& fs, std::vector<std::uint64_t>& keys, std::vector<std::uint64_t>& values) {
    namespace fsys = std::filesystem;
    const fsys::path root_path = fs.backing_root.root();
    std::error_code ec;

    std::uint64_t next_inode = 1;
    insert_cached_entry(fs, fs.mount_point, next_inode++, true, static_cast<mode_t>(S_IFDIR | 0755), static_cast<uid_t>(::getuid()), static_cast<gid_t>(::getgid()), 0);
    if (auto root_key = encode_path(fs.mount_point, fs.path_cfg); root_key.value != EncodedKey::INVALID_KEY) {
        keys.push_back(root_key.value);
        values.push_back(1);
    }

    if (!fsys::exists(root_path, ec)) {
        return;
    }

    for (fsys::recursive_directory_iterator it(root_path, ec), end; it != end && !ec; it.increment(ec)) {
        const auto rel = it->path().lexically_relative(root_path);
        if (rel.empty()) {
            continue;
        }
        const std::string abs_mount_path = normalize_path(fs.mount_point + "/" + rel.generic_string());
        const auto status = it->symlink_status(ec);
        if (ec) {
            break;
        }
        const bool is_dir = fsys::is_directory(status);
        const mode_t mode = static_cast<mode_t>(is_dir ? (S_IFDIR | 0755) : (S_IFREG | 0644));
        std::uint64_t size = 0;
        if (!is_dir) {
            size = static_cast<std::uint64_t>(fsys::file_size(it->path(), ec));
            if (ec) {
                size = 0;
                ec.clear();
            }
        }
        insert_cached_entry(fs, abs_mount_path, next_inode++, is_dir, mode, static_cast<uid_t>(::getuid()), static_cast<gid_t>(::getgid()), size);
        if (auto key = encode_path(abs_mount_path, fs.path_cfg); key.value != EncodedKey::INVALID_KEY) {
            keys.push_back(key.value);
            values.push_back(next_inode - 1);
        }
    }

    for (auto& [_, child_list] : fs.children) {
        std::sort(child_list.begin(), child_list.end());
        child_list.erase(std::unique(child_list.begin(), child_list.end()), child_list.end());
    }
    fs.children[fs.mount_point];
}

void refresh_cached_path(GPULearnedFS& fs, const std::string& abs_path) {
    if (!fs.usable_mode) {
        return;
    }
    std::vector<std::string> entries;
    struct stat st{};
    if (fs.backing_root.getattr(abs_path, &st) != 0) {
        fs.nodes.erase(abs_path);
        return;
    }
    auto it = fs.nodes.find(abs_path);
    if (it == fs.nodes.end()) {
        GPULearnedFS::NodeEntry entry;
        entry.inode = fs.next_inode++;
        entry.is_dir = S_ISDIR(st.st_mode);
        entry.mode = st.st_mode;
        entry.uid = st.st_uid;
        entry.gid = st.st_gid;
        entry.size = static_cast<std::uint64_t>(st.st_size);
        entry.atime = st.st_atim;
        entry.mtime = st.st_mtim;
        entry.ctime = st.st_ctim;
        fs.nodes[abs_path] = std::move(entry);
    } else {
        it->second.is_dir = S_ISDIR(st.st_mode);
        it->second.mode = st.st_mode;
        it->second.uid = st.st_uid;
        it->second.gid = st.st_gid;
        it->second.size = static_cast<std::uint64_t>(st.st_size);
        it->second.atime = st.st_atim;
        it->second.mtime = st.st_mtim;
        it->second.ctime = st.st_ctim;
    }
    const auto parent = parent_of(abs_path);
    if (!parent.empty()) {
        auto& kids = fs.children[parent];
        const auto name = basename_of(abs_path);
        if (std::find(kids.begin(), kids.end(), name) == kids.end() && abs_path != fs.mount_point) {
            kids.push_back(name);
            std::sort(kids.begin(), kids.end());
        }
    }
}

int apply_fallback_getattr(GPULearnedFS& fs, const std::string& abs, struct stat* stbuf) {
    if (!fs.usable_mode) {
        return -ENOENT;
    }
    return fs.backing_root.getattr(abs, stbuf);
}

int apply_fallback_readdir(GPULearnedFS& fs, const std::string& abs, void* buf, fuse_fill_dir_t filler) {
    if (!fs.usable_mode) {
        return -ENOENT;
    }
    std::vector<std::string> entries;
    const auto rc = fs.backing_root.listdir(abs, entries);
    if (rc != 0) {
        return rc;
    }
    struct stat dir_stub{};
    if (fs.backing_root.getattr(abs, &dir_stub) != 0) {
        return -ENOENT;
    }
    filler(buf, ".", &dir_stub, 0, static_cast<fuse_fill_dir_flags>(0));
    filler(buf, "..", &dir_stub, 0, static_cast<fuse_fill_dir_flags>(0));
    for (const auto& name : entries) {
        struct stat st{};
        const auto child = abs == "/" ? "/" + name : abs + "/" + name;
        fs.backing_root.getattr(child, &st);
        filler(buf, name.c_str(), &st, 0, static_cast<fuse_fill_dir_flags>(0));
    }
    return 0;
}

}  // namespace

GPULearnedFS* active_fs() {
    return active_fs_storage();
}

void set_active_fs(GPULearnedFS* fs) {
    active_fs_storage() = fs;
}

void gpufs_init(GPULearnedFS& fs,
                IGPUControlPlane* control_plane,
                const FSConfig& cfg) {
    fs.control_plane = control_plane;
    fs.path_cfg.mount_point = cfg.fs.mount_point;
    fs.mount_point = cfg.fs.mount_point;
    fs.backing_root.set_root(cfg.fs.backing_root);
    fs.backing_root.set_mount_point(cfg.fs.mount_point);
    fs.backing_root.set_mount_root(normalize_path(cfg.fs.mount_point));
    fs.strict_mode = cfg.fs.strict_mode;
    fs.usable_mode = !cfg.fs.strict_mode;
    fs.nodes.clear();
    fs.children.clear();
    fs.next_inode = 2;

    if (fs.backing_root.ensure_root() != 0) {
        // leave usable mode on but allow the mount to continue; fallback will surface errors
    }
    if (fs.verbose) {
        std::cerr << "[gpufs] init mount=" << fs.mount_point
                  << " backing_root=" << fs.backing_root.root()
                  << " mount_root=" << normalize_path(cfg.fs.mount_point)
                  << '\n';
    }

    std::vector<std::uint64_t> keys;
    std::vector<std::uint64_t> values;
    load_cache_from_backing_root(fs, keys, values);

    if (fs.control_plane) {
        TrainingConfig tcfg;
        tcfg.index_type = "g-index";
        tcfg.sample_ratio = 1.0f;
        fs.control_plane->initialize(tcfg.index_type);
        if (!keys.empty() && keys.size() == values.size()) {
            fs.control_plane->train(keys, values, tcfg);
        }
    }
    active_fs_storage() = &fs;
}

int gpufs_getattr(const char* path, struct stat* stbuf, ::fuse_file_info*) {
    const auto start = now_ns();
    if (!path || !stbuf) {
        return -EINVAL;
    }
    GPULearnedFS* fs = active_fs_storage();
    if (!fs) {
        return -EIO;
    }
    std::lock_guard<std::mutex> lock(fs->global_lock);
    const std::string abs = join_mount_path(fs->mount_point, path);
    if (fs->verbose) {
        std::cerr << "[gpufs] getattr " << abs << '\n';
    }

    if (fs->usable_mode) {
        const auto rc = fs->backing_root.getattr(abs, stbuf);
        if (rc == 0) {
            perfetto_track_event("fuse.getattr", start, now_ns() - start);
            return 0;
        }
        return rc;
    }

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
    if (fs->control_plane) {
        const auto result = fs->control_plane->lookup(key.value);
        if (result.inode != INVALID_INODE) {
            fill_stat_from_inode(result.inode, stbuf);
            perfetto_track_event("fuse.getattr", start, now_ns() - start);
            return 0;
        }
        if (!result.fallback_to_backing_root && fs->strict_mode) {
            return -ENOENT;
        }
    }
    const auto rc = apply_fallback_getattr(*fs, abs, stbuf);
    if (rc == 0) {
        perfetto_track_event("fuse.getattr", start, now_ns() - start);
    }
    return rc;
}

int gpufs_readdir(const char* path, void* buf, ::fuse_fill_dir_t filler, off_t, ::fuse_file_info*, enum ::fuse_readdir_flags) {
    const auto start = now_ns();
    GPULearnedFS* fs = active_fs_storage();
    if (!fs || !filler) {
        return -EIO;
    }
    std::lock_guard<std::mutex> lock(fs->global_lock);
    const std::string abs = join_mount_path(fs->mount_point, path);
    if (fs->verbose) {
        std::cerr << "[gpufs] readdir " << abs << '\n';
    }

    auto node_it = fs->nodes.find(abs);
    if (fs->usable_mode) {
        const auto rc = apply_fallback_readdir(*fs, abs, buf, filler);
        if (rc == 0) {
            perfetto_track_event("fuse.readdir", start, now_ns() - start);
        }
        return rc;
    }
    if (node_it == fs->nodes.end()) {
        const auto rc = apply_fallback_readdir(*fs, abs, buf, filler);
        if (rc == 0) {
            perfetto_track_event("fuse.readdir", start, now_ns() - start);
        }
        return rc;
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
    if (kids && !kids->empty()) {
        for (const auto& name : *kids) {
            std::string child_path = abs == "/" ? "/" + name : abs + "/" + name;
            auto node = fs->nodes.find(child_path);
            struct stat st{};
            if (node != fs->nodes.end()) {
                fill_stat_from_node(node->second, &st);
            }
            filler(buf, name.c_str(), &st, 0, static_cast<fuse_fill_dir_flags>(0));
        }
    } else if (fs->usable_mode) {
        std::vector<std::string> entries;
        if (fs->backing_root.listdir(abs, entries) == 0) {
            for (const auto& name : entries) {
                struct stat st{};
                const auto child = abs == "/" ? "/" + name : abs + "/" + name;
                fs->backing_root.getattr(child, &st);
                filler(buf, name.c_str(), &st, 0, static_cast<fuse_fill_dir_flags>(0));
            }
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
    if (fs->verbose) {
        std::cerr << "[gpufs] open " << abs << '\n';
    }
    if (fs->usable_mode) {
        struct stat st{};
        if (fs->backing_root.getattr(abs, &st) != 0) {
            return -ENOENT;
        }
        return S_ISDIR(st.st_mode) ? -EISDIR : 0;
    }
    auto it = fs->nodes.find(abs);
    if (it == fs->nodes.end()) {
        if (!fs->usable_mode) {
            return -ENOENT;
        }
        struct stat st{};
        if (fs->backing_root.getattr(abs, &st) != 0) {
            return -ENOENT;
        }
        return S_ISDIR(st.st_mode) ? -EISDIR : 0;
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
    if (fs->verbose) {
        std::cerr << "[gpufs] create " << abs << " mode=" << std::oct << mode << std::dec << '\n';
    }
    if (fs->usable_mode) {
        const auto rc = fs->backing_root.create(abs, mode);
        if (rc < 0) {
            return rc;
        }
        refresh_cached_path(*fs, abs);
        return 0;
    }
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
    if (fs->usable_mode) {
        fs->backing_root.create(abs, mode);
        refresh_cached_path(*fs, abs);
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
    if (fs->verbose) {
        std::cerr << "[gpufs] unlink " << abs << '\n';
    }
    if (fs->usable_mode) {
        const auto rc = fs->backing_root.unlink(abs);
        if (rc < 0) {
            return rc;
        }
        fs->nodes.erase(abs);
        const auto parent = parent_of(abs);
        auto kids_it = fs->children.find(parent);
        if (kids_it != fs->children.end()) {
            auto& kids = kids_it->second;
            kids.erase(std::remove(kids.begin(), kids.end(), basename_of(abs)), kids.end());
        }
        fs->children.erase(abs);
        return 0;
    }
    auto it = fs->nodes.find(abs);
    if (it == fs->nodes.end() || it->second.is_dir) {
        if (!fs->usable_mode) {
            return -ENOENT;
        }
        return fs->backing_root.unlink(abs);
    }
    fs->nodes.erase(it);
    const auto parent = parent_of(abs);
    auto kids_it = fs->children.find(parent);
    if (kids_it != fs->children.end()) {
        auto& kids = kids_it->second;
        kids.erase(std::remove(kids.begin(), kids.end(), basename_of(abs)), kids.end());
    }
    if (fs->usable_mode) {
        fs->backing_root.unlink(abs);
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
    if (fs->verbose) {
        std::cerr << "[gpufs] mkdir " << abs << " mode=" << std::oct << mode << std::dec << '\n';
    }
    if (fs->usable_mode) {
        const auto rc = fs->backing_root.mkdir(abs, mode);
        if (rc < 0) {
            return rc;
        }
        refresh_cached_path(*fs, abs);
        return 0;
    }
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
    if (fs->usable_mode) {
        fs->backing_root.mkdir(abs, mode);
        refresh_cached_path(*fs, abs);
    }
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
    if (fs->verbose) {
        std::cerr << "[gpufs] rename " << abs_from << " -> " << abs_to << '\n';
    }
    if (fs->usable_mode) {
        const auto rc = fs->backing_root.rename(abs_from, abs_to);
        if (rc == 0) {
            fs->nodes.erase(abs_from);
            fs->children.erase(abs_from);
            refresh_cached_path(*fs, abs_to);
        }
        return rc;
    }
    auto it = fs->nodes.find(abs_from);
    if (it == fs->nodes.end()) {
        if (!fs->usable_mode) {
            return -ENOENT;
        }
        return fs->backing_root.rename(abs_from, abs_to);
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

    auto from_kids_it = fs->children.find(from_parent);
    if (from_kids_it != fs->children.end()) {
        auto& kids = from_kids_it->second;
        kids.erase(std::remove(kids.begin(), kids.end(), basename_of(abs_from)), kids.end());
    }
    fs->children[to_parent].push_back(basename_of(abs_to));
    std::sort(fs->children[to_parent].begin(), fs->children[to_parent].end());

    auto ch_it = fs->children.find(abs_from);
    if (ch_it != fs->children.end()) {
        fs->children[abs_to] = std::move(ch_it->second);
        fs->children.erase(ch_it);
    }

    if (fs->usable_mode) {
        const auto rc = fs->backing_root.rename(abs_from, abs_to);
        if (rc == 0) {
            refresh_cached_path(*fs, abs_to);
        }
        return rc;
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
    if (fs->verbose) {
        std::cerr << "[gpufs] truncate " << abs << " size=" << size << '\n';
    }
    if (fs->usable_mode) {
        const auto rc = fs->backing_root.truncate(abs, size);
        if (rc == 0) {
            refresh_cached_path(*fs, abs);
        }
        return rc;
    }
    auto it = fs->nodes.find(abs);
    if (it == fs->nodes.end()) {
        if (!fs->usable_mode) {
            return -ENOENT;
        }
        return fs->backing_root.truncate(abs, size);
    }
    if (it->second.is_dir) {
        return -EISDIR;
    }
    it->second.data.resize(static_cast<std::size_t>(size), 0);
    it->second.size = static_cast<std::uint64_t>(size);
    it->second.mtime = it->second.ctime = now_ts();
    if (fs->usable_mode) {
        const auto rc = fs->backing_root.truncate(abs, size);
        if (rc == 0) {
            refresh_cached_path(*fs, abs);
        }
        return rc;
    }
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
    if (fs->verbose) {
        std::cerr << "[gpufs] read " << abs << " size=" << size << " offset=" << offset << '\n';
    }
    if (fs->usable_mode) {
        return fs->backing_root.read(abs, buf, size, offset);
    }
    auto it = fs->nodes.find(abs);
    if (it == fs->nodes.end()) {
        if (!fs->usable_mode) {
            return -ENOENT;
        }
        return fs->backing_root.read(abs, buf, size, offset);
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
    if (fs->usable_mode) {
        refresh_cached_path(*fs, abs);
    }
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
    if (fs->verbose) {
        std::cerr << "[gpufs] write " << abs << " size=" << size << " offset=" << offset << '\n';
    }
    if (fs->usable_mode) {
        const auto rc = fs->backing_root.write(abs, buf, size, offset);
        if (rc >= 0) {
            refresh_cached_path(*fs, abs);
        }
        return rc;
    }
    auto it = fs->nodes.find(abs);
    if (it == fs->nodes.end()) {
        if (!fs->usable_mode) {
            return -ENOENT;
        }
        return fs->backing_root.write(abs, buf, size, offset);
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
    if (fs->usable_mode) {
        const auto rc = fs->backing_root.write(abs, buf, size, offset);
        if (rc >= 0) {
            refresh_cached_path(*fs, abs);
        }
        return rc;
    }
    return static_cast<int>(size);
}

int gpufs_utimens(const char* path, const struct timespec tv[2], struct fuse_file_info*) {
    GPULearnedFS* fs = active_fs_storage();
    if (!fs || !path || !tv) {
        return -EIO;
    }
    std::lock_guard<std::mutex> lock(fs->global_lock);
    const std::string abs = join_mount_path(fs->mount_point, path);
    if (fs->verbose) {
        std::cerr << "[gpufs] utimens " << abs << '\n';
    }
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
    if (fs->verbose) {
        std::cerr << "[gpufs] rmdir " << abs << '\n';
    }
    if (fs->usable_mode) {
        const auto rc = fs->backing_root.rmdir(abs);
        if (rc == 0) {
            fs->nodes.erase(abs);
            fs->children.erase(abs);
        }
        return rc;
    }
    auto it = fs->nodes.find(abs);
    if (it == fs->nodes.end() || !it->second.is_dir) {
        if (!fs->usable_mode) {
            return -ENOENT;
        }
        return fs->backing_root.rmdir(abs);
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
    if (fs->usable_mode) {
        fs->backing_root.rmdir(abs);
    }
    return 0;
}

namespace {

static void* fuse_init_thunk(struct fuse_conn_info*, struct fuse_config* cfg) {
    if (cfg) {
        cfg->attr_timeout = 0.0;
        cfg->entry_timeout = 0.0;
        cfg->negative_timeout = 0.0;
        cfg->kernel_cache = 0;
        cfg->auto_cache = 0;
    }
    return nullptr;
}

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
    ops.init = fuse_init_thunk;
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
