#include "fuse/backing_root_proxy.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <system_error>

namespace glfs {

namespace fs = std::filesystem;

BackingRootProxy::BackingRootProxy(std::string root) : root_(std::move(root)) {}

const std::string& BackingRootProxy::root() const {
    return root_;
}

void BackingRootProxy::set_root(std::string root) {
    root_ = std::filesystem::absolute(std::filesystem::path(std::move(root))).lexically_normal().string();
}

void BackingRootProxy::set_mount_point(std::string mount_point) {
    mount_point_ = std::move(mount_point);
}

void BackingRootProxy::set_mount_root(std::string mount_root) {
    mount_root_ = std::move(mount_root);
}

std::string BackingRootProxy::resolve(const std::string& absolute_path) const {
    fs::path rel = absolute_path;
    if (rel.is_absolute()) {
        if (!mount_root_.empty() && absolute_path == mount_root_) {
            return fs::path(root_).lexically_normal().string();
        }
        if (!mount_root_.empty() && absolute_path.rfind(mount_root_ + "/", 0) == 0) {
            rel = fs::path(absolute_path).lexically_relative(mount_root_);
        } else if (!mount_point_.empty() && absolute_path.rfind(mount_point_, 0) == 0) {
            if (absolute_path == mount_point_) {
                return fs::path(root_).lexically_normal().string();
            }
            if (absolute_path.rfind(mount_point_ + "/", 0) != 0) {
                rel = rel.relative_path();
            } else {
                rel = fs::path(absolute_path).lexically_relative(mount_point_);
            }
        } else {
            rel = rel.relative_path();
        }
    }
    if (rel == ".") {
        return fs::path(root_).lexically_normal().string();
    }
    return (fs::path(root_) / rel).lexically_normal().string();
}

int BackingRootProxy::ensure_root() const {
    std::error_code ec;
    fs::create_directories(root_, ec);
    return ec ? -ec.value() : 0;
}

int BackingRootProxy::getattr(const std::string& absolute_path, struct stat* stbuf) const {
    if (!stbuf) {
        return -EINVAL;
    }
    std::error_code ec;
    const auto resolved = resolve(absolute_path);
    std::cerr << "[backing_root] getattr " << absolute_path << " -> " << resolved << '\n';
    if (!fs::exists(resolved, ec)) {
        return -ENOENT;
    }
    if (auto status = fs::status(resolved, ec); ec) {
        return -ec.value();
    } else {
        std::memset(stbuf, 0, sizeof(*stbuf));
        stbuf->st_size = static_cast<off_t>(fs::is_directory(status) ? 4096 : fs::file_size(resolved, ec));
        stbuf->st_mode = fs::is_directory(status) ? (S_IFDIR | 0755) : (S_IFREG | 0644);
        stbuf->st_nlink = fs::is_directory(status) ? 2 : 1;
        stbuf->st_blksize = 4096;
        stbuf->st_blocks = (stbuf->st_size + 511) / 512;
        return 0;
    }
}

int BackingRootProxy::listdir(const std::string& absolute_path, std::vector<std::string>& entries) const {
    entries.clear();
    std::error_code ec;
    const auto resolved = resolve(absolute_path);
    std::cerr << "[backing_root] listdir " << absolute_path << " -> " << resolved << '\n';
    if (!fs::exists(resolved, ec)) {
        return -ENOENT;
    }
    if (!fs::is_directory(resolved, ec)) {
        return -ENOTDIR;
    }
    for (const auto& entry : fs::directory_iterator(resolved, ec)) {
        if (ec) {
            return -ec.value();
        }
        entries.push_back(entry.path().filename().string());
    }
    return 0;
}

int BackingRootProxy::mkdir(const std::string& absolute_path, mode_t) const {
    std::error_code ec;
    const auto resolved = resolve(absolute_path);
    std::cerr << "[backing_root] mkdir " << absolute_path << " -> " << resolved << '\n';
    fs::create_directories(resolved, ec);
    return ec ? -ec.value() : 0;
}

int BackingRootProxy::rmdir(const std::string& absolute_path) const {
    std::error_code ec;
    const auto resolved = resolve(absolute_path);
    std::cerr << "[backing_root] rmdir " << absolute_path << " -> " << resolved << '\n';
    auto removed = fs::remove(resolved, ec);
    if (ec) {
        return -ec.value();
    }
    return removed ? 0 : -ENOENT;
}

int BackingRootProxy::create(const std::string& absolute_path, mode_t) const {
    std::error_code ec;
    auto resolved = resolve(absolute_path);
    std::cerr << "[backing_root] create " << absolute_path << " -> " << resolved << '\n';
    fs::create_directories(fs::path(resolved).parent_path(), ec);
    if (ec) {
        return -ec.value();
    }
    std::ofstream out(resolved, std::ios::binary | std::ios::app);
    if (!out) {
        return -errno;
    }
    out.flush();
    return 0;
}

int BackingRootProxy::unlink(const std::string& absolute_path) const {
    std::error_code ec;
    const auto resolved = resolve(absolute_path);
    std::cerr << "[backing_root] unlink " << absolute_path << " -> " << resolved << '\n';
    auto removed = fs::remove(resolved, ec);
    if (ec) {
        return -ec.value();
    }
    return removed ? 0 : -ENOENT;
}

int BackingRootProxy::rename(const std::string& from, const std::string& to) const {
    std::error_code ec;
    const auto resolved_from = resolve(from);
    const auto resolved_to = resolve(to);
    std::cerr << "[backing_root] rename " << from << " -> " << to << " => " << resolved_from << " -> " << resolved_to << '\n';
    fs::create_directories(fs::path(resolved_to).parent_path(), ec);
    if (ec) {
        return -ec.value();
    }
    fs::rename(resolved_from, resolved_to, ec);
    return ec ? -ec.value() : 0;
}

int BackingRootProxy::truncate(const std::string& absolute_path, off_t size) const {
    std::error_code ec;
    const auto resolved = resolve(absolute_path);
    std::cerr << "[backing_root] truncate " << absolute_path << " -> " << resolved << " size=" << size << '\n';
    fs::resize_file(resolved, static_cast<std::uintmax_t>(size), ec);
    return ec ? -ec.value() : 0;
}

int BackingRootProxy::read(const std::string& absolute_path, char* buf, size_t size, off_t offset) const {
    if (!buf) {
        return -EINVAL;
    }
    const auto resolved = resolve(absolute_path);
    std::cerr << "[backing_root] read " << absolute_path << " -> " << resolved << " size=" << size << " offset=" << offset << '\n';
    std::ifstream in(resolved, std::ios::binary);
    if (!in) {
        return -ENOENT;
    }
    in.seekg(offset, std::ios::beg);
    if (!in) {
        return -EINVAL;
    }
    in.read(buf, static_cast<std::streamsize>(size));
    return static_cast<int>(in.gcount());
}

int BackingRootProxy::write(const std::string& absolute_path, const char* buf, size_t size, off_t offset) const {
    if (!buf) {
        return -EINVAL;
    }
    auto resolved = resolve(absolute_path);
    std::cerr << "[backing_root] write " << absolute_path << " -> " << resolved << " size=" << size << " offset=" << offset << '\n';
    std::fstream file(resolved, std::ios::in | std::ios::out | std::ios::binary);
    if (!file) {
        std::ofstream create(resolved, std::ios::binary);
        if (!create) {
            return -errno;
        }
        create.close();
        file.open(resolved, std::ios::in | std::ios::out | std::ios::binary);
        if (!file) {
            return -errno;
        }
    }
    file.seekp(offset, std::ios::beg);
    if (!file) {
        return -EINVAL;
    }
    file.write(buf, static_cast<std::streamsize>(size));
    if (!file) {
        return -errno;
    }
    file.flush();
    return static_cast<int>(size);
}

}  // namespace glfs
