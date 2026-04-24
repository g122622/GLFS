#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <sys/stat.h>

namespace glfs {

class BackingRootProxy {
public:
    explicit BackingRootProxy(std::string root = "./backing_root");

    const std::string& root() const;
    void set_root(std::string root);

    std::string resolve(const std::string& absolute_path) const;
    int ensure_root() const;

    int getattr(const std::string& absolute_path, struct stat* stbuf) const;
    int listdir(const std::string& absolute_path, std::vector<std::string>& entries) const;
    int mkdir(const std::string& absolute_path, mode_t mode) const;
    int rmdir(const std::string& absolute_path) const;
    int create(const std::string& absolute_path, mode_t mode) const;
    int unlink(const std::string& absolute_path) const;
    int rename(const std::string& from, const std::string& to) const;
    int truncate(const std::string& absolute_path, off_t size) const;
    int read(const std::string& absolute_path, char* buf, size_t size, off_t offset) const;
    int write(const std::string& absolute_path, const char* buf, size_t size, off_t offset) const;

private:
    std::string root_;
};

}  // namespace glfs
