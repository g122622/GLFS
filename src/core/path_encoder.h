#pragma once

#include <cstdint>
#include <string>

namespace glfs {

struct EncodedKey {
    std::uint64_t value = 0;
    std::uint32_t depth = 0;
    static constexpr std::uint64_t INVALID_KEY = 0;
};

struct PathConfig {
    std::string mount_point = "/home/user/data";
    std::uint32_t max_depth = 32;
    std::uint32_t bits_per_level = 8;
};

std::string normalize_path(const std::string& path);
EncodedKey encode_path(const std::string& path, const PathConfig& cfg);

}  // namespace glfs
