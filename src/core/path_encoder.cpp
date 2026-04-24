#include "core/path_encoder.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace glfs {

namespace {

std::uint64_t fnv1a64(const std::string& text) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (unsigned char c : text) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::vector<std::string> split_path(const std::string& path) {
    std::vector<std::string> parts;
    std::string current;
    for (char c : path) {
        if (c == '/') {
            if (!current.empty()) {
                parts.push_back(current);
                current.clear();
            }
            continue;
        }
        current.push_back(c);
    }
    if (!current.empty()) {
        parts.push_back(current);
    }
    return parts;
}

bool is_valid_component(const std::string& part) {
    return part.find('\0') == std::string::npos && part.find('/') == std::string::npos;
}

}  // namespace

std::string normalize_path(const std::string& path) {
    if (path.empty()) {
        return "/";
    }
    std::vector<std::string> stack;
    const bool absolute = path.front() == '/';
    for (const auto& part : split_path(path)) {
        if (part == ".") {
            continue;
        }
        if (part == "..") {
            if (!stack.empty()) {
                stack.pop_back();
            }
            continue;
        }
        if (!is_valid_component(part)) {
            throw std::invalid_argument("invalid path component");
        }
        stack.push_back(part);
    }
    std::ostringstream out;
    if (absolute) {
        out << '/';
    }
    for (std::size_t i = 0; i < stack.size(); ++i) {
        out << stack[i];
        if (i + 1 != stack.size()) {
            out << '/';
        }
    }
    std::string normalized = out.str();
    if (normalized.empty()) {
        return "/";
    }
    return normalized;
}

EncodedKey encode_path(const std::string& path, const PathConfig& cfg) {
    if (cfg.max_depth == 0 || cfg.bits_per_level == 0 || cfg.bits_per_level > 16) {
        throw std::invalid_argument("invalid PathConfig");
    }
    const std::string normalized = normalize_path(path);
    if (normalized.rfind(cfg.mount_point, 0) != 0) {
        throw std::invalid_argument("path must start with mount_point");
    }

    std::string relative = normalized.substr(cfg.mount_point.size());
    if (!relative.empty() && relative.front() == '/') {
        relative.erase(relative.begin());
    }
    const auto parts = split_path(relative);
    if (parts.size() > cfg.max_depth) {
        return {EncodedKey::INVALID_KEY, 0};
    }

    const std::uint64_t mask = (cfg.bits_per_level == 64) ? ~0ULL : ((1ULL << cfg.bits_per_level) - 1ULL);
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        const std::uint64_t component_hash = fnv1a64(parts[i]) & mask;
        const std::uint32_t shift = cfg.bits_per_level * static_cast<std::uint32_t>(cfg.max_depth - 1 - i);
        if (shift >= 64) {
            continue;
        }
        value |= (component_hash << shift);
    }
    value &= ~0x3FULL;
    value |= static_cast<std::uint64_t>(parts.size() & 0x3FULL);
    if (value == 0) {
        value = 1;
    }
    return {value, static_cast<std::uint32_t>(parts.size())};
}

}  // namespace glfs
