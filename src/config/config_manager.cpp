#include "config/config_manager.h"

#include <fstream>
#include <sstream>

#include "utils/json_parser.h"

namespace glfs {

namespace {

std::string read_file(const std::string& filepath) {
    std::ifstream in(filepath);
    if (!in) {
        throw std::runtime_error("unable to open config file: " + filepath);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string get_string_or(const JsonValue& obj, const std::string& key, const std::string& fallback) {
    return obj.contains(key) ? obj.at(key).as_string() : fallback;
}

std::uint32_t get_u32_or(const JsonValue& obj, const std::string& key, std::uint32_t fallback) {
    return obj.contains(key) ? static_cast<std::uint32_t>(obj.at(key).as_number()) : fallback;
}

std::uint64_t get_u64_or(const JsonValue& obj, const std::string& key, std::uint64_t fallback) {
    return obj.contains(key) ? static_cast<std::uint64_t>(obj.at(key).as_number()) : fallback;
}

float get_float_or(const JsonValue& obj, const std::string& key, float fallback) {
    return obj.contains(key) ? static_cast<float>(obj.at(key).as_number()) : fallback;
}

std::vector<std::string> get_string_array_or(const JsonValue& obj, const std::string& key, std::vector<std::string> fallback) {
    if (!obj.contains(key)) {
        return fallback;
    }
    std::vector<std::string> values;
    for (const auto& item : obj.at(key).as_array()) {
        values.push_back(item.as_string());
    }
    return values;
}

}  // namespace

FSConfig load_config(const std::string& filepath) {
    const auto parsed = parse_json(read_file(filepath));
    if (!parsed.is_object()) {
        throw ValidationError("config root must be a JSON object");
    }

    FSConfig cfg;
    const auto& root = parsed;
    if (!root.contains("fs") || !root.contains("index") || !root.contains("benchmark")) {
        throw ValidationError("missing required top-level section");
    }

    cfg.fs.mount_point = get_string_or(root.at("fs"), "mount_point", "/home/user/data");
    cfg.fs.fuse_opts = get_string_array_or(root.at("fs"), "fuse_opts", {"-o", "auto_cache"});
    cfg.fs.backing_root = get_string_or(root.at("fs"), "backing_root", "./backing_root");
    cfg.fs.strict_mode = root.at("fs").contains("strict_mode") ? root.at("fs").at("strict_mode").as_bool() : false;

    cfg.index.type = get_string_or(root.at("index"), "type", "g-index");
    if (root.at("index").contains("training")) {
        const auto& training = root.at("index").at("training");
        cfg.index.training.sample_ratio = get_float_or(training, "sample_ratio", 1.0f);
        cfg.index.training.key_encoding = get_string_or(training, "key_encoding", "trie_prefix");
    }
    if (root.at("index").contains("inference")) {
        const auto& inference = root.at("index").at("inference");
        cfg.index.inference.batch_size = get_u32_or(inference, "batch_size", 256);
        cfg.index.inference.fallback_on_miss = inference.contains("fallback_on_miss") ? inference.at("fallback_on_miss").as_bool() : false;
    }
    if (root.at("index").contains("resource")) {
        const auto& resource = root.at("index").at("resource");
        cfg.index.resource.max_vram_bytes = get_u64_or(resource, "max_vram_bytes", 1024ULL * 1024ULL * 1024ULL);
    }

    cfg.benchmark.warmup_iters = get_u32_or(root.at("benchmark"), "warmup_iters", 100);
    cfg.benchmark.metrics = get_string_array_or(root.at("benchmark"), "metrics", {"p50", "p99", "throughput"});

    validate_config(cfg);
    return cfg;
}

void validate_config(const FSConfig& cfg) {
    if (cfg.fs.mount_point.empty()) {
        throw ValidationError("missing: fs.mount_point");
    }
    if (cfg.fs.backing_root.empty()) {
        throw ValidationError("missing: fs.backing_root");
    }
    if (cfg.index.type.empty()) {
        throw ValidationError("missing: index.type");
    }
    if (cfg.index.training.sample_ratio < 0.0f || cfg.index.training.sample_ratio > 1.0f) {
        throw ValidationError("expected sample_ratio in [0,1]");
    }
    if (cfg.index.resource.max_vram_bytes > 1024ULL * 1024ULL * 1024ULL) {
        throw ValidationError("index.resource.max_vram_bytes exceeds 1GB constraint");
    }
    if (cfg.index.inference.fallback_on_miss) {
        throw ValidationError("config.index.inference.fallback_on_miss must be false");
    }
    if (cfg.fs.strict_mode && cfg.index.inference.fallback_on_miss) {
        throw ValidationError("strict_mode cannot be combined with fallback_on_miss");
    }
    if (cfg.benchmark.warmup_iters == 0) {
        throw ValidationError("benchmark.warmup_iters must be > 0");
    }
}

}  // namespace glfs
