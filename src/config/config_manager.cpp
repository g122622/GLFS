#include "config/config_manager.h"

#include <cmath>
#include <fstream>
#include <limits>
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

const JsonValue& require_section(const JsonValue& root, const std::string& key, const std::string& path) {
    if (!root.is_object()) {
        throw ValidationError("config root must be a JSON object");
    }
    if (!root.contains(key)) {
        throw ValidationError("missing required field: " + path);
    }
    const auto& section = root.at(key);
    if (!section.is_object()) {
        throw ValidationError("field must be an object: " + path);
    }
    return section;
}

const JsonValue& require_member(const JsonValue& obj, const std::string& key, const std::string& path) {
    if (!obj.is_object()) {
        throw ValidationError("field must be an object: " + path);
    }
    if (!obj.contains(key)) {
        throw ValidationError("missing required field: " + path);
    }
    return obj.at(key);
}

std::string require_string(const JsonValue& obj, const std::string& key, const std::string& path) {
    const auto& value = require_member(obj, key, path);
    if (!value.is_string()) {
        throw ValidationError("field must be a string: " + path);
    }
    const auto& result = value.as_string();
    if (result.empty()) {
        throw ValidationError("field must not be empty: " + path);
    }
    return result;
}

bool require_bool(const JsonValue& obj, const std::string& key, const std::string& path) {
    const auto& value = require_member(obj, key, path);
    if (!value.is_bool()) {
        throw ValidationError("field must be a boolean: " + path);
    }
    return value.as_bool();
}

std::uint32_t require_u32(const JsonValue& obj, const std::string& key, const std::string& path) {
    const auto& value = require_member(obj, key, path);
    if (!value.is_number()) {
        throw ValidationError("field must be a number: " + path);
    }
    const double n = value.as_number();
    if (!std::isfinite(n) || n < 0.0 || n > static_cast<double>(std::numeric_limits<std::uint32_t>::max()) || std::floor(n) != n) {
        throw ValidationError("field must be a non-negative integer: " + path);
    }
    return static_cast<std::uint32_t>(n);
}

std::uint64_t require_u64(const JsonValue& obj, const std::string& key, const std::string& path) {
    const auto& value = require_member(obj, key, path);
    if (!value.is_number()) {
        throw ValidationError("field must be a number: " + path);
    }
    const double n = value.as_number();
    if (!std::isfinite(n) || n < 0.0 || n > static_cast<double>(std::numeric_limits<std::uint64_t>::max()) || std::floor(n) != n) {
        throw ValidationError("field must be a non-negative integer: " + path);
    }
    return static_cast<std::uint64_t>(n);
}

float require_float(const JsonValue& obj, const std::string& key, const std::string& path) {
    const auto& value = require_member(obj, key, path);
    if (!value.is_number()) {
        throw ValidationError("field must be a number: " + path);
    }
    const double n = value.as_number();
    if (!std::isfinite(n)) {
        throw ValidationError("field must be finite: " + path);
    }
    return static_cast<float>(n);
}

std::vector<std::string> require_string_array(const JsonValue& obj, const std::string& key, const std::string& path) {
    const auto& value = require_member(obj, key, path);
    if (!value.is_array()) {
        throw ValidationError("field must be an array: " + path);
    }
    std::vector<std::string> result;
    result.reserve(value.size());
    for (std::size_t i = 0; i < value.size(); ++i) {
        const auto& item = value.at(i);
        if (!item.is_string()) {
            throw ValidationError("array entries must be strings: " + path);
        }
        result.push_back(item.as_string());
    }
    return result;
}

}  // namespace

FSConfig load_config(const std::string& filepath) {
    const auto parsed = parse_json(read_file(filepath));
    if (!parsed.is_object()) {
        throw ValidationError("config root must be a JSON object");
    }

    FSConfig cfg{};
    const auto& root = parsed;

    const auto& fs = require_section(root, "fs", "fs");
    cfg.fs.mount_point = require_string(fs, "mount_point", "fs.mount_point");
    cfg.fs.fuse_opts = require_string_array(fs, "fuse_opts", "fs.fuse_opts");
    cfg.fs.backing_root = require_string(fs, "backing_root", "fs.backing_root");
    cfg.fs.strict_mode = require_bool(fs, "strict_mode", "fs.strict_mode");

    const auto& index = require_section(root, "index", "index");
    cfg.index.type = require_string(index, "type", "index.type");

    const auto& training = require_section(index, "training", "index.training");
    cfg.index.training.sample_ratio = require_float(training, "sample_ratio", "index.training.sample_ratio");
    cfg.index.training.key_encoding = require_string(training, "key_encoding", "index.training.key_encoding");
    cfg.index.training.max_epochs = require_u32(training, "max_epochs", "index.training.max_epochs");

    const auto& inference = require_section(index, "inference", "index.inference");
    cfg.index.inference.batch_size = require_u32(inference, "batch_size", "index.inference.batch_size");
    cfg.index.inference.batch_timeout_us = require_u32(inference, "batch_timeout_us", "index.inference.batch_timeout_us");
    cfg.index.inference.fallback_on_miss = require_bool(inference, "fallback_on_miss", "index.inference.fallback_on_miss");

    const auto& resource = require_section(index, "resource", "index.resource");
    cfg.index.resource.max_vram_bytes = require_u64(resource, "max_vram_bytes", "index.resource.max_vram_bytes");

    const auto& path_encoding = require_section(index, "path_encoding", "index.path_encoding");
    cfg.index.path_encoding.max_depth = require_u32(path_encoding, "max_depth", "index.path_encoding.max_depth");
    cfg.index.path_encoding.bits_per_level = require_u32(path_encoding, "bits_per_level", "index.path_encoding.bits_per_level");

    const auto& backend = require_section(index, "backend", "index.backend");
    cfg.index.backend.segment_base_width = require_u32(backend, "segment_base_width", "index.backend.segment_base_width");
    cfg.index.backend.segment_min_width = require_u32(backend, "segment_min_width", "index.backend.segment_min_width");
    cfg.index.backend.segment_max_width = require_u32(backend, "segment_max_width", "index.backend.segment_max_width");
    cfg.index.backend.segment_epoch_cap = require_u32(backend, "segment_epoch_cap", "index.backend.segment_epoch_cap");
    cfg.index.backend.lookup_window = require_u32(backend, "lookup_window", "index.backend.lookup_window");
    cfg.index.backend.cuda_block_size = require_u32(backend, "cuda_block_size", "index.backend.cuda_block_size");
    cfg.index.backend.latency_history_limit = require_u32(backend, "latency_history_limit", "index.backend.latency_history_limit");
    cfg.index.backend.vram_overhead_bytes = require_u64(backend, "vram_overhead_bytes", "index.backend.vram_overhead_bytes");

    const auto& benchmark = require_section(root, "benchmark", "benchmark");
    cfg.benchmark.warmup_iters = require_u32(benchmark, "warmup_iters", "benchmark.warmup_iters");
    cfg.benchmark.measure_iters = require_u32(benchmark, "measure_iters", "benchmark.measure_iters");
    cfg.benchmark.read_size_bytes = require_u32(benchmark, "read_size_bytes", "benchmark.read_size_bytes");
    cfg.benchmark.report_csv_path = require_string(benchmark, "report_csv_path", "benchmark.report_csv_path");
    cfg.benchmark.report_plot_path = require_string(benchmark, "report_plot_path", "benchmark.report_plot_path");
    const auto& report_plot_paths = require_section(benchmark, "report_plot_paths", "benchmark.report_plot_paths");
    cfg.benchmark.report_plot_paths.traversal_latency_path = require_string(report_plot_paths, "traversal_latency_path", "benchmark.report_plot_paths.traversal_latency_path");
    cfg.benchmark.report_plot_paths.resource_summary_path = require_string(report_plot_paths, "resource_summary_path", "benchmark.report_plot_paths.resource_summary_path");
    cfg.benchmark.mode = require_string(benchmark, "mode", "benchmark.mode");
    cfg.benchmark.mount_wait_timeout_ms = require_u32(benchmark, "mount_wait_timeout_ms", "benchmark.mount_wait_timeout_ms");
    cfg.benchmark.mount_poll_interval_ms = require_u32(benchmark, "mount_poll_interval_ms", "benchmark.mount_poll_interval_ms");
    cfg.benchmark.daemon_stop_timeout_ms = require_u32(benchmark, "daemon_stop_timeout_ms", "benchmark.daemon_stop_timeout_ms");
    cfg.benchmark.metrics = require_string_array(benchmark, "metrics", "benchmark.metrics");

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
    if (cfg.index.training.key_encoding.empty()) {
        throw ValidationError("missing: index.training.key_encoding");
    }
    if (cfg.index.training.sample_ratio < 0.0f || cfg.index.training.sample_ratio > 1.0f) {
        throw ValidationError("expected index.training.sample_ratio in [0,1]");
    }
    if (cfg.index.training.max_epochs == 0) {
        throw ValidationError("index.training.max_epochs must be > 0");
    }
    if (cfg.index.inference.batch_size == 0) {
        throw ValidationError("index.inference.batch_size must be > 0");
    }
    if (cfg.index.inference.batch_timeout_us == 0) {
        throw ValidationError("index.inference.batch_timeout_us must be > 0");
    }
    if (cfg.index.inference.fallback_on_miss) {
        throw ValidationError("index.inference.fallback_on_miss must be false");
    }
    if (cfg.index.resource.max_vram_bytes == 0) {
        throw ValidationError("index.resource.max_vram_bytes must be > 0");
    }
    if (cfg.index.path_encoding.max_depth == 0) {
        throw ValidationError("index.path_encoding.max_depth must be > 0");
    }
    if (cfg.index.path_encoding.bits_per_level == 0 || cfg.index.path_encoding.bits_per_level > 16) {
        throw ValidationError("index.path_encoding.bits_per_level must be in [1,16]");
    }
    if (cfg.index.backend.segment_base_width == 0) {
        throw ValidationError("index.backend.segment_base_width must be > 0");
    }
    if (cfg.index.backend.segment_min_width == 0) {
        throw ValidationError("index.backend.segment_min_width must be > 0");
    }
    if (cfg.index.backend.segment_max_width < cfg.index.backend.segment_min_width) {
        throw ValidationError("index.backend.segment_max_width must be >= segment_min_width");
    }
    if (cfg.index.backend.segment_epoch_cap == 0) {
        throw ValidationError("index.backend.segment_epoch_cap must be > 0");
    }
    if (cfg.index.backend.lookup_window == 0) {
        throw ValidationError("index.backend.lookup_window must be > 0");
    }
    if (cfg.index.backend.cuda_block_size == 0) {
        throw ValidationError("index.backend.cuda_block_size must be > 0");
    }
    if (cfg.index.backend.latency_history_limit == 0) {
        throw ValidationError("index.backend.latency_history_limit must be > 0");
    }
    if (cfg.index.backend.vram_overhead_bytes == 0) {
        throw ValidationError("index.backend.vram_overhead_bytes must be > 0");
    }
    if (cfg.benchmark.warmup_iters < 0) {
        throw ValidationError("benchmark.warmup_iters must be >= 0");
    }
    if (cfg.benchmark.measure_iters == 0) {
        throw ValidationError("benchmark.measure_iters must be > 0");
    }
    if (cfg.benchmark.read_size_bytes == 0) {
        throw ValidationError("benchmark.read_size_bytes must be > 0");
    }
    if (cfg.benchmark.report_csv_path.empty()) {
        throw ValidationError("missing: benchmark.report_csv_path");
    }
    if (cfg.benchmark.report_plot_path.empty()) {
        throw ValidationError("missing: benchmark.report_plot_path");
    }
    if (cfg.benchmark.report_plot_paths.traversal_latency_path.empty()) {
        throw ValidationError("missing: benchmark.report_plot_paths.traversal_latency_path");
    }
    if (cfg.benchmark.report_plot_paths.resource_summary_path.empty()) {
        throw ValidationError("missing: benchmark.report_plot_paths.resource_summary_path");
    }
    if (cfg.benchmark.mode != "dual_compare") {
        throw ValidationError("benchmark.mode must be dual_compare");
    }
    if (cfg.benchmark.mount_wait_timeout_ms == 0) {
        throw ValidationError("benchmark.mount_wait_timeout_ms must be > 0");
    }
    if (cfg.benchmark.mount_poll_interval_ms == 0) {
        throw ValidationError("benchmark.mount_poll_interval_ms must be > 0");
    }
    if (cfg.benchmark.daemon_stop_timeout_ms == 0) {
        throw ValidationError("benchmark.daemon_stop_timeout_ms must be > 0");
    }
    if (cfg.benchmark.metrics.empty()) {
        throw ValidationError("benchmark.metrics must not be empty");
    }
}

}  // namespace glfs
