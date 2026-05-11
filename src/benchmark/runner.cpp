#include "benchmark/runner.h"

#include <algorithm>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "fuse/client_api.h"
#include "fuse/gpufs_ops.h"
#include "utils/perfetto_integration.h"
#include "utils/timer.h"

namespace glfs {

namespace {

volatile std::sig_atomic_t g_benchmark_stop_requested = 0;

void throw_if_stopped() {
    if (g_benchmark_stop_requested != 0) {
        throw std::runtime_error("benchmark interrupted");
    }
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    const std::size_t idx = std::min<std::size_t>(values.size() - 1, static_cast<std::size_t>(p * (values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + idx, values.end());
    return values[idx];
}

std::vector<double> measure_us(std::uint32_t warmup_iters,
                               std::uint32_t measure_iters,
                               const std::function<void()>& op) {
    std::vector<double> durations;
    durations.reserve(measure_iters);
    for (std::uint32_t i = 0; i < warmup_iters + measure_iters; ++i) {
        throw_if_stopped();
        const auto start = now_ns();
        op();
        const auto dur = static_cast<double>(now_ns() - start) / 1000.0;
        if (i >= warmup_iters) {
            durations.push_back(dur);
        }
    }
    return durations;
}

std::string csv_escape(const std::string& s) {
    if (s.find_first_of(",\"\n") == std::string::npos) {
        return s;
    }
    std::string out;
    out.push_back('"');
    for (char c : s) {
        if (c == '"') {
            out += "\"\"";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('"');
    return out;
}

std::string workload_relative(const std::filesystem::path& root, const std::filesystem::path& p) {
    const auto rel = std::filesystem::relative(p, root);
    if (rel.empty()) {
        return "/";
    }
    return "/" + rel.generic_string();
}

std::filesystem::path resolve_workload_path(const std::filesystem::path& root, const std::string& rel_path) {
    if (rel_path == "/") {
        return root;
    }
    return root / rel_path.substr(1);
}

BenchmarkResult make_result(const std::string& backend,
                            const std::string& operation,
                            const std::string& path,
                            const std::vector<double>& durations,
                            const IndexStats& stats) {
    BenchmarkResult result;
    result.backend = backend;
    result.operation = operation;
    result.path = path;
    result.p50_us = percentile(durations, 0.50);
    result.p99_us = percentile(durations, 0.99);
    result.p999_us = percentile(durations, 0.999);
    result.throughput_qps = durations.empty() ? 0.0 : (static_cast<double>(durations.size()) * 1000000.0 / std::accumulate(durations.begin(), durations.end(), 0.0));
    result.index_stats = stats;
    return result;
}

std::vector<BenchmarkTreeEntry> collect_tree(const std::filesystem::path& root) {
    if (root.empty()) {
        throw std::runtime_error("benchmark root path is empty");
    }
    std::error_code ec;
    if (!std::filesystem::exists(root, ec)) {
        throw std::runtime_error("benchmark root does not exist: " + root.string());
    }

    std::vector<BenchmarkTreeEntry> entries;
    for (std::filesystem::recursive_directory_iterator it(root, ec), end; it != end && !ec; it.increment(ec)) {
        throw_if_stopped();
        entries.push_back(BenchmarkTreeEntry{it->path(), it->is_directory(ec)});
    }
    if (ec) {
        throw std::runtime_error("failed to traverse benchmark root: " + ec.message());
    }

    std::sort(entries.begin(), entries.end(), [](const BenchmarkTreeEntry& a, const BenchmarkTreeEntry& b) {
        return a.path.generic_string() < b.path.generic_string();
    });
    return entries;
}

std::vector<BenchmarkResult> run_filesystem_backend(const FSConfig& cfg,
                                                    const std::string& backend,
                                                    const std::filesystem::path& root,
                                                    const std::vector<BenchmarkTreeEntry>& tree) {
    std::vector<BenchmarkResult> results;
    results.reserve(tree.size() * 2);

    std::cerr << "[benchmark] " << backend << ": " << tree.size() << " entries\n";
    const std::size_t progress_step = std::max<std::size_t>(1, tree.size() / 20);

    for (std::size_t index = 0; index < tree.size(); ++index) {
        throw_if_stopped();
        const auto& entry = tree[index];
        const std::string rel_path = workload_relative(cfg.fs.backing_root, entry.path);
        const auto abs_path = resolve_workload_path(root, rel_path);
        if (index == 0 || index + 1 == tree.size() || ((index + 1) % progress_step) == 0) {
            std::cerr << "[benchmark] " << backend << " " << (index + 1) << "/" << tree.size()
                      << " " << rel_path << '\n';
        }
        if (entry.is_dir) {
            const auto stat_durations = measure_us(cfg.benchmark.warmup_iters, cfg.benchmark.measure_iters, [&]() {
                struct stat st{};
                std::error_code ec;
                (void)std::filesystem::status(abs_path, ec);
                if (ec) {
                    throw std::runtime_error(ec.message());
                }
                (void)st;
            });
            results.push_back(make_result(backend, "stat_getattr", rel_path, stat_durations, IndexStats{}));

            const auto readdir_durations = measure_us(cfg.benchmark.warmup_iters, cfg.benchmark.measure_iters, [&]() {
                std::error_code ec;
                for (std::filesystem::directory_iterator it(abs_path, ec), end; it != end && !ec; it.increment(ec)) {
                    (void)it->path();
                }
                if (ec) {
                    throw std::runtime_error(ec.message());
                }
            });
            results.push_back(make_result(backend, "readdir", rel_path, readdir_durations, IndexStats{}));
        } else {
            const auto open_durations = measure_us(cfg.benchmark.warmup_iters, cfg.benchmark.measure_iters, [&]() {
                std::ifstream in(abs_path, std::ios::binary);
                if (!in) {
                    throw std::runtime_error("unable to open file: " + abs_path.string());
                }
            });
            results.push_back(make_result(backend, "open", rel_path, open_durations, IndexStats{}));

            const auto read_durations = measure_us(cfg.benchmark.warmup_iters, cfg.benchmark.measure_iters, [&]() {
                std::ifstream in(abs_path, std::ios::binary);
                if (!in) {
                    throw std::runtime_error("unable to open file: " + abs_path.string());
                }
                std::vector<char> buf(cfg.benchmark.read_size_bytes);
                in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
            });
            results.push_back(make_result(backend, "read_lookup", rel_path, read_durations, IndexStats{}));
        }
    }

    std::cerr << "[benchmark] " << backend << ": done\n";

    return results;
}

}  // namespace

std::vector<BenchmarkTreeEntry> collect_benchmark_tree(const std::filesystem::path& root) {
    return collect_tree(root);
}

std::vector<BenchmarkResult> run_benchmarks(const FSConfig& cfg) {
    throw_if_stopped();
    if (cfg.benchmark.report_csv_path.empty()) {
        throw std::runtime_error("missing benchmark.report_csv_path");
    }

    const auto tree = collect_tree(cfg.fs.backing_root);

    std::cerr << "[benchmark] tree entries: " << tree.size() << '\n';

    auto glfs_results = run_filesystem_backend(cfg, "glfs", cfg.fs.mount_point, tree);
    auto backing_results = run_filesystem_backend(cfg, "backing_root", cfg.fs.backing_root, tree);

    std::vector<BenchmarkResult> results;
    results.reserve(glfs_results.size() + backing_results.size());
    results.insert(results.end(), glfs_results.begin(), glfs_results.end());
    results.insert(results.end(), backing_results.begin(), backing_results.end());

    perfetto_flush();
    std::cerr << "[benchmark] collected " << results.size() << " result rows\n";
    return results;
}

void request_benchmark_stop() {
    g_benchmark_stop_requested = 1;
}

void reset_benchmark_stop() {
    g_benchmark_stop_requested = 0;
}

bool benchmark_stop_requested() {
    return g_benchmark_stop_requested != 0;
}

void write_benchmark_report_csv(const std::string& csv_path, const std::vector<BenchmarkResult>& results) {
    if (csv_path.empty()) {
        throw std::runtime_error("missing benchmark.report_csv_path");
    }
    std::ofstream out(csv_path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("unable to open benchmark CSV report: " + csv_path);
    }

    out << "backend,operation,path,p50_us,p99_us,p999_us,throughput_qps,query_count,miss_count,gpu_util_percent,vram_usage_bytes\n";
    for (const auto& r : results) {
        out << csv_escape(r.backend) << ','
            << csv_escape(r.operation) << ','
            << csv_escape(r.path) << ','
            << r.p50_us << ','
            << r.p99_us << ','
            << r.p999_us << ','
            << r.throughput_qps << ','
            << r.index_stats.query_count << ','
            << r.index_stats.miss_count << ','
            << r.index_stats.gpu_util_percent << ','
            << r.index_stats.vram_usage_bytes << '\n';
    }
}

}  // namespace glfs
