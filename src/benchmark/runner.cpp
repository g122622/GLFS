#include "benchmark/runner.h"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <random>

#include "core/path_encoder.h"
#include "fuse/client_api.h"
#include "fuse/gpufs_ops.h"
#include "utils/perfetto_integration.h"
#include "utils/timer.h"

namespace glfs {

namespace {

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    const std::size_t idx = std::min<std::size_t>(values.size() - 1, static_cast<std::size_t>(p * (values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + idx, values.end());
    return values[idx];
}

std::vector<std::pair<std::string, std::string>> scenarios() {
    return {{"random_stat", "/train/img/cat.jpg"}, {"seq_readdir", "/train/img"}, {"mixed_rw", "/train/text/readme.txt"}};
}

}  // namespace

std::vector<BenchmarkResult> run_benchmarks(const std::string& mount_point, const FSConfig& cfg) {
    std::vector<BenchmarkResult> results;
    auto* control_plane = create_control_plane(cfg.index.type);
    FSConfig runtime_cfg = cfg;
    runtime_cfg.fs.mount_point = mount_point;
    GPULearnedFS fs;
    fs.path_cfg.mount_point = mount_point;
    gpufs_init(fs, control_plane, runtime_cfg);
    set_active_fs(&fs);

    for (const auto& [scenario, rel_path] : scenarios()) {
        std::vector<double> durations;
        const std::uint32_t warmup = cfg.benchmark.warmup_iters;
        const std::uint32_t iters = 100;
        struct stat st{};
        for (std::uint32_t i = 0; i < warmup + iters; ++i) {
            const auto start = now_ns();
            if (scenario == "seq_readdir") {
                fuse_client_readdir(mount_point.c_str(), "/train/img", nullptr, nullptr);
            } else {
                fuse_client_stat(mount_point.c_str(), rel_path.c_str(), &st);
            }
            const auto dur = static_cast<double>(now_ns() - start) / 1000.0;
            if (i >= warmup) {
                durations.push_back(dur);
            }
        }

        BenchmarkResult result;
        result.scenario = scenario;
        result.p50_us = percentile(durations, 0.50);
        result.p99_us = percentile(durations, 0.99);
        result.p999_us = percentile(durations, 0.999);
        result.throughput_qps = durations.empty() ? 0.0 : (static_cast<double>(durations.size()) * 1000000.0 / std::accumulate(durations.begin(), durations.end(), 0.0));
        result.index_stats = control_plane->get_stats();
        results.push_back(result);
    }

    perfetto_flush();
    destroy_control_plane(control_plane);
    return results;
}

}  // namespace glfs
