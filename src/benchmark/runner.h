#pragma once

#include <string>
#include <vector>

#include "config/config_manager.h"
#include "core/gpu_index_adapter.h"

namespace glfs {

struct BenchmarkResult {
    std::string scenario;
    double p50_us = 0.0;
    double p99_us = 0.0;
    double p999_us = 0.0;
    double throughput_qps = 0.0;
    IndexStats index_stats;
};

std::vector<BenchmarkResult> run_benchmarks(const std::string& mount_point, const FSConfig& cfg);

}  // namespace glfs
