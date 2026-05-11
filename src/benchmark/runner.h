#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "config/config_manager.h"
#include "core/gpu_index_adapter.h"

namespace glfs {

struct BenchmarkResult {
     std::string backend;
    std::string operation;
    std::string path;
    double p50_us = 0.0;
    double p99_us = 0.0;
    double p999_us = 0.0;
    double throughput_qps = 0.0;
    IndexStats index_stats;
};

struct BenchmarkTreeEntry {
    std::filesystem::path path;
    bool is_dir = false;
};

std::vector<BenchmarkTreeEntry> collect_benchmark_tree(const std::filesystem::path& root);
std::vector<BenchmarkResult> run_benchmarks(const FSConfig& cfg);
void write_benchmark_report_csv(const std::string& csv_path, const std::vector<BenchmarkResult>& results);

}  // namespace glfs
