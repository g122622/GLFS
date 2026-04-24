#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace glfs {

using cudaStream_t = void*;

static constexpr std::uint64_t INVALID_INODE = 0;

struct IndexStats {
    double p50_latency_us = 0.0;
    double p99_latency_us = 0.0;
    double throughput_qps = 0.0;
    float gpu_util_percent = 0.0f;
    std::size_t vram_usage_bytes = 0;
    std::uint64_t query_count = 0;
    std::uint64_t miss_count = 0;
};

struct TrainingConfig {
    std::string index_type = "g-index";
    float sample_ratio = 1.0f;
    std::uint32_t max_epochs = 1;
    std::size_t max_vram_mb = 1024;
};

struct ControlResult {
    std::uint64_t inode = INVALID_INODE;
    bool fallback_to_backing_root = false;
    std::string reason;
};

class IGPUIndex {
public:
    virtual ~IGPUIndex() = default;
    virtual void train(const std::vector<std::uint64_t>& keys,
                       const std::vector<std::uint64_t>& values,
                       const TrainingConfig& cfg) = 0;
    virtual std::vector<std::uint64_t> batch_lookup(const std::vector<std::uint64_t>& keys,
                                                    cudaStream_t stream = nullptr) = 0;
    virtual bool save(const std::string& filepath) = 0;
    virtual bool load(const std::string& filepath) = 0;
    virtual IndexStats get_stats() const = 0;
    virtual void enable_profiling(bool enabled) = 0;
    virtual std::size_t get_vram_usage() const = 0;
};

class IGPUControlPlane {
public:
    virtual ~IGPUControlPlane() = default;
    virtual void initialize(const std::string& index_type) = 0;
    virtual void train(const std::vector<std::uint64_t>& keys,
                       const std::vector<std::uint64_t>& values,
                       const TrainingConfig& cfg) = 0;
    virtual ControlResult lookup(std::uint64_t key) = 0;
    virtual IndexStats get_stats() const = 0;
    virtual void enable_profiling(bool enabled) = 0;
    virtual IGPUIndex* index() = 0;
};

IGPUIndex* create_index(const std::string& type);
void destroy_index(IGPUIndex* index);

IGPUControlPlane* create_control_plane(const std::string& type);
void destroy_control_plane(IGPUControlPlane* control_plane);

}  // namespace glfs
