#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace glfs {

struct ValidationError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct FSConfig {
    struct {
        std::string mount_point;
        std::vector<std::string> fuse_opts;
        std::string backing_root;
        bool strict_mode;
    } fs;

    struct {
        std::string type;
        struct {
            float sample_ratio;
            std::string key_encoding;
            std::uint32_t max_epochs;
        } training;
        struct {
            std::uint32_t batch_size;
            bool fallback_on_miss;
        } inference;
        struct {
            std::uint64_t max_vram_bytes;
        } resource;
        struct {
            std::uint32_t max_depth;
            std::uint32_t bits_per_level;
        } path_encoding;
        struct {
            std::uint32_t segment_base_width;
            std::uint32_t segment_min_width;
            std::uint32_t segment_max_width;
            std::uint32_t segment_epoch_cap;
            std::uint32_t lookup_window;
            std::uint32_t cuda_block_size;
            std::uint32_t latency_history_limit;
            std::uint64_t vram_overhead_bytes;
        } backend;
    } index;

    struct {
        std::uint32_t warmup_iters;
        std::uint32_t measure_iters;
        std::vector<std::string> metrics;
    } benchmark;
};

FSConfig load_config(const std::string& filepath);
void validate_config(const FSConfig& cfg);

}  // namespace glfs
