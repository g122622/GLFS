#pragma once

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
    } fs;

    struct {
        std::string type;
        struct {
            float sample_ratio = 1.0f;
            std::string key_encoding = "trie_prefix";
        } training;
        struct {
            std::uint32_t batch_size = 256;
            bool fallback_on_miss = false;
        } inference;
        struct {
            std::uint64_t max_vram_bytes = 1024ULL * 1024ULL * 1024ULL;
        } resource;
    } index;

    struct {
        std::uint32_t warmup_iters = 100;
        std::vector<std::string> metrics = {"p50", "p99", "throughput"};
    } benchmark;
};

FSConfig load_config(const std::string& filepath);
void validate_config(const FSConfig& cfg);

}  // namespace glfs
