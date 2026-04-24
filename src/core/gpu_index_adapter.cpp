#include "core/gpu_index_adapter.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <numeric>
#include <sstream>

namespace glfs {

namespace {

class StaticMapIndex final : public IGPUIndex {
public:
    explicit StaticMapIndex(std::string type) : type_(std::move(type)) {}

    void train(const std::vector<std::uint64_t>& keys,
               const std::vector<std::uint64_t>& values,
               const TrainingConfig& cfg) override {
        if (keys.size() != values.size()) {
            throw std::invalid_argument("keys and values size mismatch");
        }
        if (keys.empty()) {
            throw std::invalid_argument("training data must not be empty");
        }
        if (cfg.sample_ratio < 0.0f || cfg.sample_ratio > 1.0f) {
            throw std::invalid_argument("invalid sample_ratio");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        map_.clear();
        for (std::size_t i = 0; i < keys.size(); ++i) {
            map_[keys[i]] = values[i];
        }
        trained_ = true;
        vram_usage_ = std::min<std::size_t>(cfg.max_vram_mb * 1024ULL * 1024ULL,
                                            keys.size() * sizeof(std::uint64_t) * 2ULL + 1024ULL);
    }

    std::vector<std::uint64_t> batch_lookup(const std::vector<std::uint64_t>& keys,
                                            cudaStream_t) override {
        std::vector<std::uint64_t> out;
        out.reserve(keys.size());
        for (auto key : keys) {
            const auto start = std::chrono::steady_clock::now();
            std::uint64_t value = INVALID_INODE;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto it = map_.find(key);
                if (it != map_.end()) {
                    value = it->second;
                }
            }
            const auto end = std::chrono::steady_clock::now();
            const auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            if (profiling_enabled_) {
                record_latency(static_cast<double>(dur) / 1000.0);
            }
            query_count_.fetch_add(1, std::memory_order_relaxed);
            if (value == INVALID_INODE) {
                miss_count_.fetch_add(1, std::memory_order_relaxed);
            }
            out.push_back(value);
        }
        return out;
    }

    bool save(const std::string& filepath) override {
        std::lock_guard<std::mutex> lock(mutex_);
        std::ofstream out(filepath, std::ios::trunc);
        if (!out) {
            return false;
        }
        out << type_ << '\n';
        out << map_.size() << '\n';
        for (const auto& [key, value] : map_) {
            out << key << ' ' << value << '\n';
        }
        return out.good();
    }

    bool load(const std::string& filepath) override {
        std::ifstream in(filepath);
        if (!in) {
            return false;
        }
        std::string file_type;
        std::size_t size = 0;
        in >> file_type >> size;
        if (!in || file_type != type_) {
            return false;
        }
        std::unordered_map<std::uint64_t, std::uint64_t> loaded;
        for (std::size_t i = 0; i < size; ++i) {
            std::uint64_t key = 0, value = 0;
            in >> key >> value;
            if (!in) {
                return false;
            }
            loaded[key] = value;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        map_ = std::move(loaded);
        trained_ = true;
        return true;
    }

    IndexStats get_stats() const override {
        IndexStats stats;
        stats.query_count = query_count_.load(std::memory_order_relaxed);
        stats.miss_count = miss_count_.load(std::memory_order_relaxed);
        stats.vram_usage_bytes = vram_usage_;
        stats.gpu_util_percent = trained_ ? 1.0f : 0.0f;
        std::vector<double> copy;
        {
            std::lock_guard<std::mutex> lock(latency_mutex_);
            copy = latencies_us_;
        }
        if (!copy.empty()) {
            std::sort(copy.begin(), copy.end());
            stats.p50_latency_us = copy[copy.size() / 2];
            stats.p99_latency_us = copy[std::min<std::size_t>(copy.size() - 1, (copy.size() * 99) / 100)];
        }
        return stats;
    }

    void enable_profiling(bool enabled) override {
        profiling_enabled_ = enabled;
    }

    std::size_t get_vram_usage() const override {
        return vram_usage_;
    }

private:
    void record_latency(double us) const {
        std::lock_guard<std::mutex> lock(latency_mutex_);
        if (latencies_us_.size() >= 1024) {
            latencies_us_.erase(latencies_us_.begin());
        }
        latencies_us_.push_back(us);
    }

    std::string type_;
    mutable std::mutex mutex_;
    std::unordered_map<std::uint64_t, std::uint64_t> map_;
    bool trained_ = false;
    std::size_t vram_usage_ = 0;
    std::atomic<std::uint64_t> query_count_{0};
    std::atomic<std::uint64_t> miss_count_{0};
    bool profiling_enabled_ = false;
    mutable std::mutex latency_mutex_;
    mutable std::vector<double> latencies_us_;
};

}  // namespace

IGPUIndex* create_index(const std::string& type) {
    return new StaticMapIndex(type.empty() ? "g-index" : type);
}

void destroy_index(IGPUIndex* index) {
    delete index;
}

}  // namespace glfs
