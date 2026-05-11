#include "core/gpu_index_adapter.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <numeric>
#include <sstream>

namespace glfs::backends::g_index {
IGPUIndex* create_backend();
}  // namespace glfs::backends::g_index

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

class LocalGPUControlPlane final : public IGPUControlPlane {
public:
    void initialize(const std::string& index_type) override {
        if (index_type.empty()) {
            throw std::invalid_argument("index_type must not be empty");
        }
        if (!index_) {
            index_.reset(create_index(index_type));
        }
        index_type_ = index_type;
    }

    void train(const std::vector<std::uint64_t>& keys,
               const std::vector<std::uint64_t>& values,
               const TrainingConfig& cfg) override {
        ensure_index(cfg.index_type);
        index_->train(keys, values, cfg);
    }

    ControlResult lookup(std::uint64_t key) override {
        auto results = lookup_batch({key});
        if (!results.empty()) {
            return results.front();
        }
        ControlResult result;
        result.fallback_to_backing_root = true;
        result.reason = "control_plane_empty_batch";
        return result;
    }

    std::vector<ControlResult> lookup_batch(const std::vector<std::uint64_t>& keys) override {
        std::vector<ControlResult> results;
        results.reserve(keys.size());
        if (keys.empty()) {
            return results;
        }
        control_stats_.queued_requests += static_cast<std::uint64_t>(keys.size());
        ensure_index(index_type_);
        auto values = index_->batch_lookup(keys);
        for (auto value : values) {
            ControlResult result;
            if (value != INVALID_INODE) {
                result.inode = value;
                result.reason = "control_plane_hit";
                ++control_stats_.completed_requests;
            } else {
                result.fallback_to_backing_root = true;
                result.reason = "control_plane_miss";
                ++control_stats_.fallback_requests;
            }
            results.push_back(std::move(result));
        }
        return results;
    }

    MutationDecision decide_mutation(const MutationRequest& request) override {
        MutationDecision decision;
        control_stats_.queued_requests++;

        switch (request.op) {
            case MutationOp::Create:
            case MutationOp::Mkdir:
                decision.allowed = true;
                decision.reason = "control_plane_allow_create_like";
                ++control_stats_.completed_requests;
                return decision;
            case MutationOp::Truncate:
                if (request.byte_size > 0) {
                    decision.allowed = true;
                    decision.reason = "control_plane_allow_truncate";
                } else {
                    decision.allowed = true;
                    decision.reason = "control_plane_allow_truncate_zero";
                }
                ++control_stats_.completed_requests;
                return decision;
            case MutationOp::Unlink:
                decision.allowed = true;
                decision.reason = "control_plane_allow_unlink";
                ++control_stats_.completed_requests;
                return decision;
            case MutationOp::Rename:
                if (request.has_secondary) {
                    decision.allowed = true;
                    decision.reason = "control_plane_allow_rename";
                } else {
                    decision.allowed = false;
                    decision.fallback_to_backing_root = true;
                    decision.reason = "control_plane_rename_missing_target";
                    ++control_stats_.fallback_requests;
                    return decision;
                }
                ++control_stats_.completed_requests;
                return decision;
        }

        decision.allowed = false;
        decision.fallback_to_backing_root = true;
        decision.reason = "control_plane_unknown_mutation";
        ++control_stats_.fallback_requests;
        return decision;
    }

    void submit_lookup_batch(const std::vector<std::uint64_t>& keys) override {
        control_stats_.queued_requests += static_cast<std::uint64_t>(keys.size());
        pending_batches_.push_back(keys);
    }

    void set_namespace(const std::map<std::string, std::vector<std::string>>& children) override {
        namespace_children_ = children;
        for (auto& [_, child_list] : namespace_children_) {
            std::sort(child_list.begin(), child_list.end());
            child_list.erase(std::unique(child_list.begin(), child_list.end()), child_list.end());
        }
    }

    std::vector<std::string> list_children(const std::string& path) const override {
        auto it = namespace_children_.find(path);
        if (it == namespace_children_.end()) {
            return {};
        }
        return it->second;
    }

    void drain() override {
        for (const auto& batch : pending_batches_) {
            if (!batch.empty()) {
                (void)index_->batch_lookup(batch);
                control_stats_.completed_requests += static_cast<std::uint64_t>(batch.size());
            }
        }
        pending_batches_.clear();
    }

    IndexStats get_stats() const override {
        return index_ ? index_->get_stats() : IndexStats{};
    }

    ControlPlaneStats get_control_plane_stats() const override {
        return control_stats_;
    }

    void enable_profiling(bool enabled) override {
        if (index_) {
            index_->enable_profiling(enabled);
        }
    }

    IGPUIndex* index() override {
        return index_.get();
    }

private:
    void ensure_index(const std::string& type) {
        if (type.empty()) {
            throw std::invalid_argument("index_type must not be empty");
        }
        if (!index_) {
            index_.reset(create_index(type));
            index_type_ = type;
        }
    }

    std::unique_ptr<IGPUIndex> index_;
    std::string index_type_;
    ControlPlaneStats control_stats_;
    std::vector<std::vector<std::uint64_t>> pending_batches_;
    std::map<std::string, std::vector<std::string>> namespace_children_;
};

}  // namespace

IGPUIndex* create_index(const std::string& type) {
    if (type.empty()) {
        throw std::invalid_argument("index type must not be empty");
    }
    if (type == "g-index") {
        return glfs::backends::g_index::create_backend();
    }
    return new StaticMapIndex(type);
}

void destroy_index(IGPUIndex* index) {
    delete index;
}

void destroy_control_plane(IGPUControlPlane* control_plane) {
    delete control_plane;
}

}  // namespace glfs
