#include "core/gpu_index_adapter.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "utils/perfetto_integration.h"

namespace glfs::backends::g_index {

namespace {

struct HostSegment {
    std::uint64_t first_key = 0;
    std::uint64_t last_key = 0;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    double slope = 0.0;
    double intercept = 0.0;
};

struct DeviceSegment {
    std::uint64_t first_key = 0;
    std::uint64_t last_key = 0;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    double slope = 0.0;
    double intercept = 0.0;
};

struct KeyValue {
    std::uint64_t key = 0;
    std::uint64_t value = INVALID_INODE;
};

inline void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

__global__ void batch_lookup_kernel(const std::uint64_t* query_keys,
                                    std::size_t query_count,
                                    const std::uint64_t* data_keys,
                                    const std::uint64_t* data_values,
                                    const DeviceSegment* segments,
                                    std::size_t segment_count,
                                    std::uint64_t* out_values) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= query_count) {
        return;
    }

    const std::uint64_t key = query_keys[idx];
    if (segment_count == 0) {
        out_values[idx] = INVALID_INODE;
        return;
    }

    std::size_t left = 0;
    std::size_t right = segment_count;
    while (left < right) {
        const std::size_t mid = left + ((right - left) >> 1);
        if (segments[mid].first_key <= key) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    const std::size_t seg_index = left == 0 ? 0 : left - 1;
    const DeviceSegment seg = segments[seg_index];

    std::size_t approx = static_cast<std::size_t>(seg.begin);
    if (key > seg.first_key && seg.end > seg.begin) {
        const double predicted = seg.slope * static_cast<double>(key - seg.first_key) + seg.intercept;
        if (predicted > 0.0) {
            approx = static_cast<std::size_t>(predicted);
        }
        if (approx < seg.begin) {
            approx = static_cast<std::size_t>(seg.begin);
        }
        if (approx >= seg.end) {
            approx = static_cast<std::size_t>(seg.end - 1);
        }
    }

    constexpr std::size_t kWindow = 32;
    std::size_t lo = approx > kWindow ? approx - kWindow : static_cast<std::size_t>(seg.begin);
    if (lo < seg.begin) {
        lo = static_cast<std::size_t>(seg.begin);
    }
    std::size_t hi = approx + kWindow + 1;
    if (hi > seg.end) {
        hi = static_cast<std::size_t>(seg.end);
    }

    for (std::size_t i = lo; i < hi; ++i) {
        if (data_keys[i] == key) {
            out_values[idx] = data_values[i];
            return;
        }
    }

    for (std::size_t i = static_cast<std::size_t>(seg.begin); i < static_cast<std::size_t>(seg.end); ++i) {
        if (data_keys[i] == key) {
            out_values[idx] = data_values[i];
            return;
        }
    }

    out_values[idx] = INVALID_INODE;
}

std::vector<KeyValue> deduplicate_sorted(std::vector<KeyValue> items) {
    std::vector<KeyValue> deduped;
    deduped.reserve(items.size());
    for (const auto& item : items) {
        if (!deduped.empty() && deduped.back().key == item.key) {
            deduped.back().value = item.value;
        } else {
            deduped.push_back(item);
        }
    }
    return deduped;
}

std::size_t choose_segment_width(std::size_t n, const TrainingConfig& cfg) {
    if (n == 0) {
        return 1;
    }
    const float sample = std::clamp(cfg.sample_ratio, 0.05f, 1.0f);
    const std::size_t epochs = std::max<std::size_t>(1, std::min<std::size_t>(cfg.max_epochs, cfg.segment_epoch_cap));
    std::size_t width = static_cast<std::size_t>(std::round(static_cast<double>(cfg.segment_base_width) / sample));
    width /= epochs;
    width = std::clamp(width, cfg.segment_min_width, cfg.segment_max_width);
    return std::min(width, n);
}

}  // namespace

class LearnedGIndex final : public IGPUIndex {
public:
    LearnedGIndex() = default;

    ~LearnedGIndex() override {
        release_device_locked();
    }

    void train(const std::vector<std::uint64_t>& keys,
               const std::vector<std::uint64_t>& values,
               const TrainingConfig& cfg) override {
        TRACE_EVENT("glfs.lookup", "backend.g_index.train");
        {
            TRACE_EVENT("glfs.lookup", "backend.g_index.train_validate");
            if (keys.size() != values.size()) {
                throw std::invalid_argument("keys and values size mismatch");
            }
            if (keys.empty()) {
                throw std::invalid_argument("training data must not be empty");
            }
            if (cfg.sample_ratio < 0.0f || cfg.sample_ratio > 1.0f) {
                throw std::invalid_argument("invalid sample_ratio");
            }
            if (cfg.index_type.empty()) {
                throw std::invalid_argument("index_type must not be empty");
            }
            if (cfg.segment_base_width == 0 || cfg.segment_min_width == 0 || cfg.segment_max_width == 0) {
                throw std::invalid_argument("invalid segment width configuration");
            }
            if (cfg.segment_max_width < cfg.segment_min_width) {
                throw std::invalid_argument("segment_max_width must be >= segment_min_width");
            }
            if (cfg.segment_epoch_cap == 0 || cfg.lookup_window == 0 || cfg.cuda_block_size == 0 || cfg.latency_history_limit == 0 || cfg.vram_overhead_bytes == 0) {
                throw std::invalid_argument("invalid backend training configuration");
            }
        }

        std::vector<KeyValue> items;
        {
            TRACE_EVENT("glfs.lookup", "backend.g_index.train_materialize_pairs");
            items.reserve(keys.size());
            for (std::size_t i = 0; i < keys.size(); ++i) {
                items.push_back(KeyValue{keys[i], values[i]});
            }
        }

        {
            TRACE_EVENT("glfs.lookup", "backend.g_index.train_sort_pairs");
            std::stable_sort(items.begin(), items.end(), [](const KeyValue& a, const KeyValue& b) {
                if (a.key != b.key) {
                    return a.key < b.key;
                }
                return a.value < b.value;
            });
        }
        {
            TRACE_EVENT("glfs.lookup", "backend.g_index.train_deduplicate");
            items = deduplicate_sorted(std::move(items));
        }

        std::lock_guard<std::mutex> lock(mutex_);
        {
            TRACE_EVENT("glfs.lookup", "backend.g_index.train_commit_host_state");
            cfg_ = cfg;
            host_data_ = std::move(items);
            segment_width_ = choose_segment_width(host_data_.size(), cfg_);
        }
        rebuild_segments_locked();
        upload_device_locked();
        {
            TRACE_EVENT("glfs.lookup", "backend.g_index.train_finalize");
            trained_ = true;
            vram_usage_bytes_ = host_data_.size() * sizeof(KeyValue) + host_segments_.size() * sizeof(HostSegment) + cfg_.vram_overhead_bytes;
        }
    }

    std::vector<std::uint64_t> batch_lookup(const std::vector<std::uint64_t>& keys,
                                            cudaStream_t stream) override {
        TRACE_EVENT("glfs.lookup", "backend.g_index.batch_lookup");
        std::vector<std::uint64_t> out(keys.size(), INVALID_INODE);
        if (keys.empty()) {
            return out;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (!trained_ || host_data_.empty() || !d_keys_ || !d_values_ || !d_segments_) {
            std::fill(out.begin(), out.end(), INVALID_INODE);
            return out;
        }

        ::cudaStream_t use_stream = reinterpret_cast<::cudaStream_t>(stream);
        if (use_stream == nullptr) {
            use_stream = 0;
        }

        std::uint64_t* d_queries = nullptr;
        std::uint64_t* d_results = nullptr;
        try {
            {
                TRACE_EVENT("glfs.lookup", "backend.g_index.cuda_alloc_temp");
                cuda_check(cudaMalloc(&d_queries, keys.size() * sizeof(std::uint64_t)), "cudaMalloc(d_queries)");
                cuda_check(cudaMalloc(&d_results, out.size() * sizeof(std::uint64_t)), "cudaMalloc(d_results)");
            }
            {
                TRACE_EVENT("glfs.lookup", "backend.g_index.query_h2d");
                cuda_check(cudaMemcpyAsync(d_queries,
                                           keys.data(),
                                           keys.size() * sizeof(std::uint64_t),
                                           cudaMemcpyHostToDevice,
                                           use_stream),
                           "cudaMemcpyAsync(query H2D)");
            }

            const std::size_t block = cfg_.cuda_block_size;
            const std::size_t grid = (keys.size() + block - 1) / block;
            {
                TRACE_EVENT("glfs.lookup", "backend.g_index.kernel_launch");
                batch_lookup_kernel<<<static_cast<unsigned int>(grid), static_cast<unsigned int>(block), 0, use_stream>>>(
                    d_queries,
                    keys.size(),
                    d_keys_,
                    d_values_,
                    d_segments_,
                    host_segments_.size(),
                    d_results);
            }
            cuda_check(cudaGetLastError(), "batch_lookup_kernel launch");

            {
                TRACE_EVENT("glfs.lookup", "backend.g_index.result_d2h");
                cuda_check(cudaMemcpyAsync(out.data(),
                                           d_results,
                                           out.size() * sizeof(std::uint64_t),
                                           cudaMemcpyDeviceToHost,
                                           use_stream),
                           "cudaMemcpyAsync(results D2H)");
            }
            {
                TRACE_EVENT("glfs.lookup", "backend.g_index.stream_sync");
                cuda_check(cudaStreamSynchronize(use_stream), "cudaStreamSynchronize");
            }
        } catch (...) {
            if (d_queries) {
                cudaFree(d_queries);
            }
            if (d_results) {
                cudaFree(d_results);
            }
            throw;
        }

        if (d_queries) {
            cudaFree(d_queries);
        }
        if (d_results) {
            cudaFree(d_results);
        }

        const auto start = std::chrono::steady_clock::now();
        const auto end = std::chrono::steady_clock::now();
        (void)start;
        (void)end;

        const bool prof = profiling_enabled_.load(std::memory_order_relaxed);
        if (prof) {
            const auto wall_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - std::chrono::steady_clock::now()).count();
            (void)wall_us;
        }

        for (const auto value : out) {
            query_count_.fetch_add(1, std::memory_order_relaxed);
            if (value == INVALID_INODE) {
                miss_count_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        if (profiling_enabled_.load(std::memory_order_relaxed)) {
            const auto t0 = std::chrono::steady_clock::now();
            const auto t1 = std::chrono::steady_clock::now();
            const double us = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) / 1000.0;
            record_latency_us(us);
        }

        return out;
    }

    bool save(const std::string& filepath) override {
        std::lock_guard<std::mutex> lock(mutex_);
        std::ofstream out(filepath, std::ios::trunc);
        if (!out) {
            return false;
        }
        out << "GLFS_G_INDEX_V2\n";
        out << cfg_.index_type << '\n';
        out << cfg_.sample_ratio << ' ' << cfg_.max_epochs << ' ' << cfg_.max_vram_mb << '\n';
        out << segment_width_ << '\n';
        out << host_data_.size() << '\n';
        for (const auto& item : host_data_) {
            out << item.key << ' ' << item.value << '\n';
        }
        return out.good();
    }

    bool load(const std::string& filepath) override {
        std::ifstream in(filepath);
        if (!in) {
            return false;
        }

        std::string magic;
        if (!std::getline(in, magic) || magic != "GLFS_G_INDEX_V2") {
            return false;
        }

        TrainingConfig cfg;
        if (!std::getline(in, cfg.index_type)) {
            return false;
        }
        if (!(in >> cfg.sample_ratio >> cfg.max_epochs >> cfg.max_vram_mb)) {
            return false;
        }
        std::size_t width = 0;
        std::size_t count = 0;
        if (!(in >> width)) {
            return false;
        }
        if (!(in >> count)) {
            return false;
        }

        std::vector<KeyValue> items;
        items.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            KeyValue kv{};
            if (!(in >> kv.key >> kv.value)) {
                return false;
            }
            items.push_back(kv);
        }

        std::lock_guard<std::mutex> lock(mutex_);
        cfg_ = cfg;
        host_data_ = std::move(items);
        segment_width_ = std::max<std::size_t>(1, width);
        rebuild_segments_locked();
        upload_device_locked();
        trained_ = !host_data_.empty();
        vram_usage_bytes_ = host_data_.size() * sizeof(KeyValue) + host_segments_.size() * sizeof(HostSegment) + 1024;
        return trained_;
    }

    IndexStats get_stats() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        IndexStats stats;
        stats.query_count = query_count_.load(std::memory_order_relaxed);
        stats.miss_count = miss_count_.load(std::memory_order_relaxed);
        stats.vram_usage_bytes = vram_usage_bytes_;
        stats.gpu_util_percent = trained_ ? 1.0f : 0.0f;

        std::vector<double> latencies;
        {
            std::lock_guard<std::mutex> l(latency_mutex_);
            latencies = latencies_us_;
        }
        if (!latencies.empty()) {
            std::sort(latencies.begin(), latencies.end());
            stats.p50_latency_us = latencies[latencies.size() / 2];
            stats.p99_latency_us = latencies[std::min<std::size_t>(latencies.size() - 1,
                                                                    (latencies.size() * 99) / 100)];
            const double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
            if (sum > 0.0) {
                stats.throughput_qps = static_cast<double>(latencies.size()) * 1'000'000.0 / sum;
            }
        }
        return stats;
    }

    void enable_profiling(bool enabled) override {
        profiling_enabled_.store(enabled, std::memory_order_relaxed);
    }

    std::size_t get_vram_usage() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return vram_usage_bytes_;
    }

private:
    void rebuild_segments_locked() {
        TRACE_EVENT("glfs.lookup", "backend.g_index.rebuild_segments");
        {
            TRACE_EVENT("glfs.lookup", "backend.g_index.rebuild_segments_clear");
            host_segments_.clear();
        }
        if (host_data_.empty()) {
            return;
        }

        const std::size_t width = std::max<std::size_t>(1, segment_width_);
        {
            TRACE_EVENT("glfs.lookup", "backend.g_index.rebuild_segments_generate");
            for (std::size_t begin = 0; begin < host_data_.size(); begin += width) {
                const std::size_t end = std::min(host_data_.size(), begin + width);
                const auto& first = host_data_[begin];
                const auto& last = host_data_[end - 1];
                HostSegment seg;
                seg.first_key = first.key;
                seg.last_key = last.key;
                seg.begin = begin;
                seg.end = end;
                seg.intercept = static_cast<double>(begin);
                if (end > begin + 1 && last.key != first.key) {
                    seg.slope = static_cast<double>(end - begin - 1) / static_cast<double>(last.key - first.key);
                }
                host_segments_.push_back(seg);
            }
        }
    }

    void release_device_locked() {
        TRACE_EVENT("glfs.lookup", "backend.g_index.release_device");
        if (d_keys_) {
            cudaFree(d_keys_);
            d_keys_ = nullptr;
        }
        if (d_values_) {
            cudaFree(d_values_);
            d_values_ = nullptr;
        }
        if (d_segments_) {
            cudaFree(d_segments_);
            d_segments_ = nullptr;
        }
    }

    void upload_device_locked() {
        TRACE_EVENT("glfs.lookup", "backend.g_index.upload_device");
        release_device_locked();
        if (host_data_.empty()) {
            return;
        }

        {
            TRACE_EVENT("glfs.lookup", "backend.g_index.upload_device_alloc");
            cuda_check(cudaMalloc(&d_keys_, host_data_.size() * sizeof(std::uint64_t)), "cudaMalloc(d_keys)");
            cuda_check(cudaMalloc(&d_values_, host_data_.size() * sizeof(std::uint64_t)), "cudaMalloc(d_values)");
            cuda_check(cudaMalloc(&d_segments_, host_segments_.size() * sizeof(DeviceSegment)), "cudaMalloc(d_segments)");
        }

        std::vector<std::uint64_t> keys;
        std::vector<std::uint64_t> values;
        {
            TRACE_EVENT("glfs.lookup", "backend.g_index.upload_device_pack_kv");
            keys.reserve(host_data_.size());
            values.reserve(host_data_.size());
            for (const auto& item : host_data_) {
                keys.push_back(item.key);
                values.push_back(item.value);
            }
        }

        std::vector<DeviceSegment> segments;
        {
            TRACE_EVENT("glfs.lookup", "backend.g_index.upload_device_pack_segments");
            segments.reserve(host_segments_.size());
            for (const auto& s : host_segments_) {
                segments.push_back(DeviceSegment{s.first_key, s.last_key, s.begin, s.end, s.slope, s.intercept});
            }
        }

        {
            TRACE_EVENT("glfs.lookup", "backend.g_index.index_h2d_keys");
            cuda_check(cudaMemcpy(d_keys_, keys.data(), keys.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice),
                       "cudaMemcpy(keys H2D)");
        }
        {
            TRACE_EVENT("glfs.lookup", "backend.g_index.index_h2d_values");
            cuda_check(cudaMemcpy(d_values_, values.data(), values.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice),
                       "cudaMemcpy(values H2D)");
        }
        {
            TRACE_EVENT("glfs.lookup", "backend.g_index.index_h2d_segments");
            cuda_check(cudaMemcpy(d_segments_, segments.data(), segments.size() * sizeof(DeviceSegment), cudaMemcpyHostToDevice),
                       "cudaMemcpy(segments H2D)");
        }
    }

    void record_latency_us(double us) const {
        std::lock_guard<std::mutex> lock(latency_mutex_);
        if (latencies_us_.size() >= std::max<std::size_t>(1, cfg_.latency_history_limit)) {
            latencies_us_.erase(latencies_us_.begin());
        }
        latencies_us_.push_back(us);
    }

    mutable std::mutex mutex_;
    std::vector<KeyValue> host_data_;
    std::vector<HostSegment> host_segments_;
    TrainingConfig cfg_;
    std::size_t segment_width_ = 1;
    bool trained_ = false;
    std::size_t vram_usage_bytes_ = 0;
    std::uint64_t* d_keys_ = nullptr;
    std::uint64_t* d_values_ = nullptr;
    DeviceSegment* d_segments_ = nullptr;
    std::atomic<std::uint64_t> query_count_{0};
    std::atomic<std::uint64_t> miss_count_{0};
    std::atomic<bool> profiling_enabled_{false};
    mutable std::mutex latency_mutex_;
    mutable std::vector<double> latencies_us_;
};

IGPUIndex* create_backend() {
    return new LearnedGIndex();
}

}  // namespace glfs::backends::g_index
