#include "core/gpu_index_adapter.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <mutex>
#include <stdexcept>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "utils/perfetto_integration.h"

namespace glfs {

namespace {

__global__ void noop_kernel() {}

struct GPURequest {
    enum class Kind {
        LookupBatch,
        TrainBatch,
        Drain
    };

    Kind kind = Kind::Drain;
    std::vector<std::uint64_t> keys;
    std::vector<std::uint64_t> values;
    TrainingConfig cfg;
};

struct PendingLookup {
    std::vector<std::uint64_t> keys;
    std::promise<std::vector<std::uint64_t>> promise;
};

struct GPUControlPlaneRuntime {
    enum class State {
        Idle,
        Running,
        Draining,
        Faulted
    };

    ::cudaStream_t stream = nullptr;
    std::deque<GPURequest> queue;
    State state = State::Idle;
    bool initialized = false;
    bool profiling_enabled = false;
    std::uint64_t queued_requests = 0;
    std::uint64_t completed_requests = 0;
    std::uint64_t fallback_requests = 0;
    std::uint64_t batch_flushes = 0;
};

GPUControlPlaneRuntime& runtime() {
    static GPUControlPlaneRuntime rt;
    return rt;
}

void ensure_stream(GPUControlPlaneRuntime& rt) {
    TRACE_EVENT("glfs.lookup", "cuda_control_plane.ensure_stream");
    if (!rt.initialized) {
        if (cudaStreamCreate(&rt.stream) != cudaSuccess) {
            throw std::runtime_error("failed to create CUDA stream");
        }
        rt.initialized = true;
    }
}

void launch_noop() {
    TRACE_EVENT("glfs.lookup", "cuda_control_plane.launch_noop");
    noop_kernel<<<1, 1>>>();
}

}  // namespace

extern "C" void glfs_launch_noop_kernel() {
    launch_noop();
}

class CUDAQueueControlPlane final : public IGPUControlPlane {
public:
    void initialize(const std::string& index_type,
                    const TrainingConfig& training_cfg,
                    std::uint32_t inference_batch_size,
                    std::uint32_t inference_batch_timeout_us) override {
        TRACE_EVENT("glfs.lookup", "cuda_control_plane.initialize");
        std::lock_guard<std::mutex> lock(mutex_);
        if (index_type.empty()) {
            throw std::invalid_argument("index_type must not be empty");
        }
        if (inference_batch_size == 0) {
            throw std::invalid_argument("inference_batch_size must be > 0");
        }
        if (inference_batch_timeout_us == 0) {
            throw std::invalid_argument("inference_batch_timeout_us must be > 0");
        }
        {
            TRACE_EVENT("glfs.lookup", "cuda_control_plane.initialize_stream");
            ensure_stream(rt_);
        }
        if (!index_) {
            TRACE_EVENT("glfs.lookup", "cuda_control_plane.initialize_index");
            index_.reset(create_index(index_type));
        }
        index_type_ = index_type;
        training_cfg_ = training_cfg;
        inference_batch_size_ = inference_batch_size;
        inference_batch_timeout_ = std::chrono::microseconds(inference_batch_timeout_us);
    }

    void train(const std::vector<std::uint64_t>& keys,
               const std::vector<std::uint64_t>& values,
               const TrainingConfig& cfg) override {
        TRACE_EVENT("glfs.lookup", "cuda_control_plane.train");
        std::lock_guard<std::mutex> lock(mutex_);
        {
            TRACE_EVENT("glfs.lookup", "cuda_control_plane.train_ensure_stream");
            ensure_stream(rt_);
        }
        {
            TRACE_EVENT("glfs.lookup", "cuda_control_plane.train_enqueue");
            enqueue_locked(GPURequest{GPURequest::Kind::TrainBatch, keys, values, cfg});
        }
        {
            TRACE_EVENT("glfs.lookup", "cuda_control_plane.train_process");
            process_locked();
        }
    }

    ControlResult lookup(std::uint64_t key) override {
        TRACE_EVENT("glfs.lookup", "cuda_control_plane.lookup_single");
        ControlResult result;
        auto batch = lookup_batch({key});
        if (!batch.empty()) {
            return batch.front();
        }
        result.fallback_to_backing_root = true;
        result.reason = "control_plane_empty_batch";
        return result;
    }

    std::vector<ControlResult> lookup_batch(const std::vector<std::uint64_t>& keys) override {
        TRACE_EVENT("glfs.lookup", "cuda_control_plane.lookup_batch");
        std::vector<ControlResult> results;
        results.reserve(keys.size());

        auto values = dispatch_lookup_batch(keys);
        for (auto value : values) {
            ControlResult result;
            if (value != INVALID_INODE) {
                result.inode = value;
                result.reason = "control_plane_hit";
            } else {
                result.fallback_to_backing_root = true;
                result.reason = "control_plane_miss";
            }
            results.push_back(std::move(result));
        }
        return results;
    }

    MutationDecision decide_mutation(const MutationRequest& request) override {
        TRACE_EVENT("glfs.lookup", "cuda_control_plane.decide_mutation");
        std::lock_guard<std::mutex> lock(mutex_);
        ensure_stream(rt_);
        MutationDecision decision;
        enqueue_locked(GPURequest{GPURequest::Kind::Drain, {}, {}, {}});
        process_locked();

        ++rt_.queued_requests;
        switch (request.op) {
            case MutationOp::Create:
            case MutationOp::Mkdir:
                decision.allowed = true;
                decision.reason = "mutation_allowed";
                ++rt_.completed_requests;
                break;
            case MutationOp::Unlink:
                decision.allowed = true;
                decision.reason = "mutation_allowed";
                ++rt_.completed_requests;
                break;
            case MutationOp::Rename:
                if (request.has_secondary) {
                    decision.allowed = true;
                    decision.reason = "mutation_allowed";
                    ++rt_.completed_requests;
                } else {
                    decision.allowed = false;
                    decision.fallback_to_backing_root = true;
                    decision.reason = "rename_requires_target";
                    ++rt_.fallback_requests;
                }
                break;
            case MutationOp::Truncate:
                decision.allowed = true;
                decision.reason = "mutation_allowed";
                ++rt_.completed_requests;
                break;
        }
        return decision;
    }

    void submit_lookup_batch(const std::vector<std::uint64_t>& keys) override {
        TRACE_EVENT("glfs.lookup", "cuda_control_plane.submit_lookup_batch");
        std::lock_guard<std::mutex> lock(mutex_);
        ensure_stream(rt_);
        enqueue_locked(GPURequest{GPURequest::Kind::LookupBatch, keys, {}, {}});
    }

    void set_namespace(const std::map<std::string, std::vector<std::string>>& children) override {
        TRACE_EVENT("glfs.lookup", "cuda_control_plane.set_namespace");
        std::lock_guard<std::mutex> lock(mutex_);
        namespace_children_ = children;
        for (auto& [_, child_list] : namespace_children_) {
            std::sort(child_list.begin(), child_list.end());
            child_list.erase(std::unique(child_list.begin(), child_list.end()), child_list.end());
        }
    }

    std::vector<std::string> list_children(const std::string& path) const override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = namespace_children_.find(path);
        if (it == namespace_children_.end()) {
            return {};
        }
        return it->second;
    }

    void drain() override {
        TRACE_EVENT("glfs.lookup", "cuda_control_plane.drain");
        std::lock_guard<std::mutex> lock(mutex_);
        ensure_stream(rt_);
        enqueue_locked(GPURequest{GPURequest::Kind::Drain, {}, {}, {}});
        process_locked();
    }

    IndexStats get_stats() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return index_ ? index_->get_stats() : IndexStats{};
    }

    ControlPlaneStats get_control_plane_stats() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        ControlPlaneStats stats;
        stats.queued_requests = rt_.queued_requests;
        stats.completed_requests = rt_.completed_requests;
        stats.fallback_requests = rt_.fallback_requests;
        return stats;
    }

    void enable_profiling(bool enabled) override {
        std::lock_guard<std::mutex> lock(mutex_);
        rt_.profiling_enabled = enabled;
        if (index_) {
            index_->enable_profiling(enabled);
        }
    }

    IGPUIndex* index() override {
        std::lock_guard<std::mutex> lock(mutex_);
        return index_.get();
    }

private:
    std::vector<std::uint64_t> dispatch_lookup_batch(const std::vector<std::uint64_t>& keys) {
        std::future<std::vector<std::uint64_t>> future;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ensure_stream(rt_);

            if (!index_) {
                std::vector<std::uint64_t> out(keys.size(), INVALID_INODE);
                rt_.fallback_requests += static_cast<std::uint64_t>(keys.size());
                rt_.queued_requests += static_cast<std::uint64_t>(keys.size());
                return out;
            }

            PendingLookup pending;
            pending.keys = keys;
            future = pending.promise.get_future();
            pending_lookups_.push_back(std::move(pending));
            rt_.queued_requests += static_cast<std::uint64_t>(keys.size());
            if (!batch_window_open_) {
                batch_window_open_ = true;
                batch_window_start_ = std::chrono::steady_clock::now();
            }

            if (pending_lookup_key_count_locked() >= inference_batch_size_) {
                flush_pending_lookups_locked();
            } else {
                const auto deadline = batch_window_start_ + inference_batch_timeout_;
                while (future.wait_for(std::chrono::microseconds(0)) != std::future_status::ready) {
                    if (pending_lookup_key_count_locked() >= inference_batch_size_) {
                        flush_pending_lookups_locked();
                        break;
                    }
                    if (pending_lookups_.empty()) {
                        break;
                    }
                    if (batch_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
                        flush_pending_lookups_locked();
                        break;
                    }
                }
            }
        }
        return future.get();
    }

    std::size_t pending_lookup_key_count_locked() const {
        std::size_t total = 0;
        for (const auto& pending : pending_lookups_) {
            total += pending.keys.size();
        }
        return total;
    }

    void flush_pending_lookups_locked() {
        if (pending_lookups_.empty()) {
            batch_window_open_ = false;
            batch_cv_.notify_all();
            return;
        }

        std::vector<PendingLookup> pending = std::move(pending_lookups_);
        pending_lookups_.clear();
        batch_window_open_ = false;

        std::vector<std::uint64_t> merged_keys;
        merged_keys.reserve(pending_lookup_key_count(pending));
        std::vector<std::size_t> offsets;
        offsets.reserve(pending.size());
        for (const auto& entry : pending) {
            offsets.push_back(merged_keys.size());
            merged_keys.insert(merged_keys.end(), entry.keys.begin(), entry.keys.end());
        }

        auto values = index_->batch_lookup(merged_keys, rt_.stream);
        for (std::size_t i = 0; i < pending.size(); ++i) {
            const auto begin = offsets[i];
            const auto end = begin + pending[i].keys.size();
            std::vector<std::uint64_t> slice(values.begin() + static_cast<std::ptrdiff_t>(begin),
                                             values.begin() + static_cast<std::ptrdiff_t>(end));
            for (auto value : slice) {
                if (value != INVALID_INODE) {
                    ++rt_.completed_requests;
                } else {
                    ++rt_.fallback_requests;
                }
            }
            pending[i].promise.set_value(std::move(slice));
        }
        batch_cv_.notify_all();
    }

    static std::size_t pending_lookup_key_count(const std::vector<PendingLookup>& pending) {
        std::size_t total = 0;
        for (const auto& entry : pending) {
            total += entry.keys.size();
        }
        return total;
    }

    void enqueue_locked(GPURequest request) {
        TRACE_EVENT("glfs.lookup", "cuda_control_plane.enqueue_locked");
        rt_.queue.push_back(std::move(request));
        ++rt_.queued_requests;
    }

    void process_locked() {
        TRACE_EVENT("glfs.lookup", "cuda_control_plane.process_locked");
        if (rt_.state == GPUControlPlaneRuntime::State::Faulted) {
            return;
        }
        rt_.state = rt_.queue.empty() ? GPUControlPlaneRuntime::State::Idle
                                      : GPUControlPlaneRuntime::State::Running;
        while (!rt_.queue.empty()) {
            const auto request = std::move(rt_.queue.front());
            rt_.queue.pop_front();
            rt_.state = GPUControlPlaneRuntime::State::Running;
            switch (request.kind) {
                case GPURequest::Kind::TrainBatch: {
                    TRACE_EVENT("glfs.lookup", "cuda_control_plane.process_train_batch");
                    if (!index_) {
                        if (request.cfg.index_type.empty()) {
                            throw std::runtime_error("missing index_type for train request");
                        }
                        index_.reset(create_index(request.cfg.index_type));
                    }
                    index_->train(request.keys, request.values, request.cfg);
                    rt_.completed_requests += static_cast<std::uint64_t>(request.keys.size());
                    break;
                }
                case GPURequest::Kind::LookupBatch: {
                    TRACE_EVENT("glfs.lookup", "cuda_control_plane.process_lookup_batch");
                    if (!index_) {
                        rt_.fallback_requests += static_cast<std::uint64_t>(request.keys.size());
                    }
                    break;
                }
                case GPURequest::Kind::Drain: {
                    TRACE_EVENT("glfs.lookup", "cuda_control_plane.process_drain");
                    ++rt_.batch_flushes;
                    break;
                }
            }
            rt_.state = rt_.queue.empty() ? GPUControlPlaneRuntime::State::Idle
                                          : GPUControlPlaneRuntime::State::Running;
        }
        if (rt_.profiling_enabled) {
            launch_noop();
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable batch_cv_;
    std::unique_ptr<IGPUIndex> index_;
    std::string index_type_;
    TrainingConfig training_cfg_{};
    std::uint32_t inference_batch_size_ = 0;
    std::chrono::microseconds inference_batch_timeout_{};
    bool batch_window_open_ = false;
    std::chrono::steady_clock::time_point batch_window_start_{};
    std::vector<PendingLookup> pending_lookups_;
    GPUControlPlaneRuntime& rt_ = runtime();
    std::map<std::string, std::vector<std::string>> namespace_children_;
};

IGPUControlPlane* create_control_plane(const std::string& type,
                                       const TrainingConfig& training_cfg,
                                       std::uint32_t inference_batch_size,
                                       std::uint32_t inference_batch_timeout_us) {
    auto* cp = new CUDAQueueControlPlane();
    cp->initialize(type, training_cfg, inference_batch_size, inference_batch_timeout_us);
    return cp;
}

}  // namespace glfs
