#include "core/gpu_index_adapter.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

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
    if (!rt.initialized) {
        if (cudaStreamCreate(&rt.stream) != cudaSuccess) {
            throw std::runtime_error("failed to create CUDA stream");
        }
        rt.initialized = true;
    }
}

void launch_noop() {
    noop_kernel<<<1, 1>>>();
}

}  // namespace

extern "C" void glfs_launch_noop_kernel() {
    launch_noop();
}

class CUDAQueueControlPlane final : public IGPUControlPlane {
public:
    void initialize(const std::string& index_type) override {
        std::lock_guard<std::mutex> lock(mutex_);
        ensure_stream(rt_);
        if (!index_) {
            index_.reset(create_index(index_type));
        }
        index_type_ = index_type.empty() ? "g-index" : index_type;
    }

    void train(const std::vector<std::uint64_t>& keys,
               const std::vector<std::uint64_t>& values,
               const TrainingConfig& cfg) override {
        std::lock_guard<std::mutex> lock(mutex_);
        ensure_stream(rt_);
        enqueue_locked(GPURequest{GPURequest::Kind::TrainBatch, keys, values, cfg});
        process_locked();
    }

    ControlResult lookup(std::uint64_t key) override {
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
        std::lock_guard<std::mutex> lock(mutex_);
        ensure_stream(rt_);

        std::vector<ControlResult> results;
        results.reserve(keys.size());

        if (!index_) {
            for (auto key : keys) {
                ControlResult result;
                result.fallback_to_backing_root = true;
                result.reason = "control_plane_uninitialized";
                result.inode = INVALID_INODE;
                results.push_back(std::move(result));
                ++rt_.fallback_requests;
                ++rt_.queued_requests;
            }
            return results;
        }

        enqueue_locked(GPURequest{GPURequest::Kind::LookupBatch, keys, {}, {}});
        process_locked();

        auto values = index_->batch_lookup(keys, rt_.stream);
        for (auto value : values) {
            ControlResult result;
            if (value != INVALID_INODE) {
                result.inode = value;
                result.reason = "control_plane_hit";
                ++rt_.completed_requests;
            } else {
                result.fallback_to_backing_root = true;
                result.reason = "control_plane_miss";
                ++rt_.fallback_requests;
            }
            results.push_back(std::move(result));
        }
        return results;
    }

    MutationDecision decide_mutation(const MutationRequest& request) override {
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
        std::lock_guard<std::mutex> lock(mutex_);
        ensure_stream(rt_);
        enqueue_locked(GPURequest{GPURequest::Kind::LookupBatch, keys, {}, {}});
    }

    void drain() override {
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
    void enqueue_locked(GPURequest request) {
        rt_.queue.push_back(std::move(request));
        ++rt_.queued_requests;
    }

    void process_locked() {
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
                case GPURequest::Kind::TrainBatch:
                    if (!index_) {
                        index_.reset(create_index(request.cfg.index_type));
                    }
                    index_->train(request.keys, request.values, request.cfg);
                    rt_.completed_requests += static_cast<std::uint64_t>(request.keys.size());
                    break;
                case GPURequest::Kind::LookupBatch:
                    if (index_) {
                        (void)index_->batch_lookup(request.keys, rt_.stream);
                    } else {
                        rt_.fallback_requests += static_cast<std::uint64_t>(request.keys.size());
                    }
                    break;
                case GPURequest::Kind::Drain:
                    ++rt_.batch_flushes;
                    break;
            }
            rt_.state = rt_.queue.empty() ? GPUControlPlaneRuntime::State::Idle
                                          : GPUControlPlaneRuntime::State::Running;
        }
        if (rt_.profiling_enabled) {
            launch_noop();
        }
    }

    mutable std::mutex mutex_;
    std::unique_ptr<IGPUIndex> index_;
    std::string index_type_ = "g-index";
    GPUControlPlaneRuntime& rt_ = runtime();
};

IGPUControlPlane* create_control_plane(const std::string& type) {
    auto* cp = new CUDAQueueControlPlane();
    cp->initialize(type);
    return cp;
}

}  // namespace glfs
