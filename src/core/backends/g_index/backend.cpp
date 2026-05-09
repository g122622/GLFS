#include "core/gpu_index_adapter.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <numeric>
#include <shared_mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace glfs::backends::g_index {

namespace {

struct KeyValue {
	std::uint64_t key = 0;
	std::uint64_t value = INVALID_INODE;
};

struct Segment {
	std::uint64_t first_key = 0;
	std::uint64_t last_key = 0;
	std::size_t begin = 0;
	std::size_t end = 0;  // one past the last element covered by the segment
	double slope = 0.0;
	double intercept = 0.0;
};

std::size_t choose_segment_width(std::size_t n, const TrainingConfig& cfg) {
	if (n == 0) {
		return 1;
	}

	const float ratio = std::clamp(cfg.sample_ratio, 0.05f, 1.0f);
	const std::size_t epoch_scale = std::max<std::size_t>(1, cfg.max_epochs);

	std::size_t width = static_cast<std::size_t>(std::round(256.0f / ratio));
	width /= std::min<std::size_t>(epoch_scale, 8);
	width = std::clamp(width, std::size_t{32}, std::size_t{4096});
	return std::min(width, n);
}

std::vector<KeyValue> compress_sorted_pairs(std::vector<KeyValue> items) {
	std::vector<KeyValue> compressed;
	compressed.reserve(items.size());
	for (const auto& item : items) {
		if (!compressed.empty() && compressed.back().key == item.key) {
			compressed.back().value = item.value;
		} else {
			compressed.push_back(item);
		}
	}
	return compressed;
}

}  // namespace

class LearnedGIndex final : public IGPUIndex {
public:
	LearnedGIndex() = default;

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

		std::vector<KeyValue> items;
		items.reserve(keys.size());
		for (std::size_t i = 0; i < keys.size(); ++i) {
			items.push_back(KeyValue{keys[i], values[i]});
		}

		std::stable_sort(items.begin(), items.end(), [](const KeyValue& a, const KeyValue& b) {
			if (a.key != b.key) {
				return a.key < b.key;
			}
			return a.value < b.value;
		});

		items = compress_sorted_pairs(std::move(items));

		std::unique_lock<std::shared_mutex> lock(state_mutex_);
		data_ = std::move(items);
		training_cfg_ = cfg;
		segment_width_ = choose_segment_width(data_.size(), cfg);
		rebuild_segments_locked();
		trained_ = true;

		const auto data_bytes = data_.size() * sizeof(KeyValue);
		const auto segment_bytes = segments_.size() * sizeof(Segment);
		vram_usage_bytes_ = data_bytes + segment_bytes + 1024;
	}

	std::vector<std::uint64_t> batch_lookup(const std::vector<std::uint64_t>& keys,
											cudaStream_t) override {
		std::vector<std::uint64_t> out;
		out.reserve(keys.size());

		std::shared_lock<std::shared_mutex> lock(state_mutex_);
		for (const auto key : keys) {
			const auto start = std::chrono::steady_clock::now();
			const auto value = lookup_one_locked(key);
			const auto end = std::chrono::steady_clock::now();
			if (profiling_enabled_.load(std::memory_order_relaxed)) {
				record_latency_us(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0);
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
		std::shared_lock<std::shared_mutex> lock(state_mutex_);
		std::ofstream out(filepath, std::ios::trunc);
		if (!out) {
			return false;
		}

		out << "GLFS_G_INDEX_V1\n";
		out << training_cfg_.index_type << '\n';
		out << training_cfg_.sample_ratio << ' ' << training_cfg_.max_epochs << ' '
			<< training_cfg_.max_vram_mb << '\n';
		out << segment_width_ << '\n';
		out << data_.size() << '\n';
		for (const auto& item : data_) {
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
		std::string index_type;
		TrainingConfig cfg;
		std::size_t width = 0;
		std::size_t size = 0;

		if (!std::getline(in, magic) || magic != "GLFS_G_INDEX_V1") {
			return false;
		}
		if (!std::getline(in, index_type)) {
			return false;
		}
		cfg.index_type = index_type;
		if (!(in >> cfg.sample_ratio >> cfg.max_epochs >> cfg.max_vram_mb)) {
			return false;
		}
		if (!(in >> width)) {
			return false;
		}
		if (!(in >> size)) {
			return false;
		}

		std::vector<KeyValue> items;
		items.reserve(size);
		for (std::size_t i = 0; i < size; ++i) {
			KeyValue item{};
			if (!(in >> item.key >> item.value)) {
				return false;
			}
			items.push_back(item);
		}

		std::unique_lock<std::shared_mutex> lock(state_mutex_);
		data_ = std::move(items);
		training_cfg_ = cfg;
		segment_width_ = std::max<std::size_t>(1, width);
		rebuild_segments_locked();
		trained_ = !data_.empty();
		const auto data_bytes = data_.size() * sizeof(KeyValue);
		const auto segment_bytes = segments_.size() * sizeof(Segment);
		vram_usage_bytes_ = data_bytes + segment_bytes + 1024;
		return trained_;
	}

	IndexStats get_stats() const override {
		std::shared_lock<std::shared_mutex> lock(state_mutex_);
		IndexStats stats;
		stats.query_count = query_count_.load(std::memory_order_relaxed);
		stats.miss_count = miss_count_.load(std::memory_order_relaxed);
		stats.vram_usage_bytes = vram_usage_bytes_;
		stats.gpu_util_percent = trained_ ? 1.0f : 0.0f;

		std::vector<double> latencies;
		{
			std::lock_guard<std::mutex> latency_lock(latency_mutex_);
			latencies = latencies_us_;
		}

		if (!latencies.empty()) {
			std::sort(latencies.begin(), latencies.end());
			stats.p50_latency_us = latencies[latencies.size() / 2];
			stats.p99_latency_us = latencies[std::min<std::size_t>(latencies.size() - 1,
																   (latencies.size() * 99) / 100)];
			const auto total_us = std::accumulate(latencies.begin(), latencies.end(), 0.0);
			if (total_us > 0.0) {
				stats.throughput_qps = static_cast<double>(latencies.size()) * 1'000'000.0 / total_us;
			}
		}

		return stats;
	}

	void enable_profiling(bool enabled) override {
		profiling_enabled_.store(enabled, std::memory_order_relaxed);
	}

	std::size_t get_vram_usage() const override {
		std::shared_lock<std::shared_mutex> lock(state_mutex_);
		return vram_usage_bytes_;
	}

private:
	void rebuild_segments_locked() {
		segments_.clear();
		segment_keys_.clear();

		if (data_.empty()) {
			return;
		}

		const std::size_t width = std::max<std::size_t>(1, segment_width_);
		for (std::size_t begin = 0; begin < data_.size(); begin += width) {
			const std::size_t end = std::min(data_.size(), begin + width);
			const auto& first = data_[begin];
			const auto& last = data_[end - 1];

			Segment seg;
			seg.first_key = first.key;
			seg.last_key = last.key;
			seg.begin = begin;
			seg.end = end;
			seg.intercept = static_cast<double>(begin);
			if (last.key != first.key && end > begin + 1) {
				seg.slope = static_cast<double>(end - begin - 1) / static_cast<double>(last.key - first.key);
			} else {
				seg.slope = 0.0;
			}

			segment_keys_.push_back(seg.first_key);
			segments_.push_back(seg);
		}
	}

	std::uint64_t lookup_one_locked(std::uint64_t key) const {
		if (data_.empty() || segments_.empty()) {
			return INVALID_INODE;
		}

		const auto segment_it = std::upper_bound(segment_keys_.begin(), segment_keys_.end(), key);
		const std::size_t segment_index = segment_it == segment_keys_.begin()
											  ? 0
											  : static_cast<std::size_t>(segment_it - segment_keys_.begin() - 1);
		const auto& seg = segments_[segment_index];

		double predicted = seg.intercept;
		if (key > seg.first_key) {
			predicted = seg.slope * static_cast<double>(key - seg.first_key) + seg.intercept;
		}

		std::size_t approx = 0;
		if (predicted > 0.0) {
			approx = static_cast<std::size_t>(predicted);
		}
		approx = std::clamp(approx, seg.begin, seg.end - 1);

		constexpr std::size_t kWindow = 16;
		const std::size_t lo = approx > kWindow ? std::max(seg.begin, approx - kWindow) : seg.begin;
		const std::size_t hi = std::min(seg.end, approx + kWindow + 1);

		auto begin_it = data_.begin() + static_cast<std::ptrdiff_t>(lo);
		auto end_it = data_.begin() + static_cast<std::ptrdiff_t>(hi);
		auto it = std::lower_bound(begin_it, end_it, key, [](const KeyValue& item, std::uint64_t needle) {
			return item.key < needle;
		});
		if (it != end_it && it->key == key) {
			return it->value;
		}

		begin_it = data_.begin() + static_cast<std::ptrdiff_t>(seg.begin);
		end_it = data_.begin() + static_cast<std::ptrdiff_t>(seg.end);
		it = std::lower_bound(begin_it, end_it, key, [](const KeyValue& item, std::uint64_t needle) {
			return item.key < needle;
		});
		if (it != end_it && it->key == key) {
			return it->value;
		}
		return INVALID_INODE;
	}

	void record_latency_us(double us) const {
		std::lock_guard<std::mutex> lock(latency_mutex_);
		if (latencies_us_.size() >= 2048) {
			latencies_us_.erase(latencies_us_.begin());
		}
		latencies_us_.push_back(us);
	}

	mutable std::shared_mutex state_mutex_;
	std::vector<KeyValue> data_;
	std::vector<Segment> segments_;
	std::vector<std::uint64_t> segment_keys_;
	TrainingConfig training_cfg_;
	std::size_t segment_width_ = 1;
	bool trained_ = false;
	std::size_t vram_usage_bytes_ = 0;

	std::atomic<std::uint64_t> query_count_{0};
	std::atomic<std::uint64_t> miss_count_{0};
	std::atomic<bool> profiling_enabled_{false};

	mutable std::mutex latency_mutex_;
	mutable std::vector<double> latencies_us_;
};

const char* backend_name() {
	return "g-index";
}

IGPUIndex* create_backend() {
	return new LearnedGIndex();
}

}  // namespace glfs::backends::g_index
