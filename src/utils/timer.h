#pragma once

#include <chrono>
#include <cstdint>

namespace glfs {

inline std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline std::int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

template <typename Fn>
auto measure_ns(Fn&& fn) {
    const auto start = std::chrono::steady_clock::now();
    auto result = fn();
    const auto end = std::chrono::steady_clock::now();
    const auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    return std::pair{dur, std::move(result)};
}

}  // namespace glfs
