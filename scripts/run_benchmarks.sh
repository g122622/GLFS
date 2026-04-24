#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-build}"
config="${2:-configs/default.json}"

cmake -S . -B "$build_dir" -DGLFS_BUILD_TESTS=ON
cmake --build "$build_dir" -j"$(nproc)"
"$build_dir/gpufs" "$config"
