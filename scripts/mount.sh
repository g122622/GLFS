#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
CONFIG_FILE="${ROOT_DIR}/configs/default.json"
MOUNT_POINT="/home/user/data"

usage() {
  cat <<'EOF'
usage: scripts/mount.sh [--config FILE] [--mount DIR] [--build] [--] [fuse args...]

Examples:
  scripts/mount.sh
  scripts/mount.sh --mount /home/user/data -- -f
EOF
}

FUSE_ARGS=()
BUILD_FIRST=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --mount)
      MOUNT_POINT="$2"
      shift 2
      ;;
    --build)
      BUILD_FIRST=1
      shift
      ;;
    --no-build)
      BUILD_FIRST=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        FUSE_ARGS+=("$1")
        shift
      done
      ;;
    *)
      FUSE_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -d "${MOUNT_POINT}" ]]; then
  mkdir -p "${MOUNT_POINT}"
fi

if [[ "${BUILD_FIRST}" -eq 1 ]]; then
  cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DGLFS_BUILD_TESTS=ON -DGLFS_BUILD_BENCHMARKS=ON
  cmake --build "${BUILD_DIR}" -j"$(nproc)"
fi

exec "${BUILD_DIR}/gpufs" --config "${CONFIG_FILE}" --mount "${MOUNT_POINT}" "${FUSE_ARGS[@]}"
