#!/usr/bin/env bash
set -euo pipefail

MOUNT_POINT="${1:-/home/user/data}"

if command -v fusermount3 >/dev/null 2>&1; then
  fusermount3 -u "${MOUNT_POINT}" || true
else
  fusermount -u "${MOUNT_POINT}" || true
fi
