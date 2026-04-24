#!/usr/bin/env python3
"""Generate a small synthetic dataset tree for the GPULearnedFS prototype."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/user/data")
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    root = Path(args.root)
    (root / "train" / "img").mkdir(parents=True, exist_ok=True)
    (root / "train" / "text").mkdir(parents=True, exist_ok=True)
    for i in range(args.count):
        (root / "train" / "img" / f"file_{i:05d}.jpg").write_text("x", encoding="utf-8")
    (root / "train" / "text" / "readme.txt").write_text("glfs", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
