#!/usr/bin/env python3
"""Placeholder results visualizer for prototype outputs."""

from __future__ import annotations

import csv
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: plot_results.py <csv>")
        return 1
    with open(sys.argv[1], newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    print(f"loaded {max(len(rows) - 1, 0)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
