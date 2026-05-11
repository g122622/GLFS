#!/usr/bin/env python3
"""Plot GLFS benchmark results using config-defined CSV and image paths."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OPERATIONS = ["stat_getattr", "readdir", "open", "read_lookup"]
METRIC_FIELDS = ["p50_us", "p99_us", "p999_us", "throughput_qps", "query_count", "miss_count", "gpu_util_percent", "vram_usage_bytes"]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def sort_path_key(path: str) -> tuple[int, str]:
    stem = Path(path).name
    digits = "".join(ch for ch in stem if ch.isdigit())
    return (int(digits) if digits else 10**18, path)


def aggregate_by_path(rows: list[dict[str, str]], backend: str, operation: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row["backend"] == backend and row["operation"] == operation]
    selected.sort(key=lambda row: sort_path_key(row["path"]))
    return selected


def plot_traversal_latency(rows: list[dict[str, str]], output_path: Path) -> None:
    backends = sorted({row["backend"] for row in rows})
    fig, axes = plt.subplots(len(OPERATIONS), 1, figsize=(15, 18), sharex=False)
    if len(OPERATIONS) == 1:
        axes = [axes]

    for ax, operation in zip(axes, OPERATIONS):
        for backend in backends:
            selected = aggregate_by_path(rows, backend, operation)
            if not selected:
                continue
            x = list(range(len(selected)))
            p50 = [float(row["p50_us"]) for row in selected]
            p99 = [float(row["p99_us"]) for row in selected]
            p999 = [float(row["p999_us"]) for row in selected]
            ax.plot(x, p50, label=f"{backend} p50", linewidth=1.4)
            ax.plot(x, p99, label=f"{backend} p99", linewidth=1.0, alpha=0.75)
            ax.plot(x, p999, label=f"{backend} p999", linewidth=0.9, alpha=0.55)
            ax.fill_between(x, p50, p99, alpha=0.08)
        ax.set_title(f"Traversal-order latency: {operation}")
        ax.set_ylabel("latency (us)")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        if rows:
            sample = aggregate_by_path(rows, backends[0], operation)
            if sample:
                tick_step = max(1, len(sample) // 10)
                tick_positions = list(range(0, len(sample), tick_step))
                tick_labels = [sample[i]["path"] for i in tick_positions]
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8)

    axes[-1].set_xlabel("path traversal order")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    backends = sorted({row["backend"] for row in rows})
    operations = [op for op in OPERATIONS if any(row["operation"] == op for row in rows)]
    if not backends or not operations:
        raise RuntimeError("benchmark CSV has no plottable rows")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = [ax for row in axes for ax in row]

    for ax, metric in zip(axes, ["p50_us", "p99_us", "throughput_qps", "gpu_util_percent"]):
        width = 0.8 / max(len(backends), 1)
        x_positions = list(range(len(operations)))
        for idx, backend in enumerate(backends):
            values = []
            for op in operations:
                subset = [float(row[metric]) for row in rows if row["backend"] == backend and row["operation"] == op]
                values.append(sum(subset) / len(subset) if subset else 0.0)
            offsets = [x + (idx - (len(backends) - 1) / 2) * width for x in x_positions]
            ax.bar(offsets, values, width=width, label=backend)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(operations, rotation=15)
        ax.set_title(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_resource_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    backends = sorted({row["backend"] for row in rows})
    operations = [op for op in OPERATIONS if any(row["operation"] == op for row in rows)]
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for ax, metric in zip(axes, ["miss_count", "vram_usage_bytes"]):
        width = 0.8 / max(len(backends), 1)
        x_positions = list(range(len(operations)))
        for idx, backend in enumerate(backends):
            values = []
            for op in operations:
                subset = [float(row[metric]) for row in rows if row["backend"] == backend and row["operation"] == op]
                values.append(sum(subset) / len(subset) if subset else 0.0)
            offsets = [x + (idx - (len(backends) - 1) / 2) * width for x in x_positions]
            ax.bar(offsets, values, width=width, label=backend)
        ax.set_title(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xticks(list(range(len(operations))))
    axes[-1].set_xticklabels(operations, rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot GLFS benchmark results")
    parser.add_argument("--config", required=True, help="Path to the config JSON")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")

    cfg = load_config(config_path)
    benchmark = cfg["benchmark"]
    csv_path = Path(benchmark["report_csv_path"])
    combined_output_path = Path(benchmark["report_plot_path"])
    traversal_output_path = Path(benchmark["report_plot_paths"]["traversal_latency_path"])
    resource_output_path = Path(benchmark["report_plot_paths"]["resource_summary_path"])

    if not csv_path.is_file():
        raise FileNotFoundError(f"csv file not found: {csv_path}")

    rows = load_rows(csv_path)
    plot_traversal_latency(rows, traversal_output_path)
    plot_metric_summary(rows, combined_output_path)
    plot_resource_summary(rows, resource_output_path)
    print(f"wrote {traversal_output_path}")
    print(f"wrote {combined_output_path}")
    print(f"wrote {resource_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
