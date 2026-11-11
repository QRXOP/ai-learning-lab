"""Simple local timing and Python-allocation benchmark."""

from __future__ import annotations

import csv
import json
import math
import statistics
import tracemalloc
from pathlib import Path
from time import perf_counter
from typing import Any

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ai_learning_lab.utils.runtime import runtime_metadata


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def run_benchmark(repeats: int = 20, warmup: int = 3) -> dict[str, Any]:
    """Measure Iris inference latency and return metrics and environment metadata.

    Training is excluded from timing. ``tracemalloc`` reports Python-tracked
    allocations, not total process or native-library memory. Tracing adds
    overhead, so latencies are comparable only between runs of this function.
    """
    if not _is_int(repeats) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not _is_int(warmup) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")

    iris = load_iris()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(iris.data, iris.target)

    for _ in range(warmup):
        model.predict(iris.data)

    timings: list[float] = []
    # Leave tracing alone if the caller already started it.
    started_tracing = not tracemalloc.is_tracing()
    if started_tracing:
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        baseline_bytes, _ = tracemalloc.get_traced_memory()
        for _ in range(repeats):
            start = perf_counter()
            model.predict(iris.data)
            timings.append(perf_counter() - start)
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        if started_tracing:
            tracemalloc.stop()

    ordered = sorted(timings)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "model": "standard_scaler_logistic_regression",
        "dataset": "iris",
        "number_of_samples": int(len(iris.data)),
        "repeats": repeats,
        "warmup": warmup,
        "mean_latency_ms": statistics.mean(timings) * 1000,
        "median_latency_ms": statistics.median(timings) * 1000,
        "p95_latency_ms": ordered[p95_index] * 1000,
        "python_tracemalloc_current_mb": (current_bytes - baseline_bytes) / (1024 * 1024),
        "python_tracemalloc_peak_mb": (peak_bytes - baseline_bytes) / (1024 * 1024),
        "measurement_scope": "inference only; model training excluded",
        **runtime_metadata(),
    }


def export_benchmark(
    results: dict[str, Any],
    output_dir: str | Path = ".",
    file_name: str = "benchmark_results",
) -> tuple[Path, Path]:
    """Export benchmark output as JSON and one-row CSV; return both paths."""
    if not isinstance(results, dict) or not results:
        raise ValueError("results must be a non-empty dictionary")
    if (
        not isinstance(file_name, str)
        or not file_name
        or file_name in {".", ".."}
        or Path(file_name).name != file_name
        or "\\" in file_name
    ):
        raise ValueError("file_name must be a plain filename stem without path separators")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / f"{file_name}.json"
    csv_path = destination / f"{file_name}.csv"

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=2, sort_keys=True)
        json_file.write("\n")

    flattened = {
        key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in results.items()
    }
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(flattened))
        writer.writeheader()
        writer.writerow(flattened)
    return json_path, csv_path
