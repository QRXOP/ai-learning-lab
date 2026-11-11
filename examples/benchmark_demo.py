"""Time Iris inference locally and write JSON and CSV results.

Results depend on the machine, its load, and installed versions. Compare runs
only under controlled conditions.
"""

from ai_learning_lab.benchmarking import export_benchmark, run_benchmark


def main() -> None:
    result = run_benchmark(repeats=20, warmup=3)
    print(f"mean   latency: {result['mean_latency_ms']:.3f} ms")
    print(f"median latency: {result['median_latency_ms']:.3f} ms")
    print(f"p95    latency: {result['p95_latency_ms']:.3f} ms")
    print(f"peak Python allocations: {result['python_tracemalloc_peak_mb']:.3f} MB")

    json_path, csv_path = export_benchmark(result)
    print(f"Wrote {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
