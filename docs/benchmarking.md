# Benchmarking methodology

`ai_learning_lab.benchmarking.run_benchmark` is a small teaching example of how to measure inference latency. It is not a general-purpose benchmark suite.

## What it measures

1. Loads scikit-learn's bundled Iris dataset (150 samples, 4 features).
2. Fits `StandardScaler` + `LogisticRegression(random_state=42)` on all of it. **Training is not timed.**
3. Runs `warmup` untimed `predict` calls over all 150 samples to reduce first-call effects.
4. Times `repeats` calls to `predict` over all 150 samples with `time.perf_counter`.
5. Reports mean, median, and p95 latency in milliseconds. p95 uses the nearest-rank method on the sorted timings; with few repeats it is close to the maximum.
6. Reports `tracemalloc` current and peak bytes during the timed loop, relative to a baseline taken just before it.
7. Records platform, Python version, and scikit-learn/NumPy versions.

Each latency is for **one batch of 150 rows**, not per row.

## Limitations

- **Machine-dependent.** CPU, power settings, background load, BLAS backend, and library versions all change results. Compare numbers only between runs on the same machine under similar conditions.
- **Tracing overhead.** `tracemalloc` is active during the timed loop, which slows Python code. Latencies are comparable between runs of this function, not with untraced measurements.
- **Partial memory picture.** `tracemalloc` sees Python-level allocations only. Memory allocated by native code (NumPy/BLAS buffers) or by accelerators is not included. Use OS-level tools for total process memory.
- **Small sample.** The default 20 repeats gives a rough estimate. Increase `repeats` and report variance before drawing conclusions.
- **Tiny workload.** Iris inference takes well under a millisecond on typical hardware, so timer resolution and interpreter overhead are a large share of the result.

## Exported files

`export_benchmark(result, output_dir, file_name)` writes:

- `<file_name>.json`: the full result, with sorted keys.
- `<file_name>.csv`: one header row and one data row; nested values (`hardware`, `dependencies`) are JSON-encoded strings.

`file_name` must be a plain stem without path separators. Output files are listed in `.gitignore` for the default name and the `results/` directory; avoid committing machine-specific results unless they are part of a documented comparison.
