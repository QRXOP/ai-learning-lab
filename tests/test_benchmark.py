import csv
import json
import tracemalloc

import pytest

from ai_learning_lab.benchmarking import export_benchmark, run_benchmark


@pytest.fixture(scope="module")
def result():
    return run_benchmark(repeats=5, warmup=1)


def test_run_benchmark_fields(result):
    assert result["repeats"] == 5
    assert result["warmup"] == 1
    assert result["number_of_samples"] == 150
    for key in ("mean_latency_ms", "median_latency_ms", "p95_latency_ms"):
        assert result[key] > 0
    assert result["p95_latency_ms"] >= result["median_latency_ms"]
    assert result["python_tracemalloc_peak_mb"] >= 0
    assert set(result["hardware"]) == {"platform", "processor", "python_version"}
    assert set(result["dependencies"]) == {"scikit_learn", "numpy"}


def test_run_benchmark_leaves_tracing_state_alone():
    assert not tracemalloc.is_tracing()
    run_benchmark(repeats=1, warmup=0)
    assert not tracemalloc.is_tracing()

    tracemalloc.start()
    try:
        run_benchmark(repeats=1, warmup=0)
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize(
    ("repeats", "warmup"),
    [(0, 0), (-1, 0), (1.5, 0), (True, 0), (1, -1), (1, 1.0), (1, False)],
)
def test_run_benchmark_rejects_bad_arguments(repeats, warmup):
    with pytest.raises(ValueError):
        run_benchmark(repeats=repeats, warmup=warmup)


def test_export_writes_json_and_csv(result, tmp_path):
    json_path, csv_path = export_benchmark(result, output_dir=tmp_path / "out", file_name="run")
    assert json_path == tmp_path / "out" / "run.json"
    assert csv_path == tmp_path / "out" / "run.csv"

    assert json.loads(json_path.read_text(encoding="utf-8")) == result

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["dataset"] == "iris"
    assert json.loads(rows[0]["hardware"]) == result["hardware"]


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\\b", "../escape"])
def test_export_rejects_unsafe_file_names(result, tmp_path, name):
    with pytest.raises(ValueError):
        export_benchmark(result, output_dir=tmp_path, file_name=name)


@pytest.mark.parametrize("bad", [{}, [], None])
def test_export_rejects_empty_results(tmp_path, bad):
    with pytest.raises(ValueError):
        export_benchmark(bad, output_dir=tmp_path)
