from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_gpu_forward_concurrency import (
    aggregate_workload,
    candidate_qualifies,
    percentile,
)


def test_percentile_interpolates_sorted_values() -> None:
    assert percentile([40.0, 10.0, 30.0, 20.0], 50) == 25.0
    assert percentile([10.0, 20.0], 95) == 19.5


def test_aggregate_workload_reports_latency_throughput_and_errors() -> None:
    result = aggregate_workload(
        latencies_ms=[100.0, 200.0, 300.0],
        request_count=3,
        text_count=12,
        elapsed_seconds=0.6,
        errors=["boom"],
    )

    assert result["p95_ms"] == 290.0
    assert result["requests_per_second"] == 5.0
    assert result["texts_per_second"] == 20.0
    assert result["error_count"] == 1


def test_candidate_qualification_enforces_every_gate() -> None:
    baseline = {
        "workloads": {
            "single": {"p95_ms": 100.0, "texts_per_second": 100.0},
            "batch": {"p95_ms": 200.0, "texts_per_second": 200.0},
        }
    }
    candidate = {
        "workloads": {
            "single": {"p95_ms": 105.0, "texts_per_second": 115.0, "error_count": 0},
            "batch": {"p95_ms": 210.0, "texts_per_second": 225.0, "error_count": 0},
        },
        "oom_count": 0,
        "safe_vram": True,
        "offload_released": True,
    }
    assert candidate_qualifies(baseline, candidate)

    for key, bad_value in (
        ("oom_count", 1),
        ("safe_vram", False),
        ("offload_released", False),
    ):
        failed = {**candidate, key: bad_value}
        assert not candidate_qualifies(baseline, failed)

    high_p95 = {**candidate, "workloads": {**candidate["workloads"]}}
    high_p95["workloads"] = {
        **candidate["workloads"],
        "single": {**candidate["workloads"]["single"], "p95_ms": 111.0},
    }
    assert not candidate_qualifies(baseline, high_p95)
