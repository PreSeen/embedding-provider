#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


WORKLOADS = (
    ("single_serial", 24, 1, 1),
    ("batch_8_serial", 12, 8, 1),
    ("batch_64_serial", 6, 64, 1),
    ("single_concurrent_16", 32, 1, 16),
    ("batch_8_concurrent_8", 16, 8, 8),
)


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    interpolated = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(interpolated, 3)


def aggregate_workload(
    *,
    latencies_ms: list[float],
    request_count: int,
    text_count: int,
    elapsed_seconds: float,
    errors: list[str],
) -> dict[str, Any]:
    elapsed = max(elapsed_seconds, 1e-9)
    return {
        "request_count": request_count,
        "text_count": text_count,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "p50_ms": percentile(latencies_ms, 50),
        "p95_ms": percentile(latencies_ms, 95),
        "requests_per_second": round(request_count / elapsed, 3),
        "texts_per_second": round(text_count / elapsed, 3),
        "error_count": len(errors),
        "errors": errors[:10],
    }


def candidate_qualifies(baseline: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if candidate.get("oom_count", 0) or not candidate.get("safe_vram") or not candidate.get("offload_released"):
        return False
    baseline_workloads = baseline.get("workloads") or {}
    candidate_workloads = candidate.get("workloads") or {}
    if not baseline_workloads or set(candidate_workloads) != set(baseline_workloads):
        return False
    baseline_throughput = 0.0
    candidate_throughput = 0.0
    for name, baseline_result in baseline_workloads.items():
        result = candidate_workloads[name]
        if result.get("error_count", 0):
            return False
        baseline_p95 = float(baseline_result.get("p95_ms") or 0.0)
        if baseline_p95 > 0 and float(result.get("p95_ms") or 0.0) > baseline_p95 * 1.10:
            return False
        baseline_throughput += float(baseline_result.get("texts_per_second") or 0.0)
        candidate_throughput += float(result.get("texts_per_second") or 0.0)
    return baseline_throughput > 0 and candidate_throughput >= baseline_throughput * 1.10


def _request_json(url: str, *, api_key: str | None, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with urlopen(request, timeout=180) as response:
            return json.loads(response.read())
    except (HTTPError, URLError) as exc:
        detail = exc.read().decode("utf-8", errors="replace") if isinstance(exc, HTTPError) else str(exc)
        raise RuntimeError(f"request failed ({getattr(exc, 'code', 'network')}): {detail[:300]}") from exc


def _make_texts(count: int, request_index: int) -> list[str]:
    return [
        f"benchmark request {request_index} text {index}: semantic retrieval concurrency measurement " * 4
        for index in range(count)
    ]


def _run_workload(
    base_url: str,
    api_key: str | None,
    *,
    request_count: int,
    texts_per_request: int,
    concurrency: int,
) -> dict[str, Any]:
    latencies: list[float] = []
    errors: list[str] = []

    def send(index: int) -> None:
        started = time.perf_counter()
        try:
            _request_json(
                f"{base_url}/v1/embeddings",
                api_key=api_key,
                payload={"input": _make_texts(texts_per_request, index)},
            )
            latencies.append((time.perf_counter() - started) * 1000)
        except Exception as exc:
            errors.append(str(exc))

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send, index) for index in range(request_count)]
        for future in as_completed(futures):
            future.result()
    elapsed = time.perf_counter() - started
    successes = request_count - len(errors)
    return aggregate_workload(
        latencies_ms=latencies,
        request_count=successes,
        text_count=successes * texts_per_request,
        elapsed_seconds=elapsed,
        errors=errors,
    )


def _gpu_memory() -> tuple[int | None, int | None]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        used, total = (int(value.strip()) for value in completed.stdout.splitlines()[0].split(","))
        return used, total
    except Exception:
        return None, None


def _gpu_compute_pids() -> set[int] | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return {int(line.strip()) for line in completed.stdout.splitlines() if line.strip()}
    except Exception:
        return None


def benchmark_running_service(
    base_url: str,
    *,
    api_key: str | None,
    warmup_count: int,
    repeats: int,
) -> dict[str, Any]:
    for index in range(warmup_count):
        _request_json(
            f"{base_url}/v1/embeddings",
            api_key=api_key,
            payload={"input": _make_texts(8, index)},
        )
    workloads: dict[str, Any] = {}
    for name, requests, texts, concurrency in WORKLOADS:
        samples = [
            _run_workload(
                base_url,
                api_key,
                request_count=requests,
                texts_per_request=texts,
                concurrency=concurrency,
            )
            for _ in range(repeats)
        ]
        workloads[name] = {
            key: round(sum(float(sample[key]) for sample in samples) / len(samples), 3)
            for key in ("p50_ms", "p95_ms", "requests_per_second", "texts_per_second")
        }
        workloads[name]["error_count"] = sum(int(sample["error_count"]) for sample in samples)
        workloads[name]["errors"] = [error for sample in samples for error in sample["errors"]][:10]
    stats = _request_json(f"{base_url}/statsz", api_key=None)
    used_mb, total_mb = _gpu_memory()
    runtime = stats.get("runtime") or {}
    all_errors = [error for workload in workloads.values() for error in workload["errors"]]
    return {
        "workloads": workloads,
        "runtime": runtime,
        "gpu_memory_used_mb": used_mb,
        "gpu_memory_total_mb": total_mb,
        "safe_vram": used_mb is not None and total_mb is not None and used_mb <= total_mb * 0.90,
        "oom_count": (
            sum("out of memory" in error.lower() for error in all_errors)
            + int((runtime.get("adaptive_batch") or {}).get("oom_target_ceiling") is not None)
        ),
        "offload_released": False,
    }


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _wait_ready(base_url: str, process: subprocess.Popen[Any], timeout: float = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"benchmark server exited with code {process.returncode}")
        try:
            _request_json(f"{base_url}/readyz", api_key=None)
            return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError("benchmark server did not become ready")


def benchmark_candidate(
    root: Path,
    env_file: Path,
    *,
    candidate: int,
    port: int,
    warmup_count: int,
    repeats: int,
) -> dict[str, Any]:
    configured = _read_env(env_file)
    env = os.environ.copy()
    env.update(configured)
    env.update(
        {
            "BIND_HOST": "127.0.0.1",
            "PORT": str(port),
            "START_DEVICE": "cuda",
            "DEVICE_SWITCH_MIN_SECONDS": "0",
            "GPU_FORWARD_CONCURRENCY": str(candidate),
            "PYTHONUNBUFFERED": "1",
        }
    )
    api_key = configured.get("API_KEY") or None
    base_url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryFile(mode="w+") as log_file:
        process = subprocess.Popen(
            [str(root / ".venv/bin/uvicorn"), "provider.app:app", "--app-dir", str(root), "--host", "127.0.0.1", "--port", str(port)],
            cwd=root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_ready(base_url, process)
            result = benchmark_running_service(
                base_url,
                api_key=api_key,
                warmup_count=warmup_count,
                repeats=repeats,
            )
            worker_pid = result["runtime"].get("worker_pid")
            _request_json(f"{base_url}/admin/device", api_key=api_key, payload={"device": "cpu"})
            offloaded = _request_json(f"{base_url}/statsz", api_key=None).get("runtime") or {}
            live_gpu_pids = _gpu_compute_pids()
            result["offload_released"] = (
                worker_pid is not None
                and live_gpu_pids is not None
                and int(worker_pid) not in live_gpu_pids
                and offloaded.get("worker_pid") is None
                and offloaded.get("loaded_device") == "cpu"
            )
            result["candidate"] = candidate
            return result
        except Exception:
            log_file.seek(0)
            tail = log_file.read()[-4000:]
            raise RuntimeError(f"candidate {candidate} failed; server log tail:\n{tail}")
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _markdown_report(results: list[dict[str, Any]]) -> str:
    baseline = results[0]
    lines = [
        "# jina-v5-small GPU Forward Concurrency Benchmark",
        "",
        "| concurrency | aggregate texts/s | worst p95 ms | errors | GPU MiB | offload released | qualifies |",
        "|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for result in results:
        throughput = sum(float(item["texts_per_second"]) for item in result["workloads"].values())
        p95 = max(float(item["p95_ms"]) for item in result["workloads"].values())
        errors = sum(int(item["error_count"]) for item in result["workloads"].values())
        qualifies = result is not baseline and candidate_qualifies(baseline, result)
        lines.append(
            f"| {result['candidate']} | {throughput:.1f} | {p95:.1f} | {errors} | "
            f"{result.get('gpu_memory_used_mb')} | {result.get('offload_released')} | {qualifies} |"
        )
    lines.extend(["", "```json", json.dumps(results, indent=2), "```", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark embedding GPU forward concurrency without printing API keys")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--candidates", default="1,2,4")
    parser.add_argument("--warmup-count", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--port", type=int, default=17997)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    candidates = [int(value) for value in args.candidates.split(",")]
    if not candidates or candidates[0] != 1 or any(value < 1 for value in candidates):
        parser.error("candidates must be positive and begin with baseline 1")
    results = [
        benchmark_candidate(
            root,
            args.env_file.resolve(),
            candidate=candidate,
            port=args.port,
            warmup_count=args.warmup_count,
            repeats=args.repeats,
        )
        for candidate in candidates
    ]
    report = _markdown_report(results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
