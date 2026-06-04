#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PhaseResult:
    name: str
    before: dict[str, Any]
    after: dict[str, Any]
    latencies_ms: list[float]


def _load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _request_json(
    method: str,
    url: str,
    *,
    api_key: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _runtime(base_url: str) -> dict[str, Any]:
    return _request_json("GET", f"{base_url}/statsz")["runtime"]


def _device(runtime: dict[str, Any]) -> str:
    return str(runtime.get("loaded_device") or "unknown")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


async def _post_embedding(
    base_url: str,
    api_key: str,
    model: str,
    texts: list[str],
    timeout: float,
) -> float:
    payload: dict[str, Any] = {"model": model, "input": texts if len(texts) != 1 else texts[0]}
    start = time.perf_counter()
    await asyncio.to_thread(
        _request_json,
        "POST",
        f"{base_url}/v1/embeddings",
        api_key=api_key,
        payload=payload,
        timeout=timeout,
    )
    return (time.perf_counter() - start) * 1000


async def _run_low_qps_phase(
    *,
    base_url: str,
    api_key: str,
    model: str,
    requests: int,
    interval_seconds: float,
    timeout: float,
) -> list[float]:
    latencies: list[float] = []
    for idx in range(requests):
        latencies.append(
            await _post_embedding(base_url, api_key, model, [f"low-qps-{idx}"], timeout)
        )
        if idx != requests - 1:
            await asyncio.sleep(interval_seconds)
    return latencies


async def _run_burst_phase(
    *,
    base_url: str,
    api_key: str,
    model: str,
    concurrent_requests: int,
    timeout: float,
) -> list[float]:
    tasks = [
        _post_embedding(base_url, api_key, model, [f"burst-{idx}"], timeout)
        for idx in range(concurrent_requests)
    ]
    return list(await asyncio.gather(*tasks))


def _switch_device(base_url: str, api_key: str, device: str) -> dict[str, Any]:
    return _request_json(
        "POST",
        f"{base_url}/admin/device",
        api_key=api_key,
        payload={"device": device},
        timeout=120,
    )["runtime"]


def _summarize_latencies(values: list[float]) -> str:
    if not values:
        return "n/a"
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    return f"count={len(values)} avg={sum(values) / len(values):.1f}ms p95={p95:.1f}ms"


async def main_async() -> int:
    parser = argparse.ArgumentParser(description="Probe embedding provider CPU/GPU scaling behavior.")
    parser.add_argument("--base-url", default=os.getenv("EMBEDDING_PROVIDER_URL", "http://127.0.0.1:7997"))
    parser.add_argument("--env-file", default="deployments/gpu4/jina-v5-small.env")
    parser.add_argument("--low-requests", type=int, default=3)
    parser.add_argument("--low-interval", type=float, default=0.25)
    parser.add_argument("--burst-requests", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--skip-manual-reset", action="store_true")
    args = parser.parse_args()

    env = _load_env(Path(args.env_file))
    api_key = env.get("API_KEY") or os.getenv("API_KEY")
    model = env.get("MODEL_ID") or os.getenv("MODEL_ID")
    _require(bool(api_key), "API_KEY is required via --env-file or environment")
    _require(bool(model), "MODEL_ID is required via --env-file or environment")

    results: list[PhaseResult] = []

    if not args.skip_manual_reset:
        _switch_device(args.base_url, api_key, "cpu")

    before = _runtime(args.base_url)
    _require(_device(before) == "cpu", f"expected CPU before low-QPS phase, got {_device(before)}")
    low_latencies = await _run_low_qps_phase(
        base_url=args.base_url,
        api_key=api_key,
        model=model,
        requests=args.low_requests,
        interval_seconds=args.low_interval,
        timeout=args.timeout,
    )
    after_low = _runtime(args.base_url)
    _require(_device(after_low) == "cpu", f"low-QPS phase should stay on CPU, got {_device(after_low)}")
    results.append(PhaseResult("low_qps_cpu_hold", before, after_low, low_latencies))

    before_burst = after_low
    burst_latencies = await _run_burst_phase(
        base_url=args.base_url,
        api_key=api_key,
        model=model,
        concurrent_requests=args.burst_requests,
        timeout=args.timeout,
    )
    after_burst = _runtime(args.base_url)
    _require(_device(after_burst) == "cuda", f"burst phase should scale up to GPU, got {_device(after_burst)}")
    _require(after_burst.get("worker_pid") is not None, "GPU phase should expose a worker_pid")
    results.append(PhaseResult("burst_gpu_scale_up", before_burst, after_burst, burst_latencies))

    before_down = after_burst
    after_down = _switch_device(args.base_url, api_key, "cpu")
    _require(_device(after_down) == "cpu", f"scale-down phase should return to CPU, got {_device(after_down)}")
    _require(after_down.get("worker_pid") is None, "CPU scale-down should clear worker_pid")
    results.append(PhaseResult("manual_gpu_scale_down", before_down, after_down, []))

    print("device scaling probe passed")
    for item in results:
        print(
            f"- {item.name}: {_device(item.before)} -> {_device(item.after)}; "
            f"{_summarize_latencies(item.latencies_ms)}"
        )
    return 0


def main() -> int:
    try:
        return asyncio.run(main_async())
    except (AssertionError, urllib.error.URLError, TimeoutError) as exc:
        print(f"device scaling probe failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
