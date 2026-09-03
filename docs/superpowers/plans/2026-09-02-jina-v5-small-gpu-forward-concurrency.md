# jina-v5-small GPU Forward Concurrency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add benchmark-gated concurrent GPU forward execution for `jina-v5-small` while keeping CPU forwards serialized and preserving continuous batching, adaptive VRAM limits, and subprocess offload.

**Architecture:** A central batch collector continues to coalesce compatible requests. CUDA dispatches run through a bounded lane scheduler and a shared text budget, while CPU dispatch remains single-lane. The existing GPU subprocess protocol gains request IDs, a response reader, and worker-side concurrent execution so complete batches can overlap without loading another model copy.

**Tech Stack:** Python 3.12, FastAPI, asyncio, `subprocess.Popen`, `ThreadPoolExecutor`, pytest, NVIDIA CLI telemetry.

**Spec:** `docs/superpowers/specs/2026-09-02-jina-v5-small-gpu-forward-concurrency-design.md`

## Global Constraints

- Work on the current `main` branch is explicitly authorized by the user.
- Preserve every pre-existing uncommitted runtime-tuning change.
- `GPU_FORWARD_CONCURRENCY` defaults to `1` and must be positive.
- Effective CPU forward concurrency is always `1`.
- Only `jina-v5-small` may receive a production value greater than `1`, after the benchmark gate passes.
- Total CUDA texts in flight must never exceed the current shared adaptive target.
- One GPU worker process and one model copy remain in use.
- Existing API response ordering, offload, OOM backoff, and cooldown behavior remain intact.

---

### Task 1: Configuration and Runtime Metrics

**Files:**
- Modify: `provider/config.py`
- Modify: `provider/app.py`
- Modify: `tests/test_service_observability.py`
- Modify: `.env.example`
- Modify: `docker-compose.yml`
- Modify: `deployments/gpu4/jina-v5-nano.env.example`
- Modify: `deployments/gpu4/qwen3-embedding-0.6b.env.example`

**Interfaces:**
- Produces: `Settings.gpu_forward_concurrency: int`, runtime concurrency metrics, and default configuration value `1`.

- [ ] Write tests proving the default is `1`, positive values parse, non-positive values raise `ValueError`, and CPU effective concurrency remains `1`.
- [ ] Run `.venv/bin/pytest -q tests/test_service_observability.py` and verify the new assertions fail for the missing setting.
- [ ] Add `_positive_env_int("GPU_FORWARD_CONCURRENCY", "1")`, store it in `Settings`, and expose configured/effective/current/peak fields from runtime status.
- [ ] Add `GPU_FORWARD_CONCURRENCY=1` to generic examples and Compose without changing `jina-v5-small.env` yet.
- [ ] Re-run the focused tests and verify they pass.

---

### Task 2: Multiplex the Existing GPU Worker

**Files:**
- Modify: `provider/app.py:718-825`
- Modify: `provider/gpu_worker.py:111-179`
- Modify: `tests/test_runtime_idle_offload.py`

**Interfaces:**
- Produces: request-ID-correlated `_GpuEmbedderWorker.encode`, pending future failure on EOF, and concurrent worker-side `encode` execution.

- [ ] Add a controllable fake process/response stream test that submits two `encode` calls, returns the second response first, and asserts both callers receive their own embeddings.
- [ ] Add tests that one correlated error does not fail another request, EOF fails all pending calls, and termination remains idempotent.
- [ ] Run `.venv/bin/pytest -q tests/test_runtime_idle_offload.py -k 'worker'` and verify RED because the existing I/O lock serializes write-through-read and responses have no IDs.
- [ ] Replace the request-wide lock with a write-only lock, request counter, pending `Future` map, and reader thread. Keep startup handshake synchronous before starting the reader.
- [ ] Add request IDs to worker responses and use a `ThreadPoolExecutor(max_workers=settings.gpu_forward_concurrency)` for `encode` operations. Serialize response writes and drain submitted tasks before shutdown.
- [ ] Ensure `empty_cache` and shutdown are exclusive lifecycle operations and preserve graceful/terminate/kill fallback.
- [ ] Re-run focused and full idle-offload tests and verify GREEN.

---

### Task 3: Concurrent Batch Dispatch with a Shared Text Budget

**Files:**
- Modify: `provider/app.py:1485-1669`
- Modify: `tests/test_service_observability.py`

**Interfaces:**
- Produces: a central `ContinuousBatcher` collector, bounded CUDA dispatch tasks, CPU serialization, and text reservation/release.

- [ ] Write async tests with a blocking real fake runtime proving CPU peak forward count is `1`, CUDA peak never exceeds configured lanes, compatible inputs still batch together, and out-of-order completions preserve each request's input order.
- [ ] Write a test with adaptive target `4` and concurrency `4` proving total texts concurrently dispatched never exceeds `4`.
- [ ] Write error/cancellation tests proving lane and text capacity are released.
- [ ] Run `.venv/bin/pytest -q tests/test_service_observability.py -k 'concurr or batch'` and verify RED because `_worker()` awaits each `_dispatch()` inline.
- [ ] Add a CUDA lane semaphore and condition-protected text reservation. Split overflow before dispatch so each task owns only reserved texts.
- [ ] Keep one collector and track active dispatch tasks; collect the next window while capacity remains, but await active tasks when CPU is selected or CUDA capacity is exhausted.
- [ ] Update current/peak forward metrics at the actual runtime encode boundary and release all permits in `finally`.
- [ ] On shutdown, cancel collection, await or cancel active dispatch tasks deterministically, then allow runtime close to reap the worker.
- [ ] Re-run focused tests and the full suite.

---

### Task 4: Benchmark and Production Selection

**Files:**
- Create: `scripts/benchmark_gpu_forward_concurrency.py`
- Create: `tests/test_benchmark_gpu_forward_concurrency.py`
- Create: `docs/benchmarks/2026-09-02-jina-v5-small-gpu-forward-concurrency.md`
- Modify only if qualified: `deployments/gpu4/jina-v5-small.env`

**Interfaces:**
- Produces: repeatable concurrency `1/2/4` workload measurements and a documented production decision.

- [ ] Write tests for percentile calculation, workload result aggregation, and the qualification rule: throughput `>= 1.10x`, every P95 `<= 1.10x`, zero errors/OOMs, safe VRAM, and successful offload.
- [ ] Run `.venv/bin/pytest -q tests/test_benchmark_gpu_forward_concurrency.py` and verify RED because the benchmark module is missing.
- [ ] Implement CLI inputs for base URL, env file, candidate list, warmup count, repeats, and output report. Never print the API key.
- [ ] Implement the five specified workloads and collect HTTP latency, text/request throughput, `/statsz` concurrency metrics, `nvidia-smi` GPU utilization/VRAM, and errors.
- [ ] Run the full automated test suite and verify it passes.
- [ ] Start a separate `jina-v5-small` test instance for each candidate, benchmark `1`, `2`, and `4`, and write the complete result table and decision to the report.
- [ ] If `2` or `4` qualifies, apply the lowest best qualifying value only to `deployments/gpu4/jina-v5-small.env`; otherwise explicitly retain `1`.
- [ ] Restart only the `jina-v5-small` instance after a qualifying config change, then verify readiness, dimensions, error counters, worker PID lifecycle, P95, and VRAM.
