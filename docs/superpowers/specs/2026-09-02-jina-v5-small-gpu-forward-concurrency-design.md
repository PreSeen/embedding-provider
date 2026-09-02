# jina-v5-small GPU Forward Concurrency Design

## Problem

The embedding provider accepts concurrent HTTP requests but deliberately serializes GPU model execution in two places:

1. `ContinuousBatcher` combines queued texts into one batch and waits for that dispatch before collecting the next dispatch.
2. `_GpuEmbedderWorker` holds one I/O lock from request write through response read, while `provider.gpu_worker` reads and executes one request at a time.

This is efficient when one large homogeneous batch saturates the GPU. It can leave the GPU underutilized when `jina-v5-small` receives short or heterogeneous batches that cannot be merged because their task or output dimensions differ. The optimization must determine whether a small number of concurrent GPU forwards improves real throughput without defeating batching, causing CUDA OOMs, or changing CPU behavior.

## Scope

The shared provider gains an opt-in GPU-forward concurrency capability. Its default remains `1`, so `jina-v5-nano`, `qwen3-embedding-0.6b`, and `qwen3-embedding-4b` keep their current behavior. Only `deployments/gpu4/jina-v5-small.env` may enable a value greater than `1`, and only after host benchmarks pass.

Existing uncommitted runtime-tuning changes in `provider/app.py`, `provider/config.py`, their tests, examples, README, and Compose configuration are treated as the implementation baseline and must be preserved.

## Goals

- Keep CPU model execution serialized with effective forward concurrency `1`.
- Permit `jina-v5-small` to run a measured, bounded number of GPU forwards concurrently.
- Preserve continuous batching and group compatibility by `(task, dimensions)`.
- Preserve GPU worker subprocess offload so CPU state has no worker PID and releases all provider-owned CUDA memory.
- Bound total concurrent text demand so concurrency cannot multiply the adaptive batch target by the number of lanes.
- Select production concurrency from `1`, `2`, and `4` using throughput, latency, VRAM, and error measurements.

## Non-goals

- Enabling concurrent forwards for other embedding deployments.
- Starting multiple GPU worker processes or loading multiple model copies.
- Removing the adaptive batch, VRAM probe, CUDA OOM backoff, cache, or device-switch cooldown.
- Changing the embedding API or result ordering.
- Assuming that a higher concurrency value is automatically faster.

## Configuration

Add:

```text
GPU_FORWARD_CONCURRENCY=1
```

Rules:

- The value must be a positive integer.
- It is an upper bound, not a guarantee of simultaneous execution.
- CPU always uses one forward lane regardless of the configured value.
- The default is `1` in `Settings`, `.env.example`, examples, and Compose.
- The final measured value is written only to `deployments/gpu4/jina-v5-small.env`.

Expose the configured and effective values in `/statsz` and `/healthz` runtime data:

- `gpu_forward_concurrency_configured`
- `gpu_forward_concurrency_effective`, equal to the configured value on CUDA and `1` on CPU
- `gpu_forwards_in_flight`
- `gpu_forward_peak_in_flight`

## Architecture

### Continuous batch scheduler

Keep one central queue and one collector so requests still coalesce during `BATCH_WINDOW_MS=50`. The collector forms compatible `(task, dimensions)` batches exactly as today, then submits complete batches to a bounded set of dispatch tasks instead of awaiting every dispatch inline.

The collector must not create one independent queue per lane because that would fragment batches. It continues to prioritize overflow from a partially dispatched group before waiting for a new batch window.

CPU dispatch always awaits the single active task before collecting another forward. CUDA dispatch may keep up to `GPU_FORWARD_CONCURRENCY` tasks active.

### Global text budget

Concurrent lanes share one text budget equal to the current adaptive batch target. A dispatch reserves its text count before entering the worker and releases it afterward. A new CUDA dispatch cannot begin until both conditions hold:

- an execution lane is free;
- `sum(texts_in_flight) + next_batch_size <= effective_batch_target`.

If the next batch is larger than the remaining budget, the scheduler splits it and leaves overflow at the front of the pending list. At least one text may run when no other dispatch is active, preserving the current low-VRAM fallback.

This prevents concurrency `4` from turning a safe adaptive target of `64` into as many as `256` simultaneous texts. It also makes the trade-off explicit: concurrency helps only when overlapping smaller forwards beats one combined forward, especially across incompatible task/dimension groups.

### Multiplexed parent IPC

Replace `_GpuEmbedderWorker`'s request-wide I/O lock with:

- a monotonically increasing request ID;
- a short write lock covering only one JSON line;
- a dictionary of request ID to `concurrent.futures.Future`;
- a dedicated reader thread that resolves the matching future.

Inference request:

```json
{"id":23,"op":"encode","texts":["..."],"dimensions":1024,"task":"text-matching"}
```

Inference response:

```json
{"id":23,"status":"ok","embeddings":[[0.1]],"sample_bytes_per_text":1048576}
```

Lifecycle operations remain exclusive. Shutdown stops new requests, waits for submitted work to settle, and retains the existing graceful wait followed by terminate/kill fallbacks.

### Concurrent GPU worker

The worker main loop reads requests continuously and submits `encode` operations to a `ThreadPoolExecutor` sized to `GPU_FORWARD_CONCURRENCY`. Response writes use one lock and include the request ID, so completion order does not affect caller correlation.

The model instance is shared, matching the current in-process model semantics. No second model copy or second CUDA context is created.

### Adaptive batching and OOM behavior

The existing `AdaptiveBatchState` lock protects target updates. Successful concurrent dispatches update demand and memory samples after completion. CUDA OOM handling must atomically lower the shared target before another queued dispatch reserves capacity.

Only the failed dispatch performs its existing chunk backoff. Other already-running forwards are allowed to finish. New dispatches observe the lower target. `empty_cache` remains a serialized worker control operation and runs only when no new write is being issued.

Device scale-down continues to require `inflight_encodes == 0`, which now covers all dispatch lanes. The GPU worker cannot terminate while a forward is active.

## Result Ordering and Failures

- Each HTTP request retains its input ordering through `_PendingRequestState.input_index`.
- Out-of-order batch completion may resolve later HTTP requests first; it cannot place an embedding in the wrong request.
- A per-request worker error fails only its correlated batch and its affected HTTP futures.
- Unexpected worker exit fails every pending IPC future, clears the worker PID, and uses the existing CPU fallback path for subsequent service.
- A dispatch task must release its lane and text reservation in `finally`, including cancellation and errors.
- Continuous batcher shutdown cancels collection, awaits active dispatch tasks, and then closes the runtime worker.

## Benchmark Gate

Benchmark the currently deployed `jina-v5-small` model on GPU 0 with concurrency values `1`, `2`, and `4`. Run each candidate after a warmup phase and repeat each workload enough times to avoid one-off startup noise.

Workloads:

1. Single-text burst: 32 concurrent HTTP requests, one short text each.
2. Homogeneous batch burst: 16 concurrent requests, eight short texts each, same task and dimensions.
3. Heterogeneous burst: concurrent requests split across two dimensions or tasks so they form separate batch groups.
4. Long-text burst: eight concurrent requests near the normal upper token distribution.
5. Low-QPS control: sequential single-text requests to ensure CPU scale-down behavior is unchanged.

Record:

- successful texts per second;
- requests per second;
- p50 and p95 request latency;
- peak `gpu_forwards_in_flight`;
- peak GPU utilization and VRAM;
- CUDA OOMs, worker errors, and HTTP failures;
- actual adaptive batch target and text count per forward.

A candidate greater than `1` may be enabled only if, compared with concurrency `1`:

- weighted throughput improves by at least 10%;
- no workload's p95 latency regresses by more than 10%;
- there are zero CUDA OOMs and worker/HTTP errors;
- peak VRAM stays below total VRAM minus the configured fixed and ratio safety headroom;
- idle CPU offload still removes the worker PID from `nvidia-smi`.

Choose the lowest concurrency value that produces the best qualifying throughput. If neither `2` nor `4` qualifies, keep `GPU_FORWARD_CONCURRENCY=1`; the benchmark result is still the optimization outcome and avoids deploying a harmful change.

## Testing

### Configuration tests

- Default is `1`.
- Positive configured values parse correctly.
- Zero and negative values are rejected.
- Effective concurrency is always `1` on CPU.

### IPC tests

- Responses arriving out of order resolve the correct callers.
- Four submitted requests may be pending without serializing on the write lock.
- One request error does not corrupt other requests.
- EOF fails all pending requests.
- Shutdown waits for active requests and reaps the worker.

### Scheduler tests

- CPU never has more than one forward active.
- CUDA never exceeds the configured lane count.
- Total texts in flight never exceeds the shared adaptive target.
- Compatible requests still combine into one batch.
- Incompatible groups can overlap when lanes and text budget are available.
- Overflow remains ahead of a new batch window.
- Results remain in per-request input order despite out-of-order completion.
- Lane and text permits are released after errors and cancellation.

### Runtime tests

- Scale-down waits for every concurrent encode to finish.
- Worker termination clears the PID after concurrent use.
- OOM backoff lowers the target seen by subsequent dispatches.
- Runtime stats expose configured/effective concurrency and current/peak in-flight forwards.

## Rollout

1. Add the shared capability with default concurrency `1` and run the full test suite.
2. Run the benchmark gate against a separate `jina-v5-small` instance so production traffic and configuration are unchanged.
3. Record the three candidate results in a dated report under `docs/benchmarks/`.
4. If a value greater than `1` qualifies, update only `deployments/gpu4/jina-v5-small.env` and restart only that instance.
5. Verify readiness, result dimensions, cache behavior, worker PID, device switching, error count, VRAM, and p95 latency.
6. Roll back by setting `GPU_FORWARD_CONCURRENCY=1` and restarting only `jina-v5-small` if production metrics regress.

## Acceptance Criteria

- All existing tests and new concurrency tests pass.
- Existing uncommitted runtime-tuning changes remain intact.
- CPU forward concurrency is always `1`.
- GPU forward concurrency and shared text budget are both enforced.
- Concurrent responses cannot cross request boundaries.
- Dynamic CPU/CUDA switching and complete GPU worker offload still work.
- A reproducible benchmark report compares `1`, `2`, and `4`.
- The production value for `jina-v5-small` satisfies every benchmark gate; otherwise it remains `1` with the non-qualifying result documented.
