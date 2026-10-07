# Embedding Provider

Standalone OpenAI-compatible embedding service for Stardust shared use.

## Purpose

- Run independently from `memory-connector`
- Host embedding models on a shared GPU server
- Reuse one service codebase for different embedder instances
- Start multiple isolated provider instances from different env files

## Control Plane Boundary

`embedding-provider` is a model-serving runtime, not a backend control plane.

- Active provider selection lives in `memory-connector` through `config/llm_profiles.json` and `targets.embedder`
- This service must not store backend graph state, dirty/re-embed state, or active profile overrides
- This service only owns model loading, batching, health/readiness, and runtime-local offload/reload behavior

## Runtime Model

The service is a thin OpenAI-compatible wrapper over one loaded embedding model. It does not carry `memory-connector` backend logic. To run multiple embedders, start multiple compose projects with different env files, ports, cache directories, and optional GPU pinning.

Use a user-writable cache path such as `./runtime-cache/<instance-name>` for host deployments. Avoid reusing Docker-created root-owned cache folders under `./data/`.

## Instance Examples

Example Jina v5 instance:

```bash
cp deployments/gpu4/jina-v5-nano.env.example deployments/gpu4/jina-v5-nano.env
docker compose --env-file deployments/gpu4/jina-v5-nano.env -p embedding-provider-jina-v5-nano up -d --build
```

Example Qwen instance:

```bash
cp deployments/gpu4/qwen3-embedding-0.6b.env.example deployments/gpu4/qwen3-embedding-0.6b.env
docker compose --env-file deployments/gpu4/qwen3-embedding-0.6b.env -p embedding-provider-qwen3-embedding-0-6b up -d --build
```

## Local Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
cp .env.example .env
set -a && source .env && set +a
uvicorn provider.app:app --host 127.0.0.1 --port 8000
```

## OpenAI-Compatible API

```bash
curl -X POST http://127.0.0.1:8000/v1/embeddings \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -H "X-Request-Id: demo-embed-001" \
  -d '{
    "model": "jinaai/jina-embeddings-v5-text-nano",
    "input": ["Alice leads Project Alpha"]
  }'
```

Observability endpoints:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
curl http://127.0.0.1:8000/statsz
```

## Stardust GPU4

- Repo target: `stardust@stardust-gpu4:~/Projects/embedding-provider`
- Private bind: prefer `127.0.0.1` on the host, one port per instance
- Keep `memory-connector` pointing at the chosen provider instance through `GRAPHITI_EMBEDDER_BASE_URL`
- On the current `gpu4` host, the recommended path is the host-venv scripts under `scripts/`, because Docker GPU runtime is not configured.
- For public exposure, bind the model server to `127.0.0.1`, enforce `API_KEY`, and publish HTTPS through a Cloudflare named tunnel.

## Current Production Endpoint

- Public base URL: `https://embed.preseen.ai/v1`
- Current model: `jinaai/jina-embeddings-v5-text-nano`
- Current service bind on `gpu4`: `127.0.0.1:7997`
- Provider defaults:
  - `MAX_LENGTH=8192`
  - `MAX_BATCH_SIZE=64`
  - `CPU_BATCH_TARGET=8`
  - `START_DEVICE=cpu`
  - `DEFAULT_DIMENSIONS=768`
  - `IDLE_OFFLOAD_SECONDS=1800`
  - `REQUEST_LOG_LIMIT=100000`
  - `EMBEDDING_CACHE_LIMIT=100000`

## Idle GPU Offload

On CUDA hosts, the provider now keeps the HTTP process alive but moves the loaded model into a dedicated GPU worker subprocess. After a long idle window, that worker exits entirely so VRAM is actually released.

- `IDLE_OFFLOAD_SECONDS`: how long the service may stay idle before the GPU worker is terminated
- `IDLE_OFFLOAD_POLL_SECONDS`: how often the background monitor checks whether idle termination should run
- `DEVICE_SWITCH_MIN_SECONDS`: minimum interval between automatic CPU/CUDA device switches
- `START_DEVICE`: startup device policy, usually `cpu`, `cuda`, or `auto`
- `CPU_BATCH_TARGET`: normal CPU queue-drain batch target
- `CPU_TO_GPU_SCALE_UP_TEXTS`: queued/requested text count threshold that promotes CPU to CUDA
- `GPU_TO_CPU_SCALE_DOWN_TEXTS`: sustained small CUDA batch threshold used for scale-down
- `GPU_TO_CPU_SCALE_DOWN_SECONDS`: how long small CUDA batches must persist before CPU scale-down
- `CUDA_RETRY_SECONDS` (default `300`): after the GPU worker fails to start (for example CUDA OOM because another service holds the VRAM) the runtime serves from CPU; after this many seconds the next batch of `CPU_TO_GPU_SCALE_UP_TEXTS` or more texts retries CUDA. A successful start ends the fallback and re-enables idle offload; a failed retry starts the wait again. `POST /admin/device` with `cuda` retries immediately. `/statsz` shows `cuda_fallback_reason` and `cuda_retry_in_seconds`.
- `REQUEST_LOG_LIMIT`: in-memory request/input log retention limit
- `EMBEDDING_CACHE_LIMIT`: in-memory exact-match embedding cache entry limit
- `CUDA_BATCH_GROWTH_FACTOR`: multiplier used when CUDA target grows after full successful dispatches
- `CUDA_BATCH_DEMAND_EMA_ALPHA`: smoothing factor for lowering CUDA target toward recent average queue demand
- `CUDA_VRAM_SAFETY_FIXED_MB`: fixed CUDA free-VRAM safety headroom
- `CUDA_VRAM_SAFETY_TOTAL_RATIO`: proportional CUDA total-VRAM safety headroom
- `GPU_FORWARD_CONCURRENCY`: maximum simultaneous forwards inside the one CUDA worker; defaults to `1`

When idle offload is enabled:

- an unused embedding model vacates VRAM after the configured idle threshold because the GPU worker exits
- the next embedding request automatically starts a fresh GPU worker before inference
- `/healthz` exposes a `runtime` block with `loaded_device`, `engine_state`, `idle_for_seconds`, `reload_in_progress`, and `worker_pid`
- `/readyz` distinguishes “process is alive” from “service is ready to accept embedding requests”
- `/statsz` exposes request counters, queue depth, last error summary, and reload/offload counters

This only affects the embedding-provider process itself. It does not touch the OCR service, system CUDA, or other GPU workloads on the host.

## GPU Forward Concurrency

CPU inference always stays serial because parallel CPU forwards compete for the same cores without improving useful throughput. CUDA can use bounded concurrent forwards when `GPU_FORWARD_CONCURRENCY` is greater than `1`: the central batcher splits only the current adaptive text budget across lanes, the parent correlates out-of-order worker responses by request ID, and the child retains one model copy. `/statsz` reports `gpu_forward_concurrency_configured`, `gpu_forward_concurrency_effective`, `gpu_forwards_in_flight`, and `gpu_forward_peak_in_flight`.

Keep the generic default at `1`. Benchmark `jina-v5-small` candidates in an isolated local instance before changing its production env:

```bash
.venv/bin/python scripts/benchmark_gpu_forward_concurrency.py \
  --env-file deployments/gpu4/jina-v5-small.env \
  --candidates 1,2,4 \
  --output docs/benchmarks/jina-v5-small-gpu-forward-concurrency.md
```

The benchmark never prints the API key and requires at least 10% aggregate throughput improvement, no workload P95 regression above 10%, no errors/OOM, safe VRAM, and successful worker offload.

## VRAM-Aware Batching

On CUDA hosts, `MAX_BATCH_SIZE` is a hard upper bound, not the batch size used blindly. Before the runtime has a GPU memory sample, it probes with a single text. The GPU worker reports peak memory growth per text for that real encode call, and later chunks use the current free VRAM minus safety headroom to choose the next batch size. Batch growth is ramped after the probe, so the runtime grows at most 1, 2, 4, 8, and onward instead of jumping directly to the hard cap. The CUDA target stays stable across small follow-up requests; sustained small batches are handled by scaling back down to CPU. If available VRAM is below the safety headroom, the runtime keeps forwarding one text at a time. CUDA OOM backoff remains the final guard for unusually large inputs. On CPU, `CPU_BATCH_TARGET` controls the normal queue-drain target, with `MAX_BATCH_SIZE` still acting as the hard upper bound.

## Remote Sync

```bash
bash scripts/deploy_gpu4.sh
```

This syncs the repo to `stardust-gpu4` without forcing one fixed instance to start.

To start an instance on the server:

```bash
ssh stardust-gpu4-stardust
cd ~/Projects/embedding-provider
./scripts/start_host_instance.sh deployments/gpu4/jina-v5-nano.env
```

To expose that instance on the public internet with a fixed hostname on a NATed host like `gpu4`:

```bash
./scripts/start_public_cloudflared.sh deployments/gpu4/jina-v5-nano.env
```

Set these values in `deployments/gpu4/jina-v5-nano.env` before starting the tunnel:

```bash
BIND_HOST=127.0.0.1
API_KEY=change-me
PUBLIC_UPSTREAM=127.0.0.1:7997
```

Create the named tunnel and DNS route from a machine already logged into Cloudflare:

```bash
cloudflared tunnel create embedding-provider-preseen-ai
cloudflared tunnel route dns embedding-provider-preseen-ai embed.preseen.ai
cloudflared tunnel token embedding-provider-preseen-ai
```

Run the token on `gpu4`:

```bash
docker rm -f embedding-provider-cloudflared >/dev/null 2>&1 || true
docker run -d \
  --name embedding-provider-cloudflared \
  --restart unless-stopped \
  --network host \
  cloudflare/cloudflared:latest \
  tunnel --no-autoupdate run --token <TOKEN> --url http://127.0.0.1:7997
```

Operational checks:

```bash
curl https://embed.preseen.ai/healthz
curl https://embed.preseen.ai/readyz
curl https://embed.preseen.ai/statsz
curl -H "Authorization: Bearer <API_KEY>" https://embed.preseen.ai/v1/models
docker logs --tail 50 embedding-provider-cloudflared
```

If you have a directly routable public IP and want a host-local reverse proxy instead of Cloudflare Tunnel, the repo also includes `./scripts/start_public_caddy.sh` plus `deployments/gpu4/public.Caddyfile`.

The host scripts will auto-create `deployments/.../*.env` from the matching
`.env.example` on first run. To inspect or stop one instance:

```bash
./scripts/status_host_instance.sh deployments/gpu4/jina-v5-nano.env
./scripts/stop_host_instance.sh deployments/gpu4/jina-v5-nano.env
```
