# Changelog

## Unreleased

### Fixed
- The GPU worker could not use CUDA once `INTERACTIVE_MAX_TOKENS` was set: the interactive queue's CPU encoder was loaded in `create_app()`, and the module-level `app = create_app()` also runs in the worker (it imports the provider package), where loading the CPU model hides CUDA. Every scale-up after the 2026-11-17 deploy failed with `No CUDA GPUs are available` and bulk requests stayed on CPU. The encoder is now loaded at startup.
- Short requests no longer wait behind bulk document forwards. The provider had one forward queue, and on CPU a batch of 8 long documents takes 59–116 s, so recall queries and memory writes from memory.preseen.ai took 60–130 s on 2026-11-17 (one sampled 1-text request: 4.4 s forward, about 125 s queued). With `INTERACTIVE_MAX_TOKENS` set, requests whose uncached inputs total at most that many tokens go to an interactive queue with its own CPU model instance, whose forwards run beside the bulk queue's. Under a burst of 8×2,400-token document batches on 8 cores, 1-text latency went from p50 203 s / p90 454 s to p50 0.33 s / p90 0.57 s (`docs/benchmarks/2026-11-17-interactive-queue-and-cuda-peak.md`). Off by default.
- Automatic CPU→CUDA scale-up no longer stalls requests. Starting the GPU worker loads the model (about 20 s on gpu4), and it used to run inside the request that crossed `CPU_TO_GPU_SCALE_UP_TEXTS`, so every queued request waited for the load. On 2026-11-07 this made 20 concurrent recall queries from CoSens take 24.5 s each. The worker now starts in a background thread while requests keep being served on CPU, and encodes switch to it once it is ready. A failed start keeps the CPU model as it is and only records `cuda_fallback_reason` for the retry cooldown. `/statsz` shows `cuda_scale_up_in_progress`. `POST /admin/device` with `cuda` still switches synchronously.

### Added
- `CUDA_MAX_FORWARD_TOKENS` caps each GPU forward at that many padded tokens, which bounds the GPU worker's VRAM footprint (about 1.3 GiB per 8192 tokens per forward, measured). llm-gateway's `GPU_NEIGHBOUR_RESERVE_MIB` is sized from it. Off by default.
- `scripts/benchmark_interactive_latency.py` (1-text latency under a document-ingest burst) and `scripts/measure_cuda_peak.py` (VRAM footprint per batch size at `MAX_LENGTH`).
- `/v1/embeddings` accepts an optional `prompt_name` (`query` | `document`) that selects the `Query: ` / `Document: ` input prefix of Jina v5 models, so `task=retrieval` can embed queries correctly. Omitting it keeps the previous behavior (model default `document`).
- `prompt_name` is part of the embedding cache key, the batcher group key, the GPU worker request, the `MAX_LENGTH` token counting, and the request log.
- `deployments/cloudflare-lb/`: Terraform module for a US/CN Cloudflare load balancer in front of `embed.preseen.ai` (`least_outstanding_requests` steering over two private-network pools). Not applied yet and not validated with `terraform validate`; see its README.
