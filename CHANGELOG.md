# Changelog

## Unreleased

### Fixed
- Automatic CPU→CUDA scale-up no longer stalls requests. Starting the GPU worker loads the model (about 20 s on gpu4), and it used to run inside the request that crossed `CPU_TO_GPU_SCALE_UP_TEXTS`, so every queued request waited for the load. On 2026-11-07 this made 20 concurrent recall queries from CoSens take 24.5 s each. The worker now starts in a background thread while requests keep being served on CPU, and encodes switch to it once it is ready. A failed start keeps the CPU model as it is and only records `cuda_fallback_reason` for the retry cooldown. `/statsz` shows `cuda_scale_up_in_progress`. `POST /admin/device` with `cuda` still switches synchronously.

### Added
- `/v1/embeddings` accepts an optional `prompt_name` (`query` | `document`) that selects the `Query: ` / `Document: ` input prefix of Jina v5 models, so `task=retrieval` can embed queries correctly. Omitting it keeps the previous behavior (model default `document`).
- `prompt_name` is part of the embedding cache key, the batcher group key, the GPU worker request, the `MAX_LENGTH` token counting, and the request log.
- `deployments/cloudflare-lb/`: Terraform module for a US/CN Cloudflare load balancer in front of `embed.preseen.ai` (`least_outstanding_requests` steering over two private-network pools). Not applied yet and not validated with `terraform validate`; see its README.
