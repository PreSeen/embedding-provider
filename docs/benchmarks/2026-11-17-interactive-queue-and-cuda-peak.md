# Interactive queue and CUDA peak, jina-v5-small on gpu4 (2026-11-17)

## Why

On 2026-11-17 single memory_write and recall embeddings from memory.preseen.ai took 60–130 s. There were two causes.

1. The provider had one forward queue. On CPU (Ryzen 9950X) one 8192-token text takes 7–15 s and a batch of 8 took 59–116 s. A sampled 1-text request ran a 4.4 s forward after about 125 s in the queue.
2. llm-gateway's vLLM instances ran at `--gpu-memory-utilization 0.950` on both RTX 5090s and left about 0.9 GiB free on each, so every CUDA scale-up failed with OOM and the provider stayed on CPU.

Traffic in the live request log, 2026-11-13 18:40 to 11-18 02:31 UTC (100,000 most recent inputs), tokenised with the model's tokenizer:

| caller (source IP) | requests | tokens per request |
|---|---:|---|
| memory.preseen.ai on gpu4 (75.103.6.230), 1 text | 2,495 | p50 64, p90 514, p99 940, max 4,059 |
| memory.preseen.ai, several texts | 10,833 | p50 21, p90 58, p99 4,822 |
| CoSens RSI on GCP (34.145.172.252), several texts | 43 | p50 3,530, p90 35,027 (8 documents of up to 8,192 chars) |
| callzhang/ceo-agent-service email classifier (223.104.40.159, 2409:8900:…, 222.128.34.78, 39.157.87.165, all in China), 1 text | 475 | p50 847, p90 7,457 |
| the same email classifier, several texts | 35 | 56-text training batches, each text truncated to 8,192 |

The China Mobile volume is the ceo-agent-service email classifier: every input is an `email-folder-model-input-v4` JSON document. It is bulk work. memory.preseen.ai's requests are almost all short.

## Rule

`INTERACTIVE_MAX_TOKENS=1024`: a request whose uncached inputs total at most 1024 tokens goes to the interactive queue, which runs on its own CPU model instance beside the bulk queue. Each interactive forward is capped at 1024 padded tokens.

CPU forward cost, one text, 8 cores (`taskset -c 8-15,24-31`, 8 torch threads), measured in-process:

| tokens | alone (p50) | beside a running 2×8192 forward (p50) |
|---:|---:|---:|
| 32 | 58 ms | 102 ms |
| 256 | 163 ms | 267 ms |
| 512 | 366 ms | 667 ms |
| 1024 | 976 ms | 1,644 ms |
| 2048 | 2,737 ms | |
| 8192 | 21.0 s | |

The bulk 2×8192 forward took 42.3 s alone and 48.3 s with interactive forwards running beside it (+14%).

## Burst benchmark

`scripts/benchmark_interactive_latency.py --duration 180`: 2 bulk clients send 8 documents of 2,407 tokens back to back. A probe starts every 2 s without waiting for earlier ones, alternating a 37-token query and a 307-token episode; every text is unique, so the cache never answers. Both runs used a throwaway instance on port 17996 with the gpu4 env (CPU, `CPU_BATCH_TARGET=8`, `MAX_LENGTH=8192`, `BATCH_WINDOW_MS=50`), pinned to 8 cores. The baseline was `origin/main` e53a3c5.

| | before (one queue) | after (`INTERACTIVE_MAX_TOKENS=1024`) |
|---|---:|---:|
| 1-text, all 90 probes | p50 203.5 s, p90 453.7 s, max 568.5 s | p50 0.33 s, p90 0.57 s, max 0.69 s |
| 37-token query (45) | p50 227.5 s, p90 451.9 s | p50 173 ms, p90 186 ms |
| 307-token episode (45) | p50 179.5 s, p90 453.9 s | p50 540 ms, p90 600 ms |
| bulk 8×2,407 request | p50 85.6 s (11 requests) | p50 78.5 s (6 requests) |

In the baseline the probes joined the bulk batches; in the new run they never did. The new run finished fewer bulk requests because it stopped sooner: it ends once the probes are answered, and the probes no longer waited.

## CUDA peak

`scripts/measure_cuda_peak.py` on GPU 1 (RTX 5090) at 19:0x, after llm-gateway's idle slot-1 replica had unloaded and before it respawned (19:06:13, still util 0.950). Every text was at 8192 tokens, `task=text-matching`, bf16:

| texts per forward | tokens | forward | torch peak allocated | process footprint (nvidia-smi) |
|---:|---:|---:|---:|---:|
| loaded | | | 1,445 MiB | 1,966 MiB |
| 1 | 8,192 | 0.50 s | 2,737 MiB | 3,438 MiB |
| 2 | 16,384 | 0.39 s | 4,017 MiB | 4,848 MiB |
| 4 | 32,768 | 0.78 s | 6,578 MiB | 8,048 MiB |
| 8 | 65,536 | 1.59 s | 11,698 MiB | 14,448 MiB |
| 16 | 131,072 | 3.11 s | 21,939 MiB | 27,248 MiB |

A slice for 8 texts × 8192 tokens (14.1 GiB) leaves the 27B less than its 24.83 GiB floor, so that is not an option. `CUDA_MAX_FORWARD_TOKENS=8192` caps the footprint at 3,438 MiB and keeps about 0.2 s per 8192-token text, 35–100 times the CPU rate (7–21 s per text). The matching llm-gateway setting is `GPU_NEIGHBOUR_RESERVE_MIB=1=4096`: the 3,438 MiB peak of one forward plus 658 MiB, because the caching allocator fragments over forwards of different shapes and an overflow would fail a single-text request (it cannot be split further). On a 5090 that caps the 27B on GPU 1 at util 0.851 instead of 0.950, about 3.2 GiB less KV cache on that replica; GPU 0 is unchanged.
