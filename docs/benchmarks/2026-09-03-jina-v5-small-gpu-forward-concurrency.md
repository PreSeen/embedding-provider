# jina-v5-small GPU Forward Concurrency Benchmark

| concurrency | aggregate texts/s | worst p95 ms | errors | GPU MiB | offload released | qualifies |
|---:|---:|---:|---:|---:|:---:|:---:|
| 1 | 8614.1 | 126.6 | 0 | 3317 | True | False |
| 2 | 8122.2 | 3272.5 | 0 | 2757 | True | False |
| 4 | 7960.5 | 2692.0 | 0 | 2785 | True | False |

## Decision

Retain `GPU_FORWARD_CONCURRENCY=1` for `jina-v5-small`. Candidates `2` and `4` both exercised their configured peak concurrency and released the worker successfully, but aggregate throughput fell by about 5.7% and 7.6%, respectively. Their worst workload P95 also regressed far beyond the allowed 10%. The single-forward configuration therefore remains the production optimum for this model and workload mix.

```json
[
  {
    "workloads": {
      "single_serial": {
        "p50_ms": 59.353,
        "p95_ms": 60.977,
        "requests_per_second": 730.264,
        "texts_per_second": 730.264,
        "error_count": 0,
        "errors": []
      },
      "batch_8_serial": {
        "p50_ms": 50.008,
        "p95_ms": 85.65,
        "requests_per_second": 202.132,
        "texts_per_second": 1617.062,
        "error_count": 0,
        "errors": []
      },
      "batch_64_serial": {
        "p50_ms": 92.18,
        "p95_ms": 126.616,
        "requests_per_second": 31.77,
        "texts_per_second": 2033.328,
        "error_count": 0,
        "errors": []
      },
      "single_concurrent_16": {
        "p50_ms": 4.244,
        "p95_ms": 55.091,
        "requests_per_second": 1322.947,
        "texts_per_second": 1322.947,
        "error_count": 0,
        "errors": []
      },
      "batch_8_concurrent_8": {
        "p50_ms": 11.662,
        "p95_ms": 68.532,
        "requests_per_second": 363.818,
        "texts_per_second": 2910.546,
        "error_count": 0,
        "errors": []
      }
    },
    "runtime": {
      "loaded_device": "cuda",
      "preferred_device": "cuda",
      "detected_device": "cuda",
      "precision": "bfloat16",
      "start_device": "cuda",
      "engine_state": "hot",
      "cuda_fallback_reason": null,
      "idle_offload_enabled": true,
      "idle_offload_seconds": 300.0,
      "idle_offload_poll_seconds": 30.0,
      "device_switch_min_seconds": 0.0,
      "device_switch_cooldown_remaining_seconds": 0.0,
      "cpu_batch_target": 8,
      "effective_batch_target": 32,
      "cpu_to_gpu_scale_up_texts": 8,
      "gpu_to_cpu_scale_down_texts": 8,
      "gpu_to_cpu_scale_down_seconds": 15.0,
      "gpu_low_batch_seconds": null,
      "inflight_encodes": 0,
      "gpu_forward_concurrency_configured": 1,
      "gpu_forward_concurrency_effective": 1,
      "gpu_forwards_in_flight": 0,
      "gpu_forward_peak_in_flight": 1,
      "idle_for_seconds": 0.033,
      "offloaded_for_seconds": null,
      "reload_in_progress": false,
      "worker_pid": 3278760,
      "adaptive_batch": {
        "current_target": 32,
        "initial_target": 1,
        "hard_cap": 64,
        "last_batch_texts": 28,
        "last_observed_demand": 28,
        "demand_ema": 25.999,
        "last_vram_cap": 64,
        "oom_target_ceiling": null,
        "adjustments_total": 12
      }
    },
    "gpu_memory_used_mb": 3317,
    "gpu_memory_total_mb": 32607,
    "safe_vram": true,
    "oom_count": 0,
    "offload_released": true,
    "candidate": 1
  },
  {
    "workloads": {
      "single_serial": {
        "p50_ms": 59.422,
        "p95_ms": 60.165,
        "requests_per_second": 736.534,
        "texts_per_second": 736.534,
        "error_count": 0,
        "errors": []
      },
      "batch_8_serial": {
        "p50_ms": 141.039,
        "p95_ms": 148.512,
        "requests_per_second": 187.804,
        "texts_per_second": 1502.436,
        "error_count": 0,
        "errors": []
      },
      "batch_64_serial": {
        "p50_ms": 909.325,
        "p95_ms": 3272.454,
        "requests_per_second": 27.616,
        "texts_per_second": 1767.455,
        "error_count": 0,
        "errors": []
      },
      "single_concurrent_16": {
        "p50_ms": 4.58,
        "p95_ms": 63.895,
        "requests_per_second": 1305.992,
        "texts_per_second": 1305.992,
        "error_count": 0,
        "errors": []
      },
      "batch_8_concurrent_8": {
        "p50_ms": 10.896,
        "p95_ms": 148.782,
        "requests_per_second": 351.221,
        "texts_per_second": 2809.77,
        "error_count": 0,
        "errors": []
      }
    },
    "runtime": {
      "loaded_device": "cuda",
      "preferred_device": "cuda",
      "detected_device": "cuda",
      "precision": "bfloat16",
      "start_device": "cuda",
      "engine_state": "hot",
      "cuda_fallback_reason": null,
      "idle_offload_enabled": true,
      "idle_offload_seconds": 300.0,
      "idle_offload_poll_seconds": 30.0,
      "device_switch_min_seconds": 0.0,
      "device_switch_cooldown_remaining_seconds": 0.0,
      "cpu_batch_target": 8,
      "effective_batch_target": 8,
      "cpu_to_gpu_scale_up_texts": 8,
      "gpu_to_cpu_scale_down_texts": 8,
      "gpu_to_cpu_scale_down_seconds": 15.0,
      "gpu_low_batch_seconds": 0.965,
      "inflight_encodes": 0,
      "gpu_forward_concurrency_configured": 2,
      "gpu_forward_concurrency_effective": 2,
      "gpu_forwards_in_flight": 0,
      "gpu_forward_peak_in_flight": 2,
      "idle_for_seconds": 0.027,
      "offloaded_for_seconds": null,
      "reload_in_progress": false,
      "worker_pid": 3282501,
      "adaptive_batch": {
        "current_target": 8,
        "initial_target": 1,
        "hard_cap": 64,
        "last_batch_texts": 2,
        "last_observed_demand": 28,
        "demand_ema": 23.0,
        "last_vram_cap": 64,
        "oom_target_ceiling": null,
        "adjustments_total": 8
      }
    },
    "gpu_memory_used_mb": 2757,
    "gpu_memory_total_mb": 32607,
    "safe_vram": true,
    "oom_count": 0,
    "offload_released": true,
    "candidate": 2
  },
  {
    "workloads": {
      "single_serial": {
        "p50_ms": 59.407,
        "p95_ms": 60.555,
        "requests_per_second": 764.121,
        "texts_per_second": 764.121,
        "error_count": 0,
        "errors": []
      },
      "batch_8_serial": {
        "p50_ms": 141.435,
        "p95_ms": 151.732,
        "requests_per_second": 184.126,
        "texts_per_second": 1473.009,
        "error_count": 0,
        "errors": []
      },
      "batch_64_serial": {
        "p50_ms": 955.827,
        "p95_ms": 2692.032,
        "requests_per_second": 28.546,
        "texts_per_second": 1826.953,
        "error_count": 0,
        "errors": []
      },
      "single_concurrent_16": {
        "p50_ms": 4.127,
        "p95_ms": 91.519,
        "requests_per_second": 1280.197,
        "texts_per_second": 1280.197,
        "error_count": 0,
        "errors": []
      },
      "batch_8_concurrent_8": {
        "p50_ms": 11.765,
        "p95_ms": 239.186,
        "requests_per_second": 327.023,
        "texts_per_second": 2616.177,
        "error_count": 0,
        "errors": []
      }
    },
    "runtime": {
      "loaded_device": "cuda",
      "preferred_device": "cuda",
      "detected_device": "cuda",
      "precision": "bfloat16",
      "start_device": "cuda",
      "engine_state": "hot",
      "cuda_fallback_reason": null,
      "idle_offload_enabled": true,
      "idle_offload_seconds": 300.0,
      "idle_offload_poll_seconds": 30.0,
      "device_switch_min_seconds": 0.0,
      "device_switch_cooldown_remaining_seconds": 0.0,
      "cpu_batch_target": 8,
      "effective_batch_target": 8,
      "cpu_to_gpu_scale_up_texts": 8,
      "gpu_to_cpu_scale_down_texts": 8,
      "gpu_to_cpu_scale_down_seconds": 15.0,
      "gpu_low_batch_seconds": 1.511,
      "inflight_encodes": 0,
      "gpu_forward_concurrency_configured": 4,
      "gpu_forward_concurrency_effective": 4,
      "gpu_forwards_in_flight": 0,
      "gpu_forward_peak_in_flight": 4,
      "idle_for_seconds": 0.028,
      "offloaded_for_seconds": null,
      "reload_in_progress": false,
      "worker_pid": 3285655,
      "adaptive_batch": {
        "current_target": 8,
        "initial_target": 1,
        "hard_cap": 64,
        "last_batch_texts": 1,
        "last_observed_demand": 28,
        "demand_ema": 23.0,
        "last_vram_cap": 64,
        "oom_target_ceiling": null,
        "adjustments_total": 8
      }
    },
    "gpu_memory_used_mb": 2785,
    "gpu_memory_total_mb": 32607,
    "safe_vram": true,
    "oom_count": 0,
    "offload_released": true,
    "candidate": 4
  }
]
```
