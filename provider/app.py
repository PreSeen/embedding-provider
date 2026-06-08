from __future__ import annotations

import asyncio
import functools
import gc
import inspect
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import threading
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from provider.config import Settings

log = logging.getLogger("embedding_provider")
logging.basicConfig(level="INFO")

_CUDA_VRAM_SAFETY_FIXED_BYTES = 512 * 1024 * 1024
_CUDA_VRAM_SAFETY_TOTAL_RATIO = 0.05
_CUDA_BATCH_GROWTH_FACTOR = 2


def _batched(values: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        return [values]
    return [values[idx: idx + batch_size] for idx in range(0, len(values), batch_size)]


def _is_cuda_oom(exc: Exception) -> bool:
    text = str(exc)
    return "CUDA out of memory" in text or "OutOfMemoryError" in text


def _resolve_gpu_index() -> str | None:
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return None
    first = visible.split(",", 1)[0].strip()
    if first.isdigit():
        return first
    return None


def _probe_cuda_memory_bytes() -> tuple[int | None, int | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    gpu_index = _resolve_gpu_index()
    if gpu_index is not None:
        command.insert(1, f"--id={gpu_index}")
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None, None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return None, None
    try:
        free_raw, total_raw = [part.strip() for part in lines[0].split(",", 1)]
        return int(free_raw) * 1024 * 1024, int(total_raw) * 1024 * 1024
    except Exception:
        return None, None


def _detect_preferred_device() -> str:
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None and visible_devices.strip().lower() in {"", "-1", "none", "cpu"}:
        return "cpu"
    free_bytes, _total_bytes = _probe_cuda_memory_bytes()
    return "cuda" if free_bytes is not None else "cpu"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _first_header_ip(value: str | None) -> str | None:
    if not value:
        return None
    first = value.split(",", 1)[0].strip()
    return first or None


def _country_for_ip(ip_text: str | None) -> str:
    if not ip_text:
        return "unknown"
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return "unknown"
    if ip.is_loopback:
        return "localhost"
    if ip.is_private or ip.is_link_local:
        return "LAN"
    return "unknown"


def _client_ip_and_country(request: Request) -> tuple[str | None, str]:
    headers = request.headers
    source_ip = (
        _first_header_ip(headers.get("cf-connecting-ip"))
        or _first_header_ip(headers.get("true-client-ip"))
        or _first_header_ip(headers.get("x-real-ip"))
        or _first_header_ip(headers.get("x-forwarded-for"))
        or (request.client.host if request.client else None)
    )
    country = (
        headers.get("cf-ipcountry")
        or headers.get("x-vercel-ip-country")
        or headers.get("x-country-code")
        or ""
    ).strip()
    return source_ip, country.upper() if country else _country_for_ip(source_ip)


def _summarize_exception(exc: Exception) -> str:
    detail = str(exc).strip()
    if detail:
        return f"{exc.__class__.__name__}: {detail}"
    return exc.__class__.__name__


class ProviderRuntimeStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests_total = 0
        self._requests_succeeded = 0
        self._requests_failed = 0
        self._texts_total = 0
        self._batches_total = 0
        self._batch_texts_total = 0
        self._last_batch_texts = 0
        self._running_batch_texts = 0
        self._queue_depth = 0
        self._oldest_queue_age_seconds = 0.0
        self._reloads_total = 0
        self._offloads_total = 0
        self._last_request_at: str | None = None
        self._last_request_id: str | None = None
        self._last_request_texts = 0
        self._last_success_at: str | None = None
        self._last_success_request_id: str | None = None
        self._last_error_at: str | None = None
        self._last_error_request_id: str | None = None
        self._last_error_summary: str | None = None
        self._last_duration_ms: float | None = None

    def record_request_start(self, *, request_id: str, text_count: int) -> None:
        with self._lock:
            self._requests_total += 1
            self._texts_total += text_count
            self._last_request_at = _utc_now_iso()
            self._last_request_id = request_id
            self._last_request_texts = max(0, text_count)

    def record_request_success(self, *, request_id: str, duration_ms: float) -> None:
        with self._lock:
            self._requests_succeeded += 1
            self._last_success_at = _utc_now_iso()
            self._last_success_request_id = request_id
            self._last_duration_ms = round(duration_ms, 3)

    def record_request_failure(self, *, request_id: str, duration_ms: float, error_summary: str) -> None:
        with self._lock:
            self._requests_failed += 1
            self._last_error_at = _utc_now_iso()
            self._last_error_request_id = request_id
            self._last_error_summary = error_summary
            self._last_duration_ms = round(duration_ms, 3)

    def record_batch_dispatch(self, *, request_count: int, text_count: int) -> None:
        with self._lock:
            self._batches_total += 1
            self._batch_texts_total += text_count
            self._last_batch_texts = max(0, text_count)

    def set_running_batch(self, *, text_count: int) -> None:
        with self._lock:
            self._running_batch_texts = max(0, text_count)

    def update_pending(self, *, depth: int, oldest_age_seconds: float | None) -> None:
        with self._lock:
            self._queue_depth = max(0, depth)
            self._oldest_queue_age_seconds = round(max(0.0, oldest_age_seconds or 0.0), 3)

    def record_reload(self) -> None:
        with self._lock:
            self._reloads_total += 1

    def record_offload(self) -> None:
        with self._lock:
            self._offloads_total += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests_total": self._requests_total,
                "requests_succeeded": self._requests_succeeded,
                "requests_failed": self._requests_failed,
                "texts_total": self._texts_total,
                "batches_total": self._batches_total,
                "batch_texts_total": self._batch_texts_total,
                "last_batch_texts": self._last_batch_texts,
                "running_batch_texts": self._running_batch_texts,
                "queue_depth": self._queue_depth,
                "oldest_queue_age_seconds": self._oldest_queue_age_seconds,
                "reloads_total": self._reloads_total,
                "offloads_total": self._offloads_total,
                "last_request_at": self._last_request_at,
                "last_request_id": self._last_request_id,
                "last_request_texts": self._last_request_texts,
                "last_success_at": self._last_success_at,
                "last_success_request_id": self._last_success_request_id,
                "last_error_at": self._last_error_at,
                "last_error_request_id": self._last_error_request_id,
                "last_error_summary": self._last_error_summary,
                "last_duration_ms": self._last_duration_ms,
            }


class RequestLogBuffer:
    def __init__(self, *, max_inputs: int = 10_000, max_requests: int = 10_000) -> None:
        self._lock = threading.Lock()
        self._inputs: deque[dict[str, Any]] = deque(maxlen=max_inputs)
        self._requests: deque[dict[str, Any]] = deque(maxlen=max_requests)
        self._request_index: dict[str, dict[str, Any]] = {}

    def record_start(
        self,
        *,
        request_id: str,
        source_ip: str | None,
        source_country: str,
        model: str | None,
        dimensions: int | None,
        task: str | None,
        texts: list[str],
    ) -> None:
        now_epoch = time.time()
        now_iso = _utc_now_iso()
        entry = {
            "request_id": request_id,
            "started_at": now_iso,
            "started_epoch": now_epoch,
            "source_ip": source_ip,
            "source_country": source_country,
            "model": model,
            "dimensions": dimensions,
            "task": task,
            "text_count": len(texts),
            "status": "running",
            "status_code": None,
            "duration_ms": None,
            "error_summary": None,
        }
        input_entries = [
            {
                "request_id": request_id,
                "input_index": idx,
                "received_at": now_iso,
                "source_ip": source_ip,
                "source_country": source_country,
                "model": model,
                "dimensions": dimensions,
                "task": task,
                "input": text,
            }
            for idx, text in enumerate(texts)
        ]
        with self._lock:
            self._requests.append(entry)
            self._request_index[request_id] = entry
            for item in input_entries:
                self._inputs.append(item)
            while len(self._request_index) > len(self._requests):
                active_ids = {item["request_id"] for item in self._requests}
                for old_id in list(self._request_index):
                    if old_id not in active_ids:
                        self._request_index.pop(old_id, None)

    def record_finish(
        self,
        *,
        request_id: str,
        status_code: int,
        duration_ms: float,
        error_summary: str | None = None,
    ) -> None:
        with self._lock:
            entry = self._request_index.get(request_id)
            if entry is None:
                return
            entry["status"] = "ok" if status_code < 400 else "error"
            entry["status_code"] = status_code
            entry["duration_ms"] = round(duration_ms, 3)
            entry["error_summary"] = error_summary

    def recent_inputs(self, *, limit: int = 200) -> list[dict[str, Any]]:
        safe_limit = max(1, min(10_000, int(limit)))
        with self._lock:
            return list(self._inputs)[-safe_limit:][::-1]

    def recent_requests(self, *, limit: int = 200) -> list[dict[str, Any]]:
        safe_limit = max(1, min(10_000, int(limit)))
        with self._lock:
            return [dict(item) for item in list(self._requests)[-safe_limit:][::-1]]

    def qps_buckets(self, *, window_seconds: int = 3600, bucket_seconds: int = 30) -> list[dict[str, Any]]:
        now = time.time()
        window_start = now - max(1, window_seconds)
        bucket_size = max(1, bucket_seconds)
        bucket_count = max(1, math.ceil(window_seconds / bucket_size))
        end_bucket = math.floor(now / bucket_size) * bucket_size
        start_bucket = end_bucket - (bucket_count - 1) * bucket_size
        buckets: list[dict[str, Any]] = []
        for idx in range(bucket_count):
            bucket_start = start_bucket + idx * bucket_size
            buckets.append(
                {
                    "start_epoch": bucket_start,
                    "start": datetime.fromtimestamp(bucket_start, timezone.utc).isoformat(),
                    "requests": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "texts": 0,
                    "qps": 0.0,
                    "tps": 0.0,
                    "avg_texts_per_request": 0.0,
                    "avg_duration_ms": None,
                    "p95_duration_ms": None,
                }
            )
        durations: list[list[float]] = [[] for _ in buckets]
        with self._lock:
            rows = [dict(item) for item in self._requests]
        for row in rows:
            started_epoch = float(row.get("started_epoch") or 0.0)
            if started_epoch < window_start or started_epoch > now:
                continue
            index = int((started_epoch - start_bucket) // bucket_size)
            if index < 0 or index >= len(buckets):
                continue
            bucket = buckets[index]
            bucket["requests"] += 1
            bucket["texts"] += int(row.get("text_count") or 0)
            status_code = row.get("status_code")
            if isinstance(status_code, int):
                if status_code < 400:
                    bucket["succeeded"] += 1
                else:
                    bucket["failed"] += 1
            duration = row.get("duration_ms")
            if isinstance(duration, (int, float)):
                durations[index].append(float(duration))
        for bucket, values in zip(buckets, durations):
            bucket["qps"] = round(bucket["requests"] / bucket_size, 4)
            bucket["tps"] = round(bucket["texts"] / bucket_size, 4)
            bucket["avg_texts_per_request"] = (
                round(bucket["texts"] / bucket["requests"], 2)
                if bucket["requests"]
                else 0.0
            )
            if values:
                values.sort()
                bucket["avg_duration_ms"] = round(sum(values) / len(values), 3)
                p95_index = min(len(values) - 1, math.ceil(len(values) * 0.95) - 1)
                bucket["p95_duration_ms"] = round(values[p95_index], 3)
        return buckets


class AdaptiveBatchState:
    _DEFAULT_MAX_TARGET = 64
    _MIN_TARGET = 1

    def __init__(self, configured_max_batch_size: int | None) -> None:
        configured_cap = configured_max_batch_size or self._DEFAULT_MAX_TARGET
        self._hard_cap = max(1, int(configured_cap))
        self._initial_target = min(self._MIN_TARGET, self._hard_cap)
        self._current_target = self._initial_target
        self._last_batch_texts = 0
        self._last_vram_cap: int | None = None
        self._adjustments_total = 0
        self._oom_target_ceiling: int | None = None
        self._lock = threading.Lock()

    @property
    def current_target(self) -> int:
        with self._lock:
            return self._current_target

    def effective_target(self, *, vram_cap: int | None = None) -> int:
        with self._lock:
            return self._effective_target_locked(vram_cap=vram_cap)

    def reset(self) -> None:
        with self._lock:
            self._current_target = self._initial_target
            self._last_batch_texts = 0
            self._last_vram_cap = None
            self._oom_target_ceiling = None
            self._adjustments_total += 1

    def record_successful_dispatch(self, *, text_count: int, vram_cap: int | None, allow_growth: bool) -> None:
        with self._lock:
            self._last_batch_texts = max(0, int(text_count))
            self._last_vram_cap = vram_cap if vram_cap is None else max(1, int(vram_cap))
            if not allow_growth or self._last_batch_texts < self._current_target:
                return
            next_target = min(self._hard_cap, max(1, self._current_target * _CUDA_BATCH_GROWTH_FACTOR))
            if self._last_vram_cap is not None:
                next_target = min(next_target, self._last_vram_cap)
            if self._oom_target_ceiling is not None:
                next_target = min(next_target, self._oom_target_ceiling)
            if next_target > self._current_target:
                self._current_target = next_target
                self._adjustments_total += 1

    def record_oom_backoff(self, *, failed_text_count: int) -> None:
        with self._lock:
            next_target = max(self._initial_target, min(self._hard_cap, max(1, int(failed_text_count) // 2)))
            self._oom_target_ceiling = (
                next_target
                if self._oom_target_ceiling is None
                else min(self._oom_target_ceiling, next_target)
            )
            if next_target < self._current_target:
                self._current_target = next_target
                self._adjustments_total += 1

    def snapshot(self) -> dict[str, int | None]:
        with self._lock:
            return {
                "current_target": self._current_target,
                "initial_target": self._initial_target,
                "hard_cap": self._hard_cap,
                "last_batch_texts": self._last_batch_texts,
                "last_vram_cap": self._last_vram_cap,
                "oom_target_ceiling": self._oom_target_ceiling,
                "adjustments_total": self._adjustments_total,
            }

    def _effective_target_locked(self, *, vram_cap: int | None) -> int:
        target = min(self._current_target, self._hard_cap)
        if vram_cap is not None:
            target = min(target, max(1, int(vram_cap)))
        return max(1, target)


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    dimensions: int | None = None
    encoding_format: Literal["float"] | None = "float"
    user: str | None = None
    task: str | None = None


class EmbeddingItem(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingItem]
    model: str
    usage: EmbeddingUsage = Field(default_factory=EmbeddingUsage)


class DeviceSwitchRequest(BaseModel):
    device: Literal["cpu", "cuda"]


class InputLengthValidator:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tokenizer: Any | None = None
        self._lock = threading.Lock()

    @property
    def max_length(self) -> int | None:
        return self._settings.max_length

    def token_counts(self, texts: list[str]) -> list[int]:
        tokenizer = self._get_tokenizer()
        counts: list[int] = []
        for text in texts:
            encoded = tokenizer(
                f"Document: {text}",
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            counts.append(len(encoded["input_ids"]))
        return counts

    def validate(self, texts: list[str]) -> list[int]:
        max_length = self.max_length
        if not max_length:
            return []
        counts = self.token_counts(texts)
        for index, count in enumerate(counts):
            if count > max_length:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"input[{index}] exceeds MAX_LENGTH={max_length}: "
                        f"{count} tokens"
                    ),
                )
        return counts

    def _get_tokenizer(self) -> Any:
        with self._lock:
            if self._tokenizer is None:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._settings.model_id,
                    trust_remote_code=self._settings.trust_remote_code,
                )
            return self._tokenizer


@dataclass
class _EncodeResult:
    embeddings: list[list[float]]
    max_forward_texts: int
    had_oom_backoff: bool = False


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "stardust"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


class _GpuEmbedderWorker:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._root_dir = Path(__file__).resolve().parents[1]
        self._process: subprocess.Popen[str] | None = None
        self._io_lock = threading.RLock()
        self._device_name = "none"

    @property
    def pid(self) -> int | None:
        if self._process is None:
            return None
        return self._process.pid

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def ensure_started(self) -> str:
        with self._io_lock:
            if self.is_running():
                return self._device_name
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["START_DEVICE"] = "cuda"
            if self._settings.cuda_visible_devices:
                env["CUDA_VISIBLE_DEVICES"] = self._settings.cuda_visible_devices
            self._process = subprocess.Popen(
                [sys.executable, "-m", "provider.gpu_worker"],
                cwd=str(self._root_dir),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            message = self._read_message()
            if str(message.get("status")) != "ready":
                self.terminate()
                raise RuntimeError(str(message.get("error") or "failed to start embedding GPU worker"))
            self._device_name = str(message.get("device") or "cuda")
            return self._device_name

    def encode(
        self,
        texts: list[str],
        *,
        dimensions: int | None = None,
        task: str | None = None,
    ) -> tuple[list[list[float]], float | None]:
        effective_task = task or self._settings.embedding_task
        response = self._request(
            {
                "op": "encode",
                "texts": texts,
                "dimensions": dimensions,
                "task": effective_task,
            }
        )
        embeddings = [[float(value) for value in row] for row in response.get("embeddings") or []]
        sample = response.get("sample_bytes_per_text")
        return embeddings, float(sample) if sample is not None else None

    def empty_cache(self) -> None:
        self._request({"op": "empty_cache"})

    def terminate(self) -> None:
        with self._io_lock:
            process = self._process
            self._process = None
            self._device_name = "none"
            if process is None:
                return
            if process.poll() is None:
                try:
                    self._write_message({"op": "shutdown"}, process=process)
                    process.wait(timeout=2)
                except Exception:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except Exception:
                        process.kill()
                        process.wait(timeout=5)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._io_lock:
            self.ensure_started()
            self._write_message(payload)
            response = self._read_message()
        if str(response.get("status")) != "ok":
            raise RuntimeError(str(response.get("error") or "embedding GPU worker request failed"))
        return response

    def _write_message(self, payload: dict[str, Any], *, process: subprocess.Popen[str] | None = None) -> None:
        target = process or self._process
        if target is None or target.stdin is None:
            raise RuntimeError("embedding GPU worker stdin is unavailable")
        target.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        target.stdin.flush()

    def _read_message(self) -> dict[str, Any]:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("embedding GPU worker stdout is unavailable")
        line = self._process.stdout.readline()
        if not line:
            return {"status": "error", "error": "embedding GPU worker exited before responding"}
        return json.loads(line)


class EmbedderRuntime:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._detected_device = _detect_preferred_device()
        if settings.start_device == "cpu":
            self._preferred_device = "cpu"
        elif settings.start_device == "cuda" and self._detected_device == "cuda":
            self._preferred_device = "cuda"
        else:
            self._preferred_device = self._detected_device
        self._device = self._preferred_device
        self.device_name = "none" if self._preferred_device == "cuda" else self._device
        self._model_lock = threading.Condition()
        self._use_gpu_worker = self._preferred_device == "cuda"
        self._gpu_worker: _GpuEmbedderWorker | None = _GpuEmbedderWorker(settings) if self._use_gpu_worker else None
        self._model = None if self._use_gpu_worker else self._load_model()
        self._encode_signature = None if self._model is None else inspect.signature(self._model.encode)
        self._bytes_per_text_ema: float | None = None
        self._ema_alpha = 0.25
        self._input_length_validator = InputLengthValidator(settings)
        self.adaptive_batch = AdaptiveBatchState(settings.max_batch_size)
        self._engine_state = "offloaded" if self._use_gpu_worker else "hot"
        self._reload_in_progress = False
        self._inflight_encodes = 0
        self._last_encode_finished_at = time.monotonic()
        self._last_offloaded_at: float | None = None
        self._gpu_low_batch_since: float | None = None
        self._idle_offload_enabled = self._detected_device == "cuda" and settings.idle_offload_seconds > 0
        self._stats: ProviderRuntimeStats | None = None
        self._cuda_fallback_reason: str | None = None

    def attach_stats(self, stats: ProviderRuntimeStats) -> None:
        self._stats = stats

    def _resolve_dtype(self) -> Any:
        import torch

        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return mapping.get(self._settings.dtype.lower(), torch.bfloat16)

    def _load_model(self, device_override: str | None = None) -> Any:
        target_device = device_override or self._preferred_device
        hide_cuda = str(target_device) == "cpu"
        old_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if hide_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            from transformers import AutoModel

            kwargs: dict[str, Any] = {
                "trust_remote_code": self._settings.trust_remote_code,
                "torch_dtype": self._resolve_dtype(),
            }
            if self._settings.attn_implementation:
                kwargs["attn_implementation"] = self._settings.attn_implementation
            try:
                model = AutoModel.from_pretrained(self._settings.model_id, **kwargs)
            except Exception:
                log.warning("model load with custom attention failed; retrying without attn_implementation", exc_info=True)
                kwargs.pop("attn_implementation", None)
                model = AutoModel.from_pretrained(self._settings.model_id, **kwargs)
            if hasattr(model, "to"):
                model = model.to(target_device)
            if hasattr(model, "eval"):
                model.eval()
            return model
        finally:
            if hide_cuda:
                if old_visible_devices is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old_visible_devices

    def runtime_status(self) -> dict[str, Any]:
        with self._model_lock:
            idle_for = max(0.0, time.monotonic() - self._last_encode_finished_at)
            offloaded_for = None
            if self._last_offloaded_at is not None:
                offloaded_for = max(0.0, time.monotonic() - self._last_offloaded_at)
            return {
                "loaded_device": self.device_name,
                "preferred_device": self._preferred_device,
                "detected_device": self._detected_device,
                "precision": self._settings.dtype,
                "start_device": self._settings.start_device,
                "engine_state": self._engine_state,
                "cuda_fallback_reason": self._cuda_fallback_reason,
                "idle_offload_enabled": self._idle_offload_enabled,
                "idle_offload_seconds": self._settings.idle_offload_seconds if self._idle_offload_enabled else None,
                "idle_offload_poll_seconds": (
                    self._settings.idle_offload_poll_seconds if self._idle_offload_enabled else None
                ),
                "cpu_batch_target": self._settings.cpu_batch_target,
                "effective_batch_target": self.estimate_max_texts(),
                "cpu_to_gpu_scale_up_texts": self._settings.cpu_to_gpu_scale_up_texts,
                "gpu_to_cpu_scale_down_texts": self._settings.gpu_to_cpu_scale_down_texts,
                "gpu_to_cpu_scale_down_seconds": self._settings.gpu_to_cpu_scale_down_seconds,
                "gpu_low_batch_seconds": (
                    round(max(0.0, time.monotonic() - self._gpu_low_batch_since), 3)
                    if self._gpu_low_batch_since is not None else None
                ),
                "inflight_encodes": self._inflight_encodes,
                "idle_for_seconds": round(idle_for, 3),
                "offloaded_for_seconds": round(offloaded_for, 3) if offloaded_for is not None else None,
                "reload_in_progress": self._reload_in_progress,
                "worker_pid": self._gpu_worker.pid if self._gpu_worker is not None and self._gpu_worker.is_running() else None,
                "adaptive_batch": self.adaptive_batch.snapshot(),
            }

    def close(self) -> None:
        if self._gpu_worker is not None:
            self._gpu_worker.terminate()

    def maybe_offload_idle(self) -> bool:
        if not self._idle_offload_enabled:
            return False
        with self._model_lock:
            if self._reload_in_progress or self.device_name == "cpu" or self._inflight_encodes > 0:
                return False
            low_batch_since = self._gpu_low_batch_since
            should_offload = False
            if (
                low_batch_since is not None
                and self._settings.gpu_to_cpu_scale_down_texts > 0
                and self._settings.gpu_to_cpu_scale_down_seconds >= 0
            ):
                low_for = time.monotonic() - low_batch_since
                if low_for >= self._settings.gpu_to_cpu_scale_down_seconds:
                    log.info(
                        "GPU stayed below scale-down threshold for %.1fs without new requests; switching to CPU",
                        low_for,
                    )
                    should_offload = True
            if not should_offload:
                idle_for = time.monotonic() - self._last_encode_finished_at
                should_offload = idle_for >= self._settings.idle_offload_seconds
            if not should_offload:
                return False
        self._switch_to_cpu(mark_offload=True)
        return True

    def _ensure_hot_model(self) -> None:
        if not self._idle_offload_enabled:
            return
        with self._model_lock:
            while self._reload_in_progress:
                self._model_lock.wait()
            if self._engine_state == "hot":
                return
            self._reload_in_progress = True
        try:
            if self._use_gpu_worker:
                if self._gpu_worker is None:
                    raise RuntimeError("embedding GPU worker is unavailable")
                try:
                    device_name = self._gpu_worker.ensure_started()
                except Exception as exc:
                    self._fallback_to_cpu(exc)
                    return
                with self._model_lock:
                    self._device = device_name
                    self.device_name = device_name
                    self._engine_state = "hot"
                    self._last_offloaded_at = None
            else:
                next_model = self._load_model(device_override=self._preferred_device)
                self._swap_model(next_model, self._preferred_device, engine_state="hot")
            if self._stats is not None:
                self._stats.record_reload()
        finally:
            with self._model_lock:
                self._reload_in_progress = False
                self._model_lock.notify_all()

    def _maybe_scale_up_for_request(self, text_count: int) -> None:
        if (
            self._settings.cpu_to_gpu_scale_up_texts <= 0
            or text_count < self._settings.cpu_to_gpu_scale_up_texts
            or self._detected_device != "cuda"
            or self._cuda_fallback_reason is not None
        ):
            return
        self._switch_to_cuda(wait_for_idle=False)

    def switch_device(self, target_device: str) -> dict[str, Any]:
        if target_device == "cpu":
            self._switch_to_cpu(mark_offload=False)
            return self.runtime_status()
        if target_device == "cuda":
            self._switch_to_cuda(wait_for_idle=True)
            return self.runtime_status()
        raise ValueError(f"unsupported device: {target_device}")

    def _switch_to_cpu(self, *, mark_offload: bool) -> None:
        with self._model_lock:
            while self._reload_in_progress or self._inflight_encodes > 0:
                self._model_lock.wait()
            if self.device_name == "cpu" and not self._use_gpu_worker:
                return
            self._reload_in_progress = True
        try:
            old_model = None
            if self._use_gpu_worker:
                if self._gpu_worker is not None:
                    self._gpu_worker.terminate()
                self._gpu_worker = None
            else:
                old_model = self._model

            next_model = self._load_model(device_override="cpu")
            with self._model_lock:
                self._model = next_model
                self._encode_signature = inspect.signature(self._model.encode)
                self._use_gpu_worker = False
                self._preferred_device = "cpu"
                self._device = "cpu"
                self.device_name = "cpu"
                self._engine_state = "hot"
                self._last_offloaded_at = time.monotonic() if mark_offload else None
            if old_model is not None:
                del old_model
                gc.collect()
            self.adaptive_batch.reset()
            with self._model_lock:
                self._gpu_low_batch_since = None
            if self._stats is not None:
                if mark_offload:
                    self._stats.record_offload()
                else:
                    self._stats.record_reload()
        finally:
            with self._model_lock:
                self._reload_in_progress = False
                self._model_lock.notify_all()

    def _switch_to_cuda(self, *, wait_for_idle: bool) -> None:
        if self._detected_device != "cuda":
            raise RuntimeError("CUDA is not available for this runtime")
        if self._cuda_fallback_reason is not None:
            raise RuntimeError(f"CUDA fallback is active: {self._cuda_fallback_reason}")
        with self._model_lock:
            while self._reload_in_progress or (wait_for_idle and self._inflight_encodes > 0):
                self._model_lock.wait()
            if self._use_gpu_worker or self.device_name != "cpu":
                return
            self._reload_in_progress = True
        worker = _GpuEmbedderWorker(self._settings)
        try:
            try:
                device_name = worker.ensure_started()
            except Exception as exc:
                worker.terminate()
                self._fallback_to_cpu(exc)
                return
            with self._model_lock:
                old_model = self._model
                self._model = None
                self._encode_signature = None
                self._gpu_worker = worker
                self._use_gpu_worker = True
                self._preferred_device = "cuda"
                self._device = device_name
                self.device_name = device_name
                self._engine_state = "hot"
                self._last_offloaded_at = None
                self._gpu_low_batch_since = None
            if old_model is not None:
                del old_model
                gc.collect()
            self.adaptive_batch.reset()
            if self._stats is not None:
                self._stats.record_reload()
        finally:
            with self._model_lock:
                self._reload_in_progress = False
                self._model_lock.notify_all()

    def _begin_encode(self, text_count: int) -> None:
        with self._model_lock:
            self._inflight_encodes += 1
            self._model_lock.notify_all()
        try:
            self._ensure_hot_model()
            self._maybe_scale_up_for_request(text_count)
        except Exception:
            self._finish_encode()
            raise

    def _finish_encode(self) -> None:
        with self._model_lock:
            self._inflight_encodes = max(0, self._inflight_encodes - 1)
            self._last_encode_finished_at = time.monotonic()
            self._model_lock.notify_all()

    def _release_cuda_cache(self) -> None:
        if self._preferred_device != "cuda":
            return
        try:
            if self._use_gpu_worker:
                if self._gpu_worker is not None and self._gpu_worker.is_running():
                    self._gpu_worker.empty_cache()
                return
            import torch

            torch.cuda.empty_cache()
        except Exception:
            log.warning("failed to release CUDA cache after adaptive batch shrink", exc_info=True)

    def _fallback_to_cpu(self, exc: Exception) -> None:
        reason = str(exc) or repr(exc)
        log.warning("embedding GPU worker failed to start; falling back to CPU: %s", reason)
        if self._gpu_worker is not None:
            self._gpu_worker.terminate()
        next_model = self._load_model(device_override="cpu")
        with self._model_lock:
            self._gpu_worker = None
            self._use_gpu_worker = False
            self._preferred_device = "cpu"
            self._device = "cpu"
            self.device_name = "cpu"
            self._model = next_model
            self._encode_signature = inspect.signature(self._model.encode)
            self._engine_state = "hot"
            self._last_offloaded_at = None
            self._idle_offload_enabled = False
            self._cuda_fallback_reason = reason

    def _swap_model(self, next_model: Any, next_device: str, *, engine_state: str) -> None:
        if self._use_gpu_worker:
            raise RuntimeError("_swap_model is unavailable in GPU worker mode")
        with self._model_lock:
            old_model = self._model
            self._model = next_model
            self._encode_signature = inspect.signature(self._model.encode)
            self._device = next_device
            self.device_name = str(next_device)
            self._engine_state = engine_state
            if engine_state == "offloaded":
                self._last_offloaded_at = time.monotonic()
            else:
                self._last_offloaded_at = None
        if old_model is not next_model:
            del old_model
            gc.collect()
            if self._preferred_device == "cuda":
                import torch

                torch.cuda.empty_cache()

    def _encode_once(self, texts: list[str], *, dimensions: int | None = None, task: str | None = None) -> list[list[float]]:
        if self._use_gpu_worker:
            if self._gpu_worker is None:
                raise RuntimeError("embedding GPU worker is unavailable")
            try:
                device_name = self._gpu_worker.ensure_started()
            except Exception as exc:
                self._fallback_to_cpu(exc)
                return self._encode_once(texts, dimensions=dimensions, task=task)
            with self._model_lock:
                self._device = device_name
                self.device_name = device_name
                self._engine_state = "hot"
                self._last_offloaded_at = None
            embeddings, sample = self._gpu_worker.encode(texts, dimensions=dimensions, task=task)
            if sample is not None and sample > 0:
                if self._bytes_per_text_ema is None:
                    self._bytes_per_text_ema = sample
                else:
                    self._bytes_per_text_ema = self._ema_alpha * sample + (1 - self._ema_alpha) * self._bytes_per_text_ema
            return embeddings
        kwargs: dict[str, Any] = {}
        if self._encode_signature is None:
            raise RuntimeError("embedding model signature is unavailable")
        input_name = "texts" if "texts" in self._encode_signature.parameters else "sentences"
        requested_dimensions = dimensions or self._settings.default_dimensions
        effective_task = task or self._settings.embedding_task
        if effective_task and "task" in self._encode_signature.parameters:
            kwargs["task"] = effective_task
        if requested_dimensions and "truncate_dim" in self._encode_signature.parameters:
            kwargs["truncate_dim"] = requested_dimensions
        if self._settings.max_length and "max_length" in self._encode_signature.parameters:
            kwargs["max_length"] = self._settings.max_length

        import torch

        track_mem = self._preferred_device == "cuda"
        if track_mem:
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated()

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self._model.encode(**{input_name: texts}, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if track_mem:
            delta = torch.cuda.max_memory_allocated() - mem_before
            if delta > 0:
                sample = delta / len(texts)
                if self._bytes_per_text_ema is None:
                    self._bytes_per_text_ema = sample
                else:
                    self._bytes_per_text_ema = self._ema_alpha * sample + (1 - self._ema_alpha) * self._bytes_per_text_ema

        log.info(
            "model=%s texts=%s elapsed_ms=%.1f bytes_per_text=%.0f",
            self._settings.model_id, len(texts), elapsed_ms,
            self._bytes_per_text_ema or 0,
        )

        array = _outputs_to_numpy(outputs)
        if array.ndim == 1:
            array = np.expand_dims(array, axis=0)
        if requested_dimensions and "truncate_dim" not in self._encode_signature.parameters:
            if requested_dimensions > array.shape[1]:
                raise ValueError(f"Requested dimensions={requested_dimensions} but model only produced {array.shape[1]}")
            array = array[:, :requested_dimensions]
        if self._settings.normalize_embeddings:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            array = array / norms
        return array.tolist()

    def _encode_with_backoff(
        self,
        texts: list[str],
        *,
        dimensions: int | None = None,
        task: str | None = None,
    ) -> _EncodeResult:
        try:
            return _EncodeResult(
                embeddings=self._encode_once(texts, dimensions=dimensions, task=task),
                max_forward_texts=len(texts),
            )
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            self._release_cuda_cache()
            if self._preferred_device == "cuda":
                self.adaptive_batch.record_oom_backoff(failed_text_count=len(texts))
            token_count = self._diagnostic_token_count(texts)
            char_count = sum(len(text) for text in texts)
            free_vram, total_vram = _probe_cuda_memory_bytes()
            log.warning(
                "CUDA OOM for model=%s batch_size=%s token_count=%s char_count=%d "
                "max_length=%s device=%s free_vram_bytes=%s total_vram_bytes=%s",
                self._settings.model_id,
                len(texts),
                token_count,
                char_count,
                self._settings.max_length,
                self.device_name,
                free_vram,
                total_vram,
            )
            if len(texts) <= 1:
                try:
                    log.warning(
                        "retrying single-input CUDA OOM after empty_cache: model=%s token_count=%s "
                        "char_count=%d max_length=%s device=%s",
                        self._settings.model_id,
                        token_count,
                        char_count,
                        self._settings.max_length,
                        self.device_name,
                    )
                    return _EncodeResult(
                        embeddings=self._encode_once(texts, dimensions=dimensions, task=task),
                        max_forward_texts=1,
                        had_oom_backoff=True,
                    )
                except RuntimeError as retry_exc:
                    if not _is_cuda_oom(retry_exc):
                        raise
                    self._release_cuda_cache()
                    retry_free_vram, retry_total_vram = _probe_cuda_memory_bytes()
                    log.warning(
                        "single-input CUDA OOM persisted after empty_cache: model=%s token_count=%s "
                        "char_count=%d max_length=%s device=%s free_vram_bytes=%s total_vram_bytes=%s",
                        self._settings.model_id,
                        token_count,
                        char_count,
                        self._settings.max_length,
                        self.device_name,
                        retry_free_vram,
                        retry_total_vram,
                    )
                raise ValueError(
                    "Embedding request exceeded GPU memory for a single input after CUDA cache retry; "
                    f"token_count={token_count}, char_count={char_count}, MAX_LENGTH={self._settings.max_length}"
                ) from exc
            split_at = max(1, len(texts) // 2)
            log.warning(
                "retrying CUDA OOM with smaller batches: model=%s batch_size=%s",
                self._settings.model_id,
                len(texts),
            )
            left = self._encode_with_backoff(texts[:split_at], dimensions=dimensions, task=task)
            right = self._encode_with_backoff(texts[split_at:], dimensions=dimensions, task=task)
            return _EncodeResult(
                embeddings=[*left.embeddings, *right.embeddings],
                max_forward_texts=max(left.max_forward_texts, right.max_forward_texts),
                had_oom_backoff=True,
            )

    def _diagnostic_token_count(self, texts: list[str]) -> int | None:
        try:
            return sum(self._input_length_validator.token_counts(texts))
        except Exception:
            return None

    def encode(
        self,
        texts: list[str],
        *,
        dimensions: int | None = None,
        task: str | None = None,
        allow_batch_growth: bool = False,
    ) -> list[list[float]]:
        self._begin_encode(len(texts))
        succeeded = False
        try:
            embeddings: list[list[float]] = []
            offset = 0
            while offset < len(texts):
                remaining = len(texts) - offset
                max_batch_size = self._settings.max_batch_size or remaining
                vram_cap: int | None = None
                chunk_target = max_batch_size
                if self._preferred_device == "cuda":
                    if self.adaptive_batch.current_target <= 1:
                        vram_cap = self._estimate_vram_text_cap()
                    chunk_target = min(max_batch_size, self.adaptive_batch.effective_target(vram_cap=vram_cap))
                chunk_size = max(1, min(chunk_target, remaining))
                chunk = texts[offset: offset + chunk_size]
                result = self._encode_with_backoff(chunk, dimensions=dimensions, task=task)
                embeddings.extend(result.embeddings)
                offset += chunk_size
                if self._preferred_device == "cuda":
                    vram_cap = self._estimate_vram_text_cap()
                    self.adaptive_batch.record_successful_dispatch(
                        text_count=result.max_forward_texts,
                        vram_cap=vram_cap,
                        allow_growth=(allow_batch_growth or offset < len(texts)) and not result.had_oom_backoff,
                    )
            succeeded = True
            return embeddings
        finally:
            self._finish_encode()
            if succeeded:
                self._maybe_scale_down_for_request(len(texts))

    def _maybe_scale_down_for_request(self, text_count: int) -> None:
        if (
            self._settings.gpu_to_cpu_scale_down_texts <= 0
            or self._settings.gpu_to_cpu_scale_down_seconds < 0
            or self._detected_device != "cuda"
            or self._preferred_device != "cuda"
            or not self._use_gpu_worker
        ):
            return

        now = time.monotonic()
        with self._model_lock:
            if text_count >= self._settings.gpu_to_cpu_scale_down_texts:
                self._gpu_low_batch_since = None
                return
            if self._gpu_low_batch_since is None:
                self._gpu_low_batch_since = now
                return
            low_for = now - self._gpu_low_batch_since
            if (
                low_for < self._settings.gpu_to_cpu_scale_down_seconds
                or self._reload_in_progress
                or self._inflight_encodes > 0
            ):
                return

        log.info(
            "GPU batch stayed below scale-down threshold: texts=%d threshold=%d seconds=%.1f; switching to CPU",
            text_count,
            self._settings.gpu_to_cpu_scale_down_texts,
            low_for,
        )
        self._switch_to_cpu(mark_offload=True)

    def estimate_max_texts(self) -> int:
        if self._preferred_device == "cuda":
            vram_cap = self._estimate_vram_text_cap() if self.adaptive_batch.current_target <= 1 else None
            return self.adaptive_batch.effective_target(vram_cap=vram_cap)
        vram_cap = self._estimate_vram_text_cap()
        return min(vram_cap, max(1, self._settings.cpu_batch_target))

    def _estimate_vram_text_cap(self) -> int:
        """Return a safe upper bound on texts for the next forward pass based on free VRAM."""
        hard_cap = self._settings.max_batch_size
        if self._preferred_device != "cuda":
            return hard_cap or 256

        free, total = _probe_cuda_memory_bytes()
        if free is None or total is None:
            return hard_cap or 256
        safety = _CUDA_VRAM_SAFETY_FIXED_BYTES + int(total * _CUDA_VRAM_SAFETY_TOTAL_RATIO)
        usable = free - safety
        if usable <= 0 or self._bytes_per_text_ema is None or self._bytes_per_text_ema <= 0:
            estimate = 1
        else:
            estimate = max(1, int(usable / self._bytes_per_text_ema))

        if hard_cap:
            estimate = min(estimate, hard_cap)
        return max(1, estimate)


@dataclass
class _PendingRequestState:
    future: asyncio.Future  # resolved with list[list[float]]
    embeddings: list[list[float] | None]
    remaining: int


@dataclass
class _PendingItem:
    text: str
    input_index: int
    dimensions: int | None
    task: str | None
    state: _PendingRequestState
    enqueued_at: float
    request_id: str


class ContinuousBatcher:
    """Collects requests within a time window and dispatches them as a single batch.

    The worker loop is sequential: it collects a window of requests, runs one
    GPU forward pass (per task/dimensions group), resolves all futures, then
    starts the next window. This keeps GPU jobs serialized while requests queue
    up naturally during inference.
    """

    def __init__(self, runtime: EmbedderRuntime, stats: ProviderRuntimeStats, window_secs: float) -> None:
        self._runtime = runtime
        self._stats = stats
        self._window = window_secs
        self._queue: asyncio.Queue[_PendingItem] = asyncio.Queue()
        self._pending_enqueued_at: deque[float] = deque()

    def _refresh_pending_snapshot(self) -> None:
        oldest_age = None
        if self._pending_enqueued_at:
            oldest_age = time.monotonic() - self._pending_enqueued_at[0]
        self._stats.update_pending(depth=len(self._pending_enqueued_at), oldest_age_seconds=oldest_age)

    def _mark_enqueued(self, enqueued_at: float) -> None:
        self._pending_enqueued_at.append(enqueued_at)
        self._refresh_pending_snapshot()

    def _mark_processed(self, count: int) -> None:
        for _ in range(max(0, count)):
            if not self._pending_enqueued_at:
                break
            self._pending_enqueued_at.popleft()
        self._refresh_pending_snapshot()

    def start(self) -> asyncio.Task:
        return asyncio.create_task(self._worker())

    async def encode(
        self,
        texts: list[str],
        *,
        dimensions: int | None = None,
        task: str | None = None,
        request_id: str,
    ) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[list[float]]] = loop.create_future()
        state = _PendingRequestState(
            future=future,
            embeddings=[None] * len(texts),
            remaining=len(texts),
        )
        for input_index, text in enumerate(texts):
            enqueued_at = time.monotonic()
            await self._queue.put(
                _PendingItem(
                    text=text,
                    input_index=input_index,
                    dimensions=dimensions,
                    task=task,
                    state=state,
                    enqueued_at=enqueued_at,
                    request_id=request_id,
                )
            )
            self._mark_enqueued(enqueued_at)
        return await future

    async def _worker(self) -> None:
        loop = asyncio.get_running_loop()
        pending: list[_PendingItem] = []
        while True:
            if not pending:
                # Block until at least one request arrives.
                pending.append(await self._queue.get())

                # Drain for the remaining window.
                deadline = loop.time() + self._window
                while True:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                        pending.append(item)
                    except (asyncio.TimeoutError, TimeoutError):
                        break
            else:
                ready_items: list[_PendingItem] = []
                while True:
                    try:
                        ready_items.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                if ready_items:
                    pending = [pending[0], *ready_items, *pending[1:]]

            # Dispatch; overflow (texts that didn't fit in VRAM) is returned
            # and processed immediately in the next iteration without a new window wait.
            pending = await self._dispatch(pending, loop)

    async def _dispatch(self, batch: list[_PendingItem], loop: asyncio.AbstractEventLoop) -> list[_PendingItem]:
        stale_count = sum(1 for item in batch if item.state.future.done())
        if stale_count:
            batch = [item for item in batch if not item.state.future.done()]
            self._mark_processed(stale_count)
        if not batch:
            return []

        # Group by (task, dimensions) — in practice almost always one group.
        groups: dict[tuple, list[_PendingItem]] = defaultdict(list)
        for item in batch:
            groups[(item.task, item.dimensions)].append(item)

        overflow: list[_PendingItem] = []

        for (task, dimensions), items in groups.items():
            # Cap this group by available VRAM; overflow is deferred to next iteration.
            max_texts = max(1, self._runtime.estimate_max_texts())
            to_process = items[:max_texts]
            overflow.extend(items[max_texts:])
            group_overflow = len(items) - len(to_process)
            all_texts = [item.text for item in to_process]
            text_count = len(all_texts)

            log.info(
                "batch dispatch: requests=%d texts=%d overflow=%d task=%s dim=%s",
                len({item.request_id for item in to_process}), text_count, group_overflow, task, dimensions,
            )
            self._stats.record_batch_dispatch(request_count=len({item.request_id for item in to_process}), text_count=text_count)
            self._stats.set_running_batch(text_count=text_count)
            fn = functools.partial(
                self._runtime.encode,
                all_texts,
                dimensions=dimensions,
                task=task,
                allow_batch_growth=group_overflow > 0,
            )
            try:
                embeddings: list[list[float]] = await loop.run_in_executor(None, fn)
                for item, embedding in zip(to_process, embeddings):
                    state = item.state
                    if state.future.done():
                        continue
                    state.embeddings[item.input_index] = embedding
                    state.remaining -= 1
                    if state.remaining <= 0:
                        state.future.set_result([row or [] for row in state.embeddings])
            except Exception as exc:
                for item in to_process:
                    if not item.state.future.done():
                        item.state.future.set_exception(exc)
            finally:
                self._stats.set_running_batch(text_count=0)
                self._mark_processed(len(to_process))

        return overflow


def _outputs_to_numpy(outputs: Any) -> np.ndarray:
    import torch

    if isinstance(outputs, torch.Tensor):
        return outputs.detach().float().cpu().numpy()
    if isinstance(outputs, np.ndarray):
        return outputs.astype(np.float32, copy=False)
    if isinstance(outputs, list):
        rows: list[np.ndarray] = []
        for item in outputs:
            if isinstance(item, torch.Tensor):
                rows.append(item.detach().float().cpu().numpy())
            else:
                rows.append(np.asarray(item, dtype=np.float32))
        return np.stack(rows, axis=0)
    return np.asarray(outputs, dtype=np.float32)


def _estimate_tokens(texts: list[str]) -> int:
    return sum(max(1, math.ceil(len(text) / 4)) for text in texts)


def _validate_inputs(value: str | list[str]) -> list[str]:
    texts = [value] if isinstance(value, str) else value
    if not texts:
        raise HTTPException(status_code=400, detail="input must not be empty")
    if not all(isinstance(item, str) and item.strip() for item in texts):
        raise HTTPException(status_code=400, detail="all inputs must be non-empty strings")
    return texts


def _require_api_key(settings: Settings, authorization: str | None) -> None:
    if not settings.api_key:
        return
    expected = f"Bearer {settings.api_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


def _dashboard_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Embedding Provider Dashboard</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; background: #111; color: #eee; }
    body { margin: 0; padding: 18px; }
    header { display: flex; gap: 16px; align-items: baseline; margin-bottom: 14px; }
    h1 { font-size: 20px; margin: 0; }
    .muted { color: #aaa; font-size: 13px; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-bottom: 14px; }
    .card { border: 1px solid #333; border-radius: 6px; padding: 10px; background: #181818; position: relative; }
    .card[data-title] { cursor: help; }
    .label { color: #aaa; font-size: 12px; }
    .value { font-size: 18px; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .value.good { color: #8fd18f; }
    .value.warn { color: #f1c75b; }
    .value.bad { color: #ff8a8a; }
    #card-tooltip {
      position: fixed; z-index: 20; display: none; max-width: min(520px, calc(100vw - 24px));
      padding: 9px 11px; border: 1px solid #444; border-radius: 6px;
      background: rgba(18, 18, 18, 0.97); color: #eee; box-shadow: 0 10px 28px rgba(0,0,0,0.4);
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      white-space: pre-wrap; overflow-wrap: anywhere; pointer-events: none;
    }
    qps-chart { display: block; height: 260px; margin-bottom: 14px; }
    table { width: 100%; border-collapse: collapse; margin-top: 14px; table-layout: fixed; }
    th, td { border-bottom: 1px solid #333; padding: 7px; text-align: left; vertical-align: top; font-size: 12px; }
    th { color: #bbb; font-weight: 600; position: sticky; top: 0; background: #111; }
    td { word-break: break-word; }
    .input { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .ok { color: #8fd18f; }
    .error { color: #ff8a8a; }
  </style>
</head>
<body>
  <header>
    <h1>Embedding Provider</h1>
    <div class="muted" id="updated">loading</div>
  </header>
  <section class="grid" id="cards"></section>
  <div id="card-tooltip" role="tooltip"></div>
  <qps-chart id="qps"></qps-chart>
  <table>
    <thead>
      <tr>
        <th style="width: 150px">time</th>
        <th style="width: 110px">source</th>
        <th style="width: 80px">country</th>
        <th>input</th>
      </tr>
    </thead>
    <tbody id="inputs"></tbody>
  </table>
<script>
const cards = document.getElementById("cards");
const updated = document.getElementById("updated");
const qpsChart = document.getElementById("qps");
const inputs = document.getElementById("inputs");
const cardTooltip = document.getElementById("card-tooltip");

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[ch]));
}

function showCardTooltip(target, event) {
  const title = target.dataset.title;
  if (!title) return;
  cardTooltip.textContent = title;
  cardTooltip.style.display = "block";
  positionCardTooltip(event);
}

function positionCardTooltip(event) {
  if (cardTooltip.style.display !== "block") return;
  const gap = 12;
  const rect = cardTooltip.getBoundingClientRect();
  let left = event.clientX + gap;
  let top = event.clientY + gap;
  if (left + rect.width > window.innerWidth - gap) {
    left = Math.max(gap, event.clientX - rect.width - gap);
  }
  if (top + rect.height > window.innerHeight - gap) {
    top = Math.max(gap, event.clientY - rect.height - gap);
  }
  cardTooltip.style.left = `${left}px`;
  cardTooltip.style.top = `${top}px`;
}

function hideCardTooltip() {
  cardTooltip.style.display = "none";
}

function durationTone(durationMs) {
  const value = Number(durationMs);
  if (!Number.isFinite(value)) return "";
  if (value < 250) return "good";
  if (value < 1000) return "warn";
  return "bad";
}

class QpsChart extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this.buckets = [];
    this.hoverIndex = null;
    this.shadowRoot.innerHTML = `
      <style>
        :host { position: relative; }
        .frame {
          position: relative; height: 100%; border: 1px solid #333; border-radius: 6px;
          background: #151515; box-sizing: border-box; overflow: hidden;
        }
        .chart-head {
          position: absolute; top: 10px; left: 14px; right: 14px; z-index: 1;
          display: flex; justify-content: space-between; align-items: baseline;
          color: #ddd; font: 600 13px system-ui;
        }
        .chart-head span { color: #999; font-weight: 400; font-size: 12px; }
        svg { width: 100%; height: 100%; display: block; }
        .axis { stroke: #444; stroke-width: 1; }
        .grid { stroke: #292929; stroke-width: 1; }
        .line { fill: none; stroke: #7cc7ff; stroke-width: 2.5; }
        .line.texts { stroke: #8fd18f; }
        .area { fill: #7cc7ff; opacity: 0.14; }
        .tick { fill: #9a9a9a; font: 11px system-ui; }
        .crosshair { stroke: #777; stroke-width: 1; stroke-dasharray: 3 3; }
        .dot { fill: #b8e1ff; stroke: #151515; stroke-width: 2; }
        .tooltip {
          position: absolute; min-width: 190px; pointer-events: none; display: none;
          padding: 8px 10px; border: 1px solid #444; border-radius: 6px;
          background: rgba(18, 18, 18, 0.96); box-shadow: 0 8px 24px rgba(0,0,0,0.35);
          color: #eee; font: 12px system-ui; line-height: 1.45;
        }
        .tooltip strong { display: block; margin-bottom: 4px; color: #fff; font-size: 12px; }
        .tooltip span { color: #aaa; }
        .legend { display: flex; gap: 12px; align-items: center; }
        .legend b { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; }
        .legend .qps b { background: #7cc7ff; }
        .legend .tps b { background: #8fd18f; }
      </style>
      <div class="frame">
        <div class="chart-head">
          <div class="legend">
            <span class="qps"><b></b>request QPS</span>
            <span class="tps"><b></b>text/s</span>
          </div>
          <span>last 1h</span>
        </div>
        <svg></svg>
        <div class="tooltip"></div>
      </div>`;
    this.svg = this.shadowRoot.querySelector("svg");
    this.tooltip = this.shadowRoot.querySelector(".tooltip");
    this.size = { width: 1000, height: 260 };
    this.resizeObserver = new ResizeObserver(entries => {
      const rect = entries[0]?.contentRect;
      if (!rect) return;
      this.size = { width: Math.max(320, rect.width), height: Math.max(220, rect.height) };
      this.render();
    });
  }

  connectedCallback() {
    this.addEventListener("mousemove", event => this.onHover(event));
    this.addEventListener("mouseleave", () => {
      this.hoverIndex = null;
      this.tooltip.style.display = "none";
      this.render();
    });
    this.resizeObserver.observe(this);
  }

  disconnectedCallback() {
    this.resizeObserver.disconnect();
  }

  setData(buckets) {
    this.buckets = Array.isArray(buckets) ? buckets : [];
    this.render();
  }

  pointFor(index, maxQps) {
    const pad = { left: 58, right: 22, top: 44, bottom: 34 };
    const width = this.size.width - pad.left - pad.right;
    const height = this.size.height - pad.top - pad.bottom;
    const bucket = this.buckets[index] || {};
    const x = pad.left + (this.buckets.length <= 1 ? 0 : (index / (this.buckets.length - 1)) * width);
    const y = pad.top + height - (((bucket.qps || 0) / maxQps) * height);
    return { x, y };
  }

  textPointFor(index, maxTps) {
    const pad = { left: 58, right: 22, top: 44, bottom: 34 };
    const width = this.size.width - pad.left - pad.right;
    const height = this.size.height - pad.top - pad.bottom;
    const bucket = this.buckets[index] || {};
    const x = pad.left + (this.buckets.length <= 1 ? 0 : (index / (this.buckets.length - 1)) * width);
    const y = pad.top + height - (((bucket.tps || 0) / maxTps) * height);
    return { x, y };
  }

  render() {
    const buckets = this.buckets;
    const maxQps = Math.max(0.01, ...buckets.map(b => b.qps || 0));
    const maxTps = Math.max(0.01, ...buckets.map(b => b.tps || 0));
    const yTicks = [0, maxQps / 2, maxQps];
    const pad = { left: 58, right: 22, top: 44, bottom: 34 };
    const width = this.size.width - pad.left - pad.right;
    const height = this.size.height - pad.top - pad.bottom;
    const points = buckets.map((_, index) => this.pointFor(index, maxQps));
    const textPoints = buckets.map((_, index) => this.textPointFor(index, maxTps));
    const line = points.map((p, index) => `${index === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
    const textLine = textPoints.map((p, index) => `${index === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
    const area = points.length
      ? `${line} L ${points[points.length - 1].x.toFixed(1)} ${pad.top + height} L ${pad.left} ${pad.top + height} Z`
      : "";
    const hover = this.hoverIndex === null ? null : points[this.hoverIndex];
    const start = buckets[0]?.start_epoch ? new Date(buckets[0].start_epoch * 1000) : null;
    const end = buckets[buckets.length - 1]?.start_epoch ? new Date(buckets[buckets.length - 1].start_epoch * 1000) : null;
    this.svg.setAttribute("viewBox", `0 0 ${this.size.width} ${this.size.height}`);

    this.svg.innerHTML = `
      ${yTicks.map(value => {
        const y = pad.top + height - (value / maxQps) * height;
        return `<line class="grid" x1="${pad.left}" y1="${y}" x2="${pad.left + width}" y2="${y}"></line>
          <text class="tick" x="12" y="${y + 4}">${value.toFixed(3)}</text>`;
      }).join("")}
      <line class="axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${pad.top + height}"></line>
      <line class="axis" x1="${pad.left}" y1="${pad.top + height}" x2="${pad.left + width}" y2="${pad.top + height}"></line>
      ${start ? `<text class="tick" x="${pad.left}" y="${this.size.height - 10}">${start.toLocaleTimeString()}</text>` : ""}
      ${end ? `<text class="tick" x="${pad.left + width - 72}" y="${this.size.height - 10}">${end.toLocaleTimeString()}</text>` : ""}
      ${area ? `<path class="area" d="${area}"></path><path class="line" d="${line}"></path><path class="line texts" d="${textLine}"></path>` : ""}
      ${hover ? `<line class="crosshair" x1="${hover.x}" y1="${pad.top}" x2="${hover.x}" y2="${pad.top + height}"></line>
        <circle class="dot" cx="${hover.x}" cy="${hover.y}" r="4"></circle>` : ""}
    `;
  }

  onHover(event) {
    if (!this.buckets.length) return;
    const rect = this.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const padLeft = 58;
    const chartWidth = this.size.width - 58 - 22;
    const ratio = Math.max(0, Math.min(1, (x - padLeft) / chartWidth));
    const index = Math.round(ratio * (this.buckets.length - 1));
    this.hoverIndex = index;
    this.render();

    const bucket = this.buckets[index];
    const time = new Date(bucket.start_epoch * 1000).toLocaleString();
    this.tooltip.innerHTML = `
      <strong>${esc(time)}</strong>
      <div><span>request QPS</span> ${Number(bucket.qps || 0).toFixed(4)}</div>
      <div><span>text/s</span> ${Number(bucket.tps || 0).toFixed(4)}</div>
      <div><span>requests</span> ${esc(bucket.requests)}</div>
      <div><span>texts</span> ${esc(bucket.texts)}</div>
      <div><span>avg texts/request</span> ${Number(bucket.avg_texts_per_request || 0).toFixed(2)}</div>
      <div><span>failed</span> ${esc(bucket.failed)}</div>
      <div><span>p95</span> ${bucket.p95_duration_ms ?? "-"} ms</div>
    `;
    this.tooltip.style.display = "block";
    const left = Math.min(rect.width - 220, Math.max(8, event.clientX - rect.left + 12));
    this.tooltip.style.left = `${left}px`;
    this.tooltip.style.top = "42px";
  }
}

customElements.define("qps-chart", QpsChart);

cards.addEventListener("mouseover", event => {
  const card = event.target.closest(".card[data-title]");
  if (!card || !cards.contains(card)) return;
  showCardTooltip(card, event);
});
cards.addEventListener("mousemove", positionCardTooltip);
cards.addEventListener("mouseout", event => {
  const card = event.target.closest(".card[data-title]");
  if (!card || card.contains(event.relatedTarget)) return;
  hideCardTooltip();
});

async function apiJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function refresh() {
  const [stats, metrics, log] = await Promise.all([
    apiJson("/statsz"),
    apiJson("/metricsz?window_seconds=3600&bucket_seconds=30"),
    apiJson("/request-logz?limit=200")
  ]);
  const s = stats.stats;
  const r = stats.runtime;
  const lastError = s.last_error_summary
    ? `${s.last_error_at ?? ""} ${s.last_error_request_id ?? ""}\n${s.last_error_summary}`
    : "no recorded errors";
  const latestBucket = [...(metrics.buckets || [])].reverse().find(bucket => Number(bucket.requests || 0) > 0)
    || (metrics.buckets || [])[metrics.buckets.length - 1]
    || {};
  const cardRows = [
    { label: "requests", value: s.requests_total },
    { label: "request qps", value: Number(latestBucket.qps || 0).toFixed(3) },
    { label: "text/s", value: Number(latestBucket.tps || 0).toFixed(1) },
    { label: "inflight / queue", value: `${r.inflight_encodes} / ${s.queue_depth}` },
    { label: "device", value: r.loaded_device },
    { label: "precision", value: r.precision ?? "-" },
    { label: "worker pid", value: r.worker_pid ?? "-" },
    { label: "request texts", value: s.last_request_texts ?? "-" },
    { label: "batch size", value: s.last_batch_texts ?? "-" },
    {
      label: "batch target",
      value: r.effective_batch_target ?? "-"
    },
    {
      label: "last duration",
      value: `${s.last_duration_ms ?? "-"} ms`,
      tone: durationTone(s.last_duration_ms)
    },
    {
      label: "errors",
      value: s.requests_failed,
      tone: Number(s.requests_failed || 0) > 0 ? "bad" : "",
      title: lastError
    }
  ];
  cards.innerHTML = cardRows.map(item => `
    <div class="card"${item.title ? ` data-title="${esc(item.title)}"` : ""}>
      <div class="label">${esc(item.label)}</div>
      <div class="value${item.tone ? ` ${esc(item.tone)}` : ""}">${esc(item.value)}</div>
    </div>
  `).join("");
  qpsChart.setData(metrics.buckets);
  inputs.innerHTML = log.inputs.map(item => `
    <tr>
      <td>${esc(item.received_at)}</td>
      <td>${esc(item.source_ip)}</td>
      <td>${esc(item.source_country)}</td>
      <td class="input">${esc(item.input)}</td>
    </tr>
  `).join("");
  updated.textContent = `updated ${new Date().toLocaleTimeString()}`;
}

refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>"""


def create_app(settings: Settings | None = None, runtime: EmbedderRuntime | None = None) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    resolved_runtime = runtime or EmbedderRuntime(resolved_settings)
    stats = ProviderRuntimeStats()
    request_log = RequestLogBuffer(max_inputs=10_000, max_requests=10_000)
    resolved_runtime.attach_stats(stats)
    batcher = ContinuousBatcher(resolved_runtime, stats=stats, window_secs=resolved_settings.batch_window_ms / 1000)
    input_length_validator = InputLengthValidator(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = batcher.start()
        offload_task: asyncio.Task | None = None
        if resolved_runtime.runtime_status()["idle_offload_enabled"]:
            async def idle_offload_worker() -> None:
                while True:
                    await asyncio.sleep(resolved_settings.idle_offload_poll_seconds)
                    await asyncio.get_running_loop().run_in_executor(None, resolved_runtime.maybe_offload_idle)

            offload_task = asyncio.create_task(idle_offload_worker())
        yield
        task.cancel()
        if offload_task is not None:
            offload_task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        if offload_task is not None:
            try:
                await offload_task
            except asyncio.CancelledError:
                pass
        resolved_runtime.close()

    app = FastAPI(title="Embedding Provider", version="0.3.0", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "service": resolved_settings.service_name,
            "model": resolved_settings.model_id,
            "model_alias": resolved_settings.model_alias,
            "task": resolved_settings.embedding_task,
            "dimensions": resolved_settings.default_dimensions,
            "device": resolved_runtime.device_name,
            "batch_window_ms": resolved_settings.batch_window_ms,
            "runtime": resolved_runtime.runtime_status(),
            "stats": stats.snapshot(),
        }

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        runtime_status = resolved_runtime.runtime_status()
        payload = {
            "ready": not bool(runtime_status.get("reload_in_progress")),
            "service": resolved_settings.service_name,
            "model": resolved_settings.model_id,
            "runtime": runtime_status,
            "stats": stats.snapshot(),
        }
        return JSONResponse(status_code=200 if payload["ready"] else 503, content=payload)

    @app.get("/statsz")
    def statsz() -> dict[str, Any]:
        return {
            "service": resolved_settings.service_name,
            "model": resolved_settings.model_id,
            "stats": stats.snapshot(),
            "runtime": resolved_runtime.runtime_status(),
        }

    @app.post("/admin/device")
    async def switch_device(
        request: DeviceSwitchRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _require_api_key(resolved_settings, authorization)
        try:
            runtime_status = await asyncio.get_running_loop().run_in_executor(
                None,
                functools.partial(resolved_runtime.switch_device, request.device),
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "service": resolved_settings.service_name,
            "model": resolved_settings.model_id,
            "runtime": runtime_status,
            "stats": stats.snapshot(),
        }

    @app.get("/metricsz")
    def metricsz(
        window_seconds: int = 3600,
        bucket_seconds: int = 30,
    ) -> dict[str, Any]:
        return {
            "service": resolved_settings.service_name,
            "model": resolved_settings.model_id,
            "window_seconds": window_seconds,
            "bucket_seconds": bucket_seconds,
            "buckets": request_log.qps_buckets(
                window_seconds=max(1, min(86_400, window_seconds)),
                bucket_seconds=max(1, min(3600, bucket_seconds)),
            ),
            "stats": stats.snapshot(),
            "runtime": resolved_runtime.runtime_status(),
        }

    @app.get("/request-logz")
    def request_logz(limit: int = 200) -> dict[str, Any]:
        return {
            "service": resolved_settings.service_name,
            "model": resolved_settings.model_id,
            "inputs": request_log.recent_inputs(limit=limit),
            "requests": request_log.recent_requests(limit=limit),
        }

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(_dashboard_html())

    @app.get("/v1/models", response_model=ModelList)
    def list_models(authorization: str | None = Header(default=None)) -> ModelList:
        _require_api_key(resolved_settings, authorization)
        return ModelList(data=[ModelInfo(id=resolved_settings.model_alias or resolved_settings.model_id)])

    @app.post("/v1/embeddings", response_model=EmbeddingResponse)
    async def create_embeddings(
        http_request: Request,
        request: EmbeddingRequest,
        response: Response,
        authorization: str | None = Header(default=None),
        x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
    ) -> EmbeddingResponse:
        _require_api_key(resolved_settings, authorization)
        request_id = (x_request_id or uuid4().hex).strip()
        response.headers["X-Request-Id"] = request_id
        if request.encoding_format not in (None, "float"):
            raise HTTPException(
                status_code=400,
                detail="Only float encoding_format is supported",
                headers={"X-Request-Id": request_id},
            )
        allowed_models = {resolved_settings.model_id}
        if resolved_settings.model_alias:
            allowed_models.add(resolved_settings.model_alias)
        if request.model and request.model not in allowed_models:
            raise HTTPException(
                status_code=400,
                detail=f"Loaded model is {resolved_settings.model_id}, got {request.model}",
                headers={"X-Request-Id": request_id},
            )
        texts = _validate_inputs(request.input)
        source_ip, source_country = _client_ip_and_country(http_request)
        request_log.record_start(
            request_id=request_id,
            source_ip=source_ip,
            source_country=source_country,
            model=request.model,
            dimensions=request.dimensions,
            task=request.task,
            texts=texts,
        )
        stats.record_request_start(request_id=request_id, text_count=len(texts))
        started_at = time.perf_counter()
        try:
            input_length_validator.validate(texts)
        except HTTPException as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000
            error_summary = f"HTTPException: {exc.detail}"
            stats.record_request_failure(
                request_id=request_id,
                duration_ms=duration_ms,
                error_summary=error_summary,
            )
            request_log.record_finish(
                request_id=request_id,
                status_code=exc.status_code,
                duration_ms=duration_ms,
                error_summary=error_summary,
            )
            log.warning(
                "embedding request rejected request_id=%s texts=%d duration_ms=%.1f error=%s",
                request_id,
                len(texts),
                duration_ms,
                error_summary,
            )
            exc.headers = {**(exc.headers or {}), "X-Request-Id": request_id}
            raise
        log.info(
            "embedding request started request_id=%s texts=%d dimensions=%s task=%s",
            request_id,
            len(texts),
            request.dimensions,
            request.task,
        )
        try:
            embeddings = await batcher.encode(
                texts,
                dimensions=request.dimensions,
                task=request.task,
                request_id=request_id,
            )
        except ValueError as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000
            error_summary = _summarize_exception(exc)
            stats.record_request_failure(
                request_id=request_id,
                duration_ms=duration_ms,
                error_summary=error_summary,
            )
            request_log.record_finish(
                request_id=request_id,
                status_code=400,
                duration_ms=duration_ms,
                error_summary=error_summary,
            )
            log.warning(
                "embedding request failed request_id=%s texts=%d duration_ms=%.1f error=%s",
                request_id,
                len(texts),
                duration_ms,
                error_summary,
            )
            raise HTTPException(status_code=400, detail=str(exc), headers={"X-Request-Id": request_id}) from exc
        except Exception as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000
            error_summary = _summarize_exception(exc)
            stats.record_request_failure(
                request_id=request_id,
                duration_ms=duration_ms,
                error_summary=error_summary,
            )
            request_log.record_finish(
                request_id=request_id,
                status_code=500,
                duration_ms=duration_ms,
                error_summary=error_summary,
            )
            log.exception(
                "embedding request crashed request_id=%s texts=%d duration_ms=%.1f error=%s",
                request_id,
                len(texts),
                duration_ms,
                error_summary,
            )
            raise HTTPException(
                status_code=500,
                detail="embedding request failed",
                headers={"X-Request-Id": request_id},
            ) from exc

        duration_ms = (time.perf_counter() - started_at) * 1000
        stats.record_request_success(request_id=request_id, duration_ms=duration_ms)
        request_log.record_finish(
            request_id=request_id,
            status_code=200,
            duration_ms=duration_ms,
        )
        log.info(
            "embedding request succeeded request_id=%s texts=%d duration_ms=%.1f",
            request_id,
            len(texts),
            duration_ms,
        )

        return EmbeddingResponse(
            data=[EmbeddingItem(index=idx, embedding=embedding) for idx, embedding in enumerate(embeddings)],
            model=resolved_settings.model_alias or resolved_settings.model_id,
            usage=EmbeddingUsage(
                prompt_tokens=_estimate_tokens(texts),
                total_tokens=_estimate_tokens(texts),
            ),
        )

    return app


app = create_app()
