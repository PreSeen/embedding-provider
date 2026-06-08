from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient

if "torch" not in sys.modules:
    fake_torch = types.ModuleType("torch")

    class _FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeTensor:
        pass

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def empty_cache() -> None:
            return None

        @staticmethod
        def reset_peak_memory_stats() -> None:
            return None

        @staticmethod
        def memory_allocated() -> int:
            return 0

        @staticmethod
        def max_memory_allocated() -> int:
            return 0

    fake_torch.cuda = _FakeCuda()
    fake_torch.no_grad = _FakeNoGrad
    fake_torch.device = lambda name: name
    fake_torch.float16 = "float16"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float32 = "float32"
    fake_torch.Tensor = _FakeTensor
    fake_torch.OutOfMemoryError = RuntimeError
    sys.modules["torch"] = fake_torch

if "transformers" not in sys.modules:
    fake_transformers = types.ModuleType("transformers")

    class _BootstrapModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def encode(self, texts: list[str], **kwargs):
            width = kwargs.get("truncate_dim") or 4
            return [[1.0] * width for _ in texts]

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _BootstrapModel()

    class _FakeTokenizer:
        def __call__(self, text: str, **kwargs):
            return {"input_ids": text.split()}

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeTokenizer()

    fake_transformers.AutoModel = _FakeAutoModel
    fake_transformers.AutoTokenizer = _FakeAutoTokenizer
    sys.modules["transformers"] = fake_transformers

from provider.app import AdaptiveBatchState, RequestLogBuffer, _country_for_ip, create_app
from provider.config import Settings


@dataclass
class FakeRuntime:
    reload_in_progress: bool = False
    fail_with: Exception | None = None
    estimate_max_texts_value: int = 128

    def __post_init__(self) -> None:
        self._stats = None
        self.device_name = "cpu"
        self.seen_batches: list[int] = []

    def attach_stats(self, stats: Any) -> None:
        self._stats = stats

    def runtime_status(self) -> dict[str, Any]:
        return {
            "loaded_device": "cpu",
            "preferred_device": "cpu",
            "engine_state": "hot",
            "idle_offload_enabled": False,
            "idle_offload_seconds": None,
            "idle_offload_poll_seconds": None,
            "inflight_encodes": 0,
            "idle_for_seconds": 0.0,
            "offloaded_for_seconds": None,
            "reload_in_progress": self.reload_in_progress,
            "worker_pid": None,
        }

    def encode(self, texts: list[str], *, dimensions: int | None = None, task: str | None = None) -> list[list[float]]:
        if self.fail_with is not None:
            raise self.fail_with
        self.seen_batches.append(len(texts))
        width = dimensions or 4
        return [[float(idx + 1)] * width for idx, _text in enumerate(texts)]

    def estimate_max_texts(self) -> int:
        return self.estimate_max_texts_value

    def close(self) -> None:
        return None


def _settings() -> Settings:
    return Settings(
        service_name="embedding-provider",
        model_id="jinaai/jina-embeddings-v5-text-nano",
        model_alias="jinaai/jina-embeddings-v5-text-nano",
        api_key="test-key",
        embedding_task="text-matching",
        default_dimensions=4,
        max_length=256,
        max_batch_size=8,
        normalize_embeddings=True,
        dtype="float32",
        attn_implementation=None,
        trust_remote_code=True,
        batch_window_ms=1,
        idle_offload_seconds=0,
        idle_offload_poll_seconds=0,
        cpu_batch_target=8,
        cpu_to_gpu_scale_up_texts=8,
        gpu_to_cpu_scale_down_texts=2,
        gpu_to_cpu_scale_down_seconds=30,
        start_device="auto",
        cuda_visible_devices=None,
    )


def test_health_and_stats_endpoints_expose_runtime_snapshot() -> None:
    app = create_app(settings=_settings(), runtime=FakeRuntime())
    with TestClient(app) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        payload = health.json()
        assert payload["ok"] is True
        assert payload["runtime"]["engine_state"] == "hot"
        assert payload["stats"]["requests_total"] == 0

        ready = client.get("/readyz")
        assert ready.status_code == 200
        assert ready.json()["ready"] is True

        stats = client.get("/statsz")
        assert stats.status_code == 200
        assert stats.json()["stats"]["queue_depth"] == 0


def test_readyz_returns_503_when_runtime_is_reloading() -> None:
    app = create_app(settings=_settings(), runtime=FakeRuntime(reload_in_progress=True))
    with TestClient(app) as client:
        response = client.get("/readyz")
        assert response.status_code == 503
        assert response.json()["ready"] is False


def test_embeddings_request_echoes_request_id_and_updates_success_stats() -> None:
    app = create_app(settings=_settings(), runtime=FakeRuntime())
    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            headers={
                "Authorization": "Bearer test-key",
                "X-Request-Id": "embed-req-1",
            },
            json={
                "model": "jinaai/jina-embeddings-v5-text-nano",
                "input": ["alpha", "beta"],
                "dimensions": 4,
            },
        )
        assert response.status_code == 200
        assert response.headers["X-Request-Id"] == "embed-req-1"

        stats = client.get("/statsz").json()["stats"]
        assert stats["requests_total"] == 1
        assert stats["requests_succeeded"] == 1
        assert stats["requests_failed"] == 0
        assert stats["texts_total"] == 2
        assert stats["batches_total"] == 1
        assert stats["last_request_id"] == "embed-req-1"
        assert stats["last_success_request_id"] == "embed-req-1"


def test_embeddings_failure_updates_error_stats_and_preserves_request_id() -> None:
    app = create_app(settings=_settings(), runtime=FakeRuntime(fail_with=ValueError("bad dims")))
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/embeddings",
            headers={
                "Authorization": "Bearer test-key",
                "X-Request-Id": "embed-req-fail",
            },
            json={
                "model": "jinaai/jina-embeddings-v5-text-nano",
                "input": ["alpha"],
            },
        )
        assert response.status_code == 400
        assert response.headers["X-Request-Id"] == "embed-req-fail"

        stats = client.get("/statsz").json()["stats"]
        assert stats["requests_total"] == 1
        assert stats["requests_succeeded"] == 0
        assert stats["requests_failed"] == 1
        assert stats["last_error_request_id"] == "embed-req-fail"
        assert "ValueError" in str(stats["last_error_summary"])


def test_embeddings_rejects_inputs_over_configured_token_limit() -> None:
    app = create_app(settings=_settings(), runtime=FakeRuntime())
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/embeddings",
            headers={
                "Authorization": "Bearer test-key",
                "X-Request-Id": "embed-req-long",
            },
            json={
                "model": "jinaai/jina-embeddings-v5-text-nano",
                "input": " ".join(f"tok-{idx}" for idx in range(300)),
            },
        )

        assert response.status_code == 400
        assert response.headers["X-Request-Id"] == "embed-req-long"
        assert "exceeds MAX_LENGTH=256" in response.json()["detail"]

        stats = client.get("/statsz").json()["stats"]
        assert stats["requests_total"] == 1
        assert stats["requests_failed"] == 1
        assert stats["last_error_request_id"] == "embed-req-long"
        assert "exceeds MAX_LENGTH=256" in str(stats["last_error_summary"])


def test_adaptive_batch_state_grows_conservatively_after_full_dispatches() -> None:
    state = AdaptiveBatchState(configured_max_batch_size=64)

    assert state.current_target == 1

    state.record_successful_dispatch(text_count=1, vram_cap=64, allow_growth=True)
    assert state.current_target == 2

    state.record_successful_dispatch(text_count=2, vram_cap=64, allow_growth=True)
    assert state.current_target == 4

    state.record_successful_dispatch(text_count=4, vram_cap=3, allow_growth=True)
    assert state.current_target == 4


def test_adaptive_batch_state_keeps_target_stable_until_reset() -> None:
    state = AdaptiveBatchState(configured_max_batch_size=64)
    state.record_successful_dispatch(text_count=1, vram_cap=64, allow_growth=True)
    state.record_successful_dispatch(text_count=2, vram_cap=64, allow_growth=True)
    state.record_successful_dispatch(text_count=4, vram_cap=64, allow_growth=True)

    assert state.current_target == 8

    state.record_successful_dispatch(text_count=2, vram_cap=64, allow_growth=False)
    assert state.current_target == 8

    state.reset()
    assert state.current_target == 1
    assert state.snapshot()["last_batch_texts"] == 0


def test_adaptive_batch_state_caps_growth_after_oom_backoff() -> None:
    state = AdaptiveBatchState(configured_max_batch_size=64)
    for text_count in [1, 2, 4, 8, 16, 32]:
        state.record_successful_dispatch(text_count=text_count, vram_cap=64, allow_growth=True)

    assert state.current_target == 64

    state.record_oom_backoff(failed_text_count=64)
    assert state.current_target == 32
    assert state.snapshot()["oom_target_ceiling"] == 32

    for _ in range(12):
        state.record_successful_dispatch(text_count=32, vram_cap=64, allow_growth=True)
        assert state.current_target == 32

    state.record_oom_backoff(failed_text_count=32)
    assert state.current_target == 16
    assert state.snapshot()["oom_target_ceiling"] == 16

    state.record_successful_dispatch(text_count=32, vram_cap=64, allow_growth=True)
    assert state.current_target == 16


def test_request_log_buffer_keeps_recent_inputs_and_qps_buckets() -> None:
    log = RequestLogBuffer(max_inputs=3, max_requests=10)
    log.record_start(
        request_id="req-1",
        source_ip="127.0.0.1",
        source_country="localhost",
        model="model-a",
        dimensions=4,
        task="search",
        texts=["alpha", "beta"],
    )
    log.record_finish(request_id="req-1", status_code=200, duration_ms=12.34)
    log.record_start(
        request_id="req-2",
        source_ip="127.0.0.2",
        source_country="localhost",
        model="model-a",
        dimensions=4,
        task="search",
        texts=["gamma", "delta"],
    )
    log.record_finish(request_id="req-2", status_code=500, duration_ms=56.78, error_summary="boom")

    inputs = log.recent_inputs(limit=10)
    assert [item["input"] for item in inputs] == ["delta", "gamma", "beta"]
    assert inputs[0]["request_id"] == "req-2"
    assert inputs[0]["source_country"] == "localhost"

    requests = log.recent_requests(limit=10)
    assert requests[0]["status"] == "error"
    assert requests[0]["error_summary"] == "boom"
    assert requests[1]["status"] == "ok"

    buckets = log.qps_buckets(window_seconds=3600, bucket_seconds=30)
    assert sum(bucket["requests"] for bucket in buckets) == 2
    assert sum(bucket["texts"] for bucket in buckets) == 4
    assert sum(bucket["failed"] for bucket in buckets) == 1


def test_country_for_ip_identifies_local_networks() -> None:
    assert _country_for_ip("127.0.0.1") == "localhost"
    assert _country_for_ip("192.168.1.10") == "LAN"
    assert _country_for_ip("not-an-ip") == "unknown"
