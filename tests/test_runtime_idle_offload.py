from __future__ import annotations

import os
import sys
import types
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np

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
        def mem_get_info():
            return (8 * 1024**3, 24 * 1024**3)

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

        def encode(self, texts: list[str], **kwargs) -> np.ndarray:
            width = kwargs.get("truncate_dim") or 4
            return np.ones((len(texts), width), dtype=np.float32)

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _BootstrapModel()

    fake_transformers.AutoModel = _FakeAutoModel
    sys.modules["transformers"] = fake_transformers

from provider.app import EmbedderRuntime
from provider.config import Settings


class FakeWorker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.running = False
        self.starts = 0
        self.stops = 0
        self.batch_sizes: list[int] = []
        self.cache_releases = 0

    @property
    def pid(self) -> int | None:
        return 4321 if self.running else None

    def is_running(self) -> bool:
        return self.running

    def ensure_started(self) -> str:
        self.running = True
        self.starts += 1
        return "cuda"

    def encode(self, texts: list[str], *, dimensions: int | None = None, task: str | None = None):
        self.ensure_started()
        self.batch_sizes.append(len(texts))
        width = dimensions or 4
        return np.ones((len(texts), width), dtype=np.float32).tolist(), 1024.0

    def empty_cache(self) -> None:
        self.cache_releases += 1

    def terminate(self) -> None:
        if self.running:
            self.stops += 1
        self.running = False


class FailingStartWorker(FakeWorker):
    def ensure_started(self) -> str:
        self.starts += 1
        raise RuntimeError("CUDA out of memory while loading model")


class EmbedderRuntimeIdleOffloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_stack = ExitStack()
        env = {
            "SERVICE_NAME": "embedding-provider",
            "MODEL_ID": "jinaai/jina-embeddings-v5-text-nano",
            "MODEL_ALIAS": "jinaai/jina-embeddings-v5-text-nano",
            "EMBEDDING_TASK": "text-matching",
            "DEFAULT_DIMENSIONS": "4",
            "MAX_LENGTH": "256",
            "MAX_BATCH_SIZE": "8",
            "BATCH_WINDOW_MS": "200",
            "IDLE_OFFLOAD_SECONDS": "1",
            "IDLE_OFFLOAD_POLL_SECONDS": "1",
            "CPU_TO_GPU_SCALE_UP_TEXTS": "3",
            "GPU_TO_CPU_SCALE_DOWN_TEXTS": "2",
            "GPU_TO_CPU_SCALE_DOWN_SECONDS": "30",
            "NORMALIZE_EMBEDDINGS": "true",
            "DTYPE": "float32",
            "TRUST_REMOTE_CODE": "true",
        }
        for key, value in env.items():
            self._env_stack.enter_context(patch.dict(os.environ, {key: value}))

    def tearDown(self) -> None:
        self._env_stack.close()

    def test_idle_runtime_scales_down_to_cpu_and_scales_up_for_batches(self) -> None:
        workers: list[FakeWorker] = []

        def fake_worker_factory(settings: Settings) -> FakeWorker:
            worker = FakeWorker(settings)
            workers.append(worker)
            return worker

        with (
            patch("provider.app._GpuEmbedderWorker", side_effect=fake_worker_factory),
            patch("provider.app._detect_preferred_device", return_value="cuda"),
            patch("provider.app._probe_cuda_memory_bytes", return_value=(8 * 1024**3, 24 * 1024**3)),
        ):
            runtime = EmbedderRuntime(Settings.from_env())
            self.assertEqual(runtime.runtime_status()["loaded_device"], "none")

            embeddings = runtime.encode(["hello world"])
            self.assertEqual(len(embeddings), 1)
            self.assertEqual(runtime.runtime_status()["loaded_device"], "cuda")
            self.assertEqual(runtime.runtime_status()["engine_state"], "hot")

            runtime._last_encode_finished_at -= 2
            offloaded = runtime.maybe_offload_idle()
            self.assertTrue(offloaded)
            self.assertEqual(runtime.runtime_status()["loaded_device"], "cpu")
            self.assertEqual(runtime.runtime_status()["preferred_device"], "cpu")
            self.assertEqual(runtime.runtime_status()["engine_state"], "hot")

            embeddings = runtime.encode(["hello world"])
            self.assertEqual(len(embeddings), 1)
            self.assertEqual(runtime.runtime_status()["loaded_device"], "cpu")
            self.assertEqual(runtime.runtime_status()["preferred_device"], "cpu")

            embeddings = runtime.encode(["hello world", "again"])
            self.assertEqual(len(embeddings), 2)
            self.assertEqual(runtime.runtime_status()["loaded_device"], "cpu")
            self.assertEqual(runtime.runtime_status()["preferred_device"], "cpu")

            embeddings = runtime.encode(["hello world", "again", "third"])
            self.assertEqual(len(embeddings), 3)
            self.assertEqual(runtime.runtime_status()["loaded_device"], "cuda")
            self.assertEqual(runtime.runtime_status()["preferred_device"], "cuda")
            self.assertEqual(runtime.runtime_status()["engine_state"], "hot")
            runtime.close()

        self.assertEqual(len(workers), 2)
        self.assertGreaterEqual(workers[0].starts, 1)
        self.assertGreaterEqual(workers[0].stops, 1)
        self.assertGreaterEqual(workers[1].starts, 1)

    def test_low_batch_scale_down_runs_without_follow_up_request(self) -> None:
        workers: list[FakeWorker] = []

        def fake_worker_factory(settings: Settings) -> FakeWorker:
            worker = FakeWorker(settings)
            workers.append(worker)
            return worker

        with (
            patch.dict(
                os.environ,
                {
                    "IDLE_OFFLOAD_SECONDS": "999",
                    "GPU_TO_CPU_SCALE_DOWN_SECONDS": "1",
                },
            ),
            patch("provider.app._GpuEmbedderWorker", side_effect=fake_worker_factory),
            patch("provider.app._detect_preferred_device", return_value="cuda"),
            patch("provider.app._probe_cuda_memory_bytes", return_value=(8 * 1024**3, 24 * 1024**3)),
        ):
            runtime = EmbedderRuntime(Settings.from_env())
            try:
                runtime.encode(["one", "two", "three"])
                self.assertEqual(runtime.runtime_status()["loaded_device"], "cuda")

                runtime.encode(["small"])
                status = runtime.runtime_status()
                self.assertEqual(status["loaded_device"], "cuda")
                self.assertIsNotNone(status["gpu_low_batch_seconds"])

                runtime._gpu_low_batch_since -= 2
                offloaded = runtime.maybe_offload_idle()
                self.assertTrue(offloaded)
                self.assertEqual(runtime.runtime_status()["loaded_device"], "cpu")
                self.assertEqual(runtime.runtime_status()["effective_batch_target"], 8)
            finally:
                runtime.close()

    def test_estimate_max_texts_drops_to_one_when_free_vram_is_below_safety_margin(self) -> None:
        with (
            patch("provider.app._GpuEmbedderWorker", FakeWorker),
            patch("provider.app._detect_preferred_device", return_value="cuda"),
            patch("provider.app._probe_cuda_memory_bytes", return_value=(512 * 1024**2, 24 * 1024**3)),
        ):
            runtime = EmbedderRuntime(Settings.from_env())
            try:
                self.assertEqual(runtime.estimate_max_texts(), 1)
            finally:
                runtime.close()

    def test_runtime_can_start_on_cpu_and_switch_devices_without_restart(self) -> None:
        with (
            patch.dict(os.environ, {"START_DEVICE": "cpu"}),
            patch("provider.app._GpuEmbedderWorker", FakeWorker),
            patch("provider.app._detect_preferred_device", return_value="cuda"),
            patch("provider.app._probe_cuda_memory_bytes", return_value=(8 * 1024**3, 24 * 1024**3)),
        ):
            runtime = EmbedderRuntime(Settings.from_env())
            try:
                status = runtime.runtime_status()
                self.assertEqual(status["loaded_device"], "cpu")
                self.assertEqual(status["detected_device"], "cuda")
                self.assertIsNone(status["worker_pid"])

                status = runtime.switch_device("cuda")
                self.assertEqual(status["loaded_device"], "cuda")
                self.assertEqual(status["preferred_device"], "cuda")
                self.assertIsNotNone(status["worker_pid"])

                status = runtime.switch_device("cpu")
                self.assertEqual(status["loaded_device"], "cpu")
                self.assertEqual(status["preferred_device"], "cpu")
                self.assertIsNone(status["worker_pid"])
            finally:
                runtime.close()

    def test_runtime_scales_down_from_gpu_after_sustained_small_batches(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "START_DEVICE": "cpu",
                    "CPU_TO_GPU_SCALE_UP_TEXTS": "3",
                    "GPU_TO_CPU_SCALE_DOWN_TEXTS": "2",
                    "GPU_TO_CPU_SCALE_DOWN_SECONDS": "0",
                },
            ),
            patch("provider.app._GpuEmbedderWorker", FakeWorker),
            patch("provider.app._detect_preferred_device", return_value="cuda"),
            patch("provider.app._probe_cuda_memory_bytes", return_value=(8 * 1024**3, 24 * 1024**3)),
        ):
            runtime = EmbedderRuntime(Settings.from_env())
            try:
                runtime.switch_device("cuda")
                self.assertEqual(runtime.runtime_status()["loaded_device"], "cuda")

                runtime.encode(["small-one"])
                self.assertEqual(runtime.runtime_status()["loaded_device"], "cuda")

                runtime.encode(["small-two"])
                self.assertEqual(runtime.runtime_status()["loaded_device"], "cpu")
                self.assertIsNone(runtime.runtime_status()["worker_pid"])
            finally:
                runtime.close()

    def test_gpu_worker_start_failure_falls_back_to_cpu(self) -> None:
        workers: list[FailingStartWorker] = []

        def fake_worker_factory(settings: Settings) -> FailingStartWorker:
            worker = FailingStartWorker(settings)
            workers.append(worker)
            return worker

        with (
            patch("provider.app._GpuEmbedderWorker", side_effect=fake_worker_factory),
            patch("provider.app._detect_preferred_device", return_value="cuda"),
            patch("provider.app._probe_cuda_memory_bytes", return_value=(8 * 1024**3, 24 * 1024**3)),
        ):
            runtime = EmbedderRuntime(Settings.from_env())
            try:
                embeddings = runtime.encode(["fallback"])
                status = runtime.runtime_status()
            finally:
                runtime.close()

        self.assertEqual(len(embeddings), 1)
        self.assertEqual(status["detected_device"], "cuda")
        self.assertEqual(status["preferred_device"], "cpu")
        self.assertEqual(status["loaded_device"], "cpu")
        self.assertEqual(status["engine_state"], "hot")
        self.assertFalse(status["idle_offload_enabled"])
        self.assertIn("CUDA out of memory", status["cuda_fallback_reason"])
        self.assertEqual(workers[0].starts, 1)

    def test_encode_splits_static_batch_size_by_available_vram_cap(self) -> None:
        workers: list[FakeWorker] = []

        def fake_worker_factory(settings: Settings) -> FakeWorker:
            worker = FakeWorker(settings)
            workers.append(worker)
            return worker

        with (
            patch("provider.app._GpuEmbedderWorker", side_effect=fake_worker_factory),
            patch("provider.app._detect_preferred_device", return_value="cuda"),
            patch("provider.app._probe_cuda_memory_bytes", return_value=(512 * 1024**2, 24 * 1024**3)),
        ):
            runtime = EmbedderRuntime(Settings.from_env())
            try:
                embeddings = runtime.encode(["alpha", "beta", "gamma"])
            finally:
                runtime.close()

        self.assertEqual(len(embeddings), 3)
        self.assertEqual(len(workers), 1)
        self.assertEqual(workers[0].batch_sizes, [1, 1, 1])

    def test_encode_ramps_cuda_batch_size_after_first_real_measurement(self) -> None:
        workers: list[FakeWorker] = []

        def fake_worker_factory(settings: Settings) -> FakeWorker:
            worker = FakeWorker(settings)
            workers.append(worker)
            return worker

        with (
            patch("provider.app._GpuEmbedderWorker", side_effect=fake_worker_factory),
            patch("provider.app._detect_preferred_device", return_value="cuda"),
            patch("provider.app._probe_cuda_memory_bytes", return_value=(8 * 1024**3, 24 * 1024**3)),
        ):
            runtime = EmbedderRuntime(Settings.from_env())
            try:
                embeddings = runtime.encode(["alpha", "beta", "gamma", "delta"])
            finally:
                runtime.close()

        self.assertEqual(len(embeddings), 4)
        self.assertEqual(len(workers), 1)
        self.assertEqual(workers[0].batch_sizes, [1, 2, 1])

    def test_cuda_batch_target_drops_when_following_request_is_smaller(self) -> None:
        workers: list[FakeWorker] = []

        def fake_worker_factory(settings: Settings) -> FakeWorker:
            worker = FakeWorker(settings)
            workers.append(worker)
            return worker

        with (
            patch("provider.app._GpuEmbedderWorker", side_effect=fake_worker_factory),
            patch("provider.app._detect_preferred_device", return_value="cuda"),
            patch("provider.app._probe_cuda_memory_bytes", return_value=(8 * 1024**3, 24 * 1024**3)),
        ):
            runtime = EmbedderRuntime(Settings.from_env())
            try:
                runtime.encode([f"large-{idx}" for idx in range(8)])
                runtime.encode(["small-a", "small-b"])
                runtime.encode([f"again-{idx}" for idx in range(5)])
            finally:
                runtime.close()

        self.assertEqual(len(workers), 1)
        self.assertEqual(workers[0].batch_sizes, [1, 2, 4, 1, 2, 2, 3])
        self.assertEqual(workers[0].cache_releases, 1)


if __name__ == "__main__":
    unittest.main()
