from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor


_write_lock = threading.Lock()


def emit(payload: dict[str, object]) -> None:
    with _write_lock:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()


def handle(request: dict[str, object]) -> None:
    texts = [str(item) for item in request.get("texts") or []]
    if texts and texts[0] == "exit":
        os._exit(7)
    time.sleep(0.20 if texts and texts[0] == "slow" else 0.02)
    if texts and texts[0] == "error":
        emit({"id": request.get("id"), "status": "error", "error": "fake failure"})
        return
    emit(
        {
            "id": request.get("id"),
            "status": "ok",
            "embeddings": [[1.0] if text == "slow" else [2.0] for text in texts],
            "sample_bytes_per_text": 123.0,
        }
    )


emit({"status": "ready", "device": "cuda", "model": "fake"})
with ThreadPoolExecutor(max_workers=4) as executor:
    for raw in sys.stdin:
        request = json.loads(raw)
        op = request.get("op")
        if op == "shutdown":
            emit({"status": "ok"})
            break
        if op == "empty_cache":
            emit({"id": request.get("id"), "status": "ok"})
            continue
        executor.submit(handle, request)
