#!/usr/bin/env python3
"""Measure 1-text latency while a document-ingest burst keeps the provider busy.

Bulk clients send batches of long documents back to back. A probe sends one short
text at a fixed interval, alternating a recall-query-sized text and a
memory-write-episode-sized text, and records each probe's latency. Every text is
unique so the embedding cache never answers.

    python scripts/benchmark_interactive_latency.py --base-url http://127.0.0.1:7998 \
        --api-key KEY --duration 180
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.request import Request, urlopen

from scripts.benchmark_gpu_forward_concurrency import percentile

# Mixed Chinese/English text like the memory and document traffic on gpu4.
_SENTENCE = "记忆系统在写入时抽取实体和关系，召回时按相关性排序。The quick brown fox jumps over the lazy dog. "


def unique_text(prefix: str, index: int, sentences: int) -> str:
    return f"{prefix} {index}: " + _SENTENCE * sentences


def post_embeddings(base_url: str, api_key: str | None, texts: list[str], timeout: float) -> float:
    body = json.dumps({"input": texts}).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    started = time.perf_counter()
    with urlopen(Request(f"{base_url}/v1/embeddings", data=body, headers=headers), timeout=timeout) as response:
        response.read()
    return (time.perf_counter() - started) * 1000


def summarize(latencies_ms: list[float]) -> dict[str, Any]:
    return {
        "count": len(latencies_ms),
        "p50_ms": percentile(latencies_ms, 50),
        "p90_ms": percentile(latencies_ms, 90),
        "max_ms": round(max(latencies_ms), 3) if latencies_ms else 0.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    stop = threading.Event()
    bulk_latencies: list[float] = []
    probe_latencies: dict[str, list[float]] = {"query": [], "episode": []}
    counter = iter(range(10**9))
    lock = threading.Lock()

    def bulk_client(client: int) -> None:
        while not stop.is_set():
            with lock:
                base = next(counter)
            texts = [unique_text(f"doc-{client}", base * 100 + i, args.doc_sentences) for i in range(args.batch_texts)]
            bulk_latencies.append(post_embeddings(args.base_url, args.api_key, texts, args.timeout))

    def probe(kind: str, sentences: int) -> None:
        with lock:
            n = next(counter)
        probe_latencies[kind].append(
            post_embeddings(args.base_url, args.api_key, [unique_text(kind, n, sentences)], args.timeout)
        )

    kinds = [("query", args.query_sentences), ("episode", args.episode_sentences)]
    # Probes are open-loop: one starts every interval whether or not earlier ones returned,
    # the way recall queries keep arriving while the provider is busy.
    probe_count = int(args.duration / args.probe_interval)
    with ThreadPoolExecutor(max_workers=args.bulk_clients + probe_count) as pool:
        bulk_futures = [pool.submit(bulk_client, client) for client in range(args.bulk_clients)]
        time.sleep(args.warmup)
        probe_futures = []
        for index in range(probe_count):
            probe_futures.append(pool.submit(probe, *kinds[index % len(kinds)]))
            time.sleep(args.probe_interval)
        for future in probe_futures:
            future.result()
        stop.set()
        for future in bulk_futures:
            future.result()

    all_probes = probe_latencies["query"] + probe_latencies["episode"]
    return {
        "probe_all": summarize(all_probes),
        "probe_query": summarize(probe_latencies["query"]),
        "probe_episode": summarize(probe_latencies["episode"]),
        "bulk": summarize(bulk_latencies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:7998")
    parser.add_argument("--api-key")
    parser.add_argument("--duration", type=float, default=180.0, help="seconds of probing under the burst")
    parser.add_argument("--warmup", type=float, default=5.0, help="seconds of burst before probing starts")
    parser.add_argument("--bulk-clients", type=int, default=2)
    parser.add_argument("--batch-texts", type=int, default=8)
    parser.add_argument("--doc-sentences", type=int, default=80, help="about 40 tokens per sentence")
    parser.add_argument("--query-sentences", type=int, default=1)
    parser.add_argument("--episode-sentences", type=int, default=10)
    parser.add_argument("--probe-interval", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
