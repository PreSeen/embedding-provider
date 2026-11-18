#!/usr/bin/env python3
"""Measure the embedder's VRAM peak per batch size at MAX_LENGTH tokens per text.

Loads the model on one GPU the way the GPU worker does, runs one forward per
batch size with every text at the token limit, and reports, per batch size, the
torch peak (allocated and reserved) and the whole process's footprint as
nvidia-smi sees it (CUDA context included). The process footprint is what a
neighbour reserve on that card has to cover.

    CUDA_VISIBLE_DEVICES=1 python scripts/measure_cuda_peak.py --batch-sizes 1,2,4,8
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time


def process_used_mib(pid: int) -> float | None:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            return float(parts[1])
    return None


def text_at_token_limit(tokenizer, max_length: int) -> str:
    sentence = "记忆系统在写入时抽取实体和关系，召回时按相关性排序。The quick brown fox jumps over the lazy dog. "
    ids = tokenizer(sentence * (max_length // 20 + 10))["input_ids"][:max_length]
    return tokenizer.decode(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID", "jinaai/jina-embeddings-v5-text-small"))
    parser.add_argument("--max-length", type=int, default=int(os.getenv("MAX_LENGTH", "8192")))
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--task", default=os.getenv("EMBEDDING_TASK", "text-matching"))
    args = parser.parse_args()

    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
    torch.cuda.synchronize()
    pid = os.getpid()
    results: dict[str, object] = {
        "model_id": args.model_id,
        "max_length": args.max_length,
        "gpu": torch.cuda.get_device_name(0),
        "loaded_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
        "loaded_process_mib": process_used_mib(pid),
        "batches": [],
    }
    text = text_at_token_limit(model.tokenizer, args.max_length)
    for batch_size in (int(value) for value in args.batch_sizes.split(",")):
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            with torch.no_grad():
                model.encode([text] * batch_size, task=args.task, max_length=args.max_length)
            torch.cuda.synchronize()
            outcome = "ok"
        except torch.OutOfMemoryError:
            outcome = "oom"
        results["batches"].append(
            {
                "batch_size": batch_size,
                "tokens": batch_size * args.max_length,
                "outcome": outcome,
                "forward_s": round(time.perf_counter() - started, 3),
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
                # The caching allocator keeps the peak reserved, so this is the high-water mark.
                "process_mib_after": process_used_mib(pid),
            }
        )
        if outcome == "oom":
            break
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
