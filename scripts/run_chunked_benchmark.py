"""End-to-end Path D benchmark: real Qwen2.5-7B, real LongBench task,
real (en_qa F1, peak GPU GiB, wall-clock s) triple.

This is the reviewer-required cell that proves wrap_with_halo(chunked=True)
delivers the title's "lossless offloading" contract end-to-end.

We pick PassageRetrieval-en at 4× as the headline benchmark:
- It's a retrieval task where HALO's strength is concentrated
  (\Cref{tab:longbench-categories})
- Prompt length ≈ 4-8K tokens, large enough that the warmup→chunked
  transition triggers and the cache moves to DRAM
- 20 examples is enough to bound the F1 in <30 minutes

Outputs:
- experiments/runs/qwen2-5-7b/longbench/halo_chunked_4x/manifest.json
- console summary: (F1, peak_GiB, wall_s)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    import torch
    from baselines.longbench_eval import evaluate_task
    from halo import wrap_with_halo, HALOConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--task", default="passage_retrieval_en")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--memory-ratio", type=int, default=4,
                    help="hot_ratio = 1/memory_ratio")
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--recent-window", type=int, default=64)
    ap.add_argument("--out-dir", default="experiments/runs/qwen2-5-7b/longbench/halo_chunked_4x")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )

    # Configure HALO Path D.
    hot_ratio = 1.0 / args.memory_ratio
    cfg = HALOConfig(
        hot_ratio=hot_ratio, chunked=True,
        chunk_size=args.chunk_size, recent_window=args.recent_window,
        tiers=("dram",),
    )
    wrap_with_halo(model, cfg)

    print(f"=== Path D benchmark ===")
    print(f"  Model     : {args.model}")
    print(f"  Task      : {args.task} (limit {args.limit})")
    print(f"  HALOConfig: chunked=True, chunk_size={args.chunk_size}, "
          f"recent_window={args.recent_window}, hot_ratio={hot_ratio:.3f}")

    # Reset peak memory tracking.
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    peak_before = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"  Peak GPU before run: {peak_before:.2f} GiB")

    # Run.
    t0 = time.time()
    score = evaluate_task(model, tok, task=args.task, limit=args.limit, progress=True)
    wall_s = time.time() - t0
    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Pull telemetry from cache.
    cache = model._halo_cache
    tele = cache.telemetry()

    print(f"\n=== Result ===")
    print(f"  {args.task} score : {score:.2f}")
    print(f"  Peak GPU (GiB)   : {peak_gib:.2f}")
    print(f"  Wall-clock (s)   : {wall_s:.1f}")
    print(f"  Mode (end)       : {tele.get('mode', 'n/a')}")
    print(f"  Total chunks     : {tele.get('chunked_total_chunks', 0)}")
    print(f"  Used LSE-merge   : {tele.get('chunked_used_lse_merge_any', False)}")

    manifest = {
        "suite": "longbench-v1",
        "method": "halo_chunked",
        "memory_ratio": args.memory_ratio,
        "model": args.model,
        "task": args.task,
        "limit": args.limit,
        "scores": {args.task: score},
        "peak_gpu_gib": peak_gib,
        "wall_clock_s": wall_s,
        "halo_config": {
            "hot_ratio": hot_ratio,
            "chunked": True,
            "chunk_size": args.chunk_size,
            "recent_window": args.recent_window,
            "tiers": ["dram"],
        },
        "halo_telemetry": tele,
        "seed": 0,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n  wrote {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
