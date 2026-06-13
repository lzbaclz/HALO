"""bench_longbench_evict.py — End-to-end LongBench benchmark with peak
memory + wall-clock for Full vs HALO(keep) vs HALO(evict).

This is the LongBench analogue of ``scripts/bench_infinite_evict.py``.
It addresses the largest reviewer gap from the prior round: an
end-to-end measurement that converts the policy claim into wall-clock
tokens/sec and on-device peak GPU memory. The strict-eviction
HALOCacheEvict variant (\\Cref{sec:halo-evict}) is what makes the
``-N\\%'' memory-reduction story go from a bench microbenchmark to a
benchmark-grade artefact.

We pick three retrieval-leaning LongBench subtasks (the ones where the
HALOPress score-and-prune approach has the largest gap to Full --- see
\\Cref{tab:longbench-categories}):
   * passage_retrieval_en  (the hardest needle-in-haystack)
   * narrativeqa           (long-doc QA)
   * gov_report            (long-doc summarization)

For each (method, task) cell we record:
   * task accuracy / F1 (LongBench's metric);
   * peak GPU memory used during the entire eval;
   * wall-clock seconds for the eval pass;
   * tokens/sec (decoded tokens / wall-clock seconds), the headline
     deployment metric.

Outputs:
   experiments/runs/qwen2-5-7b/longbench_evict/manifest.json

Run:
   CUDA_VISIBLE_DEVICES=1 python scripts/bench_longbench_evict.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models/qwen2-5-7b.yaml")
    ap.add_argument("--tasks", nargs="+",
                    default=["passage_retrieval_en", "narrativeqa", "gov_report"])
    ap.add_argument("--memory-ratio", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--limit", type=int, default=20,
                    help="examples per (task, method, seed) cell.")
    ap.add_argument("--output", default="experiments/runs/qwen2-5-7b/longbench_evict")
    ap.add_argument("--methods", nargs="+",
                    default=["full", "halo_keep", "halo_evict"],
                    help="full | halo_keep | halo_evict")
    args = ap.parse_args()

    import torch
    from baselines.longbench_eval import evaluate_task
    from halo.utils import load_yaml, seed_everything

    model_cfg = load_yaml(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model + tok once.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = model_cfg["name_or_path"]
    print(f"[load] {name}", flush=True)
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[model_cfg.get("dtype", "bfloat16")]
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    results: dict = {}

    for method in args.methods:
        per_method: list[dict] = []
        for task in args.tasks:
            for seed in args.seeds:
                seed_everything(seed)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # ---- HALO config.
                press_ctx = None
                if method.startswith("halo"):
                    from halo import HALOConfig
                    from halo.policy import wrap_with_halo
                    tiers = ("dram",) if method == "halo_evict" else ("gpu", "dram")
                    cfg = HALOConfig(
                        hot_ratio=1.0 / args.memory_ratio,
                        sink_tokens=64,
                        eviction=(method == "halo_evict"),
                        tiers=tiers,
                    )
                    wrap_with_halo(model, cfg)
                    from halo.halo_press import HALOPress
                    press_ctx = HALOPress(
                        compression_ratio=1.0 - 1.0 / args.memory_ratio,
                        sink_tokens=64,
                    )

                # ---- Time the eval (prefill + decode).
                t0 = time.perf_counter()
                sub_dir = out_dir / f"{method}_{task}_seed{seed}"
                sub_dir.mkdir(parents=True, exist_ok=True)
                score = float(evaluate_task(
                    model, tok, task=task,
                    limit=args.limit, output_dir=sub_dir,
                    press_ctx=press_ctx,
                ))
                elapsed = time.perf_counter() - t0
                peak_gb = torch.cuda.max_memory_allocated() / 1e9

                # Read decoded-token count from preds.jsonl for tokens/sec.
                preds_path = sub_dir / "preds.jsonl"
                decoded_tokens = 0
                if preds_path.exists():
                    for line in preds_path.read_text(encoding="utf-8").splitlines():
                        try:
                            row = json.loads(line)
                            text = row.get("pred", "")
                            decoded_tokens += len(tok.encode(text, add_special_tokens=False))
                        except (json.JSONDecodeError, KeyError):
                            continue
                tokens_per_sec = decoded_tokens / elapsed if elapsed > 0 else float("nan")
                print(f"[{method:11s} {task:25s} seed={seed}] "
                      f"score={score:.4f}  elapsed={elapsed:.1f}s  "
                      f"peak={peak_gb:.2f}GiB  tok/s={tokens_per_sec:.2f}",
                      flush=True)
                per_method.append({
                    "task": task, "seed": seed, "score": score,
                    "elapsed_s": elapsed, "peak_mem_gb": peak_gb,
                    "decoded_tokens": decoded_tokens,
                    "tokens_per_sec": tokens_per_sec,
                })

                # Detach HALO so next iter starts clean.
                if method.startswith("halo"):
                    if hasattr(model, "_halo_generate_patched"):
                        delattr(model, "_halo_generate_patched")
                    if hasattr(model, "_halo_cache"):
                        delattr(model, "_halo_cache")

        scores = [r["score"] for r in per_method]
        peaks = [r["peak_mem_gb"] for r in per_method]
        elapsed = [r["elapsed_s"] for r in per_method]
        tps = [r["tokens_per_sec"] for r in per_method
               if r["tokens_per_sec"] == r["tokens_per_sec"]]
        results[method] = {
            "n_cells": len(per_method),
            "mean_score": mean(scores) if scores else float("nan"),
            "std_score": stdev(scores) if len(scores) > 1 else 0.0,
            "mean_peak_gb": mean(peaks) if peaks else float("nan"),
            "mean_elapsed_s": mean(elapsed) if elapsed else float("nan"),
            "mean_tokens_per_sec": mean(tps) if tps else float("nan"),
            "per_cell": per_method,
        }
        print(f"[{method}] mean_score={results[method]['mean_score']:.4f} "
              f"peak={results[method]['mean_peak_gb']:.2f}GiB "
              f"tok/s={results[method]['mean_tokens_per_sec']:.2f}",
              flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps({
        "suite": "longbench-evict",
        "model": model_cfg["name_or_path"],
        "memory_ratio": args.memory_ratio,
        "tasks": args.tasks,
        "seeds": args.seeds,
        "methods": args.methods,
        "results": results,
    }, indent=2))
    print(f"\nmanifest -> {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
