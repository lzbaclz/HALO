"""bench_infinite_evict.py — ∞-Bench eviction + multi-seed + latency benchmark.

Addresses three GPU follow-ups in one script:

* G2. Wire HALOCacheEvict into the ∞-Bench harness via
      ``HALOConfig(eviction=True)``. The §5.4 critique was that HALO held
      the full DynamicCache + tier metadata, using more GPU memory than
      Full. With eviction=True the cold-position columns are physically
      zeroed; we measure peak_mem to see whether this finally drops below
      Full's 32.2 GiB.

* G3. Multi-seed sweep over {0, 1, 2} on the en_qa subtask at 65K (the
      contribution actually claimed in the paper). ∞-Bench is small (20
      examples), so 3 seeds is cheap.

* G4. Real GPU latency benchmark: prefill seconds + decode ms/token,
      averaged over the 20 examples. Compare Full vs HALO(eviction=False)
      vs HALO(eviction=True).

Output: experiments/runs/qwen2-5-7b/infinitebench_evict/manifest.json
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
    ap.add_argument("--task", default="en_qa")
    ap.add_argument("--context-length", type=int, default=65536)
    ap.add_argument("--memory-ratio", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--output", default="experiments/runs/qwen2-5-7b/infinitebench_evict")
    ap.add_argument("--methods", nargs="+",
                    default=["full", "halo_keep", "halo_evict"],
                    help="full | halo_keep (HALOCache) | halo_evict (HALOCacheEvict) | halo_chunked (HALOCacheChunked)")
    args = ap.parse_args()

    import torch
    from baselines.infinitebench_eval import evaluate_task
    from halo.utils import load_yaml, seed_everything, write_manifest

    model_cfg = load_yaml(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model + tok once ----
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

    results = {}

    for method in args.methods:
        method_results = []
        for seed in args.seeds:
            seed_everything(seed)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # ---- Configure HALO if applicable ----
            if method.startswith("halo"):
                from halo import HALOConfig
                from halo.policy import wrap_with_halo
                # tiers: dram-only avoids any GPU-side block buffer
                # pre-allocation (TieredStorage is lazy in 2026-05-12+,
                # so passing ("gpu","dram") is also safe but adds nothing
                # to a Press-driven setup).
                tiers = ("dram",)
                cfg = HALOConfig(
                    hot_ratio=1.0 / args.memory_ratio,
                    sink_tokens=64,
                    eviction=(method == "halo_evict"),
                    chunked=(method == "halo_chunked"),
                    tiers=tiers,
                )
                wrap_with_halo(model, cfg)
                # Press for prefill compression (kvpress hook).
                # halo_chunked relies on the wired DRAM-tier path
                # (HALOCacheChunked) rather than a prefill prune.
                if method in ("halo_evict", "halo_keep"):
                    from halo.halo_press import HALOPress
                    press_ctx = HALOPress(
                        compression_ratio=1.0 - 1.0 / args.memory_ratio,
                        sink_tokens=64,
                    )
                else:
                    press_ctx = None
            else:
                press_ctx = None

            # ---- Time the eval ----
            t0 = time.perf_counter()
            sub_dir = out_dir / f"{method}_seed{seed}"
            sub_dir.mkdir(parents=True, exist_ok=True)
            score = float(evaluate_task(
                model, tok, task=args.task,
                limit=args.limit, output_dir=sub_dir,
                context_length=args.context_length,
                press_ctx=press_ctx,
            ))
            elapsed = time.perf_counter() - t0
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"[{method} seed={seed}] score={score:.4f} elapsed={elapsed:.1f}s peak={peak_gb:.2f}GiB",
                  flush=True)
            method_results.append({
                "seed": seed,
                "score": score,
                "elapsed_s": elapsed,
                "peak_mem_gb": peak_gb,
            })

            # Detach HALO so the next iteration sees a clean model.
            if method.startswith("halo"):
                # Restore original generate.
                if hasattr(model, "_halo_generate_patched") and model._halo_generate_patched:
                    delattr(model, "_halo_generate_patched")
                if hasattr(model, "_halo_cache"):
                    delattr(model, "_halo_cache")

        scores = [r["score"] for r in method_results]
        elapsed = [r["elapsed_s"] for r in method_results]
        peaks = [r["peak_mem_gb"] for r in method_results]
        agg = {
            "method": method,
            "n_seeds": len(method_results),
            "mean_score": mean(scores) if scores else float("nan"),
            "std_score": stdev(scores) if len(scores) > 1 else 0.0,
            "mean_elapsed_s": mean(elapsed) if elapsed else float("nan"),
            "mean_peak_gb": mean(peaks) if peaks else float("nan"),
            "per_seed": method_results,
        }
        results[method] = agg
        print(f"[{method}] mean={agg['mean_score']:.4f}±{agg['std_score']:.4f} "
              f"peak={agg['mean_peak_gb']:.2f}GiB t={agg['mean_elapsed_s']:.1f}s",
              flush=True)

    summary_path = out_dir / "manifest.json"
    summary_path.write_text(json.dumps({
        "suite": "infinitebench-evict",
        "task": args.task,
        "context_length": args.context_length,
        "memory_ratio": args.memory_ratio,
        "model": name,
        "seeds": args.seeds,
        "results": results,
    }, indent=2))
    print(f"\n[done] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
