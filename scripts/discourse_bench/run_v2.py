#!/usr/bin/env python3
"""Discourse-bench v2 runner with band-stratified analysis + bootstrap CI95.

Compared to v1:
  - reads discourse_eval_v2.jsonl (n=50+ per subtask, 3 bands × 8-10 templates)
  - per-band F1 (near / mid / far antecedent)
  - per-template F1 (so reviewers can see whether scores collapse to specific templates)
  - bootstrap CI95 on per-prompt scores
  - same Path D / Full / KIVI dispatch as v1

Usage:
  python scripts/discourse_bench/run_v2.py --method path_d \
    --input experiments/discourse_benchmark/discourse_eval_v2.jsonl \
    --output experiments/discourse_benchmark/v2/path_d
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def score(pred: str, gold: list[str]) -> float:
    if not pred:
        return 0.0
    p = pred.lower()
    for g in gold:
        if g.lower() in p:
            return 1.0
    return 0.0


def bootstrap_ci95(scores: list[float], n_iters: int = 10000, seed: int = 0):
    if not scores:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(scores)
    means = []
    for _ in range(n_iters):
        sample = [scores[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_iters)]
    hi = means[int(0.975 * n_iters) - 1]
    return sum(scores) / n, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["path_d", "kivi", "full"], required=True)
    ap.add_argument("--input", default="experiments/discourse_benchmark/discourse_eval_v2.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=0,
                    help="0 → use all prompts in input; otherwise truncate to first N.")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--model-path", default="/public/model_zoo/Qwen2.5-7B-Instruct")
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--hot-ratio", type=float, default=0.25)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input)]
    if args.n > 0:
        rows = rows[: args.n]
    print(f"[discourse_v2] using {len(rows)} prompts from {args.input}")

    print(f"[discourse_v2] loading {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path)
    cache_cfg = None
    if args.method == "kivi":
        from transformers.cache_utils import HQQQuantizedCache
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            device_map="auto", attn_implementation="sdpa")

        def make_kivi_cache():
            return HQQQuantizedCache(
                config=model.config, nbits=4,
                axis_key=0, axis_value=0,
                q_group_size=64, residual_length=128)
        cache_cfg = make_kivi_cache
    elif args.method == "path_d":
        from halo import HALOConfig, wrap_with_halo, install_preforward_peel
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            device_map="auto", attn_implementation="sdpa")
        cfg = HALOConfig(chunked=True, chunk_size=args.chunk_size,
                         recent_window=64, hot_ratio=args.hot_ratio, use_triton=True)
        wrap_with_halo(model, cfg)
        install_preforward_peel(model, prefill_chunk_tokens=4096, activation_threshold=8192)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            device_map="auto", attn_implementation="sdpa")

    print(f"[discourse_v2] baseline GPU = "
          f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
    model.eval()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_f = (out_dir / "preds.jsonl").open("w")

    scores_all: list[float] = []
    by_subtask: dict[str, list[float]] = defaultdict(list)
    by_band: dict[str, list[float]] = defaultdict(list)
    by_subtask_band: dict[tuple, list[float]] = defaultdict(list)
    by_template: dict[tuple, list[float]] = defaultdict(list)

    t_total = time.time()
    for i, r in enumerate(rows):
        prompt = r["input"]; gold = r["outputs"]
        subtask = r["subtask"]; band = r.get("band", "n/a")
        tpl = r.get("template_id", -1)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            inp = tok(prompt, return_tensors="pt", truncation=False).to(model.device)
            with torch.inference_mode():
                if args.method == "kivi":
                    cache = cache_cfg()
                    out_ids = model.generate(
                        **inp, past_key_values=cache,
                        max_new_tokens=args.max_new_tokens, do_sample=False,
                        pad_token_id=tok.eos_token_id)
                else:
                    out_ids = model.generate(
                        **inp, max_new_tokens=args.max_new_tokens, do_sample=False,
                        pad_token_id=tok.eos_token_id)
            pred = tok.decode(out_ids[0, inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
            s = score(pred, gold)
        except torch.cuda.OutOfMemoryError:
            pred = "[OOM]"; s = 0.0
            torch.cuda.empty_cache()
        wall = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**3

        scores_all.append(s)
        by_subtask[subtask].append(s)
        by_band[band].append(s)
        by_subtask_band[(subtask, band)].append(s)
        by_template[(subtask, tpl)].append(s)
        print(f"  [{i+1}/{len(rows)}] {subtask}/{band} tpl={tpl} idx={r.get('index',i)} "
              f"score={s} wall={wall:.1f}s pred[:60]={pred[:60]!r}")
        preds_f.write(json.dumps({
            "index": r.get("index", i), "subtask": subtask, "band": band,
            "template_id": tpl,
            "pred": pred, "gold": gold,
            "score": s, "wall_s": wall, "peak_gpu_gib": peak,
        }) + "\n")
        preds_f.flush()

    preds_f.close()
    wall_total = time.time() - t_total

    overall_mean, overall_lo, overall_hi = bootstrap_ci95(scores_all)
    summary = {
        "method": args.method,
        "n_total": len(scores_all),
        "overall_F1_pct": round(100 * overall_mean, 2),
        "overall_CI95_pct": [round(100 * overall_lo, 2), round(100 * overall_hi, 2)],
        "subtasks": {
            sub: {
                "n": len(v),
                "F1_pct": round(100 * sum(v) / max(1, len(v)), 2),
                "CI95_pct": [round(100 * x, 2) for x in bootstrap_ci95(v)[1:]],
            } for sub, v in by_subtask.items()
        },
        "bands": {
            band: {
                "n": len(v),
                "F1_pct": round(100 * sum(v) / max(1, len(v)), 2),
                "CI95_pct": [round(100 * x, 2) for x in bootstrap_ci95(v)[1:]],
            } for band, v in by_band.items()
        },
        "subtask_band": {
            f"{sub}/{band}": {
                "n": len(v),
                "F1_pct": round(100 * sum(v) / max(1, len(v)), 2),
            } for (sub, band), v in by_subtask_band.items()
        },
        "per_template": {
            f"{sub}/tpl{tpl}": {
                "n": len(v),
                "F1_pct": round(100 * sum(v) / max(1, len(v)), 2),
            } for (sub, tpl), v in by_template.items()
        },
        "wall_clock_s": wall_total,
        "model_path": args.model_path,
        "chunk_size": args.chunk_size if args.method == "path_d" else None,
        "hot_ratio": args.hot_ratio if args.method == "path_d" else None,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[discourse_v2] {args.method} overall: {summary['overall_F1_pct']:.2f}% "
          f"CI95={summary['overall_CI95_pct']} (n={summary['n_total']}) "
          f"wall={wall_total:.1f}s")
    for sub, s in summary["subtasks"].items():
        print(f"  {sub}: {s['F1_pct']:.2f}% CI95={s['CI95_pct']} (n={s['n']})")
    for band, s in summary["bands"].items():
        print(f"  band={band}: {s['F1_pct']:.2f}% CI95={s['CI95_pct']} (n={s['n']})")


if __name__ == "__main__":
    main()
