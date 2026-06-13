#!/usr/bin/env python3
"""Run ShadowKV head-to-head against Path D / KIVI / Full on ∞-Bench EnQA
or RULER NIAH adversarial.

Mirrors the shape of `scripts/discourse_bench/run_v2.py` so the same JSONL
preds.jsonl + summary.json schema is produced.
"""
from __future__ import annotations

import os
import argparse
import json
import random
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def score_substr(pred: str, gold: list[str]) -> float:
    if not pred:
        return 0.0
    p = pred.lower()
    for g in gold:
        if g.lower() in p:
            return 1.0
    return 0.0


def bootstrap_ci95(scores, n_iters=10000, seed=0):
    if not scores:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(scores)
    means = []
    for _ in range(n_iters):
        sample = [scores[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return sum(scores) / n, means[int(0.025 * n_iters)], means[int(0.975 * n_iters) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="JSONL with keys input, outputs[, subtask, band, ...]")
    ap.add_argument("--output", required=True)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--top-k-pages", type=int, default=32)
    ap.add_argument("--sink-pages", type=int, default=1)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--model-path", default=os.environ.get("HALO_DEFAULT_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input)]
    if args.n > 0:
        rows = rows[: args.n]
    print(f"[shadowkv] using {len(rows)} prompts from {args.input}")

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa")
    model.eval()

    from baselines.shadowkv_wrapper import (
        ShadowKVWrapperConfig, make_shadowkv_cache, wrap_with_shadowkv)
    cfg = ShadowKVWrapperConfig(
        rank=args.rank, page_size=args.page_size,
        top_k_pages=args.top_k_pages, sink_pages=args.sink_pages,
        cpu_offload_value=True,
    )
    wrap_with_shadowkv(model, cfg)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_f = (out_dir / "preds.jsonl").open("w")

    scores: list[float] = []
    t_total = time.time()
    for i, r in enumerate(rows):
        prompt = r["input"]; gold = r["outputs"]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            inp = tok(prompt, return_tensors="pt", truncation=False).to(model.device)
            with torch.inference_mode():
                cache = make_shadowkv_cache(cfg)
                out_ids = model.generate(
                    **inp, past_key_values=cache,
                    max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tok.eos_token_id)
            pred = tok.decode(out_ids[0, inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
            s = score_substr(pred, gold)
        except torch.cuda.OutOfMemoryError:
            pred = "[OOM]"; s = 0.0
            torch.cuda.empty_cache()
        wall = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**3
        scores.append(s)
        print(f"  [{i+1}/{len(rows)}] score={s} wall={wall:.1f}s peak={peak:.2f}GiB pred[:60]={pred[:60]!r}")
        preds_f.write(json.dumps({
            "index": r.get("index", i),
            "subtask": r.get("subtask"), "band": r.get("band"),
            "pred": pred, "gold": gold, "score": s,
            "wall_s": wall, "peak_gpu_gib": peak,
        }) + "\n")
        preds_f.flush()
    preds_f.close()
    wall_total = time.time() - t_total

    mean, lo, hi = bootstrap_ci95(scores)
    summary = {
        "method": "shadowkv",
        "rank": args.rank, "page_size": args.page_size,
        "top_k_pages": args.top_k_pages, "sink_pages": args.sink_pages,
        "n_total": len(scores),
        "overall_F1_pct": round(100 * mean, 2),
        "overall_CI95_pct": [round(100 * lo, 2), round(100 * hi, 2)],
        "wall_clock_s": wall_total,
        "model_path": args.model_path,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[shadowkv] overall {summary['overall_F1_pct']:.2f}% CI95={summary['overall_CI95_pct']}"
          f" n={summary['n_total']} wall={wall_total:.1f}s")


if __name__ == "__main__":
    main()
