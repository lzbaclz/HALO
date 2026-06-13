#!/usr/bin/env python3
"""Run the k-hop benchmark (Plan A empirical validation).

Subtask scoring: for a k-hop prompt with gold keys [K_1, ..., K_k]:
  - correct iff ALL k keys appear in the prediction AND in correct order.
  - partial-credit (per-hop): fraction of keys found in any order.

This dual scoring lets us separate "did the cache lose a hop" from
"did the model give the keys out-of-order".
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# NB: must be a 4-digit run that isn't part of a longer digit sequence.
# Cannot use \b...\b because that fails on outputs like "7259Human:" where
# the trailing letter is a word character (no \b between digit and letter
# in the regex sense). The lookbehind/lookahead constraints below only
# reject adjacent digits, which is what we actually want.
_DIGIT_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")


def score_kha(pred: str, gold_keys: list[str]) -> tuple[float, float]:
    """Return (strict_correct, per_hop_recall)."""
    if not pred:
        return 0.0, 0.0
    # Extract 4-digit numbers in order
    found = _DIGIT_RE.findall(pred)
    # Strict: gold_keys is a prefix of found (in order, allowing extras)
    # Simpler: check that found, restricted to gold_keys' set, is equal to gold_keys.
    found_in_gold = [x for x in found if x in gold_keys]
    strict = 1.0 if found_in_gold[: len(gold_keys)] == gold_keys else 0.0
    recall = sum(1 for k in gold_keys if k in found) / max(1, len(gold_keys))
    return strict, recall


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
    ap.add_argument("--method", choices=["path_d", "kivi", "full", "h2o", "streamingllm"],
                    required=True)
    ap.add_argument("--input", default="experiments/khop_bench/eval.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--model-path", default="/public/model_zoo/Qwen2.5-7B-Instruct")
    # Path D config
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--hot-ratio", type=float, default=0.25)
    # Eviction-policy config (h2o, streamingllm)
    ap.add_argument("--retention-ratio", type=float, default=0.10,
                    help="r in Thm 3.3 — fraction of positions retained.")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input)]
    if args.n > 0:
        rows = rows[: args.n]
    print(f"[khop_bench] {args.method} on {len(rows)} prompts")

    tok = AutoTokenizer.from_pretrained(args.model_path)
    cache_factory = None
    press_ctx = None  # callable(model) -> contextmanager; set by H2O/StreamingLLM branches
    if args.method == "kivi":
        from transformers.cache_utils import HQQQuantizedCache
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            device_map="auto", attn_implementation="sdpa")
        cache_factory = lambda: HQQQuantizedCache(
            config=model.config, nbits=4, axis_key=0, axis_value=0,
            q_group_size=64, residual_length=128)
    elif args.method == "path_d":
        from halo import HALOConfig, wrap_with_halo, install_preforward_peel
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            device_map="auto", attn_implementation="sdpa")
        cfg = HALOConfig(chunked=True, chunk_size=args.chunk_size,
                         recent_window=64, hot_ratio=args.hot_ratio, use_triton=True)
        wrap_with_halo(model, cfg)
        install_preforward_peel(model, prefill_chunk_tokens=4096, activation_threshold=8192)
    elif args.method in ("h2o", "streamingllm", "snapkv"):
        # IMPORTANT: use baselines.REGISTRY (the same path
        # baselines/runner.py uses for §5.2 commitment baselines), which
        # returns the kvpress Press object. We then wrap model.generate()
        # in `with press_ctx(model):` to actually fire the eviction hooks.
        # The earlier version called `wrapper.apply(model)` and threw away
        # the return value, so the press never fired (measured Full attention).
        from baselines import REGISTRY
        # H2O (ObservedAttentionPress) requires eager attention to read
        # post-softmax attention weights; StreamingLLM works with sdpa.
        attn_impl = "eager" if args.method in ("h2o", "snapkv") else "sdpa"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            device_map="auto", attn_implementation=attn_impl)
        mr = int(round(1.0 / max(0.01, args.retention_ratio)))
        press_ctx = REGISTRY[args.method](model, memory_ratio=mr)
    else:  # full
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            device_map="auto", attn_implementation="sdpa")
    model.eval()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_f = (out_dir / "preds.jsonl").open("w")

    strict_by_k: dict[int, list[float]] = defaultdict(list)
    recall_by_k: dict[int, list[float]] = defaultdict(list)
    t_total = time.time()
    for i, r in enumerate(rows):
        prompt = r["input"]; gold_keys = r["keys"]; k = r["k"]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            inp = tok(prompt, return_tensors="pt", truncation=False).to(model.device)
            from contextlib import nullcontext
            cm = press_ctx(model) if press_ctx is not None else nullcontext()
            with torch.inference_mode(), cm:
                if cache_factory is not None:
                    out_ids = model.generate(
                        **inp, past_key_values=cache_factory(),
                        max_new_tokens=args.max_new_tokens, do_sample=False,
                        pad_token_id=tok.eos_token_id)
                else:
                    out_ids = model.generate(
                        **inp, max_new_tokens=args.max_new_tokens, do_sample=False,
                        pad_token_id=tok.eos_token_id)
            pred = tok.decode(out_ids[0, inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
            strict, recall = score_kha(pred, gold_keys)
        except torch.cuda.OutOfMemoryError:
            pred = "[OOM]"; strict = 0.0; recall = 0.0
            torch.cuda.empty_cache()
        wall = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**3
        strict_by_k[k].append(strict)
        recall_by_k[k].append(recall)
        print(f"  [{i+1}/{len(rows)}] k={k} strict={strict:.0f} recall={recall:.2f} "
              f"wall={wall:.1f}s pred[:80]={pred[:80]!r}")
        preds_f.write(json.dumps({
            "index": r["index"], "k": k, "gold_keys": gold_keys,
            "pred": pred, "strict": strict, "recall": recall,
            "wall_s": wall, "peak_gpu_gib": peak,
        }) + "\n")
        preds_f.flush()
    preds_f.close()
    wall_total = time.time() - t_total

    summary = {
        "method": args.method,
        "n_total": len(rows),
        "wall_clock_s": wall_total,
        "by_k": {
            str(k): {
                "n": len(v),
                "strict_pct": round(100 * sum(v) / max(1, len(v)), 2),
                "recall_pct": round(100 * sum(recall_by_k[k]) / max(1, len(recall_by_k[k])), 2),
                "strict_CI95_pct": [round(100 * x, 2) for x in bootstrap_ci95(v)[1:]],
            } for k, v in strict_by_k.items()
        },
        "retention_ratio": args.retention_ratio if args.method in ("h2o", "streamingllm") else None,
        "chunk_size": args.chunk_size if args.method == "path_d" else None,
        "hot_ratio": args.hot_ratio if args.method == "path_d" else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[khop_bench] {args.method} (n={len(rows)}, wall={wall_total:.1f}s)")
    for k, s in sorted(summary["by_k"].items(), key=lambda x: int(x[0])):
        print(f"  k={k}: strict={s['strict_pct']}% recall={s['recall_pct']}% (n={s['n']})")


if __name__ == "__main__":
    main()
