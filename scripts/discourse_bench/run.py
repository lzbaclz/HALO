#!/usr/bin/env python3
"""Run Path D / KIVI / Full attention on the pilot discourse benchmark.

Mirrors scripts/run_pathd_ruler.py's structure but operates on our
discourse_eval.jsonl (3 subtasks: DA / BR / CM).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def score(pred: str, gold: list[str]) -> float:
    """Case-insensitive substring containment."""
    if not pred: return 0.0
    p = pred.lower()
    for g in gold:
        if g.lower() in p:
            return 1.0
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["path_d", "kivi", "full"], required=True)
    ap.add_argument("--input", default="experiments/discourse_benchmark/discourse_eval.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=18, help="Total prompts (split across DA/BR/CM evenly)")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--model-path", default="/public/model_zoo/Qwen2.5-7B-Instruct")
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--hot-ratio", type=float, default=0.25)
    args = ap.parse_args()

    # Load examples
    rows = [json.loads(l) for l in open(args.input)]
    # Take first n//3 of each subtask
    per_sub = args.n // 3
    selected = []
    for sub in ("DA", "BR", "CM"):
        sub_rows = [r for r in rows if r["subtask"] == sub]
        selected.extend(sub_rows[:per_sub])
    print(f"[discourse_bench] selected {len(selected)} prompts ({per_sub} per subtask)")

    # Load model + tokenizer
    print(f"[discourse_bench] loading {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if args.method == "kivi":
        # Reuse the KIVI HF int4 wrapper used in W7e (FU_W5/W7e)
        from transformers.cache_utils import HQQQuantizedCache
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            device_map="auto", attn_implementation="sdpa")
        # nbits=4, q_group_size=64 matches the W7e/FU_W5 config
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
        cache_cfg = None
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            device_map="auto", attn_implementation="sdpa")
        cache_cfg = None

    print(f"[discourse_bench] model loaded; baseline GPU = "
          f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
    model.eval()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_f = (out_dir / "preds.jsonl").open("w")

    scores_by_subtask: dict[str, list[float]] = {"DA": [], "BR": [], "CM": []}
    walls_by_subtask: dict[str, list[float]] = {"DA": [], "BR": [], "CM": []}
    t_total = time.time()
    for i, r in enumerate(selected):
        prompt = r["input"]
        gold = r["outputs"]
        subtask = r["subtask"]

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            inp = tok(prompt, return_tensors="pt", truncation=False).to(model.device)
            with torch.inference_mode():
                if args.method == "kivi":
                    cache = cache_cfg()  # fresh HQQQuantizedCache per prompt
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
        except torch.cuda.OutOfMemoryError as e:
            pred = "[OOM]"
            s = 0.0
            print(f"  [{i+1}/{len(selected)}] {subtask}: OOM at len(input)={len(prompt)}")
            torch.cuda.empty_cache()
        wall = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**3

        scores_by_subtask[subtask].append(s)
        walls_by_subtask[subtask].append(wall)
        print(f"  [{i+1}/{len(selected)}] {subtask} idx={r['index']} "
              f"score={s} wall={wall:.1f}s peak={peak:.2f}GiB "
              f"pred[:80]={pred[:80]!r}")
        preds_f.write(json.dumps({
            "index": r["index"], "subtask": subtask,
            "pred": pred, "gold": gold,
            "score": s, "wall_s": wall, "peak_gpu_gib": peak,
        }) + "\n")
        preds_f.flush()

    preds_f.close()
    wall_total = time.time() - t_total

    summary = {
        "method": args.method,
        "n_per_subtask": per_sub,
        "n_total": len(selected),
        "subtasks": {
            sub: {
                "n": len(scores_by_subtask[sub]),
                "mean_score": (sum(scores_by_subtask[sub]) / len(scores_by_subtask[sub]))
                              if scores_by_subtask[sub] else 0.0,
                "mean_wall_s": (sum(walls_by_subtask[sub]) / len(walls_by_subtask[sub]))
                               if walls_by_subtask[sub] else 0.0,
            } for sub in ("DA", "BR", "CM")
        },
        "overall_score_pct": 100.0 * sum(sum(v) for v in scores_by_subtask.values())
                             / max(1, sum(len(v) for v in scores_by_subtask.values())),
        "wall_clock_s": wall_total,
        "model_path": args.model_path,
        "chunk_size": args.chunk_size if args.method == "path_d" else None,
        "hot_ratio": args.hot_ratio if args.method == "path_d" else None,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[discourse_bench] {args.method} overall: {summary['overall_score_pct']:.2f}% "
          f"over n={summary['n_total']} wall={wall_total:.1f}s")
    for sub, s in summary["subtasks"].items():
        print(f"  {sub}: {s['mean_score']*100:.1f}% (n={s['n']}, wall={s['mean_wall_s']:.1f}s)")


if __name__ == "__main__":
    main()
