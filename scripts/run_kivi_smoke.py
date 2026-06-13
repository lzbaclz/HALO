"""Smoke run of the self-contained KIVI baseline on a small LongBench subset.

We exercise the KIVI cache end-to-end on Qwen2.5-7B (the substituted
Llama-3-8B-Instruct), 3 representative LongBench subtasks
(narrativeqa, gov_report, passage_retrieval_en — covering single-doc QA,
summarization, retrieval), 25 examples each, at memory_ratio∈{4, 8}.

This is the minimum-viable answer to reviewer Q8 / W9 ("add KIVI to the
LongBench head-to-head"). The numbers are produced with the bespoke
implementation in ``baselines/kivi_cache.py`` rather than kvpress 0.5.3
(which does not ship KIVIPress); we cite the discrepancy in the paper.

Outputs:
- ``experiments/runs/qwen2-5-7b/longbench/kivi_4x/manifest.json``
- ``experiments/runs/qwen2-5-7b/longbench/kivi_8x/manifest.json``
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--tasks", nargs="+",
                    default=["narrativeqa", "gov_report", "passage_retrieval_en"])
    ap.add_argument("--memory-ratios", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--out-dir", default="experiments/runs/qwen2-5-7b/longbench")
    args = ap.parse_args()

    from baselines.kivi_cache import wrap_with_kivi
    from baselines.longbench_eval import evaluate_task

    print(f"Loading {args.model} (this will take ~1-2 min in bf16)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )

    for mr in args.memory_ratios:
        print(f"\n=== memory_ratio = {mr} (KIVI bits = {max(16 // mr, 2)}) ===")
        wrap_with_kivi(model, memory_ratio=mr)
        per_task_scores = {}
        for task in args.tasks:
            print(f"  scoring {task}...")
            score = evaluate_task(model, tok, task=task, limit=args.limit, progress=False)
            per_task_scores[task] = score
            print(f"    {task} = {score:.2f}")
        out = {
            "suite": "longbench-v1",
            "method": "kivi",
            "memory_ratio": mr,
            "bits": max(16 // mr, 2),
            "implementation": "baselines.kivi_cache (self-contained; "
                              "kvpress 0.5.3 does not ship KIVIPress)",
            "model": args.model,
            "scores": per_task_scores,
            "seed": 0,
            "limit": args.limit,
        }
        out_path = Path(args.out_dir) / f"kivi_{mr}x" / "manifest.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  wrote {out_path}")

        # Reset the patched generate so the next ratio's wrap takes effect.
        if getattr(model, "_kivi_patched", False):
            model._kivi_patched = False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
