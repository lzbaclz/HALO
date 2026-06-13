"""Re-run the self-contained KIVI baseline on Llama-2-7B to test the
RoPE-diagnosis hypothesis from the appendix.

The reviewer's SHOULD-1 ask: the original KIVI paper used Llama-2 in
its main tables; if our per-channel implementation produces sensible
numbers on Llama-2 but not on Qwen2.5, the bug is Qwen-specific (most
likely a RoPE-frequency-mismatch with Qwen's larger rope_theta) and
the paper's diagnosis is verified.

Output: experiments/runs/llama-2-7b/longbench/kivi_4x/manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    import torch
    from baselines.kivi_cache import wrap_with_kivi
    from baselines.longbench_eval import evaluate_task

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--tasks", nargs="+",
                    default=["narrativeqa", "gov_report", "passage_retrieval_en"])
    ap.add_argument("--memory-ratios", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out-dir", default="experiments/runs/llama-2-7b/longbench")
    args = ap.parse_args()

    print(f"Loading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )

    for mr in args.memory_ratios:
        bits = max(16 // mr, 2)
        print(f"\n=== memory_ratio={mr} (KIVI bits={bits}) — Llama-2 ===")
        # Reset patched state across ratios.
        if getattr(model, "_kivi_patched", False):
            model._kivi_patched = False
        wrap_with_kivi(model, memory_ratio=mr)
        per_task = {}
        for task in args.tasks:
            score = evaluate_task(model, tok, task=task, limit=args.limit, progress=False)
            per_task[task] = score
            print(f"  {task} = {score:.2f}")
        out = {
            "suite": "longbench-v1",
            "method": "kivi",
            "memory_ratio": mr,
            "bits": bits,
            "implementation": "baselines.kivi_cache (self-contained)",
            "model": args.model,
            "scores": per_task,
            "seed": 0,
            "limit": args.limit,
        }
        out_path = Path(args.out_dir) / f"kivi_{mr}x" / "manifest.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  wrote {out_path}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
