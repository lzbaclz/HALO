"""Run Full attention on the same 20 PassageRetrieval-en examples that
Path D was scored on, for a fair F1 comparison.

This closes the loop on the reviewer's MUST-2 ask: not just (F1, peak,
wall), but a comparable F1 to verify Path D is lossless on real data.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    import torch
    from baselines.longbench_eval import evaluate_task

    print("Loading Qwen/Qwen2.5-7B (Full attention baseline)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B", torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    t0 = time.time()
    score = evaluate_task(model, tok, task="passage_retrieval_en", limit=20, progress=True)
    wall_s = time.time() - t0
    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print(f"\n=== Full attention, 20 PR-en examples ===")
    print(f"  F1               : {score:.2f}")
    print(f"  Peak GPU (GiB)   : {peak_gib:.2f}")
    print(f"  Wall-clock (s)   : {wall_s:.1f}")

    out = {
        "suite": "longbench-v1", "method": "full",
        "memory_ratio": 1, "model": "Qwen/Qwen2.5-7B",
        "task": "passage_retrieval_en", "limit": 20,
        "scores": {"passage_retrieval_en": score},
        "peak_gpu_gib": peak_gib, "wall_clock_s": wall_s,
        "seed": 0,
    }
    out_dir = Path("experiments/runs/qwen2-5-7b/longbench/full_1x_pr20")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(out, indent=2))
    print(f"  wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
