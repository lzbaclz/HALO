"""FU_W5: KIVI-style 4-bit KV-cache via HF QuantizedCache (HQQ backend).

Runs Qwen 2.5-7B on $\infty$-Bench En.QA at 65K context with the official
HF transformers `cache_implementation="quantized"` path. This replaces the
from-scratch 226-LOC port at baselines/kivi_cache.py (which gave 0.13% F1
on the same cell) with a baseline that uses the upstream-supported quantized
cache backend.

Usage:
    python scripts/run_kivi_hf.py --output experiments/auxiliary_cells/W5_KIVI_official
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import QuantizedCache

from baselines.infinitebench_eval import qa_f1_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models/qwen2-5-7b.yaml")
    ap.add_argument("--task", default="en_qa")
    ap.add_argument("--context-length", type=int, default=65000)
    ap.add_argument("--n-examples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--nbits", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(open(args.config))
    model_name = cfg["name"]

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    # Load $\infty$-Bench En.QA prompts. Use the same harness path as
    # baselines/infinitebench_eval.py.
    from datasets import load_dataset
    ds = load_dataset("xinrongzhang2022/InfiniteBench", split="longbook_qa_eng")
    # Filter to ~65K context window
    examples = []
    for ex in ds:
        prompt = ex["input"]
        if 60000 <= len(tok.encode(prompt)) <= args.context_length:
            examples.append(ex)
        if len(examples) >= args.n_examples:
            break

    print(f"[FU_W5] loaded {len(examples)} prompts at ~{args.context_length} ctx")

    preds = []
    correct_f1 = []
    for i, ex in enumerate(examples):
        prompt = ex["input"]
        gold = ex["answer"]
        # Apply Qwen chat template lightly — base model, no chat template.
        ids = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=args.context_length).to(model.device)

        # Create quantized cache with HQQ int4 backend.
        # transformers 5.x signature:
        #   QuantizedCache(backend, config, nbits, axis_key, axis_value, ...)
        cache = QuantizedCache(
            backend="hqq",
            config=model.config,
            nbits=args.nbits,
            axis_key=0, axis_value=0,
            q_group_size=64,
            residual_length=128,
        )

        t0 = time.time()
        with torch.no_grad():
            out_ids = model.generate(
                **ids,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                past_key_values=cache,
            )
        wall = time.time() - t0

        pred = tok.decode(out_ids[0, ids.input_ids.shape[1]:],
                          skip_special_tokens=True)
        f1 = qa_f1_score(pred, gold)
        preds.append({"index": i, "pred": pred[:200],
                      "gold": gold, "f1": f1, "wall_s": wall})
        correct_f1.append(f1)
        print(f"  [{i+1}/{len(examples)}] f1={f1:.3f} wall={wall:.1f}s")

    mean_f1 = sum(correct_f1) / max(1, len(correct_f1))
    summary = {
        "method": "kivi_hf_quantized_int4",
        "backend": "hqq",
        "nbits": args.nbits,
        "n": len(correct_f1),
        "mean_f1": mean_f1 * 100,
        "context_length": args.context_length,
        "task": args.task,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "preds.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    print(f"\n[FU_W5] mean F1 = {mean_f1*100:.2f}% over n={len(correct_f1)}")


if __name__ == "__main__":
    main()
