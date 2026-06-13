"""LongBench-v2 evaluator for Full + Path D lossless verification.

LongBench-v2 (Bai et al. 2024) is a multiple-choice extension to LongBench
designed to be more discriminating than the noisy partial-F1 metrics of
LongBench-v1. We use the multiple-choice (A/B/C/D) accuracy metric.

Usage:
    python scripts/run_longbench_v2.py --method full --n-examples 30 --seed 0 \\
        --output experiments/cell_LBv2/full

For paired comparison, run twice with --method full and --method path_d at
same --seed --n-examples to get the same prompts.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


PROMPT = """Please read the following text and answer the question below.

<text>
{context}
</text>

Question: {question}

Choices:
A. {choice_A}
B. {choice_B}
C. {choice_C}
D. {choice_D}

The correct answer is (one letter only):"""


def parse_letter(s: str) -> str:
    m = re.search(r"\b([ABCD])\b", s.strip())
    return m.group(1) if m else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["full", "path_d", "kivi", "quest", "quest_path_d"])
    ap.add_argument("--config", default="configs/models/qwen2-5-7b.yaml")
    ap.add_argument("--memory-ratio", type=int, default=4)
    ap.add_argument("--context-length", type=int, default=16384)
    ap.add_argument("--n-examples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    ds = load_dataset("THUDM/LongBench-v2", split="train")
    n_total = len(ds)
    import numpy as np
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(n_total, size=min(args.n_examples, n_total),
                         replace=False).tolist()
    indices.sort()

    print(f"[LBv2] method={args.method} n={len(indices)} seed={args.seed} ctx={args.context_length}")
    print(f"[LBv2] indices first 5: {indices[:5]}")

    # Load model with chosen method
    import torch
    import yaml
    cfg = yaml.safe_load(open(args.config))
    name = cfg["name_or_path"]

    from baselines.runner import _load_model_and_tokenizer
    cfg_for_load = dict(cfg)
    if args.method == "path_d":
        cfg_for_load["attn_implementation"] = "eager"
    try:
        model, tok = _load_model_and_tokenizer(cfg_for_load, method=args.method,
                                                memory_ratio=args.memory_ratio)
    except TypeError:
        model, tok = _load_model_and_tokenizer(cfg_for_load)
    model.eval()

    # Method-specific wrapping
    if args.method == "kivi":
        from baselines.kivi_cache import wrap_with_kivi
        wrap_with_kivi(model, memory_ratio=args.memory_ratio)
    elif args.method == "path_d":
        from halo import HALOConfig, wrap_with_halo, install_preforward_peel
        cfg_obj = HALOConfig(chunked=True, chunk_size=512, recent_window=64,
                              hot_ratio=1.0 / max(1.0, float(args.memory_ratio)),
                              use_triton=True)
        wrap_with_halo(model, cfg_obj)
        install_preforward_peel(model, prefill_chunk_tokens=4096,
                                 activation_threshold=8192)

    preds = []
    scores = []
    from tqdm import tqdm
    for i, idx in enumerate(tqdm(indices, desc="LBv2")):
        ex = ds[int(idx)]
        ctx = ex["context"]
        # Truncate context to fit in token budget. Reserve ~512 for prompt+choices+gen
        prompt = PROMPT.format(context=ctx[:args.context_length * 4],  # rough char approx
                                question=ex["question"],
                                choice_A=ex["choice_A"], choice_B=ex["choice_B"],
                                choice_C=ex["choice_C"], choice_D=ex["choice_D"])
        ids = tok(prompt, return_tensors="pt", truncation=True,
                   max_length=args.context_length - 32).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**ids, max_new_tokens=8, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
        pred = tok.decode(out_ids[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
        letter = parse_letter(pred)
        gold = ex["answer"]
        score = 1.0 if letter == gold else 0.0
        preds.append({"_id": ex["_id"], "pred": letter, "raw": pred[:40],
                       "gold": gold, "score": score})
        scores.append(score)

    preds_path = out / "preds.jsonl"
    with open(preds_path, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")

    import numpy as np
    acc = float(np.mean(scores))
    summary = {"method": args.method, "task": "LongBench-v2",
                "n": len(indices), "seed": args.seed, "accuracy": acc,
                "model": name, "context_length": args.context_length}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[LBv2] {args.method} n={len(indices)} acc={acc*100:.2f}%")


if __name__ == "__main__":
    main()
