#!/usr/bin/env python3
"""k-hop adversarial benchmark generator for Plan A
(Thm 3.3 / appendix sec:appendix-khop-bound empirical validation).

Each prompt:
  1. Sample k secret positions uniformly in [0, n_filler_paras).
  2. Inject a sentence at each: "The {j}-th secret key is {DDDD}." for
     j = 1..k.
  3. Append a question: "Report the secret keys in order, separated by ', '."
  4. Gold: "{key_1}, {key_2}, ..., {key_k}".

This realises the D_k adversary of Thm 3.3: the k indices are random,
the policy's S is a function of (K, V) only, so the probability of
retaining all k indices under a fraction-r commitment policy is ≤ r^k.

Usage:
  python scripts/khop_bench/generate.py --k-values 1 2 3 4 \\
    --n-per-k 20 --seed 0 \\
    --out experiments/khop_bench/eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Reuse the discourse-bench v1 filler primitives to keep the distractor
# distribution identical across benchmarks (same wiki-style content).
import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "discourse_bench"))
from generate import make_filler_paragraph  # noqa: E402


def make_filler_block(rng: random.Random, n_paras: int) -> list[str]:
    """Return list of filler paragraphs (kept as a list so we can splice in keys)."""
    return [make_filler_paragraph(rng, n_sentences=rng.randint(4, 7))
            for _ in range(n_paras)]


def make_khop_prompt(rng: random.Random, k: int, n_filler_paras: int) -> dict:
    """Build a k-hop prompt with k randomly-placed secret keys."""
    paragraphs = make_filler_block(rng, n_filler_paras)
    # Sample k unique positions
    n = len(paragraphs)
    positions = rng.sample(range(n), k)
    positions.sort()
    keys = [f"{rng.randint(1000, 9999)}" for _ in range(k)]

    # Insert each "The j-th secret key is K." at its position (j = 1..k by position order)
    for j, (pos, key) in enumerate(zip(positions, keys), start=1):
        anchor = f"The {ordinal(j)} secret key is {key}."
        paragraphs[pos] = anchor + " " + paragraphs[pos]

    body = "\n\n".join(paragraphs)
    question = (
        f"\n\nQuestion: The passage above contains {k} secret keys, "
        f"labeled {ordinal_list(k)}. "
        f"Report all {k} secret keys in label order, separated by commas. "
        f"Answer with only the keys.\nAnswer:"
    )
    gold_str = ", ".join(keys)
    return {
        "k": k,
        "positions": positions,
        "keys": keys,
        "prompt": body + question,
        "gold_str": gold_str,
        # gold list for substring scoring: require all k keys present, in order
        "gold_substrings": keys,
    }


def ordinal(j: int) -> str:
    if 10 <= j % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(j % 10, "th")
    return f"{j}{suf}"


def ordinal_list(k: int) -> str:
    if k == 1:
        return "the first"
    if k == 2:
        return "first and second"
    return ", ".join(ordinal(j) for j in range(1, k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-values", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--n-per-k", type=int, default=20)
    ap.add_argument("--target-ctx-tokens", type=int, default=32000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="experiments/khop_bench/eval.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    paras_needed = max(20, args.target_ctx_tokens // 100)

    prompts = []
    for k in args.k_values:
        for _ in range(args.n_per_k):
            p = make_khop_prompt(rng, k=k, n_filler_paras=paras_needed)
            prompts.append(p)

    char_counts = [len(p["prompt"]) for p in prompts]
    print(f"Generated {len(prompts)} prompts "
          f"({args.n_per_k} per k, k in {args.k_values}).")
    print(f"  chars   min={min(char_counts)} median={sorted(char_counts)[len(char_counts)//2]} max={max(char_counts)}")
    print(f"  tokens (approx) min={min(char_counts)//4} median={sorted(char_counts)[len(char_counts)//2]//4} max={max(char_counts)//4}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, p in enumerate(prompts):
            row = {
                "index": i,
                "k": p["k"],
                "positions": p["positions"],
                "keys": p["keys"],
                "input": p["prompt"],
                "outputs": [p["gold_str"]] + p["gold_substrings"],
            }
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
