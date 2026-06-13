#!/usr/bin/env python3
"""Re-score existing preds.jsonl with the fixed digit regex.

The original ``_DIGIT_RE = r"\b(\d{4})\b"`` failed when a key was followed
by a letter (e.g. ``"7259Human:"``) because there is no word boundary
between a digit and a letter in regex terms. This drops ~all multi-key
matches when the model continues after the comma-separated answer.

Fix: ``(?<!\d)(\d{4})(?!\d)`` — only reject adjacent digits, accept
adjacency to any non-digit character (letters, punctuation, whitespace).
Re-scores every existing preds.jsonl in-place and rewrites summary.json.

Usage:
    python scripts/khop_bench/rescore.py experiments/khop_bench
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

_DIGIT_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")


def score_kha(pred: str, gold_keys: list[str]) -> tuple[float, float]:
    if not pred:
        return 0.0, 0.0
    found = _DIGIT_RE.findall(pred)
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


def rescore_method(method_dir: Path) -> dict | None:
    preds_path = method_dir / "preds.jsonl"
    if not preds_path.exists():
        return None
    rows = [json.loads(l) for l in preds_path.open()]
    new_rows = []
    by_k_strict = defaultdict(list)
    by_k_recall = defaultdict(list)
    n_flipped = 0
    for r in rows:
        old_strict, old_recall = r.get("strict"), r.get("recall")
        new_strict, new_recall = score_kha(r["pred"], r["gold_keys"])
        if (old_strict is not None and old_strict != new_strict) \
           or (old_recall is not None and abs((old_recall or 0) - new_recall) > 1e-6):
            n_flipped += 1
        r["strict"] = new_strict
        r["recall"] = new_recall
        new_rows.append(r)
        by_k_strict[r["k"]].append(new_strict)
        by_k_recall[r["k"]].append(new_recall)

    # Rewrite preds.jsonl with updated strict/recall
    with preds_path.open("w") as f:
        for r in new_rows:
            f.write(json.dumps(r) + "\n")

    # Rewrite summary.json
    summary_path = method_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {"method": method_dir.name, "n_total": len(new_rows)}
    summary["by_k"] = {
        str(k): {
            "n": len(by_k_strict[k]),
            "strict_pct": round(100 * sum(by_k_strict[k]) / max(1, len(by_k_strict[k])), 2),
            "recall_pct": round(100 * sum(by_k_recall[k]) / max(1, len(by_k_recall[k])), 2),
            "strict_CI95_pct": [round(100 * x, 2) for x in bootstrap_ci95(by_k_strict[k])[1:]],
        } for k in sorted(by_k_strict.keys())
    }
    summary["rescored_with_fixed_regex"] = True
    summary["rescored_flipped_count"] = n_flipped
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, default="experiments/khop_bench", nargs="?")
    ap.add_argument("--methods", nargs="+",
                    default=["full", "kivi", "path_d", "streamingllm", "h2o"])
    args = ap.parse_args()

    print(f"Re-scoring under {args.root} with fixed regex")
    print(f"  pattern: {_DIGIT_RE.pattern}")
    print()
    for m in args.methods:
        d = args.root / m
        if not d.exists():
            print(f"  {m}: <no dir>")
            continue
        s = rescore_method(d)
        if s is None:
            print(f"  {m}: <no preds.jsonl>")
            continue
        print(f"  {m}: flipped {s.get('rescored_flipped_count', 0)}/{s['n_total']} preds")
        for k, v in sorted(s["by_k"].items(), key=lambda x: int(x[0])):
            print(f"     k={k}: strict={v['strict_pct']:5.2f}% "
                  f"CI95={v['strict_CI95_pct']}  recall={v['recall_pct']:5.2f}% (n={v['n']})")


if __name__ == "__main__":
    main()
