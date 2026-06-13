#!/usr/bin/env python3
"""Plan B — NLI semantic-identity bridge between Path D and Full attention.

The Prop. 4.5 result gives bit-equivalence per-step on fixed KV state, but
free-running fp32 trajectories may diverge (Prop. 4.5 iii). Bit-divergence
does not necessarily imply semantic divergence: two slightly different
token strings can still mean the same thing. Plan B closes the gap by
adding a *semantic-identity* tier between (ii) bit-equivalent-per-step and
(iii) downstream non-inferiority:

  P(Path D pred ⟷ Full pred under NLI)

NLI procedure (matches the convention from FEVER / GenIE evaluation):
  - score(premise, hypothesis) → P(entailment), P(neutral), P(contradiction)
  - Bidirectional check: classify the pair as
      - "equivalent" if P(entail | A→B) ≥ τ AND P(entail | B→A) ≥ τ
      - "compatible" if neither direction's P(contradict) ≥ τ
      - "contradictory" if either P(contradict) ≥ τ

We use a standard MNLI/SNLI-fine-tuned classifier — defaults to
``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`` (3-label entail/neutral/
contradict, ~184 MB). Override with --nli-model for a larger checkpoint.

Inputs:
  Two preds.jsonl files (e.g. Path D vs Full from discourse_bench or NIAH).
  Each line: {"index", "pred", "gold", ...}

Output:
  per-pair NLI scores + summary aggregating equivalence/compatibility
  rates with bootstrap CI95.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def bootstrap_ci95(values, n_iters=10000, seed=0):
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return sum(values) / n, means[int(0.025 * n_iters)], means[int(0.975 * n_iters) - 1]


def nli_score_pair(tok, model, premise: str, hypothesis: str) -> dict:
    """Return {entailment, neutral, contradiction} probability dict for the
    ordered pair (premise, hypothesis)."""
    if not premise.strip() or not hypothesis.strip():
        # empty preds: degenerate to "neutral" (no signal)
        return {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}
    inputs = tok(premise, hypothesis, return_tensors="pt", truncation=True,
                 max_length=512).to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).tolist()
    # MoritzLaurer DeBERTa MNLI labels: 0=entail, 1=neutral, 2=contradict
    # (per the model's config.id2label; we read it dynamically below)
    label2id = {v.lower(): k for k, v in model.config.id2label.items()}
    def pid(name):
        for k, v in label2id.items():
            if name in k:
                return v
        return None
    e = pid("entail"); n = pid("neutral"); c = pid("contradict")
    return {
        "entailment": probs[e] if e is not None else 0.0,
        "neutral": probs[n] if n is not None else 0.0,
        "contradiction": probs[c] if c is not None else 0.0,
    }


def classify_pair(forward: dict, backward: dict, tau: float = 0.5) -> str:
    """Bidirectional classification: equivalent / compatible / contradictory."""
    if (forward["entailment"] >= tau and backward["entailment"] >= tau):
        return "equivalent"
    if (forward["contradiction"] >= tau or backward["contradiction"] >= tau):
        return "contradictory"
    return "compatible"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-a", required=True, type=Path,
                    help="Path D preds.jsonl (or any method A)")
    ap.add_argument("--preds-b", required=True, type=Path,
                    help="Full attention preds.jsonl (or any method B)")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--nli-model", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                    help="HF checkpoint of an MNLI/SNLI-fine-tuned classifier.")
    ap.add_argument("--tau", type=float, default=0.5,
                    help="Probability threshold for entail/contradict decisions.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="0 → score all; otherwise truncate.")
    args = ap.parse_args()

    print(f"[nli-bridge] loading NLI classifier: {args.nli_model}")
    tok = AutoTokenizer.from_pretrained(args.nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.nli_model, torch_dtype=torch.float16 if "cuda" in args.device else torch.float32
    ).to(args.device)
    model.eval()
    print(f"  labels: {model.config.id2label}")

    rows_a = {r["index"]: r for r in (json.loads(l) for l in open(args.preds_a))}
    rows_b = {r["index"]: r for r in (json.loads(l) for l in open(args.preds_b))}
    common = sorted(set(rows_a) & set(rows_b))
    if args.max_pairs > 0:
        common = common[: args.max_pairs]
    print(f"[nli-bridge] scoring {len(common)} matched pairs "
          f"(A={args.preds_a.name}, B={args.preds_b.name})")

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_f = (out_dir / "pairs.jsonl").open("w")

    classifications: list[str] = []
    by_subtask: dict[str, list[str]] = {}
    by_band: dict[str, list[str]] = {}
    n_equiv = n_compat = n_contra = 0
    for idx in common:
        ra = rows_a[idx]; rb = rows_b[idx]
        a = (ra.get("pred") or "").strip()
        b = (rb.get("pred") or "").strip()
        # Skip OOM tokens
        if a == "[OOM]" or b == "[OOM]":
            continue
        forward = nli_score_pair(tok, model, premise=a, hypothesis=b)
        backward = nli_score_pair(tok, model, premise=b, hypothesis=a)
        cls = classify_pair(forward, backward, tau=args.tau)
        classifications.append(cls)
        subtask = ra.get("subtask") or rb.get("subtask")
        band = ra.get("band") or rb.get("band")
        if subtask:
            by_subtask.setdefault(subtask, []).append(cls)
        if band:
            by_band.setdefault(band, []).append(cls)
        if cls == "equivalent": n_equiv += 1
        elif cls == "compatible": n_compat += 1
        else: n_contra += 1
        pair_f.write(json.dumps({
            "index": idx, "subtask": subtask, "band": band,
            "pred_a": a, "pred_b": b,
            "forward": forward, "backward": backward,
            "class": cls,
        }) + "\n")
        if (len(classifications)) % 20 == 0:
            print(f"  {len(classifications)} scored — equiv={n_equiv} "
                  f"compat={n_compat} contra={n_contra}")
    pair_f.close()

    n = max(1, len(classifications))
    eq_rate = n_equiv / n
    cp_rate = n_compat / n
    ct_rate = n_contra / n
    eq_bin = [1.0 if c == "equivalent" else 0.0 for c in classifications]
    cp_bin = [1.0 if c in ("equivalent", "compatible") else 0.0 for c in classifications]
    eq_mean, eq_lo, eq_hi = bootstrap_ci95(eq_bin)
    cp_mean, cp_lo, cp_hi = bootstrap_ci95(cp_bin)

    def per_group(d):
        return {
            k: {
                "n": len(v),
                "equivalent_pct": round(100 * sum(1 for x in v if x == "equivalent") / max(1, len(v)), 2),
                "compatible_pct": round(100 * sum(1 for x in v if x == "compatible") / max(1, len(v)), 2),
                "contradictory_pct": round(100 * sum(1 for x in v if x == "contradictory") / max(1, len(v)), 2),
            }
            for k, v in d.items()
        }

    summary = {
        "preds_a": str(args.preds_a),
        "preds_b": str(args.preds_b),
        "nli_model": args.nli_model,
        "tau": args.tau,
        "n_pairs": len(classifications),
        "equivalent_pct": round(100 * eq_rate, 2),
        "compatible_pct": round(100 * cp_rate, 2),
        "contradictory_pct": round(100 * ct_rate, 2),
        "equivalent_CI95_pct": [round(100 * eq_lo, 2), round(100 * eq_hi, 2)],
        "equivalent_or_compatible_pct": round(100 * cp_mean, 2),
        "equivalent_or_compatible_CI95_pct": [round(100 * cp_lo, 2), round(100 * cp_hi, 2)],
        "by_subtask": per_group(by_subtask),
        "by_band": per_group(by_band),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print(f"[nli-bridge] DONE")
    print(f"  equivalent:    {summary['equivalent_pct']:.2f}%  CI95={summary['equivalent_CI95_pct']}")
    print(f"  +compatible:   {summary['equivalent_or_compatible_pct']:.2f}%  "
          f"CI95={summary['equivalent_or_compatible_CI95_pct']}")
    print(f"  contradictory: {summary['contradictory_pct']:.2f}%")
    print(f"  out: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
