"""Generate synthetic attention traces for CPU smoke testing.

The output is byte-compatible with :mod:`scripts.extract_attention_trace` so
that the rest of the analysis pipeline (``compute_hotness``, ``findings``,
``train_classifier``) can be exercised without GPU access. Each generated
trace mimics the empirically observed sink + recency + retrieval-needle
pattern: a small number of "needle" positions get a sharp boost, the first
``sink_tokens`` positions are always hot, and a recency window at the tail
contributes a smoothly decaying mass.

This file is **not** a substitute for real GPU traces. Numbers derived from it
should never enter the paper. It exists to keep CI fast and to let the user
debug downstream tooling without waiting on an A100.

Usage::

    python scripts/synthesize_traces.py \\
        --out experiments/traces/synthetic/qwen2-5-7b \\
        --tasks narrativeqa qasper hotpotqa gov_report passage_retrieval_en \\
        --context-len 1024 --n-steps 64 --n-layers 8 --num-heads 8 \\
        --seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


_TASK_PROFILES = {
    # Sharp retrieval tasks → very concentrated, narrow needle.
    "narrativeqa":         {"needles": 3, "needle_boost": 25.0, "noise": 0.04},
    "qasper":              {"needles": 5, "needle_boost": 18.0, "noise": 0.05},
    "multifieldqa_en":     {"needles": 4, "needle_boost": 20.0, "noise": 0.05},
    "hotpotqa":            {"needles": 6, "needle_boost": 15.0, "noise": 0.06},
    "2wikimqa":            {"needles": 6, "needle_boost": 15.0, "noise": 0.06},
    "musique":             {"needles": 8, "needle_boost": 12.0, "noise": 0.07},
    "passage_retrieval_en":{"needles": 2, "needle_boost": 30.0, "noise": 0.03},
    "niah_single_1":       {"needles": 1, "needle_boost": 40.0, "noise": 0.02},
    "niah_multikey_1":     {"needles": 4, "needle_boost": 18.0, "noise": 0.05},
    # Diffuse tasks → broader hot region, more noise.
    "gov_report":          {"needles": 0, "needle_boost": 0.0,  "noise": 0.10},
    "trec":                {"needles": 0, "needle_boost": 0.0,  "noise": 0.12},
    "triviaqa":            {"needles": 4, "needle_boost": 10.0, "noise": 0.08},
    "repobench-p":         {"needles": 6, "needle_boost": 12.0, "noise": 0.07},
    # Default profile.
    "default":             {"needles": 3, "needle_boost": 18.0, "noise": 0.06},
}


def _profile(task: str) -> dict:
    return _TASK_PROFILES.get(task, _TASK_PROFILES["default"])


def synthesize_trace(
    *,
    task: str,
    context_len: int = 1024,
    n_steps: int = 64,
    n_layers: int = 8,
    num_heads: int = 8,
    sink_tokens: int = 4,
    recent_tokens: int = 32,
    topk: int = 256,
    model_name: str = "synthetic",
    seed: int = 0,
) -> dict:
    """Build a synthetic attention trace mimicking real LLM behavior."""
    g = torch.Generator().manual_seed(seed + abs(hash(task)) % (2**31))
    profile = _profile(task)

    K_full = context_len + n_steps
    # 1) Sink mass — first `sink_tokens` positions always strongly hot.
    sink_mask = torch.zeros(K_full)
    sink_mask[:sink_tokens] = 8.0
    # 2) Needle mass — pick a few stable retrieval positions.
    needle_idx = torch.randperm(context_len, generator=g)[: profile["needles"]]
    needle_mask = torch.zeros(K_full)
    needle_mask[needle_idx] = profile["needle_boost"]

    trace: dict = {
        "context_len": context_len,
        "n_steps": n_steps,
        "n_layers": n_layers,
        "topk": topk,
        "hot_indices": [],
        "hot_values": [],
        "head_mean_full": [],
        "model_name": model_name,
        "task": task,
        "synthetic": True,
        "synthesis_profile": profile,
    }

    for step_idx in range(n_steps):
        # Recency window covers the last `recent_tokens` positions of the *current* sequence.
        cur_K = context_len + step_idx + 1
        positions = torch.arange(cur_K, dtype=torch.float32)
        recency = torch.exp(-(positions[-1] - positions) / max(recent_tokens, 1)) * 4.0

        step_idxs, step_vals = [], []
        for _ in range(n_layers):
            base = sink_mask[:cur_K] + needle_mask[:cur_K] + recency
            attn = base + profile["noise"] * torch.randn(cur_K, generator=g).abs()
            attn = attn.clamp(min=1e-6)
            attn = attn / attn.sum()

            k = min(topk, cur_K)
            v, i = torch.topk(attn, k=k)
            step_idxs.append(i.cpu())
            step_vals.append(v.float().cpu())
            if step_idx == 0:
                # Save a low-resolution full attention matrix for the temporal heatmap.
                head_mean = attn.unsqueeze(0).repeat(1, 1).to(torch.float16)
                trace["head_mean_full"].append(head_mean.cpu())
        trace["hot_indices"].append(step_idxs)
        trace["hot_values"].append(step_vals)

    return trace


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--out", required=True,
                    help="Output directory; one .pt file per task is written here.")
    ap.add_argument("--tasks", nargs="+",
                    default=["narrativeqa", "qasper", "hotpotqa",
                             "gov_report", "passage_retrieval_en"])
    ap.add_argument("--context-len", type=int, default=1024)
    ap.add_argument("--n-steps", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--model-name", default="synthetic")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        trace = synthesize_trace(
            task=task,
            context_len=args.context_len,
            n_steps=args.n_steps,
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            topk=args.topk,
            model_name=args.model_name,
            seed=args.seed,
        )
        path = out_dir / f"{task}.pt"
        torch.save(trace, path)
        print(f"[synth] {path} (context_len={trace['context_len']}, "
              f"n_steps={trace['n_steps']}, layers={trace['n_layers']})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
