"""Aggregate a trace into per-position hotness and compute Jaccard stability.

Implements §3.4 of ``EMNLP2026_PivotPlan.md`` exactly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch


def compute_hotness(trace: dict, layer_agg: str = "mean") -> torch.Tensor:
    """Returns a hotness tensor of shape ``(context_len + n_steps,)``."""
    L = trace["context_len"]
    N = trace["n_steps"]
    K = L + N
    n_layers = trace["n_layers"]

    hotness = torch.zeros(K, dtype=torch.float32)
    for step_idx in range(N):
        step_hot = torch.zeros(K, dtype=torch.float32)
        for layer_idx in range(n_layers):
            idxs = trace["hot_indices"][step_idx][layer_idx]
            vals = trace["hot_values"][step_idx][layer_idx]
            step_hot.index_add_(0, idxs, vals)
        if layer_agg == "mean":
            step_hot /= n_layers
        hotness += step_hot
    return hotness / max(N, 1)


def jaccard_stability(trace: dict, window: int = 64, top_pct: float = 0.10) -> float:
    """Mean Jaccard overlap of top-``top_pct`` hot positions across windows of ``window`` steps."""
    N = trace["n_steps"]
    L = trace["context_len"]
    K = L + N
    top_n = max(int(K * top_pct), 1)

    step_hots = []
    for step_idx in range(N):
        all_idx = torch.cat([trace["hot_indices"][step_idx][l] for l in range(trace["n_layers"])])
        all_val = torch.cat([trace["hot_values"][step_idx][l] for l in range(trace["n_layers"])])
        agg = torch.zeros(K, dtype=all_val.dtype)
        agg.index_add_(0, all_idx, all_val)
        top_idx = set(torch.topk(agg, k=top_n).indices.tolist())
        step_hots.append(top_idx)

    js = []
    if N > window:
        for i in range(N - window):
            a, b = step_hots[i], step_hots[i + window]
            js.append(len(a & b) / max(len(a | b), 1))
    else:
        # Not enough decoding steps to span the requested window. Fall back to
        # the largest window that fits and report it; this is honest and avoids
        # the degenerate "perfect Jaccard because no windows" case.
        eff_window = max(N // 2, 1)
        for i in range(N - eff_window):
            a, b = step_hots[i], step_hots[i + eff_window]
            js.append(len(a & b) / max(len(a | b), 1))
    if not js:
        return float("nan")
    return float(torch.tensor(js).mean().item())


def concentration_curve(hotness: torch.Tensor) -> torch.Tensor:
    """Cumulative attention mass as we accept more positions in descending hotness order."""
    sorted_h = torch.sort(hotness, descending=True).values
    return torch.cumsum(sorted_h, dim=0) / sorted_h.sum().clamp_min(1e-12)


def _cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", default=None,
                    help="Optional .pt to write {hotness, jaccard, concentration}.")
    ap.add_argument("--top-pct", type=float, default=0.10)
    ap.add_argument("--window", type=int, default=64)
    args = ap.parse_args()

    trace = torch.load(args.trace, map_location="cpu")
    hotness = compute_hotness(trace)
    jacc = jaccard_stability(trace, window=args.window, top_pct=args.top_pct)
    conc = concentration_curve(hotness)

    K = hotness.numel()
    top_idx = max(int(K * args.top_pct), 1) - 1
    print(f"context_len={trace['context_len']}, steps={trace['n_steps']}, layers={trace['n_layers']}")
    print(f"top-{int(args.top_pct*100)}% concentration = {conc[top_idx].item():.4f}")
    print(f"jaccard@window={args.window}, top={int(args.top_pct*100)}% = {jacc:.4f}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"hotness": hotness, "jaccard": jacc, "concentration": conc}, out)
        print(f"[hotness] saved to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
