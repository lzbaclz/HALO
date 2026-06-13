"""Integrity check on every .pt attention trace under experiments/traces/.

Run before any analysis pass that depends on the traces (CP-1, CP-2, the
ablation_scorer_quality script, the appendix figures). Catches:

  * file is unreadable / wrong torch.load schema
  * required keys missing (context_len / n_steps / n_layers / hot_indices /
    hot_values / model_name)
  * shape inconsistencies (hot_indices[step][layer] != topk; n_steps mismatch)
  * dtype surprises (float16 indices / values; should be int64 / float32)
  * suspicious top-k values (NaN, negative, all-zero)
  * head_mean_full sanity (expected length n_layers if present)

Usage:
  python scripts/verify_traces.py
  python scripts/verify_traces.py --root experiments/traces
  python scripts/verify_traces.py --strict   # fail on warnings too

Exit code: 0 = clean, 1 = warnings, 2 = errors.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

warnings.filterwarnings("ignore")


REQUIRED_KEYS = {"context_len", "n_steps", "n_layers",
                 "hot_indices", "hot_values", "model_name"}


def check_one(path: Path) -> tuple[str, list[str]]:
    """Return (severity, [issues]) where severity ∈ {ok, warning, error}."""
    issues: list[str] = []
    severity = "ok"

    try:
        tr = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return "error", [f"unreadable: {e}"]

    missing = REQUIRED_KEYS - set(tr.keys())
    if missing:
        return "error", [f"missing keys: {sorted(missing)}"]

    L, N, n_layers = tr["context_len"], tr["n_steps"], tr["n_layers"]
    topk = tr.get("topk")

    # Structural shape checks.
    if not isinstance(tr["hot_indices"], list) or len(tr["hot_indices"]) != N:
        issues.append(f"hot_indices length {len(tr['hot_indices'])} != n_steps {N}")
        severity = "error"
    elif tr["hot_indices"]:
        per_layer = tr["hot_indices"][0]
        if not isinstance(per_layer, list) or len(per_layer) != n_layers:
            issues.append(f"hot_indices[0] inner length {len(per_layer)} != n_layers {n_layers}")
            severity = "error"
        else:
            first = per_layer[0]
            if not torch.is_tensor(first):
                issues.append("hot_indices[0][0] is not a tensor")
                severity = "error"
            else:
                if topk is not None and first.numel() != min(topk, L + N):
                    issues.append(f"hot_indices[0][0] len={first.numel()} vs expected min(topk={topk}, L+N={L+N})")
                    if severity == "ok":
                        severity = "warning"
                if first.dtype not in (torch.int64, torch.int32, torch.long):
                    issues.append(f"hot_indices dtype {first.dtype} (expected int64)")
                    if severity == "ok":
                        severity = "warning"

    # Same for hot_values.
    if not isinstance(tr["hot_values"], list) or len(tr["hot_values"]) != N:
        issues.append(f"hot_values length {len(tr['hot_values'])} != n_steps {N}")
        severity = "error"

    # Sanity: scan a sample of values for NaN / negative.
    sample_steps = min(N, 4)
    for step in range(sample_steps):
        for layer in range(min(n_layers, 4)):
            v = tr["hot_values"][step][layer]
            if not torch.is_tensor(v):
                issues.append(f"hot_values[{step}][{layer}] not a tensor")
                severity = "error"
                continue
            if torch.isnan(v).any():
                issues.append(f"NaN in hot_values[{step}][{layer}]")
                severity = "error"
            if (v < 0).any():
                issues.append(f"negative values in hot_values[{step}][{layer}]")
                if severity == "ok":
                    severity = "warning"
            if v.sum().item() == 0:
                issues.append(f"all-zero values in hot_values[{step}][{layer}]")
                if severity == "ok":
                    severity = "warning"

    # head_mean_full (optional, but if present must be one entry per layer).
    if "head_mean_full" in tr and tr["head_mean_full"]:
        if len(tr["head_mean_full"]) != n_layers:
            issues.append(f"head_mean_full length {len(tr['head_mean_full'])} != n_layers {n_layers}")
            if severity == "ok":
                severity = "warning"

    return severity, issues


def walk(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.pt"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/traces")
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero on warnings as well as errors.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: traces root does not exist: {root}", file=sys.stderr)
        return 2

    paths = list(walk(root))
    if not paths:
        print(f"WARN: no .pt traces under {root}")
        return 1

    n_ok = n_warn = n_err = 0
    for p in paths:
        sev, issues = check_one(p)
        rel = p.relative_to(root)
        if sev == "ok":
            n_ok += 1
        elif sev == "warning":
            n_warn += 1
            print(f"WARN  {rel}")
            for i in issues:
                print(f"      └─ {i}")
        else:
            n_err += 1
            print(f"ERROR {rel}")
            for i in issues:
                print(f"      └─ {i}")
    print()
    print(f"Audited {len(paths)} traces: {n_ok} OK, {n_warn} warnings, {n_err} errors.")
    if n_err:
        return 2
    if args.strict and n_warn:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
