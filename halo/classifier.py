"""Closed-form fit of the Finding-3 hot/cold classifier.

The default scorer in :mod:`halo.scorer` uses fixed weights ``(α, β, γ)``. This
module fits those weights on a small per-model trace dataset using logistic
regression (one model fits all six checkpoints in our study, see Table 3 in
the paper). Output is a tiny ``.npz`` of three floats.

Usage::

    python -m halo.classifier fit --traces traces/qwen2-5-7b/ --out classifier_llama3.npz
    python -m halo.classifier eval --traces traces/qwen2-5-7b/ --classifier classifier_llama3.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def fit(traces_dir: str, out_path: str, *, hot_threshold: float = 0.10) -> dict:
    """Fit a logistic regression with three features: ``attn``, ``recency``, ``sink``."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    feats, labels = _build_dataset(traces_dir, hot_threshold=hot_threshold)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(feats, labels)
    auc = roc_auc_score(labels, clf.predict_proba(feats)[:, 1])

    coefs = clf.coef_.ravel()
    bias = float(clf.intercept_[0])
    np.savez(out_path, alpha=float(coefs[0]), beta=float(coefs[1]),
             gamma=float(coefs[2]), bias=bias, auc=float(auc))
    return {"alpha": coefs[0], "beta": coefs[1], "gamma": coefs[2], "bias": bias, "auc": auc}


def evaluate(traces_dir: str, classifier_path: str, *, hot_threshold: float = 0.10) -> dict:
    from sklearn.metrics import roc_auc_score

    p = np.load(classifier_path)
    alpha, beta, gamma, bias = float(p["alpha"]), float(p["beta"]), float(p["gamma"]), float(p["bias"])

    feats, labels = _build_dataset(traces_dir, hot_threshold=hot_threshold)
    score = alpha * feats[:, 0] + beta * feats[:, 1] + gamma * feats[:, 2] + bias
    auc = roc_auc_score(labels, score)
    return {"auc": float(auc)}


def _build_dataset(traces_dir: str, *, hot_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Walk a traces folder and assemble per-position (feat, label) pairs.

    Each ``.pt`` trace is a dict with keys produced by
    ``scripts/extract_attention_trace.py`` — see that file for the schema.
    """
    import torch

    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for trace_path in sorted(Path(traces_dir).glob("*.pt")):
        trace = torch.load(trace_path, map_location="cpu")
        L = trace["context_len"]
        N = trace["n_steps"]
        K = L + N
        for step_idx in range(N):
            agg = torch.zeros(K)
            for layer_idx in range(trace["n_layers"]):
                idx = trace["hot_indices"][step_idx][layer_idx]
                val = trace["hot_values"][step_idx][layer_idx]
                agg.index_add_(0, idx, val)
            agg = agg / trace["n_layers"]

            attn = (agg / (agg.sum() + 1e-12)).numpy()
            positions = np.arange(K, dtype=np.float32)
            recency = np.exp(-(step_idx - positions).clip(min=0.0) / 64.0)
            recency = recency / (recency.sum() + 1e-12)
            sink = np.zeros(K, dtype=np.float32)
            sink[:4] = 1.0

            X = np.stack([attn, recency, sink], axis=1)
            top_k = max(int(K * hot_threshold), 1)
            label_idx = np.argsort(-attn)[:top_k]
            y = np.zeros(K, dtype=np.int64)
            y[label_idx] = 1

            feats.append(X)
            labels.append(y)

    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Fit / eval the HALO Finding-3 classifier.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit")
    f.add_argument("--traces", required=True)
    f.add_argument("--out", required=True)

    e = sub.add_parser("eval")
    e.add_argument("--traces", required=True)
    e.add_argument("--classifier", required=True)

    args = ap.parse_args()
    if args.cmd == "fit":
        print(fit(args.traces, args.out))
    else:
        print(evaluate(args.traces, args.classifier))


if __name__ == "__main__":  # pragma: no cover
    _cli()
