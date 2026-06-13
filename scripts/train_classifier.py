"""Closed-form fit / eval of the Finding-3 (α, β, γ) classifier.

Thin wrapper around :func:`halo.classifier.fit` / :func:`halo.classifier.evaluate`
that adds nicer logging and writes a ``manifest.json`` for the experiment manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from halo.classifier import evaluate, fit
from halo.utils import get_logger, write_manifest


def main() -> int:
    log = get_logger("halo.train_classifier")

    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True,
                    help="Directory containing .pt traces from extract_attention_trace.py")
    ap.add_argument("--out", required=True, help="Path to write the .npz classifier.")
    ap.add_argument("--eval-traces", default=None,
                    help="If supplied, evaluate cross-model AUC on this directory.")
    ap.add_argument("--hot-threshold", type=float, default=0.10)
    args = ap.parse_args()

    log.info("fitting classifier on %s ...", args.traces)
    summary = fit(args.traces, args.out, hot_threshold=args.hot_threshold)
    log.info("alpha=%.4f beta=%.4f gamma=%.4f bias=%.4f train-AUC=%.4f",
             summary["alpha"], summary["beta"], summary["gamma"],
             summary["bias"], summary["auc"])

    if args.eval_traces:
        log.info("evaluating on %s ...", args.eval_traces)
        eval_summary = evaluate(args.eval_traces, args.out, hot_threshold=args.hot_threshold)
        log.info("eval-AUC=%.4f", eval_summary["auc"])
        summary["eval_auc"] = eval_summary["auc"]

    out_dir = Path(args.out).parent
    write_manifest(out_dir, classifier=str(args.out), **summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
