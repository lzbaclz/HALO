"""∞-Bench runner: En.QA + En.MC, supports the 1M-token claim.

Per §2.3 of ``EMNLP2026_PivotPlan.md``: only on Llama-3-8B and Qwen2.5-7B,
2 subtasks each. The 1M-token configuration is tuned for a single 80GB A100.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from halo.utils import get_logger, load_yaml, seed_everything, write_manifest


def main() -> int:
    log = get_logger("halo.infinitebench")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True,
                    choices=["full", "h2o", "streamingllm", "snapkv", "kivi", "halo", "quest", "quest_path_d", "path_d"])
    ap.add_argument("--tasks", nargs="+", default=["en_qa", "en_mc"])
    ap.add_argument("--context-length", type=int, default=1_000_000)
    ap.add_argument("--memory-ratio", type=int, default=4)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on examples per task (for smoke tests / time-budgeted runs).")
    args = ap.parse_args()

    seed_everything(args.seed)
    model_cfg = load_yaml(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from baselines.runner import evaluate as run_one
    except ImportError:  # pragma: no cover
        log.warning("baselines.runner is not installed; emitting a dry-run manifest only.")
        run_one = None

    scores: dict[str, float] = {}
    for task in args.tasks:
        log.info("task=infinite:%s @ %d tokens", task, args.context_length)
        if run_one is None:
            scores[task] = float("nan")
            continue
        scores[task] = run_one(
            model_cfg=model_cfg,
            task=f"infinitebench:{task}",
            method=args.method,
            memory_ratio=args.memory_ratio,
            extra={"context_length": args.context_length},
            output_dir=out_dir / task,
            limit=args.limit,
        )

    write_manifest(
        out_dir,
        suite="infinitebench",
        method=args.method,
        context_length=args.context_length,
        model=model_cfg["name_or_path"],
        scores=scores,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
