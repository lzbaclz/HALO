"""LongBench v1 runner under a unified memory-budget protocol.

Implements Protocol A from §2.5 of ``EMNLP2026_PivotPlan.md``: each method runs
at memory ratios {1×, 2×, 4×, 8×}. The 10 ✅ subtasks from §2.3 are the default.

Usage::

    python scripts/run_longbench.py \\
        --config configs/models/qwen2-5-7b.yaml \\
        --tasks  configs/tasks/longbench.yaml \\
        --method halo \\
        --memory-ratio 4 \\
        --output experiments/runs/qwen2-5-7b/longbench/halo_4x
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from halo.utils import get_logger, load_yaml, seed_everything, write_manifest


_REQUIRED_TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "musique",
    "gov_report", "trec", "triviaqa",
    "passage_retrieval_en", "repobench-p",
]


def main() -> int:
    log = get_logger("halo.longbench")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Model YAML config.")
    ap.add_argument("--tasks", required=True, help="Task list YAML.")
    ap.add_argument("--method", required=True,
                    choices=["full", "h2o", "streamingllm", "snapkv", "kivi", "halo", "path_d", "quest", "quest_path_d"])
    ap.add_argument("--memory-ratio", type=int, default=4,
                    help="1=full, 2=50%, 4=25%, 8=12.5%")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on examples per task (for smoke tests).")
    args = ap.parse_args()

    seed_everything(args.seed)
    model_cfg = load_yaml(args.config)
    task_cfg = load_yaml(args.tasks)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("model = %s", model_cfg["name_or_path"])
    log.info("method = %s, memory ratio = %dx", args.method, args.memory_ratio)
    log.info("tasks = %s", task_cfg["tasks"])

    # ----- LongBench harness wiring -----
    # The KVPress-compatible harness lives in baselines/runner.py and dispatches
    # by `method`. We keep this file as the orchestration / manifest writer so
    # that runs are reproducible without depending on a particular harness build.
    try:
        from baselines.runner import evaluate as run_one
    except ImportError:  # pragma: no cover - harness not installed
        log.warning("baselines.runner is not installed; emitting a dry-run manifest only.")
        run_one = None

    results: dict[str, float] = {}
    if run_one is not None:
        for task in task_cfg["tasks"]:
            log.info("running task %s ...", task)
            score = run_one(model_cfg=model_cfg, task=task, method=args.method,
                            memory_ratio=args.memory_ratio, limit=args.limit,
                            output_dir=out_dir / task)
            results[task] = score
            log.info("  → %.4f", score)
    else:
        # Dry-run placeholders so the manifest schema is exercised in CI.
        results = {task: float("nan") for task in task_cfg["tasks"]}

    manifest_path = write_manifest(
        out_dir,
        suite="longbench-v1",
        method=args.method,
        memory_ratio=args.memory_ratio,
        model=model_cfg["name_or_path"],
        scores=results,
        seed=args.seed,
    )
    log.info("manifest → %s", manifest_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
