"""RULER runner: 10 subtasks × 4 lengths (8K/32K/64K/128K).

Implements §2.3 of ``EMNLP2026_PivotPlan.md``. Mistral-7B truncates to 8K/32K
because of its native window. Usage::

    python scripts/run_ruler.py \\
        --config configs/models/qwen2-5-7b.yaml \\
        --method halo --memory-ratio 4 \\
        --lengths 8192 32768 65536 131072 \\
        --output experiments/runs/qwen2-5-7b/ruler/halo_4x
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from halo.utils import get_logger, load_yaml, seed_everything, write_manifest


_DEFAULT_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2",
    "niah_multiquery", "niah_multivalue",
    "vt", "qa_1", "qa_2",
]


def main() -> int:
    log = get_logger("halo.ruler")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True,
                    choices=["full", "h2o", "streamingllm", "snapkv", "kivi",
                             "halo", "halo_hybrid", "quest", "quest_path_d",
                             "path_d"])
    ap.add_argument("--memory-ratio", type=int, default=4)
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[8192, 32768, 65536, 131072])
    ap.add_argument("--tasks", nargs="+", default=_DEFAULT_TASKS)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on examples per (task,length) cell.")
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

    grid: dict[str, dict[int, float]] = {t: {} for t in args.tasks}
    for task in args.tasks:
        for length in args.lengths:
            log.info("task=%s length=%d ...", task, length)
            if run_one is None:
                grid[task][length] = float("nan")
                continue
            score = run_one(model_cfg=model_cfg, task=f"ruler:{task}",
                            method=args.method, memory_ratio=args.memory_ratio,
                            extra={"context_length": length},
                            output_dir=out_dir / f"{task}_{length}",
                            limit=args.limit)
            grid[task][length] = score
            log.info("  → %.4f", score)

    write_manifest(
        out_dir,
        suite="ruler",
        method=args.method,
        memory_ratio=args.memory_ratio,
        model=model_cfg["name_or_path"],
        lengths=args.lengths,
        scores=grid,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
