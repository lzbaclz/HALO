"""Multi-seed bootstrap-CI ∞-Bench runner.

Solves reviewer 3's "n=10 single seed F1=11 is statistical noise" concern.
Runs each method on the same N examples (we pre-sample a seed-controlled
subset), records per-example scores, then bootstrap-resamples scores B times
to compute mean and 95% CI.

Resumable: every per-example score is written to `<output>/preds.jsonl`
incrementally; re-runs skip examples already scored. Crash-safe.

Usage on a free 80 GB A100 once SEER frees the GPU:

    python scripts/run_infinitebench_bootstrap.py \
        --config configs/models/qwen2.5-7b.yaml \
        --methods full path_d \
        --tasks en_qa \
        --context-length 65000 \
        --n-examples 30 \
        --bootstrap-iters 10000 \
        --seeds 0 1 2 \
        --output experiments/infinitebench_bootstrap

Outputs:
    <output>/<method>/seed_<s>/<task>/preds.jsonl
    <output>/summary.json   — per-method bootstrap mean + 95% CI
    <output>/summary.tex    — paper-ready row (matched-sample mean, CI95)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from halo.utils import get_logger, load_yaml, seed_everything


def _sample_indices(*, total: int, n: int, seed: int) -> list[int]:
    """Pick n indices from [0, total) deterministically from seed."""
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), min(n, total)))


def _load_per_example_scores(preds_path: Path) -> list[float]:
    """Read scores from a previously-completed preds.jsonl."""
    if not preds_path.exists():
        return []
    out = []
    with preds_path.open() as f:
        for line in f:
            try:
                out.append(float(json.loads(line)["score"]))
            except Exception:
                continue
    return out


def _bootstrap(scores: list[float], *, n_iters: int, seed: int,
               ci_pct: float = 95.0) -> dict:
    """Standard percentile bootstrap on a list of per-example scores."""
    if not scores:
        return {"mean": float("nan"), "n": 0,
                "ci_low": float("nan"), "ci_high": float("nan")}
    import statistics
    rng = random.Random(seed)
    boot_means = []
    n = len(scores)
    for _ in range(n_iters):
        sample = [scores[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(n_iters * (50 - ci_pct / 2) / 100)]
    hi = boot_means[int(n_iters * (50 + ci_pct / 2) / 100) - 1]
    return {
        "mean": statistics.mean(scores),
        "boot_mean": statistics.mean(boot_means),
        "n": n,
        "ci_low": lo,
        "ci_high": hi,
        "ci_pct": ci_pct,
        "stdev": statistics.stdev(scores) if n > 1 else 0.0,
    }


def _run_one(*, method: str, model_cfg: dict, task: str, context_length: int,
             memory_ratio: int, output_dir: Path, indices: list[int]) -> list[float]:
    """Run one (method, seed, task) cell. Resumable from preds.jsonl."""
    preds = output_dir / "preds.jsonl"
    cached = _load_per_example_scores(preds)
    if len(cached) >= len(indices):
        return cached[: len(indices)]

    # The runner streams indices in order; resuming means we already have the
    # first len(cached) entries. We delete and re-run if cache length doesn't
    # match (e.g. partial run with different limit) — easier than diffing.
    if cached:
        # Keep the partial preds.jsonl; baselines.runner.evaluate will append.
        # But to keep alignment with the indices list we re-run from scratch.
        preds.unlink()
        cached = []

    from baselines.runner import evaluate as run_one

    # Sub-sample by writing a filtered jsonl that the eval will read.
    # We patch via HALO_INFINITEBENCH_DIR pointing to a per-seed staging dir
    # containing the picked indices only.
    # The HF dataset uses long filenames (longbook_qa_eng.jsonl) while the
    # eval harness uses short subtask names (en_qa); accept either.
    _FILE_ALIASES = {
        "en_qa": ["longbook_qa_eng.jsonl", "en_qa.jsonl"],
        "en_mc": ["longbook_choice_eng.jsonl", "en_mc.jsonl"],
    }
    src_dir = Path(os.environ.get(
        "HALO_INFINITEBENCH_DIR",
        str(_REPO_ROOT / "experiments" / "infinitebench_raw"),
    ))
    src = None
    for fname in _FILE_ALIASES.get(task, [f"{task}.jsonl"]):
        cand = src_dir / fname
        if cand.exists():
            src = cand
            break
    if src is None:
        raise FileNotFoundError(
            f"InfiniteBench data missing under {src_dir}; tried "
            f"{_FILE_ALIASES.get(task, [f'{task}.jsonl'])}"
        )

    stage = output_dir / "_stage"
    stage.mkdir(parents=True, exist_ok=True)
    # We always write the short-name file so the downstream evaluator
    # (which uses task='en_qa' → 'en_qa.jsonl' or via aliases) finds it.
    with src.open("r", encoding="utf-8") as f, \
            (stage / f"{task}.jsonl").open("w", encoding="utf-8") as g:
        for i, line in enumerate(f):
            if i in indices:
                g.write(line)

    old_env = os.environ.get("HALO_INFINITEBENCH_DIR")
    os.environ["HALO_INFINITEBENCH_DIR"] = str(stage)
    try:
        score = run_one(
            model_cfg=model_cfg,
            task=f"infinitebench:{task}",
            method=method,
            memory_ratio=memory_ratio,
            extra={"context_length": context_length},
            output_dir=output_dir,
            limit=None,
        )
    finally:
        if old_env is None:
            os.environ.pop("HALO_INFINITEBENCH_DIR", None)
        else:
            os.environ["HALO_INFINITEBENCH_DIR"] = old_env

    return _load_per_example_scores(preds)


def main() -> int:
    log = get_logger("halo.infinitebench.bootstrap")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--methods", nargs="+", required=True,
                    help="Methods to compare, e.g. 'full path_d quest_path_d'")
    ap.add_argument("--tasks", nargs="+", default=["en_qa"])
    ap.add_argument("--context-length", type=int, default=65000)
    ap.add_argument("--memory-ratio", type=int, default=4)
    ap.add_argument("--n-examples", type=int, default=30,
                    help="Per-seed example count.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="Seeds for the example-sub-sampling (deterministic).")
    ap.add_argument("--bootstrap-iters", type=int, default=10000)
    ap.add_argument("--total-pool", type=int, default=200,
                    help="The full ∞-Bench task has ~200 examples; we draw "
                         "n-examples without replacement from [0, this).")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    model_cfg = load_yaml(args.config)

    seed_everything(0)  # global determinism for tokenizer / model

    all_per_seed_scores: dict[str, dict[int, dict[str, list[float]]]] = {}
    matched_pairs_pool: dict[tuple[str, str, int], list[tuple[float, float]]] = {}

    # Peak telemetry: torch.cuda.max_memory_allocated() per (method, seed, task).
    # Captured at the end of every _run_one; reset between cells so we record
    # per-cell peaks rather than monotonically-increasing global peaks.
    peak_gib_by_cell: dict[tuple[str, int, str], float] = {}

    try:
        import torch  # noqa: F401  (peak telemetry is best-effort)
        _have_torch = torch.cuda.is_available()
    except Exception:
        _have_torch = False
        torch = None  # type: ignore

    t0 = time.time()
    for seed in args.seeds:
        indices = _sample_indices(total=args.total_pool,
                                  n=args.n_examples, seed=seed)
        log.info("seed=%d indices=%s", seed, indices[:8])

        for task in args.tasks:
            for method in args.methods:
                cell_dir = out_root / method / f"seed_{seed}" / task
                cell_dir.mkdir(parents=True, exist_ok=True)
                log.info("running method=%s seed=%d task=%s n=%d",
                         method, seed, task, len(indices))
                if _have_torch:
                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass
                scores = _run_one(
                    method=method, model_cfg=model_cfg, task=task,
                    context_length=args.context_length,
                    memory_ratio=args.memory_ratio,
                    output_dir=cell_dir, indices=indices,
                )
                if _have_torch:
                    try:
                        peak_b = torch.cuda.max_memory_allocated()
                        peak_gib = peak_b / (1024 ** 3)
                        peak_gib_by_cell[(method, seed, task)] = peak_gib
                        log.info("  peak GiB (%s/seed=%d/%s) = %.3f",
                                 method, seed, task, peak_gib)
                    except Exception as e:
                        log.warning("peak telemetry failed: %s", e)
                all_per_seed_scores.setdefault(method, {})\
                    .setdefault(seed, {})[task] = scores
                log.info("  seed=%d method=%s task=%s n=%d mean=%.2f",
                         seed, method, task, len(scores),
                         100 * (sum(scores) / max(len(scores), 1)))

    # Aggregate.
    summary = {"args": vars(args), "wall_clock_s": time.time() - t0,
               "per_method": {}}
    for method, by_seed in all_per_seed_scores.items():
        summary["per_method"][method] = {"per_seed": {}, "pooled": {}}
        all_scores_pooled: dict[str, list[float]] = {t: [] for t in args.tasks}
        for seed, by_task in by_seed.items():
            summary["per_method"][method]["per_seed"][seed] = {}
            for task, sc in by_task.items():
                b = _bootstrap(sc, n_iters=args.bootstrap_iters, seed=seed)
                summary["per_method"][method]["per_seed"][seed][task] = b
                all_scores_pooled[task].extend(sc)
        for task, sc in all_scores_pooled.items():
            summary["per_method"][method]["pooled"][task] = \
                _bootstrap(sc, n_iters=args.bootstrap_iters, seed=0)

    # Embed peak telemetry per (method, seed, task) and per-method max.
    if peak_gib_by_cell:
        peak_section: dict[str, dict] = {}
        for (method, seed, task), pg in peak_gib_by_cell.items():
            peak_section.setdefault(method, {"per_cell": {}, "max_gib": 0.0})
            peak_section[method]["per_cell"][f"seed_{seed}/{task}"] = pg
            if pg > peak_section[method]["max_gib"]:
                peak_section[method]["max_gib"] = pg
        summary["peak_gib"] = peak_section

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))

    # LaTeX row (one row per method).
    _pooled_n = (
        sum(len(s) for m in all_per_seed_scores.values()
            for sb in m.values() for s in sb.values())
        // max(1, len(args.tasks))
    )
    tex_lines = [
        f"%% pooled n = {_pooled_n} examples "
        f"({len(args.seeds)} seeds x {args.n_examples}), "
        f"bootstrap CI95 ({args.bootstrap_iters} iters)",
    ]
    for method, body in summary["per_method"].items():
        for task in args.tasks:
            pooled = body["pooled"][task]
            tex_lines.append(
                f"  {method} & {task} & "
                f"{pooled['mean'] * 100:.2f} & "
                f"[{pooled['ci_low'] * 100:.2f}, "
                f"{pooled['ci_high'] * 100:.2f}] \\\\"
            )
    (out_root / "summary.tex").write_text("\n".join(tex_lines))
    log.info("wrote %s and %s", out_root / "summary.json", out_root / "summary.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
