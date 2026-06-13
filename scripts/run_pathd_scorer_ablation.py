"""FU_W12: Path D × {Quest, HALOPress, FIFO, uniform-random} scorer ablation.

Drives :mod:`baselines.scorer_ablations` on RULER NIAH adversarial 32K.
Every cell uses the same checkpoint, prompts, decoding, and seed; only
the placement scorer changes. If selection-rule invariance holds, the
four cells must agree within Wilson CI95 on every subtask.

Usage::

    python scripts/run_pathd_scorer_ablation.py \\
        --scorer uniform_random \\
        --subtask niah_multikey_1 \\
        --context-length 32768 --n-examples 30 \\
        --output experiments/fu_w12/uniform_random_mk1_32k

The runner is byte-for-byte aligned with :mod:`scripts/run_pathd_ruler`
for the Quest case (delegates to :mod:`baselines.quest_path_d`); the
other three scorers go through
:func:`baselines.scorer_ablations.wrap_with_path_d_ablation_scorer`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from baselines.scorer_ablations import (  # noqa: E402
    AblationScorerConfig,
    wrap_with_path_d_ablation_scorer,
)


def _ruler_score(pred: str, gold) -> float:
    if isinstance(gold, str):
        golds = [gold]
    else:
        golds = list(gold)
    for g in golds:
        if str(g) in pred:
            return 1.0
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-path",
        default=os.environ.get("HALO_DEFAULT_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    )
    ap.add_argument("--scorer", required=True,
                    choices=["uniform_random", "halopress", "fifo", "quest",
                             "magicpig_sampled"])
    ap.add_argument("--subtask", required=True)
    ap.add_argument("--context-length", type=int, default=32768)
    ap.add_argument("--n-examples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--memory-ratio", type=float, default=4.0)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--recent-window", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--disable-triton", action="store_true",
                    help="Force the reference Python LSE-merge loop (no Triton).")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    raw_path = Path(
        f"experiments/ruler_data/qwen2.5-7b/{args.context_length}/"
        f"{args.subtask}/validation.jsonl"
    )
    if not raw_path.exists():
        raise FileNotFoundError(
            f"RULER {args.subtask}@{args.context_length} not found at {raw_path}"
        )
    examples = [
        json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()
    ]
    print(f"[FU_W12 {args.scorer}] loaded {len(examples)} examples from {raw_path}",
          flush=True)

    print(f"[FU_W12 {args.scorer}] loading {args.model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()
    print(
        f"[FU_W12 {args.scorer}] model loaded; "
        f"baseline GPU = {torch.cuda.memory_allocated()/1e9:.2f} GiB",
        flush=True,
    )

    if args.disable_triton:
        os.environ["HALO_DISABLE_TRITON"] = "1"
    abl_cfg = AblationScorerConfig(
        scorer_name=args.scorer,
        page_size=args.page_size,
        memory_ratio=args.memory_ratio,
        chunk_size=args.chunk_size,
        recent_window=args.recent_window,
        use_triton=not args.disable_triton,
        seed=args.seed,
    )
    wrap_with_path_d_ablation_scorer(model, abl_cfg)
    print(
        f"[FU_W12 {args.scorer}] wrapped (memory_ratio={args.memory_ratio}, "
        f"page={args.page_size}, chunk={args.chunk_size}, "
        f"triton={'on' if abl_cfg.use_triton else 'off'})",
        flush=True,
    )

    selected = examples[: args.n_examples]
    preds = []
    for i, ex in enumerate(selected):
        prompt = ex["input"]
        if ex.get("answer_prefix") and not prompt.rstrip().endswith(
            ex["answer_prefix"].rstrip()
        ):
            prompt = prompt + ex["answer_prefix"]
        gold = ex["outputs"]

        torch.cuda.reset_peak_memory_stats()
        ids = tok(
            prompt, return_tensors="pt", truncation=True,
            max_length=args.context_length,
        ).to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out_ids = model.generate(
                **ids, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tok.eos_token_id,
            )
        wall = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        pred = tok.decode(
            out_ids[0, ids.input_ids.shape[1]:], skip_special_tokens=True,
        )
        score = _ruler_score(pred, gold)
        preds.append({
            "index": i,
            "ctx_len": ids.input_ids.shape[1],
            "pred": pred[:300],
            "gold": gold if isinstance(gold, str) else list(gold),
            "score": score,
            "wall_s": wall,
            "peak_gpu_gib": peak,
        })
        print(
            f"  [{i+1}/{len(selected)}] scorer={args.scorer} "
            f"{args.subtask}@{args.context_length} "
            f"score={score:.0f} wall={wall:.1f}s peak={peak:.2f}GiB",
            flush=True,
        )
        torch.cuda.empty_cache()

    scores = [p["score"] for p in preds]
    walls = [p["wall_s"] for p in preds]
    peaks = [p["peak_gpu_gib"] for p in preds]
    summary = {
        "method": f"halo_path_d_{args.scorer}_scorer",
        "scorer": args.scorer,
        "memory_ratio": args.memory_ratio,
        "page_size": args.page_size,
        "chunk_size": args.chunk_size,
        "recent_window": args.recent_window,
        "subtask": args.subtask,
        "context_length": args.context_length,
        "seed": args.seed,
        "n": len(scores),
        "mean_score_pct": (100.0 * sum(scores) / len(scores)) if scores else None,
        "mean_wall_s": (sum(walls) / len(walls)) if walls else None,
        "mean_peak_gpu_gib": (sum(peaks) / len(peaks)) if peaks else None,
        "max_peak_gpu_gib": max(peaks) if peaks else None,
        "model_path": args.model_path,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "preds.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    if summary["mean_score_pct"] is None:
        print(f"[FU_W12 {args.scorer}] no scores", flush=True)
    else:
        print(
            f"\n[FU_W12 {args.scorer}] {args.subtask}@{args.context_length}: "
            f"mean score = {summary['mean_score_pct']:.2f}% over n={summary['n']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
