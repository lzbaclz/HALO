"""FU_W17: Path D head-to-head Full vs Path D on BABILong (distribution-over-context).

BABILong (Kuratov et al., 2024) extends the bAbI reasoning benchmark to
long context by embedding the original short bAbI tasks into a long
distractor corpus. Quality is measured by exact-match on the final
answer. The benchmark stresses *distribution-over-context* tasks where
the answer requires aggregating information from multiple distant
positions — this is the regime where NIAH single-needle saturates and
where Path D's identity contract is most informative.

Run::

    python scripts/run_pathd_babilong.py \\
        --task qa1 --context-length 32768 \\
        --n-examples 50 --seed 0 \\
        --output experiments/fu_w17_babilong/path_d_qa1_32k

Two cells per task: Full attention and Path D. Output schema matches
the existing ``scripts/run_pathd_ruler.py`` format so the aggregator
can pool by ``method`` field.

Dataset: downloaded from HuggingFace at runtime
(``RMT-team/babilong-1k-samples``), 1K-sample chunks per length tier.
We pull the 32K-token tier for the headline cell and the 64K tier for
the longer-context follow-up; both are within the local
``Qwen2.5-7B`` context budget under Path D.
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


def _build_prompt(example: dict) -> str:
    """Construct the BABILong prompt from an example dict.

    BABILong examples ship with keys ``input``, ``question``, ``answer``.
    The ``input`` field already contains the distractor corpus with the
    needle facts interleaved; ``question`` is the bAbI question.
    """
    inp = example.get("input", "")
    question = example.get("question", "")
    prompt = (
        f"{inp}\n\n"
        f"Question: {question}\n"
        "Answer (one or two words only):"
    )
    return prompt


def _em_score(pred: str, gold: str) -> float:
    """Exact-match scoring (case-insensitive, strip whitespace)."""
    p = pred.strip().lower()
    g = gold.strip().lower()
    if not p or not g:
        return 0.0
    # Accept gold as a substring of pred's first 64 chars (mirrors the
    # RULER convention).
    return 1.0 if g in p[:64] else 0.0


def _wrap_model(model, *, method: str, hot_ratio: float, chunk_size: int):
    """Install Full / Path D / Quest+Path D on a freshly loaded HF model."""
    if method == "full":
        return model
    if method == "path_d":
        from halo import HALOConfig, wrap_with_halo
        cfg = HALOConfig(
            chunked=True,
            chunk_size=chunk_size,
            hot_ratio=hot_ratio,
            use_triton=False,  # match FU_W12 reference path
            tiers=("gpu", "dram"),
        )
        wrap_with_halo(model, cfg)
        return model
    if method == "quest_path_d":
        from baselines.quest_path_d import (
            QuestPathDConfig, wrap_with_quest_path_d,
        )
        cfg = QuestPathDConfig(
            memory_ratio=1.0 / max(hot_ratio, 1e-3),
            chunk_size=chunk_size,
            use_triton=False,
        )
        wrap_with_quest_path_d(model, cfg)
        return model
    raise ValueError(f"unknown method {method!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-path",
        default=os.environ.get("HALO_DEFAULT_MODEL", "Qwen/Qwen2.5-7B"),
    )
    ap.add_argument("--method", choices=["full", "path_d", "quest_path_d"],
                    required=True)
    ap.add_argument("--task", default="qa1",
                    help="BABILong task (qa1..qa20)")
    ap.add_argument("--context-length", type=int, default=32768)
    ap.add_argument("--n-examples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--hot-ratio", type=float, default=0.25)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument(
        "--data-dir",
        default=os.environ.get("HALO_BABILONG_DIR",
                               "experiments/babilong_data"),
        help="Local jsonl mirror of the BABILong task; expected file "
             "<data_dir>/<context_kib>/<task>.jsonl",
    )
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    ctx_kib = max(args.context_length // 1024, 1)
    src = Path(args.data_dir) / f"{ctx_kib}k" / f"{args.task}.jsonl"
    if not src.exists():
        sys.exit(
            f"[fu_w17] BABILong data missing at {src}.\n"
            f"  Prepare via scripts/prepare_babilong_data.py "
            f"(or symlink an existing mirror)."
        )

    examples = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    selected = examples[: args.n_examples]
    print(f"[fu_w17] task={args.task} ctx={args.context_length} "
          f"n_loaded={len(examples)} n_selected={len(selected)}", flush=True)

    print(f"[fu_w17] loading {args.model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
    ).eval()
    _wrap_model(model, method=args.method, hot_ratio=args.hot_ratio,
                chunk_size=args.chunk_size)
    print(f"[fu_w17] wrapped method={args.method}  "
          f"baseline GPU={torch.cuda.memory_allocated()/1e9:.2f} GiB",
          flush=True)

    preds = []
    for i, ex in enumerate(selected):
        prompt = _build_prompt(ex)
        gold = ex.get("answer", "")
        torch.cuda.reset_peak_memory_stats()
        ids = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=args.context_length).to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out_ids = model.generate(
                **ids, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tok.eos_token_id,
            )
        wall = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        pred = tok.decode(out_ids[0, ids.input_ids.shape[1]:],
                          skip_special_tokens=True)
        score = _em_score(pred, gold)
        preds.append({
            "index": i,
            "ctx_len": int(ids.input_ids.shape[1]),
            "pred": pred[:200],
            "gold": gold,
            "score": score,
            "wall_s": wall,
            "peak_gpu_gib": peak,
        })
        print(f"  [{i+1}/{len(selected)}] {args.method}@{args.task}@{args.context_length} "
              f"score={score:.0f} wall={wall:.1f}s peak={peak:.2f}GiB", flush=True)
        torch.cuda.empty_cache()

    scores = [p["score"] for p in preds]
    walls = [p["wall_s"] for p in preds]
    peaks = [p["peak_gpu_gib"] for p in preds]
    summary = {
        "method": args.method,
        "task": args.task,
        "context_length": args.context_length,
        "seed": args.seed,
        "n": len(scores),
        "mean_em_pct": (100.0 * sum(scores) / len(scores)) if scores else None,
        "mean_wall_s": (sum(walls) / len(walls)) if walls else None,
        "mean_peak_gib": (sum(peaks) / len(peaks)) if peaks else None,
        "max_peak_gib": max(peaks) if peaks else None,
        "model_path": args.model_path,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "preds.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    if summary["mean_em_pct"] is None:
        print(f"[fu_w17] no scores", flush=True)
    else:
        print(f"\n[fu_w17] {args.method}@{args.task}@{args.context_length}: "
              f"EM = {summary['mean_em_pct']:.2f}% over n={summary['n']}",
              flush=True)


if __name__ == "__main__":
    main()
