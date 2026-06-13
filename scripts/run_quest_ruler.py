"""Quest standalone on RULER adversarial NIAH (apples-to-apples vs Path D / KIVI).

Same protocol as scripts/run_pathd_ruler.py and scripts/run_kivi_hf_ruler.py:
- Same model checkpoint (Qwen 2.5-7B-Instruct)
- Same RULER subtasks (niah_multikey_1, niah_multikey_2, niah_multivalue, niah_multiquery)
- Same context length (32768)
- Same n=20 prompts
- Same scoring (RULER exact-match substring)
- memory_ratio = 4 (so top_k = num_pages / 4, i.e. 25% of pages kept per step)
  matching Path D's hot_ratio=0.25 and KIVI HF int4 nbits=4 budget.

The expectation: Quest is a *query-aware* per-step retriever
(re-picks top-K pages on every decoding step using its per-page upper
bound against the current query q). On adversarial NIAH this should
significantly outperform Quest-like query-unaware commitment policies
(H2O / SnapKV / StreamingLLM) but may still under-perform Path D's
non-committing tiered execution on multi-needle cases where the
needle pages happen to fall below Quest's top-K cutoff.

Usage:
    python scripts/run_quest_ruler.py \\
        --subtask niah_multikey_1 --context-length 32768 \\
        --n-examples 20 --output experiments/.../W12_quest_niah_multikey_1_32k
"""
from __future__ import annotations

import os
import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.quest_cache import QuestConfig, wrap_with_quest


def _ruler_score(pred: str, gold) -> float:
    if isinstance(gold, str):
        golds = [gold]
    else:
        golds = list(gold)
    for g in golds:
        if str(g) in pred:
            return 1.0
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.environ.get("HALO_DEFAULT_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    ap.add_argument("--subtask", required=True)
    ap.add_argument("--context-length", type=int, default=32768)
    ap.add_argument("--n-examples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--memory-ratio", type=float, default=4.0,
                    help="Quest memory ratio (top_k = num_pages / memory_ratio); "
                         "default 4.0 matches Path D's hot_ratio=0.25")
    ap.add_argument("--page-size", type=int, default=16,
                    help="Quest page size (default 16, paper standard)")
    ap.add_argument("--sink-pages", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    raw_path = Path(f"experiments/ruler_data/qwen2.5-7b/{args.context_length}/{args.subtask}/validation.jsonl")
    if not raw_path.exists():
        raise FileNotFoundError(f"RULER {args.subtask}@{args.context_length} not found at {raw_path}")
    examples = [json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()]
    print(f"[quest] loaded {len(examples)} examples from {raw_path}", flush=True)

    print(f"[quest] loading {args.model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()
    print(f"[quest] model loaded; baseline GPU = {torch.cuda.memory_allocated()/1e9:.2f} GiB", flush=True)

    cfg = QuestConfig(
        page_size=args.page_size,
        sink_pages=args.sink_pages,
        memory_ratio=args.memory_ratio,
    )
    wrap_with_quest(model, cfg)
    print(f"[quest] wrapped with Quest (page_size={args.page_size}, memory_ratio={args.memory_ratio})", flush=True)

    selected = examples[: args.n_examples]
    preds = []
    for i, ex in enumerate(selected):
        prompt = ex["input"]
        if ex.get("answer_prefix") and not prompt.rstrip().endswith(ex["answer_prefix"].rstrip()):
            prompt = prompt + ex["answer_prefix"]
        gold = ex["outputs"]

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
        pred = tok.decode(out_ids[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
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
        print(f"  [{i+1}/{len(selected)}] {args.subtask}@{args.context_length} score={score:.0f} wall={wall:.1f}s peak={peak:.2f}GiB", flush=True)
        torch.cuda.empty_cache()

    scores = [p["score"] for p in preds]
    walls = [p["wall_s"] for p in preds]
    peaks = [p["peak_gpu_gib"] for p in preds]
    summary = {
        "method": "quest_standalone",
        "memory_ratio": args.memory_ratio,
        "page_size": args.page_size,
        "sink_pages": args.sink_pages,
        "subtask": args.subtask,
        "context_length": args.context_length,
        "n": len(scores),
        "mean_score_pct": 100.0 * sum(scores) / len(scores) if scores else None,
        "mean_wall_s": sum(walls) / len(walls) if walls else None,
        "mean_peak_gpu_gib": sum(peaks) / len(peaks) if peaks else None,
        "max_peak_gpu_gib": max(peaks) if peaks else None,
        "model_path": args.model_path,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "preds.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    if summary["mean_score_pct"] is None:
        print("[quest] no scores", flush=True)
    else:
        print(f"\n[quest] {args.subtask}@{args.context_length}: mean score = {summary['mean_score_pct']:.2f}% over n={summary['n']}", flush=True)


if __name__ == "__main__":
    main()
