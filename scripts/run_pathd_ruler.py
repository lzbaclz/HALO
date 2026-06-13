"""FU_W7b: Path D on RULER adversarial NIAH (Qwen Instruct, same checkpoint as KIVI).

Direct apples-to-apples comparison vs scripts/run_kivi_hf_ruler.py:
- Same model checkpoint (Qwen 2.5-7B-Instruct)
- Same RULER subtasks (niah_multikey_1, niah_multikey_2, niah_multivalue, niah_multiquery)
- Same context length (32768)
- Same n=20 prompts
- Same scoring (RULER exact-match substring)

The expectation under the lower bound (Thm 3.2 + Prop 4.5) is that
Path D's algebraic identity preserves the needle attention weight on
every position, so multi-key / multi-value tasks should not degrade
on the chunked path. KIVI's int4 quantization, by contrast, perturbs
q·k inner products precisely on the answer-bearing position and
should collapse on adversarial NIAH (this run is to verify).

Usage:
    python scripts/run_pathd_ruler.py \
        --subtask niah_multikey_1 --context-length 32768 \
        --n-examples 20 --output experiments/.../W7_pathd_niah_multikey_1_32k
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

from halo import wrap_with_halo, HALOConfig


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
    ap.add_argument("--hot-ratio", type=float, default=0.25,
                    help="Path D hot ratio (= 1/memory_ratio); default 0.25 matches r=04")
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    raw_path = Path(f"experiments/ruler_data/qwen2.5-7b/{args.context_length}/{args.subtask}/validation.jsonl")
    if not raw_path.exists():
        raise FileNotFoundError(f"RULER {args.subtask}@{args.context_length} not found at {raw_path}")
    examples = [json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()]
    print(f"[FU_W7b] loaded {len(examples)} examples from {raw_path}", flush=True)

    print(f"[FU_W7b] loading {args.model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()
    print(f"[FU_W7b] model loaded; baseline GPU = {torch.cuda.memory_allocated()/1e9:.2f} GiB", flush=True)

    cfg = HALOConfig(
        hot_ratio=args.hot_ratio,
        tiers=("gpu", "dram"),
        chunked=True,
        chunk_size=args.chunk_size,
    )
    wrap_with_halo(model, cfg)
    print(f"[FU_W7b] wrapped with Path D (hot_ratio={args.hot_ratio}, chunk={args.chunk_size})", flush=True)

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
        "method": "halo_path_d_chunked",
        "hot_ratio": args.hot_ratio,
        "chunk_size": args.chunk_size,
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
        print("[FU_W7b] no scores", flush=True)
    else:
        print(f"\n[FU_W7b] {args.subtask}@{args.context_length}: mean score = {summary['mean_score_pct']:.2f}% over n={summary['n']}", flush=True)


if __name__ == "__main__":
    main()
