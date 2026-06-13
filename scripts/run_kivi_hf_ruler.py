"""FU_W7: KIVI HF int4 on RULER adversarial NIAH (where commitment policies collapse).

The reviewer wants a setting where Path D demonstrably outperforms KIVI.
The NIAH adversarial r-sweep already shows commitment policies (HALOPress)
collapse on niah_multikey/multivalue at low hot ratios while Path D holds.
This script runs KIVI HF int4 on the same cells to see whether
quantization noise compounds with multi-key retrieval in a way that
commitment-policy noise does not.

The hypothesis under the lower bound (Thm 3.2) is: commitment policies
lose (1-r)V_max on adversarial NIAH; KIVI is NOT a commitment policy
(keeps every position), so the lower bound does NOT cover it. But
multi-key NIAH stresses the *precision* of the q·k inner product on
multiple positions simultaneously, which is exactly what int4
quantization perturbs. We expect KIVI to degrade smoothly as the
needle count increases (multikey_1 > multikey_2 > multikey_3) and on
the multivalue / multiquery variants.

Usage:
    python scripts/run_kivi_hf_ruler.py \
        --subtask niah_multikey_2 --context-length 32768 \
        --n-examples 30 --output experiments/.../W7_kivi_niah_multikey_2_32k
"""
from __future__ import annotations

import os
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import HQQQuantizedCache


def _ruler_score(pred: str, gold) -> float:
    """RULER's exact-match scoring: 1.0 if any of the gold answers appears as
    a substring (case-sensitive) in the model's prediction, else 0.0.
    Matches the convention in run_ruler.py / RULER's reference evaluator."""
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
    ap.add_argument("--subtask", required=True,
                    help="e.g. niah_multikey_2, niah_multivalue, niah_multiquery")
    ap.add_argument("--context-length", type=int, default=32768)
    ap.add_argument("--n-examples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--nbits", type=int, default=4)
    ap.add_argument("--q-group-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    raw_path = Path(f"experiments/ruler_data/qwen2.5-7b/{args.context_length}/{args.subtask}/validation.jsonl")
    if not raw_path.exists():
        raise FileNotFoundError(f"RULER {args.subtask}@{args.context_length} not found at {raw_path}")
    examples = [json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()]
    print(f"[FU_W7] loaded {len(examples)} examples from {raw_path}", flush=True)

    print(f"[FU_W7] loading {args.model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()
    print(f"[FU_W7] model loaded; baseline GPU = {torch.cuda.memory_allocated()/1e9:.2f} GiB", flush=True)

    selected = examples[: args.n_examples]
    preds = []
    for i, ex in enumerate(selected):
        # RULER prompts are already at the requested context length;
        # use as-is plus a short answer prefix (e.g. "Answer:") which the
        # dataset provides via answer_prefix.
        prompt = ex["input"]
        if ex.get("answer_prefix"):
            # Some RULER variants embed the answer_prefix inside `input` and
            # others provide it separately. The validation.jsonl shipped here
            # already includes it in `input`; we test for presence and append
            # only if absent.
            if not prompt.rstrip().endswith(ex["answer_prefix"].rstrip()):
                prompt = prompt + ex["answer_prefix"]
        gold = ex["outputs"]

        torch.cuda.reset_peak_memory_stats()
        ids = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=args.context_length).to(model.device)
        cache = HQQQuantizedCache(
            config=model.config, nbits=args.nbits,
            axis_key=0, axis_value=0,
            q_group_size=args.q_group_size, residual_length=128,
        )
        t0 = time.time()
        with torch.no_grad():
            out_ids = model.generate(
                **ids, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tok.eos_token_id,
                past_key_values=cache,
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
        print(f"  [{i+1}/{len(selected)}] {args.subtask}@{args.context_length} score={score:.0f} wall={wall:.1f}s peak={peak:.2f}GiB pred='{pred[:60]}...'", flush=True)
        del cache, out_ids
        torch.cuda.empty_cache()

    scores = [p["score"] for p in preds]
    walls = [p["wall_s"] for p in preds]
    peaks = [p["peak_gpu_gib"] for p in preds]
    summary = {
        "method": "kivi_hf_quantized_int4",
        "backend": "hqq",
        "nbits": args.nbits,
        "q_group_size": args.q_group_size,
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
        print("[FU_W7] no scores", flush=True)
    else:
        print(f"\n[FU_W7] {args.subtask}@{args.context_length}: mean score = {summary['mean_score_pct']:.2f}% over n={summary['n']}", flush=True)


if __name__ == "__main__":
    main()
