#!/usr/bin/env python3
"""Run kvpress baselines on RULER NIAH adversarial — matches the W20
protocol (Qwen 2.5-7B-Instruct, 32K, n=20, subtasks mk_1/mk_2/mv/mq).

Round-38 round: adds the 5 newer kvpress baselines
(ExpectedAttention, AdaKV, ThinK, PyramidKV, TOVA) to address the
reviewer ask about "compare with newer baselines". All five are
SDPA-compatible (verified) so they can run at the same 32K context as
the body's headline NIAH cell, unlike H2O / SnapKV which need eager
attention and OOM at 32K (8K-only — see Table commitment-baselines-niah).

Usage:
    python scripts/run_kvpress_niah.py \\
        --method expected_attention --subtask niah_multikey_1 \\
        --context-length 32768 --n-examples 20 \\
        --memory-ratio 4 \\
        --output experiments/newer_baselines/expected_attention_mk_1_32k
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines import REGISTRY


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
    from baselines import REGISTRY as _REG
    ap.add_argument("--method", required=True, choices=sorted(_REG.keys()))
    ap.add_argument("--subtask", required=True)
    ap.add_argument("--context-length", type=int, default=32768)
    ap.add_argument("--n-examples", type=int, default=20)
    ap.add_argument("--memory-ratio", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--model-path",
                    default=os.environ.get("HALO_DEFAULT_MODEL",
                                           "Qwen/Qwen2.5-7B-Instruct"))
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    # RULER data path (Qwen2.5-7B layout: experiments/ruler_data/<task>/<length>.jsonl)
    candidates = [
        Path(f"experiments/ruler_data/{args.subtask}/{args.context_length}.jsonl"),
        Path(f"experiments/ruler_data/qwen2.5-7b/{args.context_length}/{args.subtask}/validation.jsonl"),
    ]
    raw_path = next((c for c in candidates if c.exists()), None)
    if raw_path is None:
        raise FileNotFoundError(f"RULER data not found at any of {candidates}")
    examples = [json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()]
    print(f"[kvpress-niah] loaded {len(examples)} examples from {raw_path}", flush=True)

    # Pick attention impl based on press requirement
    factory = REGISTRY[args.method]
    attn_impl = getattr(factory, "required_attn_impl", "sdpa")
    if args.method == "full":
        attn_impl = "sdpa"
    print(f"[kvpress-niah] method={args.method} attn_impl={attn_impl} mr={args.memory_ratio}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    print(f"[kvpress-niah] loading {args.model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation=attn_impl,
    )
    model.eval()
    print(f"[kvpress-niah] model loaded; baseline GPU = "
          f"{torch.cuda.memory_allocated()/1e9:.2f} GiB", flush=True)

    # Build press
    if args.method == "full":
        press = None
    else:
        press = factory(model, memory_ratio=args.memory_ratio)
        print(f"[kvpress-niah] press: {type(press).__name__}", flush=True)

    selected = examples[: args.n_examples]
    preds = []
    for i, ex in enumerate(selected):
        prompt = ex["input"]
        if ex.get("answer_prefix") and not prompt.rstrip().endswith(ex["answer_prefix"].rstrip()):
            prompt = prompt + ex["answer_prefix"]
        gold = ex["outputs"]

        # Middle-truncate if longer than ctx
        ids = tok(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(ids) > args.context_length:
            half = args.context_length // 2
            prompt = tok.decode(ids[:half], skip_special_tokens=True) + \
                     tok.decode(ids[-half:], skip_special_tokens=True)

        torch.cuda.reset_peak_memory_stats()
        inputs = tok(prompt, return_tensors="pt", truncation=True,
                     max_length=args.context_length).to(model.device)
        ctx_len = inputs.input_ids.shape[-1]
        t0 = time.time()
        cm = press(model) if press is not None else nullcontext()
        with torch.no_grad(), cm:
            out_ids = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tok.eos_token_id,
            )
        wall = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        pred = tok.decode(out_ids[0, ctx_len:], skip_special_tokens=True)
        score = _ruler_score(pred, gold)
        preds.append({
            "index": i, "ctx_len": ctx_len, "pred": pred[:300],
            "gold": gold if isinstance(gold, str) else list(gold),
            "score": score, "wall_s": wall, "peak_gpu_gib": peak,
        })
        print(f"  [{i+1}/{len(selected)}] {args.subtask}@{args.context_length} "
              f"score={score:.0f} wall={wall:.1f}s peak={peak:.2f}GiB pred[:50]={pred[:50]!r}",
              flush=True)
        torch.cuda.empty_cache()

    scores = [p["score"] for p in preds]
    summary = {
        "method": args.method,
        "memory_ratio": args.memory_ratio,
        "subtask": args.subtask,
        "context_length": args.context_length,
        "n": len(scores),
        "mean_score_pct": 100.0 * sum(scores) / len(scores) if scores else None,
        "mean_wall_s": sum(p["wall_s"] for p in preds) / max(1, len(preds)),
        "mean_peak_gpu_gib": sum(p["peak_gpu_gib"] for p in preds) / max(1, len(preds)),
        "max_peak_gpu_gib": max((p["peak_gpu_gib"] for p in preds), default=None),
        "model_path": args.model_path,
        "attn_impl": attn_impl,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "preds.jsonl").open("w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    print(f"\n[kvpress-niah] {args.method} {args.subtask}@{args.context_length}: "
          f"mean = {summary['mean_score_pct']:.2f}% over n={summary['n']}", flush=True)


if __name__ == "__main__":
    main()
