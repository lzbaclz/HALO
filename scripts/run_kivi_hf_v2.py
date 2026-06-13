"""FU_W5 v2: KIVI-style 4-bit KV cache via HF HQQQuantizedCache (Py 3.11 + transformers 4.57.6).

Runs Qwen2.5-7B-Instruct on InfiniteBench En.QA at the requested context length
with the upstream-supported quantized cache backend. Replaces the v1 attempt
that was blocked by Python 3.13 / HQQ Dynamo import-time incompatibility:
this v2 runs in the orchkv conda env (Py 3.11.15 + torch 2.5.1+cu121) where
HQQ imports cleanly.

Usage:
    python scripts/run_kivi_hf_v2.py \
        --context-length 65000 --n-examples 5 \
        --output experiments/auxiliary_cells/W5_kivi_hf_v2
"""
from __future__ import annotations

import os
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import HQQQuantizedCache


def _qa_f1(pred: str, gold) -> float:
    """Minimal partial-F1 (matches the harness in baselines/infinitebench_eval.py)."""
    import re
    import string
    from collections import Counter

    def norm(s):
        s = s.lower()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = "".join(ch for ch in s if ch not in set(string.punctuation))
        return " ".join(s.split())

    pred_n = norm(pred)
    if isinstance(gold, str):
        golds = [gold]
    else:
        golds = list(gold)
    best = 0.0
    for g in golds:
        gold_n = norm(g)
        if not pred_n or not gold_n:
            continue
        p_tok = pred_n.split()
        g_tok = gold_n.split()
        common = Counter(p_tok) & Counter(g_tok)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        p = num_same / len(p_tok)
        r = num_same / len(g_tok)
        f1 = 2 * p * r / (p + r)
        if f1 > best:
            best = f1
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.environ.get("HALO_DEFAULT_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    ap.add_argument("--context-length", type=int, default=65000)
    ap.add_argument("--n-examples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--nbits", type=int, default=4)
    ap.add_argument("--q-group-size", type=int, default=64)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    print(f"[FU_W5 v2] loading tokenizer + model from {args.model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    base_alloc_gb = torch.cuda.memory_allocated() / 1e9
    print(f"[FU_W5 v2] model loaded; baseline GPU alloc = {base_alloc_gb:.2f} GiB", flush=True)

    print("[FU_W5 v2] loading InfiniteBench longbook_qa_eng from local jsonl", flush=True)
    raw_path = Path("experiments/infinitebench_raw/longbook_qa_eng.jsonl")
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Local InfiniteBench EnQA jsonl not found at {raw_path}; "
            "fall back to HF datasets if needed.")
    examples = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    print(f"[FU_W5 v2] loaded {len(examples)} raw EnQA examples", flush=True)

    # The InfiniteBench EnQA prompt is `context` (book) + `input` (question);
    # the upstream eval scripts and our baselines/infinitebench_eval.py
    # concatenate them as:
    #   "Read the book below and answer a question.\n\n{context}\n\nQuestion: {input}\n\nBe very concise.\nAnswer:"
    # We replicate the same template here.
    template = (
        "Read the book below and answer a question.\n\n{context}\n\n"
        "Question: {input}\n\nBe very concise.\nAnswer:"
    )

    # InfiniteBench EnQA prompts are 88K-237K tokens under Qwen tokenization;
    # we apply middle-truncation (keep first half + last half) to preserve
    # the question (at the end of the prompt) while fitting the context
    # window. This matches baselines/infinitebench_eval.py:_run_one (the
    # canonical harness used for Cells A/B/C/D/E in the paper).
    selected = []
    for ex in examples:
        prompt = template.format(context=ex["context"], input=ex["input"])
        ids = tok(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(ids) > args.context_length:
            half = args.context_length // 2
            head = tok.decode(ids[:half], skip_special_tokens=True)
            tail = tok.decode(ids[-half:], skip_special_tokens=True)
            prompt = head + tail
        L = len(tok.encode(prompt))
        selected.append({"prompt": prompt, "gold": ex["answer"], "ctx_len": L})
        if len(selected) >= args.n_examples:
            break
    print(f"[FU_W5 v2] selected {len(selected)} prompts at ~{args.context_length} ctx", flush=True)

    preds = []
    f1s = []
    walls = []
    peak_gpus = []
    for i, ex in enumerate(selected):
        prompt = ex["prompt"]
        gold = ex["gold"]
        L = ex["ctx_len"]

        torch.cuda.reset_peak_memory_stats()

        ids = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=args.context_length).to(model.device)

        cache = HQQQuantizedCache(
            config=model.config,
            nbits=args.nbits,
            axis_key=0, axis_value=0,
            q_group_size=args.q_group_size,
            residual_length=128,
        )

        t0 = time.time()
        with torch.no_grad():
            out_ids = model.generate(
                **ids,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                past_key_values=cache,
            )
        wall = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9

        pred = tok.decode(out_ids[0, ids.input_ids.shape[1]:],
                          skip_special_tokens=True)
        f1 = _qa_f1(pred, gold)

        preds.append({
            "index": i,
            "context_len_tokens": L,
            "pred": pred[:400],
            "gold": gold if isinstance(gold, str) else list(gold),
            "f1": f1,
            "wall_s": wall,
            "peak_gpu_gib": peak,
        })
        f1s.append(f1)
        walls.append(wall)
        peak_gpus.append(peak)
        print(f"  [{i+1}/{len(selected)}] ctx={L:>6} f1={f1:.3f} wall={wall:.1f}s peak_gpu={peak:.2f}GiB", flush=True)

        # Free the cache so the next iteration starts fresh
        del cache, out_ids
        torch.cuda.empty_cache()

    if not f1s:
        print("[FU_W5 v2] WARN: no prompts matched the context-length filter; widen the band", flush=True)
        summary = {
            "method": "kivi_hf_quantized_int4",
            "backend": "hqq",
            "nbits": args.nbits,
            "q_group_size": args.q_group_size,
            "n": 0,
            "mean_f1_pct": None,
            "context_length_target": args.context_length,
        }
    else:
        summary = {
            "method": "kivi_hf_quantized_int4",
            "backend": "hqq",
            "nbits": args.nbits,
            "q_group_size": args.q_group_size,
            "n": len(f1s),
            "mean_f1_pct": 100.0 * sum(f1s) / len(f1s),
            "mean_wall_s": sum(walls) / len(walls),
            "mean_peak_gpu_gib": sum(peak_gpus) / len(peak_gpus),
            "max_peak_gpu_gib": max(peak_gpus),
            "context_length_target": args.context_length,
            "model_path": args.model_path,
        }

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "preds.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")

    if summary.get("mean_f1_pct") is None:
        print(f"\n[FU_W5 v2] no prompts ran (n=0); see WARN above", flush=True)
    else:
        print(f"\n[FU_W5 v2] mean F1 = {summary['mean_f1_pct']:.2f}% over n={summary['n']}", flush=True)
        print(f"[FU_W5 v2] mean wall = {summary['mean_wall_s']:.1f}s, max peak GPU = {summary['max_peak_gpu_gib']:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
