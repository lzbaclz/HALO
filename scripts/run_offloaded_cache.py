"""FU_W6: HF OffloadedCache baseline (KV offload without LSE merge).

Same model / task / context / template as scripts/run_kivi_hf_v2.py but with
HuggingFace's built-in `OffloadedCache`, which moves KV pages to CPU when
not in use and pages them back on demand. The closest in-tree analogue of
"FlexGen-style" offloading; covered by Theorem 1's commitment lower bound
only if pages outside the active selection are zeroed (HF OffloadedCache
keeps everything and pages back, so it is *not* a commitment policy and
should match Full F1 if it runs at all).

Usage:
    python scripts/run_offloaded_cache.py \
        --context-length 65536 --n-examples 5 \
        --output experiments/auxiliary_cells/W6_offloaded_cache_65k
"""
from __future__ import annotations

import os
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import OffloadedCache


def _qa_f1(pred: str, gold) -> float:
    import re, string
    from collections import Counter

    def norm(s):
        s = s.lower()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = "".join(ch for ch in s if ch not in set(string.punctuation))
        return " ".join(s.split())

    pred_n = norm(pred)
    golds = [gold] if isinstance(gold, str) else list(gold)
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
    ap.add_argument("--context-length", type=int, default=65536)
    ap.add_argument("--n-examples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"[FU_W6] loading from {args.model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()
    print(f"[FU_W6] model loaded; baseline GPU = {torch.cuda.memory_allocated()/1e9:.2f} GiB", flush=True)

    raw_path = Path("experiments/infinitebench_raw/longbook_qa_eng.jsonl")
    examples = [json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()]
    template = ("Read the book below and answer a question.\n\n{context}\n\n"
                "Question: {input}\n\nBe very concise.\nAnswer:")
    # Middle-truncation: preserve the question (at the prompt end) by keeping
    # first half + last half of the tokenised prompt. Matches the canonical
    # harness in baselines/infinitebench_eval.py.
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
    print(f"[FU_W6] selected {len(selected)} prompts truncated to {args.context_length}", flush=True)

    preds = []
    for i, ex in enumerate(selected):
        torch.cuda.reset_peak_memory_stats()
        ids = tok(ex["prompt"], return_tensors="pt", truncation=True,
                  max_length=args.context_length).to(model.device)
        cache = OffloadedCache()
        t0 = time.time()
        with torch.no_grad():
            out_ids = model.generate(
                **ids, max_new_tokens=64, do_sample=False,
                pad_token_id=tok.eos_token_id, past_key_values=cache,
            )
        wall = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        pred = tok.decode(out_ids[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
        f1 = _qa_f1(pred, ex["gold"])
        preds.append({"index": i, "ctx_len": ex["ctx_len"], "pred": pred[:400],
                      "gold": ex["gold"] if isinstance(ex["gold"], str) else list(ex["gold"]),
                      "f1": f1, "wall_s": wall, "peak_gpu_gib": peak})
        print(f"  [{i+1}/{len(selected)}] ctx={ex['ctx_len']:>6} f1={f1:.3f} wall={wall:.1f}s peak={peak:.2f}GiB", flush=True)
        del cache, out_ids
        torch.cuda.empty_cache()

    f1s = [p["f1"] for p in preds]
    walls = [p["wall_s"] for p in preds]
    peaks = [p["peak_gpu_gib"] for p in preds]
    summary = {
        "method": "hf_offloaded_cache",
        "n": len(f1s),
        "mean_f1_pct": 100.0 * sum(f1s) / len(f1s) if f1s else None,
        "mean_wall_s": sum(walls) / len(walls) if walls else None,
        "mean_peak_gpu_gib": sum(peaks) / len(peaks) if peaks else None,
        "max_peak_gpu_gib": max(peaks) if peaks else None,
        "context_length_target": args.context_length,
        "model_path": args.model_path,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "preds.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")

    if summary["mean_f1_pct"] is None:
        print("\n[FU_W6] no prompts ran", flush=True)
    else:
        print(f"\n[FU_W6] mean F1 = {summary['mean_f1_pct']:.2f}% over n={summary['n']}", flush=True)
        print(f"[FU_W6] mean wall = {summary['mean_wall_s']:.1f}s, max peak = {summary['max_peak_gpu_gib']:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
