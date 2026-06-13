"""Compare logits Full-SDPA vs Path D chunked-prefill on identical prompt.

The token-by-token agreement check on greedy decoding can give 0/N
even when logits differ by less than a millionth — if two tokens are
near-tied in the original softmax, bf16 noise can flip which one
wins. This diagnostic computes:

  max|logits_full - logits_path_d|
  top-5 token overlap
  top-1 match per step

so we can distinguish "math is right, bf16 noise on tied logits"
from "math is wrong, decode is broken".
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from halo.chunked_prefill import (  # noqa: E402
    _force_chunked_mode, _peel_all_layers,
)
from halo.policy import HALOConfig, wrap_with_halo  # noqa: E402


def _make_prompt(tok, target_tokens):
    base = "Below are passages followed by a question.\n\n"
    passage = (
        "Passage: The history of the printing press began in the early "
        "fifteenth century when Johannes Gutenberg combined existing "
        "technologies in a novel way. Before this innovation, manuscripts "
        "had to be hand-copied, an expensive and slow process.\n"
    )
    q = "\nQuestion: Who created the printing press?\nAnswer:"
    ids = tok.encode(base, add_special_tokens=False)
    pas_ids = tok.encode(passage, add_special_tokens=False)
    while len(ids) + len(pas_ids) + len(tok.encode(q, add_special_tokens=False)) < target_tokens:
        ids.extend(pas_ids)
    ids.extend(tok.encode(q, add_special_tokens=False))
    return ids[:target_tokens]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--target-tokens", type=int, default=4096)
    ap.add_argument("--prefill-chunk", type=int, default=1024)
    ap.add_argument("--chunk-size", type=int, default=512)
    args = ap.parse_args()

    print(f"[diag] loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa",
    )

    ids = _make_prompt(tok, args.target_tokens)
    input_ids = torch.tensor([ids], device="cuda:0")
    T = input_ids.shape[-1]
    print(f"[diag] prompt tokens = {T}")

    # ---- Full one-shot prefill ----
    out_full = model(input_ids=input_ids, use_cache=False)
    full_last_logits = out_full.logits[:, -1, :].detach().clone()
    del out_full
    torch.cuda.empty_cache()
    print(f"[diag] Full last-position logits shape = {tuple(full_last_logits.shape)}")
    print(f"[diag] Full top-5 = {torch.topk(full_last_logits.float(), 5).indices.tolist()}")

    # ---- Path D chunked prefill ----
    cfg = HALOConfig(chunked=True, chunk_size=args.chunk_size,
                     recent_window=64, hot_ratio=1.0, tiers=("dram",))
    wrap_with_halo(model, cfg)
    cache = model._halo_cache
    cache.reset()
    _force_chunked_mode(cache)

    pos = 0
    pd_last_logits = None
    while pos < T:
        end = min(pos + args.prefill_chunk, T)
        slice_ids = input_ids[:, pos:end]
        cache_position = torch.arange(pos, end, device=input_ids.device)
        out = model(input_ids=slice_ids, past_key_values=cache,
                    cache_position=cache_position, use_cache=True,
                    return_dict=True)
        if end == T:
            pd_last_logits = out.logits[:, -1, :].detach().clone()
        del out
        _peel_all_layers(cache)
        pos = end

    print(f"[diag] Path D last-position logits shape = {tuple(pd_last_logits.shape)}")
    print(f"[diag] Path D top-5 = {torch.topk(pd_last_logits.float(), 5).indices.tolist()}")

    # ---- compare ----
    diff = (full_last_logits.float() - pd_last_logits.float()).abs()
    rel = diff / full_last_logits.float().abs().clamp_min(1e-6)
    print(f"[diag] max |diff|         = {diff.max().item():.4e}")
    print(f"[diag] mean |diff|        = {diff.mean().item():.4e}")
    print(f"[diag] median |diff|      = {diff.median().item():.4e}")
    print(f"[diag] max |rel diff|     = {rel.max().item():.4e}")
    print(f"[diag] L2 |diff|/L2 full  = "
          f"{(diff.norm() / full_last_logits.float().norm()).item():.4e}")

    full_top5 = set(torch.topk(full_last_logits.float(), 5).indices.tolist()[0])
    pd_top5 = set(torch.topk(pd_last_logits.float(), 5).indices.tolist()[0])
    print(f"[diag] top-5 overlap        = {len(full_top5 & pd_top5)}/5")
    print(f"[diag] top-1 match          = "
          f"{full_last_logits.argmax(-1).item() == pd_last_logits.argmax(-1).item()}")


if __name__ == "__main__":
    main()
