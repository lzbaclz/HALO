"""Diagnose Path D's apparent off-by-one token mismatch vs Full.

Reviewer 2nd-round observation:
- Full first 4 tokens: [25, 54105, 51586, 151643]
- Path D first 4 tokens: [54105, 51586, 151643, 33975]
- Path D's tokens look like a LEFT-SHIFT of Full's tokens.

Two competing hypotheses:
(A) Off-by-one BUG: Path D's `last_logits` captured at the wrong position
    (e.g., position T-2 instead of T-1 of the prompt). This would
    produce a shift by one.
(B) bf16 NOISE: at 65K with many LSE-merges, accumulated bf16 error
    flips the argmax on near-tied top-2 logits. Shift looks
    deterministic only because the prompt is structured.

This script tests both by comparing:
1. Full's logits at prompt[:T].last_position
2. Path D's last_logits after chunked prefill on prompt[:T]

If hypothesis (A): argmax of (1) and (2) differ AND the diff is large
(top-1 not in top-5 of other). Then there's a code bug.
If hypothesis (B): argmax differs but top-5 overlap is high. Then
it's bf16 noise on near-tied logits — not a bug.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from halo.chunked_prefill import _force_chunked_mode, _peel_all_layers  # noqa: E402
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
    q_ids = tok.encode(q, add_special_tokens=False)
    while len(ids) + len(pas_ids) + len(q_ids) < target_tokens:
        ids.extend(pas_ids)
    ids.extend(q_ids)
    return ids[:target_tokens]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=8192)
    ap.add_argument("--prefill-chunk", type=int, default=2048)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
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

    # ============= 1. Full one-shot forward (reference) =============
    print(f"[diag] Full one-shot forward ...")
    out_full = model(input_ids=input_ids, use_cache=False)
    full_last_logits = out_full.logits[:, -1, :].detach().clone()
    full_top5 = torch.topk(full_last_logits.float(), 5).indices[0].tolist()
    full_argmax = full_last_logits.argmax(-1).item()
    print(f"  Full last-position argmax = {full_argmax}")
    print(f"  Full last-position top-5  = {full_top5}")
    del out_full
    torch.cuda.empty_cache()

    # ============= 2. Full one-shot via .generate() (for output_logits comparison) =============
    print(f"\n[diag] Full via .generate() with output_logits=True ...")
    gen_out = model.generate(
        input_ids, max_new_tokens=4, do_sample=False,
        return_dict_in_generate=True, output_logits=True, use_cache=True,
    )
    gen_ids = gen_out.sequences[0, T:].tolist()
    gen_logits_argmax = [lg.argmax(-1).item() for lg in gen_out.logits]
    print(f"  generated_token_ids = {gen_ids}")
    print(f"  len(gen_out.logits) = {len(gen_out.logits)}")
    print(f"  per_step argmax     = {gen_logits_argmax}")
    print(f"  -> gen_out.logits[0].argmax = {gen_logits_argmax[0]}; "
          f"generated_token_ids[0] = {gen_ids[0]}")
    if gen_logits_argmax[0] == gen_ids[0]:
        print(f"  ✓ output_logits[0] aligns with generated_token_ids[0]")
    else:
        print(f"  ✗ MISALIGNMENT: output_logits[0] != generated_token_ids[0]")
        print(f"    This is the bug the reviewer flagged.")
    torch.cuda.empty_cache()

    # ============= 3. Path D chunked prefill last_logits =============
    print(f"\n[diag] Path D chunked prefill ...")
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
        cache_position = torch.arange(pos, end, device="cuda:0")
        out = model(input_ids=slice_ids, past_key_values=cache,
                    cache_position=cache_position, use_cache=True,
                    return_dict=True)
        if end == T:
            pd_last_logits = out.logits[:, -1, :].detach().clone()
        del out
        _peel_all_layers(cache)
        pos = end

    pd_top5 = torch.topk(pd_last_logits.float(), 5).indices[0].tolist()
    pd_argmax = pd_last_logits.argmax(-1).item()
    print(f"  Path D last_logits.argmax = {pd_argmax}")
    print(f"  Path D last_logits top-5  = {pd_top5}")

    # ============= 4. Comparison =============
    print(f"\n[diag] ===== Comparison Full vs Path D last-position logits =====")
    print(f"  Full   argmax = {full_argmax}")
    print(f"  Path D argmax = {pd_argmax}")
    if full_argmax == pd_argmax:
        print(f"  ✓ argmax MATCHES — no off-by-one")
    else:
        print(f"  ✗ argmax DIFFERS by {full_argmax} vs {pd_argmax}")

    diff = (full_last_logits.float() - pd_last_logits.float()).abs()
    print(f"  max |logit diff|  = {diff.max().item():.4e}")
    print(f"  mean |logit diff| = {diff.mean().item():.4e}")
    print(f"  L2 |diff|/L2 full = "
          f"{(diff.norm() / full_last_logits.float().norm()).item():.4e}")
    overlap = len(set(full_top5) & set(pd_top5))
    print(f"  top-5 overlap     = {overlap}/5")
    is_full_top5 = full_argmax in pd_top5
    is_pd_top5   = pd_argmax in full_top5
    print(f"  Full argmax in Path D top-5? {is_full_top5}")
    print(f"  Path D argmax in Full top-5? {is_pd_top5}")

    print(f"\n[diag] ===== Verdict =====")
    if full_argmax == pd_argmax:
        print("  Same argmax: no divergence at all on this prompt.")
    elif is_full_top5 and is_pd_top5 and overlap >= 4:
        print("  Hypothesis (B) — bf16 noise on near-tied top-2 logits.")
        print("  argmax differs, but both winners are in each other's top-5 and")
        print("  top-5 overlap is high. The Path D implementation is approximately")
        print("  correct; the difference is reduction-order noise.")
    else:
        print("  Hypothesis (A) — code bug (off-by-one in logits-capture).")
        print("  Logit distributions diverge beyond bf16 noise.")
    print()
    print("  ALSO note the HF .generate output_logits alignment:")
    if gen_logits_argmax[0] == gen_ids[0]:
        print("    HF output_logits[0] predicts generated_token_ids[0] (correct)")
    else:
        print("    HF output_logits[0] predicts generated_token_ids[1] (OFF-BY-ONE)")
        print("    The 'per-step top-1 match = 100%' in the manifest is misleading;")
        print("    it compared Path D step k to Full step k+1. The token-by-token")
        print("    metric (`agreement_top1_seq`) is the correct one to trust.")


if __name__ == "__main__":
    main()
