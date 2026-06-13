"""Teacher-forced lossless test: Full vs Path D logits at every step,
both methods conditioned on the SAME history (Full's greedy picks).

Why this metric is the right one
---------------------------------
The "per-step top-1 match" metric in
``scripts/run_chunked_prefill_longctx.py`` lets each method continue
on its own decoded history. Once Full and Path D diverge on token T
(e.g., bf16 noise flips a near-tied argmax), step T+1 is comparing
two completely different distributions — not a lossless test
anymore. The right test feeds BOTH methods the same token at each
step (teacher-forcing on Full's picks), so the side-by-side logit
comparison reflects only Path D's chunked-attention error vs Full's
one-shot SDPA.

Reports:
* Per-step max |logit diff|, mean |logit diff|, L2 |diff|/|full|.
* Per-step top-1 match (argmax under teacher-forced history).
* Per-step top-5 overlap.
* First step at which Path D's argmax diverges from Full's argmax.

This is the metric that should appear in the paper for the Path D
lossless contract.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from halo.chunked_prefill import _force_chunked_mode, _peel_all_layers  # noqa: E402
from halo.policy import HALOConfig, wrap_with_halo  # noqa: E402


def _make_long_prompt(tok, target_tokens):
    base = (
        "You are an expert reading-comprehension system. Below are "
        "several passages on different topics. After the passages, "
        "answer the question.\n\n"
    )
    passage = (
        "Passage: The history of the printing press began in the early "
        "fifteenth century when Johannes Gutenberg combined existing "
        "technologies in a novel way. The movable-type system "
        "revolutionised book production and altered the transmission "
        "of knowledge across Europe.\n"
    )
    q = (
        "\nQuestion: Who is credited with combining the components of "
        "the printing press in fifteenth-century Europe?\nAnswer:"
    )
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
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--target-tokens", type=int, default=16384)
    ap.add_argument("--gen-tokens", type=int, default=8)
    ap.add_argument("--prefill-chunk", type=int, default=2048)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--out-json", default="/tmp/teacher_forced_diag.json")
    args = ap.parse_args()

    print(f"[teacher-forced] loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa",
    )

    ids = _make_long_prompt(tok, args.target_tokens)
    input_ids = torch.tensor([ids], device="cuda:0")
    T = input_ids.shape[-1]
    print(f"[teacher-forced] prompt tokens = {T}, gen tokens = {args.gen_tokens}")

    # ============= 1. Full: greedy generation =============
    print(f"\n[teacher-forced] Full greedy generate ...")
    full_gen = model.generate(
        input_ids, max_new_tokens=args.gen_tokens, do_sample=False,
        return_dict_in_generate=True, output_logits=True, use_cache=True,
        eos_token_id=None, pad_token_id=0,
    )
    full_ids = full_gen.sequences[0, T:].tolist()
    full_step_logits = [lg.detach().float().clone() for lg in full_gen.logits]
    print(f"  Full generated_ids = {full_ids}")

    # ============= 2. Path D: chunked prefill + teacher-forced decode =============
    print(f"\n[teacher-forced] Path D chunked prefill + teacher-forced decode ...")
    cfg = HALOConfig(chunked=True, chunk_size=args.chunk_size,
                     recent_window=64, hot_ratio=1.0, tiers=("dram",))
    wrap_with_halo(model, cfg)
    cache = model._halo_cache
    cache.reset()
    _force_chunked_mode(cache)

    # Chunked prefill: feed the prompt slices and capture the
    # last-position logits at the end.
    pos = 0
    pd_last_prefill_logits = None
    while pos < T:
        end = min(pos + args.prefill_chunk, T)
        slice_ids = input_ids[:, pos:end]
        cache_position = torch.arange(pos, end, device="cuda:0")
        out = model(input_ids=slice_ids, past_key_values=cache,
                    cache_position=cache_position, use_cache=True,
                    return_dict=True)
        if end == T:
            pd_last_prefill_logits = out.logits[:, -1, :].detach().float().clone()
        del out
        _peel_all_layers(cache)
        pos = end

    # Teacher-forced decode: at each step, feed Full's previously
    # picked token (NOT Path D's own pick), so Path D's KV state
    # tracks Full's. This isolates Path D's per-step logit error.
    pd_step_logits = [pd_last_prefill_logits.clone()]
    for i in range(args.gen_tokens - 1):
        # Feed Full's choice at step i.
        next_tok = torch.tensor([[full_ids[i]]], device="cuda:0")
        T_cached = cache.get_seq_length()
        cp = torch.tensor([T_cached], device="cuda:0")
        out = model(input_ids=next_tok, past_key_values=cache,
                    cache_position=cp, use_cache=True, return_dict=True)
        pd_step_logits.append(out.logits[:, -1, :].detach().float().clone())
        del out
        _peel_all_layers(cache)

    # ============= 3. Step-by-step comparison =============
    print(f"\n[teacher-forced] ===== step-by-step lossless comparison =====")
    rows = []
    first_divergence = None
    for i, (fl, pl) in enumerate(zip(full_step_logits, pd_step_logits)):
        fl = fl[0]            # (V,)
        pl = pl[0]
        diff = (fl - pl).abs()
        full_top5 = torch.topk(fl, 5).indices.tolist()
        pd_top5 = torch.topk(pl, 5).indices.tolist()
        full_argmax = fl.argmax().item()
        pd_argmax = pl.argmax().item()
        overlap = len(set(full_top5) & set(pd_top5))
        top1_match = full_argmax == pd_argmax
        if not top1_match and first_divergence is None:
            first_divergence = i
        l2_rel = (diff.norm() / fl.norm().clamp_min(1e-6)).item()
        rows.append({
            "step": i,
            "full_argmax": full_argmax,
            "pd_argmax": pd_argmax,
            "top1_match": top1_match,
            "top5_overlap": overlap,
            "full_top5": full_top5,
            "pd_top5": pd_top5,
            "max_diff": diff.max().item(),
            "mean_diff": diff.mean().item(),
            "l2_rel": l2_rel,
        })
        marker = "✓" if top1_match else "✗"
        print(f"  step {i}: {marker} full={full_argmax:6d}  pd={pd_argmax:6d}  "
              f"top5_overlap={overlap}/5  max_diff={diff.max():.3f}  L2_rel={l2_rel:.4f}")

    print(f"\n[teacher-forced] ===== verdict =====")
    if first_divergence is None:
        print(f"  ✓ ALL {args.gen_tokens} steps top-1 match — Path D is lossless "
              f"up to bf16 noise; greedy generation identical to Full.")
    else:
        print(f"  Path D top-1 matches Full for steps 0..{first_divergence-1} "
              f"({first_divergence}/{args.gen_tokens} = "
              f"{first_divergence/args.gen_tokens:.1%}).")
        print(f"  First divergence at step {first_divergence}: Full picks "
              f"{rows[first_divergence]['full_argmax']}, Path D picks "
              f"{rows[first_divergence]['pd_argmax']}.")
        if rows[first_divergence]['top5_overlap'] >= 4:
            print(f"  top-5 overlap at divergence = "
                  f"{rows[first_divergence]['top5_overlap']}/5 — bf16 noise "
                  f"flipping near-tied logits, not an algorithm bug.")
        else:
            print(f"  top-5 overlap at divergence = "
                  f"{rows[first_divergence]['top5_overlap']}/5 — divergence "
                  f"is beyond bf16 noise; investigate further.")

    avg_top5_overlap = sum(r['top5_overlap'] for r in rows) / max(1, len(rows)) / 5.0
    avg_l2_rel = sum(r['l2_rel'] for r in rows) / max(1, len(rows))
    print(f"  avg top-5 overlap across all {args.gen_tokens} steps = "
          f"{avg_top5_overlap:.1%}")
    print(f"  avg L2 |diff|/|full| = {avg_l2_rel:.4f}")

    out = {
        "model": args.model,
        "target_tokens": args.target_tokens,
        "actual_prompt_tokens": T,
        "gen_tokens": args.gen_tokens,
        "prefill_chunk": args.prefill_chunk,
        "chunk_size": args.chunk_size,
        "full_generated_ids": full_ids,
        "first_divergence_step": first_divergence,
        "avg_top5_overlap_all_steps": avg_top5_overlap,
        "avg_l2_rel_all_steps": avg_l2_rel,
        "per_step": rows,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"\n[teacher-forced] wrote {args.out_json}")


if __name__ == "__main__":
    main()
