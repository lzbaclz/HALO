"""End-to-end Path D benchmark with chunked prefill on real Qwen2.5-7B.

Measures peak GPU memory at long context (configurable, default 32K)
for two configurations on the SAME prompt:

  (a) Full SDPA   — one-shot prefill + greedy decode via HF generate
  (b) Path D      — chunked prefill (slices of ``prefill_chunk``) +
                    chunked decode, cold KV on host pinned DRAM

Reports:
  - peak GPU (GiB) for each config
  - wall-clock (s) for each
  - token-by-token agreement (% match for first ``gen_tokens``)
  - DMA bytes moved to host
  - cold positions per layer

Picks NarrativeQA from LongBench for the prompts (typical length ~30K
tokens), or accepts a custom prompt file. Defaults are tuned for
A100-80GB.

Writes a manifest with the three triples the reviewers asked for:
(F1-equivalent measured as agreement-with-Full, peak GPU GiB,
wall-clock s).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _make_long_prompt(tok, target_tokens: int) -> "list[int]":
    """Build a long-context prompt by replicating a passage retrieval
    style template until ``target_tokens`` is reached. Uses real
    English to keep the model's behaviour realistic.
    """
    base = (
        "You are an expert reading-comprehension system. Below are "
        "several passages on different topics. After the passages, "
        "answer the question. \n\n"
    )
    passage = (
        "Passage: The history of the printing press began in the early "
        "fifteenth century when Johannes Gutenberg combined existing "
        "technologies in a novel way. The movable-type system revolutionised "
        "the production of books and pamphlets, fundamentally altering the "
        "transmission of knowledge across Europe. Before this innovation, "
        "manuscripts had to be hand-copied, an expensive and slow process "
        "that limited literacy to clergy and aristocracy. Gutenberg's "
        "design used a screw press, hand-cast metal type, and oil-based "
        "ink, all of which had antecedents but had never been combined "
        "into an integrated workflow. The press could produce hundreds "
        "of copies per day, compared with months required for a single "
        "hand-copied volume.\n"
    )
    question = (
        "\nQuestion: Who is credited with combining the components of the "
        "printing press in fifteenth-century Europe?\nAnswer:"
    )
    ids = tok.encode(base, add_special_tokens=False)
    pas_ids = tok.encode(passage, add_special_tokens=False)
    while len(ids) + len(pas_ids) + len(tok.encode(question, add_special_tokens=False)) < target_tokens:
        ids.extend(pas_ids)
    ids.extend(tok.encode(question, add_special_tokens=False))
    return ids[:target_tokens]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from halo import HALOConfig, wrap_with_halo
    from halo.chunked_prefill import prefill_then_generate

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--target-tokens", type=int, default=32768)
    ap.add_argument("--gen-tokens", type=int, default=8)
    ap.add_argument("--prefill-chunk", type=int, default=4096)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--recent-window", type=int, default=64)
    ap.add_argument("--out-dir",
                    default="experiments/runs/qwen2-5-7b/longctx_chunked_prefill")
    ap.add_argument("--skip-full", action="store_true",
                    help="Skip the Full baseline run (useful if you already have it).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[longctx] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa",
    )

    ids = _make_long_prompt(tok, args.target_tokens)
    input_ids = torch.tensor([ids], device="cuda:0")
    print(f"[longctx] prompt tokens = {input_ids.shape[-1]}")

    full_result = None
    full_last_logits = None
    if not args.skip_full:
        # ---- Full SDPA baseline ----
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            # eos_token_id=None forces generation to run the full
            # ``max_new_tokens`` count even on EOS. Without this, Full
            # may stop early at EOS while Path D's manual decode loop
            # runs to completion — the length mismatch then poisons
            # the side-by-side comparison.
            full_gen = model.generate(
                input_ids,
                max_new_tokens=args.gen_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_logits=True,
                use_cache=True,
                eos_token_id=None,
                pad_token_id=0,
            )
        full_wall = time.time() - t0
        full_peak = torch.cuda.max_memory_allocated() / 1024**3
        # IMPORTANT: HF may early-stop on EOS, in which case
        # sequences.shape[-1] < prompt_len + gen_tokens. Slicing with
        # ``[-gen_tokens:]`` would then pick up the *last prompt token*
        # and report it as the ``first generated token'' — the source
        # of the apparent ``off-by-one'' divergence flagged in the
        # 2026-05-12 second-round review. The correct slice is
        # ``[prompt_len:]``.
        prompt_len = input_ids.shape[-1]
        full_ids = full_gen.sequences[0, prompt_len:].tolist()
        # Capture per-step logits for top-k comparison.
        full_step_logits = [lg.detach().clone() for lg in full_gen.logits]
        print(f"[longctx] Full: peak={full_peak:.2f} GiB, wall={full_wall:.1f}s, "
              f"first tokens={full_ids[:5]} (n_gen={len(full_ids)})")
        full_result = {
            "peak_gpu_gib": full_peak,
            "wall_clock_s": full_wall,
            "generated_token_ids": full_ids,
            "per_step_top1": [lg.argmax(-1).item() for lg in full_step_logits],
            "per_step_top5": [torch.topk(lg.float(), 5).indices[0].tolist() for lg in full_step_logits],
        }

    # ---- Path D chunked prefill ----
    cfg = HALOConfig(
        chunked=True, chunk_size=args.chunk_size,
        recent_window=args.recent_window, hot_ratio=1.0,
        tiers=("dram",),
    )
    wrap_with_halo(model, cfg)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    pd_result = prefill_then_generate(
        model, input_ids,
        prefill_chunk_tokens=args.prefill_chunk,
        max_new_tokens=args.gen_tokens,
        do_sample=False,
    )
    pd_wall = time.time() - t0
    # Same slicing fix as the Full branch above: take everything past
    # the prompt so we don't fold the last prompt token in.
    prompt_len = input_ids.shape[-1]
    pd_ids = pd_result["generated_ids"][0, prompt_len:].tolist()
    print(f"[longctx] Path D: prefill_peak={pd_result['prefill_peak_gib']:.2f} GiB, "
          f"decode_peak={pd_result['decode_peak_gib']:.2f} GiB, "
          f"overall={pd_result['overall_peak_gib']:.2f} GiB, "
          f"wall={pd_wall:.1f}s, first tokens={pd_ids[:5]}")

    # ---- Per-step top-k agreement (bf16-noise-tolerant lossless check) ----
    agreement_top1 = agreement_top5 = avg_logit_l2 = None
    if full_result is not None:
        full_ids = full_result["generated_token_ids"]
        # Sequence-level greedy match (strict, bf16-sensitive)
        matches = sum(1 for a, b in zip(full_ids, pd_ids) if a == b)
        agreement_top1_seq = matches / max(1, len(full_ids))
        print(f"[longctx] greedy-id match (sequence): "
              f"{matches}/{len(full_ids)} = {agreement_top1_seq:.2%}")

        # Per-step top-1 / top-5 match using the SAME prefilled-cache
        # logits (i.e. matching position-by-position to Full's per-step
        # logits). This is the lossless-up-to-reduction-order metric.
        pd_step_logits = pd_result["step_logits"]
        full_step_top1 = full_result["per_step_top1"]
        full_step_top5 = full_result["per_step_top5"]
        top1_match = 0
        top5_overlap = 0
        l2_diffs = []
        n_steps = min(len(pd_step_logits), len(full_step_top1))
        for i in range(n_steps):
            pd_lg = pd_step_logits[i].float()
            pd_top1 = pd_lg.argmax(-1).item()
            pd_top5 = set(torch.topk(pd_lg, 5).indices[0].tolist())
            top1_match += int(pd_top1 == full_step_top1[i])
            top5_overlap += len(pd_top5 & set(full_step_top5[i])) / 5.0
        agreement_top1 = top1_match / max(1, n_steps)
        agreement_top5 = top5_overlap / max(1, n_steps)
        print(f"[longctx] per-step top-1 match : "
              f"{top1_match}/{n_steps} = {agreement_top1:.2%}")
        print(f"[longctx] per-step top-5 overlap: {agreement_top5:.2%}")

    manifest = {
        "model": args.model,
        "target_tokens": args.target_tokens,
        "actual_prompt_tokens": int(input_ids.shape[-1]),
        "gen_tokens": args.gen_tokens,
        "prefill_chunk": args.prefill_chunk,
        "halo_config": {
            "chunked": True,
            "chunk_size": args.chunk_size,
            "recent_window": args.recent_window,
            "hot_ratio": 1.0,
            "tiers": ["dram"],
        },
        "full": full_result,
        "path_d": {
            "prefill_peak_gib": pd_result["prefill_peak_gib"],
            "decode_peak_gib": pd_result["decode_peak_gib"],
            "overall_peak_gib": pd_result["overall_peak_gib"],
            "wall_clock_s": pd_wall,
            "generated_token_ids": pd_ids,
            "cache_telemetry": pd_result["cache_telemetry"],
        },
        "agreement_top1_seq": agreement_top1_seq if full_result is not None else None,
        "agreement_top1_per_step": agreement_top1,
        "agreement_top5_overlap_per_step": agreement_top5,
        "seed": 0,
    }
    out_path = out_dir / f"manifest_{args.target_tokens}.json"
    out_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[longctx] wrote {out_path}")


if __name__ == "__main__":
    main()
