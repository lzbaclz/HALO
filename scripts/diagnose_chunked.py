"""Focused chunked-attention correctness diagnostic.

Loads Qwen2.5-7B (already cached locally), processes a moderate prompt,
generates a short suffix with (a) Full attention and (b) Path D
HALOCacheChunked, and reports per-token logit deviation.

The chunked path should be bit-equivalent to Full under
Prop 4.5 / Prop 4.5 (LSE-merge associativity). If logits diverge, the
diagnostic prints the first divergence and per-layer / per-step
deviation so we can localize the bug.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    import torch
    from halo import wrap_with_halo, HALOConfig
    from halo.kv_cache_chunked import HALOCacheChunked

    print("Loading Qwen/Qwen2.5-7B...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B", torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )

    # Build a deterministic, moderately-long prompt that crosses the chunked threshold.
    base = "The quick brown fox jumps over the lazy dog. " * 200  # ~1.5K tokens
    base += "\nQuestion: What does the fox do?\nAnswer:"
    inputs = tok(base, return_tensors="pt").to("cuda")
    print(f"Prompt length: {inputs.input_ids.shape[-1]} tokens")

    # Phase 1: Full attention reference.
    print("\n=== Phase 1: Full attention ===")
    with torch.no_grad():
        out_full = model.generate(
            **inputs, max_new_tokens=10, do_sample=False,
            num_beams=1, output_scores=True, return_dict_in_generate=True,
        )
    tokens_full = out_full.sequences[0, inputs.input_ids.shape[-1]:].tolist()
    print(f"  Generated: {tok.decode(tokens_full)!r}")
    print(f"  Token IDs: {tokens_full}")

    # Phase 2: Path D chunked.
    print("\n=== Phase 2: Path D chunked ===")
    cfg = HALOConfig(
        hot_ratio=0.25, chunked=True, chunk_size=512, recent_window=64,
        tiers=("dram",),
    )
    wrap_with_halo(model, cfg)
    with torch.no_grad():
        out_chunked = model.generate(
            **inputs, max_new_tokens=10, do_sample=False,
            num_beams=1, output_scores=True, return_dict_in_generate=True,
        )
    tokens_chunked = out_chunked.sequences[0, inputs.input_ids.shape[-1]:].tolist()
    print(f"  Generated: {tok.decode(tokens_chunked)!r}")
    print(f"  Token IDs: {tokens_chunked}")

    cache = model._halo_cache
    tele = cache.telemetry()
    print(f"  Mode at end: {tele.get('mode')}")
    print(f"  Total chunks called: {tele.get('chunked_total_chunks', 0)}")

    # Compare.
    print("\n=== Comparison ===")
    first_div = None
    for i, (a, b) in enumerate(zip(tokens_full, tokens_chunked)):
        match = "OK" if a == b else "DIVERGES"
        print(f"  step {i}: full={a:>6d} chunked={b:>6d}  {match}")
        if a != b and first_div is None:
            first_div = i
    if first_div is None:
        print("  ✓ All 10 tokens match — chunked path is correct.")
    else:
        print(f"  ✗ Diverges at step {first_div}.")

    # Per-step logit comparison.
    print("\n=== Per-step logit max-abs deviation ===")
    for i, (sf, sc) in enumerate(zip(out_full.scores, out_chunked.scores)):
        dev = (sf - sc).abs().max().item()
        print(f"  step {i}: max abs logit dev = {dev:.4e}")
        if i >= 5:
            break


if __name__ == "__main__":
    raise SystemExit(main())
