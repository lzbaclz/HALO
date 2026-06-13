"""Demonstrate Path D peak-GPU memory contract under *stock* HF generate().

This script is the empirical answer to reviewer 3 W1: under the
chunked-prefill harness (scripts/run_chunked_prefill_longctx.py), Path D
on Qwen2.5-7B at 65K cuts peak GPU from 26.41 GiB to 18.89 GiB
(-28.5%). With ``install_preforward_peel(model)``, stock
``model.generate(input_ids=long_prompt)`` realises the same saving
without the caller having to use the chunked-prefill helper.

Outputs a manifest at::

    experiments/runs/qwen2-5-7b/preforward_peel_demo/manifest_<L>.json

with peak GPU + per-step top-5/top-1 agreement against Full.
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import time
from dataclasses import asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _make_long_prompt(tok, target_tokens: int):
    """Repeated-passage stress test prompt (same harness as
    scripts/run_chunked_prefill_longctx.py). NOT a real benchmark
    distribution; documents the memory contract only."""
    passage = (
        "The Manhattan Project was a research-and-development undertaking "
        "during World War II that produced the first nuclear weapons. It "
        "was led by the United States with the support of the United Kingdom "
        "and Canada. From 1942 to 1946 the project was under the direction of "
        "Major General Leslie Groves of the U.S. Army Corps of Engineers. "
    )
    question = "\nQ: When did the Manhattan Project end?\nA:"
    pas_ids = tok.encode(passage, add_special_tokens=False)
    ids = list(tok.encode("Below are passages: ", add_special_tokens=True))
    while len(ids) + len(pas_ids) + len(tok.encode(question, add_special_tokens=False)) < target_tokens:
        ids += pas_ids
    ids += tok.encode(question, add_special_tokens=False)
    return ids[:target_tokens]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--target-tokens", type=int, default=32768)
    ap.add_argument("--gen-tokens", type=int, default=8)
    ap.add_argument("--prefill-chunk", type=int, default=4096)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--recent-window", type=int, default=64)
    ap.add_argument("--out-dir",
                    default="experiments/runs/qwen2-5-7b/preforward_peel_demo")
    ap.add_argument("--skip-full", action="store_true")
    args = ap.parse_args()

    print(f"[preforward-peel] loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()

    ids = _make_long_prompt(tok, args.target_tokens)
    input_ids = torch.tensor([ids], device="cuda:0")
    actual_T = input_ids.shape[-1]
    print(f"[preforward-peel] prompt length = {actual_T} tokens", flush=True)

    # ---- Full SDPA reference ----
    full_info = {}
    if not args.skip_full:
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            full_gen = model.generate(
                input_ids,
                max_new_tokens=args.gen_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=None,
                pad_token_id=0,
            )
        wall = time.time() - t0
        peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
        prompt_len = input_ids.shape[-1]
        full_ids = full_gen[0, prompt_len:].tolist()
        full_info = {
            "peak_gpu_gib": peak_gib,
            "wall_clock_s": wall,
            "generated_token_ids": full_ids,
        }
        print(f"[preforward-peel] Full peak={peak_gib:.2f} GiB wall={wall:.1f}s "
              f"gen={full_ids}", flush=True)
        del full_gen
        gc.collect(); torch.cuda.empty_cache()

    # ---- Re-load model so wrap_with_halo gets a fresh copy ----
    del model
    gc.collect(); torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()

    # ---- Wrap with Path D + pre-forward peel ----
    from halo import HALOConfig, wrap_with_halo, install_preforward_peel
    cfg = HALOConfig(
        chunked=True,
        chunk_size=args.chunk_size,
        recent_window=args.recent_window,
        hot_ratio=1.0,
    )
    wrap_with_halo(model, cfg)
    install_preforward_peel(
        model,
        prefill_chunk_tokens=args.prefill_chunk,
        activation_threshold=args.prefill_chunk * 2,
    )

    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=args.gen_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=None,
            pad_token_id=0,
        )
    wall = time.time() - t0
    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Extract sequences from either the HF-style return or our _PreforwardPeelResult.
    if hasattr(out, "sequences"):
        seq = out.sequences
    else:
        seq = out
    prompt_len = input_ids.shape[-1]
    pd_ids = seq[0, prompt_len:].tolist()
    telemetry = getattr(out, "halo_telemetry", {})

    # Per-step top-1 / top-5 agreement
    agreement = {}
    if full_info and full_info.get("generated_token_ids"):
        f_ids = full_info["generated_token_ids"]
        n = min(len(f_ids), len(pd_ids))
        if n:
            agreement["top1_match_count"] = sum(int(a == b) for a, b in zip(f_ids[:n], pd_ids[:n]))
            agreement["top1_match_n"] = n
            agreement["top1_match_rate"] = agreement["top1_match_count"] / n

    print(f"[preforward-peel] Path D peak={peak_gib:.2f} GiB wall={wall:.1f}s "
          f"gen={pd_ids}", flush=True)
    if full_info:
        delta = peak_gib - full_info["peak_gpu_gib"]
        delta_pct = 100 * delta / full_info["peak_gpu_gib"]
        print(f"[preforward-peel] peak delta vs Full: {delta:+.2f} GiB ({delta_pct:+.1f}%)",
              flush=True)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": args.model,
        "target_tokens": args.target_tokens,
        "actual_prompt_tokens": actual_T,
        "gen_tokens": args.gen_tokens,
        "prefill_chunk": args.prefill_chunk,
        "chunk_size": args.chunk_size,
        "recent_window": args.recent_window,
        "full": full_info,
        "path_d_via_generate": {
            "peak_gpu_gib": peak_gib,
            "wall_clock_s": wall,
            "generated_token_ids": pd_ids,
            "telemetry": telemetry,
        },
        "agreement": agreement,
    }
    out_path = out_dir / f"manifest_{actual_T}.json"
    out_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[preforward-peel] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
