#!/usr/bin/env python3
"""Benchmark Path D Triton variants at decode-time at multiple context lengths.

Compares three dispatch paths:
  (a) reference per-chunk loop (synchronous DMA + per-chunk Triton launch),
  (b) HALO_TRITON_STREAMED=1 double-buffered async DMA,
  (c) HALO_TRITON_SINGLE_LAUNCH=1 single-coalesced-DMA + single-Triton-launch.

For each path we time the cold-stream forward of one transformer-layer
attention call with realistic shapes (Qwen 2.5-7B GQA 4:1, hot_ratio=0.25,
chunk_size=512). Outputs JSON for paper integration.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

# Allow running from repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _benchmark_one(
    *, T_total: int, hot_ratio: float, chunk_size: int,
    B: int, H: int, H_kv: int, D: int, dtype: torch.dtype, mode: str,
    warmup: int, iters: int,
) -> dict:
    """One configuration × one mode × multiple iterations."""
    from halo.triton_chunked import (
        init_acc, update_chunk, finalize,
        triton_chunked_attention_streamed,
        triton_chunked_attention_single_launch,
    )

    device = torch.device("cuda")
    T_hot = int(T_total * hot_ratio)
    T_cold = T_total - T_hot

    # Synthetic KV in pinned host memory (for cold) and on GPU (for hot).
    cold_k = torch.empty((B, H_kv, T_cold, D), dtype=dtype, pin_memory=True)
    cold_v = torch.empty((B, H_kv, T_cold, D), dtype=dtype, pin_memory=True)
    cold_k.normal_(mean=0.0, std=0.1)
    cold_v.normal_(mean=0.0, std=0.1)
    hot_k = torch.randn(B, H_kv, T_hot, D, device=device, dtype=dtype) * 0.1
    hot_v = torch.randn(B, H_kv, T_hot, D, device=device, dtype=dtype) * 0.1

    q = torch.randn(B, H, 1, D, device=device, dtype=dtype) * 0.1
    q3 = q.squeeze(2).contiguous()
    q_pos = torch.full((B,), T_total - 1, device=device, dtype=torch.int64)
    scale = 1.0 / math.sqrt(D)

    def run_reference():
        # Synchronous: per-chunk DMA then per-chunk Triton launch.
        m, l, o = init_acc(B=B, H=H, D_v=D, device=device, dtype=torch.float32)
        for c0 in range(0, T_cold, chunk_size):
            c1 = min(c0 + chunk_size, T_cold)
            k_h = cold_k[..., c0:c1, :].to(device, non_blocking=True).contiguous()
            v_h = cold_v[..., c0:c1, :].to(device, non_blocking=True).contiguous()
            update_chunk(
                q=q3, k=k_h, v=v_h, m=m, l=l, o=o,
                scale=scale, chunk_start=c0, q_pos=q_pos,
                apply_causal=True,
            )
            del k_h, v_h
        for c0 in range(0, T_hot, chunk_size):
            c1 = min(c0 + chunk_size, T_hot)
            update_chunk(
                q=q3, k=hot_k[..., c0:c1, :].contiguous(),
                v=hot_v[..., c0:c1, :].contiguous(),
                m=m, l=l, o=o, scale=scale,
                chunk_start=T_cold + c0, q_pos=q_pos,
                apply_causal=True,
            )
        return finalize(m=m, l=l, o=o, out_dtype=q.dtype).unsqueeze(2)

    def run_streamed():
        cold_chunks = []
        for c0 in range(0, T_cold, chunk_size):
            c1 = min(c0 + chunk_size, T_cold)
            cold_chunks.append((cold_k[..., c0:c1, :], cold_v[..., c0:c1, :], c0))
        hot_chunks = []
        for c0 in range(0, T_hot, chunk_size):
            c1 = min(c0 + chunk_size, T_hot)
            hot_chunks.append((hot_k[..., c0:c1, :].contiguous(),
                               hot_v[..., c0:c1, :].contiguous(),
                               T_cold + c0))
        return triton_chunked_attention_streamed(
            q, cold_chunks, hot_chunks,
            scale=scale, q_pos=q_pos,
        )

    def run_single_launch():
        return triton_chunked_attention_single_launch(
            q, cold_k_host=cold_k, cold_v_host=cold_v,
            hot_k_gpu=hot_k, hot_v_gpu=hot_v,
            scale=scale, q_pos=q_pos, apply_causal=True,
        )

    fn = {"reference": run_reference, "streamed": run_streamed,
          "single_launch": run_single_launch}[mode]

    # Warmup
    for _ in range(warmup):
        _ = fn()
        torch.cuda.synchronize()

    # Timed iterations using CUDA events for accuracy.
    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = fn()
        stop.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(stop))

    return {
        "mode": mode,
        "T_total": T_total,
        "T_cold": T_cold,
        "T_hot": T_hot,
        "iters": iters,
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "stdev_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, nargs="+",
                    default=[8192, 16384, 32768, 65536])
    ap.add_argument("--hot-ratio", type=float, default=0.25)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=28,
                    help="Qwen2.5-7B has 28 query heads")
    ap.add_argument("--H-kv", type=int, default=4,
                    help="Qwen2.5-7B has 4 KV heads (GQA 7:1)")
    ap.add_argument("--D", type=int, default=128,
                    help="head_dim")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--modes", nargs="+",
                    default=["reference", "streamed", "single_launch"])
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("Skipping: no CUDA", file=sys.stderr)
        return
    try:
        import triton  # noqa: F401
    except ImportError:
        print("Skipping: no Triton", file=sys.stderr)
        return

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]
    device = torch.device("cuda")
    torch.cuda.empty_cache()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"shapes: B={args.B} H={args.H} H_kv={args.H_kv} D={args.D} "
          f"dtype={args.dtype} chunk_size={args.chunk_size} "
          f"hot_ratio={args.hot_ratio}")

    results = []
    for T in args.T:
        for mode in args.modes:
            torch.cuda.empty_cache()
            gc.collect()
            r = _benchmark_one(
                T_total=T, hot_ratio=args.hot_ratio,
                chunk_size=args.chunk_size,
                B=args.B, H=args.H, H_kv=args.H_kv, D=args.D, dtype=dtype,
                mode=mode, warmup=args.warmup, iters=args.iters,
            )
            results.append(r)
            print(f"  T={T:>6} mode={mode:>13}  "
                  f"median={r['median_ms']:7.3f} ms  "
                  f"min/max={r['min_ms']:.2f}/{r['max_ms']:.2f}  "
                  f"stdev={r['stdev_ms']:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "gpu": torch.cuda.get_device_name(0),
        "config": vars(args),
        "results": results,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out}")

    # Print speedup ratios at each T for paper consumption.
    print("\n=== Speedup ratios (streamed and single_launch vs. reference) ===")
    for T in args.T:
        ref = next((r for r in results if r["T_total"] == T and r["mode"] == "reference"), None)
        if ref is None:
            continue
        for mode in ["streamed", "single_launch"]:
            r = next((x for x in results if x["T_total"] == T and x["mode"] == mode), None)
            if r is None:
                continue
            spd = ref["median_ms"] / r["median_ms"]
            saving = (1 - r["median_ms"] / ref["median_ms"]) * 100
            print(f"  T={T:>6} {mode:>13}: {spd:.2f}x speedup ({saving:+.1f}% wall)")


if __name__ == "__main__":
    main()
