"""Live GPU measurement of HALOCacheChunked peak memory + DMA latency.

This replaces (and supersedes) the older ``measure_chunked_peak.py``,
addressing the reviewer's SHOULD-2 / SHOULD-3: include the actual
H2D round-trip for cold-tier chunks, not just the GPU staging buffer.

We measure two quantities at each context length L:
1. **Peak GPU GiB**: from ``torch.cuda.max_memory_allocated()``
2. **DMA wall-clock per chunk**: time spent in the H2D copy step alone

The setup keeps cold K/V on pinned host DRAM (as the real cache would
in chunked mode), then runs the LSE-merge attention loop. This is a
faithful end-to-end reproduction of the chunked path, sans the
surrounding model.

Output: experiments/measured_peak_memory_dma.json + console summary.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

# Qwen2.5-7B architecture.
ARCH = dict(num_layers=28, num_kv_heads=4, head_dim=128,
            num_q_heads=28, dtype=torch.bfloat16)


def _peak_gib():
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def _reset():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def measure(method: str, T: int, *, r: float = 0.25, chunk_size: int = 512):
    """Run one (method, T) cell on Qwen-shape KV. Returns (peak_gib, wall_s,
    avg_dma_ms_per_chunk)."""
    device = torch.device("cuda")
    L, H_kv, D, H = ARCH["num_layers"], ARCH["num_kv_heads"], ARCH["head_dim"], ARCH["num_q_heads"]
    dtype = ARCH["dtype"]
    B = 1

    _reset()

    if method == "full":
        # All KV on GPU, all layers simultaneously, one SDPA per layer.
        kv_cache = []
        for ell in range(L):
            kv_cache.append((
                torch.randn(B, H_kv, T, D, device=device, dtype=dtype),
                torch.randn(B, H_kv, T, D, device=device, dtype=dtype),
            ))
        torch.cuda.synchronize()
        t0 = time.time()
        for ell in range(L):
            q = torch.randn(B, H, 1, D, device=device, dtype=dtype)
            k, v = kv_cache[ell]
            # Repeat KV heads if GQA.
            if H != H_kv:
                k = k.repeat_interleave(H // H_kv, dim=1)
                v = v.repeat_interleave(H // H_kv, dim=1)
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        wall_s = time.time() - t0
        peak = _peak_gib()
        del kv_cache
        return peak, wall_s, 0.0

    if method == "chunked_dma":
        # Cold KV on pinned host DRAM (the realistic case).
        # Recent on GPU is just the LAST recent_window tokens per layer.
        from halo.kv_cache_chunked import HALOCacheChunked

        recent_window = 64
        cold_T = max(T - recent_window, 0)

        # Allocate pinned host KV for cold tier (per layer).
        cold_k = []
        cold_v = []
        recent_k_gpu = []
        recent_v_gpu = []
        for ell in range(L):
            ck = torch.empty(B, H_kv, cold_T, D, dtype=dtype, pin_memory=True)
            cv = torch.empty(B, H_kv, cold_T, D, dtype=dtype, pin_memory=True)
            ck.normal_()
            cv.normal_()
            cold_k.append(ck)
            cold_v.append(cv)
            recent_k_gpu.append(
                torch.randn(B, H_kv, recent_window, D, device=device, dtype=dtype))
            recent_v_gpu.append(
                torch.randn(B, H_kv, recent_window, D, device=device, dtype=dtype))

        torch.cuda.synchronize()
        t0 = time.time()
        dma_times = []
        for ell in range(L):
            q = torch.randn(B, H, 1, D, device=device, dtype=dtype)

            running_out, running_lse = None, None
            # Iterate cold chunks.
            for c0 in range(0, cold_T, chunk_size):
                c1 = min(c0 + chunk_size, cold_T)
                # Time the H2D copy specifically.
                torch.cuda.synchronize()
                tdma = time.time()
                k_gpu = cold_k[ell][..., c0:c1, :].to(device, non_blocking=True)
                v_gpu = cold_v[ell][..., c0:c1, :].to(device, non_blocking=True)
                torch.cuda.synchronize()
                dma_times.append(time.time() - tdma)

                out_c, lse_c = HALOCacheChunked._chunk_attention(q, k_gpu, v_gpu)
                if running_out is None:
                    running_out, running_lse = out_c, lse_c
                else:
                    running_out, running_lse = HALOCacheChunked._lse_merge(
                        running_out, running_lse, out_c, lse_c)
                del k_gpu, v_gpu

            # Recent (no DMA — already on GPU).
            out_c, lse_c = HALOCacheChunked._chunk_attention(
                q, recent_k_gpu[ell], recent_v_gpu[ell])
            if running_out is None:
                running_out, running_lse = out_c, lse_c
            else:
                running_out, running_lse = HALOCacheChunked._lse_merge(
                    running_out, running_lse, out_c, lse_c)
        torch.cuda.synchronize()
        wall_s = time.time() - t0
        peak = _peak_gib()

        avg_dma_ms = (sum(dma_times) / len(dma_times) * 1000) if dma_times else 0.0
        del cold_k, cold_v, recent_k_gpu, recent_v_gpu
        return peak, wall_s, avg_dma_ms

    raise ValueError(f"unknown method {method}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[4096, 8192, 16384, 32768, 65536])
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--out", default="experiments/measured_peak_memory_dma.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[skip] no CUDA available")
        return 1

    print(f"Device: {torch.cuda.get_device_name(0)}")
    rows = []
    for T in args.lengths:
        print(f"\n=== T = {T} ===")
        for method in ("full", "chunked_dma"):
            try:
                peak, wall_s, dma_ms = measure(method, T, chunk_size=args.chunk_size)
            except torch.cuda.OutOfMemoryError:
                print(f"  {method:14s} T={T:6d}  OOM")
                rows.append({"method": method, "context": T, "peak_gib": None,
                             "wall_s": None, "dma_ms_per_chunk": None,
                             "oom": True})
                _reset()
                continue
            label = method
            print(f"  {label:14s} T={T:6d}  peak={peak:.2f} GiB  "
                  f"wall={wall_s*1000:.1f}ms  dma/chunk={dma_ms:.2f}ms")
            rows.append({"method": method, "context": T, "peak_gib": peak,
                         "wall_s": wall_s, "dma_ms_per_chunk": dma_ms,
                         "chunk_size": args.chunk_size})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "chunk_size": args.chunk_size,
        "rows": rows,
    }, indent=2))
    print(f"\n[ok] wrote {args.out}")

    # Summary: ratio.
    print("\n=== Chunked vs Full (KV-only peak memory) ===")
    for T in args.lengths:
        rf = next((r for r in rows if r["method"] == "full" and r["context"] == T), None)
        rc = next((r for r in rows if r["method"] == "chunked_dma" and r["context"] == T), None)
        if rf and rc and rf["peak_gib"] and rc["peak_gib"]:
            ratio = rf["peak_gib"] / rc["peak_gib"]
            print(f"  T={T:>6d}  full={rf['peak_gib']:.2f}  "
                  f"chunked_dma={rc['peak_gib']:.2f}  ratio={ratio:.1f}x")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
