"""Micro-benchmark: per-call wall-clock of the Path D attention loop,
reference Python path vs fused Triton kernel.

Runs a synthetic 64K cold cache + 64 recent on GPU (no real model;
no SDPA prefill), simulating one decode step's attention call across
N decoder layers. Reports per-call wall-clock and the projected
fraction of the reported 21x infinitebench gap that this kernel
closes if Python overhead is the dominant term.

Usage:
    python scripts/benchmark_triton_chunked.py \
        --T_cold 65536 --n_layers 28 --chunk_size 512 \
        --H 28 --H_kv 4 --D 128 --dtype bf16 \
        --out experiments/triton_kernel_bench.json

The benchmark is hardware-agnostic for the relative comparison
(reference vs triton are run back-to-back on the same GPU); absolute
numbers depend on the GPU, but the ratio is informative.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch


def _dtype_of(name: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16,
            "fp32": torch.float32}[name]


def _make_cache(
    *, T_cold: int, T_recent: int, n_layers: int,
    B: int, H_kv: int, D: int, dtype, device,
    cold_on_host: bool,
):
    """Allocate synthetic K/V caches with the layout the runtime expects."""
    cold_k_layers = []
    cold_v_layers = []
    for _ in range(n_layers):
        if cold_on_host:
            k = torch.randn(B, H_kv, T_cold, D, dtype=dtype,
                            device="cpu", pin_memory=True)
            v = torch.randn(B, H_kv, T_cold, D, dtype=dtype,
                            device="cpu", pin_memory=True)
        else:
            k = torch.randn(B, H_kv, T_cold, D, dtype=dtype, device=device)
            v = torch.randn(B, H_kv, T_cold, D, dtype=dtype, device=device)
        cold_k_layers.append(k)
        cold_v_layers.append(v)
    gpu_k = [torch.randn(B, H_kv, T_recent, D, dtype=dtype, device=device)
             for _ in range(n_layers)]
    gpu_v = [torch.randn(B, H_kv, T_recent, D, dtype=dtype, device=device)
             for _ in range(n_layers)]
    return cold_k_layers, cold_v_layers, gpu_k, gpu_v


def _reference_path_one_layer(
    q, cold_k, cold_v, gpu_k, gpu_v, *, chunk_size, scale, q_pos_offset,
):
    """Reproduce HALOCacheChunked.compute_attention's per-chunk math
    inline (without the telemetry / scorer side-effects) for a clean
    timing comparison.
    """
    T_cold = cold_k.shape[-2]
    T_gpu = gpu_k.shape[-2]
    B, H, _, D = q.shape

    running_out = None
    running_lse = None

    def _chunk(q_, k_, v_, causal):
        H_kv = k_.shape[1]
        rep = H // H_kv
        if rep > 1:
            k_ = k_.repeat_interleave(rep, dim=1)
            v_ = v_.repeat_interleave(rep, dim=1)
        qk = torch.matmul(q_.to(torch.float32),
                          k_.to(torch.float32).transpose(-1, -2)) * scale
        if causal is not None:
            qk = qk + causal
        lse = torch.logsumexp(qk, dim=-1)
        w = (qk - lse.unsqueeze(-1)).exp()
        out = torch.matmul(w, v_.to(torch.float32))
        return out, lse

    def _merge(oa, la, ob, lb):
        lse = torch.logaddexp(la, lb)
        wa = (la - lse).exp().unsqueeze(-1)
        wb = (lb - lse).exp().unsqueeze(-1)
        return wa * oa + wb * ob, lse

    device = q.device
    for c0 in range(0, T_cold, chunk_size):
        c1 = min(c0 + chunk_size, T_cold)
        if c0 > q_pos_offset:
            continue
        k_h = cold_k[..., c0:c1, :].to(device, non_blocking=True)
        v_h = cold_v[..., c0:c1, :].to(device, non_blocking=True)
        out_c, lse_c = _chunk(q, k_h, v_h, None)
        if running_out is None:
            running_out, running_lse = out_c, lse_c
        else:
            running_out, running_lse = _merge(
                running_out, running_lse, out_c, lse_c,
            )

    for c0 in range(0, T_gpu, chunk_size):
        c1 = min(c0 + chunk_size, T_gpu)
        k_h = gpu_k[..., c0:c1, :]
        v_h = gpu_v[..., c0:c1, :]
        out_c, lse_c = _chunk(q, k_h, v_h, None)
        if running_out is None:
            running_out, running_lse = out_c, lse_c
        else:
            running_out, running_lse = _merge(
                running_out, running_lse, out_c, lse_c,
            )
    return running_out.to(q.dtype)


def _triton_path_one_layer(
    q, cold_k, cold_v, gpu_k, gpu_v, *, chunk_size, scale, q_pos_offset,
):
    """Fused Triton path for the same per-layer attention call."""
    from halo.triton_chunked import init_acc, update_chunk, finalize

    device = q.device
    q3 = q.squeeze(2).contiguous()
    B, H, D = q3.shape
    D_v = cold_v.shape[-1]
    m, l, o = init_acc(B=B, H=H, D_v=D_v, device=device, dtype=torch.float32)
    q_pos = torch.full((B,), int(q_pos_offset),
                       device=device, dtype=torch.int64)
    T_cold = cold_k.shape[-2]
    T_gpu = gpu_k.shape[-2]
    for c0 in range(0, T_cold, chunk_size):
        c1 = min(c0 + chunk_size, T_cold)
        if c0 > q_pos_offset:
            continue
        k_h = cold_k[..., c0:c1, :].to(device, non_blocking=True).contiguous()
        v_h = cold_v[..., c0:c1, :].to(device, non_blocking=True).contiguous()
        update_chunk(
            q=q3, k=k_h, v=v_h, m=m, l=l, o=o,
            scale=scale, chunk_start=c0, q_pos=q_pos, apply_causal=True,
        )
        del k_h, v_h
    for c0 in range(0, T_gpu, chunk_size):
        c1 = min(c0 + chunk_size, T_gpu)
        global_c0 = T_cold + c0
        if global_c0 > q_pos_offset:
            continue
        k_h = gpu_k[..., c0:c1, :].contiguous()
        v_h = gpu_v[..., c0:c1, :].contiguous()
        update_chunk(
            q=q3, k=k_h, v=v_h, m=m, l=l, o=o,
            scale=scale, chunk_start=global_c0, q_pos=q_pos, apply_causal=True,
        )
    return finalize(m=m, l=l, o=o, out_dtype=q.dtype).unsqueeze(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T_cold", type=int, default=65536)
    ap.add_argument("--T_recent", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=28)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--H", type=int, default=28)
    ap.add_argument("--H_kv", type=int, default=4)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--dtype", default="bf16",
                    choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--cold_on_host", action="store_true",
                    help="Place cold tier on pinned host DRAM (default).")
    ap.add_argument("--cold_on_gpu", dest="cold_on_host",
                    action="store_false")
    ap.set_defaults(cold_on_host=True)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default="experiments/triton_kernel_bench.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the wall-clock benchmark.")

    dtype = _dtype_of(args.dtype)
    device = torch.device(args.device)

    torch.manual_seed(0)
    print(f"[bench] allocating synthetic cache: "
          f"T_cold={args.T_cold}, T_recent={args.T_recent}, "
          f"n_layers={args.n_layers}, H={args.H}, H_kv={args.H_kv}, D={args.D}, "
          f"dtype={args.dtype}, cold_on_host={args.cold_on_host}")
    cold_k, cold_v, gpu_k, gpu_v = _make_cache(
        T_cold=args.T_cold, T_recent=args.T_recent, n_layers=args.n_layers,
        B=args.B, H_kv=args.H_kv, D=args.D, dtype=dtype, device=device,
        cold_on_host=args.cold_on_host,
    )
    q = torch.randn(args.B, args.H, 1, args.D, device=device, dtype=dtype) * 0.1
    scale = 1.0 / (args.D ** 0.5)
    q_pos_offset = args.T_cold + args.T_recent - 1

    def _run(path_fn, label):
        for _ in range(args.warmup):
            for li in range(args.n_layers):
                path_fn(q, cold_k[li], cold_v[li],
                        gpu_k[li], gpu_v[li],
                        chunk_size=args.chunk_size,
                        scale=scale,
                        q_pos_offset=q_pos_offset)
        torch.cuda.synchronize()
        wall_per_iter = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            for li in range(args.n_layers):
                out = path_fn(q, cold_k[li], cold_v[li],
                              gpu_k[li], gpu_v[li],
                              chunk_size=args.chunk_size,
                              scale=scale,
                              q_pos_offset=q_pos_offset)
            torch.cuda.synchronize()
            wall_per_iter.append(time.perf_counter() - t0)
        mean = sum(wall_per_iter) / len(wall_per_iter)
        print(f"  [{label}] wall per decode step ({args.n_layers} layers): "
              f"{mean * 1000:.1f} ms  (min {min(wall_per_iter) * 1000:.1f} ms)")
        return {"label": label, "mean_s": mean,
                "min_s": min(wall_per_iter), "all_s": wall_per_iter}

    print("[bench] reference path (per-chunk python matmul + logsumexp):")
    ref = _run(_reference_path_one_layer, "reference")
    print("[bench] fused Triton path (one kernel launch per chunk):")
    trt = _run(_triton_path_one_layer, "triton")

    # DMA-overlap variant: only meaningful when cold tier is on host.
    streamed_summary = None
    if args.cold_on_host:
        from halo.triton_chunked import triton_chunked_attention_streamed
        scale_ = scale

        def _streamed_one_layer(
            q_, cold_k_, cold_v_, gpu_k_, gpu_v_, *,
            chunk_size, scale, q_pos_offset,
        ):
            cold_chunks = []
            T_c = cold_k_.shape[-2]
            for c0 in range(0, T_c, chunk_size):
                c1 = min(c0 + chunk_size, T_c)
                cold_chunks.append((cold_k_[..., c0:c1, :],
                                    cold_v_[..., c0:c1, :], c0))
            hot_chunks = []
            T_g = gpu_k_.shape[-2]
            for c0 in range(0, T_g, chunk_size):
                c1 = min(c0 + chunk_size, T_g)
                hot_chunks.append((
                    gpu_k_[..., c0:c1, :].contiguous(),
                    gpu_v_[..., c0:c1, :].contiguous(),
                    T_c + c0,
                ))
            q_pos = torch.full((q_.shape[0],), int(q_pos_offset),
                               device=q_.device, dtype=torch.int64)
            return triton_chunked_attention_streamed(
                q_, cold_chunks, hot_chunks, scale=scale, q_pos=q_pos,
            )

        print("[bench] streamed Triton path (DMA overlaps compute):")
        streamed_summary = _run(_streamed_one_layer, "streamed")

    ratio = ref["mean_s"] / trt["mean_s"] if trt["mean_s"] > 0 else float("inf")
    ratio_s = (ref["mean_s"] / streamed_summary["mean_s"]
               if streamed_summary and streamed_summary["mean_s"] > 0
               else None)
    print(f"\n[bench] speedup (reference / triton): {ratio:.2f}x")
    if ratio_s is not None:
        print(f"[bench] speedup (reference / streamed): {ratio_s:.2f}x")
    print(f"[bench] inferred residual wall vs Full attention if Python-bound:")
    print(f"        if reference is 21x Full, triton is {21 / ratio:.2f}x Full")
    if ratio_s is not None:
        print(f"        if reference is 21x Full, streamed is "
              f"{21 / ratio_s:.2f}x Full")

    out = {
        "args": vars(args),
        "gpu": torch.cuda.get_device_name(device),
        "reference": ref,
        "triton": trt,
        "streamed": streamed_summary,
        "speedup_triton_x": ratio,
        "speedup_streamed_x": ratio_s,
        "extrapolated_residual_x_full_triton": 21.0 / ratio,
        "extrapolated_residual_x_full_streamed": (
            21.0 / ratio_s if ratio_s else None
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
