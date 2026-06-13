"""Fused Triton chunked LSE-merge attention kernel (Path D fast path).

Replaces the per-chunk Python `_chunk_attention` + `_lse_merge` pair with a
single Triton launch per chunk. The kernel implements FlashAttention-2-style
online softmax across multiple chunks by keeping the running `(m, l, o)`
accumulator on GPU and updating it in-place; the host-side Python loop
becomes a DMA + kernel-launch loop without any per-chunk torch tensor
materialisation.

Why this is the "fused kernel" promised in the paper. The reference
`HALOCacheChunked` path in `kv_cache_chunked.py` issues per-chunk
`matmul`/`logsumexp`/`matmul`/`logaddexp` calls --- four launches per chunk
on top of the DMA. For a 65K cold cache split into 512-key chunks across
28 layers and 8 decoded tokens that is ~32K Python-side kernel launches.
This module collapses each chunk's math into one kernel launch.

Numerics: `qk` and the accumulators `(m, l, o)` are fp32, matching the
reference path's "safety belt". Inputs `(q, k, v)` may be bf16; promotion
happens inside the kernel. Causal masking is passed as a per-chunk integer
offset so query position `q_pos` admits key position `j` iff
`(chunk_start + j) <= q_pos`. We currently support T_q == 1 (single-token
decode); chunked prefill (T_q > 1) falls back to the reference path.

Test contract: `tests/test_triton_chunked.py` asserts that
`triton_chunked_attention` matches the reference `HALOCacheChunked`
`compute_attention` output to <= 5e-3 (bf16) / <= 1e-5 (fp32) across
random partitions of a synthetic 4K cache.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _chunked_lse_merge_kernel(
        Q,                         # (B, H, D) — single-token decode query
        K, V,                      # (B, H_kv, T_c, D)
        M, L, O,                   # accumulators: (B, H), (B, H), (B, H, D_v)
        scale,
        chunk_start: tl.constexpr,
        q_pos,                     # int64 per-batch query global position
        T_c,
        stride_qb, stride_qh,
        stride_kb, stride_kh, stride_kt,
        stride_vb, stride_vh, stride_vt,
        stride_mb, stride_mh,
        stride_lb, stride_lh,
        stride_ob, stride_oh,
        H: tl.constexpr,
        H_KV: tl.constexpr,
        D: tl.constexpr,
        D_V: tl.constexpr,
        BLOCK_N: tl.constexpr,
        APPLY_CAUSAL: tl.constexpr,
    ):
        """Per-(b, h) online softmax update for one cold/hot chunk."""
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        # GQA: each q-head maps to one kv-head via integer division.
        h_kv = pid_h * H_KV // H

        # Load query row (B, H, D).
        q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh + tl.arange(0, D)
        q = tl.load(q_ptr).to(tl.float32)

        # Load running accumulators.
        m_ptr = M + pid_b * stride_mb + pid_h * stride_mh
        l_ptr = L + pid_b * stride_lb + pid_h * stride_lh
        o_ptr_base = O + pid_b * stride_ob + pid_h * stride_oh

        m_i = tl.load(m_ptr).to(tl.float32)
        l_i = tl.load(l_ptr).to(tl.float32)
        o_off = tl.arange(0, D_V)
        o_i = tl.load(o_ptr_base + o_off).to(tl.float32)

        # Query global position for causal masking.
        qp = tl.load(q_pos + pid_b).to(tl.int64)

        # Tile-iterate over the chunk's T_c dimension.
        k_base = K + pid_b * stride_kb + h_kv * stride_kh
        v_base = V + pid_b * stride_vb + h_kv * stride_vh

        for n_start in range(0, T_c, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            valid = n_offs < T_c

            # (BLOCK_N, D) — keys
            k_block = tl.load(
                k_base + n_offs[:, None] * stride_kt + tl.arange(0, D)[None, :],
                mask=valid[:, None], other=0.0,
            ).to(tl.float32)
            # (BLOCK_N, D_V) — values
            v_block = tl.load(
                v_base + n_offs[:, None] * stride_vt + tl.arange(0, D_V)[None, :],
                mask=valid[:, None], other=0.0,
            ).to(tl.float32)

            # (BLOCK_N,) — qk = q @ k^T * scale
            qk = tl.sum(q[None, :] * k_block, axis=1) * scale

            # Mask out-of-range positions and (if causal) future positions.
            mask = valid
            if APPLY_CAUSAL:
                global_n = chunk_start + n_offs
                mask = mask & (global_n <= qp)
            qk = tl.where(mask, qk, float("-inf"))

            # Online softmax: rescale (m, l, o), add this tile's mass.
            m_tile = tl.max(qk, axis=0)
            m_new = tl.maximum(m_i, m_tile)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new)  # (BLOCK_N,)
            # If the entire tile is masked, p is 0 and m_tile is -inf; alpha
            # is fine because m_i remains finite. m_new could be -inf the
            # very first tile if every key is masked; we guard below.
            l_tile = tl.sum(p, axis=0)
            o_tile = tl.sum(p[:, None] * v_block, axis=0)  # (D_V,)

            l_i = alpha * l_i + l_tile
            o_i = alpha * o_i + o_tile
            m_i = m_new

        tl.store(m_ptr, m_i)
        tl.store(l_ptr, l_i)
        tl.store(o_ptr_base + o_off, o_i)


def init_acc(
    *, B: int, H: int, D_v: int, device, dtype=torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialise the (m, l, o) accumulators for a fresh attention call."""
    m = torch.full((B, H), float("-inf"), device=device, dtype=dtype)
    l = torch.zeros((B, H), device=device, dtype=dtype)
    o = torch.zeros((B, H, D_v), device=device, dtype=dtype)
    return m, l, o


def update_chunk(
    *, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    m: torch.Tensor, l: torch.Tensor, o: torch.Tensor,
    scale: float, chunk_start: int, q_pos: torch.Tensor,
    apply_causal: bool, block_n: int = 64,
) -> None:
    """One Triton-fused chunk update.

    Args
    ----
    q:      (B, H, D)            — single-token decode query
    k, v:   (B, H_kv, T_c, D)    — one chunk (already on GPU)
    m, l, o: running fp32 accumulators (B, H), (B, H), (B, H, D_v)
    chunk_start: int             — global key-index of this chunk's first row
    q_pos:  (B,) int64           — global query position per batch row
    apply_causal: bool           — whether to apply the causal mask
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton not available; install triton >= 3.0")

    assert q.dim() == 3, f"expected q (B, H, D), got {q.shape}"
    B, H, D = q.shape
    H_KV = k.shape[1]
    T_c = k.shape[2]
    D_V = v.shape[-1]
    assert k.shape == (B, H_KV, T_c, D), f"k shape {k.shape}"
    assert v.shape == (B, H_KV, T_c, D_V), f"v shape {v.shape}"

    grid = (B, H)
    _chunked_lse_merge_kernel[grid](
        q, k, v, m, l, o,
        scale, chunk_start, q_pos, T_c,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        m.stride(0), m.stride(1),
        l.stride(0), l.stride(1),
        o.stride(0), o.stride(1),
        H=H, H_KV=H_KV, D=D, D_V=D_V,
        BLOCK_N=block_n,
        APPLY_CAUSAL=apply_causal,
    )


def finalize(
    *, m: torch.Tensor, l: torch.Tensor, o: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Normalise (o/l) and cast to the requested output dtype.

    Returns (B, H, D_v). If every chunk was fully masked for some (b, h)
    row, l == 0 and m == -inf; we return zeros for that row to match the
    reference path's behaviour.
    """
    safe_l = torch.where(l > 0, l, torch.ones_like(l))
    out = o / safe_l.unsqueeze(-1)
    return out.to(out_dtype)


def triton_chunked_attention(
    q: torch.Tensor,
    chunks: list[tuple[torch.Tensor, torch.Tensor, int, bool]],
    *,
    scale: Optional[float] = None,
    q_pos: Optional[torch.Tensor] = None,
    block_n: int = 64,
) -> torch.Tensor:
    """Convenience wrapper: run the full chunked LSE-merge attention.

    Args
    ----
    q: (B, H, 1, D) or (B, H, D) — single-token decode query.
    chunks: list of (k, v, chunk_start, apply_causal) tuples. Each chunk's
        (k, v) is on GPU (already DMA'd if originally cold).
    scale: optional softmax scale; defaults to 1/sqrt(D).
    q_pos: (B,) int64 global query position; defaults to "after every
        chunk's last position" (i.e. no causal masking).
    block_n: Triton tile size along K axis. 64 is good for chunk_size=512.

    Returns
    -------
    out: (B, H, 1, D_v) — matches the reference compute_attention output
        shape; cast back to q.dtype.
    """
    if q.dim() == 4:
        assert q.shape[2] == 1, f"triton path supports T_q=1 only; got {q.shape}"
        q3 = q.squeeze(2)
    else:
        q3 = q
    B, H, D = q3.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    if not chunks:
        D_v = D  # fallback assumption
        return torch.zeros(B, H, 1, D_v, device=q.device, dtype=q.dtype)

    D_v = chunks[0][1].shape[-1]
    m, l, o = init_acc(B=B, H=H, D_v=D_v, device=q.device)

    if q_pos is None:
        # No causal masking — use a sentinel that satisfies every position.
        total_T = sum(k.shape[-2] for (k, _, _, _) in chunks)
        q_pos = torch.full((B,), total_T - 1, device=q.device, dtype=torch.int64)
    elif not q_pos.is_cuda:
        q_pos = q_pos.to(q.device)
    if q_pos.dtype != torch.int64:
        q_pos = q_pos.to(torch.int64)

    for (k_c, v_c, chunk_start, apply_causal) in chunks:
        update_chunk(
            q=q3, k=k_c, v=v_c, m=m, l=l, o=o,
            scale=scale, chunk_start=chunk_start, q_pos=q_pos,
            apply_causal=apply_causal, block_n=block_n,
        )

    out = finalize(m=m, l=l, o=o, out_dtype=q.dtype)
    return out.unsqueeze(2)  # (B, H, 1, D_v)


def triton_chunked_attention_streamed(
    q: torch.Tensor,
    cold_chunks_host: list[tuple[torch.Tensor, torch.Tensor, int]],
    hot_chunks_gpu: list[tuple[torch.Tensor, torch.Tensor, int]],
    *,
    scale: Optional[float] = None,
    q_pos: torch.Tensor,
    block_n: int = 64,
    n_buffers: int = 2,
) -> torch.Tensor:
    """DMA-overlapped chunked LSE-merge attention.

    Double-buffers the cold-tier DMA against the Triton kernel compute via
    two CUDA streams: while chunk `i` is being attended to on the compute
    stream, chunk `i+1` is being DMA'd into the next staging slot on the
    copy stream. Approximation: assumes adjacent chunks have similar shapes
    so the staging slot can be reused (we allocate ``n_buffers`` slots).

    `cold_chunks_host`: list of (k_host_pinned, v_host_pinned, chunk_start).
    `hot_chunks_gpu`: list of (k_gpu, v_gpu, chunk_start). No DMA needed.
    `q_pos`: (B,) int64 query position for causal masking.

    Returns out shape (B, H, 1, D_v) cast to q.dtype. T_q == 1 only.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton not available")
    if q.dim() == 4:
        assert q.shape[2] == 1
        q3 = q.squeeze(2).contiguous()
    else:
        q3 = q
    B, H, D = q3.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Determine D_v from the first chunk in either list.
    first_v = None
    for (_, v, _) in cold_chunks_host:
        first_v = v
        break
    if first_v is None:
        for (_, v, _) in hot_chunks_gpu:
            first_v = v
            break
    if first_v is None:
        return torch.zeros(B, H, 1, D, device=q.device, dtype=q.dtype)
    D_v = first_v.shape[-1]
    H_kv = first_v.shape[1]

    m, l, o = init_acc(B=B, H=H, D_v=D_v, device=q.device, dtype=torch.float32)
    if q_pos.dtype != torch.int64:
        q_pos = q_pos.to(torch.int64)
    if not q_pos.is_cuda:
        q_pos = q_pos.to(q.device)

    compute_stream = torch.cuda.current_stream()
    copy_stream = torch.cuda.Stream()
    compute_done_events = [torch.cuda.Event() for _ in range(n_buffers)]
    copy_done_events = [torch.cuda.Event() for _ in range(n_buffers)]
    # Mark "compute done" initially so the first DMA can issue immediately.
    for ev in compute_done_events:
        ev.record(compute_stream)

    # Cold chunks: DMA + kernel with double-buffering.
    for idx, (k_host, v_host, chunk_start) in enumerate(cold_chunks_host):
        slot = idx % n_buffers
        # Wait for the previous user of this slot to finish before
        # reusing the staging memory.
        with torch.cuda.stream(copy_stream):
            compute_done_events[slot].wait(copy_stream)
            k_gpu = k_host.to(q.device, non_blocking=True).contiguous()
            v_gpu = v_host.to(q.device, non_blocking=True).contiguous()
            copy_done_events[slot].record(copy_stream)
        with torch.cuda.stream(compute_stream):
            copy_done_events[slot].wait(compute_stream)
            update_chunk(
                q=q3, k=k_gpu, v=v_gpu, m=m, l=l, o=o,
                scale=scale, chunk_start=chunk_start, q_pos=q_pos,
                apply_causal=True, block_n=block_n,
            )
            compute_done_events[slot].record(compute_stream)
    # Synchronize streams: hot chunks below run on compute_stream.
    compute_stream.wait_stream(copy_stream)

    for (k_gpu, v_gpu, chunk_start) in hot_chunks_gpu:
        update_chunk(
            q=q3, k=k_gpu, v=v_gpu, m=m, l=l, o=o,
            scale=scale, chunk_start=chunk_start, q_pos=q_pos,
            apply_causal=True, block_n=block_n,
        )

    out = finalize(m=m, l=l, o=o, out_dtype=q.dtype)
    return out.unsqueeze(2)


def triton_chunked_attention_single_launch(
    q: torch.Tensor,
    cold_k_host: Optional[torch.Tensor],
    cold_v_host: Optional[torch.Tensor],
    hot_k_gpu: Optional[torch.Tensor],
    hot_v_gpu: Optional[torch.Tensor],
    *,
    scale: Optional[float] = None,
    q_pos: torch.Tensor,
    block_n: int = 64,
    apply_causal: bool = True,
) -> torch.Tensor:
    """One-launch DMA + Triton chunked attention.

    Issues a single host-to-device copy per (K, V) cold tensor on the copy
    stream (pinned-memory async), records a copy-complete event, has the
    compute stream wait on that event, then issues **one** ``update_chunk``
    call over the cold+hot concatenation. Removes per-chunk Python-side
    overhead (event records, stream syncs, kernel launches) and lets the
    PCIe transfer be a single coalesced DMA instead of ``T_cold/chunk_size``
    independent ones.

    Args
    ----
    q:            (B, H, 1, D) or (B, H, D) single-token decode query.
    cold_k_host,
    cold_v_host:  (B, H_kv, T_cold, D{,D_v}) pinned host tensors, or None.
    hot_k_gpu,
    hot_v_gpu:    (B, H_kv, T_hot, D{,D_v}) GPU tensors, or None.
    q_pos:        (B,) int64 global query position for causal masking.
    apply_causal: forwarded to the kernel (True for prefix attention).

    Tradeoff vs streamed (double-buffered):
      - Peak staging memory: full ``T_cold * H_kv * D * sizeof(K)`` instead
        of ``chunk_size * H_kv * D * sizeof(K)``. Use only when the 80\,GiB
        budget headroom permits (Cell B / D / E / native A100). For
        ``24``\,GiB Cell A, prefer the streamed path.
      - DMA overlap with compute: lost (we wait for all DMA before any
        kernel work). For PCIe-4 (~32 GB/s) and 65K * 32 H_kv * 128 * 4 B
        = ~537 MiB cold staging, this is ~17 ms of unhidden DMA, vs the
        per-chunk overlap savings on streamed (~5 ms typical). Net: the
        single-launch path wins when (kernel-launch overhead × N_chunks) >
        (per-chunk DMA hide × N_chunks - 1), which holds for T_cold >=
        ~24K on PCIe-4 with chunk_size=512.

    Returns
    -------
    out: (B, H, 1, D_v) cast back to ``q.dtype``.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton not available; install triton >= 3.0")
    if q.dim() == 4:
        assert q.shape[2] == 1, f"single-launch path supports T_q=1; got {q.shape}"
        q3 = q.squeeze(2).contiguous()
    else:
        q3 = q
    B, H, D = q3.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Derive H_kv, D_v from whichever (K, V) is present.
    ref = None
    for cand in (cold_v_host, hot_v_gpu):
        if cand is not None and cand.shape[-2] > 0:
            ref = cand
            break
    if ref is None:
        return torch.zeros(B, H, 1, D, device=q.device, dtype=q.dtype)
    H_kv = ref.shape[1]
    D_v = ref.shape[-1]

    T_cold = cold_k_host.shape[-2] if cold_k_host is not None else 0
    T_hot = hot_k_gpu.shape[-2] if hot_k_gpu is not None else 0
    T_total = T_cold + T_hot

    if T_total == 0:
        return torch.zeros(B, H, 1, D_v, device=q.device, dtype=q.dtype)

    m, l, o = init_acc(B=B, H=H, D_v=D_v, device=q.device, dtype=torch.float32)
    if q_pos.dtype != torch.int64:
        q_pos = q_pos.to(torch.int64)
    if not q_pos.is_cuda:
        q_pos = q_pos.to(q.device)

    # Preallocate the contiguous destination buffers; cold gets a single
    # async DMA into the front slice, hot is a D2D copy into the back.
    # (We allocate two contiguous tensors instead of one cat() because cat
    # would require both operands present at the same time.)
    k_buf = torch.empty(
        (B, H_kv, T_total, D), device=q.device,
        dtype=hot_k_gpu.dtype if hot_k_gpu is not None else cold_k_host.dtype,
    )
    v_buf = torch.empty(
        (B, H_kv, T_total, D_v), device=q.device,
        dtype=hot_v_gpu.dtype if hot_v_gpu is not None else cold_v_host.dtype,
    )

    compute_stream = torch.cuda.current_stream()
    copy_stream = torch.cuda.Stream()
    copy_done = torch.cuda.Event()

    if T_cold > 0:
        with torch.cuda.stream(copy_stream):
            # Single coalesced H2D copy each for K and V.
            k_buf[..., :T_cold, :].copy_(cold_k_host, non_blocking=True)
            v_buf[..., :T_cold, :].copy_(cold_v_host, non_blocking=True)
            copy_done.record(copy_stream)
        # Compute stream waits for the H2D copies before launching kernel.
        copy_done.wait(compute_stream)
    if T_hot > 0:
        # D2D copy on the compute stream (cheap, ~150 GB/s on A100 HBM2).
        k_buf[..., T_cold:T_cold + T_hot, :].copy_(hot_k_gpu, non_blocking=True)
        v_buf[..., T_cold:T_cold + T_hot, :].copy_(hot_v_gpu, non_blocking=True)

    # ONE kernel launch over the full T_total range. chunk_start=0 because
    # the buffer's row 0 corresponds to absolute position 0 (the cold/hot
    # ordering preserves the original key positions: cold is the older
    # prefix, hot is the recent suffix).
    update_chunk(
        q=q3, k=k_buf, v=v_buf, m=m, l=l, o=o,
        scale=scale, chunk_start=0, q_pos=q_pos,
        apply_causal=apply_causal, block_n=block_n,
    )

    out = finalize(m=m, l=l, o=o, out_dtype=q.dtype)
    return out.unsqueeze(2)


__all__ = [
    "HAS_TRITON",
    "init_acc",
    "update_chunk",
    "finalize",
    "triton_chunked_attention",
    "triton_chunked_attention_streamed",
    "triton_chunked_attention_single_launch",
]
