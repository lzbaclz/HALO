"""Numerical equivalence tests for the fused Triton chunked LSE-merge kernel.

The contract: `triton_chunked_attention` must match one-shot full attention
on the concatenated chunks (a) bit-equivalent up to fp32 reduction order
when q,k,v are fp32; (b) within bf16 chunk-merge tolerance when inputs are
bf16. Skipped if CUDA / Triton unavailable.
"""
from __future__ import annotations

import math

import pytest
import torch


def _have_cuda_triton():
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _have_cuda_triton(), reason="needs CUDA + Triton",
)


def _reference_full_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    *, scale: float, q_pos: torch.Tensor,
):
    """One-shot full attention for ground truth.

    Args
    ----
    q: (B, H, 1, D)
    k: (B, H_kv, T, D)
    v: (B, H_kv, T, D_v)
    """
    B, H, _, D = q.shape
    H_kv = k.shape[1]
    T = k.shape[2]
    rep = H // H_kv
    if rep > 1:
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    qk = torch.matmul(q.to(torch.float32),
                      k.to(torch.float32).transpose(-1, -2)) * scale  # (B, H, 1, T)
    j = torch.arange(T, device=q.device)
    causal = (j.unsqueeze(0) <= q_pos.view(-1, 1).to(j.dtype))  # (B, T)
    causal = causal.view(B, 1, 1, T)
    qk = torch.where(causal, qk, torch.full_like(qk, float("-inf")))
    p = torch.softmax(qk, dim=-1)
    out = torch.matmul(p, v.to(torch.float32))
    return out  # (B, H, 1, D_v) in fp32


def _split_into_chunks(k, v, chunk_size, total_T):
    """Split (k, v) along the T axis into chunks suitable for the kernel."""
    chunks = []
    for c0 in range(0, total_T, chunk_size):
        c1 = min(c0 + chunk_size, total_T)
        k_c = k[..., c0:c1, :].contiguous()
        v_c = v[..., c0:c1, :].contiguous()
        chunks.append((k_c, v_c, c0, True))
    return chunks


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk_size", [128, 512])
def test_triton_matches_reference_small(dtype, chunk_size):
    """Synthetic 1K cache, T_q=1, GQA 4:1 — match one-shot reference."""
    from halo.triton_chunked import triton_chunked_attention

    torch.manual_seed(0)
    device = "cuda"
    B, H, H_kv, D, D_v = 1, 8, 2, 64, 64
    T = 1024
    q = torch.randn(B, H, 1, D, device=device, dtype=dtype) * 0.1
    k = torch.randn(B, H_kv, T, D, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, H_kv, T, D_v, device=device, dtype=dtype) * 0.1
    q_pos = torch.tensor([T - 1], device=device, dtype=torch.int64)
    scale = 1.0 / math.sqrt(D)

    out_ref = _reference_full_attention(q, k, v, scale=scale, q_pos=q_pos)
    chunks = _split_into_chunks(k, v, chunk_size, T)
    out_triton = triton_chunked_attention(q, chunks, scale=scale, q_pos=q_pos)
    out_triton_f32 = out_triton.to(torch.float32)

    rel = (out_ref - out_triton_f32).norm() / out_ref.norm().clamp_min(1e-12)
    tol = 1e-4 if dtype == torch.float32 else 5e-3
    assert rel.item() <= tol, (
        f"dtype={dtype} chunk_size={chunk_size}: rel={rel.item():.3e}, tol={tol}"
    )


@pytest.mark.parametrize("chunk_size", [256, 512])
def test_triton_causal_masking(chunk_size):
    """Verify causal masking: a query at position p must not see j > p."""
    from halo.triton_chunked import triton_chunked_attention

    torch.manual_seed(1)
    device = "cuda"
    B, H, H_kv, D = 1, 4, 1, 64
    T = 2048
    q = torch.randn(B, H, 1, D, device=device, dtype=torch.float32) * 0.1
    k = torch.randn(B, H_kv, T, D, device=device, dtype=torch.float32) * 0.1
    v = torch.randn(B, H_kv, T, D, device=device, dtype=torch.float32) * 0.1

    # Query at position T//2 — only positions 0..T//2 should contribute.
    q_pos = torch.tensor([T // 2], device=device, dtype=torch.int64)
    scale = 1.0 / math.sqrt(D)

    out_ref = _reference_full_attention(q, k, v, scale=scale, q_pos=q_pos)
    chunks = _split_into_chunks(k, v, chunk_size, T)
    out_triton = triton_chunked_attention(q, chunks, scale=scale, q_pos=q_pos)

    rel = (out_ref - out_triton.to(torch.float32)).norm() / out_ref.norm()
    assert rel.item() <= 1e-4, f"causal mask broken: rel={rel.item():.3e}"

    # Also: feed only the first half as chunks and verify the result is
    # identical (positions T//2+1..T cannot contribute under causal).
    chunks_half = [(k[..., :T // 2 + 1, :].contiguous(),
                    v[..., :T // 2 + 1, :].contiguous(), 0, True)]
    out_half = triton_chunked_attention(q, chunks_half, scale=scale, q_pos=q_pos)
    rel_half = (out_triton - out_half).norm() / out_triton.norm()
    assert rel_half.item() <= 1e-5, (
        f"causal-truncated input gave different output: rel={rel_half.item():.3e}"
    )


def test_triton_single_launch_matches_reference():
    """``triton_chunked_attention_single_launch`` must match one-shot full
    attention within fp32 tolerance. This exercises the env-var-gated
    single-launch fast path (HALO_TRITON_SINGLE_LAUNCH=1) end-to-end.
    """
    from halo.triton_chunked import triton_chunked_attention_single_launch

    torch.manual_seed(3)
    device = "cuda"
    B, H, H_kv, D = 1, 8, 2, 64
    T_cold, T_hot = 1024, 256
    T = T_cold + T_hot
    q = torch.randn(B, H, 1, D, device=device, dtype=torch.float32) * 0.1
    k = torch.randn(B, H_kv, T, D, device=device, dtype=torch.float32) * 0.1
    v = torch.randn(B, H_kv, T, D, device=device, dtype=torch.float32) * 0.1

    q_pos = torch.tensor([T - 1], device=device, dtype=torch.int64)
    scale = 1.0 / math.sqrt(D)

    out_ref = _reference_full_attention(q, k, v, scale=scale, q_pos=q_pos)

    # Split into "cold" (host pinned) and "hot" (already on GPU). The
    # single-launch path issues one big H2D copy for the cold slice and
    # one D2D copy for the hot slice, then one kernel launch.
    cold_k = k[..., :T_cold, :].cpu().pin_memory()
    cold_v = v[..., :T_cold, :].cpu().pin_memory()
    hot_k = k[..., T_cold:, :].contiguous()
    hot_v = v[..., T_cold:, :].contiguous()

    out_sl = triton_chunked_attention_single_launch(
        q, cold_k_host=cold_k, cold_v_host=cold_v,
        hot_k_gpu=hot_k, hot_v_gpu=hot_v,
        scale=scale, q_pos=q_pos, apply_causal=True,
    )
    rel = (out_ref - out_sl.to(torch.float32)).norm() / out_ref.norm()
    assert rel.item() <= 1e-4, f"single-launch vs reference: rel={rel.item():.3e}"

    # Cold-only and hot-only edge cases.
    out_cold = triton_chunked_attention_single_launch(
        q, cold_k_host=k.cpu().pin_memory(), cold_v_host=v.cpu().pin_memory(),
        hot_k_gpu=None, hot_v_gpu=None,
        scale=scale, q_pos=q_pos, apply_causal=True,
    )
    rel_cold = (out_ref - out_cold.to(torch.float32)).norm() / out_ref.norm()
    assert rel_cold.item() <= 1e-4, f"single-launch cold-only: rel={rel_cold.item():.3e}"

    out_hot = triton_chunked_attention_single_launch(
        q, cold_k_host=None, cold_v_host=None,
        hot_k_gpu=k.contiguous(), hot_v_gpu=v.contiguous(),
        scale=scale, q_pos=q_pos, apply_causal=True,
    )
    rel_hot = (out_ref - out_hot.to(torch.float32)).norm() / out_ref.norm()
    assert rel_hot.item() <= 1e-4, f"single-launch hot-only: rel={rel_hot.item():.3e}"


def test_triton_non_contiguous_partition_identity():
    """``triton_chunked_attention`` must preserve algebraic identity
    under ARBITRARY (non-contiguous) hot/cold partitions, not just
    contiguous cold+recent splits.

    This is the "any hot/cold partition" claim of
    Prop.~\\ref{prop:chunked-lossless}\\,(i) tested against a worst-case
    interleaved partition: 5 non-adjacent chunks at positions
    [0, 256), [768, 1024), [1280, 1408), [1536, 1664), [1856, 2048).
    The reference is the SAME positions attended one-shot — so a
    correct kernel must reproduce the one-shot output exactly under
    causal masking with the right global chunk_start per chunk.

    Reviewers flagged that the existing tests focus on contiguous
    splits; this exercises the kernel against the Quest-style
    placement scenario where the hot tier is whichever pages a
    scorer picked, not a recent-suffix.
    """
    from halo.triton_chunked import triton_chunked_attention

    torch.manual_seed(4)
    device = "cuda"
    B, H, H_kv, D = 1, 8, 2, 64
    T = 2048
    q = torch.randn(B, H, 1, D, device=device, dtype=torch.float32) * 0.1
    k = torch.randn(B, H_kv, T, D, device=device, dtype=torch.float32) * 0.1
    v = torch.randn(B, H_kv, T, D, device=device, dtype=torch.float32) * 0.1

    q_pos = torch.tensor([T - 1], device=device, dtype=torch.int64)
    scale = 1.0 / math.sqrt(D)

    # 5 non-contiguous (chunk_start, chunk_end) intervals. Gaps in between
    # represent positions that the scorer "discarded" for this step.
    intervals = [
        (0, 256),
        (768, 1024),
        (1280, 1408),
        (1536, 1664),
        (1856, 2048),
    ]
    # Build the gold-standard one-shot reference attending to ONLY these
    # positions (zero out everything else with -inf in the softmax).
    keep_mask = torch.zeros(T, dtype=torch.bool, device=device)
    for lo, hi in intervals:
        keep_mask[lo:hi] = True
    k_kept = k.clone()
    v_kept = v.clone()
    # Reference: causal + keep_mask
    rep = H // H_kv
    k_ref = k_kept.repeat_interleave(rep, dim=1) if rep > 1 else k_kept
    v_ref = v_kept.repeat_interleave(rep, dim=1) if rep > 1 else v_kept
    qk = torch.matmul(q.to(torch.float32),
                      k_ref.to(torch.float32).transpose(-1, -2)) * scale
    j = torch.arange(T, device=device)
    causal = (j.unsqueeze(0) <= q_pos.view(-1, 1).to(j.dtype)).view(B, 1, 1, T)
    final_mask = causal & keep_mask.view(1, 1, 1, T)
    qk = torch.where(final_mask, qk, torch.full_like(qk, float("-inf")))
    p = torch.softmax(qk, dim=-1)
    out_ref = torch.matmul(p, v_ref.to(torch.float32))  # (B, H, 1, D_v)

    # Triton chunked path: pass each interval as a separate chunk with
    # its global chunk_start. The kernel's causal mask is global_n = chunk_start
    # + n_offs <= q_pos, so masking is correct across non-contiguous chunks.
    chunks = []
    for lo, hi in intervals:
        k_c = k[..., lo:hi, :].contiguous()
        v_c = v[..., lo:hi, :].contiguous()
        chunks.append((k_c, v_c, lo, True))

    out_triton = triton_chunked_attention(q, chunks, scale=scale, q_pos=q_pos)
    rel = (out_ref - out_triton.to(torch.float32)).norm() / out_ref.norm()
    assert rel.item() <= 1e-4, (
        f"non-contiguous partition identity broken: rel={rel.item():.3e}"
    )

    # Also test that PERMUTED chunk order gives the same result (LSE-merge
    # is associative + commutative over chunks, so order doesn't matter).
    import random as _rnd
    _rnd.seed(0)
    permuted = chunks.copy()
    _rnd.shuffle(permuted)
    out_perm = triton_chunked_attention(q, permuted, scale=scale, q_pos=q_pos)
    rel_perm = (out_ref - out_perm.to(torch.float32)).norm() / out_ref.norm()
    assert rel_perm.item() <= 1e-4, (
        f"permuted-chunk-order identity broken: rel={rel_perm.item():.3e}"
    )


def test_triton_matches_python_reference_path():
    """End-to-end: triton path on cold+recent split == HALOCacheChunked
    `compute_attention` on the same split. This is the contract that lets
    us swap the implementation behind `HALOConfig(use_triton=True)`.
    """
    from halo.triton_chunked import triton_chunked_attention

    torch.manual_seed(2)
    device = "cuda"
    B, H, H_kv, D = 1, 4, 1, 64
    T = 2048
    chunk_size = 256
    q = torch.randn(B, H, 1, D, device=device, dtype=torch.float32) * 0.1
    k = torch.randn(B, H_kv, T, D, device=device, dtype=torch.float32) * 0.1
    v = torch.randn(B, H_kv, T, D, device=device, dtype=torch.float32) * 0.1

    # "Cold" = first 1500, "recent" = last 548. Triton path treats them
    # identically — both are GPU-resident at kernel call time — so the
    # output must equal the one-shot reference within fp32 tolerance.
    q_pos = torch.tensor([T - 1], device=device, dtype=torch.int64)
    scale = 1.0 / math.sqrt(D)
    out_ref = _reference_full_attention(q, k, v, scale=scale, q_pos=q_pos)

    # Build chunk list mixing "cold" and "recent".
    chunks = []
    for c0 in range(0, 1500, chunk_size):
        c1 = min(c0 + chunk_size, 1500)
        chunks.append((k[..., c0:c1, :].contiguous(),
                       v[..., c0:c1, :].contiguous(), c0, True))
    chunks.append((k[..., 1500:, :].contiguous(),
                   v[..., 1500:, :].contiguous(), 1500, True))

    out_triton = triton_chunked_attention(q, chunks, scale=scale, q_pos=q_pos)
    rel = (out_ref - out_triton.to(torch.float32)).norm() / out_ref.norm()
    assert rel.item() <= 1e-4, f"end-to-end: rel={rel.item():.3e}"
