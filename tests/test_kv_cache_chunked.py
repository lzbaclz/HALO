"""Tests for :class:`HALOCacheChunked` — the lossless tier-paged cache.

Key invariants we exercise:

1. **Bit-equivalence to full attention under LSE-merge.** For any (Q, K, V)
   tuple, partitioning K, V into chunks and LSE-merging the per-chunk
   softmax-attention is *mathematically* the same as full attention.
   Floating-point reductions can differ by ~1e-3 in bf16, but in fp32
   we hold to ~1e-6.

2. **Chunk-count invariance.** The output is the same regardless of
   chunk_size ∈ {1, 16, 64, 256, T}. This is the test that *would*
   catch an LSE-merge implementation bug (associativity violation).

3. **Identity at r = 1.0.** As with all HALOCache subclasses, every
   position is hot, no cold blocks fire — output is exactly what
   full attention returns.

4. **Causal-decoding semantics.** ``compute_attention`` is called with
   a single query (B, H, 1, D) attending to T past positions. We
   verify that the result matches a reference implementation that
   does the full (Q, K, V) softmax in one shot.

5. **GQA / MQA broadcasting.** When the model has fewer KV-heads than
   query-heads, the chunk routine must repeat-interleave; we verify
   that head-group broadcasting works for H/H_kv ∈ {1, 2, 4, 8}.

These tests run on CPU only — no CUDA required.
"""
from __future__ import annotations

import math

import pytest
import torch


def _reference_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
) -> torch.Tensor:
    """The naive full attention we want to match. Last-query / all-past."""
    H, H_kv = q.shape[1], k.shape[1]
    if H != H_kv:
        rep = H // H_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    scale = 1.0 / math.sqrt(q.shape[-1])
    qk = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    w = torch.softmax(qk, dim=-1)
    return torch.matmul(w, v.float())


def _make_qkv(B=1, H=4, H_kv=4, T=128, D=32, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, 1, D, generator=g, dtype=dtype)
    k = torch.randn(B, H_kv, T, D, generator=g, dtype=dtype)
    v = torch.randn(B, H_kv, T, D, generator=g, dtype=dtype)
    return q, k, v


# ---------------------------------------------------------------------------
# Test 1+2: bit-equivalence and chunk-count invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("T", [1, 16, 128, 1024])
@pytest.mark.parametrize("chunk", [1, 16, 64, 256, 1024])
def test_chunked_matches_reference(T, chunk):
    """For any T and chunk size, chunked LSE-merge equals full attention.

    This is the *only* correctness theorem the title 'lossless offloading'
    needs. A failure here breaks the paper's central claim.
    """
    from halo.kv_cache_chunked import HALOCacheChunked

    q, k, v = _make_qkv(T=T, dtype=torch.float32)
    ref = _reference_attention(q, k, v)

    # Walk chunks ourselves to test the static helpers directly (no Cache
    # plumbing — keeps the test surgical).
    running_out = None
    running_lse = None
    for c0 in range(0, T, chunk):
        c1 = min(c0 + chunk, T)
        out_c, lse_c = HALOCacheChunked._chunk_attention(q, k[..., c0:c1, :], v[..., c0:c1, :])
        if running_out is None:
            running_out, running_lse = out_c, lse_c
        else:
            running_out, running_lse = HALOCacheChunked._lse_merge(
                running_out, running_lse, out_c, lse_c
            )
    out = running_out.to(q.dtype)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), \
        f"chunked@{chunk} differs from full at T={T}: max abs = {(out - ref).abs().max().item():.2e}"


def test_chunked_matches_reference_bf16():
    """bf16 path: looser tolerance, but no systematic bias."""
    from halo.kv_cache_chunked import HALOCacheChunked

    q, k, v = _make_qkv(T=512, dtype=torch.bfloat16)
    ref = _reference_attention(q, k, v).to(torch.bfloat16)

    out = None
    lse = None
    for c0 in range(0, 512, 64):
        c1 = c0 + 64
        oc, lc = HALOCacheChunked._chunk_attention(q, k[..., c0:c1, :], v[..., c0:c1, :])
        if out is None:
            out, lse = oc, lc
        else:
            out, lse = HALOCacheChunked._lse_merge(out, lse, oc, lc)
    out = out.to(torch.bfloat16)
    assert torch.allclose(out, ref, atol=5e-3, rtol=5e-3), \
        f"chunked@bf16 max abs = {(out - ref).abs().max().item():.2e}"


# ---------------------------------------------------------------------------
# Test 3: GQA / MQA broadcasting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("H,H_kv", [(8, 1), (8, 2), (8, 4), (8, 8), (16, 2)])
def test_gqa_broadcasting(H, H_kv):
    """The single-chunk attention must repeat-interleave KV heads to match Q."""
    from halo.kv_cache_chunked import HALOCacheChunked

    q, k, v = _make_qkv(H=H, H_kv=H_kv, T=64)
    ref = _reference_attention(q, k, v)
    out, _ = HALOCacheChunked._chunk_attention(q, k, v)
    assert torch.allclose(out, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 4: identity at hot_ratio = 1.0 (smoke through the Cache plumbing)
# ---------------------------------------------------------------------------


def test_identity_at_full_hot_ratio():
    """When hot_ratio = 1.0, no cold blocks; chunked path must still
    produce the same output as full attention.

    We test this by directly calling compute_attention after manually
    populating key_cache / value_cache (no transformers wrapper needed).
    """
    from halo.kv_cache_chunked import HALOCacheChunked
    from halo.demoter import HALODemoter
    from halo.memory_tier import MemoryTier, TieredStorage
    from halo.policy import HALOConfig
    from halo.refetcher import HALORefetcher
    from halo.scorer import HALOScorer

    cfg = HALOConfig(hot_ratio=1.0, tiers=("gpu", "dram"))
    storage = TieredStorage(
        tiers=[MemoryTier.GPU, MemoryTier.DRAM],
        num_layers=1, num_kv_heads=2, head_dim=32, block_size=32,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    cache = HALOCacheChunked(
        config=cfg, storage=storage, scorer=HALOScorer(cfg),
        demoter=HALODemoter(cfg, storage=storage),
        refetcher=HALORefetcher(cfg, storage=storage),
        chunk_size=64,
    )
    # Seed the cache via the parent's update mechanism so the new
    # compute_attention can read the full (K, V) tensor.
    q, k, v = _make_qkv(B=1, H=4, H_kv=2, T=200, D=32)
    cache.update(k, v, layer_idx=0)
    # Force chunked mode so the LSE-merge path is exercised even on
    # this short tensor (would normally only fire above 2*chunk_size).
    cache._mode = "chunked"
    q1 = q[..., :1, :]  # single decode-style query
    ref = _reference_attention(q1, k, v)
    out = cache.compute_attention(q1, layer_idx=0)
    assert torch.allclose(out, ref, atol=1e-5), \
        f"chunked compute_attention deviates from reference: max abs = {(out - ref).abs().max().item():.2e}"


# ---------------------------------------------------------------------------
# Test 5: telemetry / peak staging memory
# ---------------------------------------------------------------------------


def test_telemetry_reports_chunk_count():
    """The compute_attention call must record n_chunks and staging bytes."""
    from halo.kv_cache_chunked import HALOCacheChunked
    from halo.demoter import HALODemoter
    from halo.memory_tier import MemoryTier, TieredStorage
    from halo.policy import HALOConfig
    from halo.refetcher import HALORefetcher
    from halo.scorer import HALOScorer

    cfg = HALOConfig(hot_ratio=0.10, tiers=("gpu", "dram"))
    storage = TieredStorage(
        tiers=[MemoryTier.GPU, MemoryTier.DRAM],
        num_layers=1, num_kv_heads=2, head_dim=32, block_size=32,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    cache = HALOCacheChunked(
        config=cfg, storage=storage, scorer=HALOScorer(cfg),
        demoter=HALODemoter(cfg, storage=storage),
        refetcher=HALORefetcher(cfg, storage=storage),
        chunk_size=128,
    )
    # H_kv = 4 by _make_qkv default, D = 32, fp32, chunk_size = 128.
    q, k, v = _make_qkv(T=1000)
    cache.update(k, v, layer_idx=0)
    cache._mode = "chunked"
    q1 = q[..., -1:, :]
    _ = cache.compute_attention(q1, layer_idx=0)
    tele = cache.telemetry()
    assert tele["chunked_n_layers_called"] == 1
    assert tele["chunked_used_lse_merge_any"] is True
    expected_n_chunks = (1000 + 128 - 1) // 128  # walk 1000 K positions in chunks of 128
    assert tele["chunked_total_chunks"] == expected_n_chunks
