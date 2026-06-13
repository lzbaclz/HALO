"""Tests for HALOCacheEvict (the strict-eviction sibling of HALOCache).

Goals
-----
1. ``hot_ratio = 1.0`` is bit-identical to full attention (no eviction fires).
2. With ``hot_ratio < 1`` and a non-trivial hot mask installed, the
   columns of (K, V) outside the hot mask are physically zeroed.
3. The first attention call (before any hot mask exists) is a strict
   pass-through.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _build_cache(cfg, storage, evict: bool = True):
    from halo.demoter import HALODemoter
    from halo.refetcher import HALORefetcher
    from halo.scorer import HALOScorer

    if evict:
        from halo.kv_cache_evict import HALOCacheEvict as Cache
    else:
        from halo.kv_cache import HALOCache as Cache

    scorer = HALOScorer(cfg)
    demoter = HALODemoter(cfg, storage=storage)
    refetcher = HALORefetcher(cfg, storage=storage)
    return Cache(config=cfg, storage=storage,
                 scorer=scorer, demoter=demoter, refetcher=refetcher)


def _fake_kv(K_len: int, num_heads: int = 2, head_dim: int = 8):
    """Generate deterministic key/value tensors of shape (1, num_heads, K_len, head_dim)."""
    g = torch.Generator().manual_seed(0)
    keys = torch.randn(1, num_heads, K_len, head_dim, generator=g)
    vals = torch.randn(1, num_heads, K_len, head_dim, generator=g)
    return keys, vals


def test_pass_through_before_hotness_is_set(small_config, storage_factory):
    storage = storage_factory()
    cache = _build_cache(small_config, storage, evict=True)
    keys, vals = _fake_kv(K_len=16)
    K, V = cache.update(keys, vals, 0)
    # No hot mask yet -> identity.
    assert torch.equal(K, keys)
    assert torch.equal(V, vals)


def test_hot_ratio_one_is_identity(small_config, storage_factory, fake_attention):
    cfg = small_config
    cfg.hot_ratio = 1.0
    storage = storage_factory()
    cache = _build_cache(cfg, storage, evict=True)

    keys, vals = _fake_kv(K_len=16)
    cache.update(keys, vals, 0)
    cache.update_hotness(layer_idx=0, attn_weights=fake_attention(K=16))
    K, V = cache.update(torch.zeros_like(keys), torch.zeros_like(vals), 0)
    # Step 2 returns the concat of step 1 KV with zeros — but no eviction
    # should have happened, so the leading 16 columns are exactly `keys`.
    assert K.shape[-2] == 32
    assert torch.equal(K[..., :16, :], keys)
    assert torch.equal(V[..., :16, :], vals)


def test_eviction_zeros_cold_columns(small_config, storage_factory):
    """At hot_ratio < 1, columns outside the hot mask should be zeroed."""
    cfg = small_config
    cfg.hot_ratio = 0.25  # 4 of 16 positions hot
    storage = storage_factory()
    cache = _build_cache(cfg, storage, evict=True)

    keys, vals = _fake_kv(K_len=16)
    cache.update(keys, vals, 0)

    # Plant a sharp attention weight on positions {2, 5, 9, 13} so the
    # scorer puts those in the hot set. The closed-form score uses
    # alpha*attention + beta*recency + gamma*sink; with sink_tokens=2 and
    # those 4 spike positions, the top-4 will be {0, 1, 13, ...} — at
    # hot_ratio=0.25 with K=16 we expect exactly 4 indices in the hot set.
    attn = torch.zeros(16)
    attn[[2, 5, 9, 13]] = 1.0
    attn = attn / attn.sum()
    cache.update_hotness(layer_idx=0, attn_weights=attn)

    # Now the next attention call should sparsify (K, V).
    K, V = cache.update(torch.zeros(1, 2, 0, 8), torch.zeros(1, 2, 0, 8), 0)
    # Exactly cfg.hot_ratio * K = 4 hot indices.
    hot = cache._last_hot_indices[0].tolist()
    assert len(hot) == 4
    cold = sorted(set(range(16)) - set(hot))
    # Hot columns preserved.
    for h in hot:
        assert torch.equal(K[..., h, :], keys[..., h, :])
        assert torch.equal(V[..., h, :], vals[..., h, :])
    # Cold columns zeroed.
    for c in cold:
        assert torch.allclose(K[..., c, :], torch.zeros_like(keys[..., c, :]))
        assert torch.allclose(V[..., c, :], torch.zeros_like(vals[..., c, :]))


def test_telemetry_still_works_under_eviction(small_config, storage_factory, fake_attention):
    storage = storage_factory()
    cache = _build_cache(small_config, storage, evict=True)
    keys, vals = _fake_kv(K_len=16)
    cache.update(keys, vals, 0)
    cache.update_hotness(layer_idx=0, attn_weights=fake_attention(K=16))
    payload = cache.telemetry()
    assert "demoted_blocks" in payload
    assert "refetch_hit_rate" in payload
