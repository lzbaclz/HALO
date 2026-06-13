"""Tests for HALOCache control flow + the hot_ratio=1.0 identity invariant."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _build_cache(cfg, storage):
    from halo.demoter import HALODemoter
    from halo.kv_cache import HALOCache
    from halo.refetcher import HALORefetcher
    from halo.scorer import HALOScorer

    scorer = HALOScorer(cfg)
    demoter = HALODemoter(cfg, storage=storage)
    refetcher = HALORefetcher(cfg, storage=storage)
    return HALOCache(config=cfg, storage=storage,
                     scorer=scorer, demoter=demoter, refetcher=refetcher)


def test_cache_starts_empty(small_config, storage_factory):
    storage = storage_factory()
    cache = _build_cache(small_config, storage)
    assert cache.step_index == 0
    assert cache.demoted_blocks_total == 0
    assert cache.refetched_blocks_total == 0


def test_update_hotness_demotes_cooled_blocks(small_config, storage_factory, fake_attention):
    """When the hot set shrinks, the previously-hot block falls out → demotion fires."""
    storage = storage_factory(num_layers=1, num_kv_heads=2, head_dim=8)
    cache = _build_cache(small_config, storage)
    K = 64

    # Step 0: the back of the cache is hot — sets baseline.
    back_hot = torch.zeros(K)
    back_hot[-K // 4:] = 1.0
    back_hot = back_hot / back_hot.sum()
    cache.update_hotness(layer_idx=0, attn_weights=back_hot)

    # Step 1: only the front gets attention — back of cache cools.
    cache.step()
    skewed = torch.zeros(K)
    skewed[: K // 4] = 1.0
    skewed = skewed / skewed.sum()
    cache.update_hotness(layer_idx=0, attn_weights=skewed)

    # Some demotions should have been issued.
    assert cache.demoted_blocks_total > 0


def test_identity_at_hot_ratio_one(small_config, storage_factory, fake_attention):
    """hot_ratio = 1.0 must never demote (HALO degenerates to full attention)."""
    cfg = small_config
    cfg.hot_ratio = 1.0
    storage = storage_factory()
    cache = _build_cache(cfg, storage)

    # Walk a few decoding steps with arbitrary attention.
    K = 64
    for step in range(8):
        cache.step()
        cache.update_hotness(layer_idx=0, attn_weights=fake_attention(K=K, seed=step))

    assert cache.demoted_blocks_total == 0, (
        f"hot_ratio=1.0 must not demote any block, got {cache.demoted_blocks_total}"
    )
    assert cache.refetcher.misses == 0, (
        f"hot_ratio=1.0 must not miss any refetch, got {cache.refetcher.misses}"
    )


def test_reset_clears_everything(small_config, storage_factory, fake_attention):
    storage = storage_factory(num_layers=1, num_kv_heads=2, head_dim=8)
    cache = _build_cache(small_config, storage)
    K = 64

    cache.update_hotness(layer_idx=0, attn_weights=fake_attention(K=K))
    cache.step()
    cache.step()
    assert cache.step_index == 2

    cache.reset()
    assert cache.step_index == 0
    assert cache.demoted_blocks_total == 0
    assert cache.refetched_blocks_total == 0


def test_telemetry_returns_jsonable(small_config, storage_factory, fake_attention):
    import json
    storage = storage_factory(num_layers=1, num_kv_heads=2, head_dim=8)
    cache = _build_cache(small_config, storage)
    cache.update_hotness(layer_idx=0, attn_weights=fake_attention(K=64))
    payload = cache.telemetry()
    json.dumps(payload)  # must serialize cleanly
    assert "demoted_blocks" in payload
    assert "refetch_hit_rate" in payload
