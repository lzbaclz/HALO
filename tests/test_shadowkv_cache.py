"""Smoke tests for the ShadowKV unofficial reimplementation.

These tests verify:
  (a) the cache.update() multi-step sequence doesn't crash (regression
      guard for the ``sentinel-replacement'' bug);
  (b) the SVD compression triggers exactly once at the prefill→decode
      transition;
  (c) the host-V offload is populated.
"""
from __future__ import annotations

import pytest
import torch

from baselines.shadowkv_cache import ShadowKVCache, ShadowKVConfig


def test_multi_decode_no_crash():
    cache = ShadowKVCache(ShadowKVConfig(rank=4, page_size=8))
    B, H, D = 1, 2, 8
    cache.update(torch.randn(B, H, 32, D), torch.randn(B, H, 32, D), layer_idx=0)
    for _ in range(5):
        K_after, V_after = cache.update(
            torch.randn(B, H, 1, D), torch.randn(B, H, 1, D), layer_idx=0)
    assert K_after.shape == (B, H, 32 + 5, D)
    assert V_after.shape == (B, H, 32 + 5, D)


def test_compression_triggers_on_first_decode():
    cache = ShadowKVCache(ShadowKVConfig(rank=4, page_size=8))
    B, H, D = 1, 2, 8
    cache.update(torch.randn(B, H, 32, D), torch.randn(B, H, 32, D), layer_idx=0)
    assert 0 not in cache._svd, "prefill must not compress"
    cache.update(torch.randn(B, H, 1, D), torch.randn(B, H, 1, D), layer_idx=0)
    assert 0 in cache._svd, "first decode step must trigger compression"
    assert 0 in cache._v_host, "V should be offloaded to host"


def test_compression_does_not_repeat():
    cache = ShadowKVCache(ShadowKVConfig(rank=4, page_size=8))
    B, H, D = 1, 2, 8
    cache.update(torch.randn(B, H, 32, D), torch.randn(B, H, 32, D), layer_idx=0)
    cache.update(torch.randn(B, H, 1, D), torch.randn(B, H, 1, D), layer_idx=0)
    U_first = cache._svd[0]["U"]
    cache.update(torch.randn(B, H, 1, D), torch.randn(B, H, 1, D), layer_idx=0)
    U_second = cache._svd[0]["U"]
    assert U_first.shape == U_second.shape
    assert torch.equal(U_first, U_second), "compression must not re-fire"


def test_parent_kv_not_freed():
    """Round-27 caveat: we intentionally do NOT free parent K/V (would
    require overriding update() to maintain a decode-side recent cache).
    """
    cache = ShadowKVCache(ShadowKVConfig(rank=4, page_size=8))
    B, H, D = 1, 2, 8
    cache.update(torch.randn(B, H, 32, D), torch.randn(B, H, 32, D), layer_idx=0)
    cache.update(torch.randn(B, H, 1, D), torch.randn(B, H, 1, D), layer_idx=0)
    assert cache._svd[0]["parent_kv_freed"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
