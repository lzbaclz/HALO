"""Unit tests for :class:`halo.refetcher.HALORefetcher`."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_predict_hot_returns_topk(small_config, fake_attention):
    from halo.refetcher import HALORefetcher

    storage = None  # not exercised by predict_hot
    refetcher = HALORefetcher(small_config, storage=storage)
    score = fake_attention(K=64)
    last_hot = torch.topk(score, k=int(small_config.hot_ratio * 64)).indices

    pred = refetcher.predict_hot(layer=0, last_hot_indices=last_hot, last_score=score)
    expected_k = int(small_config.hot_ratio * 64) + small_config.lookahead
    assert pred.numel() == expected_k


def test_schedule_counts_hits_and_misses(small_config, storage_factory):
    from halo.refetcher import HALORefetcher

    cfg = small_config
    cfg.async_refetch = False  # synchronous so we can assert post-conditions
    storage = storage_factory()
    block_size = storage.block_size

    # Block 0 lives on GPU, block 1 on DRAM.
    for blk, tier in ((0, "gpu"), (1, "dram")):
        keys = torch.randn(2, block_size, 8)
        values = torch.randn(2, block_size, 8)
        storage.write_new_block(layer=0, block=blk, keys=keys, values=values, tier=tier)

    refetcher = HALORefetcher(cfg, storage=storage)
    # Positions 0 and 1 → block 0 (already on GPU). Positions 4 and 5 → block 1 (on DRAM).
    refetcher.schedule(layer=0, predicted_indices=[0, 1, 4, 5])

    assert refetcher.hits == 1   # block 0 was a hit
    assert refetcher.misses == 1  # block 1 was a miss → refetched
    # Synchronous path moved block 1 back to GPU.
    assert storage.locate_block(layer=0, block=1) == "gpu"
