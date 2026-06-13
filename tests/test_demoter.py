"""Unit tests for :class:`halo.demoter.HALODemoter`."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_demote_pending_drains_with_flush(small_config, storage_factory):
    from halo.demoter import HALODemoter

    cfg = small_config
    cfg.async_refetch = False  # synchronous flush path
    storage = storage_factory()
    block_size = storage.block_size

    # Seed two blocks on GPU so the demote has something to move.
    for blk in (0, 1):
        keys = torch.randn(2, block_size, 8)
        values = torch.randn(2, block_size, 8)
        storage.write_new_block(layer=0, block=blk, keys=keys, values=values, tier="gpu")

    demoter = HALODemoter(cfg, storage=storage)
    demoter.demote(layer=0, block_indices=[0, 1], src="gpu", dst="dram")

    # Synchronous path drained right away.
    assert storage.locate_block(layer=0, block=0) == "dram"
    assert storage.locate_block(layer=0, block=1) == "dram"


def test_demote_accepts_tensor_indices(small_config, storage_factory):
    from halo.demoter import HALODemoter

    cfg = small_config
    cfg.async_refetch = False
    storage = storage_factory()
    block_size = storage.block_size

    for blk in (0, 2):
        keys = torch.randn(2, block_size, 8)
        values = torch.randn(2, block_size, 8)
        storage.write_new_block(layer=0, block=blk, keys=keys, values=values, tier="gpu")

    demoter = HALODemoter(cfg, storage=storage)
    demoter.demote(layer=0, block_indices=torch.tensor([0, 2]),
                   src="gpu", dst="dram")
    assert storage.locate_block(layer=0, block=0) == "dram"
    assert storage.locate_block(layer=0, block=2) == "dram"
