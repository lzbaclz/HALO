"""Unit tests for the multi-tier KV storage."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_tier_directory_starts_empty(storage_factory):
    storage = storage_factory()
    # Default tier for an unknown block is GPU.
    assert storage.locate_block(layer=0, block=0) == "gpu"


def test_write_then_locate(storage_factory):
    storage = storage_factory(num_layers=2, num_kv_heads=2, head_dim=8)
    block_size = storage.block_size
    keys = torch.randn(2, block_size, 8)
    values = torch.randn(2, block_size, 8)
    storage.write_new_block(layer=0, block=3, keys=keys, values=values, tier="gpu")
    assert storage.locate_block(layer=0, block=3) == "gpu"


def test_move_block_updates_directory(storage_factory):
    storage = storage_factory()
    block_size = storage.block_size
    keys = torch.arange(2 * block_size * 8, dtype=torch.float32).reshape(2, block_size, 8)
    values = torch.zeros_like(keys)
    storage.write_new_block(layer=0, block=1, keys=keys, values=values, tier="gpu")

    storage.move_block(layer=0, block=1, src="gpu", dst="dram")
    assert storage.locate_block(layer=0, block=1) == "dram"

    rk, rv = storage.read_block(layer=0, block=1)
    torch.testing.assert_close(rk, keys)
    torch.testing.assert_close(rv, values)


def test_move_block_noop_same_tier(storage_factory):
    """Moving GPU → GPU is a no-op and must not touch the directory."""
    storage = storage_factory()
    storage.move_block(layer=0, block=0, src="gpu", dst="gpu")
    assert storage.locate_block(layer=0, block=0) == "gpu"
