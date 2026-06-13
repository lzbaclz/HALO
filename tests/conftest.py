"""Shared fixtures for the HALO test suite.

Tests run on CPU by default and skip GPU-only paths automatically. The fixtures
here exercise the *control flow* of the policy (scorer → demoter → refetcher)
without requiring a HuggingFace checkpoint to be downloaded.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def small_config():
    """Tiny HALOConfig so unit tests run in milliseconds."""
    from halo import HALOConfig

    return HALOConfig(
        hot_ratio=0.25,
        score_alpha=1.0,
        score_beta=0.5,
        score_gamma=2.0,
        sink_tokens=2,
        refresh_window=8,
        tiers=("gpu", "dram"),
        block_size=4,
        async_refetch=False,
        lookahead=1,
    )


@pytest.fixture
def fake_attention(monkeypatch):
    """Generate a deterministic head-mean attention vector for K positions."""
    import torch

    def _make(K: int, *, seed: int = 0) -> "torch.Tensor":
        g = torch.Generator().manual_seed(seed)
        return torch.softmax(torch.randn(K, generator=g), dim=0)

    return _make


@pytest.fixture
def storage_factory(small_config):
    """Build a TieredStorage on CPU for tests that touch demote / refetch paths."""
    import torch

    from halo.memory_tier import MemoryTier, TieredStorage

    def _make(num_layers: int = 2, num_kv_heads: int = 2, head_dim: int = 8,
              max_blocks: int = 16):
        return TieredStorage(
            tiers=[MemoryTier.GPU, MemoryTier.DRAM],
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=small_config.block_size,
            dtype=torch.float32,
            device=torch.device("cpu"),
            max_blocks=max_blocks,
        )

    return _make
