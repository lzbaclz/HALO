"""Smoke tests for :func:`halo.wrap_with_halo` configuration plumbing.

These tests do not load a real LLM checkpoint — they assert that the
wrapper installs the right attributes and that defaults match the §4 spec.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_default_config_matches_paper_defaults():
    from halo import HALOConfig

    cfg = HALOConfig()
    assert cfg.hot_ratio == 0.10        # §4.1: F1 → 10% on GPU
    assert cfg.refresh_window == 64     # §4.1: F2 → 64-step refresh
    assert cfg.sink_tokens == 4         # StreamingLLM-style anchors
    assert "gpu" in cfg.tiers and "dram" in cfg.tiers


def test_make_storage_returns_tiered_storage():
    from halo import HALOConfig
    from halo.memory_tier import TieredStorage

    cfg = HALOConfig(tiers=("gpu", "dram"))
    storage = cfg.make_storage(
        num_layers=2, num_kv_heads=2, head_dim=8,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    assert isinstance(storage, TieredStorage)
    assert storage.num_layers == 2
    assert storage.block_size == cfg.block_size
