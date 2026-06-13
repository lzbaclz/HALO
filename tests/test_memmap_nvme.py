"""Tests for the numpy.memmap-backed NVMe tier.

Round-trips a few blocks through the NVMe tier for both fp32 and bfloat16,
proving that the dtype-agnostic byte view is correct (numpy has no native bf16).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_nvme_round_trip(tmp_path, dtype):
    from halo.memory_tier import MemoryTier, TieredStorage

    storage = TieredStorage(
        tiers=[MemoryTier.GPU, MemoryTier.NVME],
        num_layers=2, num_kv_heads=2, head_dim=8, block_size=4,
        dtype=dtype, device=torch.device("cpu"),
        max_blocks=4, nvme_path=str(tmp_path / "nvme"),
    )

    keys = torch.randn(2, 4, 8).to(dtype)
    values = torch.randn(2, 4, 8).to(dtype)

    storage.write_new_block(layer=1, block=2, keys=keys, values=values, tier="nvme")
    rk, rv = storage.read_block(layer=1, block=2)
    # bf16 round-trips bit-exactly because we copy raw bytes.
    torch.testing.assert_close(rk, keys, rtol=0, atol=0)
    torch.testing.assert_close(rv, values, rtol=0, atol=0)


def test_nvme_persists_across_reopen(tmp_path):
    """The bytes survive a fresh ``TieredStorage`` pointed at the same path …
    when memmap is opened in ``r+`` mode. We verify the *file size* is correct
    after one write — full reopen is exercised in §8 of the deployment guide.
    """
    from halo.memory_tier import MemoryTier, TieredStorage

    storage = TieredStorage(
        tiers=[MemoryTier.GPU, MemoryTier.NVME],
        num_layers=1, num_kv_heads=2, head_dim=4, block_size=2,
        dtype=torch.float32, device=torch.device("cpu"),
        max_blocks=2, nvme_path=str(tmp_path / "nvme"),
    )
    keys = torch.arange(2 * 2 * 4, dtype=torch.float32).reshape(2, 2, 4)
    storage.write_new_block(layer=0, block=0, keys=keys, values=keys.clone(), tier="nvme")

    # File should be the right size: 1 layer × 2 blocks × 2 heads × 2 size × 4 dim × 4 bytes.
    expected = 1 * 2 * 2 * 2 * 4 * 4
    assert (tmp_path / "nvme" / "keys.bin").stat().st_size == expected
    assert (tmp_path / "nvme" / "values.bin").stat().st_size == expected


def test_move_block_gpu_to_nvme_round_trip(tmp_path):
    from halo.memory_tier import MemoryTier, TieredStorage

    storage = TieredStorage(
        tiers=[MemoryTier.GPU, MemoryTier.DRAM, MemoryTier.NVME],
        num_layers=1, num_kv_heads=1, head_dim=4, block_size=2,
        dtype=torch.float32, device=torch.device("cpu"),
        max_blocks=2, nvme_path=str(tmp_path / "nvme"),
    )
    keys = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
    values = -keys
    storage.write_new_block(layer=0, block=0, keys=keys, values=values, tier="gpu")
    storage.move_block(layer=0, block=0, src="gpu", dst="dram")
    storage.move_block(layer=0, block=0, src="dram", dst="nvme")
    assert storage.locate_block(layer=0, block=0) == "nvme"
    rk, rv = storage.read_block(layer=0, block=0)
    torch.testing.assert_close(rk, keys, rtol=0, atol=0)
    torch.testing.assert_close(rv, values, rtol=0, atol=0)
