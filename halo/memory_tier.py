"""Multi-tier KV storage abstraction.

Tier ladder (top is fastest):

    GPU (HBM)  ─►  DRAM (host pinned)  ─►  NVMe (mmap'd file)

Each tier exposes the same block-addressable interface. The block is the
smallest demote/refetch granularity; default ``block_size = 32`` tokens.

The NVMe tier is backed by :class:`numpy.memmap`. Because numpy lacks a native
``bfloat16`` dtype we always store the underlying bytes as ``uint8`` and
reinterpret on read using :func:`torch.frombuffer`. This is the simplest scheme
that round-trips arbitrary torch dtypes through the kernel page cache; for true
zero-copy NVMe DMA one would need libaio / liburing or GPU-Direct Storage,
which we leave to follow-up work.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch


class MemoryTier(str, Enum):
    """Symbolic tier names. Order is fast → slow."""

    GPU = "gpu"
    DRAM = "dram"
    NVME = "nvme"

    @classmethod
    def from_name(cls, name) -> "MemoryTier":
        if isinstance(name, cls):
            return name
        return cls(str(name).lower())


@dataclass
class _BlockBuffer:
    """A contiguous tier buffer, indexed by ``(layer, block, kv)``.

    Shapes: keys ``(num_layers, num_blocks, num_kv_heads, block_size, head_dim)``,
    same for values.
    """

    keys: "torch.Tensor"
    values: "torch.Tensor"

    def block(self, layer: int, block: int) -> tuple["torch.Tensor", "torch.Tensor"]:
        return self.keys[layer, block], self.values[layer, block]


@dataclass
class _MemmapBuffer:
    """NVMe tier: numpy.memmap-backed bytes view + torch reinterpret on read."""

    keys_path: str
    values_path: str
    shape: tuple
    dtype: "torch.dtype"
    itemsize: int  # bytes per element of ``dtype``
    _keys_mm: np.memmap = field(repr=False)
    _values_mm: np.memmap = field(repr=False)

    def block(self, layer: int, block: int) -> tuple["torch.Tensor", "torch.Tensor"]:
        return self._read(layer, block, which="keys"), self._read(layer, block, which="values")

    def _slice_offsets(self, layer: int, block: int) -> tuple[int, int]:
        per_block = int(np.prod(self.shape[2:])) * self.itemsize
        per_layer = self.shape[1] * per_block
        start = layer * per_layer + block * per_block
        return start, start + per_block

    def _read(self, layer: int, block: int, *, which: str) -> "torch.Tensor":
        import torch

        mm = self._keys_mm if which == "keys" else self._values_mm
        s, e = self._slice_offsets(layer, block)
        buf = bytes(mm[s:e])  # contiguous copy out of the mmap
        per_block_elems = int(np.prod(self.shape[2:]))
        flat = torch.frombuffer(bytearray(buf), dtype=self.dtype, count=per_block_elems)
        return flat.view(*self.shape[2:]).clone()

    def write(self, layer: int, block: int, *, keys: "torch.Tensor",
              values: "torch.Tensor") -> None:
        import torch as _t

        s, e = self._slice_offsets(layer, block)
        for which, src in (("keys", keys), ("values", values)):
            mm = self._keys_mm if which == "keys" else self._values_mm
            src = src.detach().contiguous().cpu()
            byte_view = src.view(_t.uint8).numpy().reshape(-1)  # dtype-agnostic, supports bf16
            mm[s:e] = byte_view
        self._keys_mm.flush()
        self._values_mm.flush()


class TieredStorage:
    """Owns one buffer per tier and a per-(layer, block) tier directory."""

    def __init__(
        self,
        *,
        tiers: list[MemoryTier],
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        dtype: "torch.dtype",
        device: "torch.device",
        nvme_path: Optional[str] = None,
        max_blocks: int = 4096,
    ) -> None:
        import torch

        self.tiers = [MemoryTier.from_name(t) for t in tiers]
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.gpu_device = device
        self.max_blocks = max_blocks

        # Per-(layer, block) tier directory.
        self._directory: dict[tuple[int, int], MemoryTier] = {}

        # Tier buffers are allocated **lazily** on first write to that
        # tier. Eagerly allocating a full-size GPU block buffer
        # (``shape = (L, max_blocks, H_kv, block_size, D)``) was the
        # cause of W7 in the 2026-05-12 review: for Qwen2.5-7B at
        # ``max_blocks=4096`` that's ~7.5 GiB of unused empty memory
        # that pushed Path A's peak GPU above one-shot full attention.
        # In the HALOPress-driven setup (which is what every benchmark
        # in §5 actually uses) the GPU block buffer is never written
        # to, so this eager allocation was pure overhead.
        self._shape = (num_layers, max_blocks, num_kv_heads, block_size, head_dim)
        self._nvme_path = nvme_path
        self._buffers: dict = {}

    # ------------------------------------------------------------------ API

    def move_block(self, *, layer: int, block: int, src, dst) -> None:
        """Copy one block from ``src`` tier to ``dst`` tier and update directory."""
        s, d = MemoryTier.from_name(src), MemoryTier.from_name(dst)
        if s == d:
            return
        sk, sv = self._read(layer, block, tier=s)
        self._write(layer, block, keys=sk, values=sv, tier=d)
        self._directory[(layer, block)] = d

    def locate_block(self, *, layer: int, block: int) -> str:
        return self._directory.get((layer, block), MemoryTier.GPU).value

    def write_new_block(
        self,
        *,
        layer: int,
        block: int,
        keys: "torch.Tensor",
        values: "torch.Tensor",
        tier: str = "gpu",
    ) -> None:
        t = MemoryTier.from_name(tier)
        self._write(layer, block, keys=keys, values=values, tier=t)
        self._directory[(layer, block)] = t

    def read_block(self, *, layer: int, block: int) -> tuple["torch.Tensor", "torch.Tensor"]:
        tier = self._directory.get((layer, block), MemoryTier.GPU)
        return self._read(layer, block, tier=tier)

    # ------------------------------------------------------------------ internals

    def _ensure_tier_buffer(self, tier: MemoryTier) -> None:
        """Lazily allocate the per-tier block buffer on first write.
        See the comment at the top of :meth:`__init__` for why
        allocation is deferred."""
        if tier in self._buffers:
            return
        if tier not in self.tiers:
            raise ValueError(f"Tier {tier} not requested at construction time.")
        import torch
        if tier is MemoryTier.GPU:
            self._buffers[tier] = _BlockBuffer(
                keys=torch.empty(self._shape, dtype=self.dtype,
                                 device=self.gpu_device),
                values=torch.empty(self._shape, dtype=self.dtype,
                                   device=self.gpu_device),
            )
        elif tier is MemoryTier.DRAM:
            pin = (torch.cuda.is_available()
                   and self.gpu_device.type == "cuda")
            self._buffers[tier] = _BlockBuffer(
                keys=torch.empty(self._shape, dtype=self.dtype,
                                 device="cpu", pin_memory=pin),
                values=torch.empty(self._shape, dtype=self.dtype,
                                   device="cpu", pin_memory=pin),
            )
        elif tier is MemoryTier.NVME:
            self._buffers[tier] = self._allocate_nvme(
                self._shape, nvme_path=self._nvme_path,
            )

    def _read(self, layer: int, block: int, *, tier: MemoryTier):
        self._ensure_tier_buffer(tier)
        buf = self._buffers[tier]
        if isinstance(buf, _MemmapBuffer):
            return buf.block(layer, block)
        return buf.block(layer, block)

    def _write(self, layer: int, block: int, *, keys, values, tier: MemoryTier) -> None:
        self._ensure_tier_buffer(tier)
        buf = self._buffers[tier]
        if isinstance(buf, _MemmapBuffer):
            buf.write(layer, block, keys=keys, values=values)
            return
        # Plain BlockBuffer (GPU / DRAM): copy_ in place.
        buf.keys[layer, block].copy_(keys)
        buf.values[layer, block].copy_(values)

    def _allocate_nvme(self, shape, *, nvme_path: Optional[str]) -> _MemmapBuffer:
        import torch

        path = Path(
            nvme_path
            or os.environ.get("HALO_NVME_PATH")
            or "/tmp/halo_nvme"
        )
        path.mkdir(parents=True, exist_ok=True)
        keys_path = path / "keys.bin"
        values_path = path / "values.bin"
        # Total bytes = numel * itemsize.
        itemsize = torch.tensor([], dtype=self.dtype).element_size()
        total_bytes = int(np.prod(shape)) * itemsize
        # Always store as uint8 so we can support arbitrary dtypes (incl. bfloat16
        # for which numpy has no native dtype).
        keys_mm = np.memmap(str(keys_path), dtype=np.uint8, mode="w+", shape=(total_bytes,))
        values_mm = np.memmap(str(values_path), dtype=np.uint8, mode="w+", shape=(total_bytes,))
        return _MemmapBuffer(
            keys_path=str(keys_path),
            values_path=str(values_path),
            shape=tuple(shape),
            dtype=self.dtype,
            itemsize=itemsize,
            _keys_mm=keys_mm,
            _values_mm=values_mm,
        )

    def __repr__(self) -> str:  # pragma: no cover
        ts = ",".join(t.value for t in self.tiers)
        return f"TieredStorage(tiers=[{ts}], blocks={self.max_blocks})"
