"""HALO-Demoter: stream cold KV blocks down the tier ladder.

Demotion is the *write* path. It moves blocks (k, v) of size ``block_size``
from the GPU tier to the next-slower tier (DRAM, then optionally NVMe). The
hot path of decoding is never blocked — demotions are issued asynchronously on
a side stream and only awaited if the GPU tier is about to overflow.
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Deque, Iterable

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from halo.memory_tier import TieredStorage
    from halo.policy import HALOConfig


class HALODemoter:
    """Move cold KV blocks from a faster tier to a slower one."""

    def __init__(self, config: "HALOConfig", *, storage: "TieredStorage") -> None:
        self.cfg = config
        self.storage = storage
        self._pending: Deque[tuple[int, int, str, str]] = deque()  # (layer, block, src, dst)
        self._stream = None  # CUDA stream — lazily created

    # ---------- public API ----------

    def demote(
        self,
        *,
        layer: int,
        block_indices: "Iterable[int] | torch.Tensor",
        src: str = "gpu",
        dst: str = "dram",
    ) -> None:
        """Schedule a demotion of the given block indices for one layer."""
        import torch

        if isinstance(block_indices, torch.Tensor):
            block_indices = block_indices.tolist()

        for blk in block_indices:
            self._pending.append((layer, int(blk), src, dst))

        if self.cfg.async_refetch:
            self._kick_async()
        else:
            self.flush()

    def flush(self) -> None:
        """Synchronously drain the pending queue."""
        while self._pending:
            layer, blk, src, dst = self._pending.popleft()
            self.storage.move_block(layer=layer, block=blk, src=src, dst=dst)

    # ---------- internals ----------

    def _kick_async(self) -> None:
        """Run demotions on a side CUDA stream so they overlap with model compute."""
        import torch

        if not torch.cuda.is_available():
            self.flush()
            return

        if self._stream is None:
            self._stream = torch.cuda.Stream()  # lazy

        cur_stream = torch.cuda.current_stream()
        # Wait for compute to release the source blocks.
        self._stream.wait_stream(cur_stream)
        with torch.cuda.stream(self._stream):
            self.flush()
        # Don't block the hot path waiting for the demotion to finish.

    def __repr__(self) -> str:  # pragma: no cover
        return f"HALODemoter(pending={len(self._pending)}, async={self.cfg.async_refetch})"
