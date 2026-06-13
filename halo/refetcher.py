"""HALO-Refetcher: asynchronously prefetch blocks predicted to become hot.

The refetcher is the *read* path that closes the loop with the demoter. Given
a hot-set prediction for the upcoming ``cfg.lookahead`` decoding steps, it
issues prefetches from the slower tiers up to the GPU tier, hiding the
tier-crossing latency under model compute.

The prefetch model is intentionally simple: any predicted-hot block whose
data is currently *not* on the GPU is issued for refetch. We rely on Finding 2
(temporal stability of the hot set) to keep the prediction good enough that
mis-prefetches are rare.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from halo.memory_tier import TieredStorage
    from halo.policy import HALOConfig


class HALORefetcher:
    """Predictive, asynchronous refetcher of cold blocks."""

    def __init__(self, config: "HALOConfig", *, storage: "TieredStorage") -> None:
        self.cfg = config
        self.storage = storage
        self._stream = None
        self._inflight: set[tuple[int, int]] = set()
        # Telemetry — exposed for ablations.
        self.hits = 0
        self.misses = 0

    # ---------- public API ----------

    def predict_hot(
        self,
        *,
        layer: int,
        last_hot_indices: "torch.Tensor",
        last_score: "torch.Tensor",
    ) -> "torch.Tensor":
        """Return the indices we expect to be hot in the next step.

        The default prediction is the union of (i) the current hot set and
        (ii) the next ``lookahead`` highest-scoring positions just outside it
        (Finding 2 says this is a tight bound for window=64).
        """
        import torch

        K = last_score.shape[-1]
        if self.cfg.hot_ratio >= 1.0:
            return torch.arange(K, device=last_score.device, dtype=torch.long)
        k = min(max(int(self.cfg.hot_ratio * K), 1) + self.cfg.lookahead, K)
        return torch.topk(last_score, k=k).indices.sort().values

    def schedule(
        self,
        *,
        layer: int,
        predicted_indices: "Iterable[int] | torch.Tensor",
    ) -> None:
        """Issue prefetches for blocks that are not currently on the GPU tier."""
        import torch

        if isinstance(predicted_indices, torch.Tensor):
            predicted_indices = predicted_indices.tolist()

        block_size = self.cfg.block_size
        seen_blocks: set[int] = set()
        for pos in predicted_indices:
            blk = int(pos) // block_size
            if blk in seen_blocks:
                continue
            seen_blocks.add(blk)

            tier_for_block = self.storage.locate_block(layer=layer, block=blk)
            key = (layer, blk)
            if tier_for_block == "gpu":
                self.hits += 1
                continue
            if key in self._inflight:
                continue

            self.misses += 1
            self._inflight.add(key)
            self._issue(layer=layer, block=blk, src=tier_for_block)

    def wait(self) -> None:
        """Block the current stream on outstanding refetches."""
        import torch

        if torch.cuda.is_available() and self._stream is not None:
            torch.cuda.current_stream().wait_stream(self._stream)
        self._inflight.clear()

    # ---------- internals ----------

    def _issue(self, *, layer: int, block: int, src: str) -> None:
        import torch

        def _do() -> None:
            self.storage.move_block(layer=layer, block=block, src=src, dst="gpu")

        if self.cfg.async_refetch and torch.cuda.is_available():
            if self._stream is None:
                self._stream = torch.cuda.Stream()
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                _do()
        else:
            _do()

    def __repr__(self) -> str:  # pragma: no cover
        total = self.hits + self.misses
        rate = (self.hits / total) if total else 0.0
        return f"HALORefetcher(hit_rate={rate:.3f}, inflight={len(self._inflight)})"
