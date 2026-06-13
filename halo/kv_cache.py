"""Tier-aware Cache subclass for HuggingFace ``transformers``.

Design
------
``HALOCache`` inherits from :class:`transformers.cache_utils.DynamicCache` so
that it is a *strict superset* in behavior: it returns the same ``(K, V)``
tensor that ``DynamicCache.update`` would return, which guarantees that the
attention call sees the same softmax input regardless of whether HALO is
installed. The only side effect of HALO is in the *physical* tier each KV
block lives on, surfaced through :meth:`update_hotness` (called by the
attention forward hook installed in :mod:`halo.policy`) and the asynchronous
demote/refetch driven by the :class:`HALODemoter` and :class:`HALORefetcher`.

Identity invariant
------------------
With ``hot_ratio = 1.0`` (every position is hot), the demoter never fires and
the refetcher only sees hits, so HALO is a no-op layer over ``DynamicCache``.
This invariant is exercised by :func:`tests.test_kv_cache.test_identity_at_hot_ratio_one`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from halo.demoter import HALODemoter
    from halo.memory_tier import TieredStorage
    from halo.policy import HALOConfig
    from halo.refetcher import HALORefetcher
    from halo.scorer import HALOScorer

try:  # transformers is an install-time soft dependency for tests of pure-Python paths.
    from transformers.cache_utils import DynamicCache as _BaseCache
    _HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover

    class _BaseCache:  # type: ignore[no-redef]
        """Lightweight stand-in so that the file imports without transformers."""

        def __init__(self) -> None:
            self.key_cache: list = []
            self.value_cache: list = []

        def update(self, key_states, value_states, layer_idx, *args, **kwargs):
            import torch as _t

            while len(self.key_cache) <= layer_idx:
                self.key_cache.append(None)  # type: ignore[arg-type]
                self.value_cache.append(None)  # type: ignore[arg-type]
            prev_k = self.key_cache[layer_idx]
            prev_v = self.value_cache[layer_idx]
            if prev_k is None:
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = _t.cat([prev_k, key_states], dim=-2)
                self.value_cache[layer_idx] = _t.cat([prev_v, value_states], dim=-2)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        def get_seq_length(self, layer_idx: int = 0) -> int:
            if layer_idx >= len(self.key_cache) or self.key_cache[layer_idx] is None:
                return 0
            return self.key_cache[layer_idx].shape[-2]

    _HAS_TRANSFORMERS = False


class HALOCache(_BaseCache):
    """KV cache that demotes cold entries to slower tiers and refetches on demand.

    Inherits the full ``Cache`` protocol from :class:`DynamicCache` (or the
    pure-Python fallback above) so that wrapping a model with HALO does not
    change the values the attention call sees. The HALO-specific bookkeeping is
    kept in three private dictionaries and surfaced via :meth:`update_hotness`.
    """

    def __init__(
        self,
        *,
        config: "HALOConfig",
        storage: "TieredStorage",
        scorer: "HALOScorer",
        demoter: "HALODemoter",
        refetcher: "HALORefetcher",
    ) -> None:
        super().__init__()
        self.cfg = config
        self.storage = storage
        self.scorer = scorer
        self.demoter = demoter
        self.refetcher = refetcher

        self._step = 0
        self._last_hot_indices: dict[int, "torch.Tensor"] = {}
        self._last_score: dict[int, "torch.Tensor"] = {}
        # Telemetry for ablations.
        self.demoted_blocks_total = 0
        self.refetched_blocks_total = 0

    # ------------------------------------------------------------------ HF API
    # We deliberately do NOT override `update` — DynamicCache's implementation
    # already returns the full (K, V) we need. HALO state is updated post-hoc
    # via `update_hotness`, called by the attention forward hook.

    # ------------------------------------------------------------------ HALO-specific

    def reset(self) -> None:
        """Clear cache state. Called once per :meth:`generate` invocation.

        On ``transformers >= 5`` the underlying ``DynamicCache`` keeps state in
        a per-layer list (``self.layers``) plus internal flags (``_seen_tokens``,
        ``layer_class_to_replicate``); the historical ``key_cache`` /
        ``value_cache`` attributes are no longer present. We re-initialize the
        base class to wipe all of that, then restore HALO-side bookkeeping. On
        the older transformers 4.x path, this still works because the lightweight
        fallback in this file just clears the two lists.
        """
        # Re-initialize the underlying transformers Cache. This wipes any
        # persistent ``layers``/``_seen_tokens`` carried over from a previous
        # ``generate`` call. We must call the *transformers* base
        # (``_BaseCache``) regardless of subclass depth — naively walking
        # ``type(self).__bases__[0]`` lands on ``HALOCache`` for any
        # ``HALOCache`` subclass (e.g. ``HALOCacheEvict``), whose
        # ``__init__`` requires kwargs we cannot supply here.
        try:
            _BaseCache.__init__(self)
        except Exception:
            # Best-effort fallback for the pure-Python stub used in CPU-only CI.
            if hasattr(self, "key_cache") and isinstance(self.key_cache, list):
                self.key_cache.clear()
            if hasattr(self, "value_cache") and isinstance(self.value_cache, list):
                self.value_cache.clear()

        # Reset HALO-side bookkeeping.
        self._step = 0
        self._last_hot_indices.clear()
        self._last_score.clear()
        self.demoted_blocks_total = 0
        self.refetched_blocks_total = 0

    def step(self) -> None:
        self._step += 1

    @property
    def step_index(self) -> int:
        return self._step

    def update_hotness(self, layer_idx: int, attn_weights: "torch.Tensor") -> None:
        """Hook: called by the model wrapper after attention is computed.

        ``attn_weights`` is expected to be the head-mean attention vector of
        shape ``(K,)`` corresponding to the most recent decoding step.

        At ``hot_ratio = 1.0``, the previous and current hot sets are identical
        (both cover every position), so :meth:`HALODemoter.demote` is called
        with an empty index list — i.e. HALO degenerates to full attention.
        """
        score = self.scorer.score(attn_weights, step=self._step)
        hot = self.scorer.topk_hot(score)

        # Demote what fell out of the hot set.
        prev_hot = self._last_hot_indices.get(layer_idx)
        if prev_hot is not None:
            cooled = _setdiff(prev_hot, hot)
            if cooled.numel() > 0:
                blocks = (cooled // self.cfg.block_size).unique().tolist()
                self.demoter.demote(layer=layer_idx, block_indices=blocks,
                                    src="gpu", dst=self._next_tier_after("gpu"))
                self.demoted_blocks_total += len(blocks)

        # Predict next-step hot set and prefetch as needed.
        predicted = self.refetcher.predict_hot(
            layer=layer_idx,
            last_hot_indices=hot,
            last_score=score,
        )
        before_misses = self.refetcher.misses
        self.refetcher.schedule(layer=layer_idx, predicted_indices=predicted)
        self.refetched_blocks_total += self.refetcher.misses - before_misses

        self._last_hot_indices[layer_idx] = hot
        self._last_score[layer_idx] = score

    def telemetry(self) -> dict:
        """Return a JSON-able snapshot of HALO state useful for ablation tables."""
        total_refetch = self.refetcher.hits + self.refetcher.misses
        hit_rate = (self.refetcher.hits / total_refetch) if total_refetch else 0.0
        return {
            "step": self._step,
            "demoted_blocks": self.demoted_blocks_total,
            "refetched_blocks": self.refetched_blocks_total,
            "refetch_hits": self.refetcher.hits,
            "refetch_misses": self.refetcher.misses,
            "refetch_hit_rate": hit_rate,
        }

    # ------------------------------------------------------------------ helpers

    def _next_tier_after(self, current: str) -> str:
        names = list(self.cfg.tiers)
        try:
            i = names.index(current)
        except ValueError:
            return current
        return names[min(i + 1, len(names) - 1)]


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def _setdiff(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    import torch

    sa, sb = set(a.tolist()), set(b.tolist())
    diff = sorted(sa - sb)
    return torch.tensor(diff, dtype=a.dtype, device=a.device)
