"""HALOCacheEvict: a sibling of :class:`HALOCache` that *physically* drops
cold KV positions from the GPU buffer at attention-call time.

Why a separate class
--------------------
The default :class:`HALOCache` deliberately preserves the full ``(K, V)``
returned to attention so that the identity invariant (``hot_ratio = 1.0``
yields bit-exact full attention) is provable. That choice means HALO does
not realize end-to-end memory savings on its own — the demoter/refetcher
fire as side effects only.

This file provides ``HALOCacheEvict``: a strict-eviction variant that
zeros out cold positions in the returned ``(K, V)`` (so attention sees a
*real* sparsified context). It is **not** information-preserving for
``hot_ratio < 1`` — it sits architecturally between H2O-style permanent
discard and the full HALO design (which would refetch needles back into
the hot buffer). The point of this class is twofold:

1. **CPU-test the eviction code path** so the demote-cold-zeros-out
   architecture is no longer vaporware (review weakness W1).
2. Provide a concrete reference for the
   "demote = zero out + restore from refetch" semantics that the
   GPU-kernel follow-up will implement.

Identity invariant: still holds at ``hot_ratio = 1.0`` because every
position is in the hot mask.

The class is a drop-in replacement: ``wrap_with_halo(model,
HALOConfig(eviction=True))`` selects this variant.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from halo.kv_cache import HALOCache, _setdiff

if TYPE_CHECKING:  # pragma: no cover
    import torch


class HALOCacheEvict(HALOCache):
    """Variant of :class:`HALOCache` that physically zeros cold positions
    in the returned ``(K, V)`` from :meth:`update`.

    Concretely: after the parent class's ``update`` returns the contiguous
    ``(K, V)`` for layer ``layer_idx``, we apply the most recent hot mask
    for that layer (cached in ``self._last_hot_indices``) and zero out
    columns that fall outside it. The first attention call of a given
    layer (before :meth:`update_hotness` has produced any hot mask) is
    pass-through: it returns the unmodified ``(K, V)``.
    """

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        import torch

        K, V = super().update(key_states, value_states, layer_idx,
                              *args, **kwargs)
        hot = self._last_hot_indices.get(layer_idx)
        # First step (no scorer call yet) -> identity pass-through.
        if hot is None or hot.numel() == 0:
            return K, V
        # If the hot mask covers every position, this is the identity
        # path — equivalent to full attention by construction.
        if int(hot.numel()) >= int(K.shape[-2]):
            return K, V
        cold = _cold_indices(hot, K.shape[-2], device=K.device)
        if cold.numel() == 0:
            return K, V
        # Zero the cold positions IN PLACE. This permanently sparsifies
        # the underlying cache buffer — that is the eviction semantics
        # (cold positions are not recoverable), and it avoids the peak-
        # memory doubling that a ``.clone()`` would incur on long
        # contexts. The (B, H, K_len, D) tensor's K_len axis is -2.
        K.index_fill_(-2, cold, 0)
        V.index_fill_(-2, cold, 0)
        return K, V


def _cold_indices(hot: "torch.Tensor", K_len: int, *, device) -> "torch.Tensor":
    """Return the complement of ``hot`` within ``range(K_len)``."""
    import torch

    full = torch.arange(K_len, device=device, dtype=hot.dtype)
    return _setdiff(full, hot)


__all__ = ["HALOCacheEvict"]
