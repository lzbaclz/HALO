"""SnapKV (Li et al., NeurIPS 2024) wrapper.

Defaults: window = 32, max_capacity scales with the unified memory ratio.
"""
from __future__ import annotations


def apply(model, *, memory_ratio: int = 4, window: int = 32,
          max_capacity: int | None = None, **kwargs):
    if max_capacity is None:
        max_capacity = max(2048 // memory_ratio, 64)
    try:
        from kvpress import SnapKVPress  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "SnapKV baseline requires kvpress. Run `pip install -e \".[eval]\"`."
        ) from e
    press = SnapKVPress(window=window, max_capacity=max_capacity)
    return press(model)
