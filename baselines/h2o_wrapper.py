"""H2O (Zhang et al., NeurIPS 2023) wrapper.

Defaults follow the paper: heavy_ratio = recent_ratio = 0.1, summed to give the
1× / 2× / 4× / 8× memory budgets requested by the unified protocol (§2.5 of
the EMNLP plan).
"""
from __future__ import annotations


def apply(model, *, memory_ratio: int = 4, heavy_ratio: float | None = None,
          recent_ratio: float | None = None, **kwargs):
    """Install H2O on ``model``.

    Implementation note: defers to KVPress's H2OPress when available, otherwise
    raises a clear ImportError telling the user how to install the harness.
    """
    if heavy_ratio is None:
        heavy_ratio = 0.5 / memory_ratio
    if recent_ratio is None:
        recent_ratio = 0.5 / memory_ratio
    try:
        from kvpress import H2OPress  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "H2O baseline requires kvpress. Run `pip install -e \".[eval]\"`."
        ) from e
    press = H2OPress(heavy_ratio=heavy_ratio, recent_ratio=recent_ratio)
    return press(model)
