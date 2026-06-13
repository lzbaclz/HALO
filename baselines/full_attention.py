"""Identity baseline — full attention, no compression."""
from __future__ import annotations


def apply(model, *, memory_ratio: int = 1, **kwargs):  # noqa: D401
    """Return ``model`` unchanged. ``memory_ratio`` must be 1."""
    if memory_ratio != 1:
        raise ValueError("Full attention has no compression knob.")
    return model
