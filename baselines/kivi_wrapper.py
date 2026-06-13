"""KIVI (Liu et al., ICML 2024) wrapper — KV cache 2-bit quantization."""
from __future__ import annotations


def apply(model, *, memory_ratio: int = 4, bits: int | None = None,
          group_size: int = 32, **kwargs):
    if bits is None:
        # 1× = bf16 (16 bit), 2× ≈ 8 bit, 4× ≈ 4 bit, 8× ≈ 2 bit.
        bits = max(16 // memory_ratio, 2)
    try:
        from kvpress import KIVIPress  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "KIVI baseline requires kvpress. Run `pip install -e \".[eval]\"`."
        ) from e
    press = KIVIPress(bits=bits, group_size=group_size)
    return press(model)
