"""StreamingLLM (Xiao et al., ICLR 2024) wrapper.

Defaults: sink = 4, window = 1020 / memory_ratio.
"""
from __future__ import annotations


def apply(model, *, memory_ratio: int = 4, sink_size: int = 4,
          window_size: int | None = None, **kwargs):
    # kvpress >=0.3 renamed StreamingLLMPress(sink_size=, window_size=) to
    # StreamingLLMPress(compression_ratio=, n_sink=). compression_ratio is
    # the fraction of positions EVICTED, so retention = 1 - compression_ratio.
    # For memory_ratio = 4 (retention = 0.25) → compression_ratio = 0.75.
    compression_ratio = max(0.0, 1.0 - 1.0 / memory_ratio)
    try:
        from kvpress import StreamingLLMPress  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "StreamingLLM baseline requires kvpress. Run `pip install -e \".[eval]\"`."
        ) from e
    press = StreamingLLMPress(compression_ratio=compression_ratio, n_sink=sink_size)
    return press(model)
