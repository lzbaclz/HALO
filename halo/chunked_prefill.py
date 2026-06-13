"""Chunked-prefill harness for Path D end-to-end memory wins.

When the prompt is very long (e.g. 16K–65K tokens), prefill itself is
the peak-GPU bottleneck — even after we wire ``HALOCacheChunked`` to
move cold KV to host DRAM, a one-shot prefill call materialises the
full prompt's per-layer ``(K, V)`` simultaneously on GPU. The way to
break this peak is to feed the model the prompt in slices, allowing
the wrapper to peel cold rows to host after each slice.

This module provides :func:`prefill_then_generate`, which:

1. Resets ``model._halo_cache`` and forces it into chunked mode.
2. Walks the prompt in ``prefill_chunk_tokens``-sized slices, calling
   ``model.forward(past_key_values=cache, use_cache=True,
   cache_position=...)`` per slice. After each slice, parent's recent
   view has grown by the slice length; we invoke
   :meth:`HALOCacheChunked._peel_to_cold` for every layer so the
   recent view shrinks back to ``recent_window`` before the next
   slice. This bounds per-layer peak GPU KV residency to
   ``slice_size + recent_window`` rows.
3. Calls ``model.generate`` for the remaining decode tokens, which
   continues to peel as the recent buffer grows.

Returned tuple: (generated_ids, telemetry). Telemetry includes
``prefill_peak_gpu_gib`` measured via
``torch.cuda.max_memory_allocated()`` straddling the prefill phase.

Notes on correctness
--------------------
Attention computed by ``cache.compute_attention`` walks the cold
suffix on host plus the recent suffix on GPU and merges via LSE,
giving output algebraically identical (real arithmetic) to one-shot
full attention on the full prompt (Prop. 4.5 part i), and per-step
bit-equivalent on a fixed KV state in fp32 (part ii). RoPE positions are preserved because (i) parent's
:meth:`get_seq_length` returns the *true* total (cold + recent) so
that ``cache_position`` rolls forward correctly, and (ii) K, V stored
in the cache are already post-RoPE for their respective positions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel
    from halo.kv_cache_chunked import HALOCacheChunked


def _force_chunked_mode(cache: "HALOCacheChunked") -> None:
    """Flip the cache into chunked mode without waiting for the warmup
    threshold. Subsequent calls to ``compute_attention`` route through
    the cold+recent LSE-merge path.
    """
    cache._mode = "chunked"  # noqa: SLF001


def _peel_all_layers(cache: "HALOCacheChunked") -> None:
    """Trigger a peel on every layer that currently has more than
    ``recent_window`` rows on GPU. Used between chunked-prefill slices.
    """
    # Iterate layers via the parent's API to find any that exceed the
    # recent threshold after the previous forward pass.
    layer_indices = []
    for li, _, k, _ in cache._iter_layer_kv():  # noqa: SLF001
        if k is None or k.numel() == 0:
            continue
        if k.shape[-2] > cache.recent_window:
            layer_indices.append(li)
    for li in layer_indices:
        cache._peel_to_cold(li)  # noqa: SLF001


def prefill_then_generate(
    model: "PreTrainedModel",
    input_ids: "torch.Tensor",
    *,
    prefill_chunk_tokens: int = 1024,
    max_new_tokens: int = 32,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    attention_mask: Optional["torch.Tensor"] = None,
) -> dict:
    """Run chunked prefill, then decode. Returns dict with generated
    ids, telemetry, and prefill peak-GPU.

    Parameters
    ----------
    model: HF model already wrapped via ``wrap_with_halo(...,
           HALOConfig(chunked=True, ...))``.
    input_ids: (1, T) prompt to prefill.
    prefill_chunk_tokens: slice size for the chunked-prefill loop.
        Smaller = lower peak GPU but more wall time. 1024 is a good
        default for 7B-class models on an 80 GB A100.
    max_new_tokens: forwarded to ``model.generate``.
    """
    import torch

    cache = model._halo_cache  # noqa: SLF001
    cache.reset()
    _force_chunked_mode(cache)

    device = input_ids.device
    B, T = input_ids.shape
    assert B == 1, "chunked-prefill harness currently assumes batch=1"

    # ---- Chunked prefill ----
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    pos = 0
    last_logits = None
    while pos < T:
        end = min(pos + prefill_chunk_tokens, T)
        slice_ids = input_ids[:, pos:end]
        slice_len = slice_ids.shape[-1]
        cache_position = torch.arange(pos, pos + slice_len, device=device)
        with torch.no_grad():
            slice_out = model(
                input_ids=slice_ids,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
        # Capture the logits at the *last* prompt position from the
        # final slice — this is the model's prediction for the first
        # token to generate (position T).
        if end == T:
            last_logits = slice_out.logits[:, -1, :].detach().clone()
        # Free the slice's activations before peeling so peak GPU
        # measurement reflects only what the next slice actually needs.
        del slice_out
        # Peel any layer that exceeds ``recent_window`` so the next
        # slice starts with a bounded GPU footprint.
        _peel_all_layers(cache)
        pos = end

    prefill_peak_gib = (torch.cuda.max_memory_allocated() / (1024 ** 3)
                        if device.type == "cuda" else 0.0)

    # ---- Decode: manual greedy / sampling loop ----
    # We bypass model.generate because HF's ``_prefill`` re-runs prefill
    # even when given a populated cache; this loop attends only to the
    # *current* token via the already-built cache, which is the actual
    # decoding cost we want to measure.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    step_logits: list = []

    def _pick_next(lg: "torch.Tensor") -> "torch.Tensor":
        if do_sample:
            probs = torch.softmax(lg / max(temperature, 1e-6), dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                cum = sorted_probs.cumsum(dim=-1)
                mask = cum > top_p
                mask[..., 0] = False
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                idx = torch.multinomial(sorted_probs, num_samples=1)
                return sorted_idx.gather(-1, idx)
            return torch.multinomial(probs, num_samples=1)
        return lg.argmax(dim=-1, keepdim=True)

    generated = []
    # The cache already contains positions [0, T). The first token is
    # predicted by the logits captured at the end of the last prefill
    # slice; we sample/argmax, append, then forward the chosen token
    # through the model so the cache absorbs it and we get logits for
    # the *next* position.
    logits = last_logits
    for _ in range(max_new_tokens):
        step_logits.append(logits.detach().clone())
        next_tok = _pick_next(logits)
        generated.append(next_tok)
        T_cached = cache.get_seq_length()
        cache_position = torch.tensor([T_cached], device=device)
        with torch.no_grad():
            out = model(
                input_ids=next_tok,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
        logits = out.logits[:, -1, :]
        del out
        # Peel layers whose recent buffer has overflowed past the
        # ``recent_window + chunk_size`` watermark.
        _peel_all_layers(cache)
    generated_ids = torch.cat([input_ids] + generated, dim=-1)

    decode_peak_gib = (torch.cuda.max_memory_allocated() / (1024 ** 3)
                       if device.type == "cuda" else 0.0)

    overall_peak_gib = max(prefill_peak_gib, decode_peak_gib)

    return {
        "generated_ids": generated_ids,
        "prefill_peak_gib": prefill_peak_gib,
        "decode_peak_gib": decode_peak_gib,
        "overall_peak_gib": overall_peak_gib,
        "cache_telemetry": cache.telemetry(),
        "step_logits": step_logits,
    }


__all__ = ["prefill_then_generate"]
