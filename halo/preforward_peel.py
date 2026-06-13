"""Pre-forward peel: make stock HF ``generate()`` realise Path D's
memory contract.

Reviewer 3 W1 (2026-05-12) identified an asymmetry: the
chunked-prefill harness in :mod:`halo.chunked_prefill` realises
-28.5% peak GPU at 65K on Qwen2.5-7B, but if a user calls stock
``model.generate(input_ids=long_prompt)`` directly, the parent
``DynamicCache`` materialises the full prompt's $(K, V)$ in a single
forward pass *before* our wrapper's peel hook fires, and the memory
saving is not realised.

This module closes that gap. After::

    wrap_with_halo(model, HALOConfig(chunked=True, ...))
    install_preforward_peel(model)

calling ``model.generate(input_ids=long_prompt, max_new_tokens=N)``
internally routes through chunked prefill: the prompt is sliced into
``prefill_chunk_tokens``-sized chunks, each chunk runs through the
model with the same parent forward, and between chunks every layer's
recent buffer is peeled so the next chunk starts with bounded GPU
residency. After prefill, the function runs the manual greedy /
sampling decode loop from :func:`halo.chunked_prefill.prefill_then_generate`
and returns a generate-like object so the caller's code is
unchanged.

The opt-in :func:`install_preforward_peel` is separate from
:func:`wrap_with_halo` because the technique is only relevant for
long prompts; we don't want it active on short LongBench prompts
where it would add slicing overhead with no memory benefit.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from transformers import PreTrainedModel


def install_preforward_peel(
    model: "PreTrainedModel",
    *,
    prefill_chunk_tokens: int = 4096,
    activation_threshold: int = 8192,
) -> "PreTrainedModel":
    """Install a transparent chunked-prefill wrapper on ``model.generate``.

    Parameters
    ----------
    model:
        A model already wrapped via ``wrap_with_halo(model,
        HALOConfig(chunked=True))``. ``ValueError`` is raised otherwise.
    prefill_chunk_tokens:
        Slice size for the chunked-prefill loop. Smaller = lower peak
        GPU but more wall time. 4096 is a good default for 7B models
        on an 80 GB A100.
    activation_threshold:
        Prompt length (in tokens) below which we delegate to stock
        ``generate()``. The threshold prevents adding slicing overhead
        on short prompts where the memory win is negligible.

    Returns
    -------
    The same ``model`` object, with ``model.generate`` patched. Idempotent.
    """
    if not getattr(model, "_halo_config", None) or not model._halo_config.chunked:
        raise ValueError(
            "install_preforward_peel requires a Path D / chunked HALO model. "
            "Wrap first via wrap_with_halo(model, HALOConfig(chunked=True))."
        )
    if getattr(model, "_halo_preforward_patched", False):
        return model

    # The previous generate hook (installed by wrap_with_halo) already
    # injects past_key_values=cache and resets per-call. We sit on top
    # of that, intercepting long-prompt calls and routing through the
    # chunked-prefill loop instead.
    original_generate = model.generate

    def _generate_with_preforward_peel(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        # HF generate accepts input_ids positionally or as kwarg.
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None or input_ids.shape[-1] <= activation_threshold:
            return original_generate(*args, **kwargs)

        # Long prompt + Path D: route through chunked prefill.
        from halo.chunked_prefill import prefill_then_generate

        max_new_tokens = kwargs.get("max_new_tokens", 32)
        do_sample = kwargs.get("do_sample", False)
        temperature = kwargs.get("temperature", 1.0)
        top_p = kwargs.get("top_p", 1.0)

        result = prefill_then_generate(
            model, input_ids,
            prefill_chunk_tokens=prefill_chunk_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        # Wrap into a HF-generate-like return so downstream code that
        # expects a ``.sequences`` attribute keeps working.
        return _PreforwardPeelResult(
            sequences=result["generated_ids"],
            telemetry=result,
        )

    model.generate = _generate_with_preforward_peel  # type: ignore[assignment]
    model._halo_preforward_patched = True            # type: ignore[attr-defined]
    model._halo_preforward_chunk = prefill_chunk_tokens      # type: ignore[attr-defined]
    model._halo_preforward_threshold = activation_threshold  # type: ignore[attr-defined]
    return model


class _PreforwardPeelResult:
    """Lightweight stand-in for ``transformers.generation.GenerateOutput``.

    Carries the generated token ids and the chunked-prefill telemetry
    on ``halo_telemetry``. Mirrors both the ``.sequences`` attribute
    (HF GenerateOutput style) AND tensor-like indexing
    (e.g. ``out[0, prompt_len:]``) so callers that treat the return
    value as a raw tensor (the standard ``model.generate`` contract
    when ``return_dict_in_generate=False``) keep working.
    """

    def __init__(self, sequences: "torch.Tensor", telemetry: dict) -> None:
        self.sequences = sequences
        self.halo_telemetry = telemetry

    # Tensor-like passthrough: most callers do ``out[0, prompt_len:]``.
    def __getitem__(self, idx):
        return self.sequences[idx]

    def __iter__(self):  # pragma: no cover (iter access for code that unpacks)
        yield self.sequences

    @property
    def shape(self):
        return self.sequences.shape

    def __len__(self):
        return len(self.sequences)


__all__ = ["install_preforward_peel"]
