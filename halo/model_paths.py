"""Centralised model-path resolution for HALO scripts.

Background
----------
Many of the standalone ``scripts/run_*.py`` runners (vLLM baseline,
KIVI baseline, Quest baseline, RULER) load checkpoints by absolute
path (``/public/model_zoo/Qwen2.5-7B-Instruct``). On the original
rented-GPU hosts this layout matches the cached safetensors;
reviewers reproducing on different infrastructure need a single
override point. This module is that point.

Order of resolution
-------------------
Given a ``hint`` (either an HF Hub id like ``Qwen/Qwen2.5-7B-Instruct``
or a local path like ``/public/model_zoo/Qwen2.5-7B-Instruct``) and
optionally a model YAML config dict:

1. If ``hint`` is an absolute path that exists on disk → return ``hint``.
2. If ``HALO_MODEL_ZOO`` env var is set and
   ``$HALO_MODEL_ZOO/<basename of hint>`` exists → return that.
3. If ``HALO_MODEL_HUB_PREFIX`` env var is set and the hint matches
   ``<prefix>/...`` → return that.
4. Otherwise return ``hint`` unchanged (HF Hub will try it).

Example
-------
>>> from halo.model_paths import resolve_model_path
>>> resolve_model_path("/public/model_zoo/Qwen2.5-7B-Instruct")
'/public/model_zoo/Qwen2.5-7B-Instruct'                       # if exists
>>> # or on a different host:
>>> # HALO_MODEL_ZOO=/scratch/models python ...
>>> resolve_model_path("/public/model_zoo/Qwen2.5-7B-Instruct")
'/scratch/models/Qwen2.5-7B-Instruct'

Standalone scripts should call ``resolve_model_path`` exactly once
near the top of ``main()`` and pass the result to
``AutoModelForCausalLM.from_pretrained`` / ``LLM(model=...)``.
"""
from __future__ import annotations

import os
from typing import Optional


def resolve_model_path(hint: str, *, model_cfg: Optional[dict] = None) -> str:
    """Return a path or HF Hub id usable by ``from_pretrained``.

    Args
    ----
    hint : the original ``name_or_path`` from a YAML config (or a CLI flag).
    model_cfg : optional model YAML dict (currently unused but reserved for
                future overrides, e.g. per-model revision pins).

    Returns
    -------
    str : resolved path / HF Hub id.
    """
    if hint and os.path.isabs(hint) and os.path.exists(hint):
        return hint
    zoo = os.environ.get("HALO_MODEL_ZOO")
    if zoo and hint:
        candidate = os.path.join(zoo, os.path.basename(hint))
        if os.path.exists(candidate):
            return candidate
    hub_prefix = os.environ.get("HALO_MODEL_HUB_PREFIX")
    if hub_prefix and hint and not os.path.isabs(hint):
        # Replace the leading org with the override (rare, mostly for mirrors).
        head, _, tail = hint.partition("/")
        if head and tail:
            return f"{hub_prefix.rstrip('/')}/{tail}"
    return hint


__all__ = ["resolve_model_path"]
