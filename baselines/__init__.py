"""Wrappers around the four baselines compared in the paper.

Each wrapper exposes a ``factory(model, *, memory_ratio, **kwargs)`` that
returns either:

  - ``None`` (full attention or HALO — runner handles those directly), or
  - a *callable context manager* ``press`` such that ``with press(model):``
    activates the compression. This is the natural shape for ``kvpress``
    Press objects (``kvpress.BasePress.__call__`` is a ``@contextmanager``).

The mapping from ``memory_ratio`` (the unified protocol's compression budget)
to each baseline's native knob is the responsibility of the wrapper.

Notes
-----
* KIVI (2-bit KV quantization) is not provided by ``kvpress`` 0.5+. We keep
  the ``kivi`` key in the registry but make it raise a clear ``NotImplementedError``
  so the experiment harness can skip it cleanly. Once a KIVI integration is
  added to ``kvpress`` (or wired in via the upstream ``kivi`` repo), this
  wrapper can be filled in without changing the runner.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

# Each ``apply`` returns either ``None`` (no wrapping needed) or a kvpress
# Press object (which doubles as a contextmanager-via-__call__).


def _full(model, *, memory_ratio: int = 1, **_: Any):
    if memory_ratio != 1:
        raise ValueError("Full attention has no compression knob.")
    return None  # runner sees None → no wrapping


def _ratio_to_compression(memory_ratio: int) -> float:
    """memory_ratio = 1×, 2×, 4×, 8× → compression_ratio = 0, 0.5, 0.75, 0.875."""
    if memory_ratio <= 1:
        return 0.0
    return 1.0 - (1.0 / float(memory_ratio))


def _h2o(model, *, memory_ratio: int = 4, **_: Any):
    """H2O ≈ kvpress.ObservedAttentionPress (heavy-hitter eviction)."""
    from kvpress import ObservedAttentionPress
    return ObservedAttentionPress(compression_ratio=_ratio_to_compression(memory_ratio))


_h2o.required_attn_impl = "eager"  # type: ignore[attr-defined]  # ObservedAttentionPress reads softmax attentions


def _streamingllm(model, *, memory_ratio: int = 4, n_sink: int = 4, **_: Any):
    from kvpress import StreamingLLMPress
    return StreamingLLMPress(
        compression_ratio=_ratio_to_compression(memory_ratio),
        n_sink=n_sink,
    )


def _snapkv(model, *, memory_ratio: int = 4, window_size: int = 32,
            kernel_size: int = 5, **_: Any):
    from kvpress import SnapKVPress
    return SnapKVPress(
        compression_ratio=_ratio_to_compression(memory_ratio),
        window_size=window_size,
        kernel_size=kernel_size,
    )


_snapkv.required_attn_impl = "eager"  # type: ignore[attr-defined]  # SnapKV peeks at the softmax window too


def _kivi(model, *, memory_ratio: int = 4, **_: Any):
    """KIVI-style 2-/4-bit KV quantization (self-contained port).

    Returns the wrapped model; the actual cache is installed via
    ``wrap_with_kivi`` (\\texttt{baselines/kivi_cache.py}). The kvpress
    package up to 0.5.x does not ship KIVIPress, so we provide a
    pure-PyTorch reference implementation. This is an ablation-only
    baseline (not a calibrated reference) — see
    \\Cref{sec:appendix-kivi-port}.
    """
    from baselines.kivi_cache import wrap_with_kivi
    return wrap_with_kivi(model, memory_ratio=memory_ratio)


_kivi.handles_attn_directly = True  # type: ignore[attr-defined]
# KIVI wraps the model's cache directly; no kvpress Press object.


# ---------------------------------------------------------------------------
# Round-38: recent (2024-2026) kvpress-supported presses, added to address
# reviewer ask about "compare against newer baselines". All five are
# SDPA-compatible (verified by inspection), so they can run at the same
# 32K context as our headline NIAH cells (unlike H2O/SnapKV which need
# eager attention and OOM at 32K).
# ---------------------------------------------------------------------------

def _expected_attention(model, *, memory_ratio: int = 4, **_: Any):
    """ExpectedAttentionPress (Jegou & Jha 2024+) — uses expected future
    attention rather than observed attention. Improves over H2O's
    observed-attention scorer."""
    from kvpress import ExpectedAttentionPress
    return ExpectedAttentionPress(compression_ratio=_ratio_to_compression(memory_ratio))


def _adakv(model, *, memory_ratio: int = 4, **_: Any):
    """AdaKV (Feng et al., 2024) — adaptive per-layer budget allocation
    wrapping a base scorer press. AdaKVPress.compress asserts SDPA (not
    eager); to keep that constraint while the inner SnapKV usually wants
    eager, we substitute ExpectedAttentionPress as the inner scorer
    (SDPA-compatible). The adaptive per-layer allocation is the AdaKV
    contribution being measured here."""
    from kvpress import AdaKVPress, ExpectedAttentionPress
    base = ExpectedAttentionPress(compression_ratio=_ratio_to_compression(memory_ratio))
    return AdaKVPress(press=base, alpha_safeguard=0.2)
# (no required_attn_impl annotation -> runner uses SDPA default; AdaKVPress
# itself asserts NOT eager, so SDPA is required.)


def _think(model, *, memory_ratio: int = 4, **_: Any):
    """ThinK (Xu et al., 2024 ICLR) — key-channel pruning instead of
    position pruning. Orthogonal axis to H2O/SnapKV.
    memory_ratio interpreted as channel compression ratio."""
    from kvpress import ThinKPress
    return ThinKPress(
        key_channel_compression_ratio=_ratio_to_compression(memory_ratio),
        window_size=32,
    )


def _pyramidkv(model, *, memory_ratio: int = 4, **_: Any):
    """PyramidKV (Cai et al., 2024) — pyramid budget across layers
    (more budget to lower layers, less to upper)."""
    from kvpress import PyramidKVPress
    return PyramidKVPress(
        compression_ratio=_ratio_to_compression(memory_ratio),
        window_size=64, kernel_size=5, beta=20,
    )


def _tova(model, *, memory_ratio: int = 4, **_: Any):
    """TOVA (Token Omission Via Attention; Oren et al., 2024) — token-omission
    scoring; conceptually similar to H2O but with a different scoring rule."""
    from kvpress import TOVAPress
    return TOVAPress(compression_ratio=_ratio_to_compression(memory_ratio))


# ---------------------------------------------------------------------------
# Round-40: three additional 2024-2025 presses (Compactor, CriticalKV,
# Finch). All SDPA-compatible. Finch requires a delimiter token between
# context and question — our NIAH harness does not insert one, so it is
# registered but the runner skips it with a clear error rather than
# fabricating a delimiter (which would alter the prompt template and
# violate apples-to-apples).
# ---------------------------------------------------------------------------

def _compactor(model, *, memory_ratio: int = 4, **_: Any):
    """Compactor (2025) — calibrated query-agnostic compression via
    approximate leverage scores on key embeddings + non-causal chunked
    attention. Prefill-only."""
    from kvpress import CompactorPress
    return CompactorPress(compression_ratio=_ratio_to_compression(memory_ratio))


def _criticalkv(model, *, memory_ratio: int = 4, **_: Any):
    """CriticalKV (2025) — two-stage compression: rescales an inner
    ScorerPress's scores by L1 norm of W_o @ values. Wrapped here over
    ExpectedAttentionPress (SDPA-compat inner press) with use_vnorm
    disabled per CriticalKV's recipe (avoids double-counting value norm).

    Compatibility patch: CriticalKVPress.vwl1norm reads
    ``module.config.head_dim``; older Qwen2 configs (incl. Qwen 2.5-7B)
    omit that attribute. Inject it from hidden_size / num_attention_heads
    before returning the press, so the score hook does not crash on the
    first decoder layer."""
    from kvpress import CriticalKVPress, ExpectedAttentionPress
    if model is not None:
        cfg = model.config
        if not hasattr(cfg, "head_dim") or getattr(cfg, "head_dim", None) is None:
            cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads
    inner = ExpectedAttentionPress(
        compression_ratio=_ratio_to_compression(memory_ratio),
        use_vnorm=False,
    )
    return CriticalKVPress(press=inner)


def _finch(model, *, memory_ratio: int = 4, **_: Any):
    """FINCH (2024) — prompt-guided compression that requires a delimiter
    token between context and question. Our NIAH harness does not insert
    one, so this baseline is recorded as REGISTERED-NOT-MEASURED: running
    it requires either modifying the harness (breaks apples-to-apples) or
    fabricating a delimiter token (breaks the published recipe). Raise a
    clear error to surface this cleanly to the runner."""
    raise NotImplementedError(
        "FinchPress requires `context + <delim> + question` prompt format; "
        "the NIAH harness emits `context + question` without a delimiter. "
        "Running Finch on this benchmark would require modifying the prompt "
        "template, which violates the apples-to-apples baseline contract."
    )


REGISTRY: dict[str, Callable[..., Optional[object]]] = {
    "full": _full,
    "h2o": _h2o,
    "streamingllm": _streamingllm,
    "snapkv": _snapkv,
    "kivi": _kivi,
    # newer baselines (all SDPA-compatible, run at 32K)
    "expected_attention": _expected_attention,
    "adakv": _adakv,
    "think": _think,
    "pyramidkv": _pyramidkv,
    "tova": _tova,
    # three more 2024-2025 presses
    "compactor": _compactor,
    "criticalkv": _criticalkv,
    "finch": _finch,  # registered but raises NotImplementedError (prompt-format incompatible)
}

__all__ = ["REGISTRY"]
