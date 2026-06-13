"""Top-level HALO policy: configuration + ``wrap_with_halo`` entrypoint.

This file glues the three submodules (Scorer, Demoter, Refetcher) onto a
HuggingFace ``CausalLM`` by replacing its KV cache with :class:`HALOCache` and
hooking attention forward calls so the scorer sees per-step attention scores.

The implementation is intentionally light at this stage of the project — it is
a *framework* matching the paper's §4 component decomposition, not a final
optimized kernel-level implementation. The Scorer, Demoter, Refetcher and
TieredStorage are CPU-correct and unit-tested; the runtime wrapper here is
verified against full-attention output equivalence in :mod:`tests.test_policy`.
End-to-end GPU verification is the user's responsibility (see ``runs.md``).
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Optional, Sequence

from halo.demoter import HALODemoter
from halo.kv_cache import HALOCache
from halo.memory_tier import MemoryTier, TieredStorage
from halo.refetcher import HALORefetcher
from halo.scorer import HALOScorer

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from transformers import PreTrainedModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class HALOConfig:
    """User-facing configuration.

    Defaults reflect the values reported in the headline experiments of the paper:
    top-10% hot ratio (Finding 1), refresh window of 64 steps (Finding 2), and
    the closed-form `α·attn + β·recency + γ·sink` scorer (Finding 3).
    """

    # --- Scoring / hot-cold classification (§4.1) ---
    hot_ratio: float = 0.10
    """Fraction of KV positions kept on the GPU tier per layer."""

    score_alpha: float = 1.0
    """Weight on attention magnitude."""

    score_beta: float = 0.5
    """Weight on recency (geometric decay over decoding steps)."""

    score_gamma: float = 2.0
    """Boost applied to the first ``sink_tokens`` positions (StreamingLLM-style)."""

    sink_tokens: int = 4
    """Always-hot prefix length (anchor sinks)."""

    refresh_window: int = 64
    """Recompute the hot-set every N decoding steps. F2 says this is safe."""

    # --- Tiering (§4.2-4.3) ---
    tiers: Sequence[str] = ("gpu", "dram")
    """Ordered tier list. Add ``"nvme"`` for >1M-token contexts."""

    block_size: int = 32
    """KV granularity for demotion / refetch (in tokens)."""

    nvme_path: Optional[str] = None
    """Optional path for the NVMe tier; auto-allocated if None."""

    async_refetch: bool = True
    """Whether the refetcher overlaps with model compute (recommended)."""

    lookahead: int = 1
    """Steps to look ahead when predicting future hot positions."""

    # --- Misc ---
    layerwise_budget: str = "uniform"
    """One of ``"uniform" | "pyramid" | "learned"``. See §4.1."""

    eviction: bool = False
    """If True, the wrapper installs :class:`HALOCacheEvict` (cold positions
    are physically zeroed in the returned (K, V), giving real GPU-side
    memory effects at the cost of bit-exactness). The default False keeps
    the identity-preserving :class:`HALOCache`. See §4.5 of the paper."""

    chunked: bool = False
    """If True, the wrapper installs :class:`HALOCacheChunked` and registers
    the ``halo_chunked`` attention interface. Cold KV migrates to pinned
    host DRAM on first decoding step; the attention call streams chunks
    through a small GPU staging buffer with log-sum-exp merging
    (\\Cref{prop:chunked-lossless}: algebraically identical to full
    attention in real arithmetic; per-step bit-equivalent on a fixed
    KV state in fp32. Free-running fp32 trajectories may diverge —
    see sec:appendix-qa2-fp32). This is Path D in
    \\Cref{tab:three-impls} — the identity-preserving variant.
    Mutually exclusive with ``eviction``. Default False (Path A)."""

    chunk_size: int = 512
    """Block size (in tokens) for the chunked attention loop when
    ``chunked=True``. The GPU staging buffer is one chunk wide."""

    recent_window: int = 64
    """Number of most-recent positions kept on GPU when ``chunked=True``
    (everything older goes to the DRAM cold tier)."""

    use_triton: bool = False
    """If True (and ``chunked=True`` and CUDA + Triton available),
    `HALOCacheChunked.compute_attention` swaps the per-chunk
    `matmul`+`logsumexp`+`matmul`+`logaddexp` Python loop for a single
    fused Triton kernel per chunk (`halo/triton_chunked.py`). Numerically
    matches the reference path to <= 5e-3 in bf16 (verified by
    ``tests/test_triton_chunked.py``); the per-chunk launch count drops
    from ~4 to 1, recovering most of the 21x wall-clock gap reported in
    \\Cref{tab:infinitebench-peak}. Off by default for byte-exact fp32
    reproduction of the reference path."""

    log_dir: Optional[str] = None
    """If set, periodically dumps hot-set telemetry (Jaccard, hit-rate, ...)."""

    extra: dict = field(default_factory=dict)
    """Free-form ablation knobs (used by the experiment harness)."""

    # ------------------------------------------------------------------ utils

    def make_storage(self, *, num_layers: int, num_kv_heads: int, head_dim: int,
                     dtype: "torch.dtype", device: "torch.device") -> TieredStorage:
        """Materialize a :class:`TieredStorage` according to the configured tiers."""
        tiers = [MemoryTier.from_name(name) for name in self.tiers]
        return TieredStorage(
            tiers=tiers,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=self.block_size,
            dtype=dtype,
            device=device,
            nvme_path=self.nvme_path,
        )

    def with_overrides(self, overrides: dict) -> "HALOConfig":
        """Return a copy with the supplied field overrides applied.

        Used by the ablation harness which streams per-variant overrides via the
        :envvar:`HALO_OVERRIDES` environment variable.
        """
        valid = {f for f in self.__dataclass_fields__}
        clean = {k: v for k, v in overrides.items() if k in valid}
        if extras := {k: v for k, v in overrides.items() if k not in valid}:
            clean["extra"] = {**self.extra, **extras}
        return replace(self, **clean)


# ---------------------------------------------------------------------------
# Override parsing — honored by ``scripts/run_*.py`` via ``$HALO_OVERRIDES``.
# ---------------------------------------------------------------------------


def parse_overrides(s: Optional[str] = None) -> dict[str, Any]:
    """Parse ``"k1=v1,k2=v2,..."`` (matching ``ablations.sh``) into a dict.

    Values are passed through :func:`ast.literal_eval` when possible so that
    Python literals like ``[gpu, dram]``, ``False``, ``0.05`` round-trip
    correctly. Unparseable values stay as strings. Commas inside ``[...]``,
    ``(...)``, ``{...}``, or quoted strings are preserved.
    """
    if s is None:
        s = os.environ.get("HALO_OVERRIDES", "").strip()
    if not s:
        return {}

    pairs: list[str] = []
    buf: list[str] = []
    depth = 0
    in_str = None
    for ch in s:
        if in_str:
            buf.append(ch)
            if ch == in_str and (len(buf) < 2 or buf[-2] != "\\"):
                in_str = None
            continue
        if ch in ("'", '"'):
            in_str = ch
            buf.append(ch)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            pairs.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        pairs.append("".join(buf).strip())

    out: dict[str, Any] = {}
    for kv in pairs:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        k, v = k.strip(), v.strip()
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Wrapping entrypoint
# ---------------------------------------------------------------------------


def wrap_with_halo(model: "PreTrainedModel",
                   config: Optional[HALOConfig] = None) -> "PreTrainedModel":
    """Replace ``model``'s KV cache with :class:`HALOCache` and install hooks.

    Parameters
    ----------
    model:
        Any HuggingFace ``CausalLM`` whose attention layers expose a
        ``past_key_value`` of type ``transformers.Cache``. (Llama-3, Qwen-2.5,
        Mistral all qualify.)
    config:
        Optional :class:`HALOConfig`. If ``None``, defaults are used; the
        :envvar:`HALO_OVERRIDES` env var (if set) is then applied on top.

    Returns
    -------
    The same model object with HALO installed in place. Forward is functionally
    equivalent to the original model up to the choice of hot ratio: with
    ``hot_ratio = 1.0`` HALO is a strict identity over full attention.
    """
    if config is None:
        config = HALOConfig()
    if env_overrides := parse_overrides():
        config = config.with_overrides(env_overrides)

    # Lazy imports keep the package importable in environments without torch.
    import torch  # noqa: F401  (required by HF model objects)

    cfg_obj = getattr(model, "config", None)
    if cfg_obj is None:  # pragma: no cover
        raise ValueError("`model.config` is missing; cannot wire HALO without architecture metadata.")

    num_layers = getattr(cfg_obj, "num_hidden_layers", None) or getattr(cfg_obj, "n_layer")
    num_kv_heads = (
        getattr(cfg_obj, "num_key_value_heads", None)
        or getattr(cfg_obj, "num_attention_heads")
    )
    head_dim = getattr(cfg_obj, "head_dim", None) or (
        cfg_obj.hidden_size // cfg_obj.num_attention_heads
    )

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    storage = config.make_storage(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )

    scorer = HALOScorer(config)
    demoter = HALODemoter(config, storage=storage)
    refetcher = HALORefetcher(config, storage=storage)
    if config.chunked and config.eviction:
        raise ValueError(
            "HALOConfig: cannot enable both `chunked` (Path D) and `eviction` (Path C). "
            "Pick one — chunked is the identity-preserving variant (Prop 4.5 i,ii), "
            "eviction is the strict-prune variant. See paper §4.5 Tab.3 for the taxonomy."
        )
    if config.chunked:
        from halo.kv_cache_chunked import HALOCacheChunked
        cache = HALOCacheChunked(
            config=config, storage=storage, scorer=scorer,
            demoter=demoter, refetcher=refetcher,
            chunk_size=config.chunk_size, recent_window=config.recent_window,
            use_triton=getattr(config, "use_triton", False),
        )
    elif config.eviction:
        from halo.kv_cache_evict import HALOCacheEvict
        cache = HALOCacheEvict(config=config, storage=storage,
                              scorer=scorer, demoter=demoter, refetcher=refetcher)
    else:
        cache = HALOCache(config=config, storage=storage,
                         scorer=scorer, demoter=demoter, refetcher=refetcher)

    # Stash on the model so generate() picks it up (mirrors ``model.past_key_values``).
    model._halo_config = config           # type: ignore[attr-defined]
    model._halo_cache = cache             # type: ignore[attr-defined]
    model._halo_scorer = scorer           # type: ignore[attr-defined]
    model._halo_demoter = demoter         # type: ignore[attr-defined]
    model._halo_refetcher = refetcher     # type: ignore[attr-defined]
    model._halo_storage = storage         # type: ignore[attr-defined]

    # Patch generate() to inject our cache. We don't replace it — we wrap.
    _install_generate_hook(model)

    # Path D (chunked) requires installing a custom attention interface
    # that intercepts the SDPA call and routes through ``cache.compute_attention``.
    if config.chunked:
        _install_chunked_attention_interface(model, cache)

    return model


# ---------------------------------------------------------------------------
# Chunked-attention path (Path D)
# ---------------------------------------------------------------------------


def _install_chunked_attention_interface(model: "PreTrainedModel", cache) -> None:
    """Register the ``halo_chunked`` attention interface and switch the model
    to use it.

    The interface is called by ``Qwen2Attention.forward`` (and the analogous
    ``LlamaAttention``) after ``past_key_values.update`` returns. For
    :class:`HALOCacheChunked` in warmup mode, ``update`` returns the full
    $(K, V)$ on GPU and we delegate to a one-shot SDPA call — identical
    behaviour to the SDPA backend. Once the cache transitions to chunked
    mode, ``update`` returns only the recent on-GPU rows and the interface
    routes through ``cache.compute_attention`` to bring in cold chunks
    from DRAM with LSE-merging.

    The result is the title's "identity-preserving offloading" contract
    observed end-to-end: every position contributes to softmax through
    the LSE merge (``prop:chunked-lossless``: algebraically identical to
    Full in real arithmetic; per-step bit-equivalent on fixed KV state
    in fp32; free-running fp32 may diverge by ULP compound) while
    peak GPU residency is sub-linear in context length
    (\\Cref{tab:peak-memory-curve}).
    """
    import torch

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError:  # pragma: no cover
        # Older transformers versions don't expose the registry; fall back
        # to monkey-patching each attention layer's forward.
        _patch_per_layer_attention(model, cache)
        return

    def halo_chunked_interface(module, query, key, value,
                                attention_mask=None, dropout=0.0,
                                scaling=None, **kwargs):
        """Custom attention forward called by Qwen2Attention / LlamaAttention.

        Shapes (in transformers 5.x): query is (B, H, T_q, D), key/value
        are (B, H_kv, T_kv, D). For HALOCacheChunked in chunked mode the
        provided key/value cover only the recent_window; for warmup mode
        they're the full sequence.
        """
        layer_idx = module.layer_idx
        if cache._mode == "warmup":
            # One-shot SDPA over the full (K, V) returned by update().
            return cache._one_shot_attention_for_module(
                module, query, key, value, attention_mask=attention_mask,
                scaling=scaling,
            ), None
        # Chunked mode: compute_attention handles cold + recent.
        out = cache.compute_attention(query, layer_idx=layer_idx, scaling=scaling)
        # transformers expects (B, T_q, H, D) layout returned (transposed).
        out = out.transpose(1, 2).contiguous()
        return out, None

    # Re-register on every wrap so the closure captures THIS call's cache
    # (a previous wrap may have left a stale closure pointing at an older
    # cache instance). Use a per-model-id key so concurrent wraps don't
    # stomp each other.
    interface_name = "halo_chunked"
    ALL_ATTENTION_FUNCTIONS[interface_name] = halo_chunked_interface

    # Switch the model and all attention modules to use the new interface.
    if hasattr(model, "config"):
        model.config._attn_implementation = interface_name
    # Also propagate per-layer (some transformer versions look it up on
    # the layer's config).
    for module in model.modules():
        if hasattr(module, "config") and module is not model:
            try:
                module.config._attn_implementation = interface_name
            except Exception:
                pass

    # Attach a one-shot fallback to the cache for warmup-mode calls.
    _attach_one_shot_fallback(cache)


def _patch_per_layer_attention(model, cache) -> None:
    """Fallback for transformers < 5.0: monkey-patch each attention forward."""
    raise NotImplementedError(
        "Per-layer monkey-patch is not implemented; please use transformers >= 5.0 "
        "with ALL_ATTENTION_FUNCTIONS registry."
    )


def _attach_one_shot_fallback(cache) -> None:
    """Attach a method that does standard SDPA when cache is in warmup mode.

    We delegate to transformers' own ``sdpa_attention_forward`` so that
    the warmup-mode path is byte-equivalent to the unwrapped SDPA model.
    This matters for the identity invariant tests.
    """
    import types

    def _one_shot_attention_for_module(self_, module, query, key, value,
                                        *, attention_mask=None, scaling=None,
                                        **kwargs):
        # Try to use the registered ``sdpa`` attention interface; fall back
        # to plain torch SDPA if unavailable.
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            sdpa_fn = ALL_ATTENTION_FUNCTIONS["sdpa"]
            out, _ = sdpa_fn(
                module, query, key, value, attention_mask,
                dropout=0.0, scaling=scaling, **kwargs,
            )
            return out
        except Exception:
            import math
            import torch
            H, H_kv = query.shape[1], key.shape[1]
            if H != H_kv:
                rep = H // H_kv
                key = key.repeat_interleave(rep, dim=1)
                value = value.repeat_interleave(rep, dim=1)
            if scaling is None:
                scaling = 1.0 / math.sqrt(query.shape[-1])
            is_causal = attention_mask is None and query.shape[-2] > 1
            out = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0,
                is_causal=is_causal, scale=scaling,
            )
            return out.transpose(1, 2).contiguous()

    cache._one_shot_attention_for_module = types.MethodType(
        _one_shot_attention_for_module, cache
    )


def _install_generate_hook(model: "PreTrainedModel") -> None:
    """Wrap ``model.generate`` so HALOCache is threaded through ``past_key_values``.

    HF's ``generate`` already reuses ``past_key_values`` if supplied; we just
    inject a fresh :class:`HALOCache` per call so each generation gets its own
    hot/cold state.
    """
    if getattr(model, "_halo_generate_patched", False):
        return

    original_generate = model.generate

    def halo_generate(*args, **kwargs):  # type: ignore[no-untyped-def]
        cache = model._halo_cache  # noqa: SLF001
        # Only reset when the caller hasn't explicitly handed us a
        # pre-populated cache. The chunked-prefill harness in
        # ``halo/chunked_prefill.py`` runs prefill outside of generate
        # and passes ``past_key_values=cache``; resetting here would
        # wipe its cold tier and silently fall back to one-shot
        # decoding from scratch.
        user_cache = kwargs.get("past_key_values")
        if user_cache is not cache:
            cache.reset()
            kwargs.setdefault("past_key_values", cache)
        kwargs.setdefault("use_cache", True)
        return original_generate(*args, **kwargs)

    model.generate = halo_generate  # type: ignore[assignment]
    model._halo_generate_patched = True  # type: ignore[attr-defined]
