"""ShadowKV model wrapper.

Installs a ShadowKVCache and registers a custom attention function that
dispatches to `cache.shadow_attention(q, layer_idx=...)` at decode time.
At prefill (T_q > 1) falls through to one-shot SDPA so the prefill quality
is unaffected (matches the paper's design).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .shadowkv_cache import ShadowKVCache, ShadowKVConfig


@dataclass
class ShadowKVWrapperConfig:
    rank: int = 8
    page_size: int = 16
    top_k_pages: int = 32
    sink_pages: int = 1
    cpu_offload_value: bool = True

    def to_cache_config(self) -> ShadowKVConfig:
        return ShadowKVConfig(
            rank=self.rank, page_size=self.page_size,
            top_k_pages=self.top_k_pages, sink_pages=self.sink_pages,
            cpu_offload_value=self.cpu_offload_value,
        )


def make_shadowkv_cache(cfg: Optional[ShadowKVWrapperConfig] = None) -> ShadowKVCache:
    """Factory: returns a fresh ShadowKVCache each call (matches HF idiom)."""
    cfg = cfg or ShadowKVWrapperConfig()
    return ShadowKVCache(cfg.to_cache_config())


def wrap_with_shadowkv(model, cfg: Optional[ShadowKVWrapperConfig] = None):
    """Register 'shadow_sdpa' and switch model+layers to use it.

    Matches the wiring in :func:`halo.policy.wrap_with_halo` — see comments
    there for why both ``model.config._attn_implementation`` and per-layer
    ``module.config._attn_implementation`` must be set.

    Caller is responsible for passing a fresh ShadowKVCache via
    ``past_key_values=`` to each ``model.generate`` call.
    """
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    cfg = cfg or ShadowKVWrapperConfig()

    def shadow_attn(module, query, key, value, attention_mask, **kw):
        """ALL_ATTENTION_FUNCTIONS attention hook.

        Returns (attn_output, attn_weights). Decode-time dispatch goes
        through ``cache.shadow_attention``; prefill and missing-cache
        fall through to SDPA on the supplied (q,k,v).
        """
        cache = kw.get("past_key_value", None)
        layer_idx = getattr(module, "layer_idx", None)
        is_decode = query.shape[-2] == 1

        if (isinstance(cache, ShadowKVCache) and is_decode and layer_idx is not None
                and layer_idx in cache._svd):
            out = cache.shadow_attention(query, layer_idx=layer_idx,
                                         scaling=kw.get("scaling"))
            if out is not None:
                return out, None

        scale = kw.get("scaling")
        if scale is None:
            scale = 1.0 / (query.shape[-1] ** 0.5)
        attn_logits = torch.matmul(query, key.transpose(-1, -2)) * scale
        if attention_mask is not None:
            attn_logits = attn_logits + attention_mask
        attn_w = attn_logits.softmax(dim=-1, dtype=torch.float32).to(query.dtype)
        out = torch.matmul(attn_w, value)
        return out, None

    interface_name = "shadow_sdpa"
    ALL_ATTENTION_FUNCTIONS[interface_name] = shadow_attn

    # Switch the model and all attention modules to the new interface
    # (mirrors halo.policy.wrap_with_halo's per-layer propagation).
    if hasattr(model, "config"):
        model.config._attn_implementation = interface_name
    for module in model.modules():
        if hasattr(module, "config") and module is not model:
            try:
                module.config._attn_implementation = interface_name
            except Exception:
                pass
        # Some HF versions cache the attn impl on the attention layer itself.
        if hasattr(module, "_attn_implementation"):
            try:
                module._attn_implementation = interface_name
            except Exception:
                pass

    return cfg
