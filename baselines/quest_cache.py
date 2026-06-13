"""Quest (Tang et al., ICML 2024) baseline under the HALO repo's
attention-interface registry protocol.

Why not under kvpress
---------------------
kvpress's ``ScorerPress`` API is a *prefill-once-prune* protocol: the
score function fires exactly once at the end of prefill, then the
selected positions are kept forever. Quest is fundamentally different:
it is *per-step query-aware* --- at each decoding step, the current
query $q$ scores every page and picks the top-$K$. So Quest cannot be
expressed as a ``ScorerPress`` --- but it fits cleanly under the same
``ALL_ATTENTION_FUNCTIONS`` registry mechanism we built for
:class:`halo.kv_cache_chunked.HALOCacheChunked`.

Algorithm (matches the original paper)
--------------------------------------
1. **Paging.** Split $(K, V)$ into pages of size ``page_size`` (default
   16). For each finalised page (i.e. once all ``page_size`` tokens
   of the page are present), store per-channel $K_{\min}, K_{\max}$
   metadata. The last partial page has no metadata and is always
   included in the attention call.
2. **Per-step scoring.** On every decoding-step attention call, the
   query $q$ scores each page via
   $\;s_p \;=\; \sum_d \max(q_d \cdot K_{\min}[p,d],\; q_d \cdot K_{\max}[p,d])$.
   This is an *upper bound* on the true $q \cdot K$ inner product for
   any actual key in the page.
3. **Top-$K$ selection.** Pick the top-$K$ pages by $s_p$ per
   $(\text{batch}, \text{KV-head}, \text{query-token})$. Always include
   (a) the first ``sink_pages`` full pages (anchor sinks) and (b) the
   last partial page if present.
4. **Masked SDPA.** Build a per-position binary mask from the selected
   pages and feed it as an additive attention mask to PyTorch's SDPA.

GQA handling
------------
``Qwen2.5-7B`` is GQA with $H_q = 28$, $H_{kv} = 4$. Metadata is stored
per KV head ($H_{kv} = 4$). At scoring time, $q$ (shape
$(B, H_q, T_q, D)$) is reshaped to $(B, H_{kv}, \text{rep}, T_q, D)$ and
mean-pooled to $(B, H_{kv}, T_q, D)$ before computing $s_p$, matching
the original paper's per-KV-group selection.

Prefill behaviour
-----------------
At prefill ($T_q > 1$), the interface falls through to one-shot SDPA on
the full $(K, V)$. This matches Quest's original implementation: the
query-aware selection is decode-only, so prefill quality is unaffected.

Lossless-vs-lossy framing (vs. HALO Path D)
-------------------------------------------
Quest is *lossy per step* --- the un-selected pages contribute zero to
the current step's softmax --- but *recoverable across steps* because
the selection is recomputed each step. HALO Path D is *algebraically identity-preserving per
step* (LSE-merge sums over every chunk). This is the architectural
delta we report in \\Cref{sec:related}.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

try:
    from transformers.cache_utils import DynamicCache as _BaseCache
    _HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover
    class _BaseCache:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.key_cache: list = []
            self.value_cache: list = []

        def update(self, key_states, value_states, layer_idx, *args, **kwargs):
            import torch as _t
            while len(self.key_cache) <= layer_idx:
                self.key_cache.append(None)
                self.value_cache.append(None)
            prev_k = self.key_cache[layer_idx]
            prev_v = self.value_cache[layer_idx]
            if prev_k is None:
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = _t.cat([prev_k, key_states], dim=-2)
                self.value_cache[layer_idx] = _t.cat([prev_v, value_states], dim=-2)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        def get_seq_length(self, layer_idx: int = 0) -> int:
            if layer_idx >= len(self.key_cache) or self.key_cache[layer_idx] is None:
                return 0
            return self.key_cache[layer_idx].shape[-2]

    _HAS_TRANSFORMERS = False


if TYPE_CHECKING:  # pragma: no cover
    import torch
    from transformers import PreTrainedModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class QuestConfig:
    """User-facing Quest knobs.

    ``memory_ratio`` is mapped to the fraction of pages selected per step:
    ``top_k = max(min_pages_selected, ceil(num_pages / memory_ratio))``.
    Always-include (sink + partial) pages count *against* this budget --
    same convention as kvpress's eviction-style baselines.
    """

    page_size: int = 16
    sink_pages: int = 1
    memory_ratio: float = 4.0
    min_pages_selected: int = 2

    def top_k_for(self, num_pages_total: int) -> int:
        budgeted = int(math.ceil(num_pages_total / max(1.0, self.memory_ratio)))
        return max(self.min_pages_selected, budgeted)


# ---------------------------------------------------------------------------
# Cache subclass
# ---------------------------------------------------------------------------


class QuestPagedCache(_BaseCache):  # type: ignore[misc]
    """KV cache with per-page K min/max metadata.

    The cache is otherwise a strict superset of ``DynamicCache``: on every
    ``update`` we call ``super().update`` to keep the parent's per-layer
    tensors consistent with what the model expects, then refresh page
    metadata for any newly-finalised pages.
    """

    def __init__(self, config: Optional[QuestConfig] = None) -> None:
        super().__init__()
        self.cfg = config or QuestConfig()
        # Per-layer page metadata.
        # _page_meta[layer_idx] = {
        #     "k_min": (B, H_kv, num_full_pages, D),
        #     "k_max": (B, H_kv, num_full_pages, D),
        #     "num_full_pages": int,
        # }
        self._page_meta: dict[int, dict] = {}
        # Decode-time telemetry.
        self._pages_selected_total = 0
        self._pages_visited_total = 0
        self._selection_steps = 0

    # --------------------------------- HF Cache API
    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        K_full, V_full = super().update(key_states, value_states, layer_idx,
                                        *args, **kwargs)
        self._refresh_page_metadata(layer_idx, K_full)
        return K_full, V_full

    def get_max_cache_shape(self):
        return None

    # --------------------------------- layout helpers
    def _iter_layer_kv(self):
        """Yield (layer_idx, layer_obj, K, V) regardless of cache layout."""
        if hasattr(self, "layers") and isinstance(self.layers, list):
            for i, layer in enumerate(self.layers):
                K = getattr(layer, "keys", None)
                if K is None:
                    K = getattr(layer, "key_cache", None)
                V = getattr(layer, "values", None)
                if V is None:
                    V = getattr(layer, "value_cache", None)
                yield i, layer, K, V
            return
        if hasattr(self, "key_cache") and isinstance(self.key_cache, list):
            for i in range(len(self.key_cache)):
                yield i, None, self.key_cache[i], self.value_cache[i]

    def _get_layer_kv(self, layer_idx):
        for li, _, k, v in self._iter_layer_kv():
            if li == layer_idx:
                return k, v
        return None, None

    # --------------------------------- metadata
    def _refresh_page_metadata(self, layer_idx, K_full):
        """Compute K_min / K_max for any newly-finalised pages.

        A page is finalised once all ``page_size`` of its tokens are
        present in ``K_full``. We never recompute metadata for already
        finalised pages, so this is $O(\\Delta T)$ per update.
        """
        import torch
        if K_full is None or K_full.numel() == 0:
            return
        B, H_kv, T, D = K_full.shape
        page_size = self.cfg.page_size
        num_full_pages = T // page_size
        meta = self._page_meta.setdefault(layer_idx, {})
        prev = int(meta.get("num_full_pages", 0))
        if num_full_pages <= prev:
            return
        new_k = K_full[..., prev * page_size: num_full_pages * page_size, :]
        num_new = num_full_pages - prev
        new_k = new_k.view(B, H_kv, num_new, page_size, D)
        new_k_min = new_k.amin(dim=-2).detach()
        new_k_max = new_k.amax(dim=-2).detach()
        if "k_min" not in meta:
            meta["k_min"] = new_k_min
            meta["k_max"] = new_k_max
        else:
            meta["k_min"] = torch.cat([meta["k_min"], new_k_min], dim=-2)
            meta["k_max"] = torch.cat([meta["k_max"], new_k_max], dim=-2)
        meta["num_full_pages"] = num_full_pages

    # --------------------------------- scoring
    def _score_pages(self, q, layer_idx):
        """Return Quest's per-page upper-bound scores.

        Output shape: ``(B, H_kv, T_q, num_full_pages)`` or ``None`` if
        no metadata yet. Score is computed in fp32 for accumulator
        stability and cast back to ``q.dtype`` on return.
        """
        import torch
        meta = self._page_meta.get(layer_idx)
        if meta is None or meta.get("num_full_pages", 0) == 0:
            return None
        k_min = meta["k_min"]
        k_max = meta["k_max"]
        H, H_kv = q.shape[1], k_min.shape[1]
        if H != H_kv:
            assert H % H_kv == 0, (
                f"GQA mismatch: q has {H} heads, metadata has {H_kv} KV heads"
            )
            rep = H // H_kv
            q_grouped = q.view(q.shape[0], H_kv, rep,
                               q.shape[2], q.shape[3]).mean(dim=2)
        else:
            q_grouped = q
        # q_grouped: (B, H_kv, T_q, D); k_min/max: (B, H_kv, P, D).
        q_exp = q_grouped.unsqueeze(-2).float()        # (B, H_kv, T_q, 1, D)
        k_min_exp = k_min.unsqueeze(-3).float()        # (B, H_kv, 1, P, D)
        k_max_exp = k_max.unsqueeze(-3).float()
        a = q_exp * k_min_exp
        b = q_exp * k_max_exp
        per_d_max = torch.maximum(a, b)                # (B, H_kv, T_q, P, D)
        scores = per_d_max.sum(dim=-1)                 # (B, H_kv, T_q, P)
        return scores

    # --------------------------------- attention
    def quest_attention(self, q, *, layer_idx, scaling=None):
        """Per-step query-aware top-K page attention (decode-only).

        For ``T_q == 1`` this is the standard Quest behaviour. For
        ``T_q > 1`` (prefill) the caller should fall through to one-shot
        SDPA via the registered interface; we still support it here
        with a per-query top-K selection so the function is well-defined
        for unit testing.
        """
        import torch
        K_full, V_full = self._get_layer_kv(layer_idx)
        if K_full is None or K_full.numel() == 0:
            B, H, T_q, D = q.shape
            return torch.zeros(B, H, T_q, D, dtype=q.dtype, device=q.device)
        B, H_kv, T, D = K_full.shape
        H = q.shape[1]
        T_q = q.shape[-2]
        page_size = self.cfg.page_size
        num_full_pages = T // page_size
        has_partial = (T % page_size) != 0
        num_pages_total = num_full_pages + (1 if has_partial else 0)

        scores = self._score_pages(q, layer_idx) if num_full_pages > 0 else None

        # Page-level binary mask: True = include in attention.
        page_mask = torch.zeros(B, H_kv, T_q, num_pages_total,
                                dtype=torch.bool, device=q.device)
        sink = min(self.cfg.sink_pages, num_pages_total)
        if sink > 0:
            page_mask[..., :sink] = True
        if has_partial:
            page_mask[..., -1] = True

        if scores is not None:
            # Top-K budget counts the sink + partial pages.
            top_k_total = self.cfg.top_k_for(num_pages_total)
            always = sink + (1 if has_partial else 0)
            k_to_pick = max(0, top_k_total - always)
            if k_to_pick > 0:
                # Exclude sink pages from the score (they are already in
                # the mask). We score the rest and pick top-k of them.
                scoreable = scores.clone()
                if sink > 0:
                    scoreable[..., :sink] = float("-inf")
                k_eff = min(k_to_pick, max(0, num_full_pages - sink))
                if k_eff > 0:
                    _, picked = scoreable.topk(k_eff, dim=-1)
                    page_mask.scatter_(-1, picked, True)

        # Upsample page mask to per-position mask.
        position_mask = page_mask.repeat_interleave(page_size, dim=-1)
        position_mask = position_mask[..., :T]

        # Causal mask (only relevant if T_q > 1).
        if T_q > 1:
            i = torch.arange(T_q, device=q.device).unsqueeze(-1)
            j = torch.arange(T, device=q.device).unsqueeze(0)
            q_pos_offset = T - T_q
            causal = (q_pos_offset + i) >= j
            position_mask = position_mask & causal

        # GQA broadcast.
        rep = H // H_kv
        if rep > 1:
            position_mask = position_mask.repeat_interleave(rep, dim=1)
            K = K_full.repeat_interleave(rep, dim=1)
            V = V_full.repeat_interleave(rep, dim=1)
        else:
            K, V = K_full, V_full

        # Telemetry (decode steps only).
        if T_q == 1:
            self._pages_selected_total += int(page_mask.sum().item())
            self._pages_visited_total += int(
                B * H_kv * T_q * num_pages_total
            )
            self._selection_steps += 1

        # Build additive mask in the same dtype as q.
        # Using bf16 -inf as the masked value is well-defined for SDPA.
        neg_inf = torch.tensor(float("-inf"), dtype=q.dtype, device=q.device)
        zero = torch.tensor(0.0, dtype=q.dtype, device=q.device)
        attn_bias = torch.where(position_mask, zero, neg_inf)

        if scaling is None:
            scaling = 1.0 / math.sqrt(q.shape[-1])
        out = torch.nn.functional.scaled_dot_product_attention(
            q, K, V, attn_mask=attn_bias, scale=scaling, is_causal=False,
        )
        return out

    # --------------------------------- API
    def telemetry(self) -> dict:
        avg_frac = (self._pages_selected_total
                    / max(1, self._pages_visited_total))
        return {
            "pages_selected_total": int(self._pages_selected_total),
            "pages_visited_total": int(self._pages_visited_total),
            "selection_steps": int(self._selection_steps),
            "avg_pages_selected_frac": float(avg_frac),
        }

    def reset(self) -> None:
        try:
            _BaseCache.__init__(self)
        except Exception:
            if hasattr(self, "key_cache") and isinstance(self.key_cache, list):
                self.key_cache.clear()
            if hasattr(self, "value_cache") and isinstance(self.value_cache, list):
                self.value_cache.clear()
        self._page_meta.clear()
        self._pages_selected_total = 0
        self._pages_visited_total = 0
        self._selection_steps = 0


# ---------------------------------------------------------------------------
# Wrap entrypoint
# ---------------------------------------------------------------------------


def wrap_with_quest(model, config: Optional[QuestConfig] = None):
    """Install :class:`QuestPagedCache` on ``model`` and register the
    ``quest`` attention interface (transformers $\\geq$ 5.0). The
    interface routes decoding-step attention through
    :meth:`QuestPagedCache.quest_attention` (top-$K$ page selection)
    and prefill through one-shot SDPA.

    Returns the same model object with attributes
    ``model._quest_cache`` and ``model._quest_config`` set, and
    ``model.generate`` patched to thread the cache automatically.
    """
    import torch  # noqa: F401

    cfg = config or QuestConfig()
    cache = QuestPagedCache(cfg)
    model._quest_cache = cache               # type: ignore[attr-defined]
    model._quest_config = cfg                # type: ignore[attr-defined]

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError:
        raise RuntimeError(
            "wrap_with_quest requires transformers >= 5.0 (ALL_ATTENTION_FUNCTIONS registry)."
        )

    def quest_attention_interface(module, query, key, value,
                                  attention_mask=None, dropout=0.0,
                                  scaling=None, **kwargs):
        layer_idx = module.layer_idx
        T_q = query.shape[-2]
        if T_q > 1:
            # Prefill — delegate to SDPA exactly as the unwrapped model
            # would. This matches the original Quest paper.
            try:
                sdpa_fn = ALL_ATTENTION_FUNCTIONS["sdpa"]
                out, _ = sdpa_fn(module, query, key, value, attention_mask,
                                 dropout=dropout, scaling=scaling, **kwargs)
                return out, None
            except Exception:
                import math
                H, H_kv = query.shape[1], key.shape[1]
                if H != H_kv:
                    rep = H // H_kv
                    key = key.repeat_interleave(rep, dim=1)
                    value = value.repeat_interleave(rep, dim=1)
                if scaling is None:
                    scaling = 1.0 / math.sqrt(query.shape[-1])
                is_causal = attention_mask is None and T_q > 1
                out = torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0,
                    is_causal=is_causal, scale=scaling,
                )
                return out.transpose(1, 2).contiguous(), None
        # Decoding — Quest top-K page selection.
        out = cache.quest_attention(query, layer_idx=layer_idx, scaling=scaling)
        out = out.transpose(1, 2).contiguous()
        return out, None

    ALL_ATTENTION_FUNCTIONS["quest"] = quest_attention_interface

    if hasattr(model, "config"):
        model.config._attn_implementation = "quest"
    for sub in model.modules():
        if hasattr(sub, "config") and sub is not model:
            try:
                sub.config._attn_implementation = "quest"
            except Exception:
                pass

    if not getattr(model, "_quest_generate_patched", False):
        original_generate = model.generate

        def quest_generate(*args, **kwargs):
            user_cache = kwargs.get("past_key_values")
            if user_cache is not cache:
                cache.reset()
                kwargs.setdefault("past_key_values", cache)
            kwargs.setdefault("use_cache", True)
            return original_generate(*args, **kwargs)

        model.generate = quest_generate
        model._quest_generate_patched = True

    return model


__all__ = ["QuestConfig", "QuestPagedCache", "wrap_with_quest"]
