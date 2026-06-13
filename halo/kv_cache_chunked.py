"""HALOCacheChunked: identity-preserving tier-paged KV cache (Path D, the title's claim).

Architecture
------------
This is the only Path that simultaneously delivers (i) per-step
algebraic identity in real arithmetic between the chunked output
and one-shot full attention (LSE-merge associativity,
``prop:chunked-lossless`` part i; bit-equivalent under fp32 on a
fixed KV state per part ii — free-running fp32 trajectories may
still diverge due to ULP-level compound, see
``sec:appendix-qa2-fp32``) and (ii) sub-linear peak GPU KV
residency in context length. The mechanism: at the boundary between
prefill and decoding, the parent ``DynamicCache``'s GPU-resident
$(K, V)$ are split into a small "recent" view (``recent_window``
positions, kept on GPU) and a "cold" suffix (everything older, moved
to pinned host DRAM). The parent's per-layer tensor is then *replaced*
in-place by the recent view, freeing the cold suffix's GPU memory.
On every subsequent attention call, ``compute_attention`` streams the
cold suffix back chunk-by-chunk via DMA into a single GPU staging
buffer, computing partial $(\\text{out}, \\text{lse})$ for each chunk
and merging them with log-sum-exp. This is mathematically equivalent
to one-shot SDPA on the full $(K, V)$ (Prop. 4.5).

Lifecycle
---------
* **WARMUP mode** (prefill + initial decode steps until the parent
  has >$2 \\cdot \\text{chunk\\_size}$ rows cached): we delegate to
  ``DynamicCache`` and return the full $(K, V)$. The model's
  attention call goes through ``halo_chunked_interface`` which
  detects warmup mode and calls one-shot SDPA. This keeps prefill
  fast (FlashAttention via SDPA) at the cost of not yet saving
  memory.
* **CHUNKED mode** (triggered on the first 1-token decode step after
  the parent exceeds the threshold): :meth:`_peel_to_cold` moves all
  but the last ``recent_window`` rows to host pinned DRAM, replaces
  the parent's GPU tensor with the recent view, and flips the mode.
  Every subsequent decode step (a) appends the new token to the
  recent view via parent's ``update`` and (b) when the recent view
  grows beyond ``recent_window + chunk_size``, peels the oldest
  ``chunk_size`` rows to cold, keeping GPU residency bounded.
  ``compute_attention`` then walks (cold chunks via DMA) + (recent
  chunks on GPU) and merges via LSE.

Result: after the prefill-→-decode boundary, per-layer GPU KV
residency is bounded by ``(recent_window + chunk_size) *
per_token_bytes`` regardless of total cached length.

Numerical safety: LSE reductions and the chunk-merge step are done
in fp32 even when $(K, V)$ are bf16, matching the FlashAttention
recipe. Causal-masked chunks where every key is masked produce
``lse == -inf``; we zero those out before the merge to avoid the
indeterminate ``exp(-inf - (-inf)) = NaN``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from halo.kv_cache import HALOCache

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from halo.demoter import HALODemoter
    from halo.memory_tier import TieredStorage
    from halo.policy import HALOConfig
    from halo.refetcher import HALORefetcher
    from halo.scorer import HALOScorer


@dataclass
class _ChunkedAttentionStats:
    """Telemetry for one attention call on one layer."""

    hot_positions: int = 0
    cold_positions: int = 0
    n_chunks: int = 0
    chunk_size: int = 0
    peak_staging_bytes: int = 0
    used_lse_merge: bool = False
    dma_bytes: int = 0



# Async-DMA implementation (R1 W3, FU_W1; see paper §5 native-hardware
# async-DMA roadmap).
#
# Set HALO_PATH_D_ASYNC_DMA=1 to issue the cold-tier device↔host copies
# on a dedicated CUDA stream with explicit event-based synchronization
# between the cold-peel write and the next-step attention-compute read.
# This is the correct version: the v1 prototype that flipped non_blocking
# without a dedicated stream + event-wait raced the cold-peel against
# the subsequent attention read and dropped accuracy 40%→0% on
# InfiniteBench En.MC n=5 at 16K. The v2 implementation below records
# a CUDA event after each cold-peel DMA and waits on it inside
# `_chunked_compute_attention` before reading the cold tier.
# Off by default to preserve the synchronous-DMA fp32-identity contract;
# the synchronous path remains the canonical reference and is what the
# 109 unit tests cover.
import os as _async_os
_HALO_PATH_D_ASYNC_DMA = _async_os.environ.get("HALO_PATH_D_ASYNC_DMA", "0") == "1"
_HALO_PATH_D_ASYNC_STREAM = None

def _async_dma_stream():
    """Return the dedicated CUDA stream for cold-tier DMA, creating on first call."""
    global _HALO_PATH_D_ASYNC_STREAM
    if _HALO_PATH_D_ASYNC_STREAM is None:
        import torch
        if torch.cuda.is_available():
            _HALO_PATH_D_ASYNC_STREAM = torch.cuda.Stream()
    return _HALO_PATH_D_ASYNC_STREAM

class HALOCacheChunked(HALOCache):
    """KV cache that holds cold positions on pinned host DRAM and streams
    them through a small GPU staging buffer at attention time.

    After the warmup→chunked transition (first decode step on a
    long-enough context), per-layer peak GPU KV residency is
    ``(recent_window + chunk_size) * per_token_bytes``, *independent*
    of total context length.

    Identity invariant: at ``hot_ratio = 1.0`` no positions are
    demoted in the scorer sense; the chunked path is then exercised
    over the entire $(K, V)$ and the output is bit-identical to full
    attention up to floating-point reduction order
    (\\Cref{prop:chunked-lossless}; tests in
    ``tests/test_kv_cache_chunked.py`` for the LSE-merge proof and
    ``tests/test_chunked_wrap_integration.py`` for the end-to-end
    forward).
    """

    def __init__(
        self,
        *,
        config: "HALOConfig",
        storage: "TieredStorage",
        scorer: "HALOScorer",
        demoter: "HALODemoter",
        refetcher: "HALORefetcher",
        chunk_size: int = 512,
        recent_window: int = 64,
        use_triton: bool = False,
    ) -> None:
        super().__init__(config=config, storage=storage, scorer=scorer,
                         demoter=demoter, refetcher=refetcher)
        self.chunk_size = chunk_size
        self.recent_window = recent_window
        # The Triton fast path collapses the per-chunk (matmul, logsumexp,
        # matmul, logaddexp) sequence into a single fused kernel launch.
        # It is opt-in because (a) it requires CUDA + Triton, (b) the
        # reference path is the byte-exact fp32 reference, and (c) it
        # currently only supports T_q == 1 (single-token decode). Chunked
        # prefill (T_q > 1) falls back to the reference path automatically.
        self.use_triton = bool(use_triton)
        self._triton_available = False
        if self.use_triton:
            try:
                from halo.triton_chunked import HAS_TRITON
                import torch as _torch
                self._triton_available = bool(HAS_TRITON) and _torch.cuda.is_available()
            except Exception:
                self._triton_available = False
        # ``warmup`` (parent holds full K, V on GPU; one-shot SDPA) or
        # ``chunked`` (cold suffix on host DRAM; LSE-merge over chunks).
        self._mode = "warmup"
        # Per-layer cold tier (CPU pinned). After the transition,
        # ``self._cold_k[i]`` holds positions ``[0, T_cold)`` for layer
        # ``i``; the parent ``DynamicCache``'s layer holds positions
        # ``[T_cold, T_total)`` on GPU.
        self._cold_k: dict[int, "torch.Tensor"] = {}
        self._cold_v: dict[int, "torch.Tensor"] = {}
        self._stats: dict[int, _ChunkedAttentionStats] = {}
        # Cumulative DMA bytes across the lifecycle for diagnostics.
        self._dma_bytes_cumulative: int = 0
        # FU_W1: per-layer CUDA events for async-DMA correctness.
        # When HALO_PATH_D_ASYNC_DMA=1 the cold-peel d2h copy runs on a
        # dedicated stream; the event is recorded on that stream and
        # waited-on inside _chunked_compute_attention before any read of
        # _cold_k[layer_idx]. Empty dict = no event pending (sync path).
        self._cold_dma_events: dict[int, "torch.cuda.Event"] = {}
        # Optional Quest-style page scorer (attached by
        # ``baselines.quest_path_d.wrap_with_quest_path_d``). If set,
        # ``_peel_to_cold`` consults it to drive query-aware GPU
        # residency instead of the recent-window default. Quality is
        # invariant under any scorer choice by Prop. 4.5; the scorer
        # only affects DMA traffic.
        self._quest_scorer = None
        # Stash latest query per layer (set in compute_attention) so
        # the *next* peel can score pages query-aware.
        self._last_query: dict[int, "torch.Tensor"] = {}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _lse_merge(
        out_a: "torch.Tensor", lse_a: "torch.Tensor",
        out_b: "torch.Tensor", lse_b: "torch.Tensor",
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Combine two partial softmax-attention results (out, lse) into one.

        Standard log-sum-exp merge (cf. FlashAttention's tiled softmax):
        the global output over disjoint key sets $A \\cup B$ is obtained by
        re-weighting per-set partial outputs with $\\exp(\\ell_{\\cdot} - \\ell)$
        where $\\ell = \\mathrm{logaddexp}(\\ell_a, \\ell_b)$.

        Numerical safety: operates in fp32, and handles the case where
        one or both of $\\ell_a, \\ell_b$ are $-\\infty$ (which occurs
        under causal masking when a chunk contributes zero allowed
        positions to a given query). In that case the corresponding
        weight is $0$ — guard against the indeterminate
        $\\exp(-\\infty - (-\\infty))$ that gives NaN.
        """
        import torch

        oa = out_a.to(torch.float32)
        ob = out_b.to(torch.float32)
        la = lse_a.to(torch.float32)
        lb = lse_b.to(torch.float32)
        lse = torch.logaddexp(la, lb)

        a_neg_inf = torch.isneginf(la)
        b_neg_inf = torch.isneginf(lb)

        wa = (la - lse).exp()
        wb = (lb - lse).exp()
        wa = torch.where(a_neg_inf, torch.zeros_like(wa), wa)
        wb = torch.where(b_neg_inf, torch.zeros_like(wb), wb)
        wa = wa.unsqueeze(-1)
        wb = wb.unsqueeze(-1)
        a_safe = torch.where(
            a_neg_inf.unsqueeze(-1).expand_as(oa),
            torch.zeros_like(oa), oa,
        )
        b_safe = torch.where(
            b_neg_inf.unsqueeze(-1).expand_as(ob),
            torch.zeros_like(ob), ob,
        )
        out = wa * a_safe + wb * b_safe
        return out, lse

    @staticmethod
    def _chunk_attention(
        q: "torch.Tensor", k_chunk: "torch.Tensor", v_chunk: "torch.Tensor",
        *, scaling: Optional[float] = None,
        causal_mask: Optional["torch.Tensor"] = None,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Single-chunk softmax attention.

        Shapes:
            q:        (B, H, T_q, D)        — queries
            k_chunk:  (B, H_kv, T_c, D)
            v_chunk:  (B, H_kv, T_c, D_v)
            causal_mask: (T_q, T_c) additive mask (-inf for masked) or None
        Returns:
            out:  (B, H, T_q, D_v) — partial softmax output for this chunk
            lse:  (B, H, T_q)      — log-sum-exp of scaled qk over this chunk
        """
        import torch

        H, H_kv = q.shape[1], k_chunk.shape[1]
        if H != H_kv:
            assert H % H_kv == 0, f"head count mismatch: q has {H}, kv has {H_kv}"
            rep = H // H_kv
            k_chunk = k_chunk.repeat_interleave(rep, dim=1)
            v_chunk = v_chunk.repeat_interleave(rep, dim=1)

        if scaling is None:
            scaling = 1.0 / math.sqrt(q.shape[-1])
        qk = torch.matmul(q.to(torch.float32),
                          k_chunk.to(torch.float32).transpose(-1, -2)) * scaling
        if causal_mask is not None:
            qk = qk + causal_mask.to(qk.dtype)
        lse = torch.logsumexp(qk, dim=-1)            # (B, H, T_q)
        weights = (qk - lse.unsqueeze(-1)).exp()      # (B, H, T_q, T_c)
        out = torch.matmul(weights, v_chunk.to(torch.float32))  # (B, H, T_q, D_v)
        return out, lse

    # ------------------------------------------------------------------ HF Cache API
    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        """Mode-aware update.

        Calls parent ``DynamicCache.update`` to append the new $(K, V)$
        rows, then (a) on the first 1-token decode step after the
        parent's per-layer cached length exceeds ``2 * chunk_size``,
        triggers the warmup→chunked transition by peeling cold rows
        to host pinned DRAM, and (b) in chunked mode, peels any
        overflow beyond ``recent_window + chunk_size`` to keep GPU
        residency bounded.

        The function returns the parent's *current* layer tensors
        after peeling — i.e. the recent-only view in chunked mode.
        The model's attention call is intercepted by
        ``halo_chunked_interface`` which ignores those K, V and
        consults :meth:`compute_attention` instead, so the returned
        shape mismatch versus the attention_mask is harmless.
        """
        T_new = key_states.shape[-2]
        K_full, V_full = super().update(key_states, value_states, layer_idx,
                                        *args, **kwargs)

        # Refresh Quest scorer page metadata if attached. This is a
        # no-op unless ``_quest_scorer`` was installed by
        # ``baselines.quest_path_d.wrap_with_quest_path_d``; if so the
        # scorer maintains per-page K_min/K_max bounding boxes used
        # during ``_peel_to_cold`` to drive query-aware placement.
        quest_scorer = getattr(self, "_quest_scorer", None)
        if quest_scorer is not None and K_full is not None and K_full.numel() > 0:
            try:
                quest_scorer.observe_keys(layer_idx, K_full)
            except Exception:
                pass

        # Warmup → chunked transition. Heuristic: only flip on the
        # first 1-token decode step (T_new == 1) so prefill itself
        # still uses one-shot SDPA (fast). For very long prompts where
        # prefill memory is the bottleneck, a future chunked-prefill
        # extension can fold the prefill into the chunked path
        # (Conclusion §7).
        if self._mode == "warmup":
            transition = (
                T_new == 1
                and K_full is not None
                and K_full.numel() > 0
                and K_full.shape[-2] > 2 * self.chunk_size
            )
            if transition:
                self._mode = "chunked"
                self._peel_to_cold(layer_idx)
                K_full, V_full = self._get_layer_kv(layer_idx)
        elif self._mode == "chunked":
            # Incremental peel: keep parent's recent view bounded.
            if K_full is not None and K_full.shape[-2] > self.recent_window + self.chunk_size:
                self._peel_to_cold(layer_idx)
                K_full, V_full = self._get_layer_kv(layer_idx)

        return K_full, V_full

    def _iter_layer_kv(self):
        """Yield (layer_idx, layer_obj, K, V) tuples regardless of the
        transformers cache layout.

        transformers ≤ 4.x stored ``self.key_cache: list``; 5.x has
        ``self.layers: list[DynamicLayer]`` where each layer exposes
        ``.keys`` / ``.values`` (the canonical 5.x attributes).
        """
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
        """Return (K, V) tensors held by the parent ``DynamicCache`` for
        layer ``layer_idx`` (the recent-only view in chunked mode)."""
        for li, _, k, v in self._iter_layer_kv():
            if li == layer_idx:
                return k, v
        return None, None

    def _replace_layer_kv(self, layer_idx, new_k, new_v):
        """Replace the parent's per-layer K, V tensors in place."""
        for li, layer_obj, _, _ in self._iter_layer_kv():
            if li != layer_idx:
                continue
            if layer_obj is not None and hasattr(layer_obj, "keys"):
                layer_obj.keys = new_k
                layer_obj.values = new_v
            elif layer_obj is not None and hasattr(layer_obj, "key_cache"):
                layer_obj.key_cache = new_k
                layer_obj.value_cache = new_v
            else:
                self.key_cache[layer_idx] = new_k
                self.value_cache[layer_idx] = new_v
            return

    def _peel_to_cold(self, layer_idx):
        """Peel all but ``recent_window`` rows of layer ``layer_idx`` from
        parent's GPU tensor into the host pinned cold tier, then replace
        parent's tensor with the recent-only view (freeing the cold
        suffix's GPU memory at the next allocator pass).

        This is the function that delivers the "Lossless Offloading"
        contract observed end-to-end. The corresponding tests:

        * ``tests/test_chunked_wrap_integration.py``
          ``test_chunked_actually_offloads_cold_tier`` asserts that
          after this call, parent's per-layer tensor length equals
          ``recent_window`` and ``cache._cold_k[i]`` lives on CPU.
        * ``tests/test_kv_cache_chunked.py`` ``test_lse_merge_bit_equiv``
          asserts the merge math.
        """
        import torch

        k_gpu, v_gpu = self._get_layer_kv(layer_idx)
        if k_gpu is None or k_gpu.numel() == 0:
            return
        T = k_gpu.shape[-2]
        recent = min(self.recent_window, T)
        peel = T - recent
        if peel <= 0:
            return

        # ---- Quest-aware placement (optional) ------------------------------
        # If a Quest scorer is attached and a query is available from the
        # previous compute_attention call, peel the Quest-bottom pages
        # rather than the oldest ones. Quality is invariant by Prop 4.5;
        # only DMA traffic and GPU residency change.
        quest_scorer = getattr(self, "_quest_scorer", None)
        last_q = self._last_query.get(layer_idx) if quest_scorer is not None else None
        quest_drove = False
        if quest_scorer is not None and last_q is not None:
            try:
                hot_pages = quest_scorer.select_hot_pages(layer_idx, last_q)
            except Exception:
                hot_pages = None
            if hot_pages is not None and hot_pages.numel() > 0:
                p = quest_scorer.page_size
                # Build a boolean mask over positions [0, T): True = keep on GPU.
                # Always keep the trailing recent_window positions on GPU
                # regardless of Quest score (recency floor).
                keep_pos = torch.zeros(T, dtype=torch.bool)
                for pg in hot_pages.tolist():
                    s = pg * p
                    e = min(s + p, T)
                    if s < T:
                        keep_pos[s:e] = True
                # Recency floor: always keep last ``recent_window`` rows.
                if recent > 0:
                    keep_pos[-recent:] = True
                peel_pos = (~keep_pos).nonzero(as_tuple=False).squeeze(-1)
                if peel_pos.numel() > 0 and peel_pos.numel() < T:
                    # Quest-driven peel: gather peel positions to cold tier.
                    cold_k = k_gpu.index_select(-2, peel_pos.to(k_gpu.device)).detach().to(
                        "cpu", non_blocking=_HALO_PATH_D_ASYNC_DMA, copy=True,
                    )
                    cold_v = v_gpu.index_select(-2, peel_pos.to(v_gpu.device)).detach().to(
                        "cpu", non_blocking=_HALO_PATH_D_ASYNC_DMA, copy=True,
                    )
                    keep_pos_t = keep_pos.nonzero(as_tuple=False).squeeze(-1)
                    new_k = k_gpu.index_select(-2, keep_pos_t.to(k_gpu.device)).contiguous()
                    new_v = v_gpu.index_select(-2, keep_pos_t.to(v_gpu.device)).contiguous()
                    quest_drove = True
                    peel = int(peel_pos.numel())
                    try:
                        cold_k = cold_k.pin_memory()
                        cold_v = cold_v.pin_memory()
                    except Exception:
                        pass

        if not quest_drove:
            # FU_W1 async-DMA: run the d2h copy on a dedicated CUDA
            # stream so it overlaps with the next layer's compute on the
            # default stream. Record an event so the next-step attention
            # read can wait on completion before touching the host tensor.
            if _HALO_PATH_D_ASYNC_DMA and torch.cuda.is_available():
                dma_stream = _async_dma_stream()
                # Ensure the dma stream waits for the parent stream's
                # producer (the k_gpu / v_gpu values must be ready).
                dma_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(dma_stream):
                    cold_k = k_gpu[..., :peel, :].detach().to(
                        "cpu", non_blocking=True, copy=True,
                    )
                    cold_v = v_gpu[..., :peel, :].detach().to(
                        "cpu", non_blocking=True, copy=True,
                    )
                # Record an event on the dma stream; the next
                # compute_attention will wait on it before reading cold_k.
                ev = torch.cuda.Event()
                ev.record(stream=dma_stream)
                self._cold_dma_events[layer_idx] = ev
            else:
                # Synchronous fallback (default, the correctness reference).
                cold_k = k_gpu[..., :peel, :].detach().to("cpu", copy=True)
                cold_v = v_gpu[..., :peel, :].detach().to("cpu", copy=True)
            new_k = k_gpu[..., peel:, :].contiguous()
            new_v = v_gpu[..., peel:, :].contiguous()
        # pin_memory must NOT race the async d2h copy: in the async path
        # we defer pin_memory until the event has been waited-on
        # (lazily in compute_attention). In the sync path we pin
        # immediately as before.
        if not _HALO_PATH_D_ASYNC_DMA:
            try:
                cold_k = cold_k.pin_memory()
                cold_v = cold_v.pin_memory()
            except Exception:
                pass

        # Pending async-DMA event must be drained before any host-side
        # read of cold_k (the cat below or storing into _cold_k). In
        # the sync path cold_k is already complete; in the async path
        # the dma_stream event drives the wait.
        pending_ev = self._cold_dma_events.get(layer_idx)
        if pending_ev is not None and _HALO_PATH_D_ASYNC_DMA and torch.cuda.is_available():
            # Drain on CPU — the d2h must be complete before torch.cat
            # reads the host buffer.
            pending_ev.synchronize()
            try:
                cold_k = cold_k.pin_memory()
                cold_v = cold_v.pin_memory()
            except Exception:
                pass
            # The event has been consumed; clear it so compute_attention
            # doesn't redundantly wait.
            self._cold_dma_events.pop(layer_idx, None)
        if layer_idx in self._cold_k:
            existing_k = self._cold_k[layer_idx]
            existing_v = self._cold_v[layer_idx]
            self._cold_k[layer_idx] = torch.cat([existing_k, cold_k], dim=-2)
            self._cold_v[layer_idx] = torch.cat([existing_v, cold_v], dim=-2)
        else:
            self._cold_k[layer_idx] = cold_k
            self._cold_v[layer_idx] = cold_v

        # Replace parent's tensor with the new on-GPU view. In the
        # default positional path new_k/new_v are k_gpu[..., peel:, :];
        # in the Quest-driven path they are the Quest-selected hot pages.
        # Either way, ``peel`` is the number of positions that left GPU.
        self._replace_layer_kv(layer_idx, new_k, new_v)

        # Track DMA bytes for telemetry.
        self._dma_bytes_cumulative += int(
            cold_k.element_size() * (cold_k.numel() + cold_v.numel())
        )

        # Telemetry: count demote events (per the HALOCache contract).
        # ``demoted_blocks`` here means ``cold rows that left the GPU
        # tier in this peel call'', sized in ``cfg.block_size`` units.
        block_size = max(1, getattr(self.cfg, "block_size", 32))
        n_blocks = (peel + block_size - 1) // block_size
        self.demoted_blocks_total += n_blocks
        # Demoter side-stream bookkeeping (no-op CUDA copy in the
        # async stream; the real copy happened above synchronously).
        try:
            self.demoter.demote(layer=layer_idx,
                                 block_indices=list(range(n_blocks)),
                                 src="gpu",
                                 dst=self._next_tier_after("gpu"))
        except Exception:
            pass

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Total cached sequence length across cold + recent (HF API contract).

        Critical for correct ``cache_position`` propagation to the
        model (RoPE applies the global position to each new token,
        so reporting only the recent length would scramble positional
        embeddings).
        """
        cold = self._cold_k.get(layer_idx)
        cold_T = cold.shape[-2] if cold is not None else 0
        try:
            gpu_T = super().get_seq_length(layer_idx)
        except Exception:
            gpu_T = 0
        return cold_T + gpu_T

    def get_max_cache_shape(self):
        """No static upper bound — required by transformers 5.x cache API."""
        return None

    # ------------------------------------------------------------------ main entry
    def _compute_attention_triton(
        self, *, q, layer_idx, scaling, query_pos_offset,
        cold_k, cold_v, gpu_k, gpu_v,
        T_cold, T_gpu, chunk_size, block_size,
    ):
        """Triton fast path for compute_attention.

        Same semantics as the reference per-chunk loop but each chunk's
        (matmul + softmax + matmul + lse-merge) collapses into one Triton
        kernel launch via online softmax updates on a shared (m, l, o)
        accumulator. T_q == 1 only; the reference path covers T_q > 1.

        Env var ``HALO_TRITON_STREAMED=1`` enables DMA-overlap via a
        second CUDA stream (`triton_chunked_attention_streamed`).
        """
        import os
        import torch
        from halo.triton_chunked import (
            init_acc, update_chunk, finalize,
            triton_chunked_attention_streamed,
            triton_chunked_attention_single_launch,
        )

        device = q.device
        # q: (B, H, 1, D). Squeeze to (B, H, D) for the kernel.
        q3 = q.squeeze(2).contiguous()
        B, H, D = q3.shape
        # D_v from the first available KV (cold or hot).
        if cold_v is not None and T_cold > 0:
            D_v = cold_v.shape[-1]
        elif gpu_v is not None and T_gpu > 0:
            D_v = gpu_v.shape[-1]
        else:
            D_v = D
        m, l, o = init_acc(B=B, H=H, D_v=D_v, device=device, dtype=torch.float32)
        q_pos = torch.full((B,), int(query_pos_offset),
                           device=device, dtype=torch.int64)
        scale = float(scaling) if scaling is not None else (1.0 / math.sqrt(D))

        # HALO_TRITON_SINGLE_LAUNCH=1: collapse the entire cold + hot prefix
        # into ONE Triton kernel launch (one big H2D DMA for cold, one
        # async stream sync, one update_chunk over T_cold + T_hot). Trades
        # peak staging memory for kernel-launch-overhead amortization;
        # measurably faster at T_cold >= ~24K on PCIe-4 80\,GiB hardware.
        # OFF by default because peak staging exceeds the Cell A 24\,GiB
        # budget; the streamed path remains the conservative default.
        if os.environ.get("HALO_TRITON_SINGLE_LAUNCH", "0") == "1" \
                and (T_cold + T_gpu) > 0:
            out = triton_chunked_attention_single_launch(
                q,
                cold_k_host=cold_k if T_cold > 0 else None,
                cold_v_host=cold_v if T_cold > 0 else None,
                hot_k_gpu=gpu_k if T_gpu > 0 else None,
                hot_v_gpu=gpu_v if T_gpu > 0 else None,
                scale=scale, q_pos=q_pos, apply_causal=True,
            ).squeeze(2)
            n_chunks = 1  # single kernel launch
            dma_bytes = 0
            if cold_k is not None and T_cold > 0:
                dma_bytes = int(cold_k.element_size()
                                * (cold_k.numel() + cold_v.numel()))
            peak_staging_bytes = dma_bytes  # full cold tensor on GPU
            cold_refetch_blocks = max(1, (T_cold + block_size - 1)
                                      // block_size) if T_cold > 0 else 0
            self._stats[layer_idx] = _ChunkedAttentionStats(
                hot_positions=T_gpu, cold_positions=T_cold, n_chunks=n_chunks,
                chunk_size=chunk_size, peak_staging_bytes=peak_staging_bytes,
                used_lse_merge=False, dma_bytes=dma_bytes,
            )
            self._dma_bytes_cumulative += dma_bytes
            self.refetched_blocks_total += cold_refetch_blocks
            try:
                self.refetcher.hits += cold_refetch_blocks
            except Exception:
                pass
            return out.unsqueeze(2)

        # Optional DMA-overlap path: dispatches all cold DMA on a copy
        # stream double-buffered against the compute stream's kernels.
        # Off by default because it changes CUDA event semantics in ways
        # that interact with the parent model's CUDA graph; opt in via
        # HALO_TRITON_STREAMED=1 for the wall-clock-tuned regime.
        if os.environ.get("HALO_TRITON_STREAMED", "0") == "1" \
                and cold_k is not None and T_cold > 0:
            cold_chunks = []
            for c0 in range(0, T_cold, chunk_size):
                c1 = min(c0 + chunk_size, T_cold)
                if c0 > int(query_pos_offset):
                    continue
                cold_chunks.append((cold_k[..., c0:c1, :],
                                    cold_v[..., c0:c1, :], c0))
            hot_chunks = []
            if gpu_k is not None and T_gpu > 0:
                for c0 in range(0, T_gpu, chunk_size):
                    c1 = min(c0 + chunk_size, T_gpu)
                    global_c0 = T_cold + c0
                    if global_c0 > int(query_pos_offset):
                        continue
                    hot_chunks.append((
                        gpu_k[..., c0:c1, :].contiguous(),
                        gpu_v[..., c0:c1, :].contiguous(),
                        global_c0,
                    ))
            out = triton_chunked_attention_streamed(
                q, cold_chunks, hot_chunks,
                scale=scale, q_pos=q_pos,
            ).squeeze(2)  # (B, H, D_v)
            n_chunks = len(cold_chunks) + len(hot_chunks)
            dma_bytes = sum(
                k.element_size() * (k.numel() + v.numel())
                for (k, v, _) in cold_chunks
            )
            peak_staging_bytes = max(
                (k.element_size() * (k.numel() + v.numel())
                 for (k, v, _) in cold_chunks), default=0,
            )
            cold_refetch_blocks = sum(
                max(1, (k.shape[-2] + block_size - 1) // block_size)
                for (k, _, _) in cold_chunks
            )
            self._stats[layer_idx] = _ChunkedAttentionStats(
                hot_positions=T_gpu, cold_positions=T_cold, n_chunks=n_chunks,
                chunk_size=chunk_size, peak_staging_bytes=peak_staging_bytes,
                used_lse_merge=(n_chunks > 1), dma_bytes=dma_bytes,
            )
            self._dma_bytes_cumulative += dma_bytes
            self.refetched_blocks_total += cold_refetch_blocks
            try:
                self.refetcher.hits += cold_refetch_blocks
            except Exception:
                pass
            return out.unsqueeze(2)

        n_chunks = 0
        dma_bytes = 0
        peak_staging_bytes = 0
        cold_refetch_blocks = 0

        # ----- cold suffix (DMA each chunk to GPU) -----
        if cold_k is not None and T_cold > 0:
            for c0 in range(0, T_cold, chunk_size):
                c1 = min(c0 + chunk_size, T_cold)
                # Causal short-circuit: every key index >= q_pos+1 is masked.
                if c0 > int(query_pos_offset):
                    continue
                k_h = cold_k[..., c0:c1, :].to(device, non_blocking=True).contiguous()
                v_h = cold_v[..., c0:c1, :].to(device, non_blocking=True).contiguous()
                bytes_this = int(
                    k_h.element_size() * (k_h.numel() + v_h.numel())
                )
                dma_bytes += bytes_this
                peak_staging_bytes = max(peak_staging_bytes, bytes_this)
                update_chunk(
                    q=q3, k=k_h, v=v_h, m=m, l=l, o=o,
                    scale=scale, chunk_start=c0, q_pos=q_pos,
                    apply_causal=True,
                )
                n_chunks += 1
                cold_refetch_blocks += max(
                    1, (c1 - c0 + block_size - 1) // block_size,
                )
                del k_h, v_h

        # ----- recent suffix (already on GPU) -----
        if gpu_k is not None and T_gpu > 0:
            for c0 in range(0, T_gpu, chunk_size):
                c1 = min(c0 + chunk_size, T_gpu)
                global_c0 = T_cold + c0
                if global_c0 > int(query_pos_offset):
                    continue
                k_h = gpu_k[..., c0:c1, :].contiguous()
                v_h = gpu_v[..., c0:c1, :].contiguous()
                update_chunk(
                    q=q3, k=k_h, v=v_h, m=m, l=l, o=o,
                    scale=scale, chunk_start=global_c0, q_pos=q_pos,
                    apply_causal=True,
                )
                n_chunks += 1

        out = finalize(m=m, l=l, o=o, out_dtype=q.dtype)  # (B, H, D_v)

        self._stats[layer_idx] = _ChunkedAttentionStats(
            hot_positions=T_gpu, cold_positions=T_cold, n_chunks=n_chunks,
            chunk_size=chunk_size, peak_staging_bytes=peak_staging_bytes,
            used_lse_merge=(n_chunks > 1), dma_bytes=dma_bytes,
        )
        self._dma_bytes_cumulative += dma_bytes
        self.refetched_blocks_total += cold_refetch_blocks
        try:
            self.refetcher.hits += cold_refetch_blocks
        except Exception:
            pass
        return out.unsqueeze(2)  # (B, H, 1, D_v)

    def compute_attention(
        self, q: "torch.Tensor", *, layer_idx: int,
        scaling: Optional[float] = None,
        query_pos_offset: Optional[int] = None,
    ) -> "torch.Tensor":
        """Identity-preserving attention over cold (host DRAM) + recent (GPU).

        Walks the cold suffix one ``chunk_size`` block at a time,
        DMA-ing each block into a GPU staging buffer for the chunk
        softmax, then walks the recent on-GPU portion the same way.
        Partial $(\\text{out}, \\text{lse})$ outputs are merged
        via :meth:`_lse_merge`, yielding output algebraically
        identical (real arithmetic) to one-shot full attention by
        Prop. 4.5 part i, and per-step bit-equivalent on a fixed KV
        state in fp32 by part ii. Free-running fp32 trajectories may
        still diverge due to ULP-level compound (sec:appendix-qa2-fp32).

        Parameters
        ----------
        q: (B, H, T_q, D), post-RoPE.
        layer_idx: which layer's KV cache to attend to.
        scaling: optional softmax scale; default 1/sqrt(D).
        query_pos_offset: global start index of the query block. For
            single-token decoding, omit (defaults to ``T_total - T_q``).
        """
        import torch

        device = q.device
        T_q = q.shape[-2]
        chunk_size = max(1, self.chunk_size)

        # Stash the latest query for the *next* peel call (Quest path).
        # Detached so we don't keep autograd graph alive across steps.
        if self._quest_scorer is not None:
            try:
                self._last_query[layer_idx] = q.detach()
            except Exception:
                pass

        cold_k = self._cold_k.get(layer_idx)
        cold_v = self._cold_v.get(layer_idx)
        # FU_W1 async-DMA correctness gate: if a pending cold-peel d2h
        # event exists for this layer, drain it on the CPU side before
        # any host-side read of cold_k/cold_v. The earlier prototype
        # used torch.cuda.current_stream().wait_event(ev), which is a
        # GPU-stream-side barrier — it does NOT block the CPU, and at
        # 65K context this raced the subsequent h2d chunk-read against
        # an incomplete CPU buffer, corrupting Path~D's output (0\% F1
        # vs sync's 20\% on RULER qa_2/65K). The correct fix below
        # CPU-sync's on the event, which preserves correctness at the
        # cost of parallelism; closing the parallelism gap requires
        # fusing the event-wait into the Triton chunked-LSE-merge
        # kernel (out of scope for this submission, FU\_W1 follow-up).
        ev = self._cold_dma_events.pop(layer_idx, None)
        if ev is not None and torch.cuda.is_available():
            ev.synchronize()
            try:
                if cold_k is not None:
                    self._cold_k[layer_idx] = cold_k = cold_k.pin_memory()
                if cold_v is not None:
                    self._cold_v[layer_idx] = cold_v = cold_v.pin_memory()
            except Exception:
                pass
        gpu_k, gpu_v = self._get_layer_kv(layer_idx)

        T_cold = cold_k.shape[-2] if cold_k is not None else 0
        T_gpu = gpu_k.shape[-2] if (gpu_k is not None and gpu_k.numel() > 0) else 0
        T_total = T_cold + T_gpu

        if T_total == 0:
            B, H = q.shape[0], q.shape[1]
            D_v = q.shape[-1]
            return torch.zeros(B, H, T_q, D_v, dtype=q.dtype, device=device)

        if query_pos_offset is None:
            query_pos_offset = T_total - T_q

        block_size = max(1, getattr(self.cfg, "block_size", 32))

        # ------------------------------------------------------------------
        # Triton fast path: T_q == 1 (single-token decode) on CUDA only.
        # Collapses the per-chunk (matmul + logsumexp + matmul + logaddexp)
        # sequence into one fused kernel launch per chunk. The result is
        # numerically equivalent to the reference path within bf16 chunk-
        # merge tolerance (tests/test_triton_chunked.py).
        # ------------------------------------------------------------------
        if (
            self.use_triton and self._triton_available
            and T_q == 1 and device.type == "cuda"
        ):
            return self._compute_attention_triton(
                q=q, layer_idx=layer_idx, scaling=scaling,
                query_pos_offset=query_pos_offset,
                cold_k=cold_k, cold_v=cold_v,
                gpu_k=gpu_k, gpu_v=gpu_v,
                T_cold=T_cold, T_gpu=T_gpu, chunk_size=chunk_size,
                block_size=block_size,
            )

        running_out = running_lse = None
        n_chunks = 0
        dma_bytes = 0
        peak_staging_bytes = 0

        # ------ cold suffix (DMA each chunk to GPU) ------
        # Track the refetch (every cold chunk we DMA in counts as one
        # refetch event for the HALOCache telemetry contract).
        cold_refetch_blocks = 0
        if cold_k is not None and T_cold > 0:
            for c0 in range(0, T_cold, chunk_size):
                c1 = min(c0 + chunk_size, T_cold)
                # Causal short-circuit: skip a chunk that has no allowed
                # keys for any query.
                if T_q > 1:
                    i = torch.arange(T_q, device=device).unsqueeze(-1)
                    j = torch.arange(c1 - c0, device=device).unsqueeze(0)
                    allowed_any = bool(((c0 + j) <= (query_pos_offset + i)).any())
                    if not allowed_any:
                        continue
                k_h = cold_k[..., c0:c1, :].to(device, non_blocking=True)
                v_h = cold_v[..., c0:c1, :].to(device, non_blocking=True)
                bytes_this = int(
                    k_h.element_size() * (k_h.numel() + v_h.numel())
                )
                dma_bytes += bytes_this
                peak_staging_bytes = max(peak_staging_bytes, bytes_this)

                causal = None
                if T_q > 1:
                    i = torch.arange(T_q, device=device).unsqueeze(-1)
                    j = torch.arange(c1 - c0, device=device).unsqueeze(0)
                    allowed = (c0 + j) <= (query_pos_offset + i)
                    if not bool(allowed.all()):
                        causal = torch.where(allowed, 0.0, float("-inf"))

                out_c, lse_c = self._chunk_attention(
                    q, k_h, v_h, scaling=scaling, causal_mask=causal,
                )
                if running_out is None:
                    running_out, running_lse = out_c, lse_c
                else:
                    running_out, running_lse = self._lse_merge(
                        running_out, running_lse, out_c, lse_c,
                    )
                n_chunks += 1
                # Refetch telemetry: this chunk came from host DRAM
                # and was DMA'd into a GPU staging buffer for attention.
                cold_refetch_blocks += max(1, (c1 - c0 + block_size - 1) // block_size)
                # Release the staging buffer immediately so the next
                # iteration's DMA reuses the same GPU memory slot.
                del k_h, v_h

        # ------ recent suffix (GPU-resident, no DMA) ------
        if gpu_k is not None and T_gpu > 0:
            for c0 in range(0, T_gpu, chunk_size):
                c1 = min(c0 + chunk_size, T_gpu)
                global_c0 = T_cold + c0
                k_h = gpu_k[..., c0:c1, :]
                v_h = gpu_v[..., c0:c1, :]
                causal = None
                if T_q > 1:
                    i = torch.arange(T_q, device=device).unsqueeze(-1)
                    j = torch.arange(c1 - c0, device=device).unsqueeze(0)
                    allowed = (global_c0 + j) <= (query_pos_offset + i)
                    if not bool(allowed.all()):
                        if bool((~allowed).all()):
                            continue
                        causal = torch.where(allowed, 0.0, float("-inf"))
                out_c, lse_c = self._chunk_attention(
                    q, k_h, v_h, scaling=scaling, causal_mask=causal,
                )
                if running_out is None:
                    running_out, running_lse = out_c, lse_c
                else:
                    running_out, running_lse = self._lse_merge(
                        running_out, running_lse, out_c, lse_c,
                    )
                n_chunks += 1

        self._stats[layer_idx] = _ChunkedAttentionStats(
            hot_positions=T_gpu, cold_positions=T_cold, n_chunks=n_chunks,
            chunk_size=chunk_size, peak_staging_bytes=peak_staging_bytes,
            used_lse_merge=(n_chunks > 1), dma_bytes=dma_bytes,
        )
        self._dma_bytes_cumulative += dma_bytes
        # Telemetry: refetch event count (one per cold chunk DMA'd in).
        # ``refetched_blocks_total`` is the HALOCache contract surface
        # for ``how many block-sized units the cache pulled back from
        # a slower tier during this run''.
        self.refetched_blocks_total += cold_refetch_blocks
        # Refetcher side-stream bookkeeping. The refetcher's own
        # hit / miss counters are not meaningful in the Path D pure-DMA
        # path (every fetch is a "hit" in the trivial sense of "the
        # block was available on host"), so we increment ``hits`` to
        # reflect that.
        try:
            self.refetcher.hits += cold_refetch_blocks
        except Exception:
            pass
        return running_out.to(q.dtype)

    def telemetry(self) -> dict:
        base = super().telemetry()
        base["mode"] = self._mode
        base["dma_bytes_cumulative"] = int(self._dma_bytes_cumulative)
        if self._stats:
            base["chunked_n_layers_called"] = len(self._stats)
            base["chunked_total_chunks"] = sum(s.n_chunks for s in self._stats.values())
            base["chunked_peak_staging_bytes_max"] = max(
                s.peak_staging_bytes for s in self._stats.values()
            )
            base["chunked_used_lse_merge_any"] = any(
                s.used_lse_merge for s in self._stats.values()
            )
            base["chunked_total_dma_bytes"] = sum(s.dma_bytes for s in self._stats.values())
        if self._cold_k:
            base["cold_positions_per_layer_max"] = max(
                v.shape[-2] for v in self._cold_k.values()
            )
            base["cold_layers"] = len(self._cold_k)
        return base

    def reset(self) -> None:
        super().reset()
        self._mode = "warmup"
        self._cold_k.clear()
        self._cold_v.clear()
        self._stats.clear()
        self._dma_bytes_cumulative = 0


__all__ = ["HALOCacheChunked"]
