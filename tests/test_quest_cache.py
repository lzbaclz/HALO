"""Unit tests for :mod:`baselines.quest_cache`.

Five contracts we want to nail down:

1. **Metadata consistency**: after several decoding-step updates, the
   per-page ``k_min``/``k_max`` tensors equal the actual per-channel
   min / max of the corresponding K slice. Catches off-by-one in the
   ``_refresh_page_metadata`` slicing.
2. **Score is an upper bound on the true qk inner product**: for any
   query and any actual key in a page, the page score must be
   $\\geq q \\cdot k$. This is the *whole reason* Quest's top-K works
   as a Bayes-style approximation.
3. **Identity at full budget**: at ``memory_ratio = 1.0`` and
   ``min_pages_selected >= num_pages``, Quest's attention output
   matches one-shot SDPA on the full $(K, V)$ to numerical precision.
   (Causal masking still applies but the page selection is a no-op.)
4. **GQA broadcast**: with $H_q = 8, H_{kv} = 2$ (rep=4), the score
   function produces the right shape and no head-dim mistakes.
5. **Telemetry**: after $S$ decoding steps with budget $K$ pages,
   ``pages_selected_total / pages_visited_total`` is close to
   ``K / num_pages_total`` (modulo always-include sink + partial).
"""
from __future__ import annotations

import math

import pytest
import torch

from baselines.quest_cache import QuestConfig, QuestPagedCache


def _make_cache(page_size=8, ratio=4.0, sink=1, min_pick=2):
    cfg = QuestConfig(
        page_size=page_size,
        memory_ratio=ratio,
        sink_pages=sink,
        min_pages_selected=min_pick,
    )
    return QuestPagedCache(cfg)


def _seed_cache(cache, T, H_kv=2, D=8, B=1, dtype=torch.float32):
    """Push T tokens into the cache via a single super().update call.

    We bypass the model and call DynamicCache.update directly so the
    test runs without any HF model. The transformers DynamicCache
    appends along dim -2.
    """
    torch.manual_seed(0)
    k = torch.randn(B, H_kv, T, D, dtype=dtype)
    v = torch.randn(B, H_kv, T, D, dtype=dtype)
    K_full, V_full = cache.update(k, v, layer_idx=0)
    return K_full, V_full


def test_metadata_consistency_full_pages():
    cache = _make_cache(page_size=8)
    K_full, _ = _seed_cache(cache, T=32, H_kv=2, D=4)
    meta = cache._page_meta[0]
    assert meta["num_full_pages"] == 4
    # Recompute reference min/max by reshaping K_full.
    B, H_kv, T, D = K_full.shape
    expected = K_full.view(B, H_kv, 4, 8, D)
    expected_min = expected.amin(dim=-2)
    expected_max = expected.amax(dim=-2)
    assert torch.allclose(meta["k_min"], expected_min)
    assert torch.allclose(meta["k_max"], expected_max)


def test_metadata_partial_page_not_finalised():
    """Tokens that don't yet fill a page should NOT contribute to metadata."""
    cache = _make_cache(page_size=8)
    _seed_cache(cache, T=20, H_kv=2, D=4)  # 2 full pages + 4 partial tokens
    meta = cache._page_meta[0]
    assert meta["num_full_pages"] == 2
    assert meta["k_min"].shape[-2] == 2


def test_metadata_incremental_update():
    """Pushing 8 then another 8 tokens must match pushing 16 at once."""
    cache_a = _make_cache(page_size=8)
    torch.manual_seed(0)
    k1 = torch.randn(1, 2, 8, 4)
    v1 = torch.randn(1, 2, 8, 4)
    k2 = torch.randn(1, 2, 8, 4)
    v2 = torch.randn(1, 2, 8, 4)
    cache_a.update(k1, v1, layer_idx=0)
    cache_a.update(k2, v2, layer_idx=0)
    meta_a = cache_a._page_meta[0]

    cache_b = _make_cache(page_size=8)
    torch.manual_seed(0)
    k1 = torch.randn(1, 2, 8, 4)
    v1 = torch.randn(1, 2, 8, 4)
    k2 = torch.randn(1, 2, 8, 4)
    v2 = torch.randn(1, 2, 8, 4)
    cache_b.update(torch.cat([k1, k2], dim=-2),
                   torch.cat([v1, v2], dim=-2),
                   layer_idx=0)
    meta_b = cache_b._page_meta[0]

    assert meta_a["num_full_pages"] == meta_b["num_full_pages"] == 2
    assert torch.allclose(meta_a["k_min"], meta_b["k_min"])
    assert torch.allclose(meta_a["k_max"], meta_b["k_max"])


def test_score_is_upper_bound():
    """For every (query, page, actual key in page), score >= q · k.

    This is the core algorithmic claim of Quest: the bounding-box score
    over [k_min, k_max] upper-bounds the true q·k for any actual key
    in the page.
    """
    cache = _make_cache(page_size=8)
    K_full, _ = _seed_cache(cache, T=32, H_kv=2, D=4)
    # One query, same heads as KV. q: (B=1, H=2, T_q=1, D=4).
    torch.manual_seed(1)
    q = torch.randn(1, 2, 1, 4)
    scores = cache._score_pages(q, layer_idx=0)
    assert scores.shape == (1, 2, 1, 4)
    page_size = 8
    # Reshape K_full to (B, H_kv, num_pages, page_size, D).
    K_pages = K_full.view(1, 2, 4, page_size, 4)
    # We want per-(B, H_kv, T_q, page) the max over tokens of q·k.
    # Add explicit broadcast dims to avoid PyTorch's left-pad behaviour.
    q_exp = q.unsqueeze(3).unsqueeze(4)        # (1, 2, 1, 1, 1, 4)
    K_exp = K_pages.unsqueeze(2)               # (1, 2, 1, num_pages, page_size, 4)
    true_qk = (q_exp * K_exp).sum(dim=-1)      # (1, 2, 1, num_pages, page_size)
    true_max_per_page = true_qk.amax(dim=-1)   # (1, 2, 1, num_pages)
    assert scores.shape == true_max_per_page.shape
    diff = scores - true_max_per_page
    assert (diff >= -1e-4).all(), (
        f"Quest score is NOT an upper bound; min(diff) = {diff.min().item():.4e}"
    )


def test_identity_at_full_budget():
    """At memory_ratio=1.0 and large min_pages_selected, Quest matches
    one-shot SDPA up to numerical precision (every page is selected)."""
    cfg = QuestConfig(page_size=8, memory_ratio=1.0,
                      sink_pages=1, min_pages_selected=1000)
    cache = QuestPagedCache(cfg)
    K_full, V_full = _seed_cache(cache, T=24, H_kv=2, D=4)
    # Query at the latest position (decoding step).
    torch.manual_seed(2)
    q = torch.randn(1, 2, 1, 4)
    out_quest = cache.quest_attention(q, layer_idx=0)
    # Reference: one-shot SDPA on full K, V.
    scaling = 1.0 / math.sqrt(q.shape[-1])
    out_ref = torch.nn.functional.scaled_dot_product_attention(
        q, K_full, V_full, scale=scaling, is_causal=False,
    )
    assert out_quest.shape == out_ref.shape
    assert torch.allclose(out_quest, out_ref, atol=1e-5, rtol=1e-5), (
        f"Quest at full budget should match SDPA exactly. Max diff: "
        f"{(out_quest - out_ref).abs().max().item():.4e}"
    )


def test_gqa_score_broadcast():
    """With H_q=8, H_kv=2 (rep=4), the score function pools q correctly."""
    cache = _make_cache(page_size=8)
    _seed_cache(cache, T=16, H_kv=2, D=4)
    torch.manual_seed(3)
    q = torch.randn(1, 8, 1, 4)
    scores = cache._score_pages(q, layer_idx=0)
    # Score is per KV head, not per Q head.
    assert scores.shape == (1, 2, 1, 2)


def test_gqa_attention_output_shape():
    cache = _make_cache(page_size=8, ratio=2.0, min_pick=1)
    _seed_cache(cache, T=24, H_kv=2, D=4)
    torch.manual_seed(4)
    q = torch.randn(1, 8, 1, 4)  # H_q=8, H_kv=2
    out = cache.quest_attention(q, layer_idx=0)
    assert out.shape == (1, 8, 1, 4)


def test_telemetry_records_selection():
    cfg = QuestConfig(page_size=8, memory_ratio=4.0, sink_pages=1,
                      min_pages_selected=2)
    cache = QuestPagedCache(cfg)
    _seed_cache(cache, T=64, H_kv=2, D=4)
    torch.manual_seed(5)
    q = torch.randn(1, 2, 1, 4)
    cache.quest_attention(q, layer_idx=0)
    tele = cache.telemetry()
    assert tele["selection_steps"] == 1
    # Per (B=1, H_kv=2, T_q=1), total pages = 8 (no partial), selected
    # should be ceil(8/4)=2 pages each. So ratio ~ 2/8 = 0.25.
    assert 0.15 < tele["avg_pages_selected_frac"] < 0.35


def test_empty_cache_returns_zero():
    cache = _make_cache(page_size=8)
    # No update call, cache is empty.
    # But cache.layers is also empty; quest_attention needs to handle this.
    torch.manual_seed(6)
    q = torch.randn(1, 2, 1, 4)
    # Manually populate the layer with an empty tensor to mimic the
    # cache state immediately after reset() before any update.
    # We just check that quest_attention doesn't crash and returns zero.
    out = cache.quest_attention(q, layer_idx=0)
    assert out.shape == (1, 2, 1, 4)
    assert torch.all(out == 0)


def test_reset_clears_state():
    cache = _make_cache(page_size=8)
    _seed_cache(cache, T=32, H_kv=2, D=4)
    torch.manual_seed(7)
    cache.quest_attention(torch.randn(1, 2, 1, 4), layer_idx=0)
    assert cache._page_meta
    assert cache._selection_steps > 0
    cache.reset()
    assert not cache._page_meta
    assert cache._selection_steps == 0
