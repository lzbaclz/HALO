"""Unit test for the Quest + Path D composition.

Verifies that Path D's LSE-merge attention, composed with Quest's
per-page partition (a partition of the token positions into pages
of size ``page_size``), produces output bit-equivalent to one-shot
Full SDPA in fp32 and to within :cref:`cor:bf16-bound`'s 2%
relative $L_2$ tolerance in bf16.

This is the architectural proof for the "Quest + Path D" extension
discussed in §5.3 and §7: Path D's LSE-merge is a partition-agnostic
softmax, so whether the partition comes from HALO's hot/cold split
or from Quest's page boundaries, the merged output is the same.

We don't need a GPU or the full HF model to verify this; the
LSE-merge invariant is a property of the algebra, tested directly
on synthetic tensors.
"""
from __future__ import annotations

import math
import pytest

torch = pytest.importorskip("torch")


def _ref_sdpa(q, k, v):
    """Reference single-pass softmax-attention. Operates in fp32
    for numerical stability and returns fp32 output.
    """
    q32 = q.to(torch.float32)
    k32 = k.to(torch.float32)
    v32 = v.to(torch.float32)
    scale = 1.0 / math.sqrt(q.shape[-1])
    logits = (q32 @ k32.transpose(-1, -2)) * scale
    weights = torch.softmax(logits, dim=-1)
    return weights @ v32


def _chunked_lse_merge(q, k, v, *, page_size: int):
    """LSE-merge attention over disjoint page chunks of size
    ``page_size``. This is the same primitive Path D uses to
    stream cold pages back per attention call. Returns fp32.
    """
    q32 = q.to(torch.float32)
    k32 = k.to(torch.float32)
    v32 = v.to(torch.float32)
    scale = 1.0 / math.sqrt(q.shape[-1])

    T = k.shape[-2]
    out_acc = None
    lse_acc = None
    for start in range(0, T, page_size):
        end = min(start + page_size, T)
        k_chunk = k32[..., start:end, :]
        v_chunk = v32[..., start:end, :]
        logits = (q32 @ k_chunk.transpose(-1, -2)) * scale
        # per-chunk lse + partial output
        lse_chunk = torch.logsumexp(logits, dim=-1, keepdim=True)  # (..., T_q, 1)
        weights = torch.softmax(logits, dim=-1)
        out_chunk = weights @ v_chunk
        if out_acc is None:
            out_acc = out_chunk
            lse_acc = lse_chunk
            continue
        # Stable two-way LSE merge: out = (e^{lse_a} out_a + e^{lse_b} out_b)
        # / (e^{lse_a} + e^{lse_b}); compute via shift to avoid overflow.
        lse_max = torch.maximum(lse_acc, lse_chunk)
        w_a = torch.exp(lse_acc - lse_max)
        w_b = torch.exp(lse_chunk - lse_max)
        denom = w_a + w_b
        out_acc = (w_a * out_acc + w_b * out_chunk) / denom
        lse_acc = lse_max + torch.log(denom)
    return out_acc


def test_quest_partition_lse_merge_matches_full_sdpa_fp32():
    """At Quest's page_size=16, the LSE-merge over pages should match
    Full SDPA to within fp32 reduction noise."""
    torch.manual_seed(0)
    B, H, T_q, T_k, D = 1, 4, 1, 256, 32
    q = torch.randn(B, H, T_q, D)
    k = torch.randn(B, H, T_k, D)
    v = torch.randn(B, H, T_k, D)

    ref = _ref_sdpa(q, k, v)
    merged = _chunked_lse_merge(q, k, v, page_size=16)
    rel = (ref - merged).norm() / ref.norm().clamp_min(1e-12)
    assert rel < 1e-5, (
        f"fp32 LSE-merge should be bit-equivalent to one-shot SDPA "
        f"up to reduction noise; got rel L2 {rel:.2e}"
    )


def test_quest_partition_lse_merge_matches_full_sdpa_bf16():
    """In bf16, LSE-merge over Quest-style pages stays within
    :cref:`cor:bf16-bound`'s 2% relative L2 tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("bf16 test requires GPU")
    torch.manual_seed(0)
    B, H, T_q, T_k, D = 1, 4, 1, 256, 32
    q = torch.randn(B, H, T_q, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, H, T_k, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, H, T_k, D, dtype=torch.bfloat16, device="cuda")

    ref = _ref_sdpa(q, k, v)
    merged = _chunked_lse_merge(q, k, v, page_size=16)
    rel = (ref - merged).norm() / ref.norm().clamp_min(1e-12)
    assert rel < 2e-2, (
        f"bf16 LSE-merge should stay within Cor 4.6's 2% bound; "
        f"got rel L2 {rel:.2e}"
    )


def test_quest_partition_irrespective_of_page_size():
    """The merged output is independent of how the partition is
    drawn: page_size=16 (Quest default), 64, 256 (one big chunk) all
    yield the same answer up to fp32 noise."""
    torch.manual_seed(0)
    B, H, T_q, T_k, D = 1, 4, 1, 256, 32
    q = torch.randn(B, H, T_q, D)
    k = torch.randn(B, H, T_k, D)
    v = torch.randn(B, H, T_k, D)

    out_a = _chunked_lse_merge(q, k, v, page_size=16)
    out_b = _chunked_lse_merge(q, k, v, page_size=64)
    out_c = _chunked_lse_merge(q, k, v, page_size=256)

    for o in (out_b, out_c):
        rel = (out_a - o).norm() / out_a.norm().clamp_min(1e-12)
        assert rel < 1e-5, f"Partition-independence violated, rel={rel:.2e}"


def test_quest_path_d_config_compose():
    """The QuestPathDConfig wrapper exists and exposes both Quest's
    selection knobs and Path D's chunked-attention knobs."""
    from baselines.quest_path_d import QuestPathDConfig

    cfg = QuestPathDConfig()
    # Quest knobs
    assert cfg.page_size == 16
    assert cfg.memory_ratio == 4.0
    assert cfg.min_pages_selected == 4
    # Path D knobs
    assert cfg.chunk_size == 512
    assert cfg.recent_window == 64


def test_quest_scorer_actually_fires():
    """The Quest scorer adapter actually computes scores when called
    (not NotImplementedError as in the v3 stub). Validates the
    selection rule's per-page upper-bound math against direct
    q·K_min / q·K_max summation.
    """
    from baselines.quest_path_d import QuestPathDConfig, _QuestScorerAdapter

    cfg = QuestPathDConfig(page_size=4, memory_ratio=2.0, min_pages_selected=1)
    adapter = _QuestScorerAdapter(cfg)

    torch.manual_seed(7)
    B, H_kv, T, D = 1, 2, 16, 8
    K = torch.randn(B, H_kv, T, D)
    adapter.observe_keys(layer_idx=0, K_full=K)
    assert adapter.observe_calls == 1
    assert adapter._num_full_pages[0] == 4  # T=16 / page_size=4

    q = torch.randn(B, H_kv, 1, D)
    scores = adapter.score_pages(0, q)
    assert scores is not None
    assert scores.shape == (B, H_kv, 1, 4)
    assert adapter.score_calls == 1

    # Manual computation: per-page upper bound = sum_d max(q*K_min, q*K_max).
    K_pages = K.view(B, H_kv, 4, 4, D)
    k_min = K_pages.amin(dim=-2)
    k_max = K_pages.amax(dim=-2)
    q_exp = q.unsqueeze(-2).float()
    expected = torch.maximum(
        q_exp * k_min.unsqueeze(-3).float(),
        q_exp * k_max.unsqueeze(-3).float(),
    ).sum(dim=-1)
    assert torch.allclose(scores, expected, atol=1e-5)

    # select_hot_pages returns top-K pages by score.
    topk = adapter.select_hot_pages(0, q)
    assert topk is not None
    K_total = max(cfg.min_pages_selected, int((4 + cfg.memory_ratio - 1) // cfg.memory_ratio))
    assert topk.shape == (K_total,)
    assert adapter.cumulative_hot_pages == K_total


def test_quest_scorer_fires_during_chunked_forward():
    """Integration test: the Quest scorer is actually invoked during a real
    HALOCacheChunked forward pass (cache.update + compute_attention), not
    just when called directly on synthetic tensors.

    This guards against the wired-vs-defined trap caught by reviewer 3 in an
    earlier draft: every method/test name that says 'actually fires' must
    drive the cache machinery end-to-end, with the scorer's call counters
    incremented by the cache itself, never by the test fixture.
    """
    from baselines.quest_path_d import QuestPathDConfig, _QuestScorerAdapter
    from halo.kv_cache_chunked import HALOCacheChunked
    from halo.demoter import HALODemoter
    from halo.memory_tier import MemoryTier, TieredStorage
    from halo.policy import HALOConfig
    from halo.refetcher import HALORefetcher
    from halo.scorer import HALOScorer

    cfg = HALOConfig(hot_ratio=0.5, tiers=("gpu", "dram"))
    storage = TieredStorage(
        tiers=[MemoryTier.GPU, MemoryTier.DRAM],
        num_layers=1, num_kv_heads=2, head_dim=32, block_size=32,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    cache = HALOCacheChunked(
        config=cfg, storage=storage, scorer=HALOScorer(cfg),
        demoter=HALODemoter(cfg, storage=storage),
        refetcher=HALORefetcher(cfg, storage=storage),
        chunk_size=32,
    )
    # Attach the Quest adapter exactly as wrap_with_quest_path_d does in
    # baselines/quest_path_d.py:247.
    qcfg = QuestPathDConfig(page_size=16, memory_ratio=2.0, min_pages_selected=1)
    adapter = _QuestScorerAdapter(qcfg)
    cache._quest_scorer = adapter

    torch.manual_seed(11)
    B, H_kv, T, D = 1, 2, 256, 32
    k = torch.randn(B, H_kv, T, D)
    v = torch.randn(B, H_kv, T, D)

    # Seed cache via update — this is the call site that invokes
    # adapter.observe_keys in halo/kv_cache_chunked.py:270.
    cache.update(k, v, layer_idx=0)
    assert adapter.observe_calls >= 1, (
        f"observe_keys was not invoked during cache.update; "
        f"observe_calls={adapter.observe_calls}. The scorer is defined but "
        "not wired into the cache.update path."
    )

    # The Quest scorer's select_hot_pages call lives inside _peel_to_cold,
    # which fires at the warmup→chunked transition (a 1-token update after a
    # multi-chunk prefill, see halo/kv_cache_chunked.py:280-289). select_hot_pages
    # also requires _last_query[layer] populated by a prior compute_attention.
    # Drive the full sequence:
    q_decode = torch.randn(B, H_kv, 1, D)
    _ = cache.compute_attention(q_decode, layer_idx=0)
    assert 0 in cache._last_query, (
        "compute_attention did not populate _last_query[0]; Quest's "
        "query-aware path needs the previous query to score pages."
    )
    # Inject a 1-token KV to trigger the warmup→chunked transition. This is the
    # call site that invokes select_hot_pages via _peel_to_cold:382.
    k_inc = torch.randn(B, H_kv, 1, D)
    v_inc = torch.randn(B, H_kv, 1, D)
    cache.update(k_inc, v_inc, layer_idx=0)
    assert adapter.score_calls >= 1 or adapter.cumulative_hot_pages >= 1, (
        f"select_hot_pages was not invoked at the warmup→chunked transition; "
        f"score_calls={adapter.score_calls}, "
        f"cumulative_hot_pages={adapter.cumulative_hot_pages}. The scorer is "
        "attached but unread by _peel_to_cold during the transition."
    )
