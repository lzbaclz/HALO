"""Unit tests for :class:`halo.scorer.HALOScorer` (Finding 3 closed-form)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_score_shape_and_range(small_config, fake_attention):
    from halo import HALOScorer

    scorer = HALOScorer(small_config)
    attn = fake_attention(K=64)
    score = scorer.score(attn, step=10)

    assert score.shape == attn.shape
    assert torch.isfinite(score).all()
    assert (score >= 0).all() and (score <= 1).all()


def test_topk_hot_respects_ratio(small_config, fake_attention):
    from halo import HALOScorer

    scorer = HALOScorer(small_config)
    attn = fake_attention(K=128)
    score = scorer.score(attn, step=0)
    hot = scorer.topk_hot(score)
    assert hot.numel() == int(small_config.hot_ratio * 128)


def test_sink_tokens_always_hot(small_config, fake_attention):
    """Sink positions get an explicit boost so they appear in topk."""
    from halo import HALOScorer

    scorer = HALOScorer(small_config)
    attn = fake_attention(K=64)
    score = scorer.score(attn, step=0)
    hot = set(scorer.topk_hot(score, ratio=0.5).tolist())
    for sink in range(small_config.sink_tokens):
        assert sink in hot, f"sink position {sink} should always be hot"


def test_jaccard_extremes(small_config):
    from halo import HALOScorer

    scorer = HALOScorer(small_config)
    a = torch.tensor([1, 2, 3, 4])
    b = torch.tensor([1, 2, 3, 4])
    assert scorer.jaccard(a, b) == pytest.approx(1.0)

    c = torch.tensor([5, 6, 7, 8])
    assert scorer.jaccard(a, c) == pytest.approx(0.0)


def test_recency_decay_monotone(small_config):
    """At a fixed step, recency should give later positions a non-decreasing score."""
    from halo import HALOScorer

    cfg = small_config
    cfg.score_alpha = 0.0
    cfg.score_gamma = 0.0
    cfg.score_beta = 1.0
    scorer = HALOScorer(cfg)

    K = 32
    attn = torch.zeros(K)
    score = scorer.score(attn, step=K - 1)
    diffs = score[1:] - score[:-1]
    assert (diffs >= -1e-6).all(), "recency should be (weakly) monotone increasing"
