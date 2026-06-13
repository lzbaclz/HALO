"""HALO-Scorer: closed-form, training-free hot/cold classifier (§4.1).

The scorer maps a per-layer attention snapshot to a per-position hotness score
in [0, 1]. The closed-form weights ``(α, β, γ)`` are taken directly from the
EMNLP paper's Finding 3:

    score(p) = α · attn(p) + β · recency(p) + γ · sink(p)

where:
    attn(p)     — head-mean attention mass placed on position ``p`` at the current step
    recency(p) — geometric decay over distance from the most recent token
    sink(p)    — 1 iff p < ``sink_tokens`` else 0  (StreamingLLM-style anchors)

The scorer is intentionally stateless and ``torch.no_grad`` friendly. A
trainable variant (closed-form fit per model family) lives in
:mod:`halo.classifier` for the Finding-3 ablation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from halo.policy import HALOConfig


class HALOScorer:
    """Compute per-position hotness scores from an attention snapshot."""

    def __init__(self, config: "HALOConfig") -> None:
        self.cfg = config

    # ---------- public API ----------

    def score(self, attention: "torch.Tensor", *, step: int) -> "torch.Tensor":
        """Return a hotness vector of shape ``(K,)`` for one (layer, head-mean) snapshot.

        Parameters
        ----------
        attention:
            1-D tensor ``(K,)`` of head-mean attention mass on the K already-cached
            positions, taken from the most recent decoding step.
        step:
            Current decoding step index (used by the recency term).
        """
        import torch

        K = attention.shape[-1]
        device = attention.device
        attn = attention.float()
        attn = attn / (attn.sum() + 1e-12)

        positions = torch.arange(K, device=device, dtype=torch.float32)
        # recency: geometric decay over (step - position). Falls off at horizon ~refresh_window.
        decay = max(self.cfg.refresh_window, 1)
        recency = torch.exp(-(step - positions).clamp(min=0.0) / decay)
        recency = recency / (recency.sum() + 1e-12)

        sink = torch.zeros(K, device=device, dtype=torch.float32)
        if self.cfg.sink_tokens > 0:
            n = min(self.cfg.sink_tokens, K)
            sink[:n] = 1.0

        score = (
            self.cfg.score_alpha * attn
            + self.cfg.score_beta * recency
            + self.cfg.score_gamma * sink
        )
        # Squash to [0, 1]
        score = score / (score.max() + 1e-12)
        return score

    def topk_hot(self, score: "torch.Tensor", *, ratio: float | None = None) -> "torch.Tensor":
        """Return indices of the hot set as defined by ``ratio`` (defaults to ``hot_ratio``).

        At ``ratio >= 1.0`` returns *every* position so that HALO degenerates
        into full attention by construction (the identity invariant exercised
        by :func:`tests.test_kv_cache.test_identity_at_hot_ratio_one`).
        """
        import torch

        ratio = self.cfg.hot_ratio if ratio is None else ratio
        K = score.shape[-1]
        k = min(max(int(ratio * K), 1), K)
        if ratio >= 1.0:
            return torch.arange(K, device=score.device, dtype=torch.long)
        return torch.topk(score, k=k).indices.sort().values

    # ---------- diagnostic helpers ----------

    def jaccard(self, a: "torch.Tensor", b: "torch.Tensor") -> float:
        """Jaccard overlap between two index sets (Finding 2)."""
        sa, sb = set(a.tolist()), set(b.tolist())
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / max(len(sa | sb), 1)

    def __repr__(self) -> str:  # pragma: no cover
        c = self.cfg
        return (
            f"HALOScorer(hot_ratio={c.hot_ratio}, "
            f"alpha={c.score_alpha}, beta={c.score_beta}, gamma={c.score_gamma}, "
            f"sink={c.sink_tokens}, refresh={c.refresh_window})"
        )
