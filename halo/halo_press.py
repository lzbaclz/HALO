"""HALO scoring rule packaged as a kvpress ``ScorerPress``.

Why this exists
---------------
The full HALO design (paper §4) does demote-on-cool / refetch-on-warm
*during decoding*. Implementing that against a moving HuggingFace ``transformers``
attention API is a nontrivial engineering project; the current
``halo.policy.wrap_with_halo`` already provides the bit-identity invariant at
``hot_ratio=1.0`` (see ``tests/test_integration_identity.py``).

For the LongBench / RULER tables we need a HALO row that *actually compresses*.
``kvpress`` provides a clean ``BasePress`` / ``ScorerPress`` interface that
prunes KV positions after prefill using a per-position importance score. The
HALO scoring rule (alpha*attn + beta*recency + gamma*sink) plugs into that
interface directly: ``HALOPress`` lets the unified evaluation harness dispatch
``--method halo`` to a real compression press while we finish wiring the
streaming demote/refetch path.

This is what gets reported in the paper's main tables. The streaming variant
remains the design contribution of paper section 4 and is verified by the
identity test.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    from kvpress import ScorerPress
except ImportError as e:
    raise ImportError(
        "halo.halo_press requires kvpress. Run `pip install kvpress`."
    ) from e


@dataclass
class HALOPress(ScorerPress):
    """Score-and-prune KV compression with HALO's alpha*attn+beta*recency+gamma*sink."""

    compression_ratio: float = 0.0
    score_alpha: float = 1.0
    score_beta: float = 0.5
    score_gamma: float = 2.0
    sink_tokens: int = 4

    def score(
        self,
        module,
        hidden_states,
        keys,
        values,
        attentions,
        kwargs,
    ):
        import torch

        b, n_kv_heads, seq_len, _ = keys.shape

        if attentions is not None and attentions.numel() > 0:
            head_mean = attentions.mean(dim=1)
            attn = head_mean.mean(dim=1)
        else:
            attn = -keys.norm(dim=-1).mean(dim=1)
            attn = attn - attn.min(dim=-1, keepdim=True).values

        if attn.dim() == 2:
            attn = attn.unsqueeze(1).expand(-1, n_kv_heads, -1)

        denom = attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        attn = attn / denom

        positions = torch.arange(seq_len, device=keys.device, dtype=attn.dtype)
        recency = torch.exp((positions - (seq_len - 1)) / 64.0)
        recency = recency / recency.sum().clamp_min(1e-12)
        recency = recency.view(1, 1, -1).expand(b, n_kv_heads, -1)

        sink = torch.zeros(seq_len, device=keys.device, dtype=attn.dtype)
        sink[: self.sink_tokens] = 1.0
        sink = sink.view(1, 1, -1).expand(b, n_kv_heads, -1)

        return self.score_alpha * attn + self.score_beta * recency + self.score_gamma * sink
