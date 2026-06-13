"""HALOHybridPress — the HALO + StreamingLLM hybrid scoring rule.

The §5.3 RULER analysis showed that HALOPress's score-and-prune drops
needles that StreamingLLM's sliding-window keeps because the needles
sit in the recent window. This press composes the two policies:

    score = halo_score   for positions OUTSIDE  the recent window
    score = +infinity    for positions INSIDE   the recent window
                          and INSIDE the sink prefix

so that the recent window + sink are *always* retained (StreamingLLM
behaviour) and the remaining $rK - |\\mathrm{recent}| - |\\mathrm{sink}|$
slots are filled by the HALO closed-form score (HALO behaviour).

The hybrid is parameterised by ``recent_tokens``: 0 reduces to plain
HALO, while a value $\\ge rK$ reduces to plain StreamingLLM. We default
to half the budget (256 of the typical $rK \\approx 512$) which lets
the hybrid keep both the latest output context and the predicted hot
positions elsewhere in the prompt.

Why this is the right design move
---------------------------------
The §5.3 honest reading was: "HALOPress drops needles that StreamingLLM
keeps because needles sit in the recent window." The fix is exactly to
preserve the recent window, then spend the remaining budget on
attention-aware selection. Empirically this should match StreamingLLM
on NIAH (where the needle is local) AND match HALO on retrieval-heavy
LongBench tasks (where the needle is far back).
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    from kvpress import ScorerPress
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "halo.halo_hybrid_press requires kvpress. "
        "Run `pip install kvpress`."
    ) from e


@dataclass
class HALOHybridPress(ScorerPress):
    """HALO + StreamingLLM hybrid score-and-prune."""

    compression_ratio: float = 0.0
    score_alpha: float = 1.0
    score_beta: float = 0.5
    score_gamma: float = 2.0
    sink_tokens: int = 4
    # Number of recent tokens (window) to *always* keep, regardless of
    # the closed-form score. Set 0 to disable hybrid behaviour (plain
    # HALO). Set high enough to mimic StreamingLLM. Default 256 matches
    # StreamingLLM's typical recent_size at 4x compression on a 4K
    # context.
    recent_tokens: int = 256

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        import torch

        b, n_kv_heads, seq_len, _ = keys.shape

        # Base attention term — same as HALOPress.
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

        score = (self.score_alpha * attn
                 + self.score_beta * recency
                 + self.score_gamma * sink)

        # Hybrid term: force the most-recent ``recent_tokens`` positions to
        # be retained. We do this by adding a large sentinel score (equal
        # to twice the maximum existing score) to the recent window, so
        # the top-rK selection always pulls them in.
        if self.recent_tokens > 0:
            recent_mask = torch.zeros(seq_len, device=keys.device, dtype=score.dtype)
            cutoff = max(0, seq_len - self.recent_tokens)
            recent_mask[cutoff:] = 1.0
            recent_mask = recent_mask.view(1, 1, -1).expand(b, n_kv_heads, -1)
            sentinel = 2.0 * score.abs().max().clamp_min(1.0)
            score = score + recent_mask * sentinel

        return score
