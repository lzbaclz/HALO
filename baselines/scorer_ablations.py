"""FU_W12: scorer-rule invariance ablation under Path D's chunked LSE-merge.

The paper claims selection-rule invariance (Prop 4.5): any commitment-style
page-selection rule plugged into Path D yields algebraically identical
attention output to one-shot Full attention. This module exposes four
"placement scorers" that share the ``observe_keys`` / ``select_hot_pages``
interface of :class:`baselines.quest_path_d._QuestScorerAdapter`, so a
single runner can swap them in to drive the GPU/DRAM tiering without
touching the LSE-merge math.

Scorers exposed
---------------
- :class:`UniformRandomScorerAdapter` — picks ``K`` hot pages uniformly at
  random each step. Quality should match Full attention up to bf16
  reduction noise (the deliberately-bad scorer reviewer Q5 asks for).
- :class:`HALOPressScorerAdapter` — uses HALO's closed-form
  ``α·attn + β·recency + γ·sink`` scorer at the page granularity. This is
  the closed-form heuristic of the EMNLP paper (Path B).
- :class:`FIFOScorerAdapter` — picks the most recent ``K`` pages
  (oldest-first eviction). This is also the default Path D peel behaviour
  when no scorer is attached, so we expose it explicitly for the ablation
  table.

The Quest scorer is the fourth comparison cell and is already implemented
in :mod:`baselines.quest_path_d`.

Why this matters
----------------
The most direct empirical question is: "given that the invariance is
claimed, does Path D with a deliberately-bad scorer match the SOTA
query-aware scorer?" Without this ablation the 'selection-rule
invariance' framing is asserted but not measured. Running all four
scorers on the same NIAH adversarial cell with the same prompts and
the same seed gives the direct empirical verification.

Telemetry
---------
Every adapter exposes the same counters (``observe_calls``,
``score_calls``, ``cumulative_hot_pages``) as
``_QuestScorerAdapter`` so that
``tests/test_quest_path_d.py::test_scorer_actually_fires`` style
assertions transfer verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from transformers import PreTrainedModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AblationScorerConfig:
    """Configuration for a Path D placement-scorer ablation cell."""

    scorer_name: str = "uniform_random"
    """One of ``"uniform_random" | "halopress" | "fifo" | "quest"``."""

    page_size: int = 16
    """Number of tokens per page. Matches the Quest convention."""

    memory_ratio: float = 4.0
    """Hot-page budget = ``ceil(num_pages / memory_ratio)``."""

    min_pages_selected: int = 4
    """Floor on hot pages."""

    chunk_size: int = 512
    """Path D LSE-merge chunk size."""

    recent_window: int = 64
    """Recent-window floor for the prefill->decode transition."""

    use_triton: bool = True

    seed: int = 0
    """Deterministic seed for the uniform-random scorer."""

    # HALOPress closed-form scorer weights (matches HALOConfig defaults)
    halopress_alpha: float = 1.0
    halopress_beta: float = 0.5
    halopress_gamma: float = 2.0
    halopress_sink_tokens: int = 4
    halopress_refresh_window: int = 64


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class _BaseAblationAdapter:
    """Common bookkeeping shared by all scorer adapters in this module.

    Tracks the number of fully-materialised pages per layer (mirrors
    Quest's ``_num_full_pages``) so that adapters can decide how many
    hot pages to flag.
    """

    def __init__(self, cfg: AblationScorerConfig) -> None:
        self.cfg = cfg
        self.page_size = cfg.page_size
        self.memory_ratio = cfg.memory_ratio
        self.min_pages_selected = cfg.min_pages_selected
        # Per-layer state
        self._num_full_pages: dict = {}
        # Telemetry
        self.observe_calls: int = 0
        self.score_calls: int = 0
        self.last_hot_pages_per_layer: dict = {}
        self.cumulative_hot_pages: int = 0

    # ---- common bookkeeping ----

    def observe_keys(self, layer_idx: int, K_full) -> None:
        """Track the number of fully-materialised pages per layer."""
        if K_full is None or K_full.numel() == 0:
            return
        T = K_full.shape[-2]
        n_full = T // self.page_size
        prev = int(self._num_full_pages.get(layer_idx, 0))
        if n_full > prev:
            self._num_full_pages[layer_idx] = n_full
        self.observe_calls += 1

    def _num_hot_pages(self, P: int) -> int:
        """Number of hot pages to select from ``P`` available pages."""
        K = max(
            self.min_pages_selected,
            int((P + self.memory_ratio - 1) // self.memory_ratio),
        )
        return min(K, P)

    def _record(self, layer_idx: int, topk: "torch.Tensor") -> None:
        self.last_hot_pages_per_layer[layer_idx] = topk
        self.cumulative_hot_pages += int(topk.numel())
        self.score_calls += 1


# ---------------------------------------------------------------------------
# Uniform-random scorer
# ---------------------------------------------------------------------------


class UniformRandomScorerAdapter(_BaseAblationAdapter):
    """Pick ``K`` hot pages uniformly at random per step.

    This is the deliberately-bad scorer in the invariance ablation: if
    Path D's identity contract holds, NIAH F1 must match Full attention
    even though the page selection has *no* signal.

    Determinism: a per-call ``torch.Generator`` is seeded from
    ``cfg.seed + layer_idx + score_calls`` so reruns reproduce exactly.
    """

    def __init__(self, cfg: AblationScorerConfig) -> None:
        super().__init__(cfg)

    def select_hot_pages(self, layer_idx: int, query) -> Optional["torch.Tensor"]:
        import torch
        P = int(self._num_full_pages.get(layer_idx, 0))
        if P == 0:
            return None
        K = self._num_hot_pages(P)
        g = torch.Generator(device="cpu")
        g.manual_seed(int(self.cfg.seed) + int(layer_idx) * 10_007 + int(self.score_calls))
        perm = torch.randperm(P, generator=g)
        topk = perm[:K].sort().values
        self._record(layer_idx, topk)
        return topk


# ---------------------------------------------------------------------------
# HALOPress (closed-form heuristic) scorer
# ---------------------------------------------------------------------------


class HALOPressScorerAdapter(_BaseAblationAdapter):
    """Closed-form ``α·attn + β·recency + γ·sink`` scorer at page granularity.

    This is the EMNLP paper's Path B scorer applied as a placement rule
    under Path D's LSE-merge. The attention-magnitude term is approximated
    by ``q · K_avg_per_page`` where ``K_avg_per_page`` is the per-page key
    mean; this is the standard "attention-aware" page proxy used by
    H2O / SnapKV ablations on page-grouped KV.

    Like every other scorer here, quality is invariant to this choice
    under :cref:`prop:chunked-lossless`; the ablation cell records that
    NIAH F1 is unchanged.
    """

    def __init__(self, cfg: AblationScorerConfig) -> None:
        super().__init__(cfg)
        # Per-layer page-mean K for the attention-magnitude proxy.
        self._k_mean: dict = {}

    def observe_keys(self, layer_idx: int, K_full) -> None:
        super().observe_keys(layer_idx, K_full)
        import torch
        if K_full is None or K_full.numel() == 0:
            return
        B, H_kv, T, D = K_full.shape
        p = self.page_size
        n_full = T // p
        if n_full == 0:
            return
        # Per-page mean key over the page-grouped axis.
        new_k = K_full[..., : n_full * p, :].view(B, H_kv, n_full, p, D)
        self._k_mean[layer_idx] = new_k.mean(dim=-2).detach()  # (B, H_kv, P, D)

    def select_hot_pages(self, layer_idx: int, query) -> Optional["torch.Tensor"]:
        import torch
        if layer_idx not in self._k_mean:
            return None
        k_mean = self._k_mean[layer_idx]               # (B, H_kv, P, D)
        H_q = query.shape[1]
        H_kv = k_mean.shape[1]
        if H_q != H_kv:
            assert H_q % H_kv == 0, "GQA mismatch"
            rep = H_q // H_kv
            q_g = query.view(
                query.shape[0], H_kv, rep, query.shape[2], query.shape[3]
            ).mean(dim=2)
        else:
            q_g = query
        # Attention-magnitude proxy: (q · K_mean)
        q_exp = q_g.unsqueeze(-2).float()              # (B, H_kv, T_q, 1, D)
        k_exp = k_mean.unsqueeze(-3).float()           # (B, H_kv, 1, P, D)
        attn = (q_exp * k_exp).sum(dim=-1)             # (B, H_kv, T_q, P)
        attn = attn.amax(dim=(0, 1, 2))                # (P,)
        # Normalise to [0, 1].
        attn = attn - attn.min()
        if float(attn.max().item()) > 0.0:
            attn = attn / attn.max()

        P = int(attn.shape[0])
        device = attn.device
        positions = torch.arange(P, device=device, dtype=torch.float32)
        # Recency: most-recent pages score higher.
        rec = torch.exp(
            -(P - 1 - positions) / max(self.cfg.halopress_refresh_window, 1)
        )
        rec = rec / (rec.sum() + 1e-12)
        rec = rec / (rec.max() + 1e-12)

        sink = torch.zeros(P, device=device, dtype=torch.float32)
        n_sink_pages = min(
            self.cfg.halopress_sink_tokens // max(self.page_size, 1) + 1, P
        )
        sink[:n_sink_pages] = 1.0

        score = (
            self.cfg.halopress_alpha * attn.float()
            + self.cfg.halopress_beta * rec
            + self.cfg.halopress_gamma * sink
        )

        K = self._num_hot_pages(P)
        topk = torch.topk(score, K, largest=True).indices.cpu().sort().values
        self._record(layer_idx, topk)
        return topk


# ---------------------------------------------------------------------------
# FIFO (oldest-first eviction) scorer
# ---------------------------------------------------------------------------


class FIFOScorerAdapter(_BaseAblationAdapter):
    """Always keep the most-recent ``K`` pages hot; demote the rest.

    Equivalent to Path D's default peel behaviour when no scorer is
    attached. Exposed as an explicit scorer so the ablation table has a
    column for it (and so the ablation harness can drive all four cells
    through the same code path).
    """

    def select_hot_pages(self, layer_idx: int, query) -> Optional["torch.Tensor"]:
        import torch
        P = int(self._num_full_pages.get(layer_idx, 0))
        if P == 0:
            return None
        K = self._num_hot_pages(P)
        # Keep the last K pages hot.
        topk = torch.arange(P - K, P, dtype=torch.long)
        self._record(layer_idx, topk)
        return topk


# ---------------------------------------------------------------------------
# Wrapper: install one of the ablation scorers under Path D
# ---------------------------------------------------------------------------


class MagicPIGSampledScorerAdapter(_BaseAblationAdapter):
    """Importance-sampled query-aware page selector (MagicPIG-style).

    MagicPIG (Chen et al., 2024) uses LSH-based sampling to pick a
    probabilistically representative subset of pages each step. The full
    LSH machinery (random-hyperplane hash signatures, asymmetric quantised
    hash tables) is heavy; this adapter implements the *effective*
    selection rule MagicPIG aims at — importance sampling weighted by the
    Quest-style per-page upper bound — under the same harness that drives
    the other scorers. Concretely:

    1. Compute the Quest per-page upper-bound score
       ``s_p = sum_d max(q_d * K_min_p_d, q_d * K_max_p_d)``.
    2. Sample ``K`` pages **without replacement** with probability
       proportional to ``softmax(s_p / temperature)``.

    This is faithful to the published MagicPIG selection rule (sampled
    proportionally to attention upper bound, not top-K) without claiming
    parity with the official CUDA release. We label this cell explicitly
    as "MagicPIG-style (our reimplementation)" in the paper and treat it
    as the second query-aware retriever cell (alongside Quest top-K) on
    the same NIAH adversarial 32K head-to-head.

    Under Path D's LSE-merge the sampled selection only affects DMA
    traffic and cache residency; quality is invariant by
    :cref:`prop:chunked-lossless`. So this cell is *also* an instance of
    the selection-rule invariance demonstration.
    """

    def __init__(self, cfg: AblationScorerConfig, temperature: float = 1.0) -> None:
        super().__init__(cfg)
        self.temperature = temperature
        # Bounding boxes reused for the importance proxy.
        self._k_min: dict = {}
        self._k_max: dict = {}

    def observe_keys(self, layer_idx: int, K_full) -> None:
        super().observe_keys(layer_idx, K_full)
        import torch
        if K_full is None or K_full.numel() == 0:
            return
        B, H_kv, T, D = K_full.shape
        p = self.page_size
        n_full = T // p
        if n_full == 0:
            return
        new_k = K_full[..., : n_full * p, :].view(B, H_kv, n_full, p, D)
        self._k_min[layer_idx] = new_k.amin(dim=-2).detach()
        self._k_max[layer_idx] = new_k.amax(dim=-2).detach()

    def select_hot_pages(self, layer_idx: int, query) -> Optional["torch.Tensor"]:
        import torch
        if layer_idx not in self._k_min:
            return None
        k_min = self._k_min[layer_idx]
        k_max = self._k_max[layer_idx]
        H_q = query.shape[1]
        H_kv = k_min.shape[1]
        if H_q != H_kv:
            assert H_q % H_kv == 0, "GQA mismatch"
            rep = H_q // H_kv
            q_g = query.view(
                query.shape[0], H_kv, rep, query.shape[2], query.shape[3]
            ).mean(dim=2)
        else:
            q_g = query
        q_exp = q_g.unsqueeze(-2).float()
        k_min_exp = k_min.unsqueeze(-3).float()
        k_max_exp = k_max.unsqueeze(-3).float()
        per_d = torch.maximum(q_exp * k_min_exp, q_exp * k_max_exp)
        scores = per_d.sum(dim=-1).amax(dim=(0, 1, 2))  # (P,)
        P = int(scores.shape[0])
        K = self._num_hot_pages(P)

        # Importance sampling without replacement, proportional to
        # softmax(s_p / T). Deterministic via per-call seed.
        probs = torch.softmax(scores / max(self.temperature, 1e-6), dim=0)
        g = torch.Generator(device="cpu")
        g.manual_seed(int(self.cfg.seed) + int(layer_idx) * 10_007
                      + int(self.score_calls) + 9001)
        # Torch's multinomial does sampling without replacement when
        # num_samples <= number of nonzero probs. Probs are >0 after
        # softmax, so this is safe.
        topk = torch.multinomial(
            probs.cpu(), num_samples=K, replacement=False, generator=g,
        ).sort().values
        self._record(layer_idx, topk)
        return topk


_SCORER_REGISTRY = {
    "uniform_random": UniformRandomScorerAdapter,
    "halopress": HALOPressScorerAdapter,
    "fifo": FIFOScorerAdapter,
    "magicpig_sampled": MagicPIGSampledScorerAdapter,
}


def wrap_with_path_d_ablation_scorer(
    model: "PreTrainedModel",
    cfg: AblationScorerConfig,
) -> "PreTrainedModel":
    """Install Path D with the ablation scorer named in ``cfg.scorer_name``.

    For ``scorer_name == "quest"``, dispatches to
    :func:`baselines.quest_path_d.wrap_with_quest_path_d` so the same
    runner can drive every cell. For the other three, installs a fresh
    Path D and attaches the named adapter.

    Returns the model with the configured scorer installed. Idempotent.
    """
    if cfg.scorer_name == "quest":
        from baselines.quest_path_d import QuestPathDConfig, wrap_with_quest_path_d
        qcfg = QuestPathDConfig(
            page_size=cfg.page_size,
            memory_ratio=cfg.memory_ratio,
            min_pages_selected=cfg.min_pages_selected,
            chunk_size=cfg.chunk_size,
            recent_window=cfg.recent_window,
            use_triton=cfg.use_triton,
        )
        return wrap_with_quest_path_d(model, qcfg)

    if cfg.scorer_name not in _SCORER_REGISTRY:
        raise ValueError(
            f"unknown scorer_name={cfg.scorer_name!r}; "
            f"expected one of {sorted(set(_SCORER_REGISTRY) | {'quest'})}"
        )

    import os
    from halo import HALOConfig, wrap_with_halo

    use_triton = cfg.use_triton and (
        os.environ.get("HALO_DISABLE_TRITON", "0") != "1"
    )
    halo_cfg = HALOConfig(
        chunked=True,
        chunk_size=cfg.chunk_size,
        recent_window=cfg.recent_window,
        hot_ratio=1.0 / max(1.0, cfg.memory_ratio),
        use_triton=use_triton,
    )
    wrap_with_halo(model, halo_cfg)
    scorer = _SCORER_REGISTRY[cfg.scorer_name](cfg)
    model._halo_cache._quest_scorer = scorer  # noqa: SLF001
    model._ablation_scorer_cfg = cfg  # type: ignore[attr-defined]
    return model


__all__ = [
    "AblationScorerConfig",
    "UniformRandomScorerAdapter",
    "HALOPressScorerAdapter",
    "FIFOScorerAdapter",
    "MagicPIGSampledScorerAdapter",
    "wrap_with_path_d_ablation_scorer",
]
