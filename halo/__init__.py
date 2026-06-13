"""HALO — Algebraically Identity-Preserving KV Tiering for Long-Context Inference.

A training-free, model-agnostic KV-cache *memory hierarchy*: every cached
position stays in full precision and contributes to every step's softmax
through log-sum-exp merge across the GPU/host-DRAM tier boundary. Under
fp32 the chunked output is algebraically identical (real arithmetic) to
one-shot Full attention for *any* hot/cold partition (Prop 4.5 (i,ii)).
Free-running autoregressive trajectories are *not* claimed bit-equivalent
(part iii — see ``tests/`` and the qa_2 8K negative result).

Quickstart (Path D, the paper's headline path)
----------------------------------------------
>>> from halo import HALOConfig, wrap_with_halo
>>> cfg = HALOConfig(chunked=True, hot_ratio=0.25)  # Path D
>>> model = wrap_with_halo(hf_causal_lm, cfg)

The historical default ``chunked=False`` runs HALOPress (Path C, an
eviction-style commitment policy — theorem-covered by §3, a
\\emph{baseline} reference row in the paper, not the contribution).
Always pass ``chunked=True`` to get Path D.

Public API
----------
- :class:`HALOConfig` — configuration dataclass
- :func:`wrap_with_halo` — wrap a HuggingFace ``CausalLM`` with HALO
- :class:`HALOCache` — drop-in replacement for ``transformers`` ``Cache``
- :class:`HALOCacheChunked` — Path D's chunked LSE-merge cache
- :class:`HALOScorer`, :class:`HALODemoter`, :class:`HALORefetcher` — submodules

The three submodules correspond directly to §4 of the paper.
"""
from halo.policy import HALOConfig, wrap_with_halo
from halo.scorer import HALOScorer
from halo.demoter import HALODemoter
from halo.refetcher import HALORefetcher
from halo.kv_cache import HALOCache
from halo.kv_cache_evict import HALOCacheEvict
from halo.kv_cache_chunked import HALOCacheChunked
from halo.memory_tier import MemoryTier, TieredStorage
from halo.preforward_peel import install_preforward_peel

__all__ = [
    "HALOConfig",
    "wrap_with_halo",
    "HALOScorer",
    "HALODemoter",
    "HALORefetcher",
    "HALOCache",
    "HALOCacheEvict",
    "HALOCacheChunked",
    "MemoryTier",
    "TieredStorage",
    "install_preforward_peel",
]

__version__ = "0.1.0"
