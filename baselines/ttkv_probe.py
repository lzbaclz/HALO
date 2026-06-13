"""TTKV (Token-Tree KV, Liao et al., 2025) — *conceptual* composability
note, not a runnable head-to-head baseline.

The submission's related-work positioning (\\S6) claims TTKV is
\\emph{orthogonal} to Path D: TTKV operates on the decoder's speculative
draft-tree trajectory, not on the prefix KV cache, so the two compose
rather than substitute.

\\textbf{Why this file does not provide an empirical orthogonality
probe.} A real orthogonality verification would require:

  (a) a target+draft speculative-decoding loop wired through
      transformers' \\texttt{assistant_model=} or vLLM's spec-dec backend,
  (b) the draft tree cache wired into Path D's hot tier so the draft
      tokens share GPU memory with the prefix's hot positions, and
  (c) a side-by-side wall-clock + accuracy comparison of
      [Path D] vs [Path D + TTKV draft cache] under both greedy and
      sampling decoding.

We did not implement (a)--(c) in this round; the pieces below are the
\\emph{conceptual sketch} of what such a probe would look like.
\\textbf{Do not} cite this file as evidence that the orthogonality
claim has been empirically verified. The related-work entry for TTKV
should be read as a positioning claim grounded in the structural
argument (prefix cache vs.\\ decoder trajectory cache are disjoint
memory regions), not as a measured result.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TTKVProbeConfig:
    n_draft: int = 4
    seed: int = 0


def make_draft_cache_overlay():
    """Conceptual placeholder. Attaches a marker attribute to a
    HALOCacheChunked instance documenting where a real TTKV draft cache
    would live. The transformers decode loop does NOT consult this
    attribute; calling this function does not produce different
    predictions from vanilla Path D.

    Kept in-tree only so that future work has a concrete attachment
    point; not invoked from any published numerical claim.
    """
    def overlay(halo_cache):
        halo_cache._ttkv_draft_overlay = {
            "installed": True,
            "n_draft": 0,
            "is_active": False,
            "note": (
                "Conceptual marker only. A real TTKV integration would "
                "implement assistant-model spec-decoding and a draft-tree "
                "cache that shares GPU memory with the hot tier."
            ),
        }
        return halo_cache
    return overlay
