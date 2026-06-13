#!/usr/bin/env python3
"""TTKV probe — DEACTIVATED in internal audit.

The previous version of this script claimed to empirically verify
TTKV orthogonality to Path D, but the overlay it installed was never
consulted by transformers' decode loop, so the comparison was vacuous
(it would have shown 100% match because both paths ran identical
vanilla Path D code).

A real TTKV head-to-head requires a target+draft speculative-decoding
loop and a draft-tree cache. We did not implement that for this paper.
The TTKV related-work entry in §6 is a structural positioning claim
(decoder draft cache vs prefix KV cache are disjoint memory regions),
not an empirical measurement. See `baselines/ttkv_probe.py` docstring
for the full caveat.

This script is retained as a stub to make the deactivation visible to
anyone who looks for it in the script directory.
"""
import sys

print(__doc__, file=sys.stderr)
sys.exit(1)
