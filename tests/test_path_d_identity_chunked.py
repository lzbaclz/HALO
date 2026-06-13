"""Path D identity test at hot_ratio < 1.0 (actual chunked LSE-merge regime).

The existing ``tests/test_chunked_wrap_integration.py`` exercises
``chunked=True`` with ``hot_ratio=1.0``, which degenerates to a pure
single-chunk evaluation (no cold tier touched) and so verifies wiring but
not the actual identity claim. ``tests/test_kv_cache_chunked.py`` tests
the cache in isolation against a numpy reference but not through a real
HuggingFace forward pass.

This test fills the gap the audit flagged ("test_path_d_identity_gpu.py":
chunked=True at hot_ratio<1, compared with full attention). It runs on
CPU with a tiny randomly-initialised Qwen2-architecture model so it is
sandbox-safe (no checkpoint download, no GPU required), and exercises
the actual cross-tier merge path::

    Path D forward at hot_ratio=0.5 (some keys cold)
      vs
    Unwrapped model forward (full attention, all keys on the same tier)

Under fp32 these must agree to within a tight tolerance (per Prop 4.5 (ii):
per-step output on a fixed KV state ≤1 ULP of the fused kernel). On a tiny
model the cumulative ULP across all layers is still << 1e-3.
"""
from __future__ import annotations

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pytest
import torch


@pytest.fixture
def tiny_model():
    """A randomly-initialised Qwen2 model small enough for CPU."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained.__wrapped__(
        AutoConfig,
        pretrained_model_name_or_path=None,
        # bypass the network and construct from scratch
    ) if False else None  # placeholder so static analysers stop complaining

    from transformers.models.qwen2 import Qwen2Config
    cfg = Qwen2Config(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA factor 2
        max_position_embeddings=4096,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()
    return model


def _short_prompt_ids(vocab_size: int, length: int):
    torch.manual_seed(1)
    return torch.randint(low=10, high=vocab_size - 10, size=(1, length))


@pytest.mark.parametrize("hot_ratio", [0.5, 0.25])
def test_path_d_chunked_identity_vs_full(tiny_model, hot_ratio: float):
    """Path D with cold-tier chunks must match Full attention in fp32.

    Asserts: with the same KV state at decode step 0, the next-token
    logits from Path D (chunked=True, hot_ratio<1) and the unwrapped
    model (single forward, full attention) agree to within 5e-4 in
    fp32 (the per-step ≤1 ULP claim of Prop 4.5 (ii) accumulated over
    2 layers × 4 heads × 64 hidden dim is well below this).
    """
    from halo import HALOConfig, wrap_with_halo

    model = tiny_model.float()  # force fp32 -- the regime where identity holds
    prompt = _short_prompt_ids(model.config.vocab_size, length=256)

    # Path A: unwrapped model, full attention (reference).
    with torch.no_grad():
        ref = model(prompt, use_cache=False)
        ref_logits = ref.logits[:, -1, :].clone()

    # Path B: wrap with Path D (chunked LSE merge, cold tier in host DRAM).
    cfg = HALOConfig(
        hot_ratio=hot_ratio,
        chunked=True,
        chunk_size=32,
        recent_window=8,
    )
    wrap_with_halo(model, cfg)
    with torch.no_grad():
        out = model(prompt, use_cache=True)
        pd_logits = out.logits[:, -1, :].clone()

    # fp32 algebraic identity (real arithmetic) + per-step ≤1 ULP bit
    # equivalence: cumulative across 2 tiny layers should be << 5e-4.
    diff = (ref_logits - pd_logits).abs().max().item()
    assert diff < 5e-4, (
        f"Path D chunked (hot_ratio={hot_ratio}) deviates {diff:.2e} > 5e-4 "
        f"from Full attention on a 256-token prompt; Prop 4.5 (i,ii) "
        f"identity is violated. Check chunk_size / recent_window / "
        f"GQA broadcast in halo/kv_cache_chunked.py."
    )


def test_path_d_chunked_argmax_unchanged(tiny_model):
    """Argmax over the next-token distribution must be unchanged.

    Stronger than the logit-tolerance test: even if some near-tied
    positions flip (qa_2-style EOS ULP-compound), the argmax on
    a non-pathological short prompt must agree. This catches regressions
    where the chunk loop drops or double-counts a position.
    """
    from halo import HALOConfig, wrap_with_halo

    model = tiny_model.float()
    prompt = _short_prompt_ids(model.config.vocab_size, length=128)

    with torch.no_grad():
        ref_argmax = model(prompt, use_cache=False).logits[:, -1, :].argmax(-1)

    cfg = HALOConfig(hot_ratio=0.5, chunked=True, chunk_size=16, recent_window=4)
    wrap_with_halo(model, cfg)
    with torch.no_grad():
        pd_argmax = model(prompt, use_cache=True).logits[:, -1, :].argmax(-1)

    assert torch.equal(ref_argmax, pd_argmax), (
        f"Path D argmax {pd_argmax.tolist()} != Full attention "
        f"{ref_argmax.tolist()} on tiny model with deterministic seed. "
        f"Identity regression."
    )
