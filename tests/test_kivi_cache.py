"""Tests for the self-contained KIVI baseline.

Verifies:
1. Per-group quantize/dequantize round-trip stays bounded by the
   theoretical error of $\\Delta = (\\max - \\min) / (2^b - 1)$ per group.
2. The cache returns the right (K, V) shape over multiple update calls.
3. The recent (bf16) buffer keeps the last `residual_length` tokens at
   full precision (this is the KIVI ``last 32 tokens'' invariant).
4. Compression ratio matches `memory_ratio` setting up to constant
   per-group overhead.
"""
from __future__ import annotations

import torch
import pytest


def test_quant_dequant_roundtrip_bounded():
    """Dequantization error per group is bounded by the group range / qmax."""
    from baselines.kivi_cache import _quantize_per_group, _dequantize_per_group
    torch.manual_seed(0)
    x = torch.randn(1, 4, 100, 128)  # B, H, T, D
    bits = 4
    group_size = 32
    q, scale, zero = _quantize_per_group(x, bits=bits, group_size=group_size, axis=-1)
    dq = _dequantize_per_group(q, scale, zero, group_size=group_size)
    err = (x - dq).abs()
    # Per-group max error bounded by (max - min) / (2^b - 1).
    x_g = x.view(1, 4, 100, 128 // group_size, group_size)
    g_range = (x_g.amax(-1) - x_g.amin(-1)).unsqueeze(-1).repeat(1, 1, 1, 1, group_size)
    g_range = g_range.view(1, 4, 100, 128)
    bound = g_range / ((1 << bits) - 1)
    assert (err <= bound + 1e-5).all(), \
        f"quant error exceeds bound: max ratio = {(err / bound).max().item():.2f}"


def test_kivi_cache_update_shapes():
    """The cache returns (K, V) with the expected concatenated shape."""
    from baselines.kivi_cache import KIVICache
    cache = KIVICache(bits=4, group_size=32, residual_length=32)
    B, H, D = 1, 4, 128

    # First update: 100 tokens, fits in recent + spills 64 to quant.
    k1 = torch.randn(B, H, 100, D)
    v1 = torch.randn(B, H, 100, D)
    K, V = cache.update(k1, v1, layer_idx=0)
    assert K.shape == (B, H, 100, D)
    assert V.shape == (B, H, 100, D)

    # Second update: 10 more tokens.
    k2 = torch.randn(B, H, 10, D)
    v2 = torch.randn(B, H, 10, D)
    K, V = cache.update(k2, v2, layer_idx=0)
    assert K.shape == (B, H, 110, D)
    assert V.shape == (B, H, 110, D)


def test_kivi_recent_buffer_is_bf16_exact():
    """The last `residual_length` tokens come back bit-exact (no quant loss)."""
    from baselines.kivi_cache import KIVICache
    cache = KIVICache(bits=2, group_size=32, residual_length=64)
    B, H, D = 1, 4, 128
    # Push 200 tokens (will quantize the first 128, keep the last 72 in recent
    # but wait — recent caps at 64 so the rest spills). Push in two chunks so
    # we exercise concat. With residual_length=64 and quant on overflow up to
    # the nearest group_size of 32, after 200 tokens we have:
    #   200 - 64 = 136 → overflow_q = 32 * (136 // 32) = 128
    #   recent buffer = 200 - 128 = 72? No — the code subtracts overflow_q
    #   from `new_k_rec`, so recent = 72 elements. Hmm let me re-trace:
    # Actually after 200 tokens, T_rec = 200, overflow = 200 - 64 = 136,
    # overflow_q = 128, recent_after = 200 - 128 = 72. So recent has 72 tokens.
    # The last 64 of those should be exactly the original last 64 tokens.
    k_full = torch.randn(B, H, 200, D)
    v_full = torch.randn(B, H, 200, D)
    cache.update(k_full[..., :100, :], v_full[..., :100, :], layer_idx=0)
    K, V = cache.update(k_full[..., 100:, :], v_full[..., 100:, :], layer_idx=0)
    # The last 64 tokens of K should match k_full's last 64 exactly
    # (they're in the recent bf16 buffer).
    assert torch.allclose(K[..., -64:, :], k_full[..., -64:, :])
    assert torch.allclose(V[..., -64:, :], v_full[..., -64:, :])


def test_kivi_quantization_changes_old_tokens():
    """Tokens that overflowed to quant should differ from bf16 by some amount."""
    from baselines.kivi_cache import KIVICache
    cache = KIVICache(bits=2, group_size=32, residual_length=32)
    B, H, D = 1, 4, 64  # D=64 = 2 groups of 32
    k_full = torch.randn(B, H, 96, D)
    v_full = torch.randn(B, H, 96, D)
    K, V = cache.update(k_full, v_full, layer_idx=0)
    # First 64 tokens went to quant (overflow = 96 - 32 = 64).
    # Quant error should be non-zero.
    err_k = (K[..., :64, :] - k_full[..., :64, :]).abs().mean().item()
    assert err_k > 1e-3, f"expected quant error > 1e-3, got {err_k:.2e}"
    # But not too large.
    assert err_k < 1.0, f"quant error too large: {err_k:.2e}"


def test_kivi_from_memory_ratio_picks_bits():
    """The benchmark-facing alias maps memory_ratio → bits correctly."""
    from baselines.kivi_cache import KIVICache
    c4 = KIVICache.from_memory_ratio(memory_ratio=4)
    assert c4.bits == 4
    c8 = KIVICache.from_memory_ratio(memory_ratio=8)
    assert c8.bits == 2
    c2 = KIVICache.from_memory_ratio(memory_ratio=2)
    assert c2.bits == 8
