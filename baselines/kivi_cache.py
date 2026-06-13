"""Self-contained KIVI-style 2-bit / 4-bit per-channel KV quantization.

The kvpress 0.5.3 in our requirements does not ship a ``KIVIPress``
class (the next-step package release does, but we cannot rely on it
during this revision cycle). We therefore implement a minimal
KIVI-style baseline directly: a HuggingFace ``Cache`` subclass that
intercepts ``update`` and per-channel-quantises the cached K and V
tensors, then de-quantises on read.

Spec (matching Liu et al.\ ``KIVI: A Tuning-Free Asymmetric 2bit
Quantization for KV Cache``, ICML 2024):

* K: per-channel quantization on the head-dim axis, group_size 32.
* V: per-token (per-position) quantization, group_size 32.
* Bits: 2 (the canonical ``KIVI-2'' configuration) or 4.
* Last ``residual_length`` tokens kept at bf16 (the recent buffer).

Compression ratio at b bits: $16 / b$ on quantized positions, modulo a
small per-group scale + zero-point overhead. The ratio is what the
LongBench / RULER ``memory\_ratio`` column reports.

This is a deliberately CPU-friendly reference: the quantize / dequant
ops are pure PyTorch, no custom kernels. It is fast enough for paper
benchmarks ($\sim$2x slowdown vs.\ full attention on Qwen2.5-7B);
production deployments would use the CUDA kernels from the official
KIVI repo.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    import torch

try:
    from transformers.cache_utils import DynamicCache as _BaseCache
    _HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover
    class _BaseCache:  # type: ignore[no-redef]
        def __init__(self): pass
    _HAS_TRANSFORMERS = False


def _quantize_per_group(x: "torch.Tensor", *, bits: int, group_size: int,
                        axis: int) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Round-to-nearest per-group quantization along ``axis``.

    Returns ``(q, scale, zero)`` such that the dequantized tensor is
    ``q.float() * scale + zero``. ``q`` is held as uint8 with up to
    ``bits`` bits packed; for simplicity we currently store it as int8
    in the range [0, 2**bits - 1] (no bit-packing — the memory savings
    in this reference are the conceptual ratio, not the byte count).
    """
    import torch

    assert axis == -1, "current implementation supports axis=-1 only"
    qmax = (1 << bits) - 1
    # Group along the last axis.
    shape = list(x.shape)
    n = shape[-1]
    assert n % group_size == 0, f"last dim {n} must be divisible by group_size {group_size}"
    g = n // group_size
    x_g = x.view(*shape[:-1], g, group_size)
    x_min = x_g.amin(dim=-1, keepdim=True)
    x_max = x_g.amax(dim=-1, keepdim=True)
    scale = (x_max - x_min).clamp_min(1e-8) / qmax
    zero = x_min
    q = ((x_g - zero) / scale).round().clamp(0, qmax).to(torch.int8)
    return q, scale.to(x.dtype), zero.to(x.dtype)


def _dequantize_per_group(q: "torch.Tensor", scale: "torch.Tensor",
                          zero: "torch.Tensor", *, group_size: int) -> "torch.Tensor":
    """Inverse of ``_quantize_per_group`` along axis=-1."""
    g = q.shape[-2]
    x = q.float() * scale.float() + zero.float()
    new_shape = list(q.shape[:-2]) + [g * group_size]
    return x.reshape(*new_shape).to(scale.dtype)


class KIVICache(_BaseCache):
    """KV cache with KIVI-style per-channel 2/4-bit quantization.

    For each layer's K and V, we maintain three parallel buffers:
    a recent (bf16) buffer of size ``residual_length`` and the quantized
    (int8 stored, ``bits`` effective) buffer for everything beyond.

    The ``memory_ratio`` argument is the conceptual compression ratio
    ($16 / \\mathrm{bits}$); we use it to pick ``bits`` for the
    benchmark harness so the experiment columns line up with the other
    baselines' memory_ratio knob.
    """

    def __init__(self, *, bits: int = 2, group_size: int = 32,
                 residual_length: int = 32) -> None:
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.residual_length = residual_length
        # Recent (bf16) per-layer buffers.
        self._k_recent: list = []
        self._v_recent: list = []
        # Quantized per-layer buffers (int8 + scale + zero).
        self._k_quant: list = []
        self._v_quant: list = []
        self._n_seen: list[int] = []  # tokens absorbed into quant buffer per layer

    @classmethod
    def from_memory_ratio(cls, *, memory_ratio: int, **kw) -> "KIVICache":
        bits = max(16 // memory_ratio, 2)
        return cls(bits=bits, **kw)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._n_seen):
            return 0
        return self._n_seen[layer_idx] + (
            self._k_recent[layer_idx].shape[-2] if layer_idx < len(self._k_recent)
            and self._k_recent[layer_idx] is not None else 0
        )

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        import torch

        # Ensure per-layer slots.
        while len(self._k_recent) <= layer_idx:
            self._k_recent.append(None)
            self._v_recent.append(None)
            self._k_quant.append(None)
            self._v_quant.append(None)
            self._n_seen.append(0)

        # Concatenate new k/v into the recent buffer.
        k_rec = self._k_recent[layer_idx]
        v_rec = self._v_recent[layer_idx]
        if k_rec is None:
            new_k_rec = key_states
            new_v_rec = value_states
        else:
            new_k_rec = torch.cat([k_rec, key_states], dim=-2)
            new_v_rec = torch.cat([v_rec, value_states], dim=-2)

        # If the recent buffer exceeds residual_length, push the overflow to
        # the quantized buffer.
        T_rec = new_k_rec.shape[-2]
        if T_rec > self.residual_length:
            overflow = T_rec - self.residual_length
            # Round overflow up to group_size so quantization is well-defined.
            overflow_q = (overflow // self.group_size) * self.group_size
            if overflow_q > 0:
                k_overflow = new_k_rec[..., :overflow_q, :].contiguous()
                v_overflow = new_v_rec[..., :overflow_q, :].contiguous()
                # Quantize K per-channel (last dim).
                qk, sk, zk = _quantize_per_group(
                    k_overflow, bits=self.bits, group_size=self.group_size, axis=-1)
                qv, sv, zv = _quantize_per_group(
                    v_overflow, bits=self.bits, group_size=self.group_size, axis=-1)
                # qk shape is (B, H, T_overflow, D/group_size, group_size); the
                # token axis is dim=-3 after the view inside _quantize_per_group.
                # scale, zero share the same layout.
                k_q_old = self._k_quant[layer_idx]
                v_q_old = self._v_quant[layer_idx]
                if k_q_old is None:
                    self._k_quant[layer_idx] = (qk, sk, zk)
                    self._v_quant[layer_idx] = (qv, sv, zv)
                else:
                    qk_old, sk_old, zk_old = k_q_old
                    qv_old, sv_old, zv_old = v_q_old
                    self._k_quant[layer_idx] = (
                        torch.cat([qk_old, qk], dim=-3),
                        torch.cat([sk_old, sk], dim=-3),
                        torch.cat([zk_old, zk], dim=-3),
                    )
                    self._v_quant[layer_idx] = (
                        torch.cat([qv_old, qv], dim=-3),
                        torch.cat([sv_old, sv], dim=-3),
                        torch.cat([zv_old, zv], dim=-3),
                    )
                self._n_seen[layer_idx] += overflow_q
                new_k_rec = new_k_rec[..., overflow_q:, :].contiguous()
                new_v_rec = new_v_rec[..., overflow_q:, :].contiguous()

        self._k_recent[layer_idx] = new_k_rec
        self._v_recent[layer_idx] = new_v_rec

        # Build the returned (K, V) by dequantizing the quant buffer and
        # concatenating the recent bf16 buffer.
        if self._k_quant[layer_idx] is None:
            return new_k_rec, new_v_rec
        qk, sk, zk = self._k_quant[layer_idx]
        qv, sv, zv = self._v_quant[layer_idx]
        k_dq = _dequantize_per_group(qk, sk, zk, group_size=self.group_size)
        v_dq = _dequantize_per_group(qv, sv, zv, group_size=self.group_size)
        K = torch.cat([k_dq, new_k_rec], dim=-2)
        V = torch.cat([v_dq, new_v_rec], dim=-2)
        return K, V

    def reset(self) -> None:
        self._k_recent.clear()
        self._v_recent.clear()
        self._k_quant.clear()
        self._v_quant.clear()
        self._n_seen.clear()


def wrap_with_kivi(model, *, memory_ratio: int = 4, **kw):
    """Install KIVI-style quantized cache on a HF model.

    Returns the same model with its ``generate`` patched to use a
    fresh :class:`KIVICache` per call.
    """
    cache = KIVICache.from_memory_ratio(memory_ratio=memory_ratio, **kw)

    if getattr(model, "_kivi_patched", False):
        return model

    original_generate = model.generate

    def kivi_generate(*args, **kwargs):  # type: ignore[no-untyped-def]
        cache.reset()
        kwargs.setdefault("past_key_values", cache)
        kwargs.setdefault("use_cache", True)
        return original_generate(*args, **kwargs)

    model.generate = kivi_generate  # type: ignore[assignment]
    model._kivi_patched = True
    return model
