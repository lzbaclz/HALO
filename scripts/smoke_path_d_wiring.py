"""Smoke test: verify Path D *actually* offloads cold K/V to host DRAM.

Runs a small randomly-initialised Qwen2-architecture model end-to-end
through ``wrap_with_halo(model, HALOConfig(chunked=True))`` with a
prompt long enough to trigger the warmup→chunked transition. After
generation, asserts:

1. ``cache._mode == "chunked"``  (we actually transitioned).
2. For each layer that saw a peel: ``cache._cold_k[i].device.type == "cpu"``
   and ``cache._cold_k[i].shape[-2] > 0``  (cold tier exists on host).
3. ``cache.layers[i].keys.shape[-2] <= recent_window + chunk_size``
   (parent's GPU tensor was actually shrunk).
4. ``cache._dma_bytes_cumulative > 0``  (real bytes moved to host).
5. Logits within 5e-2 of unwrapped baseline (loose bf16 noise tolerance;
   bit-equivalence proven separately by the LSE-merge unit tests).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, Qwen2ForCausalLM

# Repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from halo.policy import HALOConfig, wrap_with_halo  # noqa: E402


def _make_small_qwen(seq_len: int, device: str, dtype) -> Qwen2ForCausalLM:
    """Random Qwen2-arch model — no checkpoint, fast load."""
    cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B")
    # Shrink for smoke test: 4 layers, 128 hidden, 4 heads, 2 KV heads.
    cfg.num_hidden_layers = 4
    cfg.hidden_size = 256
    cfg.intermediate_size = 512
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 64
    cfg.max_position_embeddings = max(8192, seq_len + 64)
    cfg.use_cache = True
    cfg._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(cfg).to(device=device, dtype=dtype).eval()
    torch.manual_seed(0)
    for p in model.parameters():
        p.data.normal_(0, 0.02)
    return model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=1800,
                    help="Prompt length. Must exceed 2 * chunk_size to trigger transition.")
    ap.add_argument("--gen-tokens", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--recent-window", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--logit-tol", type=float, default=5e-2)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)

    print(f"[smoke] device={device} dtype={dtype} seq_len={args.seq_len} "
          f"chunk_size={args.chunk_size} recent_window={args.recent_window}")

    # Build model + baseline reference
    model = _make_small_qwen(args.seq_len, device, dtype)
    input_ids = torch.randint(0, model.config.vocab_size, (1, args.seq_len), device=device)

    print("[smoke] baseline forward (unwrapped, full SDPA)…")
    out_full = model(input_ids=input_ids, use_cache=False)
    last_logits_full = out_full.logits[:, -1, :].detach().clone()

    # Wrap with Path D
    print("[smoke] wrapping model with HALOCacheChunked…")
    cfg = HALOConfig(
        chunked=True,
        chunk_size=args.chunk_size,
        recent_window=args.recent_window,
        hot_ratio=1.0,
    )
    wrap_with_halo(model, cfg)
    cache = model._halo_cache
    assert cache._mode == "warmup", "cache should start in warmup mode"

    # Run prefill + a few decode steps
    print("[smoke] generating…")
    out = model.generate(
        input_ids,
        max_new_tokens=args.gen_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        return_dict_in_generate=True,
        output_logits=True,
    )
    gen_logits = out.logits  # tuple of length gen_tokens, each (1, V)

    # ---- assertions ----
    failures = []

    # 1: mode is chunked
    if cache._mode != "chunked":
        failures.append(f"ASSERT 1 FAILED: mode is '{cache._mode}', expected 'chunked'")
    else:
        print("  [OK 1] cache._mode == 'chunked'")

    # 2: cold tier exists on CPU with positions
    cold_layers = list(cache._cold_k.keys())
    if not cold_layers:
        failures.append("ASSERT 2 FAILED: no cold tier was populated")
    else:
        for li in cold_layers:
            ck = cache._cold_k[li]
            if ck.device.type != "cpu":
                failures.append(
                    f"ASSERT 2 FAILED: cold_k[{li}].device={ck.device}, expected cpu"
                )
            elif ck.shape[-2] == 0:
                failures.append(
                    f"ASSERT 2 FAILED: cold_k[{li}].shape[-2]==0"
                )
        if not failures:
            print(f"  [OK 2] cold tier populated on CPU, {len(cold_layers)} layers, "
                  f"per-layer cold positions={cache._cold_k[cold_layers[0]].shape[-2]}")

    # 3: parent's GPU tensor was actually shrunk
    max_recent_allowed = args.recent_window + args.chunk_size + 1
    for li, layer_obj, k_gpu, _ in cache._iter_layer_kv():
        if li not in cold_layers:
            continue
        gpu_T = k_gpu.shape[-2] if k_gpu is not None and k_gpu.numel() > 0 else 0
        if gpu_T > max_recent_allowed:
            failures.append(
                f"ASSERT 3 FAILED: parent layers[{li}].keys.shape[-2]={gpu_T} "
                f"> recent_window+chunk_size+1={max_recent_allowed}"
            )
            break
    else:
        sample_layer = cold_layers[0] if cold_layers else 0
        gpu_T = cache._get_layer_kv(sample_layer)[0].shape[-2]
        cold_T = cache._cold_k[sample_layer].shape[-2]
        print(f"  [OK 3] parent layer {sample_layer}: GPU rows={gpu_T} (≤{max_recent_allowed}), "
              f"CPU rows={cold_T}, total={gpu_T + cold_T}")

    # 4: real DMA bytes moved
    if cache._dma_bytes_cumulative == 0:
        failures.append("ASSERT 4 FAILED: _dma_bytes_cumulative == 0")
    else:
        mb = cache._dma_bytes_cumulative / (1024 ** 2)
        print(f"  [OK 4] DMA bytes cumulative = {mb:.2f} MiB")

    # 5: forward equivalence — we compare the FIRST generated logit
    # (which corresponds to position seq_len, the next token after the
    # prompt) to the unwrapped baseline's last-position logits. They
    # should match within bf16 noise tolerance.
    last_logits_halo = gen_logits[0]
    max_diff = (last_logits_full - last_logits_halo).abs().max().item()
    rel_diff = max_diff / max(1e-6, last_logits_full.abs().max().item())
    if max_diff > args.logit_tol:
        failures.append(
            f"ASSERT 5 FAILED: max logit diff {max_diff:.4e} > tol {args.logit_tol:.4e} "
            f"(rel {rel_diff:.4e})"
        )
    else:
        print(f"  [OK 5] max logit diff {max_diff:.4e} ≤ tol {args.logit_tol:.4e} "
              f"(rel {rel_diff:.4e})")

    # Print telemetry
    print("\n[smoke] telemetry:")
    print(json.dumps(cache.telemetry(), indent=2, default=str))

    if failures:
        print("\n[smoke] ===== FAILED =====")
        for f in failures:
            print(" ", f)
        sys.exit(1)
    else:
        print("\n[smoke] ===== ALL PASS =====")


if __name__ == "__main__":
    main()
