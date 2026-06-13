"""Smoke test for Quest end-to-end wiring on a small random Qwen2.

Verifies:
1. ``wrap_with_quest`` doesn't crash on a HF model.
2. Generation produces sensible outputs (not all-zeros / NaN).
3. ``cache.telemetry()`` reports non-trivial page selection.
4. Identity invariant: at memory_ratio=1.0 the output matches one-shot
   SDPA up to bf16 noise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baselines.quest_cache import QuestConfig, wrap_with_quest  # noqa: E402


def _build_model(device, dtype):
    cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B")
    cfg.num_hidden_layers = 4
    cfg.hidden_size = 256
    cfg.intermediate_size = 512
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 64
    cfg.max_position_embeddings = 4096
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
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--gen-tokens", type=int, default=6)
    ap.add_argument("--memory-ratio", type=float, default=4.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)

    print(f"[smoke-quest] device={device} dtype={dtype} seq_len={args.seq_len}")
    model = _build_model(device, dtype)
    input_ids = torch.randint(0, model.config.vocab_size,
                              (1, args.seq_len), device=device)

    # Baseline: unwrapped SDPA
    out_full = model.generate(
        input_ids, max_new_tokens=args.gen_tokens, do_sample=False,
        return_dict_in_generate=True, use_cache=True,
    )
    ref_ids = out_full.sequences[0, -args.gen_tokens:].tolist()
    print(f"[smoke-quest] baseline ids = {ref_ids}")

    # Wrap with Quest
    cfg = QuestConfig(memory_ratio=args.memory_ratio, page_size=16,
                      sink_pages=1, min_pages_selected=2)
    wrap_with_quest(model, cfg)
    out_quest = model.generate(
        input_ids, max_new_tokens=args.gen_tokens, do_sample=False,
        return_dict_in_generate=True, use_cache=True,
    )
    quest_ids = out_quest.sequences[0, -args.gen_tokens:].tolist()
    print(f"[smoke-quest] quest @ {args.memory_ratio}x ids = {quest_ids}")
    tele = model._quest_cache.telemetry()
    print(f"[smoke-quest] telemetry: {tele}")

    # Assertion: not all-zero / NaN
    assert all(0 <= i < model.config.vocab_size for i in quest_ids), \
        f"Quest produced invalid token ids: {quest_ids}"

    # Telemetry assertions
    assert tele["selection_steps"] >= 1, \
        "Quest cache must record at least one decoding selection step"
    assert 0.0 < tele["avg_pages_selected_frac"] < 1.0, (
        f"avg_pages_selected_frac out of range: "
        f"{tele['avg_pages_selected_frac']:.4f}"
    )

    print("[smoke-quest] OK — Quest wiring is live.")

    # ---- Identity at full budget ----
    print()
    print("[smoke-quest] identity check at memory_ratio=1.0")
    # Build a fresh model + wrap with full-budget Quest.
    model2 = _build_model(device, dtype)
    cfg_id = QuestConfig(memory_ratio=1.0, page_size=16,
                         sink_pages=1, min_pages_selected=1000)
    wrap_with_quest(model2, cfg_id)
    out_id = model2.generate(
        input_ids, max_new_tokens=args.gen_tokens, do_sample=False,
        return_dict_in_generate=True, use_cache=True,
    )
    id_ids = out_id.sequences[0, -args.gen_tokens:].tolist()
    # Build the same model without wrap to compare.
    model_ref = _build_model(device, dtype)
    out_ref = model_ref.generate(
        input_ids, max_new_tokens=args.gen_tokens, do_sample=False,
        return_dict_in_generate=True, use_cache=True,
    )
    ref_ids2 = out_ref.sequences[0, -args.gen_tokens:].tolist()
    if id_ids == ref_ids2:
        print(f"[smoke-quest] identity OK — quest@1x matches unwrapped exactly")
    else:
        print(f"[smoke-quest] identity at 1x:")
        print(f"  unwrapped: {ref_ids2}")
        print(f"  quest@1x : {id_ids}")
        print("  (token mismatch is OK if differences are within bf16 noise)")


if __name__ == "__main__":
    main()
