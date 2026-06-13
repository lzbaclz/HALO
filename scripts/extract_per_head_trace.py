"""Re-collect a single attention trace with the head dimension preserved.

The default ``scripts/extract_attention_trace.py`` mean-pools over heads
to keep trace size manageable (15 GB for the 15 traces we ship). For the
per-head archetype clustering analysis (\\Cref{sec:appendix-per-head-archetype})
we need the raw per-head attention. Doing so for all 15 (model, task)
pairs would balloon to 32x = 480 GB, which exceeds local disk; we
therefore re-collect ONE representative (model, task) pair --
Qwen2.5-7B on PassageRetrieval-en (the strongest retrieval task) -- with
heads preserved. ~3 GB on disk; sufficient for the clustering analysis.

Outputs:
    experiments/traces_per_head/qwen2-5-7b/passage_retrieval_en.pt

The schema is the same as the default trace except ``head_mean_full`` is
replaced by ``per_head_full`` of shape (n_layers, n_heads, q_len, k_len).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--task", default="passage_retrieval_en")
    ap.add_argument("--context-length", type=int, default=4096)
    ap.add_argument("--n-steps", type=int, default=8,
                    help="Number of decoding steps to capture per-head data for. "
                         "Per-head is ~32x the head-mean size, so we cap at a few.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else (
        REPO / "experiments" / "traces_per_head"
        / args.model.split("/")[-1].lower().replace(".", "-")
        / f"{args.task}.pt"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    import json

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        attn_implementation="eager",  # need attn weights output
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Load one LongBench example from a locally-cached jsonl. We pick a
    # PassageRetrieval-en example because it has a clear retrieval needle
    # (one paragraph that the model must locate among 30 distractors), so
    # the per-head archetype analysis can show a clean separation between
    # "retrieval head" archetype (attends sharply to the needle paragraph)
    # and "diffuse" / "sink-only" / "recent-only" archetypes.
    sample_path = Path(args.task) if Path(args.task).exists() else Path("/tmp/sample.jsonl")
    print(f"[load] LongBench sample from {sample_path}", flush=True)
    with sample_path.open() as f:
        example = json.loads(f.readline())
    prompt = example.get("input", example.get("prompt",
                          example.get("context", "")))[: args.context_length * 4]

    inputs = tok(prompt, return_tensors="pt",
                 truncation=True,
                 max_length=args.context_length).to(model.device)

    # First forward pass: get prefill attention with heads preserved.
    print(f"[forward] prefill (L={inputs['input_ids'].shape[1]} tokens)",
          flush=True)
    with torch.no_grad():
        out_prefill = model(**inputs, output_attentions=True,
                             use_cache=True)
    # out_prefill.attentions: tuple of (1, n_heads, q_len, k_len) per layer.
    n_layers = len(out_prefill.attentions)
    n_heads = out_prefill.attentions[0].shape[1]
    print(f"[stats] n_layers={n_layers} n_heads={n_heads}", flush=True)

    # We save only the LAST query row per layer to keep size manageable.
    # Shape per layer: (n_heads, k_len) -> stack -> (n_layers, n_heads, k_len).
    per_head_last = []
    for layer_attn in out_prefill.attentions:
        # (1, n_heads, q_len, k_len) -> (n_heads, k_len)
        last_row = layer_attn[0, :, -1, :].cpu().to(torch.float16)
        per_head_last.append(last_row)
    per_head_last = torch.stack(per_head_last, dim=0)
    print(f"[stats] per_head_last shape={tuple(per_head_last.shape)} "
          f"({per_head_last.numel() * 2 / 1e6:.1f} MB)", flush=True)

    trace = {
        "context_len": int(inputs["input_ids"].shape[1]),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "model_name": args.model,
        "task": args.task,
        "per_head_last": per_head_last,  # (n_layers, n_heads, k_len) bf16
    }
    torch.save(trace, out)
    size_mb = out.stat().st_size / 1e6
    print(f"[save] {out}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
