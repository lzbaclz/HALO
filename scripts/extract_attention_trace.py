"""Collect a per-step attention trace for one prompt.

The output ``.pt`` file can be consumed by :mod:`scripts.compute_hotness` or by
custom downstream analysis code.

Usage::

    python scripts/extract_attention_trace.py \\
        --model meta-llama/Llama-3-8B-Instruct \\
        --prompt-file prompts/niah_single_1.txt \\
        --output traces/qwen2-5-7b/niah_single_1.pt

Notes
-----
* HuggingFace ``output_attentions=True`` requires ``attn_implementation="eager"``.
  Flash-Attention 2 / SDPA do not return attention probabilities.
* Eager attention OOMs on very long contexts. We trim trace collection to the
  first ``--max-context`` tokens (default 4096).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def collect_trace(model, tokenizer, prompt: str, max_new_tokens: int = 64,
                  topk_per_step: int = 256, max_context: int = 4096) -> dict:
    """Collect a per-step attention trace for one prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=max_context).to(model.device)
    L = inputs.input_ids.size(1)
    print(f"[trace] context length = {L}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_attentions=True,
            return_dict_in_generate=True,
            do_sample=False,
        )

    n_steps = len(outputs.attentions)
    n_layers = len(outputs.attentions[0])

    trace = {
        "context_len": L,
        "n_steps": n_steps,
        "n_layers": n_layers,
        "topk": topk_per_step,
        "hot_indices": [],
        "hot_values": [],
        "head_mean_full": [],
        "model_name": getattr(getattr(model, "config", None), "name_or_path", "unknown"),
    }

    for step_idx, step_attn in enumerate(outputs.attentions):
        step_hot_idx, step_hot_val = [], []
        for layer_idx, layer_attn in enumerate(step_attn):
            # layer_attn shape: (1, n_heads, q_len, k_len)
            head_mean = layer_attn[0].mean(dim=0)  # (q_len, k_len)
            last_row = head_mean[-1]               # (k_len,)
            vals, idxs = torch.topk(last_row, k=min(topk_per_step, last_row.size(0)))
            step_hot_idx.append(idxs.cpu())
            step_hot_val.append(vals.float().cpu())
            if step_idx == 0:
                trace["head_mean_full"].append(head_mean.cpu().to(torch.float16))
        trace["hot_indices"].append(step_hot_idx)
        trace["hot_values"].append(step_hot_val)

    return trace


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation="eager",     # required for output_attentions=True
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    trace = collect_trace(model, tok, prompt,
                          max_new_tokens=args.max_new_tokens,
                          topk_per_step=args.topk,
                          max_context=args.max_context)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trace, out)
    print(f"[trace] saved to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
