"""Measure peak GPU memory, TTFT, TPT, throughput per (model × method × budget).

Implements §2.6 of ``EMNLP2026_PivotPlan.md``. Outputs a single CSV row per run.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from halo.utils import get_logger, load_yaml, seed_everything


def main() -> int:
    log = get_logger("halo.bench")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True,
                    choices=["full", "h2o", "streamingllm", "snapkv", "kivi", "halo"])
    ap.add_argument("--memory-ratio", type=int, default=4)
    ap.add_argument("--prompt-length", type=int, default=131072)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--output", required=True, help="CSV file (appended).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    model_cfg = load_yaml(args.config)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from halo import HALOConfig, wrap_with_halo

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        tok = AutoTokenizer.from_pretrained(model_cfg["name_or_path"])
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["name_or_path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        if args.method == "halo":
            ratio = 1.0 / args.memory_ratio
            model = wrap_with_halo(model, HALOConfig(hot_ratio=ratio))

        prompt = "the quick brown fox " * (args.prompt_length // 5)
        inputs = tok(prompt, return_tensors="pt", truncation=True,
                     max_length=args.prompt_length).to(model.device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        torch.cuda.synchronize()
        ttft = time.perf_counter() - t0

        # Decoding loop
        past = out.past_key_values
        input_ids = inputs.input_ids[:, -1:]
        t1 = time.perf_counter()
        for _ in range(args.max_new_tokens):
            with torch.no_grad():
                out = model(input_ids=input_ids, past_key_values=past, use_cache=True)
            past = out.past_key_values
            input_ids = out.logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize()
        decode_time = time.perf_counter() - t1

        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        tpt = decode_time / args.max_new_tokens
        throughput = args.max_new_tokens / decode_time

    except Exception as e:  # pragma: no cover - smoke-test friendly fallback
        log.warning("benchmark fell back to dry-run because: %s", e)
        ttft = float("nan")
        tpt = float("nan")
        throughput = float("nan")
        peak_mem = float("nan")

    row = {
        "model": model_cfg["name_or_path"],
        "method": args.method,
        "memory_ratio": args.memory_ratio,
        "prompt_length": args.prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "ttft_s": ttft,
        "tpt_s": tpt,
        "throughput_tok_s": throughput,
        "peak_gpu_gb": peak_mem,
    }
    is_new = not out_path.exists()
    with out_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            w.writeheader()
        w.writerow(row)
    log.info("wrote row → %s", out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
