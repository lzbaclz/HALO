"""FU_W5: official CUDA-kernel KIVI on RULER NIAH adversarial.

Runs the upstream jy-yuan/KIVI repository's ``LlamaForCausalLM_KIVI``
(2/4-bit per-channel K / per-token V quantisation, with the CUDA
``kivi_gemv`` kernel) on the same RULER NIAH subtasks the paper
evaluates Path D on. This is the calibrated head-to-head that the
paper registers as FU_W5 in
``scripts/repro/auxiliary_cells.sh``.

Constraint: the upstream KIVI repo ships
``LlamaForCausalLM_KIVI`` and ``MistralForCausalLM_KIVI`` only — no
Qwen wrapper. We therefore run on the canonical KIVI deployment
target (Llama-2-7b-hf, which is in the local HF cache); the resulting
KIVI vs. Path D comparison is apples-to-apples on the same Llama
backbone Path D runs on in Cell C (Llama-3.1-8B-Instruct), with the
caveat that Llama-2-7b-hf is the older checkpoint KIVI was developed
against. We report this honestly in the paper.

Usage::

    ~/miniconda3/envs/kivi_py312/bin/python scripts/run_kivi_official_ruler.py \\
        --subtask niah_multikey_1 --context-length 32768 --n-examples 30 \\
        --k-bits 4 --v-bits 4 --group-size 32 --residual-length 32 \\
        --output experiments/fu_w5_kivi/llama2_7b_mk1_32k

KIVI's example uses K_BITS=V_BITS=2 (the headline configuration) but
the HF-canonical comparison in the paper measured KIVI at int4 (HQQ
4-bit). We default to ``--k-bits 4 --v-bits 4`` to land the apples-to-
apples comparison with the existing Cell C HQQ KIVI int4 numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _ruler_score(pred: str, gold) -> float:
    if isinstance(gold, str):
        golds = [gold]
    else:
        golds = list(gold)
    for g in golds:
        if str(g) in pred:
            return 1.0
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-path",
        default=os.environ.get("KIVI_DEFAULT_MODEL", "meta-llama/Llama-2-7b-hf"),
        help="HF model id; KIVI supports Llama/Mistral architectures only.",
    )
    ap.add_argument("--subtask", required=True)
    ap.add_argument("--context-length", type=int, default=32768)
    ap.add_argument("--n-examples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--k-bits", type=int, default=4,
                    choices=[2, 4],
                    help="K quantisation bits (KIVI supports 2 or 4)")
    ap.add_argument("--v-bits", type=int, default=4,
                    choices=[2, 4])
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--residual-length", type=int, default=32,
                    help="Number of recent fp16 tokens KIVI keeps unquantised")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--use-flash", action="store_true", default=False,
                    help="Enable flash-attention during prefill (KIVI option)")
    ap.add_argument("--ruler-data-root",
                    default="experiments/ruler_data",
                    help="Root dir containing <subtask>/<ctx>.jsonl (non-qwen layout)")
    args = ap.parse_args()

    import torch  # noqa: E402

    # KIVI's repo isn't a package on sys.path by default; we cloned to /tmp/KIVI.
    if "/tmp/KIVI" not in sys.path:
        sys.path.insert(0, "/tmp/KIVI")
    from models.llama_kivi import LlamaForCausalLM_KIVI  # type: ignore
    from transformers import LlamaConfig, AutoTokenizer  # noqa: E402

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    # RULER path: the qwen-specific dir uses qwen2.5-7b/<ctx>/<sub>/validation.jsonl;
    # the non-qwen dir uses <sub>/<ctx>.jsonl. Llama 2 uses the non-qwen layout.
    raw_path = Path(args.ruler_data_root) / args.subtask / f"{args.context_length}.jsonl"
    if not raw_path.exists():
        sys.exit(f"[fu_w5] RULER {args.subtask}@{args.context_length} not found at {raw_path}")
    examples = [
        json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()
    ]
    selected = examples[: args.n_examples]
    print(f"[fu_w5 KIVI] subtask={args.subtask} ctx={args.context_length} "
          f"n_loaded={len(examples)} n_selected={len(selected)}", flush=True)

    print(f"[fu_w5 KIVI] loading {args.model_path}", flush=True)
    config = LlamaConfig.from_pretrained(args.model_path)
    config.k_bits = args.k_bits
    config.v_bits = args.v_bits
    config.group_size = args.group_size
    config.residual_length = args.residual_length
    config.use_flash = args.use_flash

    model = LlamaForCausalLM_KIVI.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    ).cuda().eval()

    tok = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=False, trust_remote_code=True,
    )
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    print(f"[fu_w5 KIVI] loaded; baseline GPU={torch.cuda.memory_allocated()/1e9:.2f} GiB "
          f"  K{args.k_bits}V{args.v_bits} group={args.group_size} "
          f"resid={args.residual_length} flash={args.use_flash}", flush=True)

    preds = []
    for i, ex in enumerate(selected):
        prompt = ex.get("input", "")
        if ex.get("answer_prefix") and not prompt.rstrip().endswith(
            ex["answer_prefix"].rstrip()
        ):
            prompt = prompt + ex["answer_prefix"]
        gold = ex.get("outputs", ex.get("answer", ""))

        torch.cuda.reset_peak_memory_stats()
        ids = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=args.context_length).to("cuda")
        t0 = time.time()
        with torch.no_grad():
            out_ids = model.generate(
                **ids, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tok.eos_token_id,
            )
        wall = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        pred = tok.decode(out_ids[0, ids.input_ids.shape[1]:],
                          skip_special_tokens=True)
        score = _ruler_score(pred, gold)
        preds.append({
            "index": i,
            "ctx_len": int(ids.input_ids.shape[1]),
            "pred": pred[:300],
            "gold": gold if isinstance(gold, str) else list(gold),
            "score": score,
            "wall_s": wall,
            "peak_gpu_gib": peak,
        })
        print(f"  [{i+1}/{len(selected)}] KIVI-K{args.k_bits}V{args.v_bits} "
              f"{args.subtask}@{args.context_length} score={score:.0f} "
              f"wall={wall:.1f}s peak={peak:.2f}GiB", flush=True)
        torch.cuda.empty_cache()

    scores = [p["score"] for p in preds]
    walls = [p["wall_s"] for p in preds]
    peaks = [p["peak_gpu_gib"] for p in preds]
    summary = {
        "method": f"kivi_official_K{args.k_bits}V{args.v_bits}",
        "k_bits": args.k_bits,
        "v_bits": args.v_bits,
        "group_size": args.group_size,
        "residual_length": args.residual_length,
        "use_flash": args.use_flash,
        "subtask": args.subtask,
        "context_length": args.context_length,
        "seed": args.seed,
        "n": len(scores),
        "mean_score_pct": (100.0 * sum(scores) / len(scores)) if scores else None,
        "mean_wall_s": (sum(walls) / len(walls)) if walls else None,
        "mean_peak_gpu_gib": (sum(peaks) / len(peaks)) if peaks else None,
        "max_peak_gpu_gib": max(peaks) if peaks else None,
        "model_path": args.model_path,
        "kivi_repo": "https://github.com/jy-yuan/KIVI (CUDA kernel kivi_gemv)",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "preds.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    if summary["mean_score_pct"] is None:
        print(f"[fu_w5 KIVI] no scores", flush=True)
    else:
        print(f"\n[fu_w5 KIVI] {args.subtask}@{args.context_length} K{args.k_bits}V{args.v_bits}: "
              f"mean = {summary['mean_score_pct']:.2f}% over n={summary['n']}",
              flush=True)


if __name__ == "__main__":
    main()
