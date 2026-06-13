"""vLLM (PagedAttention) baseline runner for the headline cells.

PagedAttention is the closest published *system* baseline to Path D's
host-tier KV offload: both partition KV memory into pages, but
PagedAttention's pages live entirely in GPU memory while Path D's
cold pages live in host DRAM and stream back via LSE merge on every
attention call. This script measures vLLM's PagedAttention end-to-end
on the same prompts (RULER NIAH adversarial + InfiniteBench EnQA) so
the paper can report PagedAttention as a head-to-head baseline rather
than only describing it taxonomically.

Note on what PagedAttention is and is not:
- PagedAttention does NOT discard or quantize KV entries. Every page
  contributes to every attention call. Quality should match Full
  attention (modulo numerics; vLLM is bf16/fp16 + flash-attn fused
  kernels). The mechanism difference from Path D is purely placement:
  vLLM = all KV on GPU in paged blocks; Path D = hot KV on GPU +
  cold KV on host DRAM with chunked LSE merge.
- Peak GPU usage: vLLM still keeps all KV on device, so on a single
  80 GiB GPU it OOMs on 65K context at 32B model weight. On 7B, it
  fits comfortably.

Usage:
    python scripts/run_vllm_baseline.py \\
        --task ruler_niah_multikey_1 --context-length 32768 \\
        --n-examples 20 --output experiments/.../W13_vllm_niah_mk1
    python scripts/run_vllm_baseline.py \\
        --task infinitebench_en_qa --context-length 65000 \\
        --n-examples 20 --output experiments/.../W13_vllm_enqa_65k
"""
from __future__ import annotations

import os
import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _ruler_score(pred: str, gold) -> float:
    if isinstance(gold, str):
        golds = [gold]
    else:
        golds = list(gold)
    for g in golds:
        if str(g) in pred:
            return 1.0
    return 0.0


def _load_ruler(subtask: str, context_length: int):
    p = Path(f"experiments/ruler_data/qwen2.5-7b/{context_length}/{subtask}/validation.jsonl")
    if not p.exists():
        raise FileNotFoundError(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _load_enqa(context_length: int, n_examples: int, tokenizer,
               harness: str = "infinitebench_eval"):
    """Load InfiniteBench EnQA with middle-truncation.

    Two harness templates are supported (matching the two paper-side
    EnQA harnesses):

    - ``infinitebench_eval`` (default): the canonical paper harness in
      ``baselines/infinitebench_eval.py``. Template:
        "Read the book and answer the question. Be concise.\\n\\n
         Book:\\n{context}\\n\\nQuestion: {input}\\n\\nAnswer:"
      max_gen = 32 (matches Cells A/B/C/D/E in the paper).
    - ``kivi_hf_v2``: the FU_W5 KIVI runner template (slightly different).
    """
    raw_path = Path("experiments/infinitebench_raw/longbook_qa_eng.jsonl")
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    examples = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    if harness == "infinitebench_eval":
        template = (
            "Read the book and answer the question. Be concise.\n\n"
            "Book:\n{context}\n\nQuestion: {input}\n\nAnswer:"
        )
    elif harness == "kivi_hf_v2":
        template = (
            "Read the book below and answer a question.\n\n{context}\n\n"
            "Question: {input}\n\nBe very concise.\nAnswer:"
        )
    else:
        raise ValueError(harness)
    selected = []
    for ex in examples:
        prompt = template.format(context=ex["context"], input=ex["input"])
        ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(ids) > context_length:
            half = context_length // 2
            head = tokenizer.decode(ids[:half], skip_special_tokens=True)
            tail = tokenizer.decode(ids[-half:], skip_special_tokens=True)
            prompt = head + tail
        # Re-pack to the IB-style dict
        gold = ex["answer"]
        selected.append({"input": prompt, "outputs": gold})
        if len(selected) >= n_examples:
            break
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.environ.get("HALO_DEFAULT_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    ap.add_argument("--task", required=True,
                    help="ruler_<subtask> or infinitebench_en_qa")
    ap.add_argument("--context-length", type=int, default=32768)
    ap.add_argument("--n-examples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Load tokenizer (used by both task paths) --------------------------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path)

    # Load task examples ------------------------------------------------
    if args.task.startswith("ruler_"):
        subtask = args.task[len("ruler_"):]
        examples = _load_ruler(subtask, args.context_length)
        task_kind = "ruler"
    elif args.task == "infinitebench_en_qa":
        examples = _load_enqa(args.context_length, args.n_examples, tok,
                              harness="infinitebench_eval")
        subtask = "en_qa"
        task_kind = "infinitebench"
    else:
        raise ValueError(f"Unknown task {args.task}")
    print(f"[vllm] loaded {len(examples)} examples for {args.task}", flush=True)

    # vLLM engine -------------------------------------------------------
    from vllm import LLM, SamplingParams
    print(f"[vllm] loading {args.model_path} with PagedAttention", flush=True)

    # vLLM's max_model_len must >= context + max_new_tokens
    max_model_len = args.context_length + args.max_new_tokens + 64

    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=False,
        trust_remote_code=False,
        seed=args.seed,
    )
    sp = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    selected = examples[: args.n_examples]
    preds = []
    for i, ex in enumerate(selected):
        if task_kind == "ruler":
            prompt = ex["input"]
            if ex.get("answer_prefix") and not prompt.rstrip().endswith(ex["answer_prefix"].rstrip()):
                prompt = prompt + ex["answer_prefix"]
            gold = ex["outputs"]
        else:  # infinitebench
            prompt = ex["input"]
            gold = ex["outputs"]

        t0 = time.time()
        outputs = llm.generate([prompt], sp, use_tqdm=False)
        wall = time.time() - t0
        pred = outputs[0].outputs[0].text
        if task_kind == "ruler":
            score = _ruler_score(pred, gold)
        else:  # infinitebench EnQA: use the same qa_f1_score as the other
               # InfiniteBench harnesses in baselines/infinitebench_eval.py.
            from baselines.longbench_eval import qa_f1_score
            if isinstance(gold, str):
                score = qa_f1_score(pred, gold)
            else:
                score = max((qa_f1_score(pred, g) for g in gold), default=0.0)
        preds.append({
            "index": i,
            "pred": pred[:300],
            "gold": gold if isinstance(gold, str) else list(gold),
            "score": score,
            "wall_s": wall,
        })
        print(f"  [{i+1}/{len(selected)}] {args.task} score={score:.0f} wall={wall:.1f}s", flush=True)

    scores = [p["score"] for p in preds]
    walls = [p["wall_s"] for p in preds]
    # Bootstrap CI95 over 10000 resamples, matching the IB / RULER harnesses.
    ci_low = ci_high = None
    if len(scores) >= 5:
        import random as _rand
        _rand.seed(args.seed)
        boot_means = []
        for _ in range(10000):
            sample = [scores[_rand.randint(0, len(scores) - 1)]
                      for _ in range(len(scores))]
            boot_means.append(sum(sample) / len(sample))
        boot_means.sort()
        ci_low = boot_means[int(0.025 * len(boot_means))]
        ci_high = boot_means[int(0.975 * len(boot_means))]
    # Peak GPU: vLLM owns the allocator, but torch.cuda.max_memory_allocated
    # gives the framework-level reading for cross-harness comparison. The
    # vLLM-internal KV-cache profile (engine.cache_profile) reports the
    # paged-block total, which differs by ~1 GiB (block-size rounding).
    import torch as _torch
    try:
        peak_torch_gib = _torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        peak_torch_gib = None
    summary = {
        "method": "vllm_pagedattention",
        "task": args.task,
        "context_length": args.context_length,
        "n": len(scores),
        "mean_score_pct": 100.0 * sum(scores) / len(scores) if scores else None,
        "ci_low_pct": 100.0 * ci_low if ci_low is not None else None,
        "ci_high_pct": 100.0 * ci_high if ci_high is not None else None,
        "ci_bootstrap_iters": 10000 if ci_low is not None else 0,
        "mean_wall_s": sum(walls) / len(walls) if walls else None,
        "peak_torch_gib": peak_torch_gib,
        "peak_telemetry_note": (
            "peak_torch_gib is torch.cuda.max_memory_allocated(); vLLM's "
            "internal KV-cache profile reports paged-block totals which "
            "differ by ~1 GiB. Use peak_torch_gib for cross-harness "
            "comparison against Path D / Full / KIVI which all report the "
            "same primitive."
        ),
        "model_path": args.model_path,
        "vllm_gpu_memory_utilization": args.gpu_memory_utilization,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "preds.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    print(f"\n[vllm] {args.task}: mean score = {summary['mean_score_pct']:.2f}% over n={summary['n']}", flush=True)


if __name__ == "__main__":
    main()
