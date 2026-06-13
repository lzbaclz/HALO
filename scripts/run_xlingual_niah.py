"""Run Full + Path D on cross-lingual NIAH (synthesized in build_cross_lingual_niah.py).

For each language sub-config (zh, en, mixed_zh_en, mixed_en_zh):
  - load data.jsonl
  - prompt the model with the haystack + question
  - extract the predicted number, exact-match against gold
  - report accuracy

Usage:
    python scripts/run_xlingual_niah.py --method full --tag zh --output experiments/cell_xlingual_niah/zh_full
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

PROMPT_TPL = """{context}

{question} Answer with just the number, no other text.

The special magic number is N"""


def parse_number(s: str) -> str:
    # The prompt ends with "N", model continues with the digits.
    m = re.match(r"\s*(\d{4,7})", s)
    if m:
        return m.group(1)
    # Fallback: any 5-7 digit number in the output.
    m = re.search(r"\b(\d{5,7})\b", s)
    return m.group(1) if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["full", "path_d", "kivi"])
    ap.add_argument("--tag", required=True, choices=["zh", "en", "mixed_zh_en", "mixed_en_zh"])
    ap.add_argument("--config", default="configs/models/qwen2-5-7b.yaml")
    ap.add_argument("--memory-ratio", type=int, default=4)
    ap.add_argument("--context-length", type=int, default=12000)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    data_path = _REPO_ROOT / f"experiments/cell_xlingual_niah/{args.tag}/data.jsonl"
    examples = [json.loads(L) for L in open(data_path)]
    print(f"[xlingual] {args.method} on {args.tag}: {len(examples)} examples")

    import torch
    import yaml
    cfg = yaml.safe_load(open(args.config))
    cfg_for_load = dict(cfg)
    if args.method == "path_d":
        cfg_for_load["attn_implementation"] = "eager"
    from baselines.runner import _load_model_and_tokenizer
    try:
        model, tok = _load_model_and_tokenizer(cfg_for_load, method=args.method,
                                                memory_ratio=args.memory_ratio)
    except TypeError:
        model, tok = _load_model_and_tokenizer(cfg_for_load)
    model.eval()

    if args.method == "kivi":
        from baselines.kivi_cache import wrap_with_kivi
        wrap_with_kivi(model, memory_ratio=args.memory_ratio)
    elif args.method == "path_d":
        from halo import HALOConfig, wrap_with_halo, install_preforward_peel
        cfg_obj = HALOConfig(chunked=True, chunk_size=512, recent_window=64,
                              hot_ratio=1.0 / max(1.0, float(args.memory_ratio)),
                              use_triton=True)
        wrap_with_halo(model, cfg_obj)
        install_preforward_peel(model, prefill_chunk_tokens=4096,
                                 activation_threshold=8192)

    preds = []
    from tqdm import tqdm
    correct = 0
    for ex in tqdm(examples, desc=f"xling/{args.tag}"):
        prompt = PROMPT_TPL.format(context=ex["input"], question=ex["question"])
        ids = tok(prompt, return_tensors="pt", truncation=True,
                   max_length=args.context_length).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**ids, max_new_tokens=12, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
        pred_text = tok.decode(out_ids[0, ids.input_ids.shape[1]:],
                                skip_special_tokens=True)
        pred_num = parse_number(pred_text)
        gold = ex["answer"].lstrip("N")
        score = 1.0 if pred_num == gold else 0.0
        preds.append({"index": ex["index"], "pred": pred_text[:60],
                       "pred_num": pred_num, "gold": gold, "score": score})
        correct += score

    with open(out / "preds.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    acc = correct / len(examples)
    summary = {"method": args.method, "tag": args.tag, "n": len(examples),
                "accuracy": acc, "context_length": args.context_length}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[xlingual] {args.method} {args.tag}: acc={acc*100:.2f}% ({int(correct)}/{len(examples)})")


if __name__ == "__main__":
    main()
