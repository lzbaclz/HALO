"""∞-Bench (InfiniteBench) evaluation harness.

Implements a minimal port of OpenBMB/InfiniteBench's ``compute_scores.py`` for
the two English subtasks the paper uses (``en_qa``, ``en_mc``). Data is loaded
from the local jsonl mirror at ``experiments/infinitebench_raw/<task>.jsonl``;
download the dataset from HuggingFace ``xinrongzhang2022/InfiniteBench`` (or the
upstream GitHub release) and unzip there before running.

The evaluator concatenates the prompt to a target ``context_length`` (the paper
fabricates 1M-token prompts by repeating the natural context, with the
limitation noted in §5.4). For ``en_qa`` we score with the same QA-F1 metric as
LongBench (string-tokenized F1 against the gold answers). For ``en_mc`` we
parse the predicted letter (A/B/C/D).
"""
from __future__ import annotations

import json
import os
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from baselines.longbench_eval import qa_f1_score


# ---------------------------------------------------------------------------
# Prompt templates — abridged from the upstream
# https://github.com/OpenBMB/InfiniteBench/blob/main/src/prompt.py
# ---------------------------------------------------------------------------


_PROMPTS: dict[str, str] = {
    "en_qa": (
        "Read the book and answer the question. Be concise.\n\n"
        "Book:\n{context}\n\nQuestion: {input}\n\nAnswer:"
    ),
    "en_mc": (
        "Read the book and answer the multiple-choice question. Just output "
        "the letter (A, B, C, or D).\n\n"
        "Book:\n{context}\n\nQuestion: {input}\n\nOptions:\n{options}\n\n"
        "Answer:"
    ),
}

_MAX_GEN: dict[str, int] = {"en_qa": 32, "en_mc": 8}


def _data_root() -> Path:
    env = os.environ.get("HALO_INFINITEBENCH_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent.parent
    return here / "experiments" / "infinitebench_raw"


# Map our short "subtask" names to the actual filenames on the HF dataset.
# The xinrongzhang2022/InfiniteBench HF repo ships long-form names like
# `longbook_qa_eng.jsonl`; the upstream InfiniteBench eval scripts and our
# HALO baselines refer to the same subtasks as `en_qa`, `en_mc`, etc. We
# accept either form (HF name first, short name as fallback for symlinks).
_FILE_ALIASES: dict[str, list[str]] = {
    "en_qa": ["longbook_qa_eng.jsonl", "en_qa.jsonl"],
    "en_mc": ["longbook_choice_eng.jsonl", "en_mc.jsonl"],
    # Pass-through for tasks that are already in canonical form.
}


def load_examples(task: str, *, limit: Optional[int] = None) -> list[dict]:
    root = _data_root()
    candidates = _FILE_ALIASES.get(task, [f"{task}.jsonl"])
    p = None
    for fname in candidates:
        cand = root / fname
        if cand.exists():
            p = cand
            break
    if p is None:
        raise FileNotFoundError(
            f"InfiniteBench data not found for task '{task}' at {root}; "
            f"tried filenames: {candidates}. Download via "
            f"`hf download xinrongzhang2022/InfiniteBench --repo-type dataset "
            f"--local-dir experiments/infinitebench_raw/` "
            f"(or the legacy `huggingface-cli download ...`)."
        )
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def _fabricate_long_context(orig: str, tokenizer, target_len: int) -> str:
    """Loop the natural context until token count ≥ target_len.

    Documented in §5.4 of the paper as a "limitation": ∞-Bench's natural
    contexts top out around 150K tokens, so 1M-token prompts in our 1M study
    are produced by tiling 6–8 copies. This function tokenizes ``orig`` once
    and concatenates copies until the encoded length reaches ``target_len``.
    """
    if target_len <= 0:
        return orig
    ids = tokenizer(orig, truncation=False, return_tensors="pt").input_ids[0]
    if len(ids) >= target_len:
        return orig
    n_copies = (target_len // len(ids)) + 1
    fab = (orig + "\n\n") * n_copies
    return fab


def _parse_mc_letter(s: str) -> str:
    s = s.strip().upper()
    m = re.search(r"[ABCD]", s)
    return m.group(0) if m else ""


def evaluate_task(
    model,
    tokenizer,
    *,
    task: str,
    context_length: int = 1_000_000,
    limit: Optional[int] = None,
    output_dir: Optional[Path] = None,
    press_ctx: Optional[Callable[[Any], Any]] = None,
) -> float:
    import torch

    if task not in _PROMPTS:
        raise ValueError(f"InfiniteBench task '{task}' not supported "
                         f"(expected one of {list(_PROMPTS)}).")

    examples = load_examples(task, limit=limit)
    template = _PROMPTS[task]
    max_gen = _MAX_GEN[task]
    metric = qa_f1_score if task == "en_qa" else None  # mc handled separately
    device = next(model.parameters()).device

    iterator: Iterable[dict] = examples
    try:
        from tqdm import tqdm  # type: ignore[import-not-found]
        iterator = tqdm(examples, desc=f"infinitebench/{task}")
    except ImportError:
        pass

    out_jsonl: Optional[Path] = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_jsonl = output_dir / "preds.jsonl"
        if out_jsonl.exists():
            out_jsonl.unlink()

    scores: list[float] = []
    for ex in iterator:
        ctx = ex.get("context") or ex.get("text") or ""
        ctx = _fabricate_long_context(ctx, tokenizer, context_length)

        if task == "en_mc":
            opts = ex.get("options") or ex.get("choices") or []
            opt_str = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(opts))
            prompt = template.format(context=ctx, input=ex.get("input", ex.get("question", "")),
                                     options=opt_str)
        else:
            prompt = template.format(context=ctx, input=ex.get("input", ex.get("question", "")))

        # Truncate to context_length tokens (middle-truncation).
        ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(ids) > context_length:
            half = context_length // 2
            head = tokenizer.decode(ids[:half], skip_special_tokens=True)
            tail = tokenizer.decode(ids[-half:], skip_special_tokens=True)
            prompt = head + tail

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        ctx_len = inputs.input_ids.shape[-1]

        with torch.no_grad():
            cm = press_ctx(model) if press_ctx is not None else nullcontext()
            with cm:
                out = model.generate(
                    **inputs, max_new_tokens=max_gen, do_sample=False,
                    pad_token_id=(tokenizer.eos_token_id or 0),
                )
        pred = tokenizer.decode(out[0, ctx_len:], skip_special_tokens=True)

        if task == "en_mc":
            gt_raw = ex.get("answer") or ex.get("ground_truth") or ""
            # InfiniteBench's longbook_choice_eng stores `answer` as a list of
            # one string; the gold here is the *option text*, not a letter, so
            # we must look it up against `options` (also a list).
            if isinstance(gt_raw, list):
                gt_raw = gt_raw[0] if gt_raw else ""
            opts = ex.get("options") or ex.get("choices") or []
            gt_letter = ""
            for i, o in enumerate(opts):
                if o == gt_raw:
                    gt_letter = chr(65 + i)
                    break
            if not gt_letter:
                gt_letter = _parse_mc_letter(str(gt_raw))
            score = 1.0 if _parse_mc_letter(pred) == gt_letter else 0.0
        else:
            answers = ex.get("answer") or ex.get("ground_truth") or ex.get("answers") or []
            if isinstance(answers, str):
                answers = [answers]
            score = max((qa_f1_score(pred, a) for a in answers), default=0.0)
        scores.append(score)

        if out_jsonl is not None:
            with out_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"pred": pred, "score": score}, ensure_ascii=False) + "\n")

    if not scores:
        return float("nan")
    return float(100.0 * sum(scores) / len(scores))
