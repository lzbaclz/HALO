"""Real LongBench v1 evaluation harness.

Self-contained: reads ``experiments/longbench_raw/data/<task>.jsonl``, builds the
official prompt template (mirrors ``THUDM/LongBench/LongBench/pred.py``), runs
the model with the supplied compression / wrapping callable, and scores
predictions with the metric functions from ``THUDM/LongBench/LongBench/metrics.py``.

This module is intentionally NOT registered as an importable package called
``longbench`` (which is what the old stub in ``baselines/runner.py`` looked for).
Instead, it is wired in directly from ``baselines.runner._longbench``.
"""
from __future__ import annotations

import json
import os
import re
import string
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


# ---------------------------------------------------------------------------
# Per-task config — copied verbatim from the LongBench v1 official repo
# (paper: github.com/THUDM/LongBench, files: config/dataset2{prompt,maxlen}.json)
# ---------------------------------------------------------------------------

DATASET2PROMPT: dict[str, str] = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, "
        "and a question. Answer the question asconcisely as you can, using a "
        "single phrase if possible. Do not provide any explanation.\n\n"
        "Story: {context}\n\nNow, answer the question based on the story "
        "asconcisely as you can, using a single phrase if possible. Do not "
        "provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question "
        "as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the "
        "article, write \"unanswerable\". If the question is a yes/no question, "
        "answer \"yes\", \"no\", or \"unanswerable\". Do not provide any "
        "explanation.\n\nArticle: {context}\n\n Answer the question based on "
        "the above article as concisely as you can, using a single phrase or "
        "sentence if possible. If the question cannot be answered based on the "
        "information in the article, write \"unanswerable\". If the question is "
        "a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not "
        "provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, "
        "answer the following question based on the above text, only give me "
        "the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nThe following are given "
        "passages.\n{context}\n\nAnswer the question based on the given "
        "passages. Only give me the answer and do not output any other words."
        "\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nThe following are given "
        "passages.\n{context}\n\nAnswer the question based on the given "
        "passages. Only give me the answer and do not output any other words."
        "\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nThe following are given "
        "passages.\n{context}\n\nAnswer the question based on the given "
        "passages. Only give me the answer and do not output any other words."
        "\n\nQuestion: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page "
        "summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page "
        "summary of the report.\n\nSummary:"
    ),
    "trec": (
        "Please determine the type of the question below. Here are some "
        "examples of questions.\n\n{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the "
        "answer and do not output any other words. The following are some "
        "examples.\n\n{context}\n\n{input}"
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please "
        "determine which paragraph the abstract is from.\n\n{context}\n\nThe "
        "following is an abstract.\n\n{input}\n\nPlease enter the number of "
        "the paragraph that the abstract is from. The answer format must be "
        "like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: "
    ),
    "repobench-p": (
        "Please complete the code given below. \n{context}{input}Next line of code:\n"
    ),
    "lcc": (
        "Please complete the code given below. \n{context}Next line of code:\n"
    ),
}

DATASET2MAXLEN: dict[str, int] = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64,
    "hotpotqa": 32, "2wikimqa": 32, "musique": 32,
    "gov_report": 512, "trec": 64, "triviaqa": 32,
    "passage_retrieval_en": 32, "lcc": 64, "repobench-p": 64,
}

NO_CHAT_WRAPPING: set[str] = {"trec", "triviaqa", "samsum", "lcc", "repobench-p"}


# ---------------------------------------------------------------------------
# Metric functions — verbatim from THUDM/LongBench/LongBench/metrics.py
# (re-implemented here so we don't depend on the external package).
# ---------------------------------------------------------------------------


def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def _f1(prediction: list[str], ground_truth: list[str]) -> float:
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(pred: str, gt: str, **_: Any) -> float:
    return _f1(_normalize_answer(pred).split(), _normalize_answer(gt).split())


def rouge_score(pred: str, gt: str, **_: Any) -> float:
    try:
        from rouge import Rouge  # type: ignore[import-not-found]
    except ImportError:
        return 0.0
    r = Rouge()
    try:
        return r.get_scores([pred], [gt], avg=True)["rouge-l"]["f"]
    except Exception:
        return 0.0


def classification_score(pred: str, gt: str, *, all_classes: Optional[list[str]] = None,
                         **_: Any) -> float:
    if not all_classes:
        return 1.0 if gt.strip() in pred else 0.0
    matches = [c for c in all_classes if c in pred]
    matches = [m for m in matches if not (m in gt and m != gt)]
    if gt in matches:
        return 1.0 / len(matches)
    return 0.0


def retrieval_score(pred: str, gt: str, **_: Any) -> float:
    m = re.findall(r"Paragraph (\d+)", gt)
    if not m:
        return 0.0
    gt_id = m[0]
    nums = re.findall(r"\d+", pred)
    if not nums:
        return 0.0
    return sum(1 for n in nums if str(n) == str(gt_id)) / len(nums)


def code_sim_score(pred: str, gt: str, **_: Any) -> float:
    try:
        from fuzzywuzzy import fuzz  # type: ignore[import-not-found]
    except ImportError:
        return 0.0
    line = ""
    for ln in pred.lstrip("\n").split("\n"):
        if "`" not in ln and "#" not in ln and "//" not in ln:
            line = ln
            break
    return fuzz.ratio(line, gt) / 100.0


DATASET2METRIC: dict[str, Callable[..., float]] = {
    "narrativeqa": qa_f1_score, "qasper": qa_f1_score, "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score, "2wikimqa": qa_f1_score, "musique": qa_f1_score,
    "gov_report": rouge_score, "trec": classification_score,
    "triviaqa": qa_f1_score, "passage_retrieval_en": retrieval_score,
    "lcc": code_sim_score, "repobench-p": code_sim_score,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _data_root() -> Path:
    env = os.environ.get("HALO_LONGBENCH_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent.parent
    return here / "experiments" / "longbench_raw" / "data"


def load_task_examples(task: str, *, limit: Optional[int] = None) -> list[dict]:
    path = _data_root() / f"{task}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"LongBench data not found: {path}. "
                                f"Run `unzip data.zip -d experiments/longbench_raw/`.")
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


# ---------------------------------------------------------------------------
# Prompt building & truncation (matches the official pred.py middle-truncation)
# ---------------------------------------------------------------------------


def truncate_middle(prompt: str, tokenizer, max_length: int) -> str:
    ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(ids) <= max_length:
        return prompt
    half = max_length // 2
    head = tokenizer.decode(ids[:half], skip_special_tokens=True)
    tail = tokenizer.decode(ids[-half:], skip_special_tokens=True)
    return head + tail


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def evaluate_task(
    model,
    tokenizer,
    *,
    task: str,
    limit: Optional[int] = None,
    output_dir: Optional[Path] = None,
    max_input_length: int = 7500,
    press_ctx: Optional[Callable[[Any], Any]] = None,
    progress: bool = True,
) -> float:
    """Run LongBench evaluation for one task and return the mean score (×100).

    ``press_ctx`` is an optional callable that takes ``model`` and returns a
    context manager; we wrap it around the model forward pass. For HALO we pass
    ``None`` since :func:`halo.policy.wrap_with_halo` already patches generate.
    """
    import torch

    if task not in DATASET2PROMPT:
        raise ValueError(f"LongBench task '{task}' not supported by this harness.")

    examples = load_task_examples(task, limit=limit)
    template = DATASET2PROMPT[task]
    max_gen = DATASET2MAXLEN[task]

    iterator: Iterable[dict] = examples
    if progress:
        try:
            from tqdm import tqdm  # type: ignore[import-not-found]
            iterator = tqdm(examples, desc=f"longbench/{task}")
        except ImportError:
            pass

    metric = DATASET2METRIC[task]
    scores: list[float] = []
    out_jsonl: Optional[Path] = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_jsonl = output_dir / "preds.jsonl"
        if out_jsonl.exists():
            out_jsonl.unlink()

    device = next(model.parameters()).device

    for ex in iterator:
        prompt = template.format(context=ex["context"], input=ex["input"])
        prompt = truncate_middle(prompt, tokenizer, max_input_length)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(device)
        ctx_len = inputs.input_ids.shape[-1]

        # Optional stochastic decoding (T2.9 in the 2026-05-12 revision).
        # ``$HALO_DO_SAMPLE=1`` toggles do_sample=True so the same harness
        # can produce real seed variance for error-bar tables. Greedy is
        # the default to preserve back-compat with the existing manifests.
        _do_sample = os.environ.get("HALO_DO_SAMPLE", "0") == "1"
        _temperature = float(os.environ.get("HALO_TEMPERATURE", "1.0"))
        _top_p = float(os.environ.get("HALO_TOP_P", "0.9"))
        with torch.no_grad():
            cm = press_ctx(model) if press_ctx is not None else nullcontext()
            with cm:
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_gen,
                    do_sample=_do_sample,
                    num_beams=1,
                    temperature=_temperature if _do_sample else 1.0,
                    top_p=_top_p if _do_sample else 1.0,
                    pad_token_id=(tokenizer.eos_token_id or 0),
                )
        gen = out[0, ctx_len:]
        pred = tokenizer.decode(gen, skip_special_tokens=True)

        if task in ("trec", "triviaqa"):
            pred = pred.lstrip("\n").split("\n")[0]

        best = 0.0
        for gt in ex["answers"]:
            best = max(best, metric(pred, gt, all_classes=ex.get("all_classes")))
        scores.append(best)

        if out_jsonl is not None:
            with out_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "pred": pred,
                    "answers": ex["answers"],
                    "all_classes": ex.get("all_classes"),
                    "length": ex.get("length"),
                    "score": best,
                }, ensure_ascii=False) + "\n")

    if not scores:
        return float("nan")
    return float(100.0 * sum(scores) / len(scores))
