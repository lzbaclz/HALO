"""RULER (NVIDIA, 2024) evaluation harness — minimal port.

The full RULER suite is a synthetic benchmark generator + per-task scorers
hosted at https://github.com/NVIDIA/RULER. This module assumes the data has
already been generated (per the user's repro instructions) and lives at
``${HALO_RULER_DIR:-~/RULER/data}/<task>/<context_length>.jsonl`` (the layout
produced by ``RULER/scripts/data/synthetic/niah/prepare_data.sh``).

Each jsonl row is expected to expose at least:
  - ``input``: the full prompt
  - ``outputs``: list of acceptable answers (substring match)
  - ``length``: nominal context length

Scoring follows RULER's official ``substring_match.py``: the prediction
"contains" the gold needle (case-insensitive substring match). The score is
the mean accuracy over the slice of examples for the given context length.
"""
from __future__ import annotations

import json
import os
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


def _data_root() -> Path:
    env = os.environ.get("HALO_RULER_DIR")
    if env:
        return Path(env)
    return Path.home() / "RULER" / "data"


def _load(task: str, length: int, limit: Optional[int]) -> list[dict]:
    root = _data_root()
    candidates = [
        root / task / f"{length}.jsonl",
        root / task / f"{length}" / "validation.jsonl",
        root / f"{length}" / "data" / f"{task}.jsonl",
        # NVIDIA/RULER prepare.py default layout:
        root / f"{length}" / task / "validation.jsonl",
    ]
    for c in candidates:
        if c.exists():
            with c.open("r", encoding="utf-8") as f:
                rows = [json.loads(l) for l in f]
            if limit:
                rows = rows[:limit]
            return rows
    raise FileNotFoundError(
        f"RULER data not found for task={task} length={length}. Searched: "
        + ", ".join(str(c) for c in candidates) + ". "
        f"Generate it via `cd ~/RULER && bash scripts/data/synthetic/niah/prepare_data.sh` "
        f"and set HALO_RULER_DIR if needed."
    )


def _substring_match(pred: str, golds: list[str]) -> float:
    p = pred.lower()
    return float(any((g or "").lower() in p for g in golds))


def evaluate_task(
    model,
    tokenizer,
    *,
    task: str,
    context_length: int = 8192,
    limit: Optional[int] = None,
    output_dir: Optional[Path] = None,
    press_ctx: Optional[Callable[[Any], Any]] = None,
) -> float:
    import torch

    try:
        rows = _load(task, context_length, limit)
    except FileNotFoundError as e:
        # RULER's prepare.py silently fails to write some (task, length) combos
        # (e.g. cwe @ 8K, vt @ 8K). Return NaN here so the surrounding sweep
        # can keep going; downstream manifest consumers treat NaN as a
        # missing cell.
        import sys
        print(f"[ruler_eval] skipping {task}@{context_length} (no data): {e}",
              file=sys.stderr)
        return float("nan")
    device = next(model.parameters()).device

    iterator: Iterable[dict] = rows
    try:
        from tqdm import tqdm  # type: ignore[import-not-found]
        iterator = tqdm(rows, desc=f"ruler/{task}@{context_length}")
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
        prompt = ex.get("input", ex.get("prompt", ""))
        # Middle-truncate just in case the row was generated for a longer
        # context window than the model can handle.
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
                    **inputs, max_new_tokens=64, do_sample=False,
                    pad_token_id=(tokenizer.eos_token_id or 0),
                )
        pred = tokenizer.decode(out[0, ctx_len:], skip_special_tokens=True)

        golds = ex.get("outputs") or ex.get("answers") or [ex.get("answer", "")]
        if isinstance(golds, str):
            golds = [golds]
        score = _substring_match(pred, golds)
        scores.append(score)

        if out_jsonl is not None:
            with out_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"pred": pred, "outputs": golds,
                                    "length": ex.get("length"),
                                    "score": score}, ensure_ascii=False) + "\n")

    if not scores:
        return float("nan")
    return float(100.0 * sum(scores) / len(scores))
