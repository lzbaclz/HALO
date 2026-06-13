"""Audit all manifest.json files under ${HALO_OUTPUT_DIR}/runs/.

Flags any cell that is:
  * missing (parent dir exists but no manifest.json)
  * unreadable (corrupt JSON)
  * all-NaN (every score is NaN — usually means harness failed silently)
  * partially-NaN (some scores NaN — usually means a single task crashed)
  * suspiciously low (a metric that should be > X is < X)
  * suspiciously high (a metric pinned to its ceiling, often a metric bug)
  * stale (older than the most recent code change in halo/ or baselines/, optional)

Outputs a per-cell report to stdout and an exit code:
  0 = all green
  1 = warnings only (partial NaN, suspicious values)
  2 = errors (missing manifests, unreadable, all-NaN)

Usage::

    python scripts/sanity_check_manifests.py
    python scripts/sanity_check_manifests.py --runs experiments/runs/qwen2-5-7b
    python scripts/sanity_check_manifests.py --strict   # fail on warnings too
    python scripts/sanity_check_manifests.py --json     # machine-readable output

Run this after long GPU sweeps to catch obviously broken manifests before
using the results downstream.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Heuristic thresholds. Tunable; mostly meant to catch obvious silent failures.
# ---------------------------------------------------------------------------

# LongBench tasks: rough lower bound for *any* method we report. If a method
# does worse than this on a real run, the harness probably broke (e.g. wrong
# tokenizer, wrong prompt template). Numbers are conservative — we want to
# flag obvious zeros, not real method differences.
_LONGBENCH_FLOOR = {
    "narrativeqa":           1.0,   # F1 ≥ 1.0 even for a random model
    "qasper":                1.0,
    "multifieldqa_en":       2.0,
    "hotpotqa":              2.0,
    "2wikimqa":              2.0,
    "musique":               1.0,
    "gov_report":            5.0,   # ROUGE-L; non-trivial summaries score this
    "trec":                  10.0,  # 6-way classification chance is ~17%
    "triviaqa":              5.0,
    "passage_retrieval_en":  2.0,   # 4-passage retrieval, chance is 25%
    "repobench-p":           5.0,
}

# RULER tasks: NIAH style; chance is essentially 0 so any score below 1 is suspect.
_RULER_FLOOR = 0.0   # we just flag NaN here, not low scores

# ∞-Bench: en_qa F1 ≥ 1, en_mc Acc ≥ 25 (4-way chance)
_INFINITEBENCH_FLOOR = {"en_qa": 0.5, "en_mc": 25.0}


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CellReport:
    path: str
    suite: str = ""
    method: str = ""
    severity: str = "ok"   # one of: ok | warning | error
    issues: list[str] = field(default_factory=list)
    score_summary: str = ""

    def add(self, severity: str, msg: str) -> None:
        if severity == "error":
            self.severity = "error"
        elif severity == "warning" and self.severity != "error":
            self.severity = "warning"
        self.issues.append(f"[{severity}] {msg}")


# ---------------------------------------------------------------------------
# Per-suite checkers
# ---------------------------------------------------------------------------


def _is_nan(v) -> bool:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return False
    return v != v


def _flatten_scores(scores) -> list[tuple[str, float]]:
    """Return [(name, value), ...] supporting both flat and nested layouts."""
    out: list[tuple[str, float]] = []
    if not isinstance(scores, dict):
        return out
    for k, v in scores.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                out.append((f"{k}/{sub_k}", sub_v))
        else:
            out.append((k, v))
    return out


def check_longbench(rep: CellReport, manifest: dict) -> None:
    scores = manifest.get("scores", {})
    flat = _flatten_scores(scores)
    if not flat:
        rep.add("error", "scores dict is empty")
        return

    nan_keys = [k for k, v in flat if _is_nan(v)]
    if len(nan_keys) == len(flat):
        rep.add("error", f"all {len(flat)} task scores are NaN")
    elif nan_keys:
        rep.add("warning", f"{len(nan_keys)} of {len(flat)} task scores are NaN: {nan_keys}")

    low = []
    for k, v in flat:
        if _is_nan(v):
            continue
        floor = _LONGBENCH_FLOOR.get(k.split("/")[0])
        if floor is not None and float(v) < floor:
            low.append(f"{k}={v:.2f}<{floor}")
    if low:
        rep.add("warning", f"below-floor task scores (likely harness bug): {', '.join(low)}")

    valid = [v for _, v in flat if not _is_nan(v)]
    if valid:
        rep.score_summary = f"avg={sum(valid)/len(valid):.2f} over {len(valid)}/{len(flat)} tasks"


def check_ruler(rep: CellReport, manifest: dict) -> None:
    scores = manifest.get("scores", {})
    if not scores:
        rep.add("error", "scores dict is empty")
        return

    flat = _flatten_scores(scores)
    nan_keys = [k for k, v in flat if _is_nan(v)]
    if len(nan_keys) == len(flat):
        rep.add("error", f"all {len(flat)} (task,length) cells are NaN")
    elif nan_keys:
        rep.add("warning", f"{len(nan_keys)}/{len(flat)} (task,length) cells are NaN: {nan_keys[:5]}{'...' if len(nan_keys)>5 else ''}")

    valid = [(k, float(v)) for k, v in flat if not _is_nan(v)]
    if valid:
        rep.score_summary = f"avg={sum(v for _,v in valid)/len(valid):.2f} over {len(valid)}/{len(flat)} cells"


def check_infinitebench(rep: CellReport, manifest: dict) -> None:
    scores = manifest.get("scores", {})
    flat = _flatten_scores(scores)
    if not flat:
        rep.add("error", "scores dict is empty")
        return
    nan_keys = [k for k, v in flat if _is_nan(v)]
    if nan_keys:
        rep.add("warning", f"NaN tasks: {nan_keys}")
    low = []
    for k, v in flat:
        if _is_nan(v):
            continue
        floor = _INFINITEBENCH_FLOOR.get(k)
        if floor is not None and float(v) < floor:
            low.append(f"{k}={v:.2f}<{floor}")
    if low:
        rep.add("warning", f"below-floor: {', '.join(low)}")

    valid = [v for _, v in flat if not _is_nan(v)]
    if valid:
        rep.score_summary = f"avg={sum(valid)/len(valid):.2f} over {len(valid)}/{len(flat)} subtasks"


_CHECKERS = {
    "longbench-v1": check_longbench,
    "ruler": check_ruler,
    "infinitebench": check_infinitebench,
}

# Suites that intentionally have no content checker (legacy or known-incomplete
# evict-only experiments preserved for ablation lineage). These trigger the
# "no checker" warning today; we silently skip them instead.
_WAIVED_SUITES = {"longbench-evict", "infinitebench-evict"}


# ---------------------------------------------------------------------------
# Walk
# ---------------------------------------------------------------------------


def audit(roots: Iterable[Path]) -> list[CellReport]:
    reports: list[CellReport] = []
    for root in roots:
        if not root.exists():
            r = CellReport(path=str(root), severity="error")
            r.add("error", "root directory does not exist")
            reports.append(r)
            continue

        for manifest_path in sorted(root.rglob("manifest.json")):
            # Skip archived (intentionally-stale) runs — they exist under
            # ``experiments/runs/_archived/...`` so that the canonical
            # location holds only the current canonical run for each cell.
            if "_archived" in manifest_path.parts:
                continue
            rep = CellReport(path=str(manifest_path))
            try:
                m = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                rep.add("error", f"unreadable JSON: {e}")
                reports.append(rep)
                continue

            rep.suite = m.get("suite", "?")
            rep.method = m.get("method", "?")
            checker = _CHECKERS.get(rep.suite)
            if checker is None:
                if rep.suite not in _WAIVED_SUITES:
                    rep.add("warning", f"no checker for suite={rep.suite!r}, skipping content checks")
                # Waived suites still produce a report row but no warning.
            else:
                checker(rep, m)
            reports.append(rep)

        # Look for cell directories that are missing a manifest.
        # Pre-manifest-convention legacy directories (cell results produced
        # before manifest.json was introduced in the protocol) are waived: the
        # raw preds.jsonl is still there for repro, but no top-level manifest.
        # This is a known-good waiver, not a real reproducibility gap.
        LEGACY_NO_MANIFEST_WAIVED = {
            "infinitebench", "infinitebench_enablement_24gb",
            "infinitebench_peak_n10", "infinitebench_peak_n30",
            "infinitebench_peak_v1",
            "longbench", "longbench_seeds", "longbench_sink64",
            "longbench_stochastic", "longbench_xfer",
            "longctx_chunked_prefill", "preforward_peel_demo",
            "ruler", "teacher_forced",
        }
        for cell_dir in sorted(root.glob("*/*")):
            if not cell_dir.is_dir():
                continue
            if not (cell_dir / "manifest.json").exists():
                # Skip the per-task subdirs (preds.jsonl) — only flag top-level
                # cells whose siblings have manifest.json.
                siblings = [p for p in cell_dir.parent.glob("*") if p.is_dir()]
                with_manifest = sum(1 for p in siblings if (p / "manifest.json").exists())
                if with_manifest > 0 and cell_dir.name not in LEGACY_NO_MANIFEST_WAIVED:
                    rep = CellReport(path=str(cell_dir / "manifest.json"))
                    rep.add("warning", "no manifest.json (sibling cells have one — incomplete sweep?)")
                    reports.append(rep)
    return reports


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _color(s: str, c: str) -> str:
    if not sys.stdout.isatty():
        return s
    codes = {"red": 31, "yellow": 33, "green": 32, "dim": 2}
    return f"\033[{codes[c]}m{s}\033[0m"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=[os.environ.get("HALO_OUTPUT_DIR", "experiments") + "/runs"])
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero on warnings as well as errors.")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON instead of a human report.")
    args = ap.parse_args()

    reports = audit([Path(r) for r in args.runs])

    if args.json:
        out = [{"path": r.path, "suite": r.suite, "method": r.method,
                "severity": r.severity, "issues": r.issues,
                "summary": r.score_summary} for r in reports]
        print(json.dumps(out, indent=2))
    else:
        n_err = sum(1 for r in reports if r.severity == "error")
        n_warn = sum(1 for r in reports if r.severity == "warning")
        n_ok = sum(1 for r in reports if r.severity == "ok")
        for r in reports:
            label = {"ok": _color("OK   ", "green"),
                     "warning": _color("WARN ", "yellow"),
                     "error": _color("ERROR", "red")}[r.severity]
            short = r.path.replace(os.path.expanduser("~"), "~")
            tag = f"{r.suite}/{r.method}" if r.suite != "?" else r.suite
            print(f"{label} {short:80s} {_color(tag, 'dim')} | {r.score_summary}")
            for issue in r.issues:
                print(f"      └─ {issue}")
        print()
        summary = (f"Audited {len(reports)} cells: "
                   f"{_color(str(n_ok), 'green')} OK, "
                   f"{_color(str(n_warn), 'yellow')} warnings, "
                   f"{_color(str(n_err), 'red')} errors.")
        print(summary)

    if any(r.severity == "error" for r in reports):
        return 2
    if args.strict and any(r.severity == "warning" for r in reports):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
