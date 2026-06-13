"""FU_W17 data prep: download BABILong from HuggingFace and write to
``experiments/babilong_data/<ctx_kib>k/<task>.jsonl`` (the format
``scripts/run_pathd_babilong.py`` expects).

The Hub dataset ``RMT-team/babilong-1k-samples`` ships as parquet shards
at ``<ctx>/qa<i>-*-of-*.parquet`` (no datasets-config script). We pull
the shards directly via ``hf_hub_download`` and convert to jsonl.

Run::

    python scripts/prepare_babilong_data.py --ctx-kib 32 --tasks qa1 qa2 --n 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _prepare(ctx_kib: int, tasks: list[str], n: int, out_root: Path) -> None:
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    repo = "RMT-team/babilong-1k-samples"
    api = HfApi()
    all_files = api.list_repo_files(repo, repo_type="dataset")

    out_dir = out_root / f"{ctx_kib}k"
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        out = out_dir / f"{task}.jsonl"
        if out.exists() and out.stat().st_size > 0:
            print(f"[fu_w17 prep] SKIP {out} already populated")
            continue
        shards = [f for f in all_files
                  if f.startswith(f"{ctx_kib}k/{task}-") and f.endswith(".parquet")]
        if not shards:
            print(f"[fu_w17 prep] no shards found for {ctx_kib}k/{task}",
                  file=sys.stderr)
            continue
        print(f"[fu_w17 prep] {ctx_kib}k/{task}: {len(shards)} shard(s)",
              flush=True)
        rows = []
        for shard in sorted(shards):
            local = hf_hub_download(repo, shard, repo_type="dataset")
            t = pq.read_table(local)
            for r in t.to_pylist():
                rows.append({
                    "input":    r.get("input", "") or "",
                    "question": r.get("question", "") or "",
                    "answer":   r.get("target", r.get("answer", "")) or "",
                })
                if len(rows) >= n:
                    break
            if len(rows) >= n:
                break
        with out.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {len(rows)} examples → {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx-kib", type=int, default=32)
    ap.add_argument("--tasks", nargs="+", default=["qa1", "qa2"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out-root", default="experiments/babilong_data")
    args = ap.parse_args()
    _prepare(args.ctx_kib, args.tasks, args.n, Path(args.out_root))


if __name__ == "__main__":
    main()
