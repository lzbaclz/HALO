#!/usr/bin/env python3
"""Write a unified run-manifest sidecar next to a summary.json.

The audit  flagged that wall-clock numbers across followups
were not always reconstructable: HALO_TRITON_STREAMED, HALO_PATH_D_ASYNC_DMA,
HALO_FENCE_GIB, transformers version, torch version, git commit, and
the model revision are all needed to interpret a peak-GiB / wall-s pair.

This script captures all of them in a side-car ``manifest.json`` so a
reviewer can ``diff`` two manifests and see exactly what differed.

Usage (inside a runner script after summary.json is written)::

    python scripts/repro/write_run_manifest.py \\
        --output-dir experiments/auxiliary_cells/W17_vllm_enqa_32k_llama

Or programmatically::

    from scripts.repro.write_run_manifest import write_run_manifest
    write_run_manifest(out_dir, extra={"vllm_version": vllm.__version__})

Schema (the fields the audit asked for)::

    {
      "git_commit": "abc123...",
      "git_dirty": false,
      "torch_version": "2.6.0+cu124",
      "transformers_version": "4.57.6",
      "cuda_version": "12.4",
      "host": "<host>-<arch>",
      "gpu_name": "NVIDIA A100-SXM4-80GB",
      "gpu_driver": "535.288.01",
      "env": {
        "HALO_TRITON_STREAMED": "1",
        "HALO_PATH_D_ASYNC_DMA": "0",
        "HALO_FENCE_GIB": "",
        "HALO_LSE_FORCE_FP32": "",
        "HALO_MODEL_ZOO": "/public/model_zoo",
        "CUDA_VISIBLE_DEVICES": "0",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": ""
      },
      "timestamp_utc": "2026-05-19T16:36:24+00:00",
      "extra": {...}
    }

The ``extra`` field is a free-form dict for runner-specific metadata
(e.g. ``vllm_version``, ``num_seeds``, ``chunk_size``).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Optional

# Env vars the audit asked us to capture in every manifest.
_AUDITED_ENV_VARS = [
    "HALO_TRITON_STREAMED",
    "HALO_TRITON_SINGLE_LAUNCH",
    "HALO_PATH_D_ASYNC_DMA",
    "HALO_FENCE_GIB",
    "HALO_LSE_FORCE_FP32",
    "HALO_MODEL_ZOO",
    "HALO_MODEL_HUB_PREFIX",
    "CUDA_VISIBLE_DEVICES",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "PYTORCH_CUDA_ALLOC_CONF",
]


def _git_info() -> dict:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[2],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        return {"git_commit": head, "git_dirty": bool(status)}
    except Exception:
        return {"git_commit": None, "git_dirty": None}


def _torch_info() -> dict:
    info: dict = {}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
        else:
            info["gpu_name"] = None
    except Exception:
        info["torch_version"] = None
        info["cuda_version"] = None
        info["gpu_name"] = None
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except Exception:
        info["transformers_version"] = None
    return info


def _gpu_driver() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
        return out[0] if out else None
    except Exception:
        return None


def write_run_manifest(out_dir: Path, *, extra: Optional[dict] = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **_git_info(),
        **_torch_info(),
        "gpu_driver": _gpu_driver(),
        "host": platform.node() + "-" + platform.machine(),
        "env": {k: os.environ.get(k, "") for k in _AUDITED_ENV_VARS},
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "extra": extra or {},
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output-dir", required=True,
                    help="Directory to write manifest.json into.")
    args = ap.parse_args()
    path = write_run_manifest(Path(args.output_dir))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
