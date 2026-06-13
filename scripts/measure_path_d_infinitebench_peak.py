"""Measure peak GPU memory for Path D vs Full on real ∞-Bench prompts.

Reviewer 1 (#W1) flagged that we lacked a joint (peak GPU, F1) cell on
real benchmark prompts. The infinitebench runner doesn't track peak
GPU; this script does. Runs both ``full`` and ``path_d`` (Path D under
stock HF generate() with install_preforward_peel) on the same
∞-Bench En.QA prompts back-to-back in fresh subprocesses and reports
``torch.cuda.max_memory_allocated()`` per method.

Output: ``experiments/runs/qwen2-5-7b/infinitebench_peak_v1/peak.json``
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]


def _load(model_cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = model_cfg.get("name_or_path") or model_cfg.get("name") or model_cfg["model"]
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def _wrap_path_d(model):
    from halo import HALOConfig, wrap_with_halo, install_preforward_peel
    cfg = HALOConfig(chunked=True, chunk_size=512, recent_window=64, hot_ratio=0.25)
    wrap_with_halo(model, cfg)
    install_preforward_peel(model, prefill_chunk_tokens=4096, activation_threshold=8192)


def run(method: str, *, n: int, ctx: int, out_dir: Path,
        model_cfg: dict) -> dict:
    from baselines.infinitebench_eval import evaluate_task

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    model, tok = _load(model_cfg)
    if method == "path_d":
        _wrap_path_d(model)

    score = evaluate_task(
        model, tok,
        task="en_qa",
        context_length=ctx,
        limit=n,
        output_dir=out_dir / method / "en_qa",
        press_ctx=None,
    )

    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    wall = time.time() - t0
    del model
    gc.collect(); torch.cuda.empty_cache()
    return {"method": method, "n": n, "f1_mean": float(score),
            "peak_gpu_gib": float(peak), "wall_s": float(wall)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--method", choices=["full", "path_d", "both"], default="both")
    ap.add_argument("--out",
                    default="experiments/runs/qwen2-5-7b/infinitebench_peak_v1")
    args = ap.parse_args()

    model_cfg = yaml.safe_load(
        (REPO / "configs/models/qwen2-5-7b.yaml").read_text())

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    if args.method in ("full", "both"):
        print("\n=== Full ===", flush=True)
        results["full"] = run("full", n=args.n, ctx=args.ctx,
                              out_dir=out_dir, model_cfg=model_cfg)
        print(f"  Full   : F1={results['full']['f1_mean']:.2f}"
              f" peak={results['full']['peak_gpu_gib']:.2f} GiB"
              f" wall={results['full']['wall_s']:.1f}s", flush=True)
    if args.method in ("path_d", "both"):
        print("\n=== Path D (install_preforward_peel) ===", flush=True)
        results["path_d"] = run("path_d", n=args.n, ctx=args.ctx,
                                out_dir=out_dir, model_cfg=model_cfg)
        print(f"  Path D : F1={results['path_d']['f1_mean']:.2f}"
              f" peak={results['path_d']['peak_gpu_gib']:.2f} GiB"
              f" wall={results['path_d']['wall_s']:.1f}s", flush=True)
    if "full" in results and "path_d" in results:
        df = results["path_d"]["peak_gpu_gib"] - results["full"]["peak_gpu_gib"]
        pct = 100.0 * (results["path_d"]["peak_gpu_gib"]
                       / results["full"]["peak_gpu_gib"] - 1.0)
        results["delta"] = {"peak_gib": df, "peak_pct": pct}
        print(f"\nΔpeak: {df:+.2f} GiB ({pct:+.1f}%)", flush=True)

    (out_dir / "peak.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_dir/'peak.json'}", flush=True)


if __name__ == "__main__":
    main()
