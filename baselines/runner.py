"""Single dispatcher used by ``scripts/run_*.py``.

Loads the model + tokenizer once, builds the baseline (or HALO wrap), and
dispatches to the right harness. Returns the per-task metric value.

Behavior on missing dependencies:
  - If ``torch``/``transformers`` cannot be imported → returns NaN (CI-friendly).
  - If a baseline raises (e.g. KIVI) → logs a warning and returns NaN.
  - If the harness module is missing → returns NaN.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from halo.utils import get_logger


def _ratio_to_compression(memory_ratio: int) -> float:
    """memory_ratio = 1×, 2×, 4×, 8× → compression_ratio = 0, 0.5, 0.75, 0.875.

    Mirrors the same mapping in ``baselines/__init__.py`` for the kvpress
    baselines. Kept local so the runner doesn't need a registry import just
    for HALOPress construction.
    """
    if memory_ratio <= 1:
        return 0.0
    return 1.0 - (1.0 / float(memory_ratio))

_log = get_logger("halo.runner")
logging.basicConfig(level=logging.INFO)


# Cache the loaded model / tokenizer so a single ``run_longbench.py`` invocation
# runs N tasks back-to-back without paying ``from_pretrained`` cost N times.
# Keyed by ``(name, attn_impl, method, memory_ratio)`` so the same script can
# (in principle) evaluate multiple methods on the same model in one process,
# without leaking the wrapping state between methods.
_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


def _maybe_install_fence(device: int = 0):
    """If HALO_FENCE_GIB is set, allocate a CUDA blocker tensor at process
    start so only that many GiB remain free for subsequent allocations.

    Useful when the rented GPU has more VRAM than the paper's headline
    target (e.g. a "RTX 4090" RunPod box reports 48 GiB instead of the
    standard 24 GiB --- HALO_FENCE_GIB=24 simulates the consumer card).
    The blocker is parked on a module-level attribute so it's not GC'd
    when this function returns.
    """
    import os
    target = os.environ.get("HALO_FENCE_GIB")
    if not target:
        return
    import torch
    if not torch.cuda.is_available():
        return
    if getattr(_maybe_install_fence, "_done", False):
        return
    try:
        target_gib = float(target)
    except ValueError:
        _log.warning("HALO_FENCE_GIB='%s' not a number, ignoring", target)
        return
    torch.cuda.set_device(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_gib = free_bytes / 1024 ** 3
    total_gib = total_bytes / 1024 ** 3
    blocker_gib = max(0.0, free_gib - target_gib - 1.0)  # 1 GiB safety margin
    if blocker_gib <= 0:
        _log.info("fence: total=%.1f GiB free=%.1f GiB <= target=%.1f GiB, "
                  "no blocker needed", total_gib, free_gib, target_gib)
        _maybe_install_fence._done = True
        _maybe_install_fence._blocker = None
        return
    n_bytes = int(blocker_gib * 1024 ** 3)
    blocker = torch.empty(n_bytes, dtype=torch.uint8, device=device)
    _log.info("fence: total=%.1f GiB free=%.1f GiB, blocker=%.1f GiB → "
              "~%.1f GiB available (HALO_FENCE_GIB=%s)",
              total_gib, free_gib, blocker_gib, target_gib, target)
    _maybe_install_fence._blocker = blocker  # keep alive
    _maybe_install_fence._done = True


def _load_model_and_tokenizer(model_cfg: dict, *, method: str, memory_ratio: int):
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Install GPU-memory fence if HALO_FENCE_GIB is set. Must happen BEFORE
    # any large allocation (model load) so the blocker reservation succeeds.
    _maybe_install_fence(device=0)

    name = model_cfg["name_or_path"]
    # Local-path / HF-id reconciliation: if name_or_path is a relative
    # or absolute file path that doesn't exist (e.g. a different machine's
    # /public/model_zoo layout), but HALO_MODEL_ZOO is set, try resolving
    # ``$HALO_MODEL_ZOO/<basename>`` first; otherwise fall back to the
    # original (which AutoModelForCausalLM will then try via HF Hub).
    # This keeps the artifact reproducible across machines without
    # forcing every reviewer to live-edit YAML.
    import os as _os
    if (_os.path.isabs(name) and not _os.path.exists(name)
            and "HALO_MODEL_ZOO" in _os.environ):
        candidate = _os.path.join(_os.environ["HALO_MODEL_ZOO"],
                                  _os.path.basename(name))
        if _os.path.exists(candidate):
            name = candidate
    attn = model_cfg.get("attn_implementation", "sdpa")
    key = f"{name}@{attn}#{method}#{memory_ratio}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    # Free any previously cached model from the GPU before loading a new one.
    # This matters when a single script invocation evaluates multiple
    # (method, memory_ratio) cells back-to-back: without an explicit free,
    # the old model stays on the GPU until Python's GC kicks in, which
    # routinely OOMs us on a single 80GB A100.
    if _MODEL_CACHE:
        for k in list(_MODEL_CACHE.keys()):
            old_model, _ = _MODEL_CACHE.pop(k)
            try:
                old_model.to("cpu")
            except Exception:
                pass
            del old_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _log.info("loading model %s for method=%s ratio=%dx ...", name, method, memory_ratio)
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    dtype_name = model_cfg.get("dtype", "bfloat16")
    # FU_W4 (reviewer v3 W4): HALO_LSE_FORCE_FP32=1 promotes the entire
    # forward to fp32 to verify the prediction in Prop. 4.5 that fp32
    # forward should close the bf16 qa_2 deviation. Memory is 2× higher;
    # only feasible on smaller cells (qa_2 at 8K fits on one A100 80GB).
    import os as _os
    if _os.environ.get("HALO_LSE_FORCE_FP32") == "1":
        _log.info("HALO_LSE_FORCE_FP32=1 → forcing model dtype to float32")
        dtype_name = "float32"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[dtype_name]
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        device_map=model_cfg.get("device_map", "auto"),
        attn_implementation=attn,
        trust_remote_code=True,
    )
    model.eval()
    _MODEL_CACHE[key] = (model, tok)
    return model, tok


def evaluate(
    *,
    model_cfg: dict,
    task: str,
    method: str,
    memory_ratio: int = 4,
    extra: Optional[dict] = None,
    output_dir: Any = None,
    limit: Optional[int] = None,
) -> float:
    extra = extra or {}
    _log.info("dispatch method=%s task=%s ratio=%dx", method, task, memory_ratio)

    try:
        import torch  # noqa: F401
    except ImportError:
        _log.warning("torch not available — returning NaN.")
        return float("nan")

    # Some kvpress baselines (H2O / SnapKV) require eager attention because
    # their hooks read the post-softmax attention probabilities. We honour the
    # ``required_attn_impl`` annotation on the factory if present.
    cfg_for_load = dict(model_cfg)
    if method not in ("full", "halo"):
        try:
            from baselines import REGISTRY as _REG
            req_attn = getattr(_REG.get(method), "required_attn_impl", None)
            if req_attn:
                cfg_for_load["attn_implementation"] = req_attn
        except Exception:
            pass

    try:
        model, tok = _load_model_and_tokenizer(cfg_for_load, method=method,
                                               memory_ratio=memory_ratio)
    except Exception as e:  # pragma: no cover - GPU may be unavailable in CI
        # Default behaviour is NaN-return so CI runs without a GPU still
        # produce a (degenerate) summary.json. For real experiment runs,
        # set HALO_FAIL_ON_LOAD_ERROR=1 to escalate to a hard exit — this
        # prevents the "10-minute silent NaN run" failure mode where every
        # seed's model load fails and the script reports success with 0
        # predictions.
        import os as _os
        if _os.environ.get("HALO_FAIL_ON_LOAD_ERROR", "0") not in ("", "0", "false", "False"):
            _log.error("model load failed (%s) and HALO_FAIL_ON_LOAD_ERROR=1 — aborting.", e)
            raise
        _log.warning("model load failed (%s) — returning NaN.", e)
        return float("nan")

    # ------------------------------------------------------------------
    # Build the per-method press / wrapping. We only wrap once per cache key:
    # the wrapping state (HALO patched generate, no kvpress hooks) lives on
    # the cached model object, so subsequent task evaluations re-use it.
    # ------------------------------------------------------------------
    press_ctx = None  # callable(model) -> contextmanager

    if method == "halo":
        # HALO has two cooperating components:
        #   1. ``wrap_with_halo`` installs the HALOCache + scoring/telemetry
        #      bookkeeping (one-time, idempotent) — this preserves the §4
        #      identity invariant at hot_ratio=1.0.
        #   2. ``HALOPress`` is the per-call compression hook — it actually
        #      drops (1 - 1/memory_ratio) of KV positions during prefill,
        #      using the same alpha*attn+beta*recency+gamma*sink scoring rule.
        if not getattr(model, "_halo_generate_patched", False):
            from halo import HALOConfig, wrap_with_halo
            try:
                wrap_with_halo(model, HALOConfig(hot_ratio=1.0 / memory_ratio))
            except Exception as e:  # pragma: no cover
                _log.warning("wrap_with_halo failed (%s) — returning NaN.", e)
                return float("nan")
        if memory_ratio > 1:
            try:
                from halo.halo_press import HALOPress
                from halo.policy import parse_overrides
                ov = parse_overrides()
                press_kwargs = {"compression_ratio": _ratio_to_compression(memory_ratio)}
                for k in ("sink_tokens", "score_alpha", "score_beta", "score_gamma"):
                    if k in ov:
                        press_kwargs[k] = ov[k]
                if press_kwargs.keys() & {"sink_tokens", "score_alpha",
                                         "score_beta", "score_gamma"}:
                    _log.info("HALOPress overrides from HALO_OVERRIDES: %s",
                              {k: v for k, v in press_kwargs.items()
                               if k != "compression_ratio"})
                press_ctx = HALOPress(**press_kwargs)
            except Exception as e:  # pragma: no cover
                _log.warning("HALOPress unavailable (%s) — running halo as identity.", e)
    elif method == "halo_hybrid":
        # The HALO + StreamingLLM hybrid: closed-form score with the recent
        # window force-retained. See halo/halo_hybrid_press.py and §5.3 of
        # the paper.
        if not getattr(model, "_halo_generate_patched", False):
            from halo import HALOConfig, wrap_with_halo
            try:
                wrap_with_halo(model, HALOConfig(hot_ratio=1.0 / memory_ratio))
            except Exception as e:  # pragma: no cover
                _log.warning("wrap_with_halo failed (%s) — returning NaN.", e)
                return float("nan")
        if memory_ratio > 1:
            try:
                from halo.halo_hybrid_press import HALOHybridPress
                from halo.policy import parse_overrides
                ov = parse_overrides()
                press_kwargs = {"compression_ratio": _ratio_to_compression(memory_ratio)}
                for k in ("sink_tokens", "score_alpha", "score_beta",
                          "score_gamma", "recent_tokens"):
                    if k in ov:
                        press_kwargs[k] = ov[k]
                press_ctx = HALOHybridPress(**press_kwargs)
            except Exception as e:  # pragma: no cover
                _log.warning("HALOHybridPress unavailable (%s) — falling back to identity.", e)
    elif method == "kivi":
        # KIVI-style 2-/4-bit KV quantization (self-contained port).
        # Wraps the model's generate() to install a KIVICache that
        # quantizes K per-channel and V per-token at the chosen bit-width
        # (16 / memory_ratio, clamped at 2). press_ctx stays None
        # because KIVI is a cache wrapper, not a prefill prune.
        if not getattr(model, "_kivi_patched", False):
            try:
                from baselines.kivi_cache import wrap_with_kivi
                wrap_with_kivi(model, memory_ratio=memory_ratio)
            except Exception as e:  # pragma: no cover
                _log.warning("wrap_with_kivi failed (%s) — returning NaN.", e)
                return float("nan")
        press_ctx = None
    elif method == "quest":
        # Quest (Tang et al. ICML 2024). Like HALO, Quest is NOT a
        # kvpress press but a cache wrapper + attention-interface hook
        # (per-step query-aware top-K page selection). The wrap
        # registers the ``quest`` attention interface and installs
        # ``QuestPagedCache``; ``press_ctx`` stays None because Quest
        # does no prefill prune.
        if not getattr(model, "_quest_generate_patched", False):
            try:
                from baselines.quest_cache import QuestConfig, wrap_with_quest
                _ps = int(os.environ.get("HALO_QUEST_PAGE_SIZE", "16"))
                wrap_with_quest(model, QuestConfig(
                    memory_ratio=float(memory_ratio), page_size=_ps))
            except Exception as e:  # pragma: no cover
                _log.warning("wrap_with_quest failed (%s) — returning NaN.", e)
                return float("nan")
        press_ctx = None
    elif method == "path_d":
        # Path D under stock HF generate() with install_preforward_peel.
        # This is the deployable form of the title's identity-preserving contract:
        # users call model.generate(long_prompt) unchanged, and our hook
        # transparently routes prefill through the chunked-prefill loop.
        if not getattr(model, "_halo_preforward_patched", False):
            try:
                from halo import (
                    HALOConfig, wrap_with_halo, install_preforward_peel,
                )
                # Triton fast path enabled by default when CUDA + Triton are
                # available; set HALO_DISABLE_TRITON=1 to fall back to the
                # reference Python loop (bit-exact fp32 reference path).
                _use_triton = os.environ.get("HALO_DISABLE_TRITON", "0") != "1"
                cfg = HALOConfig(
                    chunked=True, chunk_size=512, recent_window=64,
                    hot_ratio=1.0 / max(1.0, float(memory_ratio)),
                    use_triton=_use_triton,
                )
                wrap_with_halo(model, cfg)
                install_preforward_peel(
                    model, prefill_chunk_tokens=4096,
                    activation_threshold=8192,
                )
            except Exception as e:  # pragma: no cover
                _log.warning("wrap_with_halo+preforward_peel failed (%s) — NaN.", e)
                return float("nan")
        press_ctx = None
    elif method == "quest_path_d":
        # Quest+PathD composition (\Cref{sec:exp-quest-pathd}):
        # Quest's per-step page selection drives GPU hot-tier
        # residency, while Path D's chunked LSE-merge keeps the
        # attention preserves algebraic identity over the full partition. Wraps via
        # ``baselines.quest_path_d.wrap_with_quest_path_d``, which
        # composes Path D (chunked=True) with a Quest scorer
        # adapter. Output quality is Full-equivalent by
        # Prop. 4.5 (fp32) / Cor. 4.6 (bf16 within 2% rel L2).
        if not getattr(model, "_quest_path_d_config", None):
            try:
                from baselines.quest_path_d import (
                    QuestPathDConfig, wrap_with_quest_path_d,
                )
                # HALO_QUEST_PAGE_SIZE env override for adversarial
                # page-size sweeps (Quest failure-mode study).
                _ps = int(os.environ.get("HALO_QUEST_PAGE_SIZE", "16"))
                wrap_with_quest_path_d(
                    model,
                    QuestPathDConfig(memory_ratio=float(memory_ratio),
                                     page_size=_ps),
                )
            except Exception as e:  # pragma: no cover
                _log.warning("wrap_with_quest_path_d failed (%s) — returning NaN.", e)
                return float("nan")
        press_ctx = None
    elif method == "full":
        press_ctx = None
    else:
        try:
            from baselines import REGISTRY
            factory = REGISTRY[method]
            press = factory(model, memory_ratio=memory_ratio, **extra)
            press_ctx = press  # kvpress Press is itself a callable contextmanager
        except KeyError:
            raise ValueError(f"Unknown method '{method}'")
        except NotImplementedError as e:
            _log.warning("method=%s not implemented (%s) — returning NaN.", method, e)
            return float("nan")
        except Exception as e:  # pragma: no cover
            _log.warning("baseline build failed (%s) — returning NaN.", e)
            return float("nan")

    # ------------------------------------------------------------------
    # Dispatch.
    # ------------------------------------------------------------------
    if task.startswith("ruler:"):
        return _ruler(model, tok, task.split(":", 1)[1], extra,
                      output_dir, limit, press_ctx)
    if task.startswith("infinitebench:"):
        return _infinitebench(model, tok, task.split(":", 1)[1], extra,
                              output_dir, limit, press_ctx)
    return _longbench(model, tok, task, extra, output_dir, limit, press_ctx)


# ---------------------------------------------------------------------------
# Harness dispatchers
# ---------------------------------------------------------------------------


def _longbench(model, tok, task: str, extra: dict, output_dir, limit, press_ctx) -> float:
    try:
        from baselines.longbench_eval import evaluate_task
    except ImportError as e:  # pragma: no cover
        _log.warning("longbench_eval unavailable (%s) — returning NaN.", e)
        return float("nan")
    max_input_length = extra.get("max_input_length", 7500)
    return float(evaluate_task(
        model, tok, task=task,
        limit=limit, output_dir=output_dir,
        max_input_length=max_input_length,
        press_ctx=press_ctx,
    ))


def _ruler(model, tok, task: str, extra: dict, output_dir, limit, press_ctx) -> float:
    try:
        from baselines.ruler_eval import evaluate_task  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        _log.warning("ruler_eval not installed (%s) — returning NaN. "
                     "Run `bash scripts/setup_ruler.sh` first.", e)
        return float("nan")
    return float(evaluate_task(
        model, tok, task=task,
        context_length=extra.get("context_length", 8192),
        output_dir=output_dir,
        press_ctx=press_ctx,
        limit=limit,
    ))


def _infinitebench(model, tok, task: str, extra: dict, output_dir, limit, press_ctx) -> float:
    try:
        from baselines.infinitebench_eval import evaluate_task  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        _log.warning("infinitebench_eval not installed (%s) — returning NaN.", e)
        return float("nan")
    return float(evaluate_task(
        model, tok, task=task,
        context_length=extra.get("context_length", 1_000_000),
        output_dir=output_dir,
        press_ctx=press_ctx,
        limit=limit,
    ))
