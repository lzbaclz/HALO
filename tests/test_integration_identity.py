"""GPU-only integration test: hot_ratio=1.0 must be bit-identical to baseline.

This is the §4 paper claim ("HALO is a strict identity over full attention at
hot_ratio=1.0") backed by an end-to-end test on a real 7B model rather than the
synthetic CPU fixtures used by ``tests/test_kv_cache.py``.

Run with::

    pytest -m gpu tests/test_integration_identity.py -v

(unmarked tests skip the GPU bits in CI).

The test loads the model in eager-attention bf16 to:
  - exercise the standard HALOCache code path (eager attention is what HALO
    hooks plug into; SDPA / FlashAttention have a separate fast path),
  - and avoid SDPA's nondeterministic kernel selection that can flip the last
    bits of the logits even when no other state changes.

Because eager attention OOMs on 128K context, we cap the prompt at 4K tokens
and decode 64 new tokens (matches the trace-collection cap in
``scripts/extract_attention_trace.py``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _gpu_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() >= 1


pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not _gpu_available(), reason="needs a CUDA-capable GPU")
def test_identity_on_qwen() -> None:
    """hot_ratio=1.0 produces the exact same token sequence as no wrapping."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from halo import HALOConfig, wrap_with_halo

    model_name = os.environ.get("HALO_IDENTITY_MODEL", "Qwen/Qwen2.5-7B")
    max_context = int(os.environ.get("HALO_IDENTITY_MAX_CONTEXT", "4096"))
    max_new = int(os.environ.get("HALO_IDENTITY_MAX_NEW", "64"))
    prompt_path = _REPO_ROOT / "experiments" / "prompts" / "narrativeqa.txt"

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()

    prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else (
        "The quick brown fox jumps over the lazy dog. " * 200
    )
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=max_context).to("cuda:0")

    torch.manual_seed(0)
    with torch.no_grad():
        base_out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False, num_beams=1,
            pad_token_id=tok.eos_token_id,
        )

    wrap_with_halo(model, HALOConfig(hot_ratio=1.0))
    torch.manual_seed(0)
    with torch.no_grad():
        halo_out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False, num_beams=1,
            pad_token_id=tok.eos_token_id,
        )

    assert torch.equal(base_out, halo_out), (
        "hot_ratio=1.0 must produce a bit-identical token sequence; "
        f"got {base_out.tolist()=} vs {halo_out.tolist()=}"
    )

    cache = model._halo_cache  # type: ignore[attr-defined]
    assert cache.demoted_blocks_total == 0
    assert cache.refetcher.misses == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
