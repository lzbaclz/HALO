"""Round-trip tests for the closed-form classifier."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
sklearn = pytest.importorskip("sklearn")


def _write_synthetic_traces(tmp_path, n: int = 3, seed: int = 0):
    from scripts.synthesize_traces import synthesize_trace

    out = tmp_path / "traces"
    out.mkdir(parents=True, exist_ok=True)
    tasks = ["narrativeqa", "qasper", "passage_retrieval_en"][:n]
    for t in tasks:
        trace = synthesize_trace(task=t, context_len=256, n_steps=24,
                                 n_layers=4, num_heads=4, topk=64,
                                 model_name="synthetic", seed=seed)
        torch.save(trace, out / f"{t}.pt")
    return out


def test_fit_writes_classifier_npz(tmp_path):
    from halo.classifier import fit

    traces = _write_synthetic_traces(tmp_path)
    out = tmp_path / "clf.npz"
    summary = fit(str(traces), str(out), hot_threshold=0.10)
    assert out.exists()
    for k in ("alpha", "beta", "gamma", "bias", "auc"):
        assert k in summary
    # Synthetic data is highly separable → AUC should be very high.
    assert summary["auc"] > 0.85, f"expected high AUC on synthetic, got {summary['auc']:.3f}"


def test_eval_recovers_high_auc(tmp_path):
    from halo.classifier import evaluate, fit

    traces = _write_synthetic_traces(tmp_path)
    out = tmp_path / "clf.npz"
    fit(str(traces), str(out), hot_threshold=0.10)
    eval_out = evaluate(str(traces), str(out), hot_threshold=0.10)
    assert eval_out["auc"] > 0.85
