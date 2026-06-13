"""Tests for HALO_OVERRIDES env-var parsing (used by ablations.sh)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_parse_simple_floats():
    from halo.policy import parse_overrides

    out = parse_overrides("hot_ratio=0.05,score_alpha=1.0")
    assert out == {"hot_ratio": 0.05, "score_alpha": 1.0}


def test_parse_lists_and_bools():
    from halo.policy import parse_overrides

    out = parse_overrides("tiers=['gpu','dram','nvme'],async_refetch=False")
    assert out == {"tiers": ["gpu", "dram", "nvme"], "async_refetch": False}


def test_parse_strings_fall_through():
    from halo.policy import parse_overrides

    out = parse_overrides("layerwise_budget=pyramid")
    assert out == {"layerwise_budget": "pyramid"}


def test_parse_empty_returns_empty():
    from halo.policy import parse_overrides

    assert parse_overrides("") == {}
    assert parse_overrides("   ") == {}


def test_with_overrides_applies_known_fields():
    from halo.policy import HALOConfig

    cfg = HALOConfig()
    new = cfg.with_overrides({"hot_ratio": 0.30, "lookahead": 4})
    assert new.hot_ratio == 0.30
    assert new.lookahead == 4
    # Original is untouched.
    assert cfg.hot_ratio == 0.10
    assert cfg.lookahead == 1


def test_with_overrides_unknown_keys_go_to_extra():
    from halo.policy import HALOConfig

    cfg = HALOConfig()
    new = cfg.with_overrides({"some_research_knob": 0.5})
    assert new.extra == {"some_research_knob": 0.5}
    assert cfg.extra == {}


def test_env_var_picked_up(monkeypatch):
    from halo.policy import parse_overrides

    monkeypatch.setenv("HALO_OVERRIDES", "hot_ratio=0.20,sink_tokens=8")
    out = parse_overrides()
    assert out == {"hot_ratio": 0.20, "sink_tokens": 8}
