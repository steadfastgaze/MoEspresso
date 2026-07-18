"""DS4 quality gates pin expert-pool residency before loading the model.

The routed prefill has two numerically valid kernels selected by pool
residency; gate evidence is only comparable at a pinned pool state. These
tests pin the contract: gates default the prewarm env before the model
loads, and an explicit caller override wins.
"""

from __future__ import annotations

import pytest

import moespresso.correctness.deepseek_v4.quality as quality

PREWARM_ENV = "MOESPRESSO_SSD_PREWARM_EXPERTS"


def test_pin_sets_prewarm_all_when_unset(monkeypatch):
    # setenv first so monkeypatch restores the original (absent) state even
    # though the helper writes os.environ directly.
    monkeypatch.setenv(PREWARM_ENV, "sentinel")
    monkeypatch.delenv(PREWARM_ENV)
    quality.pin_full_expert_residency()
    import os

    assert os.environ[PREWARM_ENV] == "all"


def test_pin_keeps_explicit_override(monkeypatch):
    monkeypatch.setenv(PREWARM_ENV, "")
    quality.pin_full_expert_residency()
    import os

    assert os.environ[PREWARM_ENV] == ""


class _SentinelLoad(RuntimeError):
    pass


def test_q1_pins_residency_before_model_load(monkeypatch, tmp_path):
    import moespresso.runtime.serve as serve_mod

    def _raise(package_dir):
        raise _SentinelLoad(str(package_dir))

    monkeypatch.setenv(PREWARM_ENV, "sentinel")
    monkeypatch.delenv(PREWARM_ENV)
    monkeypatch.setattr(serve_mod, "load_served_model", _raise)
    with pytest.raises(_SentinelLoad):
        quality.q1_deepseek_v4_official_top20_parity(tmp_path / "pkg")
    import os

    assert os.environ[PREWARM_ENV] == "all"
