"""Hardware-gated KQ_SEG_TILE default for the routed GEMM.

Apple M5 and newer default to the device-A half-weight tile; M4 and older
keep the float default so the committed token-identity anchors stay exact.
An explicit KQ_SEG_TILE always wins and off-Apple hardware is untouched.
"""

from __future__ import annotations

import os

from moespresso.runtime.serve import default_kq_seg_tile_for_hardware

TILE_ENV = "KQ_SEG_TILE"


def _clear_env(monkeypatch):
    monkeypatch.setenv(TILE_ENV, "sentinel")
    monkeypatch.delenv(TILE_ENV)


def test_m5_defaults_half_weight_tile(monkeypatch, capsys):
    _clear_env(monkeypatch)
    assert default_kq_seg_tile_for_hardware(5) == "t48x128x16ah"
    assert os.environ[TILE_ENV] == "t48x128x16ah"
    out = capsys.readouterr().out
    # The notice is written for end users: plain words, no kernel jargon
    # beyond the kill-switch value itself.
    assert "faster GPU math path" in out
    assert "KQ_SEG_TILE=t48x128x16a" in out
    assert "staging" not in out and "float default" not in out


def test_m6_and_newer_also_default(monkeypatch):
    _clear_env(monkeypatch)
    assert default_kq_seg_tile_for_hardware(6) == "t48x128x16ah"


def test_m3_and_m4_keep_float_default(monkeypatch):
    for generation in (1, 3, 4):
        _clear_env(monkeypatch)
        assert default_kq_seg_tile_for_hardware(generation) is None
        assert TILE_ENV not in os.environ


def test_unknown_hardware_untouched(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(
        "moespresso.runtime.serve._apple_silicon_generation", lambda: None)
    assert default_kq_seg_tile_for_hardware() is None
    assert TILE_ENV not in os.environ


def test_explicit_tile_always_wins(monkeypatch):
    monkeypatch.setenv(TILE_ENV, "t48x128x16a")
    assert default_kq_seg_tile_for_hardware(5) is None
    assert os.environ[TILE_ENV] == "t48x128x16a"
