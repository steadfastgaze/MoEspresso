from __future__ import annotations

from importlib import metadata

import pytest

from moespresso.correctness import environment
from moespresso.correctness.environment import MLX_WHEEL_UNKNOWN, mlx_wheel_tag


def test_mlx_wheel_tag_reads_the_installed_dist_info():
    try:
        text = metadata.distribution("mlx").read_text("WHEEL")
    except metadata.PackageNotFoundError:
        pytest.skip("mlx is not installed in this environment")
    assert text
    expected = [
        line.partition(":")[2].strip()
        for line in text.splitlines()
        if line.lower().startswith("tag:")
    ]
    assert expected
    tag = mlx_wheel_tag()
    assert tag == ",".join(expected)
    assert tag != MLX_WHEEL_UNKNOWN


def test_mlx_wheel_tag_returns_unknown_when_lookup_raises(monkeypatch):
    def _boom(name: str):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(environment.metadata, "distribution", _boom)
    assert mlx_wheel_tag() == MLX_WHEEL_UNKNOWN


def test_mlx_wheel_tag_returns_unknown_without_a_tag_line(monkeypatch):
    class _Dist:
        def read_text(self, name: str) -> str:
            assert name == "WHEEL"
            return "Wheel-Version: 1.0\nRoot-Is-Purelib: false\n"

    monkeypatch.setattr(environment.metadata, "distribution", lambda name: _Dist())
    assert mlx_wheel_tag() == MLX_WHEEL_UNKNOWN


def test_mlx_wheel_tag_returns_unknown_when_wheel_file_is_missing(monkeypatch):
    class _Dist:
        def read_text(self, name: str):
            return None

    monkeypatch.setattr(environment.metadata, "distribution", lambda name: _Dist())
    assert mlx_wheel_tag() == MLX_WHEEL_UNKNOWN
