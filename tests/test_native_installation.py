"""Installed native-module discovery and optional runtime fallback."""

from types import SimpleNamespace

import pytest

from moespresso.runtime import native_gate


@pytest.fixture
def loader(monkeypatch):
    monkeypatch.setattr(native_gate, "_ENV_DIR", None)
    monkeypatch.setattr(native_gate, "_GATE", [None])
    monkeypatch.setenv("MOESPRESSO_SSD_GATE_DECODE", "1")
    return native_gate, native_gate.load_gate, "moespresso._native._moespresso_gate"


def test_loader_uses_installed_package_and_caches_result(loader, monkeypatch):
    module, load, expected = loader
    extension = SimpleNamespace()
    imports = []
    checks = []

    def import_module(name):
        imports.append(name)
        return extension

    def self_test(value):
        checks.append(value)
        return True

    monkeypatch.setattr(module.importlib, "import_module", import_module)
    monkeypatch.setattr(module, "_self_test", self_test)
    assert load() is extension
    assert load() is extension
    assert imports == [expected]
    assert checks == [extension]


@pytest.mark.parametrize("error", [ImportError("missing"), RuntimeError("incompatible")])
def test_loader_preserves_fallback_if_installed_extension_cannot_load(loader, monkeypatch, error):
    module, load, expected = loader
    imports = []

    def import_module(name):
        imports.append(name)
        raise error

    monkeypatch.setattr(module.importlib, "import_module", import_module)
    assert load() is None
    assert load() is None
    assert imports == [expected]
