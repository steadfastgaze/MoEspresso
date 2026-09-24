"""Read-only Darwin process counters and unavailable-counter behavior."""

import ctypes
from types import SimpleNamespace

import pytest

from moespresso.runtime import process_resources as p


def test_gpu_discovery_uses_an_allowlisted_environment(monkeypatch):
    from moespresso.runtime.diagnostic_environment import diagnostic_tool_environment

    monkeypatch.setattr(p, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setenv("HF_TOKEN", "synthetic-secret")
    monkeypatch.setenv("UNRELATED_SETTING", "synthetic-secret")

    def run(command, **kwargs):
        assert kwargs["env"] == diagnostic_tool_environment()
        assert "synthetic-secret" not in kwargs["env"].values()
        return SimpleNamespace(stdout='{"SPDisplaysDataType": []}')

    monkeypatch.setattr(p.subprocess, "run", run)
    assert p.gpu_inventory() == []


def test_gpu_inventory_omits_device_names(monkeypatch):
    monkeypatch.setattr(p, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(
        p.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout='{"SPDisplaysDataType": [{"sppci_model": "Personal GPU", "sppci_cores": "24"}]}'
        ),
    )

    assert p.gpu_inventory() == [{"core_count": 24}]


def test_darwin_rusage_v2_layout():
    assert ctypes.sizeof(p._RusageV2) == 160
    assert p._RusageV2.phys_footprint.offset == 72
    assert p._RusageV2.diskio_bytesread.offset == 144
    assert p._RusageV2.diskio_byteswritten.offset == 152


def test_process_counters_include_a_real_zero(monkeypatch):
    monkeypatch.setattr(p, "sys", SimpleNamespace(platform="darwin"))

    def reader(pid, flavor, pointer):
        assert pid == p.os.getpid()
        assert flavor == 2
        value = ctypes.cast(pointer, ctypes.POINTER(p._RusageV2)).contents
        value.diskio_bytesread = 0
        value.diskio_byteswritten = 42
        value.phys_footprint = 99
        return 0

    monkeypatch.setattr(p, "_reader", lambda: reader)
    assert p.process_resources() == {
        "counter_kind": "darwin_proc_pid_rusage_v2", "disk_read_bytes": 0,
        "disk_write_bytes": 42, "physical_footprint_bytes": 99,
    }


def test_unsupported_platform_does_not_load_darwin_library(monkeypatch):
    monkeypatch.setattr(p, "sys", SimpleNamespace(platform="linux"))

    def unexpected():
        raise AssertionError("must not read Darwin counters")

    monkeypatch.setattr(p, "_reader", unexpected)
    assert p.process_resources() is None


@pytest.mark.parametrize("failure", [1, OSError("unavailable"), AttributeError("missing")])
def test_failed_reader_is_unavailable_not_zero(monkeypatch, failure):
    monkeypatch.setattr(p, "sys", SimpleNamespace(platform="darwin"))

    def reader(*args):
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setattr(p, "_reader", lambda: reader)
    assert p.process_resources() is None
