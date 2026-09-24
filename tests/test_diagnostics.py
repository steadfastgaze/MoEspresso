"""Stdout speed checks and server-side resource telemetry."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from moespresso.agentlib.client import ChatCompletion, ClientError
from moespresso.runtime import diagnostics as d


def usage(rate=2):
    return {
        "completion_tokens": 4,
        "moespresso": {
            "generation_seconds": 0.5 + 3 / rate,
            "first_token_seconds": 0.5,
            "generation_tps": 999,
        },
    }


def health(*, capacity=223, source="automatic-wired-headroom"):
    gib = 1 << 30
    mib = 1 << 20
    return {
        "status": "ok",
        "diagnostics": {
            "identity": {
                "family": "qwen4_exp",
                "context_limit": 131072,
                "template_kwargs": {"reasoning_effort": "medium"},
                "gpu_inventory": [{"core_count": 32}],
            },
            "device": {"device_name": "Apple M1 Max"},
            "system_memory": {"total_bytes": 32 * gib},
        },
        "ssd_streaming": {
            "enabled": True,
            "capacity_per_layer": capacity,
            "capacity_overrides": {},
            "capacity_budget": {
                "resident_base_bytes": 10 * gib,
                "runtime_resident_bytes": 0,
                "kv_activation_allowance_bytes": 0,
                "safety_margin_bytes": 64 * mib,
                "bytes_per_capacity_unit": 64 * mib,
                "min_capacity": 12,
                "max_capacity": 512,
                "full_resident_expert_bytes": 32 * gib,
                "planner_resolution": {
                    "resolved_bytes": (18 if source == "live-available" else 24) * gib,
                    "limiting_source": source,
                    "physical_reserve_ceiling_bytes": 27 * gib,
                    "automatic_ceiling_bytes": 24 * gib,
                },
            },
            "cache_routing": {
                "policy": "prefer-resident",
                "cache_factor": 2,
                "protected_routes": 2,
            },
        },
    }


class Client:
    def __init__(self, usages=None, failure=None, fail_after=0, health_value=None,
                 health_failure=None):
        self.calls = []
        self.usages = usages
        self.failure = failure
        self.fail_after = fail_after
        self.health_value = health() if health_value is None else health_value
        self.health_failure = health_failure
        self.health_calls = 0

    def health(self):
        self.health_calls += 1
        if self.health_failure is not None:
            raise self.health_failure
        return self.health_value

    def complete(self, messages, **kwargs):
        index = len(self.calls)
        self.calls.append((messages, kwargs))
        if self.failure is not None and index == self.fail_after:
            raise self.failure
        return ChatCompletion(
            message={"role": "assistant", "content": "response not printed"},
            finish_reason="stop",
            usage=self.usages[index] if self.usages is not None else usage(),
        )


def test_speed_check_prints_split_summary_and_run_conditions(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    client = Client([
        usage(100),
        usage(1), usage(10),
        usage(3), usage(12),
        usage(5), usage(14),
    ])
    assert d.run_speed_check(client, max_tokens=128) == 0
    output = capsys.readouterr()
    assert output.out == (
        "MoEspresso speed\n"
        "Apple M1 Max · 32 GPU cores · 32 GiB\n"
        "Qwen3.8-Flash-Next · context 131072 · thinking medium\n"
        "SSD experts 223/512 per layer · auto memory 24.00 GiB · Cache-Prior 2/2\n"
        "Decode: topic switch 3.00 · same topic 12.00 · overall 7.50 tok/s\n"
        "TTFT: 0.50 s median · greedy · 6/6 requests · 128-token cap\n"
        "Conditions: normal\n"
    )
    assert "Warming up..." in output.err
    assert "response not printed" not in output.out + output.err
    assert list(tmp_path.iterdir()) == []
    assert client.health_calls == 1
    assert len(client.calls) == 7
    assert [kw["max_tokens"] for _, kw in client.calls] == [64] + [128] * 6
    assert [messages[0]["content"] for messages, _kwargs in client.calls[1:]] == [
        prompt for _name, first, second in d.SPEED_TOPICS for prompt in (first, second)
    ]
    for _messages, kwargs in client.calls:
        assert all(kwargs[name] == value for name, value in d.SAMPLING.items())


def test_health_failure_is_nonfatal_and_reported(capsys):
    client = Client(health_failure=ClientError(404, "missing"))
    assert d.run_speed_check(client, max_tokens=64) == 0
    output = capsys.readouterr()
    assert "Run conditions unavailable; continuing with timing only." in output.err
    assert "Decode: topic switch 2.00 · same topic 2.00 · overall 2.00 tok/s" in output.out
    assert "Conditions: unavailable" in output.out


@pytest.mark.parametrize("error,code", [(ClientError(500, "failed"), 1), (KeyboardInterrupt(), 130)])
def test_failure_keeps_completed_measurements_on_stdout(capsys, error, code):
    client = Client(failure=error, fail_after=2)
    assert d.run_speed_check(client, max_tokens=64) == code
    output = capsys.readouterr()
    assert "Decode: topic switch 2.00 · same topic unavailable · overall 2.00 tok/s" in output.out
    assert "1/6 requests" in output.out
    assert "failed" in output.err or "interrupted" in output.err


def test_missing_server_timing_does_not_become_client_timing(capsys):
    client = Client([{}] * 7)
    assert d.run_speed_check(client, max_tokens=64) == 1
    output = capsys.readouterr().out
    assert "Decode: topic switch unavailable · same topic unavailable · overall unavailable tok/s" in output
    assert "TTFT: unavailable · greedy · 0/6 requests" in output


def test_unavailable_request_does_not_discard_other_measurements(capsys):
    client = Client([usage(), usage(2), {}, usage(4), {}, usage(6), {}])
    assert d.run_speed_check(client, max_tokens=64) == 0
    output = capsys.readouterr().out
    assert "Decode: topic switch 4.00 · same topic unavailable · overall 4.00 tok/s" in output
    assert "3/6 requests" in output


def test_live_memory_warning_reports_material_capacity_loss(capsys):
    client = Client(health_value=health(capacity=130, source="live-available"))
    assert d.run_speed_check(client, max_tokens=64) == 0
    output = capsys.readouterr()
    assert (
        "Live available memory limited the expert pool to 130 resident experts per layer "
        "(normal planner ceiling: about 223)" in output.err
    )
    assert "Conditions: memory constrained;" in output.out


def test_live_memory_limit_without_slot_loss_does_not_warn():
    result = d._speed_environment(health(capacity=223, source="live-available"))
    assert result["warning"] is None
    assert result["condition"] == "normal"


def test_small_live_memory_capacity_change_does_not_trigger_severe_warning():
    result = d._speed_environment(health(capacity=222, source="live-available"))
    assert result["warning"] is None
    assert result["condition"] == "normal"


def test_normal_capacity_honors_explicit_ceiling_and_full_residency():
    value = health(capacity=130, source="live-available")["ssd_streaming"]
    resolution = value["capacity_budget"]["planner_resolution"]
    resolution["explicit_ceiling_bytes"] = 20 << 30
    assert d._normal_planner_capacity(value, "qwen4_exp") == 159
    value["capacity_budget"]["full_resident_expert_bytes"] = 9 << 30
    assert d._normal_planner_capacity(value, "qwen4_exp") == 512


def test_capacity_overrides_are_reported_as_a_range():
    value = health()
    value["ssd_streaming"]["capacity_overrides"] = {"3": 211, "9": 240}
    assert any(
        "SSD experts 211-240/512 per layer" in line
        for line in d._speed_environment(value)["lines"]
    )


@pytest.mark.parametrize("count,total,first", [
    (1, 1, 0.5), (4, 1, 1), (4, 1, -1), (4, float("nan"), 0.5),
    (4, 1, float("inf")), (True, 2, 0.5),
])
def test_unmeasurable_decode_rate_is_unavailable(count, total, first):
    result = d._metrics({
        "completion_tokens": count,
        "moespresso": {"generation_seconds": total, "first_token_seconds": first},
    })
    assert result["after_first_token_tps"] is None


@pytest.mark.parametrize(
    "platform,kind", [("darwin", "vm_pageins_pageouts"), ("linux", "psutil_sin_sout")]
)
def test_snapshot_uses_read_only_resource_functions(monkeypatch, platform, kind):
    monkeypatch.setattr(d, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(d, "gpu_inventory", lambda: [{"core_count": 24}])
    monkeypatch.setattr(d, "process_resources", lambda: {
        "counter_kind": "darwin_proc_pid_rusage_v2", "disk_read_bytes": 123,
        "disk_write_bytes": 456, "physical_footprint_bytes": 789,
    })
    vm = SimpleNamespace(total=32, available=4, used=28)
    swap = SimpleNamespace(used=6, sin=7, sout=8)
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(_asdict=lambda: {"rss": 20}))
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(
            virtual_memory=lambda: vm,
            swap_memory=lambda: swap,
            Process=lambda: process,
            sensors_battery=lambda: SimpleNamespace(percent=80, power_plugged=True),
        ),
    )
    mx = SimpleNamespace(
        device_info=lambda: {"device_name": "test device"},
        get_active_memory=lambda: 10,
        get_cache_memory=lambda: 2,
        get_peak_memory=lambda: 12,
    )
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    callback = d.make_server_diagnostics(
        {"artifact_id": "test"}, context_limit=1024, template_kwargs=None, generation_defaults={}
    )
    first, second = callback(), callback()
    assert first["schema"] == "moespresso-server-diagnostics-v3"
    assert first["identity"] == second["identity"]
    assert first["system_memory"]["available_bytes"] == 4
    assert first["system_memory"]["swap_used_bytes"] == 6
    assert first["system_memory"]["paging_in_bytes"] == 7
    assert first["system_memory"]["paging_out_bytes"] == 8
    assert first["system_memory"]["paging_counter_kind"] == kind
    assert "swap_in_bytes" not in first["system_memory"]
    assert "swap_out_bytes" not in first["system_memory"]
    assert "psutil" in first["identity"]["versions"]
    assert first["identity"]["gpu_inventory"] == [{"core_count": 24}]
    assert first["mlx_memory"] == {"active_bytes": 10, "cached_bytes": 2, "peak_active_bytes": 12}
    assert not first["errors"]
    assert first["process_io"]["disk_read_bytes"] == 123
    assert first["process_io"]["physical_footprint_bytes"] == 789


@pytest.mark.parametrize("args", [
    ["--url", "ftp://example.org"],
    ["--url", "http://user:secret@example.org"],
    ["--url", "http://example.org/v1"],
    ["--repeats", "0"],
    ["--max-tokens", "9000"],
    ["--timeout", "nan"],
])
def test_invalid_options_fail_before_network(args):
    with pytest.raises(SystemExit) as exc:
        d.main(args)
    assert exc.value.code == 2


def test_diagnostic_client_import_does_not_import_mlx():
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from moespresso.runtime.diagnostics import main; "
            "import sys; assert not any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("stream", [False, True])
def test_real_http_speed_check_preserves_serving_sampling_and_telemetry(stream, capsys):
    from http.server import ThreadingHTTPServer
    import threading

    from moespresso.agentlib.client import CompletionsClient
    from moespresso.runtime.generation import GenerationResult
    from moespresso.runtime.http import make_handler

    seen = []

    def generate(prompt, **kwargs):
        seen.append(kwargs)
        return GenerationResult(
            text="answer", prompt_tokens=5, completion_tokens=4,
            first_token_seconds=0.5, generation_seconds=2, cached_tokens=0,
        )

    health_calls = 0

    def server_diagnostics():
        nonlocal health_calls
        health_calls += 1
        return health()["diagnostics"]

    handler = make_handler(
        generate,
        model_id="test",
        diagnostics=server_diagnostics,
        runtime_stats=lambda: health()["ssd_streaming"],
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = CompletionsClient(
            f"http://127.0.0.1:{server.server_port}", timeout=10, stream=stream,
        )
        assert d.run_speed_check(client, max_tokens=64) == 0
        assert health_calls == 1
        assert len(seen) == 7
        assert all(all(opts[key] == value for key, value in d.SAMPLING.items()) for opts in seen)
        output = capsys.readouterr().out
        assert "Decode: topic switch 2.00 · same topic 2.00 · overall 2.00 tok/s" in output
        assert "6/6 requests" in output
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("args,stream", [([], False), (["--stream"], True)])
def test_cli_uses_the_selected_transport(monkeypatch, args, stream):
    from moespresso.cli import main

    def run(client, **kwargs):
        assert client.stream is stream
        assert kwargs == {"max_tokens": 256}
        return 0

    monkeypatch.setattr(d, "run_speed_check", run)
    assert main(["speed", *args]) == 0


def test_diagnose_command_has_no_compatibility_alias():
    from moespresso.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["diagnose"])
    assert exc.value.code == 2
