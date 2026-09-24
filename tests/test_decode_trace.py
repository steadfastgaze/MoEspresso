from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from moespresso.runtime import decode_trace as trace


def schema(columns):
    return "".join(f"<col><mnemonic>{name}</mnemonic></col>" for name in columns)


GPU_COLUMNS = [
    "start", "duration", "start-latency", "channel-name", "state", "process",
    "gpu", "cmdbuffer-id", "encoder-id",
]


def gpu_row(start, duration, *, pid=42, command_buffer=100, encoder=200, channel="Compute"):
    return (
        f"<row><start-time>{start}</start-time><duration>{duration}</duration>"
        f"<duration>900</duration><gpu-channel-name>{channel}</gpu-channel-name>"
        f"<gpu-state>Active</gpu-state><process><pid>{pid}</pid></process>"
        f"<metal-device-name>M1</metal-device-name>"
        f"<metal-command-buffer-id>{command_buffer}</metal-command-buffer-id>"
        f"<metal-command-buffer-id>{encoder}</metal-command-buffer-id></row>"
    )


def gpu_xml(*rows, columns=None):
    return (
        '<trace-query-result><node><schema name="metal-gpu-intervals">'
        + schema(GPU_COLUMNS if columns is None else columns)
        + "</schema>" + "".join(rows) + "</node></trace-query-result>"
    )


def clock_xml():
    return (
        '<trace-query-result><node><schema name="time-info"/>'
        '<row><sample-time>0</sample-time><mach-absolute-time>1000</mach-absolute-time>'
        '<mach-timebase-info><mach-timebase-info-field>125</mach-timebase-info-field>'
        '<mach-timebase-info-field>3</mach-timebase-info-field></mach-timebase-info>'
        '</row></node></trace-query-result>'
    )


def test_intervals_are_unioned_and_command_buffers_not_confused_with_encoders():
    result = trace.summarize_trace_xml(
        gpu_xml(
            gpu_row(100, 200, encoder=201),
            gpu_row(200, 200, encoder=202),
            gpu_row(500, 100, command_buffer=101),
            gpu_row(0, 1000, pid=99),
            gpu_row(0, 1000, channel="Fragment"),
        ),
        pid=42,
    )
    assert result["status"] == "complete"
    assert result["compute_interval_count"] == 3
    assert result["compute_interval_union_ms"] == 400 / 1e6
    assert result["compute_interval_span_ms"] == 500 / 1e6
    assert result["compute_interval_gap_ms"] == 100 / 1e6
    assert result["command_buffer_count"] == 2
    assert result["dispatch_count"] is None
    assert result["kernel_timing"]["status"] == "unavailable"


def test_references_resolve_and_pid_filter_does_not_accept_launcher():
    first = gpu_row(100, 200).replace("<process><pid>42</pid></process>",
                                    '<process id="P"><pid id="PID">42</pid></process>')
    second = gpu_row(300, 200).replace("<process><pid>42</pid></process>", '<process ref="P"/>')
    result = trace.summarize_trace_xml(gpu_xml(first, second), pid=42)
    assert result["compute_interval_count"] == 2
    assert trace.summarize_trace_xml(gpu_xml(first, second), pid=41)["status"] == "unavailable"


@pytest.mark.parametrize("pid", [None, 0, -1, True, "42"])
def test_explicit_worker_pid_is_required(pid):
    assert trace.summarize_trace_xml(gpu_xml(gpu_row(0, 10)), pid=pid)["status"] == "unavailable"


def test_clock_mapping_aligns_and_clips_declared_windows():
    result = trace.summarize_trace_xml(
        gpu_xml(gpu_row(0, 200), gpu_row(300, 500)),
        pid=42,
        clock_xml_text=clock_xml(),
        trace_duration_ns=1000,
        windows=[
            {"label": "decode", "start_mach_ticks": 1003, "end_mach_ticks": 1018},
            {"label": "partial", "start_mach_ticks": 997, "end_mach_ticks": 1030},
            {"label": "outside", "start_mach_ticks": 1030, "end_mach_ticks": 1033},
        ],
    )
    decode, partial, outside = result["windows"]
    assert decode["start_trace_ns"] == 125
    assert decode["end_trace_ns"] == 750
    assert decode["compute_interval_union_ms"] == 525 / 1e6
    assert decode["window_without_target_compute_intervals_ms"] == 100 / 1e6
    assert decode["coverage"] == "inside_recording"
    assert partial["status"] == "partial"
    assert partial["recorded_fraction"] == pytest.approx(1000 / 1375)
    assert outside["status"] == "unavailable"


def test_clock_ids_are_independent_between_separate_exports():
    gpu = gpu_xml(gpu_row(0, 100)).replace("<duration>100</duration>", '<duration id="1">100</duration>')
    clock = clock_xml().replace("<mach-absolute-time>", '<mach-absolute-time id="1">')
    result = trace.summarize_trace_xml(gpu, pid=42, clock_xml_text=clock)
    assert result["compute_interval_union_ms"] == 100 / 1e6
    assert result["clock_mapping"]["mach_epoch"] == 1000


def test_missing_clock_does_not_infer_alignment_from_elapsed_time():
    result = trace.summarize_trace_xml(
        gpu_xml(gpu_row(0, 100)), pid=42,
        windows=[{"label": "decode", "start_mach_ticks": 1000, "end_mach_ticks": 2000}],
    )
    assert result["status"] == "complete"
    assert result["windows"][0]["status"] == "unavailable"
    assert "compute_interval_union_ms" not in result["windows"][0]


def test_duration_absence_does_not_claim_full_recording_coverage():
    result = trace.summarize_trace_xml(
        gpu_xml(gpu_row(0, 100)), pid=42, clock_xml_text=clock_xml(),
        windows=[{"start_mach_ticks": 1000, "end_mach_ticks": 1003}],
    )
    assert result["windows"][0]["coverage"] == "unknown"
    assert result["windows"][0]["status"] == "partial"


@pytest.mark.parametrize("content", ["<broken", "<trace-query-result/>",
    '<trace-query-result><node><row><start-time>0</start-time></row></node></trace-query-result>'])
def test_missing_or_malformed_schema_is_unavailable(content):
    result = trace.summarize_trace_xml(content, pid=42)
    assert result["status"] == "unavailable"
    assert "compute_interval_union_ms" not in result


def test_invalid_rows_mark_partial_without_hiding_valid_rows():
    broken = gpu_row(300, 100).replace("<process><pid>42</pid></process>", '<process ref="missing"/>')
    result = trace.summarize_trace_xml(gpu_xml(gpu_row(0, 100), broken), pid=42)
    assert result["status"] == "partial"
    assert result["invalid_rows"] == 1
    assert result["compute_interval_count"] == 1


def test_unattributed_noncompute_row_does_not_make_target_coverage_partial():
    vertex = gpu_row(200, 10, channel="Vertex").replace(
        "<process><pid>42</pid></process>", "<sentinel/>"
    )
    result = trace.summarize_trace_xml(gpu_xml(gpu_row(0, 100), vertex), pid=42)
    assert result["status"] == "complete"
    assert result["invalid_rows"] == 0
    assert result["excluded_noncompute_or_inactive_rows"] == 1


def test_unattributed_active_compute_row_makes_aligned_windows_partial():
    unattributed = gpu_row(200, 10).replace("<process><pid>42</pid></process>", "<sentinel/>")
    result = trace.summarize_trace_xml(
        gpu_xml(gpu_row(0, 100), unattributed), pid=42, clock_xml_text=clock_xml(),
        trace_duration_ns=1000,
        windows=[{"start_mach_ticks": 1000, "end_mach_ticks": 1018}],
    )
    assert result["status"] == "partial"
    window = result["windows"][0]
    assert window["status"] == "partial"
    assert window["coverage"] == "incomplete_interval_rows"
    assert window["recording_bounds_coverage"] == "inside_recording"


def test_missing_command_buffer_ids_do_not_become_zero_buffers():
    content = gpu_xml(gpu_row(0, 100, command_buffer=2**64 - 1))
    result = trace.summarize_trace_xml(content, pid=42)
    assert result["command_buffer_count"] is None
    assert result["intervals_without_command_buffer_id"] == 1


def test_export_uses_separate_schema_calls_and_preserves_raw_files(tmp_path, monkeypatch):
    from moespresso.runtime.diagnostic_environment import diagnostic_tool_environment

    calls = []
    monkeypatch.setenv("HF_TOKEN", "synthetic-secret")
    monkeypatch.setenv("UNRELATED_SETTING", "synthetic-secret")

    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["env"] == diagnostic_tool_environment()
        assert "synthetic-secret" not in kwargs["env"].values()
        output = Path(command[command.index("--output") + 1])
        if "--toc" in command:
            text = ('<trace-toc><run number="1"><info><summary><duration>1</duration>'
                    '</summary></info><data><table schema="metal-gpu-intervals"/>'
                    '</data></run></trace-toc>')
        elif "time-info" in command[command.index("--xpath") + 1]:
            text = clock_xml()
        else:
            text = gpu_xml(gpu_row(0, 100))
        output.write_text(text)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(trace.subprocess, "run", run)
    result = trace.export_and_summarize_trace(tmp_path / "capture.trace", tmp_path, pid=42)
    assert result["status"] == "complete"
    assert len(calls) == 3
    assert result["trace_duration_ns"] == 1_000_000_000
    assert calls[1][1]["timeout"] <= calls[0][1]["timeout"]
    assert calls[2][1]["timeout"] <= calls[1][1]["timeout"]
    assert all(command[:3] == ["xcrun", "xctrace", "export"] for command, _ in calls)
    assert (tmp_path / "trace-clock.xml").is_file()
    assert trace.export_and_summarize_trace(tmp_path / "capture.trace", tmp_path, pid=42)["status"] == "unavailable"
    assert len(calls) == 3


def test_export_timeout_retains_partial_file(tmp_path, monkeypatch):
    def run(command, **kwargs):
        Path(command[command.index("--output") + 1]).write_text("partial")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(trace.subprocess, "run", run)
    result = trace.export_and_summarize_trace(tmp_path / "capture.trace", tmp_path, pid=42, timeout_s=1)
    assert result["status"] == "unavailable"
    assert (tmp_path / "trace-toc.xml").read_text() == "partial"


def test_export_errors_are_not_summarized(tmp_path, monkeypatch):
    monkeypatch.setattr(trace.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 1, "", "permission denied"))
    result = trace.export_and_summarize_trace(tmp_path / "capture.trace", tmp_path, pid=42)
    assert result["status"] == "unavailable"
    assert result["exports"][0]["stderr"] == "permission denied"


@pytest.mark.parametrize("duration", ["NaN", "sNaN", "Infinity", "-Infinity", "1e1000000", "0", "-1"])
def test_export_rejects_nonfinite_or_unrepresentable_duration(tmp_path, monkeypatch, duration):
    def run(command, **kwargs):
        Path(command[command.index("--output") + 1]).write_text(
            '<trace-toc><run number="1"><info><summary><duration>' + duration
            + '</duration></summary></info><data><table schema="metal-gpu-intervals"/>'
            '</data></run></trace-toc>'
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(trace.subprocess, "run", run)
    result = trace.export_and_summarize_trace(tmp_path / "capture.trace", tmp_path, pid=42)
    assert result["status"] == "unavailable"
    assert result["reason"] == "invalid recording duration"
    assert len(result["exports"]) == 1


@pytest.mark.parametrize("oversized_name", ["trace-toc.xml", "trace-intervals.xml", "trace-clock.xml"])
def test_every_export_read_is_bounded(tmp_path, monkeypatch, oversized_name):
    def run(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        if output.name == oversized_name:
            text = "x" * 1025
        elif output.name == "trace-toc.xml":
            text = ('<trace-toc><run number="1"><info><summary><duration>1</duration>'
                    '</summary></info><data><table schema="metal-gpu-intervals"/>'
                    '</data></run></trace-toc>')
        elif output.name == "trace-clock.xml":
            text = clock_xml()
        else:
            text = gpu_xml(gpu_row(0, 100))
        output.write_text(text)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(trace, "_MAX_XML_BYTES", 1024)
    monkeypatch.setattr(trace.subprocess, "run", run)
    result = trace.export_and_summarize_trace(tmp_path / "capture.trace", tmp_path, pid=42)
    assert result["status"] == "unavailable"
    assert oversized_name in result["reason"]
    assert (tmp_path / oversized_name).stat().st_size == 1025
