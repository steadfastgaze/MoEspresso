"""Export process-scoped Metal compute intervals without inferring kernel timing."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, DecimalException
import math
from pathlib import Path
import subprocess
import time
from typing import Any
import xml.etree.ElementTree as ET

from moespresso.runtime.diagnostic_environment import diagnostic_tool_environment


_MAX_XML_BYTES = 128 * 1024 * 1024
_GPU_SCHEMA = "metal-gpu-intervals"
_LIMITATIONS = [
    "Compute intervals describe traced GPU activity, not individual kernel dispatches.",
    "Interval gaps are not proof of CPU, SSD, or synchronization stalls.",
    "This export does not measure ALU utilization, memory bandwidth, or register occupancy.",
]


def _resolve(element: ET.Element, ids: dict[str, ET.Element]) -> ET.Element:
    seen = set()
    while "ref" in element.attrib:
        reference = element.attrib["ref"]
        if reference in seen or reference not in ids:
            raise ValueError("unresolved or cyclic XML reference")
        seen.add(reference)
        element = ids[reference]
    return element


def _text(element: ET.Element, ids: dict[str, ET.Element]) -> str:
    return (_resolve(element, ids).text or "").strip()


def _number(element: ET.Element, ids: dict[str, ET.Element]) -> int:
    value = _text(element, ids)
    return int(value, 16 if value.startswith("0x") else 10)


def _pid(element: ET.Element, ids: dict[str, ET.Element]) -> int:
    process = _resolve(element, ids)
    value = process.find("pid")
    if value is None:
        raise ValueError("process has no PID")
    return _number(value, ids)


def _union_ns(intervals: list[tuple[int, int]]) -> int:
    total = 0
    right = 0
    for start, end in sorted(intervals):
        total += max(0, end - max(right, start))
        right = max(right, end)
    return total


def _metrics(rows: list[dict[str, Any]], bounds: tuple[int, int] | None = None) -> dict:
    selected = []
    for row in rows:
        start, end = row["start_ns"], row["end_ns"]
        if bounds is not None:
            start, end = max(start, bounds[0]), min(end, bounds[1])
        if end > start:
            selected.append((start, end, row))
    intervals = [(start, end) for start, end, _ in selected]
    union = _union_ns(intervals)
    span = max((end for _, end in intervals), default=0) - min(
        (start for start, _ in intervals), default=0
    )
    missing_ids = sum(row["command_buffer"] is None for _, _, row in selected)
    result = {
        "compute_interval_count": len(selected),
        "compute_interval_union_ms": union / 1e6,
        "compute_interval_span_ms": span / 1e6,
        "compute_interval_gap_ms": (span - union) / 1e6,
        "command_buffer_count": (
            len({(row["device"], row["command_buffer"]) for _, _, row in selected})
            if not missing_ids
            else None
        ),
        "intervals_without_command_buffer_id": missing_ids,
        "dispatch_count": None,
    }
    if bounds is not None:
        duration = bounds[1] - bounds[0]
        result.update(
            window_ms=duration / 1e6,
            window_without_target_compute_intervals_ms=(duration - union) / 1e6,
        )
    return result


def _clock_mapping(node: ET.Element, ids: dict[str, ET.Element]) -> tuple[int, int, int]:
    mappings = set()
    for row in node.findall("row"):
        epoch = row.find("mach-absolute-time")
        timebase = row.find("mach-timebase-info")
        if epoch is None or timebase is None:
            raise ValueError("time-info omits its absolute epoch or timebase")
        fields = list(_resolve(timebase, ids))
        if len(fields) != 2:
            raise ValueError("unsupported timebase fields")
        numerator, denominator = (_number(field, ids) for field in fields)
        if numerator <= 0 or denominator <= 0:
            raise ValueError("invalid timebase ratio")
        mappings.add((_number(epoch, ids), numerator, denominator))
    if len(mappings) != 1:
        raise ValueError("missing or changing trace clock mapping")
    return mappings.pop()


def _window_summary(
    window: dict[str, Any],
    rows: list[dict[str, Any]],
    mapping: tuple[int, int, int] | None,
    duration_ns: int | None,
) -> dict:
    result = {"label": str(window.get("label", "unnamed"))}
    if mapping is None:
        return {**result, "status": "unavailable", "reason": "trace clock mapping unavailable"}
    try:
        epoch, numerator, denominator = mapping
        ticks = [window["start_mach_ticks"], window["end_mach_ticks"]]
        if any(type(value) is not int for value in ticks):
            raise ValueError("window bounds must be integer mach_absolute_time ticks")
        start, end = [(value - epoch) * numerator // denominator for value in ticks]
        if end <= start:
            raise ValueError("window end must follow its start")
    except (KeyError, TypeError, ValueError) as exc:
        return {**result, "status": "unavailable", "reason": str(exc)}
    result.update(start_trace_ns=start, end_trace_ns=end)
    if duration_ns is None:
        result.update(status="partial", coverage="unknown", reason="trace duration unavailable")
        bounds = (max(0, start), end)
    else:
        bounds = (max(0, start), min(end, duration_ns))
        complete = start >= 0 and end <= duration_ns
        result.update(
            status="complete" if complete else "partial",
            coverage="inside_recording" if complete else "clipped_to_recording",
            recorded_fraction=max(0, bounds[1] - bounds[0]) / (end - start),
        )
    if bounds[1] <= bounds[0]:
        return {**result, "status": "unavailable", "reason": "window is outside recording"}
    result.update(_metrics(rows, bounds))
    if not result["compute_interval_count"]:
        result.update(status="unavailable", reason="no target compute intervals in window")
    return result


def summarize_trace_xml(
    xml_text: str | bytes,
    *,
    pid: int | None = None,
    windows: list[dict[str, Any]] | None = None,
    trace_duration_ns: int | None = None,
    clock_xml_text: str | bytes | None = None,
) -> dict:
    """Summarize the supported xctrace schema and explicitly mark missing evidence.

    Window bounds are raw ``mach_absolute_time`` ticks from the worker. They
    are mapped with the trace's exported ``time-info`` epoch and timebase.
    No wall-clock or elapsed-time alignment is inferred.
    """
    result: dict[str, Any] = {
        "status": "unavailable",
        "pid": pid,
        "schemas": [],
        "warnings": [],
        "limitations": list(_LIMITATIONS),
        "kernel_timing": {"status": "unavailable", "reason": "no validated per-kernel schema"},
    }
    if type(pid) is not int or pid <= 0:
        return {**result, "reason": "an explicit positive worker PID is required"}
    try:
        root = ET.fromstring(xml_text)
        ids = {element.attrib["id"]: element for element in root.iter() if "id" in element.attrib}
    except (ET.ParseError, ValueError) as exc:
        return {**result, "reason": f"invalid XML: {exc}"}
    rows: list[dict[str, Any]] = []
    mapping = None
    if clock_xml_text is not None:
        try:
            clock_root = ET.fromstring(clock_xml_text)
            clock_ids = {
                element.attrib["id"]: element
                for element in clock_root.iter()
                if "id" in element.attrib
            }
            clock_nodes = [
                node for node in clock_root.findall("node")
                if node.find("schema") is not None
                and node.find("schema").attrib.get("name") == "time-info"
            ]
            if len(clock_nodes) != 1:
                raise ValueError("expected one time-info table")
            mapping = _clock_mapping(clock_nodes[0], clock_ids)
        except (ET.ParseError, ValueError, TypeError) as exc:
            result["warnings"].append(f"Clock mapping unavailable: {exc}")
    invalid_rows = 0
    excluded_noncompute_or_inactive_rows = 0
    observed = Counter()
    for node in root.findall("node"):
        schema = node.find("schema")
        if schema is None:
            result["warnings"].append("An exported node has no schema; its rows were not inferred.")
            continue
        name = schema.attrib.get("name", "unknown")
        columns = [col.findtext("mnemonic", "") for col in schema.findall("col")]
        result["schemas"].append({"name": name, "columns": columns, "rows": len(node.findall("row"))})
        if name == "time-info":
            try:
                mapping = _clock_mapping(node, ids)
            except (ValueError, TypeError) as exc:
                result["warnings"].append(f"Clock mapping unavailable: {exc}")
            continue
        if name != _GPU_SCHEMA:
            continue
        required = {"start", "duration", "channel-name", "state", "process"}
        if not required.issubset(columns) or len(set(columns)) != len(columns):
            result["warnings"].append("Unsupported Metal GPU interval columns.")
            continue
        for row in node.findall("row"):
            try:
                if len(row) != len(columns):
                    raise ValueError("row does not match its schema")
                fields = dict(zip(columns, row, strict=True))
                channel = _text(fields["channel-name"], ids)
                if channel != "Compute" or _text(fields["state"], ids) != "Active":
                    excluded_noncompute_or_inactive_rows += 1
                    continue
                row_pid = _pid(fields["process"], ids)
                observed[(row_pid, channel)] += 1
                if row_pid != pid:
                    continue
                start = _number(fields["start"], ids)
                duration = _number(fields["duration"], ids)
                if start < 0 or duration < 0:
                    raise ValueError("negative interval time")
                command_buffer = (
                    _number(fields["cmdbuffer-id"], ids) if "cmdbuffer-id" in fields else None
                )
                if command_buffer in (0, 2**64 - 1):
                    command_buffer = None
                rows.append(
                    {
                        "start_ns": start,
                        "end_ns": start + duration,
                        "command_buffer": command_buffer,
                        "device": _text(fields["gpu"], ids) if "gpu" in fields else "unknown",
                    }
                )
            except (ValueError, TypeError, KeyError):
                invalid_rows += 1
    result["observed_process_channels"] = [
        {"pid": row_pid, "channel": channel, "intervals": count}
        for (row_pid, channel), count in sorted(observed.items())
    ]
    result["invalid_rows"] = invalid_rows
    result["excluded_noncompute_or_inactive_rows"] = excluded_noncompute_or_inactive_rows
    result["clock_mapping"] = (
        {"mach_epoch": mapping[0], "numerator": mapping[1], "denominator": mapping[2]}
        if mapping is not None
        else None
    )
    result["windows"] = [
        _window_summary(window, rows, mapping, trace_duration_ns) for window in windows or []
    ]
    if invalid_rows:
        for window in result["windows"]:
            if window["status"] != "unavailable":
                window.update(
                    status="partial",
                    recording_bounds_coverage=window.get("coverage", "unknown"),
                    coverage="incomplete_interval_rows",
                    reason="malformed rows may omit target compute intervals",
                )
    if not rows:
        return {**result, "reason": "no supported target-process active compute intervals"}
    result.update(status="partial" if invalid_rows else "complete", **_metrics(rows))
    result["scope"] = "whole traced process; use aligned windows for decode-only attribution"
    result["devices"] = {
        device: _metrics([row for row in rows if row["device"] == device])
        for device in sorted({row["device"] for row in rows})
    }
    if invalid_rows:
        result["warnings"].append("Malformed rows were excluded; interval coverage is incomplete.")
    return result


def _read_xml_bytes(path: Path) -> bytes:
    with path.open("rb") as source:
        content = source.read(_MAX_XML_BYTES + 1)
    if len(content) > _MAX_XML_BYTES:
        raise ValueError(f"{path.name} exceeds the XML parsing limit; raw export retained")
    return content


def _export(command: list[str], output: Path, timeout_s: float) -> dict:
    if timeout_s <= 0:
        return {"status": "unavailable", "reason": "trace export time budget exhausted"}
    if output.exists():
        return {"status": "unavailable", "reason": f"export already exists: {output.name}"}
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_s, check=False,
            env=diagnostic_tool_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "reason": str(exc), "output": str(output)}
    return {
        "status": "complete" if completed.returncode == 0 and output.is_file() else "unavailable",
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "output": str(output),
    }


def export_and_summarize_trace(
    trace_path: Path,
    output_dir: Path,
    *,
    pid: int | None = None,
    windows: list[dict[str, Any]] | None = None,
    timeout_s: float = 60,
) -> dict:
    """Export an existing recording. Failures retain raw files for inspection."""
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        return {"status": "unavailable", "reason": "export timeout must be positive"}
    deadline = time.monotonic() + timeout_s
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return {"status": "unavailable", "reason": f"cannot create export directory: {exc}"}
    toc_path = output_dir / "trace-toc.xml"
    intervals_path = output_dir / "trace-intervals.xml"
    clock_path = output_dir / "trace-clock.xml"
    base = ["xcrun", "xctrace", "export", "--input", str(trace_path)]
    toc_export = _export(
        base + ["--toc", "--output", str(toc_path)], toc_path, deadline - time.monotonic()
    )
    if toc_export["status"] != "complete":
        return {"status": "unavailable", "reason": "trace table export failed", "exports": [toc_export]}
    try:
        toc = ET.fromstring(_read_xml_bytes(toc_path))
        runs = toc.findall("run")
        if len(runs) != 1:
            raise ValueError("exactly one recording run is required")
        run = runs[0]
        number = int(run.attrib["number"])
        if not any(table.attrib.get("schema") == _GPU_SCHEMA for table in run.findall("data/table")):
            raise ValueError("trace has no Metal GPU interval table")
        duration_text = run.findtext("info/summary/duration")
        duration_ns = None
        if duration_text:
            duration_seconds = Decimal(duration_text)
            if not duration_seconds.is_finite() or not 0 < duration_seconds <= 2**63 / 1e9:
                raise ValueError("invalid recording duration")
            duration_ns = int(duration_seconds * 1_000_000_000)
            if duration_ns <= 0:
                raise ValueError("invalid recording duration")
    except (OSError, ET.ParseError, ValueError, KeyError, DecimalException, OverflowError) as exc:
        return {"status": "unavailable", "reason": str(exc), "exports": [toc_export]}
    xpath = f'/trace-toc/run[@number="{number}"]/data/table[@schema="{_GPU_SCHEMA}"]'
    interval_export = _export(
        base + ["--xpath", xpath, "--output", str(intervals_path)],
        intervals_path,
        deadline - time.monotonic(),
    )
    exports = [toc_export, interval_export]
    if interval_export["status"] != "complete":
        return {"status": "unavailable", "reason": "GPU interval export failed", "exports": exports}
    clock_xpath = f'/trace-toc/run[@number="{number}"]/data/table[@schema="time-info"]'
    clock_export = _export(
        base + ["--xpath", clock_xpath, "--output", str(clock_path)],
        clock_path,
        deadline - time.monotonic(),
    )
    exports.append(clock_export)
    try:
        result = summarize_trace_xml(
            _read_xml_bytes(intervals_path), pid=pid, windows=windows, trace_duration_ns=duration_ns,
            clock_xml_text=(_read_xml_bytes(clock_path) if clock_export["status"] == "complete" else None),
        )
    except (OSError, ValueError) as exc:
        return {"status": "unavailable", "reason": str(exc), "exports": exports}
    return {**result, "exports": exports, "trace_duration_ns": duration_ns}
