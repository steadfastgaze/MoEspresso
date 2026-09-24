"""Join sampled CPU activity and Metal submissions with decode compute gaps."""

from collections import Counter
from bisect import bisect_right
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from moespresso.runtime.decode_trace import _export, _number, _pid, _read_xml_bytes, _resolve, _text


TABLES = {
    "submissions": "metal-application-command-buffer-submissions",
    "cpu": "time-profile",
    "allocations": "metal-resource-allocations",
    "compiler": "graphics-compiler-activity-intervals",
}
FAMILIES = {
    "preparation": "Graph preparation / Python execution",
    "submission": "Native encoding / submission",
    "data": "Data reads / service",
    "coordination": "Runtime waits / publication",
    "allocation_compile": "Allocation / compilation",
    "os_driver": "OS / driver scheduling",
}


def export_detail_tables(trace_path, output_dir, *, timeout_s=60):
    """Export only named timing tables; never copy target environment metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    result = {}
    for name, schema in TABLES.items():
        path = output_dir / (name + ".xml")
        result[name] = _export([
            "xcrun", "xctrace", "export", "--input", str(trace_path),
            "--xpath", f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]',
            "--output", str(path),
        ], path, deadline - time.monotonic())
    return result


def _rows(content, schema, required):
    root = ET.fromstring(content)
    ids = {e.attrib["id"]: e for e in root.iter() if "id" in e.attrib}
    nodes = [n for n in root.findall("node") if n.find("schema") is not None
             and n.find("schema").attrib.get("name") == schema]
    if len(nodes) != 1:
        raise ValueError("expected exactly one supported table")
    node = nodes[0]
    columns = [c.findtext("mnemonic") for c in node.findall("schema/col")]
    if len(set(columns)) != len(columns) or not set(required) <= set(columns):
        raise ValueError("unsupported table columns")
    rows = []
    for row in node.findall("row"):
        if len(row) != len(columns):
            raise ValueError("row does not match table columns")
        rows.append(dict(zip(columns, row, strict=True)))
    return rows, ids


def _frames(element, ids):
    stack = _resolve(element, ids)
    backtrace = stack.find("backtrace")
    if backtrace is None:
        return []
    return [_resolve(frame, ids).attrib.get("name", "") for frame in _resolve(backtrace, ids)]


def classify_stack(frames):
    """Classify the nearest recognized stack frame, not its Python ancestors."""
    for frame in frames:
        if any(part in frame for part in (
            "MetalAllocator::malloc", "newBufferWith", "newComputePipelineState",
            "newLibraryWithSource", "MTLCompiler", "compile_to_metal_ir",
        )):
            return "allocation_compile"
        if any(part in frame for part in ("pread", "read_into", "readv", "BundleRowCache")):
            return "data"
        if any(part in frame for part in (
            "AllHitAttempt", "semaphore_wait", "__psynch_cvwait", "__psynch_mutexwait",
            "condition_variable::wait", "waitUntilCompleted", "wait_until_completed",
            "wait_for_one", "signal_event", "signaled_value",
        )):
            return "coordination"
        if any(part in frame for part in (
            "::eval_gpu", "::eval_impl", "CommandEncoder::", "ComputeCommandEncoder",
            "dispatchThreadgroups", "::set_input_array", "::finalize", "CommandBuffer commit",
        )):
            return "submission"
        if frame.startswith(("_Py", "Py", "_PyEval")):
            return "preparation"
    return "unclassified"


def _parse(content, kind, pid):
    required = {
        "cpu": ("time", "process", "thread", "thread-state", "stack"),
        "submissions": ("start", "duration", "process", "gpu", "cmdbuffer-id", "event-type"),
        "allocations": ("timestamp", "process", "resource-size", "event-type"),
        "compiler": ("start", "duration", "process"),
    }[kind]
    rows, ids = _rows(content, TABLES[kind], required)
    parsed = []
    invalid = 0
    for fields in rows:
        try:
            if _pid(fields["process"], ids) != pid:
                continue
            if kind == "cpu":
                thread = _resolve(fields["thread"], ids)
                tid = _number(thread.find("tid"), ids)
                row = {
                    "time": _number(fields["time"], ids), "tid": tid,
                    "main_label": thread.attrib.get("fmt", "").startswith("Main Thread "),
                    "state": _text(fields["thread-state"], ids),
                    "family": classify_stack(_frames(fields["stack"], ids)),
                }
            elif kind == "submissions":
                if _text(fields["event-type"], ids) != "CommandBufferSubmission":
                    continue
                start = _number(fields["start"], ids)
                duration = _number(fields["duration"], ids)
                if duration < 0:
                    raise ValueError("negative duration")
                row = {"time": start, "commit": start + duration,
                       "key": (_text(fields["gpu"], ids), _number(fields["cmdbuffer-id"], ids))}
            elif kind == "allocations":
                if _text(fields["event-type"], ids) != "Allocation":
                    continue
                row = {"time": _number(fields["timestamp"], ids), "bytes": _number(fields["resource-size"], ids)}
                if row["bytes"] < 0:
                    raise ValueError("negative allocation size")
            else:
                row = {"time": _number(fields["start"], ids), "duration": _number(fields["duration"], ids)}
                if row["duration"] < 0:
                    raise ValueError("negative compiler duration")
            if row["time"] < 0:
                raise ValueError("negative timestamp")
            parsed.append(row)
        except (ValueError, TypeError, KeyError, AttributeError):
            invalid += 1
    return {"status": "partial" if invalid else "available", "invalid_rows": invalid,
            "table_rows": len(rows), "target_rows": len(parsed)}, parsed


def _compute(content, pid):
    rows, ids = _rows(content, "metal-gpu-intervals",
                      ("start", "duration", "process", "channel-name", "state", "gpu", "cmdbuffer-id"))
    result = []
    for fields in rows:
        if _text(fields["channel-name"], ids) != "Compute" or _text(fields["state"], ids) != "Active":
            continue
        if _pid(fields["process"], ids) != pid:
            continue
        start = _number(fields["start"], ids)
        duration = _number(fields["duration"], ids)
        if start < 0 or duration < 0:
            raise ValueError("negative compute interval")
        result.append((start, start + duration,
                       (_text(fields["gpu"], ids), _number(fields["cmdbuffer-id"], ids))))
    return sorted(result)


def _gaps(intervals, start, end):
    cursor = start
    result = []
    for left, right, key in intervals:
        if right <= start or left >= end:
            continue
        left, right = max(start, left), min(end, right)
        if left > cursor:
            result.append((cursor, left, key))
        cursor = max(cursor, right)
    if cursor < end:
        result.append((cursor, end, None))
    return result


def analyze_detail(compute_xml, table_contents, *, pid, windows, main_tid=None):
    """Return interval-aligned observations without claiming a causal partition."""
    result = {"status": "available", "tables": {}, "windows": [], "limitations": [
        "Stack categories are sampled activity, not milliseconds saved or proof of causation.",
        "Creation-to-commit intervals include waits and application work; they are not encoder CPU time.",
        "Post-commit gaps can contain GPU dependencies or OS/driver delays; ready-to-run state is not established.",
        "Allocation events count creations, not allocation latency or reclamation. Empty tables do not prove absence.",
    ]}
    try:
        intervals = _compute(compute_xml, pid)
    except (ET.ParseError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return {**result, "status": "unavailable", "reason": str(exc)}
    parsed = {}
    for name in TABLES:
        try:
            if name not in table_contents:
                raise ValueError("table export unavailable")
            result["tables"][name], parsed[name] = _parse(table_contents[name], name, pid)
        except (ET.ParseError, ValueError, TypeError, KeyError, AttributeError) as exc:
            result["tables"][name] = {"status": "unavailable", "reason": str(exc)}
            parsed[name] = []
    labelled = {row["tid"] for row in parsed["cpu"] if row["main_label"]}
    if main_tid is None and len(labelled) == 1:
        main_tid = labelled.pop()
    result["main_thread_id"] = main_tid
    commits = {row["key"]: row["commit"] for row in parsed["submissions"]}
    for window in windows:
        if not window.get("label", "").startswith("decode-") or window.get("status") != "complete":
            continue
        start, end = window["start_trace_ns"], window["end_trace_ns"]
        gaps = _gaps(intervals, start, end)
        gap_starts = [left for left, _, _ in gaps]
        before_commit = after_commit = unmatched = 0
        for left, right, key in gaps:
            commit = commits.get(key)
            if commit is None or commit > right:
                unmatched += right - left
            else:
                before_commit += max(0, commit - left)
                after_commit += right - max(left, commit)
        samples = [row for row in parsed["cpu"] if start <= row["time"] < end]
        gap_samples = []
        for row in samples:
            index = bisect_right(gap_starts, row["time"]) - 1
            if index >= 0 and row["time"] < gaps[index][1]:
                gap_samples.append(row)
        main_samples = [row for row in gap_samples if row["tid"] == main_tid]
        allocations = [row for row in parsed["allocations"] if start <= row["time"] < end]
        compilers = [row for row in parsed["compiler"] if row["time"] < end and row["time"] + row["duration"] > start]
        result["windows"].append({
            "label": window["label"], "gap_ms": sum(right - left for left, right, _ in gaps) / 1e6,
            "before_next_command_commit_ms": before_commit / 1e6,
            "after_next_command_commit_ms": after_commit / 1e6,
            "unmatched_gap_ms": unmatched / 1e6,
            "cpu_samples": len(samples), "gap_cpu_samples": len(gap_samples),
            "main_thread_gap_samples": len(main_samples),
            "main_thread_gap_categories": dict(Counter(row["family"] for row in main_samples)),
            "all_thread_gap_categories": dict(Counter(row["family"] for row in gap_samples)),
            "sampled_thread_states": dict(Counter(row["state"] for row in gap_samples)),
            "allocation_events": len(allocations), "allocation_bytes": sum(row["bytes"] for row in allocations),
            "compiler_events": len(compilers),
        })
    if not result["windows"]:
        result.update(status="unavailable", reason="no complete decode windows")
    elif any(table["status"] != "available" for table in result["tables"].values()):
        result["status"] = "partial"
    return result


def analyze_detail_files(compute_path, table_dir, *, pid, windows, main_tid=None):
    contents = {}
    for name in TABLES:
        try:
            contents[name] = _read_xml_bytes(Path(table_dir) / (name + ".xml"))
        except (OSError, ValueError):
            pass
    try:
        compute = _read_xml_bytes(Path(compute_path))
    except (OSError, ValueError) as exc:
        return {"status": "unavailable", "reason": str(exc)}
    return analyze_detail(compute, contents, pid=pid, windows=windows, main_tid=main_tid)
