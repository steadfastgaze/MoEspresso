import xml.etree.ElementTree as ET

from moespresso.runtime import decode_trace_detail as detail


def table(schema, columns, rows):
    root = ET.Element("trace-query-result")
    node = ET.SubElement(root, "node")
    declaration = ET.SubElement(node, "schema", name=schema)
    for name in columns:
        ET.SubElement(ET.SubElement(declaration, "col"), "mnemonic").text = name
    for values in rows:
        row = ET.SubElement(node, "row")
        for value in values:
            row.append(ET.fromstring(value))
    return ET.tostring(root)


def number(value):
    return f"<n>{value}</n>"


def text(value):
    return f"<s>{value}</s>"


PID = "<process><pid>42</pid></process>"
THREAD = '<thread fmt="Main Thread (1)"><tid>1</tid></thread>'
WINDOW = {"label": "decode-0", "status": "complete", "start_trace_ns": 0, "end_trace_ns": 1000}


def compute():
    columns = ("start", "duration", "process", "channel-name", "state", "gpu", "cmdbuffer-id")
    rows = [[number(start), number(100), PID, text("Compute"), text("Active"), text("GPU"), number(key)]
            for start, key in ((100, 1), (400, 2), (800, 3))]
    return table("metal-gpu-intervals", columns, rows)


def contents():
    return {
        "submissions": table(detail.TABLES["submissions"],
                             ("start", "duration", "process", "gpu", "cmdbuffer-id", "event-type"),
                             [[number(0), number(end), PID, text("GPU"), number(key), text("CommandBufferSubmission")]
                              for end, key in ((50, 1), (300, 2), (450, 3))]),
        "cpu": table(detail.TABLES["cpu"], ("time", "process", "thread", "thread-state", "stack"), [
            [number(250), PID, THREAD, text("Running"), '<stack><backtrace><frame name="mlx::core::eval_impl"/></backtrace></stack>'],
            [number(450), PID, THREAD, text("Running"), '<stack><backtrace><frame name="PyEval"/></backtrace></stack>'],
        ]),
        "allocations": table(detail.TABLES["allocations"], ("timestamp", "process", "resource-size", "event-type"),
                             [[number(500), PID, number(128), text("Allocation")]]),
        "compiler": table(detail.TABLES["compiler"], ("start", "duration", "process"), []),
    }


def test_compute_gap_join_does_not_double_count_overlaps_or_invent_readiness():
    result = detail.analyze_detail(compute(), contents(), pid=42, windows=[WINDOW])
    assert result["status"] == "available"
    row = result["windows"][0]
    assert row["gap_ms"] == 700 / 1e6
    assert row["before_next_command_commit_ms"] == 150 / 1e6
    assert row["after_next_command_commit_ms"] == 450 / 1e6
    assert row["unmatched_gap_ms"] == 100 / 1e6
    assert row["cpu_samples"] == 2
    assert row["gap_cpu_samples"] == 1
    assert row["main_thread_gap_categories"] == {"submission": 1}
    assert row["allocation_events"] == 1
    assert row["allocation_bytes"] == 128
    assert row["compiler_events"] == 0


def test_missing_cpu_export_remains_unavailable():
    data = contents()
    del data["cpu"]
    result = detail.analyze_detail(compute(), data, pid=42, windows=[WINDOW])
    assert result["status"] == "partial"
    assert result["tables"]["cpu"]["status"] == "unavailable"
    assert result["main_thread_id"] is None


def test_nearest_recognized_native_frame_does_not_become_python():
    assert detail.classify_stack(["memcpy", "mlx::core::Reduce::eval_gpu", "PyEval"]) == "submission"
    assert detail.classify_stack(["pread", "PyEval"]) == "data"
    assert detail.classify_stack(["MetalAllocator::malloc", "::eval_gpu", "PyEval"]) == "allocation_compile"
    assert detail.classify_stack(["semaphore_wait_trap", "PyEval"]) == "coordination"
    assert detail.classify_stack(["unresolved frame"]) == "unclassified"


def test_malformed_sample_does_not_support_ranking():
    data = contents()
    data["cpu"] = data["cpu"].replace(b"<tid>1</tid>", b"<tid>invalid</tid>")
    result = detail.analyze_detail(compute(), data, pid=42, windows=[WINDOW])
    assert result["tables"]["cpu"]["status"] == "partial"
    assert result["tables"]["cpu"]["invalid_rows"] == 2


def test_unmatched_commit_is_not_assumed_to_be_ready():
    data = contents()
    data["submissions"] = data["submissions"].replace(b"<n>300</n>", b"<n>10000</n>")
    result = detail.analyze_detail(compute(), data, pid=42, windows=[WINDOW])
    assert result["windows"][0]["unmatched_gap_ms"] == 300 / 1e6


def test_other_process_samples_and_allocations_are_excluded():
    data = {name: value.replace(b"<pid>42</pid>", b"<pid>7</pid>") for name, value in contents().items()}
    result = detail.analyze_detail(compute(), data, pid=42, windows=[WINDOW])
    assert result["windows"][0]["cpu_samples"] == 0
    assert result["windows"][0]["allocation_events"] == 0


def test_unsupported_compute_schema_fails_closed():
    result = detail.analyze_detail(b"<trace/>", contents(), pid=42, windows=[WINDOW])
    assert result["status"] == "unavailable"
