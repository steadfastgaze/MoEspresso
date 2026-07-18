"""Deterministic fixture workspace for the road-test.

The road-test agent operates on a generated project, never a real repository,
so every run sees byte-identical files: planted marker strings for grep, a
known defect to read and fix with the edit tool, and data segments large
enough that reading them accumulates real context. All content derives from a
fixed linear congruential generator, so regeneration is reproducible and the
per-file facts (line counts, error-record counts) are stable test anchors.

The large reference file exists for the restart-probe turn: its text is
pasted into a user message so that single turn adds more prompt tokens than
one disk-checkpoint stride, which guarantees the turn's prefill crosses at
least one frontier before the server restart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Sized so the scripted session's extension reserve can grow the main
# session well past a 110k-token context target: each segment read adds
# roughly 3.5k to 4k prompt tokens. Runs with a deeper context target pass
# a larger count to ``generate_fixture``; segment content depends only on
# the segment index, so a larger fixture is a superset of a smaller one.
DATA_FILE_COUNT = 32
DATA_LINES = 240
LARGE_REFERENCE_LINES = 220

MARKER_ONE = "kiwi-lantern-042"
MARKER_TWO = "ember-quartz-777"
MARKER_REFERENCE = "cobalt-meridian-118"

MARKER_ONE_FILE = "src/module_golf.py"
MARKER_TWO_FILE = "src/module_kilo.py"

DEFECT_FILE = "src/metrics.py"
DEFECT_LINE = "    return total / (len(values) - 1)"
DEFECT_FIX_LINE = "    return total / len(values)"

_MODULE_NAMES = (
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliett", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango",
)

_ZONES = ("delta", "echo", "harbor", "ridge", "basin", "mesa", "summit", "gorge")
_NOTES = (
    "coolant pump within tolerance",
    "sensor drift within tolerance window",
    "voltage ripple observed on rail two",
    "scheduled calibration completed",
    "filter housing nominal after purge",
    "telemetry burst deferred to next window",
    "bearing temperature steady under load",
    "manifold pressure recovered after vent",
)


def _lcg_stream(seed: int):
    """Deterministic 31-bit linear congruential generator stream."""
    state = (seed * 2654435761 + 1013904223) & 0x7FFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state


@dataclass(frozen=True)
class DataFileInfo:
    """Facts about one generated data segment."""

    relpath: str
    lines: int
    error_lines: int


@dataclass(frozen=True)
class Fixture:
    """The generated workspace layout plus its planted facts."""

    root: Path
    source_files: tuple[str, ...]
    data_files: tuple[DataFileInfo, ...]
    large_reference: str
    large_reference_text: str
    markers: dict[str, str]
    defect_file: str
    defect_line: str
    defect_fix_line: str


def data_file_text(index: int, lines: int = DATA_LINES) -> tuple[str, int]:
    """The full text of data segment ``index`` plus its error-record count."""
    rng = _lcg_stream(1000 + index)
    out = []
    errors = 0
    for line_no in range(1, lines + 1):
        r1, r2, r3, r4 = next(rng), next(rng), next(rng), next(rng)
        status = "error" if r2 % 17 == 0 else "ok"
        if status == "error":
            errors += 1
        note = _NOTES[r4 % len(_NOTES)]
        out.append(
            f"record {index:02d}{line_no:04d} zone {_ZONES[r1 % len(_ZONES)]} "
            f"shard {r1 % 24:02d} checksum {r3:08x} status {status} "
            f'latency_ms {r2 % 2000:04d} note "{note}"'
        )
    return "\n".join(out) + "\n", errors


def large_reference_text() -> str:
    """The reference document pasted into the restart-probe user turn."""
    rng = _lcg_stream(77)
    out = [
        "reference log: consolidated field readings",
        f"trace tag of record: {MARKER_REFERENCE}",
        "",
    ]
    for line_no in range(1, LARGE_REFERENCE_LINES + 1):
        r1, r2, r3 = next(rng), next(rng), next(rng)
        flag = "REVIEW" if r2 % 11 == 0 else "logged"
        note = _NOTES[r3 % len(_NOTES)]
        out.append(
            f"reading {line_no:04d} station {_ZONES[r1 % len(_ZONES)]}-{r1 % 40:02d} "
            f"value {r2 % 100000:05d} flag {flag} note \"{note}\""
        )
    return "\n".join(out) + "\n"


def large_reference_review_count() -> int:
    """The number of REVIEW-flagged lines in the reference document."""
    return sum(
        1 for line in large_reference_text().splitlines() if " flag REVIEW " in line
    )


def _module_text(name: str, index: int) -> str:
    rng = _lcg_stream(2000 + index)
    lines = [
        f'"""Utility transforms for the {name} channel."""',
        "",
    ]
    if name == "golf":
        lines.append(f'TRACE_TAG = "{MARKER_ONE}"')
        lines.append("")
    if name == "kilo":
        lines.append(f'AUDIT_TAG = "{MARKER_TWO}"')
        lines.append("")
    for fn in range(4):
        modulus = 89 + (next(rng) % 31)
        offset = next(rng) % 13
        lines.extend(
            [
                f"def transform_{name}_{fn}(value):",
                f'    """Fold a raw {name} reading into bucket {fn}."""',
                f"    return ((value + {offset}) % {modulus}) / {modulus}.0",
                "",
                "",
            ]
        )
    return "\n".join(lines).rstrip("\n") + "\n"


def _metrics_text() -> str:
    return (
        '"""Aggregate statistics over channel readings."""\n'
        "\n"
        "\n"
        "def mean(values):\n"
        '    """Average of a numeric sequence."""\n'
        "    if not values:\n"
        '        raise ValueError("mean of an empty sequence")\n'
        "    total = 0.0\n"
        "    for value in values:\n"
        "        total += value\n"
        f"{DEFECT_LINE}\n"
        "\n"
        "\n"
        "def spread(values):\n"
        '    """Peak-to-peak spread of a numeric sequence."""\n'
        "    if not values:\n"
        '        raise ValueError("spread of an empty sequence")\n'
        "    return max(values) - min(values)\n"
    )


def _readme_text() -> str:
    return (
        "# Field telemetry fixture\n"
        "\n"
        "A small telemetry-processing project used as an agent workspace.\n"
        "Source transforms live under `src/`; captured data segments live\n"
        "under `data/` as plain-text record logs.\n"
    )


def generate_fixture(root: Path, *,
                     data_file_count: int = DATA_FILE_COUNT) -> Fixture:
    """Write the fixture workspace under ``root`` and return its layout."""
    if data_file_count < 18:
        # The script assigns segments 15 and 16 to the second session and
        # needs at least one extension segment past them.
        raise ValueError("data_file_count must be at least 18")
    root = Path(root)
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)

    (root / "README.md").write_text(_readme_text(), encoding="utf-8")

    source_files = []
    for index, name in enumerate(_MODULE_NAMES):
        relpath = f"src/module_{name}.py"
        (root / relpath).write_text(_module_text(name, index), encoding="utf-8")
        source_files.append(relpath)
    (root / DEFECT_FILE).write_text(_metrics_text(), encoding="utf-8")
    source_files.append(DEFECT_FILE)

    data_files = []
    for index in range(1, data_file_count + 1):
        relpath = f"data/segment_{index:02d}.txt"
        text, errors = data_file_text(index)
        (root / relpath).write_text(text, encoding="utf-8")
        data_files.append(
            DataFileInfo(relpath=relpath, lines=DATA_LINES, error_lines=errors)
        )

    reference_text = large_reference_text()
    reference_relpath = "data/large_reference.txt"
    (root / reference_relpath).write_text(reference_text, encoding="utf-8")

    return Fixture(
        root=root,
        source_files=tuple(source_files),
        data_files=tuple(data_files),
        large_reference=reference_relpath,
        large_reference_text=reference_text,
        markers={
            MARKER_ONE: MARKER_ONE_FILE,
            MARKER_TWO: MARKER_TWO_FILE,
            MARKER_REFERENCE: reference_relpath,
        },
        defect_file=DEFECT_FILE,
        defect_line=DEFECT_LINE,
        defect_fix_line=DEFECT_FIX_LINE,
    )
