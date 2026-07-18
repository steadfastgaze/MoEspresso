"""Road-test fixture workspace: determinism and planted facts.

The fixture is the ground the road-test agent walks on, so its facts must be
stable: regeneration is byte-identical, the planted markers and the defect
anchor sit where the script expects them, and the per-file error counts the
scripted questions ask about match the generated content.
"""

from __future__ import annotations

from moespresso.agentlib.roadtest.fixture import (
    DATA_FILE_COUNT,
    DATA_LINES,
    DEFECT_LINE,
    MARKER_ONE,
    MARKER_ONE_FILE,
    MARKER_REFERENCE,
    MARKER_TWO,
    MARKER_TWO_FILE,
    data_file_text,
    generate_fixture,
    large_reference_review_count,
    large_reference_text,
)


def _tree_bytes(root):
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = path.read_bytes()
    return out


def test_generation_is_deterministic(tmp_path):
    first = generate_fixture(tmp_path / "one")
    second = generate_fixture(tmp_path / "two")
    assert _tree_bytes(first.root) == _tree_bytes(second.root)


def test_layout_and_planted_markers(tmp_path):
    fixture = generate_fixture(tmp_path / "ws")
    assert len(fixture.data_files) == DATA_FILE_COUNT
    assert MARKER_ONE in (fixture.root / MARKER_ONE_FILE).read_text()
    assert MARKER_TWO in (fixture.root / MARKER_TWO_FILE).read_text()
    assert MARKER_REFERENCE in fixture.large_reference_text
    # Each marker lives in exactly one file so grep answers are unambiguous.
    for marker in (MARKER_ONE, MARKER_TWO):
        hits = [
            path for path in fixture.root.rglob("*")
            if path.is_file() and marker in path.read_text()
        ]
        assert len(hits) == 1, marker


def test_defect_anchor_present_exactly_once(tmp_path):
    fixture = generate_fixture(tmp_path / "ws")
    text = (fixture.root / fixture.defect_file).read_text()
    assert text.count(DEFECT_LINE) == 1
    assert fixture.defect_fix_line not in text


def test_data_file_error_counts_match_content(tmp_path):
    fixture = generate_fixture(tmp_path / "ws")
    for info in fixture.data_files:
        text = (fixture.root / info.relpath).read_text()
        lines = text.splitlines()
        assert len(lines) == DATA_LINES
        assert info.error_lines == sum(
            1 for line in lines if " status error " in line)
        assert info.error_lines > 0


def test_data_file_text_matches_written_file(tmp_path):
    fixture = generate_fixture(tmp_path / "ws")
    text, errors = data_file_text(3)
    assert (fixture.root / "data/segment_03.txt").read_text() == text
    assert errors == fixture.data_files[2].error_lines


def test_large_reference_is_probe_sized():
    # The probe turn pastes this text into one user message; the turn must
    # add more prompt tokens than the default checkpoint stride (4096), so
    # the text needs a healthy character margin over it.
    text = large_reference_text()
    assert len(text) > 17_000
    assert large_reference_review_count() > 0
    assert " flag REVIEW " in text
