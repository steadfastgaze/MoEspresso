"""Road-test script table: structure and phase invariants."""

from __future__ import annotations

from moespresso.agentlib.roadtest.fixture import generate_fixture
from moespresso.agentlib.roadtest.script import build_script


def _script(tmp_path):
    return build_script(generate_fixture(tmp_path / "ws"))


def test_turn_ids_are_unique(tmp_path):
    script = _script(tmp_path)
    ids = [turn.turn_id for turn in script.all_turns()]
    assert len(ids) == len(set(ids))


def test_sessions_are_labeled(tmp_path):
    script = _script(tmp_path)
    assert all(t.session == "a" for t in script.opening_a)
    assert all(t.session == "a" for t in script.growth_a)
    assert all(t.session == "a" for t in script.extensions_a)
    assert script.probe_a.session == "a"
    assert script.wrap_a.session == "a"
    assert script.resume_b.session == "b"
    interleave_sessions = [t.session for t in script.interleaved]
    assert "a" in interleave_sessions and "b" in interleave_sessions


def test_probe_turn_embeds_the_reference_text(tmp_path):
    fixture = generate_fixture(tmp_path / "ws")
    script = build_script(fixture)
    assert fixture.large_reference_text in script.probe_a.user_text
    assert "Without using any tools" in script.probe_a.user_text


def test_read_turns_reference_existing_files(tmp_path):
    fixture = generate_fixture(tmp_path / "ws")
    script = build_script(fixture)
    known = {info.relpath for info in fixture.data_files}
    for turn in script.all_turns():
        for relpath in known:
            if relpath in turn.user_text:
                assert (fixture.root / relpath).is_file()


def test_data_segments_are_not_read_twice_in_one_session(tmp_path):
    fixture = generate_fixture(tmp_path / "ws")
    script = build_script(fixture)
    per_session: dict[str, list[str]] = {"a": [], "b": []}
    for turn in script.all_turns():
        for info in fixture.data_files:
            if info.relpath in turn.user_text:
                per_session[turn.session].append(info.relpath)
    for session, reads in per_session.items():
        assert len(reads) == len(set(reads)), session


def test_extension_reserve_exists(tmp_path):
    script = _script(tmp_path)
    assert len(script.extensions_a) >= 3


def test_subagent_scenario_fields(tmp_path):
    fixture = generate_fixture(tmp_path / "ws")
    script = build_script(fixture)
    assert script.delegate_a.session == "a"
    assert script.followup_a.session == "a"
    # The brief pushes the child through at least two tool actions, and the
    # file it names really exists in the fixture.
    assert "grep tool" in script.subagent_task
    assert "read_file tool" in script.subagent_task
    named = [info.relpath for info in fixture.data_files
             if info.relpath in script.subagent_task]
    assert len(named) == 1
    assert (fixture.root / named[0]).is_file()
    assert script.subagent_context
