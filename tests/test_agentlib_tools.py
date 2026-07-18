"""The core tool registry, the four tool implementations, and the choke point.

Every execution goes through execute_tool_call against a temp working
directory; nothing here touches a model or the GPU. Failures come back as
ToolResult(ok=False, ...) with a message the model can act on; the choke
point does not raise.
"""

from __future__ import annotations

import pytest

from moespresso.agentlib import (
    ToolCall,
    ToolSpec,
    build_core_registry,
    execute_tool_call,
)
from moespresso.agentlib.tools import GREP_MAX_MATCHES


@pytest.fixture()
def registry():
    return build_core_registry()


def _run(registry, workdir, name, **arguments):
    return execute_tool_call(registry, ToolCall(name=name, arguments=arguments),
                             workdir=workdir)


# --- registry ---

def test_core_registry_has_exactly_the_four_tools(registry):
    assert registry.names() == ["read_file", "grep", "edit", "bash"]


def test_openai_tools_shape(registry):
    tools = registry.openai_tools()
    assert len(tools) == 4
    for tool in tools:
        assert tool["type"] == "function"
        fn = tool["function"]
        assert fn["name"] and fn["description"]
        schema = fn["parameters"]
        assert schema["type"] == "object"
        assert set(schema.get("required", [])) <= set(schema["properties"])


def test_duplicate_registration_rejected(registry):
    spec = ToolSpec(name="read_file", description="dup",
                    parameters={"type": "object", "properties": {}})
    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec, lambda arguments, workdir: "")


def test_unknown_tool_lookup_raises(registry):
    with pytest.raises(KeyError):
        registry.spec("nope")
    with pytest.raises(KeyError):
        registry.executor("nope")


# --- choke point ---

def test_unknown_tool_is_a_failed_result_not_an_exception(registry, tmp_path):
    result = _run(registry, tmp_path, "browse", url="http://x")
    assert not result.ok
    assert "unknown tool" in result.output
    assert "read_file" in result.output  # the available tools are listed


def test_missing_required_argument_is_a_failed_result(registry, tmp_path):
    result = _run(registry, tmp_path, "grep")
    assert not result.ok
    assert "missing required argument" in result.output
    assert "pattern" in result.output


def test_unknown_argument_name_is_a_failed_result(registry, tmp_path):
    result = _run(registry, tmp_path, "grep", pattern="x", flle="typo.py")
    assert not result.ok
    assert "unknown argument" in result.output
    assert "flle" in result.output
    assert "glob" in result.output  # the allowed names are listed


def test_executor_valueerror_becomes_failed_result(registry, tmp_path):
    result = _run(registry, tmp_path, "read_file", path="missing.txt")
    assert not result.ok
    assert "read_file" in result.output
    assert "no such file" in result.output


def test_wrong_typed_argument_value_becomes_failed_result(registry, tmp_path):
    # Argument values come from an untrusted model; a wrong type must come
    # back as a failed result, never escape the choke point as an exception.
    result = _run(registry, tmp_path, "read_file", path=123)
    assert not result.ok
    assert "read_file" in result.output


# --- read_file ---

def test_read_file_relative_path(registry, tmp_path):
    (tmp_path / "a.txt").write_text("line 1\nline 2\n")
    result = _run(registry, tmp_path, "read_file", path="a.txt")
    assert result.ok
    assert result.output == "line 1\nline 2\n"


def test_read_file_absolute_path(registry, tmp_path):
    target = tmp_path / "b.txt"
    target.write_text("content")
    result = _run(registry, tmp_path, "read_file", path=str(target))
    assert result.ok
    assert result.output == "content"


def test_read_file_offset_and_limit(registry, tmp_path):
    (tmp_path / "n.txt").write_text("one\ntwo\nthree\nfour\n")
    result = _run(registry, tmp_path, "read_file", path="n.txt", offset=2, limit=2)
    assert result.ok
    assert result.output == "two\nthree\n"


def test_read_file_offset_only_reads_to_end(registry, tmp_path):
    (tmp_path / "n.txt").write_text("one\ntwo\nthree\n")
    result = _run(registry, tmp_path, "read_file", path="n.txt", offset=3)
    assert result.ok
    assert result.output == "three\n"


def test_read_file_offset_past_end_fails(registry, tmp_path):
    (tmp_path / "n.txt").write_text("one\n")
    result = _run(registry, tmp_path, "read_file", path="n.txt", offset=5)
    assert not result.ok
    assert "past the end" in result.output


def test_read_file_zero_offset_fails(registry, tmp_path):
    (tmp_path / "n.txt").write_text("one\n")
    result = _run(registry, tmp_path, "read_file", path="n.txt", offset=0)
    assert not result.ok
    assert "1-based" in result.output


def test_read_file_directory_fails(registry, tmp_path):
    result = _run(registry, tmp_path, "read_file", path=".")
    assert not result.ok
    assert "no such file" in result.output


# --- grep ---

def test_grep_reports_relative_path_line_and_text(registry, tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text("def alpha():\n    pass\ndef beta():\n")
    (tmp_path / "top.py").write_text("x = 1\ndef gamma():\n")
    result = _run(registry, tmp_path, "grep", pattern=r"^def ")
    assert result.ok
    assert result.output.splitlines() == [
        "pkg/m.py:1:def alpha():",
        "pkg/m.py:3:def beta():",
        "top.py:2:def gamma():",
    ]


def test_grep_glob_filter(registry, tmp_path):
    (tmp_path / "a.py").write_text("needle\n")
    (tmp_path / "a.txt").write_text("needle\n")
    result = _run(registry, tmp_path, "grep", pattern="needle", glob="*.py")
    assert result.ok
    assert result.output == "a.py:1:needle"


def test_grep_single_file_path(registry, tmp_path):
    (tmp_path / "one.txt").write_text("hit\nmiss\nhit\n")
    result = _run(registry, tmp_path, "grep", pattern="hit", path="one.txt")
    assert result.ok
    assert result.output.splitlines() == ["one.txt:1:hit", "one.txt:3:hit"]


def test_grep_no_matches(registry, tmp_path):
    (tmp_path / "a.txt").write_text("nothing here\n")
    result = _run(registry, tmp_path, "grep", pattern="absent_token")
    assert result.ok
    assert result.output == "no matches"


def test_grep_invalid_regex_fails(registry, tmp_path):
    result = _run(registry, tmp_path, "grep", pattern="(unclosed")
    assert not result.ok
    assert "invalid regex" in result.output


def test_grep_missing_path_fails(registry, tmp_path):
    result = _run(registry, tmp_path, "grep", pattern="x", path="nowhere/")
    assert not result.ok
    assert "no such path" in result.output


def test_grep_skips_binary_files(registry, tmp_path):
    (tmp_path / "blob.bin").write_bytes(b"needle\x00\xff\xfe")
    (tmp_path / "a.txt").write_text("needle\n")
    result = _run(registry, tmp_path, "grep", pattern="needle")
    assert result.ok
    assert result.output == "a.txt:1:needle"


def test_grep_skips_git_directory(registry, tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("needle\n")
    (tmp_path / "a.txt").write_text("needle\n")
    result = _run(registry, tmp_path, "grep", pattern="needle")
    assert result.ok
    assert result.output == "a.txt:1:needle"


def test_grep_truncates_at_the_match_cap(registry, tmp_path):
    (tmp_path / "big.txt").write_text("hit\n" * (GREP_MAX_MATCHES + 5))
    result = _run(registry, tmp_path, "grep", pattern="hit")
    assert result.ok
    lines = result.output.splitlines()
    assert len(lines) == GREP_MAX_MATCHES + 1
    assert lines[-1] == f"... truncated at {GREP_MAX_MATCHES} matches"


# --- edit ---

def test_edit_unique_replace(registry, tmp_path):
    target = tmp_path / "f.py"
    target.write_text("value = old\nother = keep\n")
    result = _run(registry, tmp_path, "edit", path="f.py",
                  old_string="value = old", new_string="value = new")
    assert result.ok
    assert "replaced 1 occurrence(s)" in result.output
    assert target.read_text() == "value = new\nother = keep\n"


def test_edit_missing_anchor_fails_loudly_and_leaves_file_alone(registry, tmp_path):
    target = tmp_path / "f.py"
    target.write_text("content\n")
    result = _run(registry, tmp_path, "edit", path="f.py",
                  old_string="not present", new_string="x")
    assert not result.ok
    assert "not found" in result.output
    assert target.read_text() == "content\n"


def test_edit_ambiguous_anchor_fails(registry, tmp_path):
    target = tmp_path / "f.py"
    target.write_text("dup\ndup\n")
    result = _run(registry, tmp_path, "edit", path="f.py",
                  old_string="dup", new_string="uniq")
    assert not result.ok
    assert "2 times" in result.output
    assert target.read_text() == "dup\ndup\n"


def test_edit_replace_all(registry, tmp_path):
    target = tmp_path / "f.py"
    target.write_text("dup\ndup\ndup\n")
    result = _run(registry, tmp_path, "edit", path="f.py",
                  old_string="dup", new_string="uniq", replace_all=True)
    assert result.ok
    assert "replaced 3 occurrence(s)" in result.output
    assert target.read_text() == "uniq\nuniq\nuniq\n"


def test_edit_identical_strings_fail(registry, tmp_path):
    (tmp_path / "f.py").write_text("same\n")
    result = _run(registry, tmp_path, "edit", path="f.py",
                  old_string="same", new_string="same")
    assert not result.ok
    assert "identical" in result.output


def test_edit_empty_old_string_fails(registry, tmp_path):
    (tmp_path / "f.py").write_text("x\n")
    result = _run(registry, tmp_path, "edit", path="f.py",
                  old_string="", new_string="y")
    assert not result.ok
    assert "non-empty" in result.output


def test_edit_missing_file_fails(registry, tmp_path):
    result = _run(registry, tmp_path, "edit", path="ghost.py",
                  old_string="a", new_string="b")
    assert not result.ok
    assert "no such file" in result.output


# --- bash ---

def test_bash_captures_stdout(registry, tmp_path):
    result = _run(registry, tmp_path, "bash", command="echo hello")
    assert result.ok
    assert result.output == "hello"


def test_bash_runs_in_the_working_directory(registry, tmp_path):
    result = _run(registry, tmp_path, "bash", command="pwd")
    assert result.ok
    # macOS tmp paths may resolve through /private; compare resolved paths.
    import pathlib
    assert pathlib.Path(result.output).resolve() == tmp_path.resolve()


def test_bash_nonzero_exit_is_a_failed_result_with_the_code(registry, tmp_path):
    result = _run(registry, tmp_path, "bash",
                  command="echo out; echo err >&2; exit 3")
    assert not result.ok
    assert "exit code 3" in result.output
    assert "out" in result.output
    assert "err" in result.output


def test_bash_captures_stderr_on_success(registry, tmp_path):
    result = _run(registry, tmp_path, "bash", command="echo warn >&2")
    assert result.ok
    assert "warn" in result.output


def test_bash_timeout_is_a_failed_result(registry, tmp_path):
    result = _run(registry, tmp_path, "bash", command="sleep 5", timeout=0.5)
    assert not result.ok
    assert "timed out" in result.output


def test_bash_empty_command_fails(registry, tmp_path):
    result = _run(registry, tmp_path, "bash", command="   ")
    assert not result.ok
    assert "non-empty" in result.output


def test_bash_writes_land_in_the_workdir(registry, tmp_path):
    result = _run(registry, tmp_path, "bash", command="printf data > made.txt")
    assert result.ok
    assert (tmp_path / "made.txt").read_text() == "data"
