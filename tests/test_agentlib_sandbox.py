"""The sandbox policy engine and its gate in the execution choke point.

The pure logic (rule evaluation, config parsing, profile generation, path
scope) is covered table-driven with no sandbox involved. Hook tests run
tiny /bin/sh scripts written at test time. The integration section invokes
the real macOS sandbox-exec with a generated profile on tmp_path and skips
cleanly where Seatbelt is unavailable.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
from pathlib import Path

import pytest

from moespresso.agentlib import (
    Decision,
    HookRule,
    PolicyConfigError,
    RegexRule,
    SandboxPolicy,
    ToolCall,
    ToolSpec,
    build_core_registry,
    build_policy,
    evaluate_command,
    execute_tool_call,
    generate_profile,
    load_policy,
    path_scope_problem,
    sandboxed_command,
)
from moespresso.agentlib import sandbox as sandbox_module
from moespresso.agentlib.sandbox import DEFAULT_RULES

requires_seatbelt = pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("sandbox-exec") is None,
    reason="needs macOS sandbox-exec",
)


@pytest.fixture()
def registry():
    return build_core_registry()


@pytest.fixture()
def ws(tmp_path):
    path = tmp_path / "ws"
    path.mkdir()
    return path


@pytest.fixture()
def outside(tmp_path):
    path = tmp_path / "outside"
    path.mkdir()
    return path


def _decide(policy, command, cwd):
    # Unit tests pass an explicit empty env so hooks never see the test
    # runner's environment.
    return evaluate_command(policy, command, cwd=cwd, env={})


def _hook(tmp_path, body, name="hook.sh"):
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return str(path)


def _run(registry, workdir, policy, name, **arguments):
    call = ToolCall(name=name, arguments=arguments)
    return execute_tool_call(registry, call, workdir=workdir, policy=policy)


# --- evaluate: defaults and the shipped git rule ---

@pytest.mark.parametrize("command,expected", [
    ("git status", Decision.ALLOW_UNSANDBOXED),
    ("git", Decision.ALLOW_UNSANDBOXED),
    ("  git log --oneline  ", Decision.ALLOW_UNSANDBOXED),
    ("git config --global user.name x", Decision.ALLOW_UNSANDBOXED),
    ("legit status", Decision.ASK),      # substring of a command word
    ("gitk", Decision.ASK),              # git prefix of a longer word
    ("digit 5", Decision.ASK),
    ("echo x && git push", Decision.ASK),  # compound command is not a git call
    ("ls -la", Decision.ASK),            # unmatched falls to ask
])
def test_default_policy_git_rule_and_ask_fallthrough(ws, command, expected):
    policy = build_policy(ws)
    assert _decide(policy, command, ws).decision is expected


def test_unmatched_decision_names_its_source(ws):
    verdict = _decide(build_policy(ws), "ls", ws)
    assert verdict.decision is Decision.ASK
    assert verdict.source == "unmatched"


def test_default_policy_carries_exactly_the_shipped_rules(ws):
    assert build_policy(ws).rules == DEFAULT_RULES


# --- evaluate: configured regex rules ---

@pytest.mark.parametrize("decision", list(Decision))
def test_regex_rule_maps_to_each_disposition(ws, decision):
    policy = build_policy(ws, rules=(RegexRule(r"do .*", decision),))
    assert _decide(policy, "do thing", ws).decision is decision


def test_matched_regex_decision_names_the_pattern(ws):
    policy = build_policy(ws, rules=(RegexRule(r"rm( .*)?", Decision.DENY),))
    verdict = _decide(policy, "rm -rf x", ws)
    assert verdict.decision is Decision.DENY
    assert "rm( .*)?" in verdict.source


@pytest.mark.parametrize("pattern,command,expected", [
    # A bare word only matches the bare command under fullmatch; a rule that
    # should cover arguments needs an explicit tail.
    ("rm", "rm", Decision.DENY),
    ("rm", "rm -rf /", Decision.ASK),
    ("rm", "informal", Decision.ASK),
    (r"rm( .*)?", "rm -rf /", Decision.DENY),
    (r"rm( .*)?", "informal", Decision.ASK),
    (r"rm( .*)?", "grm x", Decision.ASK),
    # Broad matching is available, spelled deliberately.
    (r".*secret.*", "export secret=1", Decision.DENY),
])
def test_fullmatch_prevents_substring_matches(ws, pattern, command, expected):
    policy = build_policy(ws, rules=(RegexRule(pattern, Decision.DENY),))
    assert _decide(policy, command, ws).decision is expected


def test_first_matching_rule_wins(ws):
    policy = build_policy(ws, rules=(
        RegexRule(r"rm( .*)?", Decision.DENY),
        RegexRule(r"rm -rf scratch", Decision.ALLOW_UNSANDBOXED),
    ))
    assert _decide(policy, "rm -rf scratch", ws).decision is Decision.DENY


def test_configured_rule_overrides_the_default_git_rule(ws):
    policy = build_policy(ws, rules=(RegexRule(r"git push.*", Decision.DENY),))
    assert _decide(policy, "git push origin", ws).decision is Decision.DENY
    assert _decide(policy, "git status", ws).decision is Decision.ALLOW_UNSANDBOXED


def test_regex_rule_rejects_an_invalid_pattern_at_construction():
    with pytest.raises(PolicyConfigError, match="invalid pattern"):
        RegexRule("(unclosed", Decision.DENY)


# --- evaluate: executable hooks ---

@pytest.mark.parametrize("body,expected,fragment", [
    ("exit 0\n", Decision.ALLOW_SANDBOXED, ""),
    ("echo allow-sandboxed\nexit 0\n", Decision.ALLOW_SANDBOXED, ""),
    ("echo allow-unsandboxed\nexit 0\n", Decision.ALLOW_UNSANDBOXED, ""),
    ("echo approved!\nexit 0\n", Decision.DENY, "unrecognized output"),
    ("echo not on the list\nexit 1\n", Decision.DENY, "not on the list"),
    ("exit 1\n", Decision.DENY, "hook exited 1"),
    ("echo needs a human\nexit 2\n", Decision.ASK, "needs a human"),
    ("exit 3\n", Decision.DENY, "hook exited 3"),
])
def test_hook_exit_status_and_output_mapping(tmp_path, ws, body, expected, fragment):
    policy = build_policy(ws, rules=(HookRule(_hook(tmp_path, body)),))
    verdict = _decide(policy, "ls -la", ws)
    assert verdict.decision is expected
    assert fragment in verdict.reason


def test_missing_hook_denies(tmp_path, ws):
    policy = build_policy(ws, rules=(HookRule(str(tmp_path / "absent.sh")),))
    verdict = _decide(policy, "ls", ws)
    assert verdict.decision is Decision.DENY
    assert "hook did not run" in verdict.reason


def test_non_executable_hook_denies(tmp_path, ws):
    path = tmp_path / "noexec.sh"
    path.write_text("#!/bin/sh\nexit 0\n")
    policy = build_policy(ws, rules=(HookRule(str(path)),))
    verdict = _decide(policy, "ls", ws)
    assert verdict.decision is Decision.DENY
    assert "hook did not run" in verdict.reason


def test_hook_timeout_denies(tmp_path, ws, monkeypatch):
    monkeypatch.setattr(sandbox_module, "HOOK_TIMEOUT_SECONDS", 0.2)
    policy = build_policy(ws, rules=(HookRule(_hook(tmp_path, "sleep 5\n")),))
    verdict = _decide(policy, "ls", ws)
    assert verdict.decision is Decision.DENY
    assert "timed out" in verdict.reason


def test_hook_receives_command_cwd_env_as_json(tmp_path, ws):
    capture = tmp_path / "captured.json"
    hook = _hook(tmp_path, f"cat > {shlex.quote(str(capture))}\nexit 0\n")
    policy = build_policy(ws, rules=(HookRule(hook),))
    evaluate_command(policy, "ls -la", cwd=ws, env={"PATH": "/usr/bin"})
    payload = json.loads(capture.read_text())
    assert payload == {
        "command": "ls -la",
        "cwd": str(ws),
        "env": {"PATH": "/usr/bin"},
    }


def test_hook_ask_is_terminal(tmp_path, ws):
    # The deny rule after the hook is never reached: a consulted hook's
    # verdict ends the evaluation, ask included.
    policy = build_policy(ws, rules=(
        HookRule(_hook(tmp_path, "exit 2\n")),
        RegexRule(r"ls( .*)?", Decision.DENY),
    ))
    assert _decide(policy, "ls -la", ws).decision is Decision.ASK


def test_regex_rule_before_hook_wins_without_consulting_it(tmp_path, ws):
    marker = tmp_path / "consulted"
    hook = _hook(tmp_path, f"touch {shlex.quote(str(marker))}\nexit 1\n")
    policy = build_policy(ws, rules=(
        RegexRule(r"ls( .*)?", Decision.ALLOW_SANDBOXED),
        HookRule(hook),
    ))
    assert _decide(policy, "ls -la", ws).decision is Decision.ALLOW_SANDBOXED
    assert not marker.exists()


# --- config loading ---

def test_missing_config_file_yields_the_default_policy(tmp_path, ws):
    policy = load_policy(ws, config_path=tmp_path / "absent.toml")
    assert policy.rules == DEFAULT_RULES
    assert policy.workspace == os.path.realpath(ws)


def test_config_rules_parse_in_order_before_the_defaults(tmp_path, ws):
    config = tmp_path / "config.toml"
    config.write_text(
        '[server]\n'
        'endpoint = "http://127.0.0.1:8080"\n'
        '\n'
        '[[sandbox.rules]]\n'
        "pattern = 'rm( .*)?'\n"
        'decision = "deny"\n'
        '\n'
        '[[sandbox.rules]]\n'
        'hook = "/abs/hook.sh"\n'
    )
    policy = load_policy(ws, config_path=config)
    assert policy.rules == (
        RegexRule(r"rm( .*)?", Decision.DENY),
        HookRule("/abs/hook.sh"),
        *DEFAULT_RULES,
    )


def test_config_hook_path_expands_the_home_prefix(tmp_path, ws):
    config = tmp_path / "config.toml"
    config.write_text('[[sandbox.rules]]\nhook = "~/hooks/h.sh"\n')
    policy = load_policy(ws, config_path=config)
    assert policy.rules[0] == HookRule(str(Path.home() / "hooks" / "h.sh"))


def test_config_without_a_sandbox_table_is_fine(tmp_path, ws):
    config = tmp_path / "config.toml"
    config.write_text('[server]\nendpoint = "http://127.0.0.1:8080"\n')
    assert load_policy(ws, config_path=config).rules == DEFAULT_RULES


@pytest.mark.parametrize("text,fragment", [
    ("not [ valid toml", "invalid TOML"),
    ("sandbox = 5\n", "must be a table"),
    ("[sandbox]\nrules = 5\n", "must be an array of tables"),
    ("[sandbox]\nrules = [1, 2]\n", "each rule must be a table"),
    ("[sandbox]\nhoook = 'x'\n", "unknown [sandbox] key"),
    ("[[sandbox.rules]]\npattern = 'x'\n", "either {pattern, decision} or {hook}"),
    ("[[sandbox.rules]]\ndecision = 'deny'\n", "either {pattern, decision} or {hook}"),
    ("[[sandbox.rules]]\npattern = 'x'\ndecision = 'deny'\nhook = '/h'\n",
     "either {pattern, decision} or {hook}"),
    ("[[sandbox.rules]]\npattern = 'x'\ndecision = 'deny'\nnote = 'y'\n",
     "either {pattern, decision} or {hook}"),
    ("[[sandbox.rules]]\npattern = 5\ndecision = 'deny'\n", "pattern must be a string"),
    ("[[sandbox.rules]]\npattern = 'x'\ndecision = 5\n", "decision must be a string"),
    ("[[sandbox.rules]]\npattern = 'x'\ndecision = 'refuse'\n", "unknown decision"),
    ("[[sandbox.rules]]\npattern = '('\ndecision = 'deny'\n", "invalid pattern"),
    ("[[sandbox.rules]]\nhook = 5\n", "hook must be a string"),
    ("[[sandbox.rules]]\nhook = 'relative/h.sh'\n", "absolute path"),
])
def test_malformed_config_fails_closed_with_a_clear_error(tmp_path, ws, text, fragment):
    config = tmp_path / "config.toml"
    config.write_text(text)
    with pytest.raises(PolicyConfigError) as excinfo:
        load_policy(ws, config_path=config)
    assert fragment in str(excinfo.value)


def test_unknown_decision_error_lists_the_allowed_values(tmp_path, ws):
    config = tmp_path / "config.toml"
    config.write_text("[[sandbox.rules]]\npattern = 'x'\ndecision = 'refuse'\n")
    with pytest.raises(PolicyConfigError, match="allow-unsandboxed"):
        load_policy(ws, config_path=config)


# --- policy construction ---

def test_build_policy_resolves_and_dedupes_writable_roots(tmp_path, ws):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    policy = build_policy(ws, scratch_paths=(ws, scratch, scratch))
    assert policy.workspace == os.path.realpath(ws)
    assert policy.writable_paths == (os.path.realpath(scratch),)


# --- profile generation ---

def test_generate_profile_exact_text(tmp_path, ws):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    policy = build_policy(ws, scratch_paths=(scratch,))
    assert generate_profile(policy) == (
        "(version 1)\n"
        "(allow default)\n"
        "(deny file-write*)\n"
        "(allow file-write*\n"
        f'    (subpath "{os.path.realpath(ws)}")\n'
        f'    (subpath "{os.path.realpath(scratch)}")\n'
        '    (literal "/dev/null"))\n'
    )


def test_generate_profile_escapes_quotes_and_backslashes():
    policy = SandboxPolicy(workspace='/ws"e\\vil')
    assert '(subpath "/ws\\"e\\\\vil")' in generate_profile(policy)


def test_sandboxed_command_quoting_round_trips():
    profile = "(version 1)\n(allow default)\n"
    command = "echo 'a b' > \"f.txt\""
    parts = shlex.split(sandboxed_command(command, profile))
    assert parts == ["sandbox-exec", "-p", profile, "bash", "-c", command]


# --- path scope for the non-bash tools ---

def test_paths_inside_the_workspace_are_in_scope(ws):
    policy = build_policy(ws)
    (ws / "sub").mkdir()
    assert path_scope_problem(policy, "a.txt", ws) is None
    assert path_scope_problem(policy, ".", ws) is None
    assert path_scope_problem(policy, "sub/b.txt", ws) is None
    assert path_scope_problem(policy, str(ws / "c.txt"), ws) is None


def test_scratch_paths_are_in_scope(tmp_path, ws):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    policy = build_policy(ws, scratch_paths=(scratch,))
    assert path_scope_problem(policy, str(scratch / "s.txt"), ws) is None


@pytest.mark.parametrize("argument", [
    "../outside/f.txt",
    "sub/../../outside/f.txt",
    "/etc/hosts",
])
def test_escaping_paths_are_out_of_scope(ws, outside, argument):
    policy = build_policy(ws)
    (ws / "sub").mkdir()
    problem = path_scope_problem(policy, argument, ws)
    assert problem is not None
    assert "outside the workspace scope" in problem


def test_symlink_pointing_outside_is_out_of_scope(ws, outside):
    target = outside / "target.txt"
    target.write_text("secret")
    (ws / "link.txt").symlink_to(target)
    policy = build_policy(ws)
    assert path_scope_problem(policy, "link.txt", ws) is not None


def test_symlink_pointing_inside_is_in_scope(ws):
    (ws / "real.txt").write_text("data")
    (ws / "alias.txt").symlink_to(ws / "real.txt")
    policy = build_policy(ws)
    assert path_scope_problem(policy, "alias.txt", ws) is None


def test_sibling_directory_sharing_the_name_prefix_is_out_of_scope(tmp_path, ws):
    evil = tmp_path / "ws-evil"
    evil.mkdir()
    policy = build_policy(ws)
    assert path_scope_problem(policy, str(evil / "f.txt"), ws) is not None


# --- the gate in execute_tool_call ---

def test_denied_bash_command_is_a_failed_result(registry, ws):
    policy = build_policy(ws, rules=(RegexRule(r"rm( .*)?", Decision.DENY),))
    result = _run(registry, ws, policy, "bash", command="rm -rf x")
    assert not result.ok
    assert "refused by sandbox policy" in result.output
    assert "rm( .*)?" in result.output


def test_hook_deny_reason_reaches_the_tool_result(tmp_path, registry, ws):
    hook = _hook(tmp_path, "echo not on the list\nexit 1\n")
    policy = build_policy(ws, rules=(HookRule(hook),))
    result = _run(registry, ws, policy, "bash", command="ls")
    assert not result.ok
    assert "refused by sandbox policy" in result.output
    assert "not on the list" in result.output


def test_missing_hook_refuses_without_raising(tmp_path, registry, ws):
    policy = build_policy(ws, rules=(HookRule(str(tmp_path / "absent.sh")),))
    result = _run(registry, ws, policy, "bash", command="ls")
    assert not result.ok
    assert "hook did not run" in result.output


def test_allow_unsandboxed_command_can_write_outside_the_scope(registry, ws, outside):
    policy = build_policy(ws, rules=(
        RegexRule(r"printf .*", Decision.ALLOW_UNSANDBOXED),))
    result = _run(registry, ws, policy, "bash",
                  command=f"printf data > {outside}/u.txt")
    assert result.ok
    assert (outside / "u.txt").read_text() == "data"


def test_sandbox_requiring_decision_refuses_off_macos(registry, ws, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    policy = build_policy(ws)
    result = _run(registry, ws, policy, "bash", command="ls")
    assert not result.ok
    assert "requires macOS sandbox-exec" in result.output


def test_empty_bash_command_still_fails_in_the_executor(registry, ws):
    # The gate leaves malformed commands to the executor's own check, so an
    # empty command cannot become a runnable sandboxed wrapper.
    policy = build_policy(ws)
    result = _run(registry, ws, policy, "bash", command="   ")
    assert not result.ok
    assert "non-empty" in result.output


def test_tool_without_a_policy_mapping_is_refused(registry, ws):
    spec = ToolSpec(name="ping", description="test tool",
                    parameters={"type": "object", "properties": {}})
    registry.register(spec, lambda arguments, workdir: "pong")
    result = _run(registry, ws, build_policy(ws), "ping")
    assert not result.ok
    assert "no rule class" in result.output


def test_read_file_inside_the_workspace_is_allowed(registry, ws):
    (ws / "a.txt").write_text("content")
    result = _run(registry, ws, build_policy(ws), "read_file", path="a.txt")
    assert result.ok
    assert result.output == "content"


def test_read_file_from_scratch_scope_is_allowed(tmp_path, registry, ws):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "s.txt").write_text("scratch data")
    policy = build_policy(ws, scratch_paths=(scratch,))
    result = _run(registry, ws, policy, "read_file", path=str(scratch / "s.txt"))
    assert result.ok
    assert result.output == "scratch data"


def test_read_file_outside_the_workspace_is_refused(registry, ws, outside):
    (outside / "o.txt").write_text("secret")
    result = _run(registry, ws, build_policy(ws), "read_file",
                  path=str(outside / "o.txt"))
    assert not result.ok
    assert "outside the workspace scope" in result.output


def test_read_file_through_an_outward_symlink_is_refused(registry, ws, outside):
    target = outside / "target.txt"
    target.write_text("secret")
    (ws / "link.txt").symlink_to(target)
    result = _run(registry, ws, build_policy(ws), "read_file", path="link.txt")
    assert not result.ok
    assert "outside the workspace scope" in result.output


def test_edit_outside_via_traversal_is_refused_and_leaves_the_file(registry, ws, outside):
    target = outside / "t.txt"
    target.write_text("original")
    result = _run(registry, ws, build_policy(ws), "edit",
                  path="../outside/t.txt", old_string="original", new_string="x")
    assert not result.ok
    assert "outside the workspace scope" in result.output
    assert target.read_text() == "original"


def test_grep_defaults_to_the_workspace_and_is_allowed(registry, ws):
    (ws / "a.py").write_text("needle\n")
    result = _run(registry, ws, build_policy(ws), "grep", pattern="needle")
    assert result.ok
    assert result.output == "a.py:1:needle"


def test_grep_outside_the_workspace_is_refused(registry, ws, outside):
    (outside / "o.txt").write_text("needle\n")
    result = _run(registry, ws, build_policy(ws), "grep", pattern="needle",
                  path=str(outside))
    assert not result.ok
    assert "outside the workspace scope" in result.output


def test_no_policy_leaves_execution_ungated(registry, ws, outside):
    result = _run(registry, ws, None, "read_file", path=str(outside))
    # Ungated, the call reaches the executor and fails there (a directory is
    # not a file); the scope check never fires.
    assert not result.ok
    assert "no such file" in result.output


# --- integration: real sandbox-exec runs ---

@requires_seatbelt
def test_sandboxed_write_inside_the_workspace_succeeds(registry, ws):
    # An unmatched command falls to ask, which the headless gate resolves to
    # a sandboxed run.
    result = _run(registry, ws, build_policy(ws), "bash",
                  command="printf data > inside.txt")
    assert result.ok
    assert (ws / "inside.txt").read_text() == "data"


@requires_seatbelt
def test_sandboxed_write_outside_the_workspace_is_denied(registry, ws, outside):
    result = _run(registry, ws, build_policy(ws), "bash",
                  command=f"printf x > {outside}/nope.txt")
    assert not result.ok
    assert "Operation not permitted" in result.output
    assert not (outside / "nope.txt").exists()


@requires_seatbelt
def test_sandboxed_read_outside_the_workspace_is_allowed(registry, ws, outside):
    (outside / "readable.txt").write_text("readable")
    result = _run(registry, ws, build_policy(ws), "bash",
                  command=f"cat {outside}/readable.txt")
    assert result.ok
    assert result.output == "readable"


@requires_seatbelt
def test_sandboxed_write_to_a_scratch_path_succeeds(tmp_path, registry, ws):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    policy = build_policy(ws, scratch_paths=(scratch,))
    result = _run(registry, ws, policy, "bash",
                  command=f"printf s > {scratch}/s.txt")
    assert result.ok
    assert (scratch / "s.txt").read_text() == "s"


@requires_seatbelt
def test_sandboxed_dev_null_redirect_works(registry, ws):
    result = _run(registry, ws, build_policy(ws), "bash",
                  command="echo hi > /dev/null")
    assert result.ok
