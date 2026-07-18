"""Sandbox policy engine for tool execution.

macOS sandbox-exec (Seatbelt) is the enforcement floor. This module is the
policy layer above it: a ``SandboxPolicy`` value holds the writable scope
(the agent workspace plus designated scratch paths) and an ordered rule
list, and pure functions derive everything else from that value.
``evaluate_command`` walks the rules in order and returns the first match;
``build_policy`` appends the shipped default rules (git runs unsandboxed)
after the configured rules; a command no rule matches falls to ask.

Decisions are allow-unsandboxed, allow-sandboxed, ask, and deny. The
headless executor in ``execution.execute_tool_call`` resolves ask to a
sandboxed run, which is the default posture: reads broadly allowed, writes
denied outside the writable scope, enforced by the Seatbelt profile from
``generate_profile``.

Configured rules come from the ``[sandbox]`` table of
``~/.moespresso/config.toml``. Each ``[[sandbox.rules]]`` entry carries
either a regex matcher (``pattern`` plus ``decision``) or an executable hook
(``hook``, an absolute path). A missing config file yields the default
policy. A malformed sandbox table raises ``PolicyConfigError`` at load time;
loading fails closed on config it cannot fully understand.

Details the design leaves open are fixed here on the conservative side:

- Patterns match with ``re.fullmatch`` against the whitespace-stripped
  command. A substring hit never fires a rule; a rule that should match
  command families needs an explicit ``.*``, so a wide approval has to be
  written deliberately.
- A hook receives ``{"command", "cwd", "env"}`` as JSON on stdin. Exit 0
  approves, exit 2 asks, any other exit denies. An approving hook may print
  exactly ``allow-unsandboxed`` or ``allow-sandboxed`` on stdout to pick the
  mode; empty stdout means allow-sandboxed, and any other stdout denies. On
  a deny or ask exit, stdout is carried as the reason. A hook that cannot
  run, times out, or violates the output contract denies.
- A consulted hook is terminal, ask included; rules after it are not
  evaluated.
- Seatbelt matches resolved vnode paths, so ``build_policy`` realpaths the
  writable roots; ``generate_profile`` is pure text generation from the
  policy value.
- Path arguments of the non-bash tools (read_file, grep, edit) are confined
  to the writable scope. ``path_scope_problem`` resolves the argument with
  realpath, so ``..`` segments and symlinks that leave the scope are caught.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tomllib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

HOOK_TIMEOUT_SECONDS = 10.0


class Decision(Enum):
    """The four dispositions a rule or hook can map a command to."""

    ALLOW_UNSANDBOXED = "allow-unsandboxed"
    ALLOW_SANDBOXED = "allow-sandboxed"
    ASK = "ask"
    DENY = "deny"


class PolicyConfigError(ValueError):
    """A malformed sandbox config. Raised at load time, never for a policy
    outcome on a command."""


@dataclass(frozen=True)
class RegexRule:
    """Maps commands that fullmatch ``pattern`` to a fixed decision."""

    pattern: str
    decision: Decision

    def __post_init__(self):
        try:
            re.compile(self.pattern)
        except re.error as e:
            raise PolicyConfigError(f"invalid pattern {self.pattern!r}: {e}") from e


@dataclass(frozen=True)
class HookRule:
    """Delegates the decision to an executable at an absolute path."""

    hook: str


Rule = RegexRule | HookRule


@dataclass(frozen=True)
class SandboxPolicy:
    """The complete policy value: writable roots plus the final ordered rules.

    ``workspace`` and ``writable_paths`` are resolved absolute paths;
    ``build_policy`` is the constructor that resolves them.
    """

    workspace: str
    writable_paths: tuple[str, ...] = ()
    rules: tuple[Rule, ...] = ()


@dataclass(frozen=True)
class PolicyDecision:
    """One evaluation outcome: the decision, what produced it, and any
    hook-provided reason text."""

    decision: Decision
    source: str
    reason: str = ""


# git operations (including .git metadata and git config writes) are approved
# outside the OS sandbox so the agent never fights git. The pattern requires
# the whole command to be a git invocation; a compound command that merely
# contains git does not match and falls through.
DEFAULT_RULES: tuple[Rule, ...] = (
    RegexRule(pattern=r"git(\s.*)?", decision=Decision.ALLOW_UNSANDBOXED),
)


def default_config_path() -> Path:
    """The durable config location, ``~/.moespresso/config.toml``."""
    return Path("~/.moespresso/config.toml").expanduser()


def build_policy(workspace: str | Path, *,
                 scratch_paths: tuple[str | Path, ...] = (),
                 rules: tuple[Rule, ...] = ()) -> SandboxPolicy:
    """Resolve the writable roots and compose the final rule order.

    Configured rules come first, the shipped defaults after them, so a
    configured rule can override a default. Roots are realpathed because
    Seatbelt matches resolved vnode paths; a profile listing a symlinked
    spelling of a directory does not cover writes into it.
    """
    resolved_workspace = os.path.realpath(workspace)
    resolved_scratch = []
    for path in scratch_paths:
        resolved = os.path.realpath(path)
        if resolved != resolved_workspace and resolved not in resolved_scratch:
            resolved_scratch.append(resolved)
    return SandboxPolicy(
        workspace=resolved_workspace,
        writable_paths=tuple(resolved_scratch),
        rules=(*rules, *DEFAULT_RULES),
    )


def load_policy(workspace: str | Path, *, config_path: str | Path,
                scratch_paths: tuple[str | Path, ...] = ()) -> SandboxPolicy:
    """Build the policy from a config.toml.

    A missing file yields the default policy. Invalid TOML or a malformed
    ``[sandbox]`` table raises ``PolicyConfigError``. Tables other than
    ``[sandbox]`` belong to other components and are ignored.
    """
    rules: tuple[Rule, ...] = ()
    path = Path(config_path).expanduser()
    if path.is_file():
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise PolicyConfigError(f"{path}: invalid TOML: {e}") from e
        rules = _parse_sandbox_rules(data, source=path)
    return build_policy(workspace, scratch_paths=scratch_paths, rules=rules)


def _parse_sandbox_rules(data: dict, *, source: Path) -> tuple[Rule, ...]:
    sandbox = data.get("sandbox", {})
    if not isinstance(sandbox, dict):
        raise PolicyConfigError(f"{source}: [sandbox] must be a table")
    unknown = set(sandbox) - {"rules"}
    if unknown:
        raise PolicyConfigError(
            f"{source}: unknown [sandbox] key(s): {', '.join(sorted(unknown))}")
    entries = sandbox.get("rules", [])
    if not isinstance(entries, list):
        raise PolicyConfigError(
            f"{source}: sandbox.rules must be an array of tables")
    return tuple(_parse_rule(entry, index=index, source=source)
                 for index, entry in enumerate(entries))


def _parse_rule(entry, *, index: int, source: Path) -> Rule:
    where = f"{source}: sandbox.rules[{index}]"
    if not isinstance(entry, dict):
        raise PolicyConfigError(f"{where}: each rule must be a table")
    keys = set(entry)
    if keys == {"pattern", "decision"}:
        pattern = entry["pattern"]
        decision = entry["decision"]
        if not isinstance(pattern, str):
            raise PolicyConfigError(f"{where}: pattern must be a string")
        if not isinstance(decision, str):
            raise PolicyConfigError(f"{where}: decision must be a string")
        try:
            parsed = Decision(decision)
        except ValueError:
            allowed = ", ".join(d.value for d in Decision)
            raise PolicyConfigError(
                f"{where}: unknown decision {decision!r} "
                f"(allowed: {allowed})") from None
        try:
            return RegexRule(pattern=pattern, decision=parsed)
        except PolicyConfigError as e:
            raise PolicyConfigError(f"{where}: {e}") from None
    if keys == {"hook"}:
        hook = entry["hook"]
        if not isinstance(hook, str):
            raise PolicyConfigError(f"{where}: hook must be a string path")
        hook_path = Path(hook).expanduser()
        if not hook_path.is_absolute():
            raise PolicyConfigError(
                f"{where}: hook must be an absolute path, got {hook!r}")
        return HookRule(hook=str(hook_path))
    raise PolicyConfigError(
        f"{where}: a rule is either {{pattern, decision}} or {{hook}}; "
        f"got keys {{{', '.join(sorted(keys))}}}")


def evaluate_command(policy: SandboxPolicy, command: str, *,
                     cwd: str | Path, env: dict | None = None) -> PolicyDecision:
    """Decide how one bash command runs. First matching rule wins.

    Regex rules fullmatch the stripped command. A hook rule, once reached,
    is consulted and its verdict is terminal. A command no rule matches
    falls to ask. Policy outcomes are returned, never raised.
    """
    stripped = command.strip()
    for rule in policy.rules:
        if isinstance(rule, RegexRule):
            if re.fullmatch(rule.pattern, stripped):
                return PolicyDecision(rule.decision, f"pattern {rule.pattern!r}")
        else:
            return _run_hook(rule, command, cwd=cwd, env=env)
    return PolicyDecision(Decision.ASK, "unmatched")


def _run_hook(rule: HookRule, command: str, *,
              cwd: str | Path, env: dict | None) -> PolicyDecision:
    """Consult one executable hook and map its verdict to a decision.

    Every failure mode of the hook itself (cannot run, timeout, unexpected
    exit, unrecognized approval output) denies.
    """
    payload = json.dumps({
        "command": command,
        "cwd": str(cwd),
        "env": dict(env if env is not None else os.environ),
    })
    source = f"hook {rule.hook}"
    try:
        completed = subprocess.run(
            [rule.hook],
            input=payload,
            capture_output=True,
            text=True,
            timeout=HOOK_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return PolicyDecision(
            Decision.DENY, source,
            f"hook timed out after {HOOK_TIMEOUT_SECONDS:g}s")
    except OSError as e:
        return PolicyDecision(Decision.DENY, source, f"hook did not run: {e}")
    out = completed.stdout.strip()
    if completed.returncode == 0:
        if out in ("", Decision.ALLOW_SANDBOXED.value):
            return PolicyDecision(Decision.ALLOW_SANDBOXED, source)
        if out == Decision.ALLOW_UNSANDBOXED.value:
            return PolicyDecision(Decision.ALLOW_UNSANDBOXED, source)
        return PolicyDecision(
            Decision.DENY, source,
            f"hook approved with unrecognized output {out!r}")
    if completed.returncode == 2:
        return PolicyDecision(Decision.ASK, source, out)
    return PolicyDecision(
        Decision.DENY, source,
        out or f"hook exited {completed.returncode}")


def generate_profile(policy: SandboxPolicy) -> str:
    """The Seatbelt profile for the sandboxed disposition.

    Pure text generation from the policy value: reads stay broadly allowed,
    writes are denied everywhere except the policy's writable roots and
    ``/dev/null`` (shell redirections need it). In SBPL the later rule wins,
    so the targeted allow overrides the blanket write deny.
    """
    roots = (policy.workspace, *policy.writable_paths)
    allows = "\n".join(f'    (subpath "{_sbpl_quote(root)}")' for root in roots)
    return (
        "(version 1)\n"
        "(allow default)\n"
        "(deny file-write*)\n"
        "(allow file-write*\n"
        f"{allows}\n"
        '    (literal "/dev/null"))\n'
    )


def _sbpl_quote(path: str) -> str:
    """Escape a path for an SBPL double-quoted string literal."""
    return path.replace("\\", "\\\\").replace('"', '\\"')


def sandboxed_command(command: str, profile: str) -> str:
    """The command line that runs ``command`` under sandbox-exec.

    Both the profile and the command are shell-quoted, so the bash executor
    can run the wrapper string unchanged; exit status and output pass
    through from the inner command.
    """
    return (f"sandbox-exec -p {shlex.quote(profile)} "
            f"bash -c {shlex.quote(command)}")


def path_scope_problem(policy: SandboxPolicy, path_argument: str,
                       workdir: str | Path) -> str | None:
    """Check one tool path argument against the writable scope.

    Relative arguments resolve against ``workdir``; the result is realpathed
    so ``..`` segments and symlinks pointing outside the scope are caught.
    Returns a message the model can act on when the path escapes the
    workspace and scratch roots, None when it is in scope.
    """
    candidate = Path(path_argument)
    if not candidate.is_absolute():
        candidate = Path(workdir) / candidate
    resolved = Path(os.path.realpath(candidate))
    for root in (policy.workspace, *policy.writable_paths):
        if resolved.is_relative_to(root):
            return None
    return (f"path {path_argument!r} resolves to {resolved}, "
            f"outside the workspace scope")
