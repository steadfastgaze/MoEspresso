"""Tool registry and the four core coding tools.

A ``ToolSpec`` is the model-facing contract (name, description, JSON schema);
the executor is the Python implementation behind it. The registry maps names
to both and renders the OpenAI ``tools`` array for a request. Callers never
invoke executors directly: every invocation flows through
``execution.execute_tool_call``, the single choke point the sandbox policy
layer later wraps.

The core implementations are deliberately minimal. Path containment and
command policy are sandbox concerns and are not half-implemented here: the
policy gate inside ``execution.execute_tool_call`` applies them when a
``SandboxPolicy`` is supplied, and without one the tools operate with the
caller's privileges relative to the given working directory.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Bounds on tool output and execution so one call cannot stall or flood a session.
GREP_MAX_MATCHES = 200
BASH_DEFAULT_TIMEOUT_SECONDS = 120.0
BASH_MAX_TIMEOUT_SECONDS = 600.0

# Executor contract: (arguments, workdir) -> output string. Raises ValueError
# on a tool-level failure; the choke point turns that into a failed result.
ToolExecutor = Callable[[dict, Path], str]


@dataclass(frozen=True)
class ToolSpec:
    """The model-facing contract of one tool."""

    name: str
    description: str
    parameters: dict

    def as_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Named tools with their executors, in registration order."""

    def __init__(self):
        self._specs: dict[str, ToolSpec] = {}
        self._executors: dict[str, ToolExecutor] = {}

    def register(self, spec: ToolSpec, executor: ToolExecutor) -> None:
        if spec.name in self._specs:
            raise ValueError(f"tool already registered: {spec.name}")
        self._specs[spec.name] = spec
        self._executors[spec.name] = executor

    def names(self) -> list[str]:
        return list(self._specs)

    def spec(self, name: str) -> ToolSpec:
        if name not in self._specs:
            raise KeyError(f"unknown tool: {name}")
        return self._specs[name]

    def executor(self, name: str) -> ToolExecutor:
        if name not in self._executors:
            raise KeyError(f"unknown tool: {name}")
        return self._executors[name]

    def openai_tools(self) -> list[dict]:
        """The request's ``tools`` array. Hold it constant for a whole session:
        it renders into the shared prompt prefix, so changing it mid-session
        invalidates every cached turn."""
        return [spec.as_openai_tool() for spec in self._specs.values()]


# --- core tool implementations ---


def _resolve(path_argument: str, workdir: Path) -> Path:
    path = Path(path_argument)
    if not path.is_absolute():
        path = workdir / path
    return path


def _read_file(arguments: dict, workdir: Path) -> str:
    path = _resolve(arguments["path"], workdir)
    if not path.is_file():
        raise ValueError(f"no such file: {arguments['path']}")
    text = path.read_text(encoding="utf-8", errors="replace")
    offset = arguments.get("offset")
    limit = arguments.get("limit")
    if offset is None and limit is None:
        return text
    lines = text.splitlines(keepends=True)
    start = int(offset) - 1 if offset is not None else 0
    if start < 0:
        raise ValueError("offset must be a 1-based line number")
    if start >= len(lines) and lines:
        raise ValueError(
            f"offset {int(offset)} is past the end of the file ({len(lines)} lines)")
    end = start + int(limit) if limit is not None else len(lines)
    return "".join(lines[start:end])


def _iter_grep_files(root: Path, glob: str | None):
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if ".git" in path.relative_to(root).parts:
            continue
        if glob is not None and not path.match(glob):
            continue
        yield path


def _grep(arguments: dict, workdir: Path) -> str:
    try:
        pattern = re.compile(arguments["pattern"])
    except re.error as e:
        raise ValueError(f"invalid regex: {e}") from e
    root = _resolve(arguments.get("path", "."), workdir)
    if not root.exists():
        raise ValueError(f"no such path: {arguments.get('path', '.')}")
    matches = []
    truncated = False
    for path in _iter_grep_files(root, arguments.get("glob")):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable files are not searchable text
        rel = path.relative_to(workdir) if path.is_relative_to(workdir) else path
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                matches.append(f"{rel}:{lineno}:{line}")
                if len(matches) >= GREP_MAX_MATCHES:
                    truncated = True
                    break
        if truncated:
            break
    if not matches:
        return "no matches"
    if truncated:
        matches.append(f"... truncated at {GREP_MAX_MATCHES} matches")
    return "\n".join(matches)


def _edit(arguments: dict, workdir: Path) -> str:
    path = _resolve(arguments["path"], workdir)
    if not path.is_file():
        raise ValueError(f"no such file: {arguments['path']}")
    old = arguments["old_string"]
    new = arguments["new_string"]
    if old == new:
        raise ValueError("old_string and new_string are identical")
    if not old:
        raise ValueError("old_string must be non-empty")
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 0:
        # A silent no-op edit is the failure mode this tool exists to prevent;
        # a missing anchor must fail loudly.
        raise ValueError("old_string not found in file")
    replace_all = bool(arguments.get("replace_all", False))
    if count > 1 and not replace_all:
        raise ValueError(
            f"old_string occurs {count} times; make it unique or set replace_all")
    path.write_text(text.replace(old, new), encoding="utf-8")
    replaced = count if replace_all else 1
    return f"replaced {replaced} occurrence(s) in {arguments['path']}"


def _bash(arguments: dict, workdir: Path) -> str:
    command = arguments["command"]
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    timeout = float(arguments.get("timeout", BASH_DEFAULT_TIMEOUT_SECONDS))
    timeout = min(timeout, BASH_MAX_TIMEOUT_SECONDS)
    try:
        completed = subprocess.run(
            ["bash", "-c", command],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise ValueError(f"command timed out after {timeout:g}s") from e
    parts = []
    if completed.stdout:
        parts.append(completed.stdout.rstrip("\n"))
    if completed.stderr:
        parts.append(f"stderr:\n{completed.stderr.rstrip()}" if parts
                     else completed.stderr.rstrip("\n"))
    output = "\n".join(parts)
    if completed.returncode != 0:
        raise ValueError(
            f"exit code {completed.returncode}\n{output}" if output
            else f"exit code {completed.returncode}")
    return output


CORE_TOOL_SPECS = (
    ToolSpec(
        name="read_file",
        description="Read a text file. Paths resolve against the working "
                    "directory. Optional offset (1-based line) and limit "
                    "select a line window.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read."},
                "offset": {"type": "integer",
                           "description": "1-based first line to read."},
                "limit": {"type": "integer",
                          "description": "Maximum number of lines to read."},
            },
            "required": ["path"],
        },
    ),
    ToolSpec(
        name="grep",
        description="Search file contents with a regular expression. Returns "
                    "path:line:text matches. Skips binary files and .git.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string",
                            "description": "Regular expression to search for."},
                "path": {"type": "string",
                         "description": "File or directory to search "
                                        "(default: the working directory)."},
                "glob": {"type": "string",
                         "description": "Filename glob filter, e.g. *.py."},
            },
            "required": ["pattern"],
        },
    ),
    ToolSpec(
        name="edit",
        description="Replace an exact string in a file. Fails if old_string "
                    "is missing or ambiguous; set replace_all to replace "
                    "every occurrence.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to edit."},
                "old_string": {"type": "string",
                               "description": "Exact text to replace."},
                "new_string": {"type": "string",
                               "description": "Replacement text."},
                "replace_all": {"type": "boolean",
                                "description": "Replace every occurrence."},
            },
            "required": ["path", "old_string", "new_string"],
        },
    ),
    ToolSpec(
        name="bash",
        description="Run a bash command in the working directory and return "
                    "its output. Non-zero exit codes are reported as errors.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string",
                            "description": "The command to execute."},
                "timeout": {"type": "number",
                            "description": "Timeout in seconds "
                                           f"(default {BASH_DEFAULT_TIMEOUT_SECONDS:g}, "
                                           f"max {BASH_MAX_TIMEOUT_SECONDS:g})."},
            },
            "required": ["command"],
        },
    ),
)

_CORE_EXECUTORS: dict[str, ToolExecutor] = {
    "read_file": _read_file,
    "grep": _grep,
    "edit": _edit,
    "bash": _bash,
}


def build_core_registry() -> ToolRegistry:
    """A registry with the four core tools: read_file, grep, edit, bash."""
    registry = ToolRegistry()
    for spec in CORE_TOOL_SPECS:
        registry.register(spec, _CORE_EXECUTORS[spec.name])
    return registry
