"""Sandboxed execution of model-emitted code against hidden test cases.

An agentic-coding task asks the model to implement a small function. The gate
extracts the emitted implementation (from a qwen3_xml `<tool_call>` submitting a
`code` parameter, or from a markdown code block as a fallback), then runs it in a
fresh subprocess against hidden test cases the model never sees. The score is the
number of tests that pass. The subprocess is killed on a wall-clock timeout so a
non-terminating solution cannot hang the gate.

The candidate code and the test driver are written to a temporary file and run
with the interpreter running the gate. Execution is isolated to its own process,
so a candidate that raises, loops, or corrupts global state affects only that
subprocess. This correctness harness assumes trusted local benchmark inputs and
does not provide a security boundary against adversarial code.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# qwen3_xml submits a tool call as
#   <tool_call>\n<function=NAME>\n<parameter=code>\n...\n</parameter>\n</function>\n</tool_call>
_PARAM_CODE_RE = re.compile(
    r"<parameter=code>\s*(.*?)\s*</parameter>", re.DOTALL | re.IGNORECASE)
_FUNCTION_BLOCK_RE = re.compile(
    r"<function=[^>]*>(.*?)</function>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str | None:
    """Extract the candidate implementation from a model reply.

    Prefers a qwen3_xml `<parameter=code>` payload; falls back to the body of a
    `<function=...>` block, then to a fenced code block. Returns None if no code
    is present.
    """
    match = _PARAM_CODE_RE.search(text)
    if match and match.group(1).strip():
        return match.group(1).strip()
    match = _FUNCTION_BLOCK_RE.search(text)
    if match:
        inner = match.group(1)
        fence = _CODE_FENCE_RE.search(inner)
        if fence and fence.group(1).strip():
            return fence.group(1).strip()
        stripped = re.sub(r"<parameter=[^>]*>|</parameter>", "", inner).strip()
        if stripped:
            return stripped
    match = _CODE_FENCE_RE.search(text)
    if match and match.group(1).strip():
        return match.group(1).strip()
    return None


_DRIVER = '''
import json
import sys

# --- candidate implementation (model-emitted) ---
{candidate}
# --- end candidate ---

_CASES = json.loads({cases_json!r})
_ENTRY = {entry!r}
_results = []
_fn = globals().get(_ENTRY)
if not callable(_fn):
    print(json.dumps({{"import_error": "entry %r not defined or not callable" % _ENTRY}}))
    sys.exit(0)
for _case in _CASES:
    _args = _case["args"]
    _expected = _case["expected"]
    try:
        _got = _fn(*_args)
        _ok = (_got == _expected)
        _results.append({{"ok": bool(_ok), "got": repr(_got)}})
    except Exception as _exc:  # noqa: BLE001 - candidate may raise anything
        _results.append({{"ok": False, "error": type(_exc).__name__ + ": " + str(_exc)}})
print(json.dumps({{"results": _results}}))
'''


@dataclass
class SandboxResult:
    extracted: bool
    n_tests: int
    n_passed: int
    per_test: list
    error: str | None
    timed_out: bool


def run_hidden_tests(
    code: str | None,
    entry: str,
    cases: list[dict],
    *,
    timeout_seconds: float = 10.0,
    python_executable: str | None = None,
) -> SandboxResult:
    """Run the candidate `code` against `cases` in an isolated subprocess.

    `entry` is the required function name. Each case is `{"args": [...],
    "expected": value}`; a test passes when `entry(*args) == expected`. Returns a
    SandboxResult with the per-test outcomes.
    """
    if not code:
        return SandboxResult(extracted=False, n_tests=len(cases), n_passed=0,
                             per_test=[], error="no_code_extracted", timed_out=False)
    driver = _DRIVER.format(
        candidate=code, cases_json=json.dumps(cases), entry=entry)
    executable = python_executable or sys.executable
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "run_candidate.py"
        script.write_text(driver)
        try:
            proc = subprocess.run(
                [executable, str(script)],
                capture_output=True, text=True,
                timeout=timeout_seconds, cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(extracted=True, n_tests=len(cases), n_passed=0,
                                 per_test=[], error="timeout", timed_out=True)
    if proc.returncode != 0 and not proc.stdout.strip():
        return SandboxResult(extracted=True, n_tests=len(cases), n_passed=0,
                             per_test=[], error=(proc.stderr or "").strip()[-400:],
                             timed_out=False)
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return SandboxResult(extracted=True, n_tests=len(cases), n_passed=0,
                             per_test=[], error="unparseable_output", timed_out=False)
    if "import_error" in payload:
        return SandboxResult(extracted=True, n_tests=len(cases), n_passed=0,
                             per_test=[], error=payload["import_error"],
                             timed_out=False)
    per_test = payload.get("results", [])
    n_passed = sum(1 for row in per_test if row.get("ok"))
    return SandboxResult(extracted=True, n_tests=len(cases), n_passed=n_passed,
                         per_test=per_test, error=None, timed_out=False)
