"""The fixed episode set for the dialect study.

Fifteen short, single-goal tasks in three families of five, identical across
arms: read a file and answer, grep or edit against a known anchor, run a
command and interpret its output. Each episode runs on a fresh workspace
built from the road-test fixture generator plus a few planted files whose
contents anchor exact-answer judging. Task text pins the intended tool the
same way the road-test script does, so temperature-zero runs stay
deterministic and call correctness has a defined target.

Judging is scripted. Call correctness: a call is correct when it matches one
of the episode's matchers (intended tool plus an argument predicate;
reasonable exploration calls are listed as matchers too). Task success: the
expected answer token appears word-bounded in the final answer text, and the
goal-state predicate over the workspace holds. Both checks are independent
of any dialect.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from moespresso.agentlib.roadtest.fixture import (
    DEFECT_FIX_LINE,
    MARKER_ONE,
    MARKER_TWO,
    data_file_text,
    generate_fixture,
)
from moespresso.agentlib.toolcalls import ToolCall

VERSION_VALUE = "7.4.2"
BUMPED_VERSION_VALUE = "7.4.3"
SETTINGS_PORT = "9741"
BUMPED_SETTINGS_PORT = "9742"
TODO_TICKET = "rc-4471"
SELFTEST_CODE = "3196"
# The fixture writes twenty channel modules plus metrics.py under src/.
SRC_FILE_COUNT = 21
SEGMENT_LINES = "240"


def _segment_one_zone() -> str:
    first_line = data_file_text(1)[0].splitlines()[0]
    match = re.search(r" zone (\w+) ", first_line)
    assert match is not None
    return match.group(1)


def _segment_three_errors() -> str:
    return str(data_file_text(3)[1])


def prepare_workspace(root: Path) -> None:
    """Generate the fixture project plus the study's planted anchor files.

    An existing directory is cleared first: goal checks judge workspace
    state, so a rerun must never inherit an earlier run's edits or created
    files.
    """
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    generate_fixture(root)
    (root / "VERSION").write_text(VERSION_VALUE + "\n", encoding="utf-8")
    (root / "config").mkdir(exist_ok=True)
    (root / "config" / "settings.ini").write_text(
        f"[server]\nport = {SETTINGS_PORT}\nlog_level = info\n", encoding="utf-8")
    (root / "notes").mkdir(exist_ok=True)
    (root / "notes" / "todo.txt").write_text(
        "notes for the next calibration pass\n"
        f"TODO({TODO_TICKET}): recalibrate the delta channel filter\n",
        encoding="utf-8")
    (root / "scripts").mkdir(exist_ok=True)
    selftest = root / "scripts" / "selftest.sh"
    selftest.write_text(
        f"#!/bin/sh\necho 'SELFTEST PASS code={SELFTEST_CODE}'\n",
        encoding="utf-8")
    os.chmod(selftest, 0o755)


@dataclass(frozen=True)
class CallMatcher:
    """One acceptable call shape: a tool name plus an argument predicate."""

    tool: str
    check: Callable[[dict], bool] | None = None

    def matches(self, call: ToolCall) -> bool:
        if call.name != self.tool:
            return False
        if self.check is None:
            return True
        try:
            return bool(self.check(call.arguments))
        except (TypeError, AttributeError):
            return False


@dataclass(frozen=True)
class Episode:
    """One task: prompt, acceptable calls, and the success definition."""

    episode_id: str
    family: str
    prompt: str
    matchers: tuple[CallMatcher, ...]
    expect_answer: str | None = None
    goal_check: Callable[[Path], bool] | None = None
    max_steps: int = 4

    def call_is_correct(self, call: ToolCall) -> bool:
        return any(matcher.matches(call) for matcher in self.matchers)

    def succeeded(self, root: Path, answer_text: str) -> bool:
        if self.expect_answer is not None:
            pattern = rf"\b{re.escape(self.expect_answer)}\b"
            if not re.search(pattern, answer_text):
                return False
        if self.goal_check is not None:
            return bool(self.goal_check(root))
        return True


def _arg_contains(key: str, needle: str) -> Callable[[dict], bool]:
    return lambda args: needle in str(args.get(key, ""))


def _file_text(root: Path, relpath: str) -> str:
    path = root / relpath
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _metrics_fixed(root: Path) -> bool:
    text = _file_text(root, "src/metrics.py")
    return DEFECT_FIX_LINE in text and "len(values) - 1" not in text


def _version_bumped(root: Path) -> bool:
    text = _file_text(root, "VERSION")
    return BUMPED_VERSION_VALUE in text and VERSION_VALUE not in text


def _port_bumped(root: Path) -> bool:
    text = _file_text(root, "config/settings.ini")
    return BUMPED_SETTINGS_PORT in text and SETTINGS_PORT not in text


def _build_tag_written(root: Path) -> bool:
    return _file_text(root, "build_tag.txt").strip() == f"build-{VERSION_VALUE}"


def build_episodes() -> tuple[Episode, ...]:
    """The fixed fifteen-episode set, identical across arms."""
    read_answer = (
        Episode(
            episode_id="r1-version",
            family="read_answer",
            prompt=("Read the file VERSION with the read_file tool and report "
                    "the version string it contains."),
            matchers=(CallMatcher("read_file", _arg_contains("path", "VERSION")),),
            expect_answer=VERSION_VALUE,
        ),
        Episode(
            episode_id="r2-trace-tag",
            family="read_answer",
            prompt=("Read src/module_golf.py with the read_file tool and report "
                    "the exact value of TRACE_TAG."),
            matchers=(CallMatcher("read_file", _arg_contains("path", "module_golf")),),
            expect_answer=MARKER_ONE,
        ),
        Episode(
            episode_id="r3-port",
            family="read_answer",
            prompt=("Read config/settings.ini with the read_file tool and "
                    "report the configured port number."),
            matchers=(CallMatcher("read_file", _arg_contains("path", "settings.ini")),),
            expect_answer=SETTINGS_PORT,
        ),
        Episode(
            episode_id="r4-defect-function",
            family="read_answer",
            prompt=("Read src/metrics.py with the read_file tool and name the "
                    "function whose divisor is wrong."),
            matchers=(CallMatcher("read_file", _arg_contains("path", "metrics")),),
            expect_answer="mean",
        ),
        Episode(
            episode_id="r5-first-zone",
            family="read_answer",
            prompt=("Read the first line of data/segment_01.txt with the "
                    "read_file tool, using its offset and limit arguments, and "
                    "report the zone named on that line."),
            matchers=(CallMatcher("read_file", _arg_contains("path", "segment_01")),),
            expect_answer=_segment_one_zone(),
        ),
    )

    grep_edit = (
        Episode(
            episode_id="g1-marker-file",
            family="grep_edit",
            prompt=(f"Search the src directory for the string {MARKER_TWO} "
                    "with the grep tool and report which file defines it."),
            matchers=(CallMatcher("grep", _arg_contains("pattern", MARKER_TWO)),),
            expect_answer="module_kilo",
        ),
        Episode(
            episode_id="g2-fix-divisor",
            family="grep_edit",
            prompt=("src/metrics.py divides by len(values) - 1 in mean(). Fix "
                    "it with the edit tool so it divides by len(values). "
                    "Change only that line."),
            matchers=(
                CallMatcher("edit", _arg_contains("path", "metrics")),
                CallMatcher("read_file", _arg_contains("path", "metrics")),
                CallMatcher("grep", _arg_contains("pattern", "len(values)")),
            ),
            goal_check=_metrics_fixed,
        ),
        Episode(
            episode_id="g3-version-bump",
            family="grep_edit",
            prompt=(f"Update the VERSION file from {VERSION_VALUE} to "
                    f"{BUMPED_VERSION_VALUE} using the edit tool."),
            matchers=(
                CallMatcher("edit", _arg_contains("path", "VERSION")),
                CallMatcher("read_file", _arg_contains("path", "VERSION")),
            ),
            goal_check=_version_bumped,
        ),
        Episode(
            episode_id="g4-todo-ticket",
            family="grep_edit",
            prompt=("Find the TODO marker under the notes directory with the "
                    "grep tool and report the ticket id in parentheses."),
            matchers=(
                CallMatcher("grep", _arg_contains("pattern", "TODO")),
                CallMatcher("grep", _arg_contains("pattern", "rc-")),
                CallMatcher("read_file", _arg_contains("path", "todo")),
            ),
            expect_answer=TODO_TICKET,
        ),
        Episode(
            episode_id="g5-port-bump",
            family="grep_edit",
            prompt=(f"Change the port in config/settings.ini from "
                    f"{SETTINGS_PORT} to {BUMPED_SETTINGS_PORT} with the edit "
                    "tool."),
            matchers=(
                CallMatcher("edit", _arg_contains("path", "settings.ini")),
                CallMatcher("read_file", _arg_contains("path", "settings.ini")),
            ),
            goal_check=_port_bumped,
        ),
    )

    command = (
        Episode(
            episode_id="c1-src-count",
            family="command",
            prompt=("Using the bash tool, count how many files are in the src "
                    "directory and report the number."),
            matchers=(CallMatcher("bash", _arg_contains("command", "src")),),
            expect_answer=str(SRC_FILE_COUNT),
        ),
        Episode(
            episode_id="c2-line-count",
            family="command",
            prompt=("Using the bash tool, count the lines in "
                    "data/segment_02.txt and report the number."),
            matchers=(CallMatcher("bash", _arg_contains("command", "segment_02")),),
            expect_answer=SEGMENT_LINES,
        ),
        Episode(
            episode_id="c3-selftest",
            family="command",
            prompt=("Run scripts/selftest.sh with the bash tool and report the "
                    "numeric code it prints."),
            matchers=(CallMatcher("bash", _arg_contains("command", "selftest")),),
            expect_answer=SELFTEST_CODE,
        ),
        Episode(
            episode_id="c4-error-count",
            family="command",
            prompt=("Using the bash tool, count how many lines in "
                    "data/segment_03.txt contain the word error and report "
                    "the count."),
            matchers=(CallMatcher("bash", _arg_contains("command", "segment_03")),),
            expect_answer=_segment_three_errors(),
        ),
        Episode(
            episode_id="c5-build-tag",
            family="command",
            prompt=("Create a file build_tag.txt in the project root "
                    "containing exactly build- followed by the version from "
                    "the VERSION file (for example build-1.2.3). Use the tools "
                    "to read the version and create the file."),
            matchers=(
                CallMatcher("read_file", _arg_contains("path", "VERSION")),
                CallMatcher("bash", _arg_contains("command", "build_tag")),
                CallMatcher("bash", _arg_contains("command", "VERSION")),
            ),
            goal_check=_build_tag_written,
        ),
    )

    return (*read_answer, *grep_edit, *command)
