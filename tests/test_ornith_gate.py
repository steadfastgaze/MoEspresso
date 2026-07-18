from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from moespresso.correctness.ornith import tasks
from moespresso.correctness.ornith.gate import PROFILE, run_gate
from moespresso.correctness.ornith.sandbox import extract_code, run_hidden_tests
from moespresso.correctness.ornith.scoring import (
    FAIL_CLASSES,
    fail_class,
    key_to_value,
    normalize_text_answer,
    normalize_value,
    score_exact_answer,
    split_think,
)


# --- scoring normalization ---------------------------------------------------

def test_normalize_integer_forms():
    assert normalize_value("968", "integer") == 968
    assert normalize_value("The answer is 19,200.", "integer") == 19200
    assert normalize_value("\\boxed{42}", "integer") == 42
    # a non-integer value is not an integer answer
    assert normalize_value("3/2", "integer") is None
    assert normalize_value("no number here", "integer") is None


def test_normalize_fraction_forms():
    assert normalize_value("24/31", "fraction") == Fraction(24, 31)
    assert normalize_value("24 over 31", "fraction") == Fraction(24, 31)
    assert normalize_value("\\frac{3}{32}", "fraction") == Fraction(3, 32)
    assert normalize_value("2.25", "fraction") == Fraction(9, 4)
    # reduces to canonical form
    assert normalize_value("48/62", "fraction") == Fraction(24, 31)


def test_key_to_value_and_unsupported_form():
    assert key_to_value("968", "integer") == 968
    assert key_to_value("3/32", "fraction") == Fraction(3, 32)
    with pytest.raises(ValueError):
        normalize_value("1+sqrt(5)", "radical")


def test_score_exact_answer_pass_wrong_no_answer():
    passed = score_exact_answer("The final answer is 968.", "968", "integer")
    assert passed["passed"] is True and passed["status"] == "pass"

    wrong = score_exact_answer("I get 42 in the end.", "968", "integer")
    assert wrong["passed"] is False and wrong["status"] == "wrong"

    empty = score_exact_answer("", "968", "integer")
    assert empty["passed"] is False and empty["status"] == "no_answer"


def test_score_prefers_closing_cue_over_earlier_value():
    text = "First I estimated 100 but final answer: 968"
    result = score_exact_answer(text, "968", "integer")
    assert result["passed"] is True


def test_split_think_marker_and_missing():
    reasoning, post = split_think("<think>\nwork\n</think>\nAnswer: 5")
    assert "work" in reasoning and post == "Answer: 5"
    reasoning2, post2 = split_think("all reasoning, no marker")
    assert post2 == ""


def test_normalize_text_answer():
    assert normalize_text_answer("  **Paris.** ") == "paris"
    assert normalize_text_answer("`ORCHID-4417`") == "orchid-4417"


# --- fail-class labeling -----------------------------------------------------

def test_fail_class_labels():
    assert fail_class(passed=True, hit_cap=False, answered=True) == "clean-pass"
    assert fail_class(passed=True, hit_cap=True, answered=True) == "pass-at-cap"
    assert fail_class(passed=False, hit_cap=False, answered=True) == "fail-genuine"
    assert fail_class(passed=False, hit_cap=True, answered=True) == "fail-genuine"
    assert fail_class(passed=False, hit_cap=True, answered=False) == "fail-truncated"
    # An unanswered completion that stopped on its own is a genuine failure.
    assert fail_class(passed=False, hit_cap=False, answered=False) == "fail-genuine"


def test_fail_class_values_are_declared():
    for passed in (True, False):
        for hit_cap in (True, False):
            for answered in (True, False):
                assert fail_class(
                    passed=passed, hit_cap=hit_cap, answered=answered) in FAIL_CLASSES


# --- tool-call extraction ----------------------------------------------------

def test_extract_code_from_qwen_xml_tool_call():
    reply = (
        "<tool_call>\n<function=submit_solution>\n<parameter=code>\n"
        "def f(x):\n    return x + 1\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    code = extract_code(reply)
    assert "def f(x):" in code and "return x + 1" in code


def test_extract_code_from_markdown_fence_fallback():
    reply = "Here is my solution:\n```python\ndef g():\n    return 7\n```\n"
    code = extract_code(reply)
    assert code == "def g():\n    return 7"


def test_extract_code_none_when_absent():
    assert extract_code("I cannot help with that.") is None


# --- code-execution sandbox --------------------------------------------------

_GOOD_RLE = (
    "def rle_encode(s):\n"
    "    out = []\n"
    "    i = 0\n"
    "    while i < len(s):\n"
    "        j = i\n"
    "        while j < len(s) and s[j] == s[i]:\n"
    "            j += 1\n"
    "        out.append(s[i] + str(j - i))\n"
    "        i = j\n"
    "    return ''.join(out)\n"
)
_RLE_CASES = [
    {"args": ["aaabbc"], "expected": "a3b2c1"},
    {"args": [""], "expected": ""},
    {"args": ["x"], "expected": "x1"},
]


def test_sandbox_passing_solution():
    res = run_hidden_tests(_GOOD_RLE, "rle_encode", _RLE_CASES, timeout_seconds=10)
    assert res.extracted is True
    assert res.n_passed == res.n_tests == 3
    assert res.error is None and res.timed_out is False


def test_sandbox_failing_solution():
    bad = "def rle_encode(s):\n    return s\n"
    res = run_hidden_tests(bad, "rle_encode", _RLE_CASES, timeout_seconds=10)
    assert res.n_passed < res.n_tests


def test_sandbox_missing_entry():
    other = "def not_it(s):\n    return s\n"
    res = run_hidden_tests(other, "rle_encode", _RLE_CASES, timeout_seconds=10)
    assert res.n_passed == 0 and res.error is not None


def test_sandbox_no_code():
    res = run_hidden_tests(None, "rle_encode", _RLE_CASES, timeout_seconds=10)
    assert res.extracted is False and res.n_passed == 0


def test_sandbox_timeout():
    loop = "def rle_encode(s):\n    while True:\n        pass\n"
    res = run_hidden_tests(loop, "rle_encode", [{"args": ["a"], "expected": "a1"}],
                           timeout_seconds=2)
    assert res.timed_out is True and res.error == "timeout"


def test_sandbox_candidate_exception_is_contained():
    boom = "def rle_encode(s):\n    raise RuntimeError('boom')\n"
    res = run_hidden_tests(boom, "rle_encode", _RLE_CASES, timeout_seconds=10)
    assert res.n_passed == 0 and res.extracted is True and res.timed_out is False


# --- task set integrity ------------------------------------------------------

def test_hard_reasoning_selection_at_most_one_token_hungry():
    hungry = [item for item in tasks.HARD_REASONING if item.token_hungry]
    assert len(hungry) <= 1
    assert len(tasks.HARD_REASONING) == 4


def test_agentic_and_long_context_counts():
    assert len(tasks.AGENTIC_CODING) == 3
    assert len(tasks.LONG_CONTEXT_ITEMS) == 2
    kinds = {item.kind for item in tasks.LONG_CONTEXT_ITEMS}
    assert kinds == {"needle", "aggregation"}


def test_long_context_carries_planted_facts():
    ctx = tasks.build_long_context()
    assert tasks.NEEDLE_TOKEN in ctx
    for number in tasks.PLANTED_NUMBERS:
        assert str(number) in ctx
    # the aggregation answer is the sum of the planted numbers
    agg = next(i for i in tasks.LONG_CONTEXT_ITEMS if i.kind == "aggregation")
    assert agg.expected == str(sum(tasks.PLANTED_NUMBERS))


# --- full runner with fakes (no model) ---------------------------------------

class _FakeTokenizer:
    chat_template = "{{ enable_thinking }}"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["enable_thinking"] is False
        tools = kwargs.get("tools")
        tag = "TOOLS" if tools else "PLAIN"
        return f"{tag}:{messages[-1]['content'][:40]}"


def _fake_manifest():
    return {"artifact_id": "pkg:fake",
            "architecture": {"family": "qwen3_5_moe", "prompt_renderer": None}}


def _synthetic_hard_reasoning_fixture():
    """Create public test data without reading the ignored benchmark fixture."""
    questions = {}
    key = {}
    for answer, item in enumerate(tasks.HARD_REASONING, start=1):
        is_fraction = item.question_number == "6"
        answer_form = "fraction" if is_fraction else "integer"
        claimed = "3/4" if is_fraction else str(answer)
        questions[item.question_number] = {
            "answer_form": answer_form,
            "prompt": f"Synthetic item {item.id}. Return the {answer_form} {claimed}.",
            "topic": "synthetic",
        }
        key[item.question_number] = {"claimed": claimed}
    return questions, key


def test_synthetic_hard_reasoning_fixture_preserves_answer_forms():
    questions, key = _synthetic_hard_reasoning_fixture()

    assert questions["6"]["answer_form"] == "fraction"
    assert key["6"]["claimed"] == "3/4"
    assert {questions[number]["answer_form"] for number in ("1", "2", "8")} == {"integer"}


def _reply_for(prompt):
    """Return a plausible correct reply based on the rendered prompt content."""
    # agentic coding tool prompts render with the TOOLS tag.
    if prompt.startswith("TOOLS"):
        if "rle_encode" in prompt or "run-length" in prompt or "encodes" in prompt:
            code = ("def rle_encode(s):\n out=[]\n i=0\n while i<len(s):\n"
                    "  j=i\n  while j<len(s) and s[j]==s[i]: j+=1\n"
                    "  out.append(s[i]+str(j-i))\n  i=j\n return ''.join(out)\n")
        elif "is_balanced" in prompt or "bracket" in prompt:
            code = ("def is_balanced(s):\n p={')':'(',']':'[','}':'{'}\n st=[]\n"
                    " for c in s:\n  if c in '([{': st.append(c)\n"
                    "  elif c in p:\n   if not st or st.pop()!=p[c]: return False\n"
                    " return not st\n")
        else:
            code = ("import re\nfrom collections import Counter\n"
                    "def most_common_word(text):\n w=re.findall(r'[a-zA-Z]+',text.lower())\n"
                    " if not w: return ''\n c=Counter(w)\n t=max(c.values())\n"
                    " return min(k for k in c if c[k]==t)\n")
        return SimpleNamespace(
            text=("<tool_call>\n<function=submit_solution>\n<parameter=code>\n"
                  f"{code}</parameter>\n</function>\n</tool_call>"),
            finish_reason="stop", prompt_tokens=10, completion_tokens=40,
            generated_token_ids=(), token_logprobs=(), top_logprobs=())
    # long-context questions: route on the appended QUESTION section, which is
    # specific, because the needle fact appears in the shared context of both.
    question_section = prompt.split("# ===== QUESTION =====", 1)[-1]
    if "three numbers" in question_section or "ledger entries" in question_section:
        return SimpleNamespace(text=str(sum(tasks.PLANTED_NUMBERS)),
                               finish_reason="stop", prompt_tokens=10,
                               completion_tokens=4, generated_token_ids=(),
                               token_logprobs=(), top_logprobs=())
    if "access token" in question_section:
        return SimpleNamespace(text=tasks.NEEDLE_TOKEN, finish_reason="stop",
                               prompt_tokens=10, completion_tokens=8,
                               generated_token_ids=(), token_logprobs=(), top_logprobs=())
    # hard reasoning: answer with the code-verified key value.
    return None


def test_run_gate_full_with_fakes(monkeypatch):
    questions, key = _synthetic_hard_reasoning_fixture()
    monkeypatch.setattr(tasks, "load_private_questions", lambda: (questions, key))

    def fake_load(package_dir):
        assert package_dir == Path("/pkg")
        return "MODEL", _FakeTokenizer(), _fake_manifest()

    def fake_render_plain(messages):
        return "PLAIN:" + messages[-1]["content"]

    def fake_render_tools(messages, tools):
        assert tools and tools[0]["function"]["name"] == "submit_solution"
        return "TOOLS:" + messages[-1]["content"]

    def fake_generate(_model, _tokenizer, prompt, *, seed, max_tokens):
        assert seed == 20260709
        assert max_tokens > 0
        reply = _reply_for(prompt)
        if reply is not None:
            return reply
        # hard reasoning: find which question this prompt renders and answer it.
        for item in tasks.HARD_REASONING:
            q = questions[item.question_number]
            if q["prompt"][:40] in prompt:
                claimed = key[item.question_number]["claimed"]
                return SimpleNamespace(
                    text=f"</think>\nThe final answer is {claimed}.",
                    finish_reason="stop", prompt_tokens=10, completion_tokens=20,
                    generated_token_ids=(), token_logprobs=(), top_logprobs=())
        raise AssertionError(f"unmatched hard-reasoning prompt: {prompt[:60]}")

    report = run_gate(
        Path("/pkg"),
        load_fn=fake_load,
        generate_fn=fake_generate,
        render_plain_fn=fake_render_plain,
        render_tools_fn=fake_render_tools,
    )

    assert report["gate_version"] == "ornith_gate_v2"
    assert report["package_manifest_id"] == "pkg:fake"
    assert report["thinking"] == "off"
    assert report["profile"] == PROFILE
    assert report["n_items"] == 9
    assert report["n_passed"] == 9
    assert report["fail_class_counts"].get("clean-pass") == 9
    families = {row["family"] for row in report["items"]}
    assert families == {"hard_reasoning", "agentic_coding", "long_context"}


def test_run_gate_labels_a_genuine_failure(monkeypatch):
    questions, key = _synthetic_hard_reasoning_fixture()
    monkeypatch.setattr(tasks, "load_private_questions", lambda: (questions, key))

    def fake_load(package_dir):
        return "MODEL", _FakeTokenizer(), _fake_manifest()

    def fake_generate(_model, _tokenizer, prompt, *, seed, max_tokens):
        # wrong integer answer for every hard-reasoning item
        return SimpleNamespace(
            text="</think>\nThe answer is 0.", finish_reason="stop",
            prompt_tokens=10, completion_tokens=5,
            generated_token_ids=(), token_logprobs=(), top_logprobs=())

    report = run_gate(
        Path("/pkg"),
        families=("hard_reasoning",),
        load_fn=fake_load,
        generate_fn=fake_generate,
        render_plain_fn=lambda messages: "PLAIN:" + messages[-1]["content"],
        render_tools_fn=lambda messages, tools: "TOOLS",
    )
    assert report["n_passed"] == 0
    assert all(row["fail_class"] == "fail-genuine" for row in report["items"])
