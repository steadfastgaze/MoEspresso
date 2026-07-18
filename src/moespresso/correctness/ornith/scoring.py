"""Answer extraction, exact normalization, and fail-class labeling.

The gate scores an exact benchmark exactly. The model opens each reply with a
think block and states its final answer after `</think>`; with thinking off the
scaffold is pre-closed and the answer text follows directly. Extraction isolates
the stated final answer, normalizes it to a canonical value for its answer form
(integer or reduced fraction), and compares against the code-verified key.

Normalization uses Python integers and `fractions.Fraction` so equivalent
surface forms collapse (24/31 and "24 over 31", 2.25 and 9/4) without a symbolic
math dependency. There is no numeric-tolerance match.

Every scored item receives a fail-class label:

  clean-pass      correct answer, generation stopped on its own (not at the cap)
  pass-at-cap     correct answer, but generation ran to the token cap
  fail-genuine    a wrong answer was stated (or no closing think marker but text)
  fail-truncated  generation hit the token cap without stating an answer

The distinction between fail-genuine and fail-truncated matters: a truncation is
a budget artifact, a genuine wrong answer is a quality signal.
"""

from __future__ import annotations

import re
from fractions import Fraction

FAIL_CLASSES = ("clean-pass", "pass-at-cap", "fail-genuine", "fail-truncated")


def split_think(raw_text: str) -> tuple[str, str]:
    """Return (reasoning_text, post_think_text).

    If a `</think>` marker is present, post_think is everything after the last
    one. If the reply never closes the think block, post_think is empty and the
    caller treats the item as unanswered rather than as a wrong answer.
    """
    if "</think>" in raw_text:
        idx = raw_text.rfind("</think>")
        reasoning = raw_text[:idx].split("<think>")[-1]
        post = raw_text[idx + len("</think>"):]
        return reasoning.strip(), post.strip()
    return raw_text.strip(), ""


# Phrases that commonly precede a final answer; used to locate the answer span.
_ANSWER_CUES = (
    r"final answer\s*[:=]?\s*",
    r"the answer is\s*[:=]?\s*",
    r"answer\s*[:=]\s*",
    r"result\s*[:=]?\s*",
    r"equals?\s*[:=]?\s*",
    r"=\s*",
)


def _strip_wrappers(text: str) -> str:
    """Remove markdown emphasis, latex delimiters, and boxed wrappers."""
    text = text.strip()
    match = re.search(r"\\boxed\{([^{}]*)\}", text)
    if match:
        text = match.group(1)
    text = text.replace("\\(", " ").replace("\\)", " ")
    text = text.replace("\\[", " ").replace("\\]", " ")
    text = text.replace("$$", " ").replace("$", " ")
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", " ").replace("\\;", " ")
    text = text.replace("{", "").replace("}", "")
    return text.strip()


# A fraction, decimal, or integer, possibly signed, possibly inside a
# parenthesized expression from \frac normalization.
_FRACTION_RE = re.compile(r"\(?\s*(-?\d[\d,]*(?:\.\d+)?)\s*\)?\s*/\s*\(?\s*(-?\d[\d,]*(?:\.\d+)?)\s*\)?")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _clean_number(token: str) -> str:
    # Strip thousands separators sitting between digits.
    return re.sub(r"(?<=\d),(?=\d)", "", token).strip()


def normalize_value(text: str, answer_form: str):
    """Return a canonical value (int or Fraction) for a candidate span, or None."""
    text = _strip_wrappers(text)
    # spoken fractions before number extraction.
    text = re.sub(r"(-?\d[\d,]*(?:\.\d+)?)\s+(?:over|out of)\s+(-?\d[\d,]*(?:\.\d+)?)",
                  r"\1/\2", text)
    if answer_form == "fraction":
        frac = _FRACTION_RE.search(text)
        if frac:
            num, den = _clean_number(frac.group(1)), _clean_number(frac.group(2))
            try:
                return Fraction(Fraction(num), Fraction(den))
            except (ValueError, ZeroDivisionError):
                return None
        num = _NUMBER_RE.search(text)
        if num:
            try:
                return Fraction(_clean_number(num.group(0)))
            except ValueError:
                return None
        return None
    if answer_form == "integer":
        # A fraction span (a/b) only counts as an integer answer when it
        # reduces to a whole number; a bare 3/2 is not an integer answer.
        frac = _FRACTION_RE.search(text)
        if frac:
            num, den = _clean_number(frac.group(1)), _clean_number(frac.group(2))
            try:
                value = Fraction(Fraction(num), Fraction(den))
            except (ValueError, ZeroDivisionError):
                return None
            return int(value) if value.denominator == 1 else None
        num = _NUMBER_RE.search(text)
        if not num:
            return None
        cleaned = _clean_number(num.group(0))
        try:
            value = Fraction(cleaned)
        except ValueError:
            return None
        if value.denominator != 1:
            return None
        return int(value)
    raise ValueError(f"unsupported answer_form: {answer_form!r}")


def key_to_value(key_answer: str, answer_form: str):
    """Parse a key answer string (e.g. '24/31', '968') to a canonical value."""
    value = normalize_value(str(key_answer), answer_form)
    if value is None:
        raise ValueError(f"key answer {key_answer!r} does not parse as {answer_form}")
    return value


def candidate_answer_spans(post_think: str) -> list[str]:
    """Yield candidate answer substrings from the post-think text, best first."""
    lowered = post_think.lower()
    cue_spans = []
    for cue in _ANSWER_CUES:
        for match in re.finditer(cue, lowered):
            tail = post_think[match.end():]
            piece = re.split(r"[\n;]", tail, maxsplit=1)[0]
            if piece.strip():
                cue_spans.append(piece)

    fallback_spans = []
    lines = [line for line in post_think.splitlines() if line.strip()]
    if lines:
        fallback_spans.append(lines[-1])
    fallback_spans.append(post_think)

    ordered = []
    seen = set()
    for span in list(reversed(cue_spans)) + fallback_spans:
        stripped = span.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            ordered.append(stripped)
    return ordered


def score_exact_answer(post_think: str, key_answer, answer_form: str) -> dict:
    """Compare an extracted answer against the key with exact normalization.

    Returns a dict with `passed`, `status` (pass, wrong, no_answer),
    `extracted_span`, and `matched_value`.
    """
    key_value = key_to_value(key_answer, answer_form)
    if not post_think.strip():
        return {"passed": False, "status": "no_answer",
                "extracted_span": None, "matched_value": None}
    best_wrong = None
    for span in candidate_answer_spans(post_think):
        value = normalize_value(span, answer_form)
        if value is None:
            continue
        if value == key_value:
            return {"passed": True, "status": "pass",
                    "extracted_span": span, "matched_value": str(value)}
        if best_wrong is None:
            best_wrong = (span, str(value))
    if best_wrong is not None:
        return {"passed": False, "status": "wrong",
                "extracted_span": best_wrong[0], "matched_value": best_wrong[1]}
    return {"passed": False, "status": "no_answer",
            "extracted_span": None, "matched_value": None}


def fail_class(*, passed: bool, hit_cap: bool, answered: bool) -> str:
    """Label an item by outcome class.

    `answered` is whether the model produced a scorable final answer at all
    (a closed think block with post-think text, or a stated answer). `hit_cap`
    is whether generation stopped because it reached the token budget.
    """
    if passed:
        return "pass-at-cap" if hit_cap else "clean-pass"
    if not answered and hit_cap:
        return "fail-truncated"
    return "fail-genuine"


def normalize_text_answer(text: str) -> str:
    """Canonicalize a free-text answer for string-match scoring.

    Lowercases, collapses whitespace, and strips surrounding punctuation and
    markdown so long-context recall is scored by value despite formatting differences.
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip("`\"'*.,:;() \t\r\n")
    return text.lower()
