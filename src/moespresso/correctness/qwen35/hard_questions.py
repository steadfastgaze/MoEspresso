"""Hard deterministic Qwen 35B package questions.

This explicit package-comparison harness runs outside the default test suite. It
loads one package, renders with the same runtime prompt seam as serving, and scores tasks
whose answer is a single unambiguous string.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class HardQuestion:
    id: str
    category: str
    expected: str
    prompt: str


QUESTION_SET_VERSION = "qwen35_hard_questions_v2"
SYSTEM_PROMPT = (
    "You are being evaluated by an exact-answer benchmark. Reply with exactly "
    "one tag of the form <answer>VALUE</answer>. Do not explain. Do not add "
    "any other text."
)


QUESTIONS: tuple[HardQuestion, ...] = (
    HardQuestion(
        id="arith_17x23_minus19",
        category="arithmetic",
        expected="372",
        prompt=(
            "Compute 17 * 23 - 19. Return only the final answer inside "
            "<answer>...</answer>."
        ),
    ),
    HardQuestion(
        id="mod_12345_mod97",
        category="arithmetic",
        expected="26",
        prompt=(
            "Compute 12345 modulo 97. Return only the final answer inside "
            "<answer>...</answer>."
        ),
    ),
    HardQuestion(
        id="base16_2a",
        category="conversion",
        expected="42",
        prompt=(
            "Convert hexadecimal 2A to decimal. Return only the final answer "
            "inside <answer>...</answer>."
        ),
    ),
    HardQuestion(
        id="sort_third",
        category="ordering",
        expected="19",
        prompt=(
            "Sort these integers in ascending order: 42, 7, 19, 73, 5. "
            "Return only the third integer in the sorted list inside "
            "<answer>...</answer>."
        ),
    ),
    HardQuestion(
        id="kv_lookup_riga",
        category="lookup",
        expected="159",
        prompt=(
            "Use this mapping: LIMA=314, OSLO=271, RIGA=159, PERTH=808. "
            "Return only the value for RIGA inside <answer>...</answer>."
        ),
    ),
    HardQuestion(
        id="reverse_moka27",
        category="string",
        expected="72AKOM",
        prompt=(
            "Reverse the exact string MOKA27. Return only the reversed string "
            "inside <answer>...</answer>."
        ),
    ),
    HardQuestion(
        id="letter_count_banana_a",
        category="string",
        expected="3",
        prompt=(
            "How many uppercase letter A characters are in BANANA? Return only "
            "the final answer inside <answer>...</answer>."
        ),
    ),
    HardQuestion(
        id="checksum_digits",
        category="checksum",
        expected="45",
        prompt=(
            "For the digit string 9081726354, add all digits. Return only the "
            "sum inside <answer>...</answer>."
        ),
    ),
    HardQuestion(
        id="matrix_cell",
        category="lookup",
        expected="K7",
        prompt=(
            "Table rows: row A has C1=Q9 and C2=B4; row B has C1=K7 and "
            "C2=M2. Return only row B column C1 inside <answer>...</answer>."
        ),
    ),
    HardQuestion(
        id="set_intersection_count",
        category="set",
        expected="3",
        prompt=(
            "Set X={red, blue, green, black, white}. Set Y={yellow, green, "
            "white, red}. Return only the number of elements in X intersect Y "
            "inside <answer>...</answer>."
        ),
    ),
    HardQuestion(
        id="ascii_capital_a",
        category="conversion",
        expected="65",
        prompt=(
            "What is the ASCII decimal code for uppercase A? Return only the "
            "number inside <answer>...</answer>."
        ),
    ),
    HardQuestion(
        id="two_step_parentheses",
        category="arithmetic",
        expected="58",
        prompt=(
            "Compute (14 + 5) * 3 + 1. Return only the final answer inside "
            "<answer>...</answer>."
        ),
    ),
    HardQuestion(
        id="divide_then_add",
        category="arithmetic",
        expected="49",
        prompt=(
            "Compute 96 / 3 + 17. Return only the final answer inside "
            "<answer>...</answer>."
        ),
    ),
    HardQuestion(
        id="mod_9876_mod123",
        category="arithmetic",
        expected="36",
        prompt=(
            "Compute 9876 modulo 123. Return only the final answer inside "
            "<answer>...</answer>."
        ),
    ),
    HardQuestion(
        id="last_digit_product",
        category="arithmetic",
        expected="3",
        prompt=(
            "What is the last digit of 17 * 29? Return only the final answer "
            "inside <answer>...</answer>."
        ),
    ),
    HardQuestion(
        id="alphabet_11",
        category="lookup",
        expected="K",
        prompt=(
            "In the English alphabet, what is the 11th uppercase letter? "
            "Return only the letter inside <answer>...</answer>."
        ),
    ),
    HardQuestion(
        id="even_positions",
        category="string",
        expected="BDFH",
        prompt=(
            "Take the string ABCDEFGH. Read characters in positions 2, 4, 6, "
            "and 8 using one-based indexing. Return only the resulting string "
            "inside <answer>...</answer>."
        ),
    ),
    HardQuestion(
        id="kv_lookup_red",
        category="lookup",
        expected="58",
        prompt=(
            "Use this mapping: blue->41, red->58, green->13, black->92. "
            "Return only the value for red inside <answer>...</answer>."
        ),
    ),
    HardQuestion(
        id="set_union_count",
        category="set",
        expected="6",
        prompt=(
            "Set A={1,2,3,5}. Set B={3,4,5,6}. Return only the number of "
            "distinct elements in A union B inside <answer>...</answer>."
        ),
    ),
    HardQuestion(
        id="descending_second_item",
        category="ordering",
        expected="67",
        prompt=(
            "Sort these integers in descending order, keeping duplicates: "
            "91, 14, 67, 67, 3. Return only the second item inside "
            "<answer>...</answer>."
        ),
    ),
    HardQuestion(
        id="weekday_number",
        category="lookup",
        expected="5",
        prompt=(
            "If Monday is day 1, Tuesday is day 2, and so on, what number is "
            "Friday? Return only the number inside <answer>...</answer>."
        ),
    ),
    HardQuestion(
        id="binary_101101",
        category="conversion",
        expected="45",
        prompt=(
            "Convert binary 101101 to decimal. Return only the number inside "
            "<answer>...</answer>."
        ),
    ),
    HardQuestion(
        id="word_count_red",
        category="string",
        expected="3",
        prompt=(
            "In the sequence 'red blue red green red', how many times does "
            "the word red appear? Return only the number inside "
            "<answer>...</answer>."
        ),
    ),
    HardQuestion(
        id="letter_checksum_cab",
        category="checksum",
        expected="6",
        prompt=(
            "Use A=1, B=2, C=3. What is the sum for CAB? Return only the "
            "number inside <answer>...</answer>."
        ),
    ),
)


_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def normalize_answer(text: str) -> str:
    """Normalize an extracted answer for exact deterministic scoring."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip("`\"' \t\r\n")


def extract_answer(text: str) -> str:
    """Extract the model's claimed answer.

    Prefer the requested XML-ish answer tag. If the model ignores the format,
    use the whole response after light normalization so verbose answers fail
    instead of being judged subjectively.
    """
    match = _ANSWER_TAG_RE.search(text)
    if match:
        return normalize_answer(match.group(1))
    return normalize_answer(text)


def score_response(expected: str, text: str) -> dict:
    extracted = extract_answer(text)
    expected_norm = normalize_answer(expected)
    return {
        "expected": expected_norm,
        "extracted": extracted,
        "correct": extracted == expected_norm,
    }


def _template_kwargs_for_thinking(tokenizer, manifest: dict, thinking: str) -> dict | None:
    if thinking == "default":
        return None
    from moespresso.runtime.thinking import resolve_thinking_kwargs

    return resolve_thinking_kwargs(
        tokenizer,
        thinking=(thinking == "on"),
        family=manifest.get("architecture", {}).get("family"),
    )


def run_questions(
    package_dir: Path,
    *,
    limit: int | None = None,
    max_tokens: int = 32,
    thinking: str = "off",
    temperature: float = 0.0,
    top_p: float = 1.0,
    load_served_model_fn=None,
    generate_with_metadata_fn=None,
) -> dict:
    if load_served_model_fn is None:
        from moespresso.runtime.serve import load_served_model as load_served_model_fn
    if generate_with_metadata_fn is None:
        from moespresso.runtime.serve import (
            generate_with_metadata as generate_with_metadata_fn,
        )
    from moespresso.runtime.http import render_prompt

    model, tokenizer, manifest = load_served_model_fn(package_dir)
    template_kwargs = _template_kwargs_for_thinking(tokenizer, manifest, thinking)
    selected = QUESTIONS[:limit] if limit is not None else QUESTIONS
    results = []
    started = time.perf_counter()
    for question in selected:
        rendered = render_prompt(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question.prompt},
            ],
            tokenizer,
            template_kwargs=template_kwargs,
            prompt_renderer=manifest.get("architecture", {}).get("prompt_renderer"),
        )
        q_started = time.perf_counter()
        result = generate_with_metadata_fn(
            model,
            tokenizer,
            rendered,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        scored = score_response(question.expected, result.text)
        results.append({
            **asdict(question),
            "rendered_prompt_chars": len(rendered),
            "text": result.text,
            "finish_reason": result.finish_reason,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "first_token_seconds": result.first_token_seconds,
            "generation_seconds": result.generation_seconds,
            "wall_seconds": time.perf_counter() - q_started,
            **scored,
        })
    correct = sum(1 for row in results if row["correct"])
    return {
        "artifact_kind": "qwen35_hard_question_run",
        "question_set_version": QUESTION_SET_VERSION,
        "package_dir": str(package_dir),
        "package_manifest_id": manifest.get("artifact_id"),
        "package_family": manifest.get("architecture", {}).get("family"),
        "thinking": thinking,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "total": len(results),
        "correct": correct,
        "accuracy": correct / len(results) if results else 0.0,
        "wall_seconds": time.perf_counter() - started,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--thinking", choices=("off", "on", "default"), default="off")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    report = run_questions(
        args.package_dir,
        limit=args.limit,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text)
    print(
        f"{report['correct']}/{report['total']} correct "
        f"({report['accuracy']:.3f}) package={report['package_manifest_id']}"
    )
    return 0 if report["correct"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
