"""Ornith quality gate v2 runner and CLI.

Runs three instrument families against one served package at the model's
recommended sampling profile (temperature 1.0, top_p 0.95, top_k 20, min_p 0,
presence_penalty 1.5) with thinking off, each item on a fixed seed and a measured
token budget. Every item is labeled by fail class. Wall time is recorded per item
and in total.

The runner takes injectable load and generate callables so the scoring,
fail-class, tool-call extraction, and sandbox paths are unit-testable without a
model or a GPU. The default path loads the real serve seam
(`runtime.serve.load_served_model` plus `generate_with_metadata`) and renders
through `runtime.http.render_prompt`, the same load and render the HTTP server
uses. The agentic tasks additionally pass the qwen3_xml tool schema straight to
the tokenizer chat template, because the shared render seam does not forward
tools on the Qwen path.

This manual gate runs only through the `moespresso-ornith-gate` command. Pytest
uses injected callables and never starts the real-package run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from moespresso.correctness.ornith import tasks
from moespresso.correctness.ornith.sandbox import extract_code, run_hidden_tests
from moespresso.correctness.ornith.scoring import (
    fail_class,
    normalize_text_answer,
    score_exact_answer,
    split_think,
)

GATE_VERSION = "ornith_gate_v2"

# The model's recommended reasoning profile. Thinking is off in every render.
PROFILE = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
}


def _default_load(package_dir):
    from moespresso.runtime.serve import load_served_model
    return load_served_model(package_dir)


def _default_generate(model, tokenizer, prompt, *, seed, max_tokens):
    import mlx.core as mx
    from moespresso.runtime.kv_policy import KVPolicy
    from moespresso.runtime.serve import generate_with_metadata

    mx.random.seed(seed)
    return generate_with_metadata(
        model, tokenizer, prompt, prompt_cache=None, kv_policy=KVPolicy(),
        max_tokens=max_tokens, temperature=PROFILE["temperature"],
        top_p=PROFILE["top_p"], top_k=PROFILE["top_k"], min_p=PROFILE["min_p"],
        presence_penalty=PROFILE["presence_penalty"])


def _think_off_kwargs(tokenizer, manifest):
    from moespresso.runtime.thinking import resolve_thinking_kwargs
    family = manifest.get("architecture", {}).get("family")
    return resolve_thinking_kwargs(tokenizer, thinking=False, family=family)


def _render_plain(tokenizer, manifest, think_kwargs, messages):
    from moespresso.runtime.http import render_prompt
    return render_prompt(
        messages, tokenizer, template_kwargs=think_kwargs,
        prompt_renderer=manifest.get("architecture", {}).get("prompt_renderer"))


def _render_with_tools(tokenizer, think_kwargs, messages, tools):
    # The shared render seam drops tools on the Qwen path, so call the chat
    # template directly for the tool schema. Thinking-off flows through the same
    # enable_thinking kwarg the plain render uses.
    return tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True,
        **think_kwargs)


def _finish_flags(result):
    hit_cap = (getattr(result, "finish_reason", None) == "length")
    return hit_cap


def run_hard_reasoning(items, questions, key, render, generate):
    rows = []
    for item in items:
        q = questions[item.question_number]
        answer_form = q["answer_form"]
        prompt = render([{"role": "user", "content": q["prompt"]}])
        t0 = time.perf_counter()
        result = generate(prompt, seed=item.seed, max_tokens=item.max_tokens)
        wall = time.perf_counter() - t0
        _, post = split_think(result.text)
        # With thinking off there is no closing think marker, so the answer is
        # the whole reply body; fall back to it when the split found no post.
        answer_text = post if post.strip() else result.text
        score = score_exact_answer(answer_text,
                                   key[item.question_number]["claimed"],
                                   answer_form)
        hit_cap = _finish_flags(result)
        answered = score["status"] != "no_answer"
        label = fail_class(passed=score["passed"], hit_cap=hit_cap,
                           answered=answered)
        rows.append({
            "family": "hard_reasoning",
            "id": item.id,
            "question_number": item.question_number,
            "topic": q.get("topic"),
            "answer_form": answer_form,
            "passed": score["passed"],
            "status": score["status"],
            "fail_class": label,
            "hit_cap": hit_cap,
            "completion_tokens": result.completion_tokens,
            "max_tokens": item.max_tokens,
            "token_hungry": item.token_hungry,
            "extracted_span": score["extracted_span"],
            "matched_value": score["matched_value"],
            "wall_seconds": round(wall, 2),
        })
    return rows


def run_agentic_coding(items, render_tools, generate, *, timeout_seconds,
                       python_executable=None):
    rows = []
    for task in items:
        messages = [{"role": "user", "content": task.instruction}]
        prompt = render_tools(messages, [tasks.SUBMIT_TOOL])
        t0 = time.perf_counter()
        result = generate(prompt, seed=task.seed, max_tokens=task.max_tokens)
        wall = time.perf_counter() - t0
        _, post = split_think(result.text)
        # Prefer the post-think text for extraction; fall back to the full reply.
        code = extract_code(post) or extract_code(result.text)
        sandbox = run_hidden_tests(
            code, task.entry, list(task.hidden_tests),
            timeout_seconds=timeout_seconds, python_executable=python_executable)
        passed = sandbox.n_passed == sandbox.n_tests and sandbox.n_tests > 0
        hit_cap = _finish_flags(result)
        answered = sandbox.extracted
        label = fail_class(passed=passed, hit_cap=hit_cap, answered=answered)
        rows.append({
            "family": "agentic_coding",
            "id": task.id,
            "entry": task.entry,
            "passed": passed,
            "n_tests": sandbox.n_tests,
            "n_passed": sandbox.n_passed,
            "extracted_code": sandbox.extracted,
            "sandbox_error": sandbox.error,
            "sandbox_timed_out": sandbox.timed_out,
            "fail_class": label,
            "hit_cap": hit_cap,
            "completion_tokens": result.completion_tokens,
            "max_tokens": task.max_tokens,
            "token_hungry": task.token_hungry,
            "wall_seconds": round(wall, 2),
        })
    return rows


def run_long_context(items, context_text, render, generate):
    rows = []
    for item in items:
        content = (
            f"{context_text}\n\n"
            f"# ===== QUESTION =====\n{item.question}"
        )
        prompt = render([{"role": "user", "content": content}])
        t0 = time.perf_counter()
        result = generate(prompt, seed=item.seed, max_tokens=item.max_tokens)
        wall = time.perf_counter() - t0
        _, post = split_think(result.text)
        answer_text = post if post.strip() else result.text
        got = normalize_text_answer(answer_text)
        expected = normalize_text_answer(item.expected)
        passed = expected in got if expected else False
        hit_cap = _finish_flags(result)
        answered = bool(answer_text.strip())
        label = fail_class(passed=passed, hit_cap=hit_cap, answered=answered)
        rows.append({
            "family": "long_context",
            "id": item.id,
            "kind": item.kind,
            "passed": passed,
            "fail_class": label,
            "hit_cap": hit_cap,
            "expected_normalized": expected,
            "got_head": got[:120],
            "completion_tokens": result.completion_tokens,
            "max_tokens": item.max_tokens,
            "token_hungry": item.token_hungry,
            "wall_seconds": round(wall, 2),
        })
    return rows


def run_gate(
    package_dir: Path,
    *,
    families: tuple[str, ...] = ("hard_reasoning", "agentic_coding", "long_context"),
    sandbox_timeout_seconds: float = 10.0,
    load_fn=None,
    generate_fn=None,
    render_plain_fn=None,
    render_tools_fn=None,
    context_builder=None,
    python_executable=None,
) -> dict:
    """Run the gate against a served package and return the result report.

    The `*_fn` and `context_builder` hooks default to the real serve seam and the
    repository context builder; tests inject fakes so the whole runner is
    exercisable without a model.
    """
    load_fn = load_fn or _default_load
    generate_fn = generate_fn or _default_generate
    context_builder = context_builder or tasks.build_long_context

    model, tokenizer, manifest = load_fn(package_dir)
    think_kwargs = _think_off_kwargs(tokenizer, manifest)

    if render_plain_fn is None:
        def render_plain_fn(messages):
            return _render_plain(tokenizer, manifest, think_kwargs, messages)
    if render_tools_fn is None:
        def render_tools_fn(messages, tools):
            return _render_with_tools(tokenizer, think_kwargs, messages, tools)

    def generate(prompt, *, seed, max_tokens):
        return generate_fn(model, tokenizer, prompt, seed=seed, max_tokens=max_tokens)

    started = time.perf_counter()
    rows = []
    if "hard_reasoning" in families:
        questions, key = tasks.load_private_questions()
        rows += run_hard_reasoning(
            tasks.HARD_REASONING, questions, key, render_plain_fn, generate)
    if "agentic_coding" in families:
        rows += run_agentic_coding(
            tasks.AGENTIC_CODING, render_tools_fn, generate,
            timeout_seconds=sandbox_timeout_seconds,
            python_executable=python_executable)
    if "long_context" in families:
        context_text = context_builder()
        rows += run_long_context(
            tasks.LONG_CONTEXT_ITEMS, context_text, render_plain_fn, generate)
    total_wall = time.perf_counter() - started

    passed = sum(1 for row in rows if row["passed"])
    by_class: dict[str, int] = {}
    for row in rows:
        by_class[row["fail_class"]] = by_class.get(row["fail_class"], 0) + 1
    return {
        "artifact_kind": "ornith_gate_v2_run",
        "gate_version": GATE_VERSION,
        "package_dir": str(package_dir),
        "package_manifest_id": manifest.get("artifact_id"),
        "package_family": manifest.get("architecture", {}).get("family"),
        "thinking": "off",
        "profile": PROFILE,
        "families": list(families),
        "n_items": len(rows),
        "n_passed": passed,
        "fail_class_counts": by_class,
        "total_wall_seconds": round(total_wall, 1),
        "items": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--families", default="hard_reasoning,agentic_coding,long_context",
        help="comma-separated subset of families to run")
    parser.add_argument("--sandbox-timeout", type=float, default=10.0)
    args = parser.parse_args(argv)

    families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    report = run_gate(
        args.package_dir, families=families,
        sandbox_timeout_seconds=args.sandbox_timeout)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text)
    print(
        f"{report['n_passed']}/{report['n_items']} passed  "
        f"classes={report['fail_class_counts']}  "
        f"wall={report['total_wall_seconds']}s  "
        f"package={report['package_manifest_id']}")
    for row in report["items"]:
        print(f"  {row['family']:15s} {row['id']:26s} "
              f"{'PASS' if row['passed'] else 'FAIL':4s} "
              f"{row['fail_class']:14s} tok={row.get('completion_tokens')} "
              f"wall={row['wall_seconds']}s")
    return 0 if report["n_passed"] == report["n_items"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
