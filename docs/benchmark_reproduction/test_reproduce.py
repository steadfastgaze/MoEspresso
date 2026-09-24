import argparse
import json
import os
from pathlib import Path

import httpx
import pytest

import common
import grade_one
import run
import score


def response(text="answer", finish="stop", tokens=7):
    return {"model": "resolved-model", "choices": [{"finish_reason": finish,
            "message": {"content": text, "reasoning_content": "reasoning"}}],
            "usage": {"completion_tokens": tokens, "cost": 0.1}}


def protocol():
    return {"base_url": "http://localhost/v1", "token_field": "max_tokens",
            "request": {"model": "model", "stream": False, "max_tokens": 20}}


def question():
    return {"question_id": "synthetic", "category": "language", "task": "typos",
            "turns": ["synthetic prompt"]}


def benchmark_questions():
    return [
        {"question_id": f"{category}-{task}-{index}", "category": category,
         "task": task, "turns": ["synthetic prompt"]}
        for category, tasks in common.TASK_QUOTAS.items()
        for task, count in tasks.items()
        for index in range(count)
    ]


def test_selection_is_derived_by_salted_rank(monkeypatch):
    monkeypatch.setattr(common, "TASK_QUOTAS", {"language": {"typos": 2}})
    rows = [
        {"question_id": f"question-{index}", "category": "language", "task": "typos",
         "livebench_release_date": "2024-01-01", "turns": [f"prompt {index}"]}
        for index in range(8)
    ]
    selected, _ = common.select_questions({"language": rows})
    expected = sorted(rows, key=lambda row: (
        common.selection_rank("language", "typos", row["question_id"]), row["question_id"]
    ))[:2]
    assert selected == expected
    assert not common.active_on_release(rows[0] | {"livebench_release_date": "2025-01-01"})


@pytest.mark.parametrize(("text", "finish", "tokens", "final", "cap"), [
    ("answer", "stop", 1, "answer", False),
    ("<think>reason</think> answer", "stop", 1, "answer", False),
    ("<think>unfinished", "length", 20, "", True),
    ("answer", "stop", 20, "answer", True),
    ("answer", "length", None, "answer", True),
])
def test_answer_and_caps(text, finish, tokens, final, cap):
    fields = common.answer_fields(response(text, finish, tokens), 20)
    assert fields["final_text"] == final
    assert fields["cap_hit"] is cap


@pytest.mark.parametrize("finish", [None, "unknown", "tool_calls", "content_filter"])
def test_incomplete_response_fails(finish):
    with pytest.raises(ValueError, match="incomplete"):
        common.answer_fields(response(finish=finish), 20)


def test_stream_keeps_text_reasoning_usage_and_requires_done():
    chunks = [
        {"choices": [{"delta": {"content": "answer", "reasoning": "thought"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"cost": 0.2}},
    ]
    data = "\n\n".join("data: " + json.dumps(chunk) for chunk in chunks)
    with pytest.raises(ValueError, match="DONE"):
        run.read_stream(httpx.Response(200, text=data))
    saved = run.read_stream(httpx.Response(200, text=data + "\n\ndata: [DONE]\n\n"))
    assert saved["choices"][0]["message"] == {"content": "answer", "reasoning_content": "thought"}
    assert saved["usage"]["cost"] == 0.2


def test_completed_questions_skip_and_mismatch_refuses(tmp_path):
    calls = []
    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=response())
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        run.run_question(client, tmp_path, question(), protocol(), 10)
        run.run_question(client, tmp_path, question(), protocol(), 10)
        changed = protocol() | {"different": True}
        with pytest.raises(ValueError, match="incompatible"):
            run.run_question(client, tmp_path, question(), changed, 10)
    assert len(calls) == 1
    assert calls[0]["messages"] == [{"role": "user", "content": "synthetic prompt"}]
    item = common.read_json(tmp_path / "synthetic.json")
    assert item["response"]["usage"]["cost"] == 0.1
    assert item["answer"]["reasoning_content"] == "reasoning"


def test_uncertain_call_does_not_retry_and_saved_response_recovers(tmp_path):
    def fail(request):
        raise httpx.ReadTimeout("interrupted")
    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(httpx.ReadTimeout):
            run.run_question(client, tmp_path, question(), protocol(), 10)
        with pytest.raises(ValueError, match="uncertain"):
            run.run_question(client, tmp_path, question(), protocol(), 10)
        common.save_json(tmp_path / "synthetic.response.json", response())
        run.run_question(client, tmp_path, question(), protocol(), 10)
    assert (tmp_path / "synthetic.json").is_file()


def test_rejected_request_can_retry_but_invalid_answer_cannot(tmp_path):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(400))) as client:
        with pytest.raises(httpx.HTTPStatusError):
            run.run_question(client, tmp_path, question(), protocol(), 10)
    assert not (tmp_path / "synthetic.pending.json").exists()
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=response(finish=None))
    )) as client:
        with pytest.raises(ValueError):
            run.run_question(client, tmp_path, question(), protocol(), 10)
    assert not (tmp_path / "synthetic.json").exists()
    assert (tmp_path / "synthetic.pending.json").exists()


def test_health_is_shareable_and_boundaries_fail_closed():
    health = {"diagnostics": {"identity": {"package_artifact_id": common.PACKAGE_ID,
              "context_limit": 262144, "template_kwargs": {"enable_thinking": True},
              "pid": 123, "private_path": "/private/example"}},
              "ssd_streaming": {"capacity_per_layer": 223, "cache_routing": {
                  "policy": "prefer-resident", "cache_factor": 2, "protected_routes": 2}}}
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=health))
    with httpx.Client(transport=transport) as client:
        settings = run.server_settings(client, "http://localhost/v1", True)
    assert "private_path" not in json.dumps(settings)
    args = argparse.Namespace(require_capacity=223, require_cache_prior=True,
                              require_package=common.PACKAGE_ID)
    run.check_settings(settings, args)
    settings["capacity_overrides"] = {"3": 224}
    with pytest.raises(ValueError, match="capacity"):
        run.check_settings(settings, args)
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(404))) as client:
        assert run.server_settings(client, "http://localhost/v1") is None
        with pytest.raises(ValueError, match="unavailable"):
            run.server_settings(client, "http://localhost/v1", True)


def test_container_boundary_and_host_guard():
    command = score.container_command("image", "name")
    for option in ("--read-only", "--cap-drop", "--pids-limit", "--memory", "--user"):
        assert option in command
    assert command[command.index("--network") + 1] == "none"
    assert "--mount" not in command and "-v" not in command
    if not Path("/.dockerenv").exists():
        with pytest.raises(RuntimeError, match="host execution"):
            grade_one.official_score({"category": "coding"}, "raise Exception()")


def test_complete_report_is_six_category_macro_and_caps_skip_execution(tmp_path, monkeypatch):
    rows = benchmark_questions()
    common.save_json(tmp_path / "protocol.json", {"version": 1,
                     "selection_sha256": common.SELECTION_SHA256})
    common.save_json(tmp_path / "questions.json", rows)
    monkeypatch.setattr(score, "load_questions", lambda offline: rows)
    monkeypatch.setattr(score, "checked_item", lambda path, q, p: {
        "answer": {"cap_hit": q["category"] == "coding", "final_text": "answer"}})
    calls = []
    def grade(q, answer, image):
        calls.append(q)
        return 1.0
    monkeypatch.setattr(score, "isolated_score", grade)
    report = score.score_run(tmp_path)
    assert len(calls) == 40
    assert report["categories"]["coding"] == 0
    assert report["macro"] == pytest.approx(5 / 6)


def test_changed_public_question_and_saved_answer_fail(tmp_path):
    with pytest.raises(ValueError, match="distinct"):
        common.check_questions([question()])
    expected = benchmark_questions()
    changed = [dict(row) for row in expected]
    changed[0]["turns"] = ["changed"]
    with pytest.raises(ValueError, match="salted selection"):
        common.check_questions(changed, expected)
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=response()))
    with httpx.Client(transport=transport) as client:
        run.run_question(client, tmp_path, question(), protocol(), 10)
    path = tmp_path / "synthetic.json"
    item = common.read_json(path)
    item["answer"]["final_text"] = "tampered"
    common.save_json(path, item)
    with pytest.raises(ValueError, match="differs"):
        common.checked_item(path, question(), protocol())


@pytest.mark.skipif(os.environ.get("TEST_GRADING_IMAGE") != "1", reason="opt-in Docker smoke")
def test_real_container_scores_correct_and_wrong_programs():
    sample = {"question_id": "synthetic", "category": "coding", "task": "LCB_generation",
              "public_test_cases": json.dumps([{"input": "2\n", "output": "4\n",
                                                 "testtype": "stdin"}]),
              "private_test_cases": "[]", "original_json": {"metadata": "{}"}}
    assert score.isolated_score(sample, "```python\nprint(int(input()) * 2)\n```", score.IMAGE) == 1
    assert score.isolated_score(sample, "```python\nprint(0)\n```", score.IMAGE) == 0


@pytest.mark.parametrize("sampling", ["fixed", "provider"])
def test_whole_runner_records_request_and_refuses_changed_resume(tmp_path, monkeypatch, sampling):
    requests = []
    def handle(request):
        if request.method == "GET":
            return httpx.Response(404)
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response())
    client_class = httpx.Client
    monkeypatch.setattr(run.httpx, "Client", lambda **kwargs: client_class(
        transport=httpx.MockTransport(handle), **kwargs))
    monkeypatch.setattr(run, "load_questions", lambda _: [question()])
    monkeypatch.setenv("REPRODUCER_TEST_KEY", "synthetic-secret")
    args = argparse.Namespace(base_url="http://localhost/v1", model="model", out=tmp_path,
        api_key_env="REPRODUCER_TEST_KEY", reasoning_effort="medium", sampling=sampling,
        max_output_tokens=20, token_field="max_tokens", timeout=5, stream=False, openrouter=True,
        offline=True, require_capacity=None, require_cache_prior=False, require_package=None)
    run.run(args)
    run.run(args)
    assert len(requests) == 1
    assert ("temperature" in requests[0]) == (sampling == "fixed")
    assert requests[0]["plugins"] == [{"id": "web", "enabled": False}]
    assert requests[0]["tool_choice"] == "none"
    assert requests[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert "synthetic-secret" not in (tmp_path / "protocol.json").read_text()
    args.reasoning_effort = "high"
    with pytest.raises(ValueError, match="settings changed"):
        run.run(args)


@pytest.mark.skipif(os.environ.get("TEST_GRADING_IMAGE") != "1", reason="opt-in Docker smoke")
def test_all_twelve_official_task_scorers_load():
    questions = common.load_questions(offline=True)
    for task in sorted({q["task"] for q in questions}):
        selected = next(q for q in questions if q["task"] == task)
        assert 0 <= score.isolated_score(selected, "", score.IMAGE) <= 1
