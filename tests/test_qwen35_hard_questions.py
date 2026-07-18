from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from moespresso.correctness.qwen35.hard_questions import (
    QUESTIONS,
    extract_answer,
    run_questions,
    score_response,
)


def test_extract_answer_prefers_answer_tag_and_normalizes():
    assert extract_answer("noise <answer>  `72AKOM`  </answer> tail") == "72AKOM"
    assert extract_answer("  '42'  ") == "42"


def test_score_response_is_exact_not_subjective():
    assert score_response("26", "<answer>26</answer>")["correct"] is True
    assert score_response("26", "The answer is 26.")["correct"] is False


def test_run_questions_loads_once_renders_and_scores(monkeypatch):
    calls = {"load": 0, "generate": 0}

    class Tokenizer:
        chat_template = "{{ enable_thinking }}"

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            assert kwargs["add_generation_prompt"] is True
            assert [m["role"] for m in messages] == ["system", "user"]
            assert "exact-answer benchmark" in messages[0]["content"]
            return "RENDERED:" + messages[0]["content"]

    def fake_load(package_dir):
        calls["load"] += 1
        assert package_dir == Path("/pkg")
        manifest = {
            "artifact_id": "pkg:test",
            "architecture": {"family": "qwen3_5_moe"},
        }
        return "MODEL", Tokenizer(), manifest

    def fake_generate(_model, _tokenizer, prompt, **kwargs):
        calls["generate"] += 1
        assert prompt.startswith("RENDERED:")
        assert kwargs["max_tokens"] == 12
        expected = QUESTIONS[calls["generate"] - 1].expected
        return SimpleNamespace(
            text=f"<answer>{expected}</answer>",
            finish_reason="stop",
            prompt_tokens=5,
            completion_tokens=3,
            first_token_seconds=0.1,
            generation_seconds=0.2,
        )

    report = run_questions(
        Path("/pkg"),
        limit=2,
        max_tokens=12,
        load_served_model_fn=fake_load,
        generate_with_metadata_fn=fake_generate,
    )

    assert calls == {"load": 1, "generate": 2}
    assert report["question_set_version"] == "qwen35_hard_questions_v2"
    assert report["package_manifest_id"] == "pkg:test"
    assert report["thinking"] == "off"
    assert report["correct"] == 2
    assert report["total"] == 2
