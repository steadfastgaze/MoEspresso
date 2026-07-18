"""Incremental reasoning and answer response shaping."""

from __future__ import annotations

import pytest

from moespresso.runtime.chat_stream import ReasoningSplitter, split_complete_text


@pytest.mark.parametrize(
    "raw,thinking,reasoning,content",
    [
        ("plan</think>answer", True, "plan", "answer"),
        ("<think>plan</think>answer", True, "plan", "answer"),
        ("<think></think>answer", False, "", "answer"),
        ("plain answer", False, "", "plain answer"),
        ("plain legacy answer", True, "", "plain legacy answer"),
    ],
)
def test_split_complete_text(raw, thinking, reasoning, content):
    assert split_complete_text(raw, thinking_enabled=thinking) == (
        reasoning,
        content,
    )


def test_split_complete_text_truncated_thinking_is_reasoning_not_content():
    # Generation stopped before the close marker while the rendered prompt
    # itself opened the reasoning section: the whole string is reasoning.
    # Without the prompt-opened fact the legacy plain-text reading stands.
    truncated = "partial plan that ran out of tok"
    assert split_complete_text(
        truncated, thinking_enabled=True, prompt_opened_thinking=True,
    ) == (truncated, "")
    assert split_complete_text(
        truncated, thinking_enabled=True,
    ) == ("", truncated)
    assert split_complete_text(
        "plan</think>answer", thinking_enabled=True,
        prompt_opened_thinking=True,
    ) == ("plan", "answer")


@pytest.mark.parametrize("cut", range(1, len("<think>plan</think>answer")))
def test_reasoning_splitter_accepts_every_two_chunk_boundary(cut):
    raw = "<think>plan</think>answer"
    events = []
    splitter = ReasoningSplitter(
        thinking_enabled=True,
        emit=lambda kind, text: events.append((kind, text)),
    )
    splitter.push(raw[:cut])
    splitter.push(raw[cut:])
    splitter.finish()

    assert splitter.reasoning == "plan"
    assert splitter.content == "answer"
    assert "".join(text for kind, text in events if kind == "reasoning") == "plan"
    assert "".join(text for kind, text in events if kind == "content") == "answer"


def test_reasoning_splitter_accepts_one_character_chunks_after_prompt_open():
    events = []
    splitter = ReasoningSplitter(
        thinking_enabled=True,
        emit=lambda kind, text: events.append((kind, text)),
    )
    for char in "work\n</think>\nFinal":
        splitter.push(char)
    splitter.finish()

    assert splitter.reasoning == "work\n"
    assert splitter.content == "\nFinal"
    assert all("</think>" not in text for _kind, text in events)


def test_unterminated_reasoning_stays_in_reasoning_channel():
    splitter = ReasoningSplitter(thinking_enabled=True)
    splitter.push("still considering")
    splitter.finish()
    assert splitter.reasoning == "still considering"
    assert splitter.content == ""
