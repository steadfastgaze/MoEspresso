"""Conversation state: append-only history, verbatim assistant turns, session key.

The engine's prefix reuse depends on the client resending an unmodified,
growing message array. These tests pin the structural guarantees: turn N's
request messages are a list prefix of turn N+1's, nothing hands out a mutable
reference into the store, and the assistant message from a response is stored
verbatim.
"""

from __future__ import annotations

import pytest

from moespresso.agentlib import Conversation


def test_system_message_renders_first():
    conv = Conversation(system="Be terse.")
    assert conv.request_messages() == [{"role": "system", "content": "Be terse."}]


def test_no_system_message_by_default():
    conv = Conversation()
    assert conv.request_messages() == []
    assert len(conv) == 0


def test_add_user_appends():
    conv = Conversation()
    conv.add_user("hello")
    conv.add_user("again")
    assert conv.request_messages() == [
        {"role": "user", "content": "hello"},
        {"role": "user", "content": "again"},
    ]


def test_add_user_rejects_non_string():
    conv = Conversation()
    with pytest.raises(ValueError):
        conv.add_user(42)


def test_history_grows_append_only():
    # The list-prefix property: every earlier request's messages are an exact
    # prefix of every later request's messages.
    conv = Conversation(system="sys")
    snapshots = [conv.request_messages()]
    conv.add_user("q1")
    snapshots.append(conv.request_messages())
    conv.add_assistant_message({"role": "assistant", "content": "a1"})
    snapshots.append(conv.request_messages())
    conv.add_user("q2")
    snapshots.append(conv.request_messages())
    for earlier, later in zip(snapshots, snapshots[1:]):
        assert later[: len(earlier)] == earlier


def test_request_messages_are_copies():
    conv = Conversation()
    conv.add_user("original")
    first = conv.request_messages()
    first[0]["content"] = "mutated"
    assert conv.request_messages()[0]["content"] == "original"


def test_messages_property_returns_copies():
    conv = Conversation()
    conv.add_user("original")
    view = conv.messages
    assert isinstance(view, tuple)
    view[0]["content"] = "mutated"
    assert conv.messages[0]["content"] == "original"


def test_assistant_message_stored_verbatim_including_tool_calls():
    conv = Conversation()
    message = {
        "role": "assistant",
        "content": "<think>\nplan\n</think>",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "grep", "arguments": '{"pattern": "x"}'},
        }],
    }
    conv.add_assistant_message(message)
    assert conv.request_messages()[0] == message


def test_assistant_message_is_deep_copied_on_append():
    conv = Conversation()
    message = {"role": "assistant", "content": "a", "tool_calls": []}
    conv.add_assistant_message(message)
    message["tool_calls"].append({"id": "late"})
    assert conv.request_messages()[0]["tool_calls"] == []


def test_assistant_message_requires_assistant_role():
    conv = Conversation()
    with pytest.raises(ValueError, match="role"):
        conv.add_assistant_message({"role": "user", "content": "x"})


def test_assistant_message_requires_content_key():
    # The serve side rejects messages without a content key; fail at append
    # time instead of as a 400 on the next request. A None value is allowed.
    conv = Conversation()
    with pytest.raises(ValueError, match="content"):
        conv.add_assistant_message({"role": "assistant"})
    conv.add_assistant_message({"role": "assistant", "content": None})


def test_assistant_message_must_be_a_dict():
    conv = Conversation()
    with pytest.raises(ValueError):
        conv.add_assistant_message("just text")


def test_tool_result_with_and_without_call_id():
    conv = Conversation()
    conv.add_tool_result("ok output", tool_call_id="call_1")
    conv.add_tool_result("plain output")
    assert conv.request_messages() == [
        {"role": "tool", "content": "ok output", "tool_call_id": "call_1"},
        {"role": "tool", "content": "plain output"},
    ]


def test_tool_result_rejects_non_string():
    conv = Conversation()
    with pytest.raises(ValueError):
        conv.add_tool_result({"not": "a string"})


def test_session_cache_key_carried():
    assert Conversation().session_cache_key is None
    assert Conversation(session_cache_key="road-test-1").session_cache_key == "road-test-1"
