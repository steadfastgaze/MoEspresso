"""Append-only conversation state for a chat-completions session.

The engine's KV prefix reuse depends on the client resending an unmodified,
growing message array: turn N's request messages must be an exact list prefix
of turn N+1's. This module enforces that discipline structurally. History
entries are deep-copied on append and on read, there is no API that edits or
removes an entry, and the assistant message from a response is stored
verbatim so the next render reproduces the served turn exactly.

The optional session cache key travels as ``metadata.moespresso_cache_key``
on every request of the session. It is a disk-cache eviction grouping hint
only; it never enters the engine's restore safety key.
"""

from __future__ import annotations

import copy


class Conversation:
    """A growing message array plus the session identity it is sent under."""

    def __init__(self, *, session_cache_key: str | None = None,
                 system: str | None = None):
        self.session_cache_key = session_cache_key
        self._messages: list[dict] = []
        if system is not None:
            self._messages.append({"role": "system", "content": system})

    def __len__(self) -> int:
        return len(self._messages)

    @property
    def messages(self) -> tuple[dict, ...]:
        """The history as deep copies; mutating them never touches the store."""
        return tuple(copy.deepcopy(m) for m in self._messages)

    def add_user(self, content: str) -> None:
        if not isinstance(content, str):
            raise ValueError("user content must be a string")
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(self, message: dict) -> None:
        """Append the response message verbatim (content, tool_calls, all keys).

        The serve side requires a ``content`` key on every message, so a
        message without one fails here, at append time, instead of as a 400 on
        the next request.
        """
        if not isinstance(message, dict):
            raise ValueError("assistant message must be a dict")
        if message.get("role") != "assistant":
            raise ValueError("assistant message must have role 'assistant'")
        if "content" not in message:
            raise ValueError("assistant message must carry a 'content' key")
        self._messages.append(copy.deepcopy(message))

    def add_tool_result(self, content: str, *, tool_call_id: str | None = None) -> None:
        if not isinstance(content, str):
            raise ValueError("tool result content must be a string")
        message: dict = {"role": "tool", "content": content}
        if tool_call_id is not None:
            message["tool_call_id"] = tool_call_id
        self._messages.append(message)

    def request_messages(self) -> list[dict]:
        """The message array for the next request body, as deep copies."""
        return [copy.deepcopy(m) for m in self._messages]
