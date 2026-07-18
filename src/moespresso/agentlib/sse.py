"""Strict Server-Sent Events decoding for Chat Completions clients."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

MAX_EVENT_BYTES = 8 * 1024 * 1024


class SSEError(ValueError):
    """A malformed or incomplete SSE stream."""


def iter_sse_data(lines: Iterable[bytes]) -> Iterator[str]:
    """Yield complete SSE data payloads from a byte-line iterable."""
    parts: list[bytes] = []
    size = 0
    for raw_line in lines:
        line = raw_line.rstrip(b"\r\n")
        if not line:
            if parts:
                try:
                    yield b"\n".join(parts).decode("utf-8")
                except UnicodeDecodeError as e:
                    raise SSEError(f"stream event is not UTF-8: {e}") from e
                parts = []
                size = 0
            continue
        if line.startswith(b":"):
            continue
        field, separator, value = line.partition(b":")
        if field != b"data":
            continue
        if separator and value.startswith(b" "):
            value = value[1:]
        size += len(value)
        if size > MAX_EVENT_BYTES:
            raise SSEError(
                f"stream event exceeds {MAX_EVENT_BYTES} bytes")
        parts.append(value)
    if parts:
        raise SSEError("stream ended inside an SSE event")
