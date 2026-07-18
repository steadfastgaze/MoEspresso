"""Small generation result contract shared by HTTP and serve.

The heavy generation implementation stays in runtime.serve. This module is pure so the
HTTP core can shape structured responses without importing mlx/jang.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ContextLimitError(ValueError):
    """A request whose token span exceeds the model's declared context limit.

    Raised before any generation or cache mutation. The serve layer maps it
    to a 400 so the client learns the limit, its prompt token count, and its
    requested completion budget instead of receiving degraded output from
    positions past the declared range.
    """

    def __init__(self, *, limit: int, prompt_tokens: int, max_tokens: int):
        self.limit = int(limit)
        self.prompt_tokens = int(prompt_tokens)
        self.max_tokens = int(max_tokens)
        super().__init__(
            f"the request exceeds the model's declared context limit of "
            f"{self.limit} tokens: {self.prompt_tokens} prompt tokens plus "
            f"max_tokens {self.max_tokens}"
        )


@dataclass
class GenerationResult:
    text: str
    finish_reason: str = "stop"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    generated_token_ids: tuple[int, ...] = ()
    prompt_cache: Any = None
    token_logprobs: tuple[float, ...] = ()
    top_logprobs: tuple[tuple[dict, ...], ...] = ()
    cache_event: str | None = None
    cache_entries: int | None = None
    cache_bytes: int | None = None
    # first-token latency is the serve lane's headline metric
    first_token_seconds: float | None = None
    generation_seconds: float | None = None
    # disk frontier checkpoints written during this call (None when the writer is off)
    disk_checkpoints_written: int | None = None
    # per-checkpoint blocking write cost in seconds, measured under the serve lock
    disk_checkpoint_write_seconds: tuple[float, ...] = ()


def as_generation_result(value: str | GenerationResult) -> GenerationResult:
    """Normalize legacy string generators and the new structured seam."""
    if isinstance(value, GenerationResult):
        return value
    return GenerationResult(text=value, finish_reason="stop")
