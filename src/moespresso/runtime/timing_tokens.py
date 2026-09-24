"""Independent token accounting for text returned by a completion server."""

from __future__ import annotations

from bisect import bisect_right
import hashlib
import importlib.metadata
from pathlib import Path

from moespresso.runtime.diagnostics import _digest


class LocalTokenCounter:
    """Tokenize complete channels and map tokens to their delivery boundaries."""

    def __init__(self, tokenizer, *, source: str):
        self.tokenizer = tokenizer
        template = tokenizer.get_chat_template()
        self.identity = {
            "source": source,
            "class": type(tokenizer).__name__,
            "vocabulary_sha256": _digest(tokenizer.get_vocab()),
            "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "versions": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers")},
        }
        if self.identity["is_fast"]:
            self.identity["pipeline_sha256"] = hashlib.sha256(
                tokenizer.backend_tokenizer.to_str().encode(),
            ).hexdigest()

    @classmethod
    def load(cls, source: str, *, chat_template: Path | None = None):
        # Loading is completed before any request. No weights or remote code run.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            source, local_files_only=True, trust_remote_code=False,
        )
        if chat_template is not None:
            tokenizer.chat_template = chat_template.read_text(encoding="utf-8")
        return cls(tokenizer, source=source)

    def prompt(self, messages, template_kwargs):
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, add_generation_prompt=True, **template_kwargs,
        )
        if not isinstance(ids, list) or not all(type(token) is int for token in ids):
            raise ValueError("local chat template must return one list of token IDs")
        return {
            "tokens": len(ids),
            "token_ids_sha256": _digest(ids),
            "scope": "Local chat template including generation prefix; server rendering may differ.",
        }

    def output(self, events):
        channels = {name: "".join(event.get(name, "") for event in events)
                    for name in ("reasoning", "content")}
        encoded = {}
        ends = {}
        for name, text in channels.items():
            if self.identity["is_fast"]:
                result = self.tokenizer(
                    text, add_special_tokens=False, return_offsets_mapping=True,
                )
                encoded[name] = result["input_ids"]
                # Multiple tokens may occupy the same character span.
                ends[name] = sorted(end for _start, end in result["offset_mapping"])
            else:
                encoded[name] = self.tokenizer.encode(text, add_special_tokens=False)
        boundaries = []
        characters = dict.fromkeys(channels, 0)
        for event in events:
            counts = {}
            for name, text in channels.items():
                characters[name] += len(event.get(name, ""))
                counts[name] = (
                    bisect_right(ends[name], characters[name]) if name in ends
                    else len(self.tokenizer.encode(text[:characters[name]], add_special_tokens=False))
                )
            boundaries.append({
                "seconds": event["seconds"], "tokens": sum(counts.values()),
                "content_tokens": counts["content"], "reasoning_tokens": counts["reasoning"],
            })
        return {
            "tokens": sum(len(ids) for ids in encoded.values()),
            "content_tokens": len(encoded["content"]),
            "reasoning_tokens": len(encoded["reasoning"]),
            "token_ids_sha256": _digest(encoded),
            "method": "complete_text_offsets" if ends else "prefix_retokenization",
            "scope": "Retokenized returned text; hidden tokens, removed markers and stop tokens are not recoverable.",
            "delivery_boundaries": boundaries,
        }
