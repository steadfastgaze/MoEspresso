from __future__ import annotations

import pytest
import numpy as np

from moespresso.correctness.deepseek_v4.stage_capture import (
    _prompt_tokens,
    _read_stage_dump,
    _reference_row,
    _stages,
)


class _Tokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool | None]] = []

    def encode(self, text: str, add_special_tokens: bool | None = None) -> list[int]:
        self.calls.append((text, add_special_tokens))
        return [ord(ch) for ch in text]


def test_rendered_prompt_mode_encodes_exact_text_without_special_tokens(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("<|rendered|>hello", encoding="utf-8")
    tokenizer = _Tokenizer()

    tokens = _prompt_tokens(tokenizer, prompt, "rendered")

    assert tokens == [ord(ch) for ch in "<|rendered|>hello"]
    assert tokenizer.calls == [("<|rendered|>hello", False)]


def test_split_final_requires_ref_row_for_chunked_ds4_dump():
    with pytest.raises(ValueError, match="need --ref-final-row"):
        _reference_row(
            mode="split-final",
            tokens=30_474,
            ref_rows=3,
            final_row=30_473,
            ref_final_row=None,
        )


def test_split_final_accepts_chunked_ds4_ref_row():
    assert _reference_row(
        mode="split-final",
        tokens=30_474,
        ref_rows=3,
        final_row=30_473,
        ref_final_row=2,
    ) == 2


def test_full_mode_still_requires_full_ds4_dump():
    with pytest.raises(ValueError, match="token count mismatch"):
        _reference_row(
            mode="full",
            tokens=30_474,
            ref_rows=3,
            final_row=30_473,
            ref_final_row=2,
        )


def test_stage_parser_rejects_unknown_stage():
    with pytest.raises(Exception, match="unknown stage"):
        _stages("hc_ffn_post,nope")


def test_read_stage_dump_uses_requested_dump_pos(tmp_path):
    prefix = tmp_path / "q3_finalchunk"
    path = tmp_path / "q3_finalchunk_hc_ffn_post-7_pos30471.bin"
    expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    expected.tofile(path)

    got = _read_stage_dump(prefix, "hc_ffn_post", 7, 30471)

    np.testing.assert_array_equal(got, expected)
