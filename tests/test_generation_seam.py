"""Generation/cache seam: token ids in, prompt cache in/out, metadata out.

No model or MLX import is required. The low-level generator accepts injected stock and
raw-greedy stream functions, so the route contract is testable without a GPU/model.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from moespresso.core.artifact import make_artifact, write_artifact
from moespresso.runtime.generation import GenerationResult, as_generation_result
from moespresso.runtime.kv_policy import parse_kv_policy
from moespresso.runtime.serve import (
    _generation_json_payload,
    _ornith_raw_greedy_eligible,
    generate_once,
    generate_with_metadata,
    main,
)


def _deepseek_v4_manifest():
    return make_artifact(
        "package_manifest",
        {"source_root": "deepseek"},
        {"tool": "test", "version": "0"},
        status="valid",
        architecture={"family": "deepseek_v4_flash"},
        tensors=[],
        files=[],
    )


def _resp(text, token, *, finish_reason=None, prompt_tokens=3, generation_tokens=1):
    return SimpleNamespace(
        text=text,
        token=token,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
    )


class _Tokenizer:
    def encode(self, text):
        return list(range(len(text.split())))

    def decode(self, token_ids):
        return "".join(chr(65 + int(i)) for i in token_ids)


def _ornith_model():
    return SimpleNamespace(model_type="qwen3_5_moe")


def test_ornith_raw_greedy_eligibility_is_narrow():
    kwargs = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": None,
        "top_logprobs": None,
    }

    assert _ornith_raw_greedy_eligible(_ornith_model(), **kwargs)
    assert not _ornith_raw_greedy_eligible(
        SimpleNamespace(model_type="deepseek_v4_flash"), **kwargs)

    for name, value in (
        ("temperature", 0.1),
        ("top_p", 0.9),
        ("top_k", 1),
        ("min_p", 0.1),
        ("presence_penalty", 1.0),
        ("top_logprobs", 0),
    ):
        blocked = dict(kwargs)
        blocked[name] = value
        assert not _ornith_raw_greedy_eligible(
            _ornith_model(), **blocked)


def test_generate_with_metadata_uses_owned_ornith_raw_greedy():
    seen = {}
    prompt_cache = ["q8-cache"]

    def fake_stream(**kwargs):
        seen.update(kwargs)
        yield _resp("g", 7, finish_reason="stop")

    result = generate_with_metadata(
        _ornith_model(),
        "TOK",
        [1, 2, 3],
        prompt_cache=prompt_cache,
        kv_policy=parse_kv_policy({}),
        max_tokens=1,
        temperature=0.0,
        raw_greedy_stream_fn=fake_stream,
        logits_processors_factory=lambda **_kwargs: [],
    )

    assert result.generated_token_ids == (7,)
    assert seen["prompt_cache"] is prompt_cache
    assert seen["kv_bits"] == 8
    assert seen["kv_group_size"] == 64
    assert seen["quantized_kv_start"] == 0
    assert "sampler" not in seen
    assert "greedy_no_logprobs" not in seen


def test_generate_with_metadata_uses_requested_custom_sampler_on_ornith(monkeypatch):
    import mlx_lm

    seen = {}
    sampler = object()

    def fake_stream(**kwargs):
        seen.update(kwargs)
        yield _resp("c", 9, finish_reason="stop")

    monkeypatch.setattr(mlx_lm, "stream_generate", fake_stream)
    generate_with_metadata(
        _ornith_model(),
        "TOK",
        [1, 2, 3],
        max_tokens=1,
        temperature=0.0,
        sampler_factory=lambda **_kwargs: sampler,
        logits_processors_factory=lambda **_kwargs: [],
    )

    assert seen["sampler"] is sampler
    assert "greedy_no_logprobs" not in seen


def test_generate_with_metadata_passes_token_ids_cache_and_returns_metadata():
    seen = {}
    cache = ["cache-object"]

    def fake_stream(**kwargs):
        seen.update(kwargs)
        yield _resp("he", 101, generation_tokens=1)
        yield _resp("llo", 102, finish_reason="stop", generation_tokens=2)

    result = generate_with_metadata(
        "MODEL",
        "TOK",
        [7, 8, 9],
        prompt_cache=cache,
        cached_tokens=6,
        max_tokens=2,
        temperature=0.0,
        top_p=1.0,
        stream_generate_fn=fake_stream,
        sampler_factory=lambda **kwargs: ("sampler", kwargs),
    )

    assert seen["prompt"] == [7, 8, 9]
    assert seen["prompt_cache"] is cache
    assert seen["max_tokens"] == 2
    assert seen["sampler"][1]["temp"] == 0.0
    assert result.text == "hello"
    assert result.finish_reason == "stop"
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 2
    assert result.cached_tokens == 6
    assert result.generated_token_ids == (101, 102)
    assert result.prompt_cache is cache


def test_generate_with_metadata_optionally_captures_top_logprobs():
    def fake_stream(**_kwargs):
        yield SimpleNamespace(
            text="C",
            token=2,
            logprobs=np.array([-3.0, -2.0, -0.1, -4.0]),
            finish_reason="stop",
            prompt_tokens=3,
            generation_tokens=1,
        )

    result = generate_with_metadata(
        "MODEL",
        _Tokenizer(),
        [7, 8, 9],
        max_tokens=1,
        top_logprobs=3,
        stream_generate_fn=fake_stream,
        sampler_factory=lambda **kwargs: None,
    )

    assert result.generated_token_ids == (2,)
    assert result.token_logprobs == (-0.1,)
    assert [entry["token_id"] for entry in result.top_logprobs[0]] == [2, 1, 0]
    assert result.top_logprobs[0][0]["token"] == {"text": "C", "bytes": [67]}


def test_generate_with_metadata_defaults_length_when_stream_has_no_final_reason():
    def fake_stream(**kwargs):
        yield _resp("x", 11, generation_tokens=1)

    result = generate_with_metadata(
        "MODEL", "TOK", [1], stream_generate_fn=fake_stream,
        sampler_factory=lambda **kwargs: None,
    )

    assert result.finish_reason == "length"
    assert result.text == "x"


def test_generate_with_metadata_calls_response_callback_after_each_token():
    seen = []

    def fake_stream(**_kwargs):
        yield _resp("a", 21, generation_tokens=1)
        yield _resp("b", 22, finish_reason="stop", generation_tokens=2)

    result = generate_with_metadata(
        "MODEL",
        "TOK",
        [1],
        stream_generate_fn=fake_stream,
        sampler_factory=lambda **kwargs: None,
        response_callback=lambda step, response: seen.append((step, response.token)),
    )

    assert result.text == "ab"
    assert seen == [(1, 21), (2, 22)]


def test_generate_with_metadata_threads_q8_policy_to_mlx_stream_generate():
    seen = {}

    def fake_stream(**kwargs):
        seen.update(kwargs)
        yield _resp("q", 12, finish_reason="stop")

    result = generate_with_metadata(
        "MODEL",
        "TOK",
        [1, 2, 3],
        kv_policy=parse_kv_policy({
            "live_kv_format": "mlx_affine_q8",
            "kv_group_size": 64,
            "quantized_kv_start": 9,
        }),
        stream_generate_fn=fake_stream,
        sampler_factory=lambda **kwargs: None,
    )

    assert result.text == "q"
    assert seen["kv_bits"] == 8
    assert seen["kv_group_size"] == 64
    assert seen["quantized_kv_start"] == 9


def test_generate_with_metadata_threads_model_prefill_step_size():
    seen = {}
    model = SimpleNamespace(
        _moespresso_prefill_step_size=4096,
        _moespresso_prefill_step_size_max_prompt_tokens=4,
    )

    def fake_stream(**kwargs):
        seen.update(kwargs)
        yield _resp("p", 13, finish_reason="stop")

    result = generate_with_metadata(
        model,
        _Tokenizer(),
        "one two three",
        stream_generate_fn=fake_stream,
        sampler_factory=lambda **kwargs: None,
    )

    assert result.text == "p"
    assert seen["prefill_step_size"] == 4096


def test_generate_with_metadata_threads_prompt_progress_callback():
    seen = {}

    def fake_stream(**kwargs):
        seen.update(kwargs)
        kwargs["prompt_progress_callback"](2, 3)
        yield _resp("p", 13, finish_reason="stop")

    progress = []
    result = generate_with_metadata(
        "MODEL",
        "TOK",
        [1, 2, 3],
        stream_generate_fn=fake_stream,
        sampler_factory=lambda **kwargs: None,
        prompt_progress_callback=lambda processed, total: progress.append(
            (processed, total)),
    )

    assert result.text == "p"
    assert "prompt_progress_callback" in seen
    assert progress == [(2, 3)]


def test_generate_with_metadata_skips_model_prefill_step_size_above_ceiling():
    seen = {}
    model = SimpleNamespace(
        _moespresso_prefill_step_size=4096,
        _moespresso_prefill_step_size_max_prompt_tokens=2,
    )

    def fake_stream(**kwargs):
        seen.update(kwargs)
        yield _resp("p", 15, finish_reason="stop")

    result = generate_with_metadata(
        model,
        _Tokenizer(),
        "one two three",
        stream_generate_fn=fake_stream,
        sampler_factory=lambda **kwargs: None,
    )

    assert result.text == "p"
    assert "prefill_step_size" not in seen


def test_generate_with_metadata_skips_model_prefill_step_size_below_floor():
    # The long-prompt gate: a prompt shorter than the min stays on the stock
    # chunk (no prefill_step_size threaded).
    seen = {}
    model = SimpleNamespace(
        _moespresso_prefill_step_size=4096,
        _moespresso_prefill_step_size_min_prompt_tokens=5,
    )

    def fake_stream(**kwargs):
        seen.update(kwargs)
        yield _resp("p", 16, finish_reason="stop")

    result = generate_with_metadata(
        model,
        "TOK",
        [1, 2, 3],  # 3-token prompt, below the 5-token floor
        stream_generate_fn=fake_stream,
        sampler_factory=lambda **kwargs: None,
    )

    assert result.text == "p"
    assert "prefill_step_size" not in seen


def test_generate_with_metadata_applies_model_prefill_step_size_at_floor():
    # A prompt at or above the min gets the promoted chunk.
    seen = {}
    model = SimpleNamespace(
        _moespresso_prefill_step_size=4096,
        _moespresso_prefill_step_size_min_prompt_tokens=3,
    )

    def fake_stream(**kwargs):
        seen.update(kwargs)
        yield _resp("p", 17, finish_reason="stop")

    result = generate_with_metadata(
        model,
        "TOK",
        [1, 2, 3],  # 3-token prompt, at the 3-token floor
        stream_generate_fn=fake_stream,
        sampler_factory=lambda **kwargs: None,
    )

    assert result.text == "p"
    assert seen["prefill_step_size"] == 4096


def test_generate_with_metadata_explicit_prefill_step_size_wins():
    seen = {}
    model = SimpleNamespace(_moespresso_prefill_step_size=4096)

    def fake_stream(**kwargs):
        seen.update(kwargs)
        yield _resp("p", 14, finish_reason="stop")

    result = generate_with_metadata(
        model,
        "TOK",
        [1, 2, 3],
        prefill_step_size=1024,
        stream_generate_fn=fake_stream,
        sampler_factory=lambda **kwargs: None,
    )

    assert result.text == "p"
    assert seen["prefill_step_size"] == 1024


def _fake_prefill_chunks(record):
    def prefill(model, prompt_cache, tokens, plan, *, progress_callback=None,
                total_tokens=None, **kv_kwargs):
        record.update({
            "model": model,
            "prompt_cache": prompt_cache,
            "tokens": list(tokens),
            "plan": list(plan),
            "total_tokens": total_tokens,
            "kv_kwargs": kv_kwargs,
        })
        consumed = 0
        for size in plan:
            consumed += size
            if progress_callback is not None:
                progress_callback(consumed, total_tokens)
        return consumed

    return prefill


def test_generate_with_metadata_prefill_plan_consumes_leading_tokens():
    seen = {}
    record = {}
    cache = ["cache-object"]
    progress = []

    def fake_stream(**kwargs):
        seen.update(kwargs)
        # The tail prefill fires the callback with tail-relative counts; the
        # seam must shift them to whole-prompt counts.
        kwargs["prompt_progress_callback"](2, 3)
        yield _resp("p", 21, finish_reason="stop", prompt_tokens=3)

    result = generate_with_metadata(
        "MODEL",
        "TOK",
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        prompt_cache=cache,
        prefill_plan=[4, 2],
        prefill_step_size=4,
        prompt_progress_callback=lambda processed, total: progress.append(
            (processed, total)),
        stream_generate_fn=fake_stream,
        sampler_factory=lambda **kwargs: None,
        prefill_chunks_fn=_fake_prefill_chunks(record),
    )

    # The executor received the whole prompt, the plan, and the shared cache.
    assert record["tokens"] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert record["plan"] == [4, 2]
    assert record["prompt_cache"] is cache
    assert record["total_tokens"] == 9
    # stream_generate sees only the tail, with the uniform step for it.
    assert seen["prompt"] == [7, 8, 9]
    assert seen["prompt_cache"] is cache
    assert seen["prefill_step_size"] == 4
    # Progress counts the whole prompt: plan chunks first, then the shifted
    # tail firing (6 consumed + 2 processed of 3-token tail, out of 9).
    assert progress == [(4, 9), (6, 9), (8, 9)]
    # prompt_tokens counts the complete prompt, including the cached prefix.
    assert result.prompt_tokens == 9
    assert result.text == "p"


def test_generate_with_metadata_prefill_plan_threads_q8_kv_kwargs():
    record = {}

    def fake_stream(**kwargs):
        yield _resp("p", 22, finish_reason="stop")

    generate_with_metadata(
        "MODEL",
        "TOK",
        [1, 2, 3, 4],
        prompt_cache=["cache"],
        prefill_plan=[2],
        kv_policy=parse_kv_policy(
            {"live_kv_format": "mlx_affine_q8", "kv_group_size": 64}),
        stream_generate_fn=fake_stream,
        sampler_factory=lambda **kwargs: None,
        prefill_chunks_fn=_fake_prefill_chunks(record),
    )

    assert record["kv_kwargs"] == {
        "kv_bits": 8, "kv_group_size": 64, "quantized_kv_start": 0}


def test_generate_with_metadata_prefill_plan_fails_loudly_on_bad_plans():
    def fake_stream(**kwargs):  # pragma: no cover - validation raises first
        yield _resp("p", 23, finish_reason="stop")

    common = {
        "stream_generate_fn": fake_stream,
        "sampler_factory": lambda **kwargs: None,
        "prefill_chunks_fn": _fake_prefill_chunks({}),
    }
    with pytest.raises(ValueError, match="token-id prompt"):
        generate_with_metadata(
            "MODEL", "TOK", "a string prompt", prompt_cache=["c"],
            prefill_plan=[2], **common)
    with pytest.raises(ValueError, match="explicit prompt cache"):
        generate_with_metadata(
            "MODEL", "TOK", [1, 2, 3], prefill_plan=[2], **common)
    with pytest.raises(ValueError, match="positive"):
        generate_with_metadata(
            "MODEL", "TOK", [1, 2, 3], prompt_cache=["c"],
            prefill_plan=[2, 0], **common)
    # The plan must leave at least one prompt token for generation to feed.
    with pytest.raises(ValueError, match="pre-consumed"):
        generate_with_metadata(
            "MODEL", "TOK", [1, 2, 3], prompt_cache=["c"],
            prefill_plan=[3], **common)


def test_generate_with_metadata_empty_plan_is_the_stock_path():
    seen = {}

    def fake_stream(**kwargs):
        seen.update(kwargs)
        yield _resp("p", 24, finish_reason="stop")

    def fail_prefill(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("an empty plan must not invoke the executor")

    result = generate_with_metadata(
        "MODEL",
        "TOK",
        [1, 2, 3],
        prompt_cache=["c"],
        prefill_plan=[],
        stream_generate_fn=fake_stream,
        sampler_factory=lambda **kwargs: None,
        prefill_chunks_fn=fail_prefill,
    )

    assert seen["prompt"] == [1, 2, 3]
    assert result.prompt_tokens == 3


def test_generate_with_metadata_threads_reasoning_sampler_knobs():
    seen = {}

    def fake_stream(**kwargs):
        seen.update(kwargs)
        yield _resp("r", 31, finish_reason="stop")

    result = generate_with_metadata(
        "MODEL",
        "TOK",
        [1, 2, 3],
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        stream_generate_fn=fake_stream,
        sampler_factory=lambda **kwargs: ("sampler", kwargs),
    )

    assert result.text == "r"
    knobs = seen["sampler"][1]
    assert knobs["temp"] == 1.0
    assert knobs["top_p"] == 0.95
    assert knobs["top_k"] == 20
    assert knobs["min_p"] == 0.0


def test_generate_with_metadata_builds_presence_penalty_logits_processor():
    seen = {}
    lp_calls = []

    def fake_stream(**kwargs):
        seen.update(kwargs)
        yield _resp("p", 41, finish_reason="stop")

    def fake_logits_processors(**kwargs):
        lp_calls.append(kwargs)
        return ["presence-processor"]

    result = generate_with_metadata(
        "MODEL",
        "TOK",
        [1, 2, 3],
        presence_penalty=1.5,
        presence_context_size=20,
        stream_generate_fn=fake_stream,
        sampler_factory=lambda **kwargs: None,
        logits_processors_factory=fake_logits_processors,
    )

    assert result.text == "p"
    assert lp_calls == [{"presence_penalty": 1.5, "presence_context_size": 20}]
    assert seen["logits_processors"] == ["presence-processor"]


def test_generate_with_metadata_omits_logits_processor_without_penalty():
    seen = {}

    def fake_stream(**kwargs):
        seen.update(kwargs)
        yield _resp("n", 42, finish_reason="stop")

    def fail_logits_processors(**_kwargs):
        raise AssertionError("no penalty must not build a logits processor")

    result = generate_with_metadata(
        "MODEL",
        "TOK",
        [1, 2, 3],
        presence_penalty=None,
        stream_generate_fn=fake_stream,
        sampler_factory=lambda **kwargs: None,
        logits_processors_factory=fail_logits_processors,
    )

    assert result.text == "n"
    assert "logits_processors" not in seen


def test_generate_once_remains_string_compatibility_wrapper():
    def fake_stream(**kwargs):
        yield _resp("ok", 5, finish_reason="stop")

    text = generate_once(
        "MODEL", "TOK", "ALREADY RENDERED",
        stream_generate_fn=fake_stream,
        sampler_factory=lambda **kwargs: None,
    )

    assert text == "ok"


def test_as_generation_result_preserves_legacy_string_generators():
    result = as_generation_result("plain text")
    assert isinstance(result, GenerationResult)
    assert result.text == "plain text"
    assert result.finish_reason == "stop"
    assert result.prompt_tokens is None


def test_generation_json_payload_is_structured_and_json_safe(tmp_path):
    result = GenerationResult(
        text="Paris is compact.",
        finish_reason="stop",
        prompt_tokens=11,
        completion_tokens=4,
        cached_tokens=2,
        generated_token_ids=(7, 8),
        prompt_cache=object(),
        first_token_seconds=0.25,
        generation_seconds=1.5,
    )

    payload = _generation_json_payload(
        package_dir=tmp_path / "pkg",
        manifest={
            "artifact_id": "pkg:abc123",
            "architecture": {"family": "deepseek_v4_flash"},
        },
        user_prompt="Say Paris.",
        rendered_prompt="<chat>Say Paris.",
        max_tokens=16,
        temperature=0.0,
        top_p=1.0,
        thinking="off",
        result=result,
    )

    assert payload["artifact_kind"] == "generation_smoke"
    assert payload["package_manifest_id"] == "pkg:abc123"
    assert payload["package_family"] == "deepseek_v4_flash"
    assert payload["generated_token_ids"] == [7, 8]
    assert payload["params"]["thinking"] == "off"
    assert "prompt_cache" not in payload
    json.dumps(payload)


def test_generate_main_writes_json_out(tmp_path, monkeypatch, capsys):
    import moespresso.runtime.http as http
    import moespresso.runtime.serve as serve

    manifest = {
        "artifact_id": "pkg:smoke",
        "architecture": {"family": "deepseek_v4_flash"},
        "tensors": [1, 2],
        "files": [1],
    }

    monkeypatch.setattr(
        serve,
        "load_served_model",
        lambda package_dir: ("MODEL", "TOKENIZER", manifest),
    )
    monkeypatch.setattr(
        http,
        "render_prompt",
        lambda messages, tokenizer, **kwargs: "RENDERED:" + messages[0]["content"],
    )
    monkeypatch.setattr(
        serve,
        "generate_with_metadata",
        lambda *args, **kwargs: GenerationResult(
            text="generated text",
            finish_reason="stop",
            prompt_tokens=3,
            completion_tokens=2,
            generated_token_ids=(101, 102),
            first_token_seconds=0.1,
            generation_seconds=0.2,
        ),
    )

    json_out = tmp_path / "generation.json"
    rc = main([
        str(tmp_path / "pkg"),
        "--prompt",
        "hello",
        "--max-tokens",
        "5",
        "--temperature",
        "0",
        "--json-out",
        str(json_out),
    ])

    assert rc == 0
    assert "generated text" in capsys.readouterr().out
    payload = json.loads(json_out.read_text())
    assert payload["text"] == "generated text"
    assert payload["finish_reason"] == "stop"
    assert payload["completion_tokens"] == 2
    assert payload["generated_token_ids"] == [101, 102]
    assert payload["params"]["max_tokens"] == 5
    assert payload["params"]["thinking"] == "off"


def test_generate_main_maps_deepseek_v4_thinking_selections(
    tmp_path, monkeypatch, capsys
):
    import moespresso.runtime.http as http
    import moespresso.runtime.serve as serve

    manifest = {
        "artifact_id": "pkg:smoke",
        "architecture": {"family": "deepseek_v4_flash"},
        "tensors": [1],
        "files": [1],
    }
    seen = {}

    monkeypatch.setattr(
        serve, "load_served_model",
        lambda package_dir: ("MODEL", "TOKENIZER", manifest))

    def fake_render(messages, tokenizer, template_kwargs=None, **kwargs):
        seen["template_kwargs"] = template_kwargs
        return "RENDERED"

    monkeypatch.setattr(http, "render_prompt", fake_render)
    monkeypatch.setattr(
        serve, "generate_with_metadata",
        lambda *args, **kwargs: GenerationResult(text="ok", finish_reason="stop"))

    rc = main([str(tmp_path / "pkg"), "--prompt", "hi", "--thinking", "max"])

    assert rc == 0
    assert "[generate] thinking=max via=deepseek_v4_contract" in capsys.readouterr().out
    assert seen["template_kwargs"] == {
        "enable_thinking": True,
        "preserve_thinking": True,
        "drop_thinking": False,
        "reasoning_effort": "max",
    }


def test_generate_main_rejects_thinking_max_without_effort_mechanism(
    tmp_path, monkeypatch, capsys
):
    import moespresso.runtime.serve as serve

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    manifest = make_artifact(
        "package_manifest",
        {"source_root": "qwen"},
        {"tool": "test", "version": "0"},
        status="valid",
        architecture={"family": "qwen3_5_moe"},
        tensors=[],
        files=[],
    )
    write_artifact(pkg / "package_manifest.json", manifest)

    monkeypatch.setattr(
        serve,
        "load_served_model",
        lambda package_dir: (_ for _ in ()).throw(
            AssertionError("effort preflight must happen before load")
        ),
    )

    rc = main([
        str(pkg),
        "--prompt",
        "hello",
        "--thinking",
        "max",
    ])

    assert rc == 2
    assert "reasoning-effort" in capsys.readouterr().out
