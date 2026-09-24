"""Thin OpenAI-compatible HTTP layer: pure core + handler over a fake generator.

No mlx/jang/model here: the request->response core is pure (inject a fake
`generate`), and the handler is exercised against a real loopback socket with the
same fake. This pins the shaping + routing + validation that wrap the loaded
model. The model build itself is verified separately via the serve adapter.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from moespresso.core.artifact import make_artifact, write_artifact
from moespresso.runtime.http import (
    PackageRequestContract,
    PackageRequestContractError,
    RequestError,
    build_cache_generator,
    chat_completion,
    make_handler,
    package_request_contract,
    render_prompt,
    request_stream_options,
    run_startup_warmup,
    serialized_generator,
    serialized_stats,
)
from moespresso.runtime.generation import ContextLimitError, GenerationResult


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


def _echo(prompt, **opts):
    # Echo back the prompt + opts so tests can assert they were threaded through.
    return f"reply to[{prompt}] mt={opts['max_tokens']} t={opts['temperature']}"


# --- pure core ---

def test_chat_completion_shapes_openai_response():
    req = {"messages": [{"role": "user", "content": "hi"}]}
    resp = chat_completion(req, _echo, model_id="m", created=123)
    assert resp["object"] == "chat.completion"
    assert resp["created"] == 123
    assert resp["model"] == "m"
    choice = resp["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert "reply to[" in choice["message"]["content"]
    assert choice["finish_reason"] == "stop"


def test_chat_completion_splits_reasoning_from_answer():
    def generated(prompt, **opts):
        return GenerationResult(text="plan</think>final")

    response = chat_completion(
        {"messages": [{"role": "user", "content": "hi"}]}, generated)
    message = response["choices"][0]["message"]
    assert message == {
        "role": "assistant",
        "reasoning_content": "plan",
        "content": "final",
    }


@pytest.mark.parametrize(
    "request_data,match",
    [
        ({"stream": "yes"}, "stream must be a boolean"),
        ({"stream_options": {}}, "requires stream=true"),
        ({"stream": True, "stream_options": []}, "must be a JSON object"),
        ({"stream": True, "stream_options": {"extra": True}}, "unknown"),
        ({"stream": True, "stream_options": {"include_usage": 1}}, "boolean"),
    ],
)
def test_stream_options_fail_closed(request_data, match):
    with pytest.raises(RequestError, match=match):
        request_stream_options(request_data)


def test_request_model_overrides_default_model_id():
    req = {"model": "client-name", "messages": [{"role": "user", "content": "x"}]}
    assert chat_completion(req, _echo, model_id="default")["model"] == "client-name"


def test_generation_options_are_threaded_through():
    req = {"messages": [{"role": "user", "content": "x"}],
           "max_tokens": 7, "temperature": 0.1}
    text = chat_completion(req, _echo)["choices"][0]["message"]["content"]
    assert "mt=7" in text and "t=0.1" in text


def test_raw_kv_policy_is_validated_and_threaded_as_explicit_fallback():
    seen = {}

    def fake_generate(prompt, **opts):
        seen.update(opts)
        return "ok"

    req = {"messages": [{"role": "user", "content": "x"}], "live_kv_format": "raw"}
    resp = chat_completion(req, fake_generate)
    assert resp["choices"][0]["message"]["content"] == "ok"
    assert seen["kv_policy"].live_kv_format == "raw"
    assert seen["effective_rendering_id"]


def test_effective_rendering_identity_is_threaded_to_generator():
    seen = {}

    def fake_generate(prompt, **opts):
        seen.update(opts)
        return "ok"

    req = {"messages": [{"role": "user", "content": "x"}],
           "chat_template_kwargs": {"enable_thinking": False}}
    chat_completion(req, fake_generate, rendering_id="base-render")
    assert seen["effective_rendering_id"] != "base-render"


def test_unsupported_kv_policy_is_a_400():
    with pytest.raises(RequestError) as e:
        chat_completion(
            {"messages": [{"role": "user", "content": "x"}],
             "live_kv_format": "mlx_affine_q6"},
            _echo,
        )
    assert e.value.status == 400
    assert "live_kv_format" in e.value.message


def test_q8_kv_policy_is_enabled_and_threaded_to_generator():
    seen = {}

    def fake_generate(prompt, **opts):
        seen.update(opts)
        return "ok"

    resp = chat_completion(
        {"messages": [{"role": "user", "content": "x"}],
         "live_kv_format": "mlx_affine_q8",
         "kv_group_size": 64,
         "quantized_kv_start": 123},
        fake_generate,
    )
    assert resp["choices"][0]["message"]["content"] == "ok"
    assert seen["kv_policy"].live_kv_format == "mlx_affine_q8"
    assert seen["kv_policy"].kv_group_size == 64
    assert seen["kv_policy"].quantized_kv_start == 123


def test_default_kv_policy_is_q8_from_the_first_token():
    seen = {}

    def fake_generate(prompt, **opts):
        seen.update(opts)
        return "ok"

    chat_completion({"messages": [{"role": "user", "content": "x"}]}, fake_generate)
    assert seen["kv_policy"].live_kv_format == "mlx_affine_q8"
    assert seen["kv_policy"].quantized_kv_start == 0


def test_structured_generation_result_shapes_finish_reason_and_usage():
    def fake_generate(prompt, **opts):
        return GenerationResult(
            text="done",
            finish_reason="length",
            prompt_tokens=12,
            completion_tokens=3,
            cached_tokens=7,
            generated_token_ids=(1, 2, 3),
            cache_event="hit",
            cache_entries=2,
            cache_bytes=4096,
        )

    resp = chat_completion({"messages": [{"role": "user", "content": "x"}]},
                           fake_generate)
    assert resp["choices"][0]["finish_reason"] == "length"
    assert resp["choices"][0]["message"]["content"] == "done"
    assert resp["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
        "prompt_tokens_details": {"cached_tokens": 7},
        "prompt_cache": {"event": "hit", "entries": 2, "bytes": 4096},
    }


def test_metadata_cache_key_is_passed_through_to_generate():
    seen = {}

    def fake_generate(prompt, **opts):
        seen.update(opts)
        return "ok"

    chat_completion(
        {"messages": [{"role": "user", "content": "x"}],
         "metadata": {"moespresso_cache_key": "sess-A"}},
        fake_generate,
    )
    assert seen["session_cache_key"] == "sess-A"


def test_absent_metadata_passes_none_session_key():
    seen = {}

    def fake_generate(prompt, **opts):
        seen.update(opts)
        return "ok"

    chat_completion({"messages": [{"role": "user", "content": "x"}]}, fake_generate)
    assert seen["session_cache_key"] is None


def test_metadata_cache_key_must_be_a_string():
    def fake_generate(prompt, **opts):
        return "ok"

    with pytest.raises(RequestError):
        chat_completion(
            {"messages": [{"role": "user", "content": "x"}],
             "metadata": {"moespresso_cache_key": 7}},
            fake_generate,
        )
    with pytest.raises(RequestError):
        chat_completion(
            {"messages": [{"role": "user", "content": "x"}], "metadata": "nope"},
            fake_generate,
        )


def test_disk_checkpoints_written_surfaces_in_prompt_cache_usage():
    def fake_generate(prompt, **opts):
        return GenerationResult(
            text="done",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=1,
            cached_tokens=0,
            cache_event="miss",
            cache_entries=1,
            cache_bytes=512,
            disk_checkpoints_written=2,
            disk_checkpoint_write_seconds=(0.01234567, 0.07654321),
        )

    resp = chat_completion({"messages": [{"role": "user", "content": "x"}]},
                           fake_generate)
    assert resp["usage"]["prompt_cache"] == {
        "event": "miss",
        "entries": 1,
        "bytes": 512,
        "disk_checkpoints_written": 2,
        "disk_checkpoint_write_seconds": [0.012346, 0.076543],
    }


def test_disk_drafter_state_status_is_orthogonal_to_target_cache_event():
    def fake_generate(prompt, **opts):
        return GenerationResult(
            text="done",
            prompt_tokens=10,
            completion_tokens=1,
            cached_tokens=8,
            cache_event="disk_hit",
            cache_entries=1,
            cache_bytes=512,
            disk_attachment_event="missing",
            disk_attachments_written=1,
            disk_attachment_write_seconds=(0.00123456,),
        )

    resp = chat_completion(
        {"messages": [{"role": "user", "content": "x"}]},
        fake_generate,
    )

    assert resp["usage"]["prompt_cache"] == {
        "event": "disk_hit",
        "entries": 1,
        "bytes": 512,
        "disk_drafter_states_written": 1,
        "disk_drafter_state_write_seconds": [0.001235],
        "drafter_state": {"event": "missing"},
    }


def test_disk_restore_and_ready_to_first_token_timings_are_surfaced():
    def fake_generate(prompt, **opts):
        return GenerationResult(
            text="done",
            prompt_tokens=10,
            completion_tokens=1,
            cached_tokens=8,
            cache_event="disk_hit",
            cache_entries=1,
            cache_bytes=512,
            first_token_seconds=0.012345,
            ready_to_first_token_seconds=0.067891,
            generation_seconds=0.2,
            disk_restore_seconds=0.00456789,
            disk_attachment_event="hit",
            disk_attachment_restore_seconds=0.00098765,
        )

    resp = chat_completion(
        {"messages": [{"role": "user", "content": "x"}]},
        fake_generate,
    )

    assert resp["usage"]["prompt_cache"] == {
        "event": "disk_hit",
        "entries": 1,
        "bytes": 512,
        "disk_restore_seconds": 0.004568,
        "drafter_state": {
            "event": "hit",
            "restore_seconds": 0.000988,
        },
    }
    assert resp["usage"]["moespresso"] == {
        "first_token_seconds": 0.0123,
        "ready_to_first_token_seconds": 0.0679,
        "generation_seconds": 0.2,
        "generation_tps": 5.0,
    }


def test_empty_disk_write_timings_are_omitted_from_prompt_cache_usage():
    def fake_generate(prompt, **opts):
        return GenerationResult(
            text="done",
            prompt_tokens=10,
            completion_tokens=1,
            cached_tokens=0,
            cache_event="miss",
            cache_entries=1,
            cache_bytes=512,
            disk_checkpoints_written=1,
            disk_attachments_written=1,
        )

    resp = chat_completion(
        {"messages": [{"role": "user", "content": "x"}]},
        fake_generate,
    )

    prompt_cache = resp["usage"]["prompt_cache"]
    assert "disk_checkpoint_write_seconds" not in prompt_cache
    assert "disk_drafter_state_write_seconds" not in prompt_cache


def test_serialized_generator_holds_lock_while_generating():
    class Lock:
        def __init__(self):
            self.held = False
            self.events = []

        def __enter__(self):
            self.held = True
            self.events.append("enter")

        def __exit__(self, _exc_type, _exc, _tb):
            self.events.append("exit")
            self.held = False

    lock = Lock()
    seen = {}

    def fake_generate(prompt, **opts):
        seen["held"] = lock.held
        seen["opts"] = opts
        return "ok"

    wrapped = serialized_generator(fake_generate, lock)
    assert wrapped("prompt", max_tokens=1) == "ok"
    assert seen == {"held": True, "opts": {"max_tokens": 1}}
    assert lock.events == ["enter", "exit"]


def test_serialized_stats_holds_lock_while_reading_cache_state():
    class Lock:
        def __init__(self):
            self.held = False
            self.events = []

        def __enter__(self):
            self.held = True
            self.events.append("enter")

        def __exit__(self, _exc_type, _exc, _tb):
            self.events.append("exit")
            self.held = False

    lock = Lock()

    def read_stats():
        return {"held": lock.held, "entries": 1}

    wrapped = serialized_stats(read_stats, lock)
    assert wrapped() == {"held": True, "entries": 1}
    assert lock.events == ["enter", "exit"]


def test_build_cache_generator_builds_in_memory_generator(tmp_path):
    memory_calls = []

    def memory_factory(max_size, max_bytes):
        memory_calls.append((max_size, max_bytes))
        return "memory-store"

    gen = build_cache_generator(
        "MODEL",
        "TOK",
        {"artifact_id": "pkg"},
        memory_store_factory=memory_factory,
        prompt_cache_size=3,
        prompt_cache_bytes=123,
    )
    assert gen.cache_store == "memory-store"
    assert gen.model == "MODEL"
    assert memory_calls == [(3, 123)]


# --- declared context limit: resolution and the 400 mapping ---


def test_declared_context_limit_resolves_both_package_shapes():
    from moespresso.runtime.prefix_cache import (
        declared_context_limit,
        effective_context_limit,
    )

    # DeepSeek-V4 Flash: top-level field, the YaRN-scaled ceiling.
    ds4 = {"architecture": {"config": {
        "max_position_embeddings": 1048576,
        "rope_scaling": {"type": "yarn", "factor": 16,
                         "original_max_position_embeddings": 65536},
    }}}
    assert declared_context_limit(ds4) == 1048576

    # Ornith: wrapped multimodal family, text_config nesting.
    ornith = {"architecture": {"config": {
        "text_config": {"max_position_embeddings": 262144},
    }}}
    assert declared_context_limit(ornith) == 262144

    # No declaration: no limit, the pre-existing behavior.
    assert declared_context_limit({"architecture": {"config": {}}}) is None
    assert declared_context_limit({}) is None

    assert effective_context_limit(ds4) == 131072
    assert effective_context_limit(ds4, runtime_default=16384) == 16384
    # An explicit operator limit wins over the runtime's lower default.
    assert effective_context_limit(
        ds4, requested=30000, runtime_default=16384
    ) == 30000
    with pytest.raises(ValueError, match="runtime context default"):
        effective_context_limit(ds4, runtime_default=0)
    assert effective_context_limit(ornith) == 131072
    assert effective_context_limit(
        {"architecture": {"config": {"max_position_embeddings": 65536}}}
    ) == 65536
    assert effective_context_limit(ornith, requested=30000) == 30000
    assert effective_context_limit(ornith, requested=262144) == 262144
    with pytest.raises(ValueError, match="must be >= 1"):
        effective_context_limit(ornith, requested=0)
    with pytest.raises(ValueError, match="exceeds the package context limit"):
        effective_context_limit(ornith, requested=262145)


def test_context_limit_warning_is_generic_and_only_below_128k():
    from moespresso.runtime.prefix_cache import context_limit_warning_lines

    assert context_limit_warning_lines(131072) == ()
    assert context_limit_warning_lines(262144) == ()
    warning = context_limit_warning_lines(65536)
    assert warning == (
        "WARNING: served context is below the 128K usability target.",
        "WARNING: context_limit=65536 usability_target=131072.",
        "WARNING: usability is substantially reduced for long conversations "
        "and agentic work.",
    )


def test_chat_completion_maps_context_limit_error_to_a_400():
    def refusing_generate(prompt, **opts):
        raise ContextLimitError(
            limit=262144, prompt_tokens=260000, max_tokens=4096)

    with pytest.raises(RequestError) as e:
        chat_completion(
            {"messages": [{"role": "user", "content": "hi"}],
             "max_tokens": 4096},
            refusing_generate,
        )
    assert e.value.status == 400
    assert e.value.code == "context_length_exceeded"
    assert e.value.error_type == "invalid_request_error"
    assert "262144" in e.value.message
    assert "260000" in e.value.message
    assert "max_tokens 4096" in e.value.message


def test_build_cache_generator_resolves_the_effective_context_limit():
    manifest = {"artifact_id": "pkg", "architecture": {"config": {
        "max_position_embeddings": 262144}}}
    gen = build_cache_generator(
        "MODEL",
        "TOK",
        manifest,
        memory_store_factory=lambda size, max_bytes: "memory-store",
    )
    assert gen.context_limit == 131072

    overridden = build_cache_generator(
        "MODEL",
        "TOK",
        manifest,
        context_limit=30000,
        memory_store_factory=lambda size, max_bytes: "memory-store",
    )
    assert overridden.context_limit == 30000


# --- sampling parameter pass-through and validation ---


def _capture_generate(seen):
    def generate(prompt, **opts):
        seen.clear()
        seen.update(opts)
        return "ok"
    return generate


def test_sampling_parameters_are_forwarded_to_generate():
    seen = {}
    chat_completion(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "top_k": 20,
            "min_p": 0.25,
            "presence_penalty": 1.5,
        },
        _capture_generate(seen),
    )
    assert seen["top_k"] == 20
    assert seen["min_p"] == 0.25
    assert seen["presence_penalty"] == 1.5


def test_absent_sampling_parameters_leave_the_generate_call_unchanged():
    # The byte-identity contract: a request without sampling parameters must
    # produce exactly the pre-existing generate call shape (no new keys with
    # default values), so behavior cannot drift for existing clients.
    seen = {}
    chat_completion(
        {"messages": [{"role": "user", "content": "hi"}]},
        _capture_generate(seen),
    )
    assert set(seen) == {
        "max_tokens", "temperature", "top_p", "kv_policy",
        "effective_rendering_id", "session_cache_key",
    }


def test_package_generation_defaults_apply_only_when_request_omits_fields():
    seen = {}
    contract = PackageRequestContract(
        family="qwen4_exp",
        modality="text",
        generation_defaults={"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    )
    chat_completion(
        {"messages": [{"role": "user", "content": "hi"}]},
        _capture_generate(seen),
        request_contract=contract,
    )
    assert seen["temperature"] == 1.0
    assert seen["top_p"] == 0.95
    assert seen["top_k"] == 20

    chat_completion(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.0,
            "top_p": 0.8,
            "top_k": 7,
        },
        _capture_generate(seen),
        request_contract=contract,
    )
    assert seen["temperature"] == 0.0
    assert seen["top_p"] == 0.8
    assert seen["top_k"] == 7


def test_package_request_contract_reads_only_declared_generation_config(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    config = package / "generation_config.json"
    config.write_text(json.dumps({
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "eos_token_id": [248046, 248044],
    }))
    manifest = {
        "architecture": {"family": "qwen4_exp", "modality": "text"},
        "tokenizer": {"files": [{"path": "generation_config.json"}]},
    }

    contract = package_request_contract(package, manifest)

    assert contract.family == "qwen4_exp"
    assert contract.modality == "text"
    assert contract.generation_defaults == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
    }
    manifest["tokenizer"]["files"] = []
    assert package_request_contract(package, manifest).generation_defaults == {}


def test_non_qwen4_package_ignores_declared_generation_config(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "generation_config.json").write_text(json.dumps({
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.2,
    }))
    manifest = {
        "architecture": {"family": "qwen3_5_moe", "modality": "text"},
        "tokenizer": {"files": [{"path": "generation_config.json"}]},
    }

    contract = package_request_contract(package, manifest)

    assert contract.generation_defaults == {}


def test_qwen4_text_family_alias_uses_declared_generation_config(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "generation_config.json").write_text(json.dumps({
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
    }))
    manifest = {
        "architecture": {"family": "qwen4_exp_text", "modality": "text"},
        "tokenizer": {"files": [{"path": "generation_config.json"}]},
    }

    contract = package_request_contract(package, manifest)

    assert contract.generation_defaults == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
    }


@pytest.mark.parametrize("payload,match", [
    ([], "JSON object"),
    ({"do_sample": "yes"}, "do_sample"),
    ({"temperature": -1}, "non-negative"),
    ({"top_p": 1.1}, "between 0 and 1"),
    ({"top_k": True}, "non-negative integer"),
    ({"repetition_penalty": 1.1}, "neutral value"),
])
def test_package_request_contract_rejects_invalid_generation_defaults(
    tmp_path, payload, match
):
    package = tmp_path / "package"
    package.mkdir()
    (package / "generation_config.json").write_text(json.dumps(payload))
    manifest = {
        "architecture": {"family": "qwen4_exp", "modality": "text"},
        "tokenizer": {"files": [{"path": "generation_config.json"}]},
    }
    with pytest.raises(PackageRequestContractError, match=match):
        package_request_contract(package, manifest)


def test_neutral_repetition_penalty_is_a_no_op():
    seen = {}
    resp = chat_completion(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "repetition_penalty": 1.0,
        },
        _capture_generate(seen),
    )
    assert resp["choices"][0]["message"]["content"] == "ok"
    assert "repetition_penalty" not in seen


def test_non_neutral_repetition_penalty_is_a_clear_400():
    with pytest.raises(RequestError) as e:
        chat_completion(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "repetition_penalty": 1.5,
            },
            _echo,
        )
    assert e.value.status == 400
    assert "repetition_penalty is not implemented" in e.value.message
    assert "presence_penalty" in e.value.message
    assert "top_k" in e.value.message


@pytest.mark.parametrize("field,value,match", [
    ("temperature", -0.1, "non-negative"),
    ("temperature", True, "must be a number"),
    ("temperature", "warm", "must be a number"),
    ("top_p", 1.1, "between 0 and 1"),
    ("top_p", True, "must be a number"),
    ("top_k", -1, "non-negative integer"),
    ("top_k", 2.5, "non-negative integer"),
    ("top_k", True, "non-negative integer"),
    ("top_k", "twenty", "non-negative integer"),
    ("min_p", 1.5, "between 0 and 1"),
    ("min_p", -0.1, "between 0 and 1"),
    ("min_p", "small", "must be a number"),
    ("presence_penalty", "big", "must be a number"),
    ("repetition_penalty", "none", "must be a number"),
])
def test_malformed_sampling_parameters_are_400s(field, value, match):
    with pytest.raises(RequestError, match=match) as e:
        chat_completion(
            {"messages": [{"role": "user", "content": "hi"}], field: value},
            _echo,
        )
    assert e.value.status == 400


def test_missing_messages_is_a_400():
    with pytest.raises(RequestError) as e:
        chat_completion({}, _echo)
    assert e.value.status == 400


def test_message_without_content_is_a_400():
    with pytest.raises(RequestError) as e:
        chat_completion({"messages": [{"role": "user"}]}, _echo)
    assert e.value.status == 400


def test_render_prompt_falls_back_without_tokenizer():
    p = render_prompt([{"role": "user", "content": "hello"}])
    assert "user: hello" in p and p.strip().endswith("assistant:")


def test_render_prompt_uses_chat_template_when_present():
    class Tok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
            assert tokenize is False and add_generation_prompt is True
            # render_prompt applies MoEspresso's thinking defaults
            assert kwargs.get("enable_thinking") is True
            assert kwargs.get("preserve_thinking") is True
            return "TEMPLATED:" + messages[0]["content"]

    assert render_prompt([{"role": "user", "content": "hi"}], Tok()) == "TEMPLATED:hi"


def test_startup_warmup_uses_isolated_deterministic_generation():
    seen = {}
    ticks = iter((10.0, 12.5))

    class Tok:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["tokenize"] is False
            assert kwargs["add_generation_prompt"] is True
            return "PROMPT:" + messages[0]["content"]

    def fake_generate(model, tokenizer, prompt, **kwargs):
        seen.update({
            "model": model,
            "tokenizer": tokenizer,
            "prompt": prompt,
            **kwargs,
        })

    elapsed = run_startup_warmup(
        "MODEL", Tok(), generate_fn=fake_generate, clock=lambda: next(ticks))

    assert elapsed == 2.5
    assert seen["model"] == "MODEL"
    assert seen["prompt"] == "PROMPT:Warm up."
    assert seen["prompt_cache"] is None
    assert seen["max_tokens"] == 4
    assert seen["temperature"] == 0.0
    assert seen["top_p"] == 1.0
    assert seen["kv_policy"].live_kv_format == "mlx_affine_q8"


def test_qwen4_startup_warmup_uses_package_owned_live_cache_policy():
    seen = {}

    class Tok:
        def apply_chat_template(self, messages, **_kwargs):
            return messages[0]["content"]

    def fake_generate(_model, _tokenizer, _prompt, **kwargs):
        seen.update(kwargs)

    run_startup_warmup(
        "MODEL",
        Tok(),
        family="qwen4_exp",
        generate_fn=fake_generate,
        clock=lambda: 1.0,
    )

    assert seen["kv_policy"].live_kv_format == "raw"


# --- handler over a real loopback socket (still no model) ---

@pytest.fixture
def server():
    handler = make_handler(_echo, model_id="testmodel", clock=lambda: 42)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    yield f"http://{host}:{port}"
    httpd.shutdown()
    httpd.server_close()


def _post(base, path, payload):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return r.status, json.loads(r.read())


def test_health_endpoint(server):
    with urllib.request.urlopen(server + "/health") as r:
        assert r.status == 200
        body = json.loads(r.read())
    assert body == {"status": "ok", "model": "testmodel"}


def test_health_endpoint_can_report_prompt_cache_stats():
    handler = make_handler(
        _echo,
        model_id="testmodel",
        stats=lambda: {
            "default_live_kv_format": "mlx_affine_q8",
            "supported_live_kv_formats": ["raw", "mlx_affine_q8"],
            "entries": 2,
            "bytes": 4096,
        },
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health") as r:
            assert r.status == 200
            body = json.loads(r.read())
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert body == {
        "status": "ok",
        "model": "testmodel",
        "prompt_cache": {
            "default_live_kv_format": "mlx_affine_q8",
            "supported_live_kv_formats": ["raw", "mlx_affine_q8"],
            "entries": 2,
            "bytes": 4096,
        },
    }


def test_health_endpoint_can_report_ssd_streaming_stats():
    handler = make_handler(
        _echo,
        model_id="testmodel",
        runtime_stats=lambda: {
            "enabled": True,
            "switch_modules": 40,
            "expert_misses": 3,
        },
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health") as r:
            assert r.status == 200
            body = json.loads(r.read())
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert body == {
        "status": "ok",
        "model": "testmodel",
        "ssd_streaming": {
            "enabled": True,
            "switch_modules": 40,
            "expert_misses": 3,
        },
    }


def test_chat_completions_endpoint(server):
    status, body = _post(
        server, "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "ping"}], "max_tokens": 5})
    assert status == 200
    assert body["created"] == 42  # injected clock
    assert "reply to[" in body["choices"][0]["message"]["content"]
    assert "mt=5" in body["choices"][0]["message"]["content"]


def test_chat_completions_streams_reasoning_content_finish_and_usage():
    raw = "plan</think>final"

    def generate(prompt, **opts):
        opts["ready_callback"]()
        opts["progress_callback"](2, 4)
        for step, text in enumerate(("pl", "an</thi", "nk>fi", "nal"), start=1):
            response = type("Response", (), {"text": text})()
            opts["response_callback"](step, response)
        return GenerationResult(
            text=raw,
            finish_reason="stop",
            prompt_tokens=4,
            completion_tokens=4,
            generated_token_ids=(1, 2, 3, 4),
        )

    handler = make_handler(generate, model_id="stream-model", clock=lambda: 42)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions",
            data=json.dumps({
                "messages": [{"role": "user", "content": "ping"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            assert response.headers.get_content_type() == "text/event-stream"
            payload = response.read().decode()
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert ": prefill\n\n" in payload
    records = [
        line.removeprefix("data: ")
        for line in payload.splitlines()
        if line.startswith("data: ")
    ]
    assert records[-1] == "[DONE]"
    chunks = [json.loads(record) for record in records[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    reasoning = "".join(
        chunk["choices"][0]["delta"].get("reasoning_content", "")
        for chunk in chunks if chunk["choices"])
    content = "".join(
        chunk["choices"][0]["delta"].get("content", "")
        for chunk in chunks if chunk["choices"])
    assert reasoning == "plan"
    assert content == "final"
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["total_tokens"] == 8


def test_bad_request_returns_400(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(server, "/v1/chat/completions", {"no": "messages"})
    assert e.value.code == 400
    body = json.loads(e.value.read())
    assert "messages" in body["error"]["message"]


@pytest.mark.parametrize("stream", [False, True])
def test_context_limit_error_returns_machine_readable_code_before_stream(stream):
    def refusing_generate(prompt, **opts):
        raise ContextLimitError(
            limit=131072,
            prompt_tokens=99141,
            max_tokens=32000,
        )

    handler = make_handler(refusing_generate, model_id="context-model")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions",
            data=json.dumps({
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 32000,
                "stream": stream,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request)
        assert exc.value.code == 400
        body = json.loads(exc.value.read())
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert body == {"error": {
        "message": (
            "the request exceeds the served context limit of 131072 tokens: "
            "99141 prompt tokens plus max_tokens 32000"
        ),
        "type": "invalid_request_error",
        "code": "context_length_exceeded",
    }}


def test_context_limit_error_in_started_stream_uses_standard_error_event():
    def started_then_refused(prompt, **opts):
        opts["ready_callback"]()
        raise ContextLimitError(
            limit=131072,
            prompt_tokens=99141,
            max_tokens=32000,
        )

    handler = make_handler(started_then_refused, model_id="context-model")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions",
            data=json.dumps({
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 32000,
                "stream": True,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            assert response.status == 200
            payload = response.read().decode()
    finally:
        httpd.shutdown()
        httpd.server_close()

    records = [
        json.loads(line.removeprefix("data: "))
        for line in payload.splitlines()
        if line.startswith("data: ")
    ]
    assert records[-1] == {
        "type": "error",
        "error": {
            "message": (
                "the request exceeds the served context limit of 131072 "
                "tokens: 99141 prompt tokens plus max_tokens 32000"
            ),
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
        },
    }


def test_invalid_json_returns_400(server):
    req = urllib.request.Request(
        server + "/v1/chat/completions", data=b"{not json",
        headers={"Content-Type": "application/json"}, method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 400


def test_unknown_route_is_404(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(server + "/nope")
    assert e.value.code == 404


def test_http_module_imports_without_loading_runtime_dependencies():
    # The core + handler are stdlib-only; importing must not need mlx/jang.
    import importlib

    import moespresso.runtime.http as h
    importlib.reload(h)
    assert hasattr(h, "chat_completion") and hasattr(h, "serve")


def test_main_threads_prompt_cache_options_to_serve(monkeypatch):
    import moespresso.runtime.http as h

    seen = {}

    def fake_serve(package_dir, **kwargs):
        seen["package_dir"] = package_dir
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(h, "serve", fake_serve)

    assert h.main(["/tmp/pkg", "--prompt-cache-size", "4"]) == 0
    assert str(seen["package_dir"]) == "/tmp/pkg"
    assert seen["prompt_cache_size"] == 4
    assert seen["startup_warmup"] is True
    assert seen["max_context_tokens"] is None
    assert seen["min_resident_experts"] is None

    assert h.main([
        "/tmp/pkg",
        "--startup-warmup",
        "off",
        "--max-context-tokens",
        "30000",
        "--min-resident-experts",
        "48",
    ]) == 0
    assert seen["startup_warmup"] is False
    assert seen["max_context_tokens"] == 30000
    assert seen["min_resident_experts"] == 48


def test_serve_maps_deepseek_v4_thinking_selection(monkeypatch, capsys):
    import moespresso.runtime.http as h

    manifest = _deepseek_v4_manifest()

    def fake_load_model(package_dir):
        return "MODEL", "TOKENIZER", manifest

    seen = {}

    class _Stop(Exception):
        pass

    def fake_warmup(model, tokenizer, **kwargs):
        seen["template_kwargs"] = kwargs.get("server_template_kwargs")
        return 1.25

    def fake_build_cache_generator(*_args, **_kwargs):
        raise _Stop

    monkeypatch.setattr(h, "build_cache_generator", fake_build_cache_generator)

    with pytest.raises(_Stop):
        h.serve("/tmp/pkg", thinking="max", load_model_fn=fake_load_model,
                startup_warmup_fn=fake_warmup)

    assert "[serve] thinking=max via=deepseek_v4_contract" in capsys.readouterr().out
    assert seen["template_kwargs"] == {
        "enable_thinking": True,
        "preserve_thinking": True,
        "drop_thinking": False,
        "reasoning_effort": "max",
    }


def test_serve_default_disk_kv_opens_under_the_user_cache(
        monkeypatch, capsys, tmp_path):
    import moespresso.runtime.http as h

    monkeypatch.delenv("MOESPRESSO_DISK_KV", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    def fake_load_model(package_dir):
        raise FileNotFoundError("stop after the disk block")

    rc = h.serve(tmp_path / "pkg", load_model_fn=fake_load_model)

    assert rc == 2
    out = capsys.readouterr().out
    assert "[serve] disk_kv=frontier root=" in out
    assert str(tmp_path / "cache") in out
    assert "stride=1024" in out
    assert "budget=8GiB" in out
    assert "write_depth=16384" in out


def test_serve_degrades_when_default_disk_kv_cannot_open(
        monkeypatch, capsys, tmp_path):
    import moespresso.runtime.disk_kv as disk_kv
    import moespresso.runtime.http as h

    monkeypatch.delenv("MOESPRESSO_DISK_KV", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    def fake_open(config):
        assert config.explicit is False
        raise disk_kv.DiskKVError("disk KV root is already locked")

    monkeypatch.setattr(disk_kv, "open_disk_store", fake_open)

    def fake_load_model(package_dir):
        raise FileNotFoundError("stop after the disk block")

    rc = h.serve(tmp_path / "pkg", load_model_fn=fake_load_model)

    assert rc == 2
    out = capsys.readouterr().out
    assert "[serve] disk_kv=off (disk KV root is already locked)" in out
    assert "FAILED: stop after the disk block" in out


def test_serve_degrades_on_a_real_corrupt_default_store(
        monkeypatch, capsys, tmp_path):
    # Not a mocked DiskKVError: a corrupt index raises a raw JSON error
    # inside the store, which must normalize into the degrade path.
    from moespresso.runtime.disk_kv import default_disk_kv_root
    import moespresso.runtime.http as h

    monkeypatch.delenv("MOESPRESSO_DISK_KV", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    package_dir = tmp_path / "pkg"
    root = default_disk_kv_root(
        package_dir, {"XDG_CACHE_HOME": str(tmp_path / "cache")})
    root.mkdir(parents=True)
    (root / "index.json").write_text("not json", encoding="utf-8")

    def fake_load_model(package_dir):
        raise FileNotFoundError("stop after the disk block")

    rc = h.serve(package_dir, load_model_fn=fake_load_model)

    assert rc == 2
    out = capsys.readouterr().out
    assert "[serve] disk_kv=off (disk KV store" in out
    assert "FAILED: stop after the disk block" in out


def test_serve_fails_when_explicit_disk_kv_cannot_open(
        monkeypatch, capsys, tmp_path):
    import moespresso.runtime.disk_kv as disk_kv
    import moespresso.runtime.http as h

    monkeypatch.setenv("MOESPRESSO_DISK_KV", "frontier")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    def fake_open(config):
        assert config.explicit is True
        raise disk_kv.DiskKVError("disk KV root is already locked")

    monkeypatch.setattr(disk_kv, "open_disk_store", fake_open)

    def fake_load_model(package_dir):
        raise AssertionError("the model must not load after a refused store")

    rc = h.serve(tmp_path / "pkg", load_model_fn=fake_load_model)

    assert rc == 2
    out = capsys.readouterr().out
    assert "FAILED: disk KV root is already locked" in out


def test_serve_rejects_thinking_max_without_effort_mechanism(capsys):
    import moespresso.runtime.http as h

    manifest = make_artifact(
        "package_manifest",
        {"source_root": "qwen"},
        {"tool": "test", "version": "0"},
        status="valid",
        architecture={"family": "qwen3_5_moe"},
        tensors=[],
        files=[],
    )

    def fake_load_model(package_dir):
        return "MODEL", "TOKENIZER", manifest

    rc = h.serve("/tmp/pkg", thinking="max", load_model_fn=fake_load_model)

    assert rc == 2
    out = capsys.readouterr().out
    assert "--thinking max" in out
    assert "reasoning-effort" in out


def test_serve_accepts_deepseek_v4_prompt_cache_bounds(monkeypatch):
    # Store capacity belongs to host resource policy and stays outside package metadata:
    # a DS4 serve must thread the operator bounds into the cache generator.
    import moespresso.runtime.http as h

    manifest = _deepseek_v4_manifest()

    class _Model:
        _moespresso_ssd_streaming_capacity = 48
        _moespresso_ssd_streaming_capacity_overrides = {}

    def fake_load_model(package_dir):
        return _Model(), "TOKENIZER", manifest

    seen = {}

    class _Stop(Exception):
        pass

    def fake_build_cache_generator(model, tokenizer, _mani, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(h, "build_cache_generator", fake_build_cache_generator)

    with pytest.raises(_Stop):
        h.serve(
            "/tmp/pkg",
            prompt_cache_size=4,
            prompt_cache_bytes=1024,
            max_context_tokens=30000,
            min_resident_experts=48,
            startup_warmup=False,
            load_model_fn=fake_load_model,
        )

    assert seen["prompt_cache_size"] == 4
    assert seen["prompt_cache_bytes"] == 1024
    assert seen["context_limit"] == 30000


def test_serve_rejects_deepseek_below_minimum_residency(capsys):
    import moespresso.runtime.http as h

    manifest = _deepseek_v4_manifest()

    class _Model:
        _moespresso_ssd_streaming_capacity = 32
        _moespresso_ssd_streaming_capacity_overrides = {}

    rc = h.serve(
        "/tmp/pkg",
        min_resident_experts=48,
        startup_warmup=False,
        load_model_fn=lambda package_dir: (_Model(), "TOKENIZER", manifest),
    )

    assert rc == 2
    assert "capacity 32 is below the requested minimum of 48" in capsys.readouterr().out


def test_serve_uses_default_context_limit(monkeypatch):
    import moespresso.runtime.http as h

    manifest = _deepseek_v4_manifest()
    seen = {}

    class _Stop(Exception):
        pass

    def fake_build_cache_generator(model, tokenizer, _mani, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(h, "build_cache_generator", fake_build_cache_generator)
    with pytest.raises(_Stop):
        h.serve(
            "/tmp/pkg",
            startup_warmup=False,
            load_model_fn=lambda package_dir: ("MODEL", "TOKENIZER", manifest),
        )

    assert seen["context_limit"] == 131072


def test_serve_reports_runtime_reduced_context_limit(monkeypatch, capsys):
    import moespresso.runtime.http as h

    manifest = _deepseek_v4_manifest()
    seen = {}

    class _Model:
        _moespresso_auto_context_limit = 16384
        _moespresso_auto_context_limit_from = 131072

    class _Stop(Exception):
        pass

    def fake_build_cache_generator(model, tokenizer, _manifest, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(h, "build_cache_generator", fake_build_cache_generator)
    with pytest.raises(_Stop):
        h.serve(
            "/tmp/pkg",
            startup_warmup=False,
            load_model_fn=lambda package_dir: (_Model(), "TOKENIZER", manifest),
        )

    assert seen["context_limit"] == 16384
    output = capsys.readouterr().out
    assert (
        "[serve] context_limit=16384 package_limit=unknown "
        "auto_reduced_from=131072"
        in output
    )
    assert "WARNING: served context is below the 128K usability target" in output
    assert "WARNING: context_limit=16384 usability_target=131072" in output
    assert "WARNING: usability is substantially reduced" in output


def test_serve_binds_manifest_request_contract_to_handler(
    monkeypatch, tmp_path
):
    import moespresso.runtime.http as h

    monkeypatch.setenv("MOESPRESSO_DISK_KV", "off")
    package = tmp_path / "package"
    package.mkdir()
    (package / "generation_config.json").write_text(json.dumps({
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
    }))
    manifest = make_artifact(
        "package_manifest",
        {"source_root": "qwen"},
        {"tool": "test", "version": "0"},
        status="valid",
        architecture={"family": "qwen4_exp", "modality": "text"},
        tokenizer={
            "files": [{"path": "generation_config.json"}],
            "rendering_id": "render",
        },
        tensors=[],
        files=[],
    )
    seen = {}

    class FakeCacheGenerator:
        def __call__(self, *_args, **_kwargs):
            return "generated"

        def cache_stats(self):
            return {}

        def close(self):
            pass

    class FakeHTTPServer:
        def __init__(self, _address, _handler):
            pass

        def serve_forever(self):
            pass

        def server_close(self):
            pass

    def fake_make_handler(*_args, **kwargs):
        seen["contract"] = kwargs["request_contract"]
        return type("Handler", (), {})

    monkeypatch.setattr(h, "build_cache_generator", lambda *_a, **_k: FakeCacheGenerator())
    monkeypatch.setattr(h, "make_handler", fake_make_handler)
    monkeypatch.setattr(h, "HTTPServer", FakeHTTPServer)

    rc = h.serve(
        package,
        startup_warmup=False,
        load_model_fn=lambda _package: ("MODEL", "TOKENIZER", manifest),
    )

    assert rc == 0
    assert seen["contract"] == h.PackageRequestContract(
        family="qwen4_exp",
        modality="text",
        generation_defaults={"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    )


def test_serve_warms_before_announcing_readiness(monkeypatch, capsys):
    import moespresso.runtime.http as h

    manifest = make_artifact(
        "package_manifest",
        {"source_root": "qwen"},
        {"tool": "test", "version": "0"},
        status="valid",
        architecture={"family": "qwen3_5_moe"},
        tensors=[],
        files=[],
    )
    events = []

    def fake_load_model(_package_dir):
        events.append("load")
        return "MODEL", "TOKENIZER", manifest

    def fake_warmup(model, tokenizer, **kwargs):
        events.append("warmup")
        assert (model, tokenizer) == ("MODEL", "TOKENIZER")
        return 1.25

    class FakeCacheGenerator:
        def __call__(self, *_args, **_kwargs):
            return "generated"

        def cache_stats(self):
            return {}

        def close(self):
            events.append("cache-close")

    def fake_build_cache_generator(*_args, **_kwargs):
        events.append("cache")
        return FakeCacheGenerator()

    class FakeHTTPServer:
        def __init__(self, address, _handler):
            events.append(("bind", address))

        def serve_forever(self):
            events.append("serve")

        def server_close(self):
            events.append("server-close")

    monkeypatch.setattr(h, "build_cache_generator", fake_build_cache_generator)
    monkeypatch.setattr(h, "HTTPServer", FakeHTTPServer)

    result = h.serve(
        "/tmp/pkg",
        load_model_fn=fake_load_model,
        startup_warmup_fn=fake_warmup,
        ready_callback=lambda: events.append("ready"),
    )

    assert result == 0
    assert events == [
        "load",
        "warmup",
        "cache",
        ("bind", ("127.0.0.1", 8080)),
        "ready",
        "serve",
        "server-close",
        "cache-close",
    ]
    out = capsys.readouterr().out
    warming = out.index("warming up generation; server is not ready")
    ready = out.index("startup warmup 1.25s; generation ready")
    serving = out.index("serving on")
    assert warming < ready < serving


def test_serve_does_not_announce_readiness_when_warmup_fails(capsys):
    import moespresso.runtime.http as h

    manifest = _deepseek_v4_manifest()
    ready = []

    def fail_warmup(*_args, **_kwargs):
        raise RuntimeError("warmup failed")

    result = h.serve(
        "/tmp/pkg",
        load_model_fn=lambda _package: ("MODEL", "TOKENIZER", manifest),
        startup_warmup_fn=fail_warmup,
        ready_callback=lambda: ready.append(True),
    )

    assert result == 2
    assert ready == []
    out = capsys.readouterr().out
    assert "FAILED: startup warmup: warmup failed" in out
    assert "serving on" not in out


def test_main_passes_deepseek_v4_thinking_selection_to_serve(
    tmp_path, monkeypatch
):
    import moespresso.runtime.http as h

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    write_artifact(pkg / h.PACKAGE_MANIFEST_NAME, _deepseek_v4_manifest())
    seen = {}

    def fake_serve(*_args, **kwargs):
        seen["thinking"] = kwargs["thinking"]
        return 0

    monkeypatch.setattr(h, "serve", fake_serve)

    assert h.main([str(pkg), "--thinking", "max"]) == 0
    assert seen["thinking"] == "max"


def test_main_rejects_thinking_max_without_effort_mechanism_before_loading(
    tmp_path, monkeypatch, capsys
):
    import moespresso.runtime.http as h

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
    write_artifact(pkg / h.PACKAGE_MANIFEST_NAME, manifest)

    def fake_serve(*_args, **_kwargs):
        raise AssertionError("effort preflight must happen before serve/load")

    monkeypatch.setattr(h, "serve", fake_serve)

    rc = h.main([str(pkg), "--thinking", "max"])

    assert rc == 2
    out = capsys.readouterr().out
    assert "--thinking max" in out
    assert "reasoning-effort" in out


def test_response_carries_first_token_latency():
    """TTFT is the serve headline metric: surfaced on every response."""
    from moespresso.runtime.generation import GenerationResult
    from moespresso.runtime.http import chat_completion

    def fake_generate(prompt, **kwargs):
        return GenerationResult(
            text="hi", finish_reason="stop", prompt_tokens=4,
            completion_tokens=2, first_token_seconds=0.42,
            generation_seconds=0.62)

    resp = chat_completion(
        {"messages": [{"role": "user", "content": "hello"}]},
        fake_generate, model_id="m", created=1)
    pg = resp["usage"]["moespresso"]
    assert pg["first_token_seconds"] == 0.42
    assert pg["generation_seconds"] == 0.62
    assert pg["generation_tps"] == round(2 / 0.62, 2)


# --- --thinking on|off resolution (runtime/thinking.py) ---

def test_resolve_thinking_via_template_sniff():
    from moespresso.runtime.thinking import resolve_thinking_kwargs

    class Tok:
        chat_template = "{%- if enable_thinking is defined %}...{%- endif %}"

    assert resolve_thinking_kwargs(Tok(), thinking=False) == {
        "enable_thinking": False}
    assert resolve_thinking_kwargs(Tok(), thinking=True) == {
        "enable_thinking": True}


def test_qwen4_thinking_toggle_adds_medium_reasoning_effort():
    from moespresso.runtime.thinking import (
        ThinkingToggleUnsupported,
        resolve_thinking_kwargs,
    )

    class Tok:
        chat_template = "{%- if enable_thinking is defined %}...{%- endif %}"

    assert resolve_thinking_kwargs(
        Tok(), thinking=True, family="qwen4_exp"
    ) == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    }
    assert resolve_thinking_kwargs(
        Tok(), thinking=False, family="qwen4_exp_text"
    ) == {"enable_thinking": False}

    class MissingSwitch:
        chat_template = "{{ messages }}"

    with pytest.raises(ThinkingToggleUnsupported):
        resolve_thinking_kwargs(
            MissingSwitch(), thinking=True, family="qwen4_exp"
        )


def test_qwen4_contract_defaults_omitted_and_explicit_on_to_medium():
    from moespresso.runtime.http import qwen4_contract_template_kwargs

    class Tok:
        chat_template = "{%- if enable_thinking is defined %}...{%- endif %}"

    assert qwen4_contract_template_kwargs(Tok()) == {
        "reasoning_effort": "medium"
    }
    assert qwen4_contract_template_kwargs(Tok(), "on") == {
        "reasoning_effort": "medium",
        "enable_thinking": True,
    }
    assert qwen4_contract_template_kwargs(Tok(), "off") == {
        "reasoning_effort": "medium",
        "enable_thinking": False,
    }


def test_resolve_thinking_refuses_loudly_for_unknown_family():
    from moespresso.runtime.thinking import (
        ThinkingToggleUnsupported,
        resolve_thinking_kwargs,
    )

    class Tok:
        chat_template = "{{ messages }}"  # no enable_thinking anywhere

    with pytest.raises(ThinkingToggleUnsupported) as e:
        resolve_thinking_kwargs(Tok(), thinking=False, family="gemma4")
    assert "gemma4" in str(e.value)


def test_resolve_thinking_refuses_deepseek_v4_family_adapter_without_jinja_kwarg():
    from moespresso.runtime.thinking import (
        ThinkingToggleUnsupported,
        resolve_thinking_kwargs,
    )

    class Tok:
        chat_template = "{{ messages }}"  # DS4 packages use the MoEspresso renderer

    with pytest.raises(ThinkingToggleUnsupported) as e:
        resolve_thinking_kwargs(Tok(), thinking=False, family="deepseek_v4_flash")
    assert "deepseek_v4_flash" in str(e.value)


def test_server_thinking_flag_reaches_template_and_request_overrides():
    seen_kwargs = {}

    class Tok:
        def apply_chat_template(self, messages, tokenize,
                                add_generation_prompt, **kwargs):
            seen_kwargs.clear()
            seen_kwargs.update(kwargs)
            return "P"

    # server-level --thinking off reaches the template...
    chat_completion(
        {"messages": [{"role": "user", "content": "x"}]}, _echo,
        tokenizer=Tok(), server_template_kwargs={"enable_thinking": False})
    assert seen_kwargs["enable_thinking"] is False
    # ...and a per-request chat_template_kwargs still wins over it
    chat_completion(
        {"messages": [{"role": "user", "content": "x"}],
         "chat_template_kwargs": {"enable_thinking": True}}, _echo,
        tokenizer=Tok(), server_template_kwargs={"enable_thinking": False})
    assert seen_kwargs["enable_thinking"] is True


def test_server_thinking_flag_changes_rendering_identity():
    seen = []

    def fake_generate(prompt, **opts):
        seen.append(opts["effective_rendering_id"])
        return "ok"

    req = {"messages": [{"role": "user", "content": "x"}]}
    chat_completion(req, fake_generate, rendering_id="base")
    chat_completion(req, fake_generate, rendering_id="base",
                    server_template_kwargs={"enable_thinking": False})
    # a thinking-off server must never reuse a thinking-on prefix cache
    assert seen[0] != seen[1]
