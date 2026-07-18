from __future__ import annotations

import moespresso.runtime.http as http_runtime
from moespresso.runtime.deepseek_v4.renderer import (
    DEEPSEEK_V4_PROMPT_RENDERER,
    ASSISTANT_SP_TOKEN,
    BOS_TOKEN,
    DSML_TOKEN,
    EOS_TOKEN,
    REASONING_EFFORT_MAX,
    THINKING_END_TOKEN,
    THINKING_START_TOKEN,
    USER_SP_TOKEN,
    encode_messages,
)
from moespresso.runtime.http import (
    chat_completion,
    deepseek_v4_contract_template_kwargs,
    effective_kv_policy,
    effective_template_kwargs,
    render_prompt,
    rendering_identity,
)


def test_deepseek_v4_renderer_matches_simple_official_goldens():
    assert (
        encode_messages([{"role": "user", "content": "Hello"}], thinking_mode="chat")
        == f"{BOS_TOKEN}{USER_SP_TOKEN}Hello{ASSISTANT_SP_TOKEN}{THINKING_END_TOKEN}"
    )
    assert (
        encode_messages([{"role": "user", "content": "Hello"}], thinking_mode="thinking")
        == f"{BOS_TOKEN}{USER_SP_TOKEN}Hello{ASSISTANT_SP_TOKEN}{THINKING_START_TOKEN}"
    )
    assert (
        encode_messages(
            [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "Hi"},
            ],
            thinking_mode="chat",
        )
        == f"{BOS_TOKEN}You are terse.{USER_SP_TOKEN}Hi"
        f"{ASSISTANT_SP_TOKEN}{THINKING_END_TOKEN}"
    )


def test_deepseek_v4_renderer_preserve_and_drop_thinking_match_goldens():
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "reasoning_content": "hidden", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]

    assert encode_messages(messages, thinking_mode="thinking", drop_thinking=True) == (
        f"{BOS_TOKEN}{USER_SP_TOKEN}Q1{ASSISTANT_SP_TOKEN}{THINKING_END_TOKEN}"
        f"A1{EOS_TOKEN}{USER_SP_TOKEN}Q2{ASSISTANT_SP_TOKEN}{THINKING_START_TOKEN}"
    )
    assert encode_messages(messages, thinking_mode="thinking", drop_thinking=False) == (
        f"{BOS_TOKEN}{USER_SP_TOKEN}Q1{ASSISTANT_SP_TOKEN}{THINKING_START_TOKEN}"
        f"hidden{THINKING_END_TOKEN}A1{EOS_TOKEN}{USER_SP_TOKEN}Q2"
        f"{ASSISTANT_SP_TOKEN}{THINKING_START_TOKEN}"
    )


def test_deepseek_v4_renderer_encodes_tool_calls_as_dsml():
    rendered = encode_messages(
        [
            {"role": "user", "content": "Need"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"q":"tea","n":2}',
                        },
                    }
                ],
            },
        ],
        thinking_mode="chat",
    )

    assert rendered == (
        f"{BOS_TOKEN}{USER_SP_TOKEN}Need{ASSISTANT_SP_TOKEN}{THINKING_END_TOKEN}\n\n"
        f"<{DSML_TOKEN}tool_calls>\n"
        f'<{DSML_TOKEN}invoke name="lookup">\n'
        f'<{DSML_TOKEN}parameter name="q" string="true">tea</{DSML_TOKEN}parameter>\n'
        f'<{DSML_TOKEN}parameter name="n" string="false">2</{DSML_TOKEN}parameter>\n'
        f"</{DSML_TOKEN}invoke>\n"
        f"</{DSML_TOKEN}tool_calls>{EOS_TOKEN}"
    )


def test_deepseek_v4_render_prompt_dispatch_and_identity_use_renderer_id():
    rendered = render_prompt(
        [{"role": "user", "content": "Hello"}],
        prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
    )
    assert rendered == f"{BOS_TOKEN}{USER_SP_TOKEN}Hello{ASSISTANT_SP_TOKEN}</think>"

    base = rendering_identity("files", {"enable_thinking": False})
    ds4 = rendering_identity("files", prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER)
    assert base != ds4


def test_deepseek_v4_renderer_contract_forces_cache_stable_history_kwargs():
    stable = {"preserve_thinking": True, "drop_thinking": False}
    assert deepseek_v4_contract_template_kwargs() == {
        "enable_thinking": False, **stable,
    }
    assert deepseek_v4_contract_template_kwargs("off") == {
        "enable_thinking": False, **stable,
    }
    assert deepseek_v4_contract_template_kwargs("on") == {
        "enable_thinking": True, **stable,
    }
    assert deepseek_v4_contract_template_kwargs("high") == (
        deepseek_v4_contract_template_kwargs("on")
    )
    assert deepseek_v4_contract_template_kwargs("max") == {
        "enable_thinking": True, "reasoning_effort": "max", **stable,
    }

    kwargs = effective_template_kwargs(
        prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
    )

    assert kwargs["enable_thinking"] is False
    assert kwargs["preserve_thinking"] is True
    assert kwargs["drop_thinking"] is False

    for selection in ("off", "on", "max"):
        shape = deepseek_v4_contract_template_kwargs(selection)
        assert effective_template_kwargs(
            shape, prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
        ) == shape


def test_deepseek_v4_partial_internal_kwargs_normalize_to_a_full_mode():
    # Internal probe/capture/quality callers pass the historical shorthand
    # {"enable_thinking": ...}; it must keep normalizing to the matching
    # full contract shape rather than refusing.
    assert effective_template_kwargs(
        {"enable_thinking": False},
        prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
    ) == deepseek_v4_contract_template_kwargs("off")
    assert effective_template_kwargs(
        {"enable_thinking": True},
        prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
    ) == deepseek_v4_contract_template_kwargs("on")
    assert effective_template_kwargs(
        {"reasoning_effort": "max"},
        prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
    ) == deepseek_v4_contract_template_kwargs("max")


def test_normalize_thinking_selection_keeps_legacy_booleans_and_high_alias():
    normalize = http_runtime.normalize_thinking_selection
    assert normalize(None) is None
    assert normalize(True) == "on"
    assert normalize(False) == "off"
    assert normalize("high") == "on"
    assert normalize("max") == "max"
    try:
        normalize("loud")
    except ValueError:
        pass
    else:  # pragma: no cover - assertion path
        raise AssertionError("unknown selections must refuse loudly")


def test_deepseek_v4_renderer_rejects_render_contract_kwargs_from_callers():
    try:
        effective_template_kwargs(
            {
                "enable_thinking": True,
                "reasoning_effort": "max",
                "preserve_thinking": False,
                "drop_thinking": True,
            },
            prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
        )
    except http_runtime.RequestError as e:
        assert e.status == 400
        assert "cache/attention contract" in e.message
        assert "preserve_thinking" in e.message
        assert "drop_thinking" in e.message
    else:  # pragma: no cover - assertion path
        raise AssertionError("DS4 cache-breaking kwargs must be refused")


def test_deepseek_v4_renderer_max_effort_matches_official_golden():
    assert encode_messages(
        [{"role": "user", "content": "Hello"}],
        thinking_mode="thinking",
        reasoning_effort="max",
    ) == (
        f"{BOS_TOKEN}{REASONING_EFFORT_MAX}{USER_SP_TOKEN}Hello"
        f"{ASSISTANT_SP_TOKEN}{THINKING_START_TOKEN}"
    )


def test_deepseek_v4_renderer_history_is_append_only_in_every_thinking_mode():
    # The KV prefix (in-memory and disk) reuses a prior turn's rendered prompt
    # only if the next turn re-renders as an exact extension. With the served
    # contract kwargs this must hold in every mode, not just the off default.
    turn1 = [{"role": "user", "content": "Q1"}]
    for selection in ("off", "on", "max"):
        contract = deepseek_v4_contract_template_kwargs(selection)
        thinking = contract["enable_thinking"]
        reply = {"role": "assistant", "content": "A1"}
        generated = f"A1{EOS_TOKEN}"
        if thinking:
            reply = {**reply, "reasoning_content": "hidden"}
            generated = f"hidden{THINKING_END_TOKEN}A1{EOS_TOKEN}"
        turn2 = turn1 + [reply, {"role": "user", "content": "Q2"}]

        kwargs = dict(
            thinking_mode="thinking" if thinking else "chat",
            drop_thinking=contract["drop_thinking"],
            reasoning_effort=contract.get("reasoning_effort"),
        )
        prompt1 = encode_messages(turn1, **kwargs)
        prompt2 = encode_messages(turn2, **kwargs)

        scaffold = THINKING_START_TOKEN if thinking else THINKING_END_TOKEN
        assert prompt2 == (
            f"{prompt1}{generated}{USER_SP_TOKEN}Q2{ASSISTANT_SP_TOKEN}{scaffold}"
        )


def test_deepseek_v4_thinking_selections_have_distinct_cache_identities():
    identities = {
        selection: rendering_identity(
            "files",
            deepseek_v4_contract_template_kwargs(selection),
            prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
        )
        for selection in ("off", "on", "high", "max")
    }
    assert identities["high"] == identities["on"]
    assert len({identities["off"], identities["on"], identities["max"]}) == 3


def test_deepseek_v4_kv_policy_is_runtime_owned_raw_contract():
    policy = effective_kv_policy(
        {"messages": [{"role": "user", "content": "x"}]},
        prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
    )

    assert policy.live_kv_format == "raw"


def test_deepseek_v4_request_cannot_set_generic_kv_policy_fields():
    seen = {}

    def fake_generate(prompt, **kwargs):
        seen["prompt"] = prompt
        return "ok"

    try:
        chat_completion(
            {
                "messages": [{"role": "user", "content": "x"}],
                "live_kv_format": "mlx_affine_q8",
                "kv_group_size": 32,
                "quantized_kv_start": 1,
                "prompt_cache_size": 4,
                "prompt_cache_bytes": 1024,
            },
            fake_generate,
            prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
        )
    except http_runtime.RequestError as e:
        assert e.status == 400
        assert "attention contract" in e.message
        assert "live_kv_format" in e.message
        assert "kv_group_size" in e.message
        assert "quantized_kv_start" in e.message
        assert "prompt_cache_size" in e.message
        assert "prompt_cache_bytes" in e.message
    else:  # pragma: no cover - assertion path
        raise AssertionError("DS4 generic KV fields must fail")

    assert seen == {}


def test_chat_completion_uses_deepseek_v4_renderer_with_top_level_tools():
    seen = {}

    def fake_generate(prompt, **kwargs):
        seen["prompt"] = prompt
        seen["effective_rendering_id"] = kwargs["effective_rendering_id"]
        return "ok"

    req = {
        "messages": [{"role": "user", "content": "Need"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "look",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                    },
                },
            }
        ],
    }

    chat_completion(
        req,
        fake_generate,
        rendering_id="render-files",
        prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
    )

    assert seen["prompt"].startswith(f"{BOS_TOKEN}\n\n## Tools")
    assert f"<{DSML_TOKEN}tool_calls>" in seen["prompt"]
    assert f"{USER_SP_TOKEN}Need{ASSISTANT_SP_TOKEN}{THINKING_END_TOKEN}" in seen["prompt"]
    assert seen["effective_rendering_id"] == rendering_identity(
        "render-files",
        deepseek_v4_contract_template_kwargs(),
        prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
    )


def test_deepseek_v4_request_cannot_set_render_mode_kwargs():
    seen = {}

    def fake_generate(prompt, **kwargs):
        seen["prompt"] = prompt
        return "ok"

    try:
        chat_completion(
            {
                "messages": [{"role": "user", "content": "x"}],
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "reasoning_effort": "max",
                },
            },
            fake_generate,
            prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
        )
    except http_runtime.RequestError as e:
        assert e.status == 400
        assert "cache/attention contract" in e.message
        assert "chat_template_kwargs" in e.message
        assert "enable_thinking" in e.message
        assert "reasoning_effort" in e.message
    else:  # pragma: no cover - assertion path
        raise AssertionError("DS4 request-owned render kwargs must fail")

    assert seen == {}


def test_deepseek_v4_chat_completion_uses_contract_thinking_default():
    seen = {}

    def fake_generate(prompt, **kwargs):
        seen["prompt"] = prompt
        seen["effective_rendering_id"] = kwargs["effective_rendering_id"]
        return "ok"

    chat_completion(
        {"messages": [{"role": "user", "content": "Need"}]},
        fake_generate,
        rendering_id="render-files",
        prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
    )

    assert (
        f"{USER_SP_TOKEN}Need{ASSISTANT_SP_TOKEN}{THINKING_END_TOKEN}"
        in seen["prompt"]
    )
    assert seen["effective_rendering_id"] == rendering_identity(
        "render-files",
        deepseek_v4_contract_template_kwargs(),
        prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
    )


def test_deepseek_v4_request_cannot_set_cache_stable_history_fields():
    seen = {}
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "reasoning_content": "hidden", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]

    def fake_generate(prompt, **kwargs):
        seen["prompt"] = prompt
        seen["effective_rendering_id"] = kwargs["effective_rendering_id"]
        return "ok"

    try:
        chat_completion(
            {
                "messages": messages,
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "preserve_thinking": False,
                    "drop_thinking": True,
                },
            },
            fake_generate,
            rendering_id="render-files",
            prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
        )
    except http_runtime.RequestError as e:
        assert e.status == 400
        assert "cache/attention contract" in e.message
    else:  # pragma: no cover - assertion path
        raise AssertionError("DS4 request-owned cache contract fields must fail")

    assert seen == {}


def test_deepseek_v4_server_template_kwargs_cannot_break_history_stability():
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "reasoning_content": "hidden", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]

    def fail_generate(prompt, **kwargs):
        raise AssertionError("generate must not run")

    # The append-only history fields stay contract-owned: a server override
    # that would break KV-prefix identity refuses before rendering.
    try:
        chat_completion(
            {"messages": messages},
            fail_generate,
            rendering_id="render-files",
            prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
            server_template_kwargs={"drop_thinking": True},
        )
    except http_runtime.RequestError as e:
        assert e.status == 400
        assert "cache/attention contract" in e.message
        assert "drop_thinking" in e.message
    else:  # pragma: no cover - assertion path
        raise AssertionError("DS4 cache-breaking server kwargs must be refused")


def test_deepseek_v4_server_thinking_selection_renders_official_mode():
    messages = [{"role": "user", "content": "Q1"}]
    seen = {}

    def fake_generate(prompt, **kwargs):
        seen["prompt"] = prompt
        return "ok"

    chat_completion(
        {"messages": messages},
        fake_generate,
        rendering_id="render-files",
        prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
        server_template_kwargs=deepseek_v4_contract_template_kwargs("max"),
    )

    assert seen["prompt"].startswith(f"{BOS_TOKEN}{REASONING_EFFORT_MAX}")
    assert seen["prompt"].endswith(THINKING_START_TOKEN)
