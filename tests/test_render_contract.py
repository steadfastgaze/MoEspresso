"""The Qwen render contract: single render, family-owned chat template, explicit
thinking kwargs, and the token-prefix-stability invariant the KV cache relies on.

These are the spec for the render seam. The cache identity (rendered token stream) is only
trustworthy if rendering is single-pass, the right template is installed, and history
renders append-only under preserve_thinking=true. Most are model-free (template + tokenizer
only) so they run without a GPU or model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from moespresso.runtime.http import (
    DEFAULT_TEMPLATE_KWARGS,
    chat_completion,
    rendering_identity,
    render_prompt,
)
from moespresso.package.templates import chat_template_for
from moespresso.runtime.chat_stream import split_complete_text
from moespresso.package.tokenizer import copy_tokenizer_into_package


class _CapturingTokenizer:
    """Fake tokenizer that records every apply_chat_template call (args + kwargs)."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return f"RENDERED[{len(self.calls)}]"


# --- Step 1: vendored family template resolver ---

def test_qwen_family_resolves_to_froggeric_template():
    t = chat_template_for("qwen3_5_moe")
    assert t is not None
    # froggeric template markers: append-only thinking, XML tool format, think toggles.
    assert "preserve_thinking" in t
    assert "<function=" in t
    assert "<|im_start|>" in t


def test_qwen35_dense_family_uses_same_froggeric_template():
    assert chat_template_for("qwen3_5_dense") == chat_template_for("qwen3_5_moe")


def test_non_overridden_families_return_none():
    assert chat_template_for("synthetic_dense") is None
    assert chat_template_for(None) is None
    assert chat_template_for("some_unknown_family") is None


# --- Step 2: install at convert (per family) ---

def _fake_source(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    (d / "tokenizer.json").write_text('{"x": 1}')                       # required
    (d / "chat_template.jinja").write_text("OLD-STANDALONE")
    (d / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "OLD-EMBEDDED", "model_max_length": 4096}))


def test_install_overwrites_both_template_sources_for_qwen(tmp_path):
    # HF gives the standalone chat_template.jinja priority, but the embedded copy must also
    # be overwritten or it lingers as a stale shadow of the old template.
    src, pkg = tmp_path / "src", tmp_path / "pkg"
    _fake_source(src)
    froggeric = chat_template_for("qwen3_5_moe")

    block = copy_tokenizer_into_package(src, pkg, family="qwen3_5_moe")

    assert (pkg / "chat_template.jinja").read_text(encoding="utf-8") == froggeric
    cfg = json.loads((pkg / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert cfg["chat_template"] == froggeric
    assert cfg["model_max_length"] == 4096, "unrelated config keys must be preserved"
    assert block["chat_template_source"] == "family:qwen3_5_moe"


def test_install_overwrites_both_template_sources_for_qwen35_dense(tmp_path):
    src, pkg = tmp_path / "src", tmp_path / "pkg"
    _fake_source(src)
    froggeric = chat_template_for("qwen3_5_dense")

    block = copy_tokenizer_into_package(src, pkg, family="qwen3_5_dense")

    assert (pkg / "chat_template.jinja").read_text(encoding="utf-8") == froggeric
    cfg = json.loads((pkg / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert cfg["chat_template"] == froggeric
    assert block["chat_template_source"] == "family:qwen3_5_dense"


def test_install_leaves_non_qwen_template_untouched(tmp_path):
    src, pkg = tmp_path / "src", tmp_path / "pkg"
    _fake_source(src)
    block = copy_tokenizer_into_package(src, pkg, family="synthetic_dense")
    assert (pkg / "chat_template.jinja").read_text() == "OLD-STANDALONE"
    assert block["chat_template_source"] == "source"


def test_rendering_id_reflects_installed_template(tmp_path):
    # rendering_id hashes file content, so swapping the template must change it (it's part of
    # the rendering identity the cache keys on).
    src_q, src_n = tmp_path / "sq", tmp_path / "sn"
    _fake_source(src_q)
    _fake_source(src_n)
    q = copy_tokenizer_into_package(src_q, tmp_path / "pq", family="qwen3_5_moe")
    n = copy_tokenizer_into_package(src_n, tmp_path / "pn", family="synthetic_dense")
    assert q["rendering_id"] != n["rendering_id"]


def test_install_adds_standalone_when_source_had_only_embedded(tmp_path):
    # Some sources ship the template only inside tokenizer_config.json. The install must
    # still produce a standalone chat_template.jinja (the authoritative source) for qwen.
    src, pkg = tmp_path / "src", tmp_path / "pkg"
    src.mkdir(parents=True)
    (src / "tokenizer.json").write_text('{"x": 1}')
    (src / "tokenizer_config.json").write_text(json.dumps({"chat_template": "OLD-EMBEDDED"}))
    block = copy_tokenizer_into_package(src, pkg, family="qwen3_5_moe")
    assert (pkg / "chat_template.jinja").exists()
    assert any(f["path"] == "chat_template.jinja" for f in block["files"])


# --- Step 3: single render + explicit thinking kwargs (the seam) ---

def test_render_prompt_defaults_thinking_kwargs_on():
    # The render layer owns templating and must pass thinking kwargs on by default
    # (preserve_thinking=true is what keeps the KV prefix valid across turns).
    tok = _CapturingTokenizer()
    render_prompt([{"role": "user", "content": "hi"}], tok)
    assert len(tok.calls) == 1
    kw = tok.calls[0]["kwargs"]
    assert kw.get("enable_thinking") is True
    assert kw.get("preserve_thinking") is True


def test_render_prompt_per_request_kwargs_override_defaults():
    tok = _CapturingTokenizer()
    render_prompt([{"role": "user", "content": "hi"}], tok,
                  template_kwargs={"enable_thinking": False})
    kw = tok.calls[0]["kwargs"]
    assert kw["enable_thinking"] is False          # overridden
    assert kw.get("preserve_thinking") is True      # default still applied


def test_chat_completion_passes_request_chat_template_kwargs_to_render():
    # OpenAI-style extension: a request's chat_template_kwargs must reach apply_chat_template
    # (mirrors mlx_lm server). This is what flips thinking per request.
    tok = _CapturingTokenizer()
    req = {"messages": [{"role": "user", "content": "x"}],
           "chat_template_kwargs": {"enable_thinking": False}}
    chat_completion(req, lambda prompt, **o: "ok", tokenizer=tok)
    assert tok.calls, "chat_completion must render via the tokenizer"
    assert tok.calls[0]["kwargs"]["enable_thinking"] is False


def test_chat_completion_renders_exactly_once_no_double_template():
    # The prompt must be templated exactly once. chat_completion renders; the injected
    # generate must receive the already-rendered string and must not re-render.
    tok = _CapturingTokenizer()
    seen = {}

    def fake_generate(prompt, **opts):
        seen["prompt"] = prompt
        return "out"

    chat_completion({"messages": [{"role": "user", "content": "hi"}]}, fake_generate,
                    tokenizer=tok)
    assert len(tok.calls) == 1, "template applied more than once"
    assert seen["prompt"] == "RENDERED[1]", "generate got a non-rendered or re-rendered prompt"


# --- Step 4: rendering identity includes the resolved template kwargs ---

def test_rendering_identity_changes_with_kwargs():
    # The cache key must distinguish renders that used different template kwargs (they
    # produce different token streams). Same file-hash + different kwargs => different id.
    base = rendering_identity("filehash-abc", {"enable_thinking": True})
    other = rendering_identity("filehash-abc", {"enable_thinking": False})
    assert base != other


def test_rendering_identity_changes_with_file_hash():
    # And it must distinguish different tokenizer/template files (different rendering_id).
    a = rendering_identity("filehash-abc", DEFAULT_TEMPLATE_KWARGS)
    b = rendering_identity("filehash-xyz", DEFAULT_TEMPLATE_KWARGS)
    assert a != b


def test_rendering_identity_is_stable_and_order_independent():
    # Deterministic: same inputs -> same id; kwargs dict order must not matter.
    i1 = rendering_identity("h", {"enable_thinking": True, "preserve_thinking": False})
    i2 = rendering_identity("h", {"preserve_thinking": False, "enable_thinking": True})
    assert i1 == i2


def test_generate_once_does_not_apply_chat_template():
    # generate_once is the low-level generator: it must consume an already-rendered prompt
    # and never call apply_chat_template (that is the render layer's job). This is the other
    # half of preventing a double render: the generator must not template a second time.
    from moespresso.runtime import serve

    captured = {}

    def fake_stream(**kwargs):
        from types import SimpleNamespace
        captured["prompt"] = kwargs["prompt"]
        yield SimpleNamespace(text="ok", token=1, prompt_tokens=1,
                              generation_tokens=1, finish_reason="stop")

    tok = _CapturingTokenizer()
    serve.generate_once(
        "MODEL", tok, "ALREADY-RENDERED", max_tokens=4,
        stream_generate_fn=fake_stream, sampler_factory=lambda **kwargs: None)
    assert tok.calls == [], "generate_once must not apply the chat template"
    assert captured["prompt"] == "ALREADY-RENDERED"


# --- Step 5: token-prefix-stability invariant (the bridge to KV) ---

def _jinja_render(template_text: str, messages: list[dict], **kwargs) -> str:
    """Render a chat template the way HF/minijinja does (jinja2 + loopcontrols)."""
    from jinja2 import Environment
    env = Environment(trim_blocks=False, lstrip_blocks=False,
                      extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda v, **k: json.dumps(v)
    return env.from_string(template_text).render(messages=messages, **kwargs)


def test_preserved_thinking_makes_history_render_append_only():
    # The invariant the KV prefix cache relies on: with preserve_thinking=True, rendering
    # [turn1] and then [turn1 + assistant(<think>..) + turn2] must keep the turn-1 rendering
    # as an exact prefix of the turn-2 rendering. If history were mutated (thoughts stripped),
    # the prefix would change and the cache would be invalid.
    tpl = chat_template_for("qwen3_5_moe")
    turn1 = [{"role": "user", "content": "First question?"}]
    assistant = {"role": "assistant",
                 "content": "<think>\nreasoning here\n</think>\nFirst answer."}
    turn2 = turn1 + [assistant, {"role": "user", "content": "Second question?"}]

    r1 = _jinja_render(tpl, turn1, add_generation_prompt=False, preserve_thinking=True)
    r2 = _jinja_render(tpl, turn2, add_generation_prompt=False, preserve_thinking=True)

    assert r2.startswith(r1), (
        "preserve_thinking=True must render history append-only "
        "(turn-1 rendering is a prefix of turn-2)")


def test_stripping_thinking_breaks_the_prefix_relation():
    # The flip side that shows why the default matters: with preserve_thinking=False the
    # past <think> block is dropped, so the turn-1 rendering is no longer a prefix of turn-2
    # (the prefix mutates -> KV cache invalidated every turn). If the froggeric template ever
    # stopped honoring preserve_thinking, this test catches it.
    tpl = chat_template_for("qwen3_5_moe")
    turn1 = [{"role": "user", "content": "First question?"}]
    assistant = {"role": "assistant",
                 "content": "<think>\nreasoning here\n</think>\nFirst answer."}
    turn2 = turn1 + [assistant, {"role": "user", "content": "Second question?"}]

    r2_stripped = _jinja_render(tpl, turn2, add_generation_prompt=False,
                               preserve_thinking=False)

    # The point is the rendered assistant turn differs. Stripped render must not contain the
    # preserved thought, so the history mutates and the prefix is no longer stable.
    assert "reasoning here" not in r2_stripped, (
        "preserve_thinking=False should strip past thoughts")
    r2_preserved = _jinja_render(tpl, turn2, add_generation_prompt=False,
                                preserve_thinking=True)
    assert "reasoning here" in r2_preserved
    assert r2_preserved != r2_stripped, "the two policies must produce different history"


# --- Step 6: the request tool surface reaches the generic render branch ---

_LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look up a record by query string.",
        "parameters": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    },
}


class _JinjaTemplateTokenizer:
    """Fake tokenizer that renders a real chat template text the way HF does."""

    def __init__(self, template_text: str):
        self.template_text = template_text

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kwargs):
        assert tokenize is False
        return _jinja_render(self.template_text, messages,
                             add_generation_prompt=add_generation_prompt, **kwargs)


def test_render_prompt_forwards_tools_and_response_format():
    # The generic branch must hand the request's tool surface to the template;
    # dropping it silently disables native tool calling for any non-DS4 family.
    tok = _CapturingTokenizer()
    render_prompt(
        [{"role": "user", "content": "hi"}], tok,
        tools=[_LOOKUP_TOOL], response_format={"type": "json_object"})
    kw = tok.calls[0]["kwargs"]
    assert kw["tools"] == [_LOOKUP_TOOL]
    assert kw["response_format"] == {"type": "json_object"}


def test_render_prompt_without_tools_keeps_the_template_call_unchanged():
    # A tool-free request must not gain new template kwargs: the rendered prompt
    # is the KV prefix identity, and existing sessions depend on it staying put.
    tok = _CapturingTokenizer()
    render_prompt([{"role": "user", "content": "hi"}], tok)
    kw = tok.calls[0]["kwargs"]
    assert "tools" not in kw
    assert "response_format" not in kw


def test_chat_completion_forwards_top_level_tools_to_the_template():
    tok = _CapturingTokenizer()
    req = {"messages": [{"role": "user", "content": "x"}],
           "tools": [_LOOKUP_TOOL],
           "response_format": {"type": "json_object"}}
    chat_completion(req, lambda prompt, **o: "ok", tokenizer=tok)
    kw = tok.calls[0]["kwargs"]
    assert kw["tools"] == [_LOOKUP_TOOL]
    assert kw["response_format"] == {"type": "json_object"}


def test_top_level_tools_win_over_chat_template_kwargs():
    # Both fields can name the same template variable. The top-level OpenAI field
    # is the request contract; a single winner keeps apply_chat_template from
    # receiving a duplicate keyword.
    tok = _CapturingTokenizer()
    shadow = {"type": "function", "function": {"name": "shadow"}}
    req = {"messages": [{"role": "user", "content": "x"}],
           "tools": [_LOOKUP_TOOL],
           "chat_template_kwargs": {"tools": [shadow]}}
    chat_completion(req, lambda prompt, **o: "ok", tokenizer=tok)
    assert tok.calls[0]["kwargs"]["tools"] == [_LOOKUP_TOOL]


def test_qwen_template_renders_forwarded_tools_section():
    # End to end through render_prompt with the vendored template: the tools
    # block must render, and it must land ahead of the first message.
    tpl = chat_template_for("qwen3_5_moe")
    tok = _JinjaTemplateTokenizer(tpl)
    out = render_prompt([{"role": "user", "content": "hi"}], tok,
                        tools=[_LOOKUP_TOOL])
    assert out.startswith("<|im_start|>system\n# Tools")
    assert "<tools>" in out
    assert '"name": "lookup"' in out
    assert out.index("</tools>") < out.index("<|im_start|>user")


def test_tools_render_append_only_across_turns():
    # A constant tool set renders as the leading system block, ahead of the
    # message loop, so it joins the shared prefix: turn N's rendering must stay
    # an exact prefix of turn N+1's even when the new history carries an
    # assistant tool_calls turn and a tool result.
    tpl = chat_template_for("qwen3_5_moe")
    tools = [_LOOKUP_TOOL]
    turn1 = [{"role": "system", "content": "Be terse."},
             {"role": "user", "content": "Find record 7."}]
    turn2 = turn1 + [
        {"role": "assistant",
         "content": "<think>\nneed the lookup tool\n</think>",
         "tool_calls": [{"type": "function",
                         "function": {"name": "lookup",
                                      "arguments": {"q": "record 7"}}}]},
        {"role": "tool", "content": '{"q": "record 7", "hit": true}'},
        {"role": "user", "content": "Summarize it."},
    ]

    r1 = _jinja_render(tpl, turn1, add_generation_prompt=False,
                       preserve_thinking=True, tools=tools)
    r2 = _jinja_render(tpl, turn2, add_generation_prompt=False,
                       preserve_thinking=True, tools=tools)

    assert r2.startswith(r1), (
        "a constant tool set must keep history append-only "
        "(turn-1 rendering is a prefix of turn-2)")


def test_changing_the_tool_set_rewrites_the_shared_prefix():
    # The flip side that shows why the tool set must stay constant per session:
    # tools render into the leading system block, so adding a tool mid-session
    # rewrites the prefix and every cached turn stops applying.
    tpl = chat_template_for("qwen3_5_moe")
    extra = {"type": "function",
             "function": {"name": "store", "description": "Store a record.",
                          "parameters": {"type": "object", "properties": {}}}}
    turn1 = [{"role": "user", "content": "Find record 7."}]
    r_one = _jinja_render(tpl, turn1, add_generation_prompt=False,
                          preserve_thinking=True, tools=[_LOOKUP_TOOL])
    r_two = _jinja_render(tpl, turn1, add_generation_prompt=False,
                          preserve_thinking=True, tools=[_LOOKUP_TOOL, extra])
    assert not r_two.startswith(r_one)


# --- Step 7: mode-matrix byte-prefix stability (thinking x tools x replay) ---
#
# The property under test is the served byte identity: the next request's
# rendered prompt must start with the previous request's rendered prompt plus
# the completion text, byte for byte, in every mode combination. Thinking off
# forces an empty think scaffold onto every generated assistant turn, so the
# replayed turn must re-emit that scaffold or the prefix breaks and every
# multi-turn request becomes a prompt-cache miss. Completions here use the
# canonical whitespace the template itself produces around the think seam.

_MATRIX_TOOL_CALL = (
    "<tool_call>\n<function=lookup>\n<parameter=q>\nrecord 7\n"
    "</parameter>\n</function>\n</tool_call>"
)


def _matrix_render(messages, *, thinking_on: bool, tools, add_generation_prompt=True):
    return _jinja_render(
        chat_template_for("qwen3_5_moe"), messages,
        add_generation_prompt=add_generation_prompt,
        preserve_thinking=True, enable_thinking=thinking_on, tools=tools)


_MATRIX_BASE = [{"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Find record 7."}]


@pytest.mark.parametrize("turn_shape", ["tool_call", "plain_answer"])
@pytest.mark.parametrize("with_tools", [False, True])
@pytest.mark.parametrize("thinking_on", [False, True])
def test_mode_matrix_replay_keeps_served_byte_prefix(
        thinking_on, with_tools, turn_shape):
    tools = [_LOOKUP_TOOL] if with_tools else None
    if turn_shape == "tool_call":
        body = _MATRIX_TOOL_CALL
        follow_up = {"role": "tool", "content": '{"q": "record 7", "hit": true}'}
    else:
        body = "Record 7 is active."
        follow_up = {"role": "user", "content": "Now check record 8."}
    completion = f"need to check this\n</think>\n{body}" if thinking_on else body

    r1 = _matrix_render(_MATRIX_BASE, thinking_on=thinking_on, tools=tools)
    scaffold = "<think>\n" if thinking_on else "<think>\n</think>\n"
    assert r1.endswith("<|im_start|>assistant\n" + scaffold)

    served_bytes = r1 + completion
    history = _MATRIX_BASE + [
        {"role": "assistant", "content": completion}, follow_up]
    r2 = _matrix_render(history, thinking_on=thinking_on, tools=tools)
    assert r2.startswith(served_bytes), (
        f"replay is not byte-stable (thinking_on={thinking_on}, "
        f"with_tools={with_tools}, turn={turn_shape})")


@pytest.mark.parametrize("turn_shape", ["tool_call", "plain_answer"])
@pytest.mark.parametrize("with_tools", [False, True])
def test_structured_reasoning_replay_matches_raw_served_bytes(
        with_tools, turn_shape):
    tools = [_LOOKUP_TOOL] if with_tools else None
    body = _MATRIX_TOOL_CALL if turn_shape == "tool_call" else "Record 7 is active."
    completion = f"need to check this\n</think>\n{body}"
    reasoning, content = split_complete_text(
        completion, thinking_enabled=True)
    raw_history = _MATRIX_BASE + [{
        "role": "assistant",
        "content": completion,
    }]
    structured_history = _MATRIX_BASE + [{
        "role": "assistant",
        "reasoning_content": reasoning,
        "content": content,
    }]

    raw_render = _matrix_render(
        raw_history, thinking_on=True, tools=tools,
        add_generation_prompt=False)
    structured_render = _matrix_render(
        structured_history, thinking_on=True, tools=tools,
        add_generation_prompt=False)
    assert structured_render == raw_render


def test_thinking_off_replay_re_emits_the_empty_scaffold():
    history = _MATRIX_BASE + [
        {"role": "assistant", "content": "Record 7 is active."},
        {"role": "user", "content": "Now check record 8."}]
    rendered = _matrix_render(history, thinking_on=False, tools=None)
    assert ("<|im_start|>assistant\n<think>\n</think>\nRecord 7 is active."
            "<|im_end|>\n") in rendered


def test_thinking_on_replay_gains_no_scaffold():
    # The scaffold re-emission is gated on thinking off; a thinking-on
    # history render must stay byte-identical to the pre-fix behavior, so
    # no empty scaffold may appear anywhere in it.
    history = _MATRIX_BASE + [
        {"role": "assistant",
         "content": "reasoning here\n</think>\nRecord 7 is active."},
        {"role": "user", "content": "Now check record 8."}]
    rendered = _matrix_render(history, thinking_on=True, tools=None,
                              add_generation_prompt=False)
    assert "reasoning here" in rendered
    assert "<think>\n</think>" not in rendered


# --- integration: the real convert pipeline installs the template (needs the runtime stack) ---

def test_real_convert_installs_froggeric_template(tmp_path):
    import struct

    import numpy as np
    import pytest
    pytest.importorskip("mlx.core")
    pytest.importorskip("jang_tools.turboquant")
    from moespresso.package.convert import convert

    def _st(path, tensors):
        header, blob, off = {}, bytearray(), 0
        for name, arr in tensors.items():
            a = np.ascontiguousarray(arr, dtype=np.float32)
            b = a.tobytes()
            header[name] = {"dtype": "F32", "shape": list(a.shape),
                            "data_offsets": [off, off + len(b)]}
            blob += b
            off += len(b)
        hjson = json.dumps(header).encode()
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(hjson)))
            f.write(hjson)
            f.write(blob)

    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(0)
    _st(src / "model-00001.safetensors", {
        "model.language_model.layers.0.self_attn.q_proj.weight":
            rng.standard_normal((128, 128)).astype(np.float32),
        "model.language_model.layers.0.mlp.experts.gate_up_proj":
            rng.standard_normal((8, 256, 128)).astype(np.float32),
        "model.language_model.layers.0.mlp.experts.down_proj":
            rng.standard_normal((8, 128, 128)).astype(np.float32),
    })
    (src / "config.json").write_text(json.dumps(
        {"model_type": "qwen3_moe",
         "text_config": {"num_hidden_layers": 1, "hidden_size": 128, "num_experts": 8,
                         "num_experts_per_tok": 2, "moe_intermediate_size": 128,
                         "layer_types": ["full_attention"], "vocab_size": 256}}))
    # a minimal source tokenizer with an OLD template (both sources)
    (src / "tokenizer.json").write_text('{"x": 1}')
    (src / "tokenizer_config.json").write_text(json.dumps({"chat_template": "OLD"}))

    out = tmp_path / "pkg"
    man = convert(src, out, allow_uniform=True, target_quality=0.5, shard_size_gb=0.0)

    froggeric = chat_template_for("qwen3_5_moe")
    assert (out / "chat_template.jinja").read_text(encoding="utf-8") == froggeric
    assert json.loads((out / "tokenizer_config.json").read_text())["chat_template"] == froggeric
    assert man["tokenizer"]["chat_template_source"] == "family:qwen3_5_moe"
