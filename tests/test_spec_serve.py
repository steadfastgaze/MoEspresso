"""Served-path speculative decoding config, engagement, and cache routing.

No model or MLX import is required. The drafter loader, the spec loop, and
the plain stream function are injected, so the serve-seam contract is
testable without a GPU or a sidecar. The tiny-model end-to-end run through
the real spec loop lives in test_dspark_decode.py beside its harness.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import moespresso.runtime.deepseek_v4.spec_serve as spec_serve
from moespresso.runtime.deepseek_v4.spec_serve import (
    DRAFTER_ENV,
    DrafterConfigError,
    ServedDrafter,
    SpecCacheCompanion,
    SpecContinuation,
    SpecContinuationError,
    install_env_drafter,
    parse_drafter_env,
    resolve_env_drafter,
    spec_generation_result,
    spec_sampler_eligible,
)
from moespresso.runtime.generation import GenerationResult
from moespresso.runtime.kv_policy import parse_kv_policy
from moespresso.runtime.prefix_cache import PrefixCacheGenerator
from moespresso.runtime.serve import generate_with_metadata, load_served_model

DS4_MANIFEST = {
    "artifact_id": "pkg:ds4",
    "architecture": {"family": "deepseek_v4_flash"},
}
ORNITH_MANIFEST = {
    "artifact_id": "pkg:ornith",
    "architecture": {"family": "qwen3_5_moe"},
}


# --- configuration parsing ---


def test_parse_drafter_env_off_shapes_stay_off():
    assert parse_drafter_env(None) is None
    assert parse_drafter_env("") is None
    assert parse_drafter_env("off") is None
    assert parse_drafter_env("  off  ") is None


def test_parse_drafter_env_accepts_released_families():
    config = parse_drafter_env("dspark:/side/car")
    assert config.family == "dspark"
    assert config.sidecar_dir == Path("/side/car")

    config = parse_drafter_env("dflash:/side/dflash")
    assert config.family == "dflash"
    assert config.sidecar_dir == Path("/side/dflash")


@pytest.mark.parametrize(
    "value",
    [
        "dspark",
        "dspark:",
        "dspark:   ",
        "dflash:",
        "mtp:/sidecar",
        "banana",
        ":path",
        "on",
    ],
)
def test_parse_drafter_env_rejects_malformed_values(value):
    with pytest.raises(DrafterConfigError, match="MOESPRESSO_DS4_DRAFTER"):
        parse_drafter_env(value)


# --- drafter installation at model load ---


class _FakeDrafter:
    block_size = 5
    tap_layer_ids = (40, 41, 42)

    def tap_transform(self, layer_id, out):
        return out


def test_install_env_drafter_explicit_off_is_silent(capsys):
    model = SimpleNamespace()
    for value in ("off", "  off  "):
        assert install_env_drafter(model, DS4_MANIFEST, env_value=value) is None
    assert not hasattr(model, "_moespresso_ds4_drafter")
    assert capsys.readouterr().out == ""


def test_resolve_explicit_off_reports_off_state():
    served, state = resolve_env_drafter(
        SimpleNamespace(), DS4_MANIFEST, env_value="off")
    assert (served, state) == (None, "off")


def test_install_env_drafter_loads_taps_and_attaches(capsys):
    model = SimpleNamespace()
    fake = _FakeDrafter()
    taps = []

    def fake_load(family, sidecar_dir, m):
        assert family == "dspark"
        assert sidecar_dir == Path("/side")
        assert m is model
        return fake

    def fake_tap(m, layer_ids, transform):
        taps.append((m, tuple(layer_ids), transform))
        return "TAP"

    served = install_env_drafter(
        model,
        DS4_MANIFEST,
        env_value="dspark:/side",
        load_drafter_fn=fake_load,
        install_tap_fn=fake_tap,
    )

    assert served is model._moespresso_ds4_drafter
    assert served.family == "dspark"
    assert served.sidecar_dir == Path("/side")
    assert served.drafter is fake
    assert served.tap == "TAP"
    assert taps == [(model, (40, 41, 42), fake.tap_transform)]
    out = capsys.readouterr().out
    assert "drafter=dspark" in out
    assert "block_size=5" in out


def test_install_env_drafter_ignores_non_deepseek_families(capsys):
    model = SimpleNamespace()

    def fail_load(*_args):
        raise AssertionError("a non-DS4 family must never load a sidecar")

    assert (
        install_env_drafter(
            model, ORNITH_MANIFEST, env_value="dspark:/side",
            load_drafter_fn=fail_load,
        )
        is None
    )
    assert not hasattr(model, "_moespresso_ds4_drafter")
    assert "DeepSeek-V4 packages only" in capsys.readouterr().out

    # A malformed value is also ignored outside DS4: the variable is a
    # DeepSeek-V4 contract and other families never read past the family gate.
    assert (
        install_env_drafter(
            model, ORNITH_MANIFEST, env_value="banana",
            load_drafter_fn=fail_load,
        )
        is None
    )


def test_install_env_drafter_malformed_value_fails_startup_on_ds4():
    with pytest.raises(DrafterConfigError, match="invalid"):
        install_env_drafter(SimpleNamespace(), DS4_MANIFEST, env_value="banana")


def test_install_env_drafter_unloadable_sidecar_fails_startup():
    def broken_load(family, sidecar_dir, model):
        raise ValueError("missing sidecar manifest")

    with pytest.raises(DrafterConfigError, match="failed to load"):
        install_env_drafter(
            SimpleNamespace(),
            DS4_MANIFEST,
            env_value="dspark:/side",
            load_drafter_fn=broken_load,
        )


def test_install_env_drafter_reads_process_environment(monkeypatch, capsys):
    model = SimpleNamespace()
    monkeypatch.setenv(DRAFTER_ENV, "dspark:/from-env")

    served = install_env_drafter(
        model,
        DS4_MANIFEST,
        load_drafter_fn=lambda family, sidecar_dir, m: _FakeDrafter(),
        install_tap_fn=lambda m, layer_ids, transform: "TAP",
    )

    assert served.family == "dspark"
    assert served.sidecar_dir == Path("/from-env")
    capsys.readouterr()


def test_load_served_model_resolves_drafter_and_prints_spec_state(
    monkeypatch, tmp_path, capsys
):
    seen = {}

    def fake_resolve(model, manifest, *, package_dir):
        seen.update(model=model, manifest=manifest, package_dir=package_dir)
        return None, "dspark(auto)"

    monkeypatch.setattr(spec_serve, "resolve_env_drafter", fake_resolve)

    model, _, _ = load_served_model(
        tmp_path, manifest=dict(DS4_MANIFEST, tensors=[], files=[]),
        build_fn=lambda m, p: ("MODEL", "TOK"))

    assert seen["model"] == "MODEL"
    assert seen["manifest"]["artifact_id"] == "pkg:ds4"
    assert seen["package_dir"] == tmp_path
    out = capsys.readouterr().out
    assert "runtime=resident" in out
    assert "spec=dspark(auto)" in out


def test_load_served_model_env_absent_resolves_auto(
    monkeypatch, tmp_path, capsys
):
    # An absent variable is automatic selection, not off: the hook runs, and
    # with a resident fake model and no sidecar anywhere the resolution is
    # off with the no-declared-drafter reason, logged once and stamped on
    # the truth line. Startup proceeds.
    monkeypatch.delenv(DRAFTER_ENV, raising=False)

    load_served_model(
        tmp_path, manifest=dict(DS4_MANIFEST, tensors=[], files=[]),
        build_fn=lambda m, p: (SimpleNamespace(), "TOK"))

    out = capsys.readouterr().out
    assert "spec: auto off (package declares no drafter component)" in out
    assert "spec=off(auto:no-declared-drafter)" in out


def test_load_served_model_non_ds4_truth_line_has_no_spec_token(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.delenv(DRAFTER_ENV, raising=False)

    load_served_model(
        tmp_path, manifest=dict(ORNITH_MANIFEST, tensors=[], files=[]),
        build_fn=lambda m, p: ("MODEL", "TOK"))

    out = capsys.readouterr().out
    assert "runtime=resident" in out
    assert "spec=" not in out


# --- automatic selection (absent variable) ---


def _auto_kwargs(**overrides):
    """Seams for the automatic path: resident target, loadable fake sidecar,
    no environment reads."""
    kwargs = dict(
        env_value=None,
        residency_fn=lambda model: True,
        load_drafter_fn=lambda family, sidecar_dir, model: _FakeDrafter(),
        install_tap_fn=lambda model, layer_ids, transform: "TAP",
    )
    kwargs.update(overrides)
    return kwargs


def test_auto_bounded_residency_stays_plain(capsys):
    model = SimpleNamespace()

    def fail_load(*_args):
        raise AssertionError("bounded residency must not search or load")

    served, state = resolve_env_drafter(
        model, DS4_MANIFEST, package_dir=Path("/pkg"),
        **_auto_kwargs(residency_fn=lambda m: False, load_drafter_fn=fail_load))

    assert served is None
    assert state == "off(auto:bounded-residency)"
    assert not hasattr(model, "_moespresso_ds4_drafter")
    assert "spec: auto off (bounded residency)" in capsys.readouterr().out


def test_auto_unreadable_residency_signal_stays_plain(capsys):
    model = SimpleNamespace()

    def broken_probe(m):
        raise RuntimeError("could not find decoder layers on model")

    served, state = resolve_env_drafter(
        model, DS4_MANIFEST, package_dir=Path("/pkg"),
        **_auto_kwargs(residency_fn=broken_probe))

    assert served is None
    assert state == "off(auto:residency-unknown)"
    out = capsys.readouterr().out
    assert "spec: auto off (residency signal unavailable:" in out


def test_auto_without_sidecar_stays_plain_with_reason(tmp_path, capsys):
    package = tmp_path / "pkg"
    package.mkdir()

    served, state = resolve_env_drafter(
        SimpleNamespace(), DS4_MANIFEST, package_dir=package, **_auto_kwargs())

    assert (served, state) == (None, "off(auto:no-declared-drafter)")
    assert ("spec: auto off (package declares no drafter component)"
            in capsys.readouterr().out)


def test_auto_is_silent_for_non_ds4_packages(capsys):
    def fail_load(*_args):
        raise AssertionError("a non-DS4 family must never load a sidecar")

    served, state = resolve_env_drafter(
        SimpleNamespace(), ORNITH_MANIFEST, package_dir=Path("/pkg"),
        **_auto_kwargs(load_drafter_fn=fail_load))

    assert (served, state) == (None, None)
    assert capsys.readouterr().out == ""


def test_explicit_selection_reports_family_state_and_ignores_residency(
    capsys,
):
    model = SimpleNamespace()

    served, state = resolve_env_drafter(
        model, DS4_MANIFEST, env_value="dspark:/side",
        residency_fn=lambda m: False,
        load_drafter_fn=lambda family, sidecar_dir, m: _FakeDrafter(),
        install_tap_fn=lambda m, layer_ids, transform: "TAP")

    assert state == "dspark"
    assert served.family == "dspark"
    assert served.auto is False
    capsys.readouterr()


def test_explicit_selection_still_fails_loud_on_bad_sidecar():
    def broken_load(family, sidecar_dir, model):
        raise ValueError("missing sidecar manifest")

    with pytest.raises(DrafterConfigError, match="failed to load"):
        resolve_env_drafter(
            SimpleNamespace(), DS4_MANIFEST, env_value="dflash:/bad",
            residency_fn=lambda m: True,
            load_drafter_fn=broken_load)


# --- the full-residency signal ---


def test_full_expert_residency_true_for_resident_builds():
    # The resident builder attaches no streaming capacity; an absent
    # attribute is the resident signal.
    assert spec_serve._full_expert_residency(SimpleNamespace()) is True


def test_full_expert_residency_pooled_builds_ask_the_live_pools(monkeypatch):
    import sys
    import types

    stub = types.ModuleType("moespresso.runtime.ssd_streaming_build")
    calls = []

    def full(model):
        calls.append(model)
        return True

    stub._all_pools_fully_resident = full
    monkeypatch.setitem(
        sys.modules, "moespresso.runtime.ssd_streaming_build", stub)

    model = SimpleNamespace(_moespresso_ssd_streaming_capacity=85)
    assert spec_serve._full_expert_residency(model) is True
    assert calls == [model]

    stub._all_pools_fully_resident = lambda m: False
    assert spec_serve._full_expert_residency(model) is False


# --- engagement rules ---


def test_spec_sampler_eligibility_matrix():
    base = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": None,
        "top_logprobs": None,
    }
    assert spec_sampler_eligible(**base)
    assert spec_sampler_eligible(**{**base, "temperature": 0.7})
    assert spec_sampler_eligible(**{**base, "presence_penalty": 0.0})

    for name, value in (
        ("top_p", 0.9),
        ("top_k", 20),
        ("min_p", 0.05),
        ("presence_penalty", 1.0),
        ("top_logprobs", 3),
    ):
        assert not spec_sampler_eligible(**{**base, name: value}), name


def test_spec_sampler_eligibility_greedy_only_gates_temperature():
    # A greedy-only drafter (DFlash) proposes argmax tokens with no draft
    # distribution, so pure-temperature sampling is ineligible for it while
    # greedy requests still engage.
    base = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": None,
        "top_logprobs": None,
    }
    assert spec_sampler_eligible(**base, greedy_only=True)
    assert not spec_sampler_eligible(
        **{**base, "temperature": 0.7}, greedy_only=True)
    assert spec_sampler_eligible(**{**base, "temperature": 0.7}, greedy_only=False)


def _drafter_model():
    return SimpleNamespace(_moespresso_ds4_drafter="BUNDLE")


def _resp(text, token, *, finish_reason=None):
    return SimpleNamespace(
        text=text,
        token=token,
        finish_reason=finish_reason,
        prompt_tokens=3,
        generation_tokens=1,
        logprobs=None,
    )


def _patch_spec_result(monkeypatch, seen):
    def fake_spec(model, tokenizer, served, prompt, **kwargs):
        seen.update(model=model, served=served, prompt=prompt, **kwargs)
        return GenerationResult(
            text="spec",
            finish_reason="stop",
            prompt_tokens=3,
            completion_tokens=1,
            generated_token_ids=(7,),
            speculative={"drafter": "dspark", "rounds": 1},
        )

    monkeypatch.setattr(spec_serve, "spec_generation_result", fake_spec)


def _patch_plain_stream(monkeypatch, calls):
    import mlx_lm

    def fake_stream(**kwargs):
        calls.append(kwargs)
        yield _resp("plain", 7, finish_reason="stop")

    monkeypatch.setattr(mlx_lm, "stream_generate", fake_stream)


def test_generate_with_metadata_routes_greedy_requests_to_spec(monkeypatch):
    seen = {}
    _patch_spec_result(monkeypatch, seen)
    plain_calls = []
    _patch_plain_stream(monkeypatch, plain_calls)

    result = generate_with_metadata(
        _drafter_model(), "TOK", [1, 2, 3], max_tokens=8, temperature=0.0)

    assert result.text == "spec"
    assert result.speculative == {"drafter": "dspark", "rounds": 1}
    assert seen["served"] == "BUNDLE"
    assert seen["prompt"] == [1, 2, 3]
    assert seen["max_tokens"] == 8
    assert seen["temperature"] == 0.0
    assert plain_calls == []


def test_generate_with_metadata_forwards_speculative_continuation(monkeypatch):
    seen = {}
    _patch_spec_result(monkeypatch, seen)
    continuation = object()

    result = generate_with_metadata(
        _drafter_model(),
        "TOK",
        [3, 4],
        max_tokens=2,
        temperature=0.0,
        cached_tokens=7,
        spec_continuation=continuation,
    )

    assert result.text == "spec"
    assert seen["cached_tokens"] == 7
    assert seen["continuation"] is continuation


def test_generate_with_metadata_routes_internal_paired_prefill_to_spec(monkeypatch):
    seen = {}
    _patch_spec_result(monkeypatch, seen)
    plain_calls = []
    _patch_plain_stream(monkeypatch, plain_calls)

    def callback(event):
        return None

    result = generate_with_metadata(
        _drafter_model(),
        "TOK",
        [1, 2, 3, 4],
        max_tokens=2,
        temperature=0.0,
        spec_prefill_plan=[2],
        spec_prefill_progress_callback=callback,
        spec_prefill_progress_frontiers=[2],
    )

    assert result.text == "spec"
    assert seen["prefill_plan"] == [2]
    assert seen["prefill_progress_callback"] is callback
    assert seen["prefill_progress_frontiers"] == [2]
    assert plain_calls == []


def test_generate_with_metadata_never_drops_paired_prefill_into_plain_path(
    monkeypatch,
):
    plain_calls = []
    _patch_plain_stream(monkeypatch, plain_calls)

    with pytest.raises(SpecContinuationError, match="eligible speculative request"):
        generate_with_metadata(
            _drafter_model(),
            "TOK",
            [1, 2, 3, 4],
            max_tokens=2,
            temperature=0.0,
            top_p=0.9,
            spec_prefill_progress_callback=lambda event: None,
        )

    assert plain_calls == []


def test_generate_with_metadata_never_drops_continuation_into_plain_path(
    monkeypatch,
):
    plain_calls = []
    _patch_plain_stream(monkeypatch, plain_calls)

    with pytest.raises(SpecContinuationError, match="eligible speculative request"):
        generate_with_metadata(
            _drafter_model(),
            "TOK",
            [3, 4],
            max_tokens=2,
            temperature=0.0,
            top_p=0.9,
            spec_continuation=object(),
        )

    assert plain_calls == []


def test_generate_with_metadata_routes_pure_temperature_to_spec(monkeypatch):
    seen = {}
    _patch_spec_result(monkeypatch, seen)
    plain_calls = []
    _patch_plain_stream(monkeypatch, plain_calls)

    result = generate_with_metadata(
        _drafter_model(), "TOK", [1, 2, 3], max_tokens=4, temperature=0.7)

    assert result.text == "spec"
    assert seen["temperature"] == 0.7
    assert plain_calls == []


def _greedy_only_drafter_model():
    return SimpleNamespace(
        _moespresso_ds4_drafter=SimpleNamespace(
            drafter=SimpleNamespace(greedy_only=True))
    )


def test_generate_with_metadata_greedy_only_temperature_bypasses(monkeypatch):
    # A greedy-only drafter must never see a sampled request; the plain
    # path serves it unchanged.
    def fail_spec(*_args, **_kwargs):
        raise AssertionError(
            "a temperature request must take the plain path with a "
            "greedy-only drafter")

    monkeypatch.setattr(spec_serve, "spec_generation_result", fail_spec)
    plain_calls = []
    _patch_plain_stream(monkeypatch, plain_calls)

    result = generate_with_metadata(
        _greedy_only_drafter_model(), "TOK", [1, 2, 3],
        max_tokens=4, temperature=0.7)

    assert result.text == "plain"
    assert len(plain_calls) == 1


def test_generate_with_metadata_greedy_only_greedy_routes_to_spec(monkeypatch):
    seen = {}
    _patch_spec_result(monkeypatch, seen)
    plain_calls = []
    _patch_plain_stream(monkeypatch, plain_calls)

    result = generate_with_metadata(
        _greedy_only_drafter_model(), "TOK", [1, 2, 3],
        max_tokens=4, temperature=0.0)

    assert result.text == "spec"
    assert seen["temperature"] == 0.0
    assert plain_calls == []


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"top_p": 0.9},
        {"top_k": 20},
        {"min_p": 0.05},
        {"presence_penalty": 1.0},
        {"top_logprobs": 2},
    ],
)
def test_generate_with_metadata_shaped_requests_bypass_to_plain(
    monkeypatch, request_kwargs
):
    def fail_spec(*_args, **_kwargs):
        raise AssertionError("a shaped request must take the plain path")

    monkeypatch.setattr(spec_serve, "spec_generation_result", fail_spec)
    plain_calls = []
    _patch_plain_stream(monkeypatch, plain_calls)

    result = generate_with_metadata(
        _drafter_model(), "TOK", [1, 2, 3],
        max_tokens=4, temperature=0.0, **request_kwargs)

    assert result.text == "plain"
    assert len(plain_calls) == 1


def test_generate_with_metadata_caller_cache_bypasses_to_plain(monkeypatch):
    def fail_spec(*_args, **_kwargs):
        raise AssertionError("a caller-provided cache must take the plain path")

    monkeypatch.setattr(spec_serve, "spec_generation_result", fail_spec)
    plain_calls = []
    _patch_plain_stream(monkeypatch, plain_calls)

    result = generate_with_metadata(
        _drafter_model(), "TOK", [1, 2, 3],
        prompt_cache=["warm"], max_tokens=4, temperature=0.0)

    assert result.text == "plain"
    assert plain_calls[0]["prompt_cache"] == ["warm"]


def test_generate_with_metadata_quantized_kv_bypasses_to_plain(monkeypatch):
    def fail_spec(*_args, **_kwargs):
        raise AssertionError("quantized live KV must take the plain path")

    monkeypatch.setattr(spec_serve, "spec_generation_result", fail_spec)
    plain_calls = []
    _patch_plain_stream(monkeypatch, plain_calls)

    result = generate_with_metadata(
        _drafter_model(), "TOK", [1, 2, 3],
        kv_policy=parse_kv_policy({"live_kv_format": "mlx_affine_q8"}),
        max_tokens=4, temperature=0.0)

    assert result.text == "plain"
    assert plain_calls[0]["kv_bits"] == 8


# --- spec result surface (injected loop, no MLX) ---


class _DecodeTokenizer:
    bos_token = None
    eos_token_id = 99

    def decode(self, token_ids):
        return " ".join(str(int(t)) for t in token_ids)

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is True
        return [11, 12]


def _generation(tokens, **stats):
    defaults = dict(
        rounds=1, proposed=0, accepted=0, mean_accepted_length=1.0,
        plain_fallbacks=0, submit_length_counts={})
    defaults.update(stats)
    return SimpleNamespace(tokens=list(tokens), stats=SimpleNamespace(**defaults))


def _served():
    return ServedDrafter(
        family="dflash", sidecar_dir=Path("/side"), drafter="DRAFT", tap="TAP")


def test_spec_generation_result_stop_surface_and_stats():
    def fake_run(model, *, drafter, tap, prompt_ids, max_new_tokens,
                 temperature, eos_ids, prefill_step_size, on_commit):
        assert model == "MODEL"
        assert drafter == "DRAFT" and tap == "TAP"
        assert prompt_ids == [1, 2, 3]
        assert max_new_tokens == 10
        assert temperature == 0.0
        assert eos_ids == [99]
        assert prefill_step_size == 512
        on_commit([5])
        on_commit([6, 99])
        return _generation(
            [5, 6, 99], rounds=2, proposed=3, accepted=1,
            mean_accepted_length=2.0, plain_fallbacks=1,
            submit_length_counts={3: 1, 0: 1})

    seen = []
    result = spec_generation_result(
        "MODEL",
        _DecodeTokenizer(),
        _served(),
        [1, 2, 3],
        max_tokens=10,
        temperature=0.0,
        cached_tokens=0,
        prefill_step_size=512,
        response_callback=lambda step, resp: seen.append(
            (step, resp.token, resp.text, resp.finish_reason)),
        spec_generate_fn=fake_run,
    )

    # The stop token ends the stream without surfacing its own text.
    assert result.text == "5 6"
    assert result.finish_reason == "stop"
    assert result.generated_token_ids == (5, 6, 99)
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 3
    assert result.cached_tokens == 0
    assert result.prompt_cache is None
    assert result.first_token_seconds is not None
    assert result.generation_seconds is not None
    assert result.speculative == {
        "drafter": "dflash",
        "schedule": "adaptive",
        "rounds": 2,
        "proposed": 3,
        "accepted": 1,
        "mean_accepted_length": 2.0,
        "plain_fallbacks": 1,
        "submit_length_counts": {"0": 1, "3": 1},
    }
    assert seen == [
        (1, 5, "5", None),
        (2, 6, " 6", None),
        (3, 99, "", "stop"),
    ]


def test_spec_generation_result_length_flushes_final_segment():
    def fake_run(model, **kwargs):
        kwargs["on_commit"]([5])
        kwargs["on_commit"]([6])
        return _generation([5, 6])

    seen = []
    result = spec_generation_result(
        "MODEL",
        _DecodeTokenizer(),
        _served(),
        [1, 2, 3],
        max_tokens=2,
        temperature=0.0,
        response_callback=lambda step, resp: seen.append(
            (step, resp.token, resp.text, resp.finish_reason)),
        spec_generate_fn=fake_run,
    )

    assert result.text == "5 6"
    assert result.finish_reason == "length"
    assert result.completion_tokens == 2
    assert seen == [(1, 5, "5", None), (2, 6, " 6", "length")]


def test_spec_generation_materializes_target_cache_before_return(monkeypatch):
    import moespresso.runtime.deepseek_v4.model as ds4_model

    target_cache = [object()]
    calls = []
    monkeypatch.setattr(
        ds4_model,
        "evaluate_deepseek_v4_cache_state",
        lambda caches, *, asynchronous: calls.append(
            (caches, asynchronous)),
    )

    def fake_run(model, **kwargs):
        kwargs["on_commit"]([5])
        generation = _generation([5])
        generation.target_cache = target_cache
        return generation

    spec_generation_result(
        "MODEL",
        _DecodeTokenizer(),
        _served(),
        [1, 2, 3],
        max_tokens=1,
        temperature=0.0,
        spec_generate_fn=fake_run,
    )

    assert calls == [(target_cache, False)]


@pytest.mark.parametrize(
    ("token", "max_tokens", "expected_reason"),
    [(99, 4, "stop"), (5, 1, "length")],
)
def test_spec_first_token_callback_covers_terminal_first_commit(
    token, max_tokens, expected_reason,
):
    callbacks = []

    def fake_run(model, **kwargs):
        kwargs["on_commit"]([token])
        return _generation([token])

    result = spec_generation_result(
        "MODEL",
        _DecodeTokenizer(),
        _served(),
        [1, 2, 3],
        max_tokens=max_tokens,
        temperature=0.0,
        first_token_callback=lambda: callbacks.append("first"),
        spec_generate_fn=fake_run,
    )

    assert result.finish_reason == expected_reason
    assert callbacks == ["first"]


def test_spec_generation_result_encodes_rendered_string_prompt():
    def fake_run(model, **kwargs):
        assert kwargs["prompt_ids"] == [11, 12]
        kwargs["on_commit"]([5])
        return _generation([5])

    result = spec_generation_result(
        "MODEL",
        _DecodeTokenizer(),
        _served(),
        "rendered prompt",
        max_tokens=4,
        temperature=0.0,
        spec_generate_fn=fake_run,
    )

    assert result.prompt_tokens == 2
    assert result.text == "5"


def test_spec_generation_result_prefers_tokenizer_eos_id_set():
    class _MultiEosTokenizer(_DecodeTokenizer):
        eos_token_ids = {7, 8}

    def fake_run(model, **kwargs):
        assert set(kwargs["eos_ids"]) == {7, 8}
        kwargs["on_commit"]([5, 8])
        return _generation([5, 8])

    result = spec_generation_result(
        "MODEL",
        _MultiEosTokenizer(),
        _served(),
        [1],
        max_tokens=4,
        temperature=0.0,
        spec_generate_fn=fake_run,
    )

    assert result.finish_reason == "stop"
    assert result.text == "5"


# --- cache routing and non-resumable bypass ---


class _UntouchableStore:
    def fetch_nearest_cache(self, *_args, **_kwargs):
        raise AssertionError("the store must not serve a speculative request")

    def insert_cache(self, *_args, **_kwargs):
        raise AssertionError("a speculative cache must never be published")


class _RecordingStore:
    def __init__(self):
        self.fetched = []
        self.inserted = []

    def fetch_nearest_cache(self, model_key, tokens):
        self.fetched.append((model_key, list(tokens)))
        return None, list(tokens)

    def insert_cache(self, model_key, tokens, prompt_cache):
        self.inserted.append((model_key, list(tokens), prompt_cache))


class _EncodeTokenizer:
    bos_token = None

    def encode(self, text, add_special_tokens=True):
        return [1, 2, 3]


def _cache_generator(store, generate_fn, model=None):
    return PrefixCacheGenerator(
        model if model is not None else _drafter_model(),
        _EncodeTokenizer(),
        DS4_MANIFEST,
        store,
        make_prompt_cache_fn=lambda model: ["fresh"],
        generate_fn=generate_fn,
    )


def test_nonresumable_spec_request_keeps_cold_bypass_with_normal_miss_event(capsys):
    seen = {}
    speculative_cache = ["spec-target"]
    companion = SimpleNamespace(frontier=3, nbytes=16)

    def fake_generate(model, tokenizer, prompt, **kwargs):
        seen.update(prompt=list(prompt), **kwargs)
        return GenerationResult(
            text="s",
            finish_reason="stop",
            prompt_tokens=3,
            completion_tokens=1,
            generated_token_ids=(9,),
            prompt_cache=None,
            speculative_prompt_cache=speculative_cache,
            cache_frontier=3,
            cache_companion=companion,
            speculative={"drafter": "dspark"},
        )

    generator = _cache_generator(_UntouchableStore(), fake_generate)
    result = generator(
        "prompt",
        kv_policy=parse_kv_policy({"live_kv_format": "raw"}),
        effective_rendering_id="rid",
        temperature=0.0,
    )

    assert seen["prompt"] == [1, 2, 3]
    assert seen["prompt_cache"] is None
    assert seen["cached_tokens"] == 0
    assert result.cache_event == "miss"
    assert result.prompt_cache is None
    assert result.speculative_prompt_cache is speculative_cache
    assert result.cache_frontier == 3
    assert result.cache_companion is companion
    out = capsys.readouterr().out
    assert "disk KV are bypassed" in out

    # The notice logs once, not per request.
    generator(
        "prompt",
        kv_policy=parse_kv_policy({"live_kv_format": "raw"}),
        effective_rendering_id="rid",
        temperature=0.0,
    )
    assert "bypassed" not in capsys.readouterr().out


def test_prefix_cache_shaped_requests_still_use_the_store():
    store = _RecordingStore()

    def fake_generate(model, tokenizer, prompt, **kwargs):
        return GenerationResult(
            text="p", finish_reason="stop", prompt_tokens=3,
            completion_tokens=1, generated_token_ids=(9,),
            prompt_cache=kwargs["prompt_cache"])

    generator = _cache_generator(store, fake_generate)
    result = generator(
        "prompt",
        kv_policy=parse_kv_policy({"live_kv_format": "raw"}),
        effective_rendering_id="rid",
        temperature=0.0,
        top_p=0.9,
    )

    assert result.cache_event == "miss"
    assert len(store.fetched) == 1
    assert store.inserted[0][1] == [1, 2, 3, 9]


def test_prefix_cache_greedy_only_temperature_still_uses_the_store():
    # The cache tiers must mirror the engagement rule: a temperature
    # request with a greedy-only drafter serves plain, so the store stays
    # in play.
    store = _RecordingStore()

    def fake_generate(model, tokenizer, prompt, **kwargs):
        return GenerationResult(
            text="p", finish_reason="stop", prompt_tokens=3,
            completion_tokens=1, generated_token_ids=(9,),
            prompt_cache=kwargs["prompt_cache"])

    generator = _cache_generator(
        store, fake_generate, model=_greedy_only_drafter_model())
    result = generator(
        "prompt",
        kv_policy=parse_kv_policy({"live_kv_format": "raw"}),
        effective_rendering_id="rid",
        temperature=0.7,
    )

    assert result.cache_event == "miss"
    assert len(store.fetched) == 1


def test_prefix_cache_without_drafter_is_unchanged():
    store = _RecordingStore()

    def fake_generate(model, tokenizer, prompt, **kwargs):
        return GenerationResult(
            text="p", finish_reason="stop", prompt_tokens=3,
            completion_tokens=1, generated_token_ids=(9,),
            prompt_cache=kwargs["prompt_cache"])

    generator = _cache_generator(
        store, fake_generate, model=SimpleNamespace())
    result = generator(
        "prompt",
        kv_policy=parse_kv_policy({"live_kv_format": "raw"}),
        effective_rendering_id="rid",
        temperature=0.0,
    )

    assert result.cache_event == "miss"
    assert len(store.fetched) == 1


# --- HTTP metadata surface ---


def test_chat_completion_reports_speculative_stats_in_usage():
    from moespresso.runtime.http import chat_completion

    stats = {
        "drafter": "dspark",
        "rounds": 4,
        "mean_accepted_length": 3.1,
        "plain_fallbacks": 0,
        "submit_length_counts": {"5": 4},
    }

    def generate(prompt, **kwargs):
        return GenerationResult(
            text="hello", finish_reason="stop", prompt_tokens=5,
            completion_tokens=3, first_token_seconds=0.2,
            generation_seconds=0.4, speculative=stats)

    response = chat_completion(
        {"messages": [{"role": "user", "content": "hi"}]}, generate)

    assert response["usage"]["moespresso"]["speculative"] == stats


def test_chat_completion_omits_speculative_block_on_plain_requests():
    from moespresso.runtime.http import chat_completion

    def generate(prompt, **kwargs):
        return GenerationResult(
            text="hello", finish_reason="stop", prompt_tokens=5,
            completion_tokens=3, first_token_seconds=0.2,
            generation_seconds=0.4)

    response = chat_completion(
        {"messages": [{"role": "user", "content": "hi"}]}, generate)

    assert "speculative" not in response["usage"]["moespresso"]


# --- bundled drafter component: capacity-gated automatic selection ---


def _bundled_manifest():
    from moespresso.inventory.architecture_profile import (
        DEEPSEEK_V4_FLASH_COMPRESS_RATIOS,
    )

    return {
        "artifact_id": "pkg:ds4ship",
        "architecture": {
            "family": "deepseek_v4_flash",
            "num_hidden_layers": 43,
            "compress_ratios": list(DEEPSEEK_V4_FLASH_COMPRESS_RATIOS),
            "attention": {"head_dim": 512, "index_head_dim": 128},
        },
        "files": [{"path": "model-00001-of-00001.safetensors",
                   "size_bytes": 1 << 20, "sha256": "0" * 64}],
        "drafter": {
            "family": "dspark",
            "optional": True,
            "manifest_path": "dspark_sidecar.json",
            "files": [
                {"path": "dspark_sidecar.json",
                 "size_bytes": 64, "sha256": "0" * 64},
                {"path": "model-dspark-00001-of-00001.safetensors",
                 "size_bytes": 1 << 20, "sha256": "0" * 64},
            ],
        },
    }


def _bundled_package(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "model-dspark-00001-of-00001.safetensors").write_bytes(b"\0")
    (package / "dspark_sidecar.json").write_text(
        json.dumps({"artifact_id": "art:sidecar",
                    "dspark": {"n_mtp_layers": 3}}))
    return package


def _fit_budget():
    return (1 << 40, "synthetic")


def _short_budget():
    return (1 << 20, "synthetic")


def test_bundled_component_enables_dspark_when_the_budget_fits(
    tmp_path, capsys
):
    package = _bundled_package(tmp_path)
    model = SimpleNamespace()
    loads = []

    def record_load(family, sidecar_dir, m):
        loads.append((family, sidecar_dir))
        return _FakeDrafter()

    served, state = resolve_env_drafter(
        model, _bundled_manifest(), package_dir=package,
        budget_fn=_fit_budget,
        **_auto_kwargs(load_drafter_fn=record_load))

    assert state == "dspark(auto)"
    assert served.family == "dspark"
    assert served.auto is True
    assert served.sidecar_dir == package
    assert served.artifact_id == "art:sidecar"
    assert loads == [("dspark", package)]
    policy = model._moespresso_ds4_drafter_policy
    assert policy["mode"] == "auto"
    assert policy["decision"] == "on"
    assert policy["reason"] == "budget-fit"
    assert policy["budget_source"] == "synthetic"
    assert policy["required_bytes"] <= policy["budget_bytes"]


def test_bundled_component_budget_miss_fails_closed_to_off(tmp_path, capsys):
    package = _bundled_package(tmp_path)
    model = SimpleNamespace()

    def fail_load(*_args):
        raise AssertionError("a budget miss must never load the sidecar")

    served, state = resolve_env_drafter(
        model, _bundled_manifest(), package_dir=package,
        budget_fn=_short_budget,
        **_auto_kwargs(load_drafter_fn=fail_load))

    assert (served, state) == (None, "off(auto:budget)")
    assert not hasattr(model, "_moespresso_ds4_drafter")
    policy = model._moespresso_ds4_drafter_policy
    assert policy["mode"] == "auto"
    assert policy["decision"] == "off"
    assert policy["reason"] == "budget-exceeded"
    assert "drafter budget: budget-exceeded" in capsys.readouterr().out


def test_bundled_component_forces_off_under_bounded_residency(
    tmp_path, capsys
):
    package = _bundled_package(tmp_path)
    model = SimpleNamespace()

    def fail_load(*_args):
        raise AssertionError("bounded residency must never load a sidecar")

    served, state = resolve_env_drafter(
        model, _bundled_manifest(), package_dir=package,
        budget_fn=_fit_budget,
        **_auto_kwargs(residency_fn=lambda m: False,
                       load_drafter_fn=fail_load))

    assert (served, state) == (None, "off(auto:bounded-residency)")
    policy = model._moespresso_ds4_drafter_policy
    assert policy == {"mode": "auto", "decision": "off",
                      "reason": "bounded-residency"}
    assert "bounded residency" in capsys.readouterr().out


def test_bundled_component_absent_counts_and_stays_plain(tmp_path, capsys):
    package = tmp_path / "pkg"
    package.mkdir()
    model = SimpleNamespace()

    served, state = resolve_env_drafter(
        model, _bundled_manifest(), package_dir=package,
        budget_fn=_fit_budget, **_auto_kwargs())

    assert (served, state) == (None, "off(auto:drafter-absent)")
    policy = model._moespresso_ds4_drafter_policy
    assert policy == {"mode": "auto", "decision": "off",
                      "reason": "component-absent"}
    assert "optional drafter component absent" in capsys.readouterr().out


def test_bundled_component_partial_presence_stays_plain(tmp_path, capsys):
    package = _bundled_package(tmp_path)
    (package / "model-dspark-00001-of-00001.safetensors").unlink()
    model = SimpleNamespace()

    served, state = resolve_env_drafter(
        model, _bundled_manifest(), package_dir=package,
        budget_fn=_fit_budget, **_auto_kwargs())

    assert (served, state) == (None, "off(auto:drafter-partial)")
    policy = model._moespresso_ds4_drafter_policy
    assert policy["reason"] == "component-partial"
    assert policy["missing_files"] == [
        "model-dspark-00001-of-00001.safetensors"]


def test_bundled_component_sidecar_load_failure_stays_plain(
    tmp_path, capsys
):
    package = _bundled_package(tmp_path)
    model = SimpleNamespace()

    def broken_load(*_args):
        raise RuntimeError("bad shard")

    served, state = resolve_env_drafter(
        model, _bundled_manifest(), package_dir=package,
        budget_fn=_fit_budget,
        **_auto_kwargs(load_drafter_fn=broken_load))

    assert (served, state) == (None, "off(auto:sidecar-failed)")
    policy = model._moespresso_ds4_drafter_policy
    assert policy["decision"] == "off"
    assert policy["reason"].startswith("sidecar-failed")


def test_env_off_with_bundled_component_records_an_override(tmp_path):
    model = SimpleNamespace()

    served, state = resolve_env_drafter(
        model, _bundled_manifest(), package_dir=_bundled_package(tmp_path),
        env_value="off")

    assert (served, state) == (None, "off")
    assert model._moespresso_ds4_drafter_policy == {
        "mode": "override", "decision": "off", "reason": "env-off",
        "env": "off"}


def test_env_selection_records_an_override(tmp_path):
    model = SimpleNamespace()
    package = _bundled_package(tmp_path)

    served, state = resolve_env_drafter(
        model, _bundled_manifest(), package_dir=package,
        env_value=f"dspark:{package}",
        load_drafter_fn=lambda family, sidecar_dir, m: _FakeDrafter(),
        install_tap_fn=lambda m, layer_ids, transform: "TAP")

    assert state == "dspark"
    assert served.auto is False
    policy = model._moespresso_ds4_drafter_policy
    assert policy["mode"] == "override"
    assert policy["decision"] == "on"
    assert policy["reason"] == "env-selected"
    assert policy["env"] == f"dspark:{package}"


def test_package_without_component_serves_plain(tmp_path, capsys):
    """A package that declares no drafter component is the whole answer.

    Automatic selection reads the declaration and nothing else, so a stray
    sidecar directory beside a package that does not name it can no longer
    pair with it.
    """
    package = tmp_path / "pkg"
    stray = package / "dflash-sidecar"
    stray.mkdir(parents=True)
    (stray / "dflash_sidecar.json").write_text(json.dumps({"artifact_id": "x"}))
    model = SimpleNamespace()

    def fail_load(*_args):
        raise AssertionError("an undeclared sidecar must never load")

    served, state = resolve_env_drafter(
        model, DS4_MANIFEST, package_dir=package, budget_fn=_fit_budget,
        **_auto_kwargs(load_drafter_fn=fail_load))

    assert (served, state) == (None, "off(auto:no-declared-drafter)")
    assert not hasattr(model, "_moespresso_ds4_drafter")
    assert ("spec: auto off (package declares no drafter component)"
            in capsys.readouterr().out)


def _tight_budget_for(manifest):
    """A synthetic budget below the parts-16 requirement and at the
    parts-32 requirement, so only the floor fits."""
    from moespresso.runtime.deepseek_v4.drafter_policy import (
        evaluate_bundled_drafter_budget,
    )

    probe = evaluate_bundled_drafter_budget(
        manifest, n_stages=3, budget_fn=lambda: (1 << 60, "probe"))
    required32 = probe["required_bytes_by_nsplit"]["32"]
    return lambda: (required32, "synthetic")


def test_tight_budget_selects_parts_32_and_raises_the_process_default(
    tmp_path, monkeypatch, capsys
):
    from moespresso.runtime.deepseek_v4 import iqk_experts

    monkeypatch.delenv("MOESPRESSO_DSV4_IQK_SORT_NSPLIT", raising=False)
    package = _bundled_package(tmp_path)
    manifest = _bundled_manifest()
    model = SimpleNamespace()
    try:
        served, state = resolve_env_drafter(
            model, manifest, package_dir=package,
            budget_fn=_tight_budget_for(manifest),
            **_auto_kwargs())

        assert state == "dspark(auto)"
        policy = model._moespresso_ds4_drafter_policy
        assert policy["decision"] == "on"
        assert policy["reason"] == "budget-fit-nsplit32"
        assert policy["sort_nsplit"] == 32
        assert policy["sort_nsplit_source"] == "policy"
        assert iqk_experts.sorted_prefill_nsplit() == 32
        assert "raised to parts 32" in capsys.readouterr().out
    finally:
        iqk_experts.set_sorted_prefill_default(16)


def test_env_pinned_split_keeps_the_process_default(
    tmp_path, monkeypatch
):
    from moespresso.runtime.deepseek_v4 import iqk_experts

    monkeypatch.setenv("MOESPRESSO_DSV4_IQK_SORT_NSPLIT", "32")
    package = _bundled_package(tmp_path)
    manifest = _bundled_manifest()
    model = SimpleNamespace()

    served, state = resolve_env_drafter(
        model, manifest, package_dir=package,
        budget_fn=_tight_budget_for(manifest),
        **_auto_kwargs())

    assert state == "dspark(auto)"
    policy = model._moespresso_ds4_drafter_policy
    assert policy["sort_nsplit"] == 32
    assert policy["sort_nsplit_source"] == "env"
    assert policy["reason"] == "budget-fit"
    # The operator pinned the split; the process default is untouched.
    monkeypatch.delenv("MOESPRESSO_DSV4_IQK_SORT_NSPLIT")
    assert iqk_experts.sorted_prefill_nsplit() == 16


# --- served speculative schedule ---


def test_dspark_schedule_defaults_to_the_measured_fixed_three():
    kind, submit, label = spec_serve.resolve_spec_schedule(
        "dspark", env_value=None)
    assert (kind, submit, label) == ("fixed", 3, "fixed:3")


def test_non_dspark_families_default_to_adaptive():
    kind, submit, label = spec_serve.resolve_spec_schedule(
        "dflash", env_value=None)
    assert (kind, submit, label) == ("adaptive", None, "adaptive")


def test_schedule_env_overrides_in_both_directions():
    kind, submit, label = spec_serve.resolve_spec_schedule(
        "dspark", env_value="adaptive")
    assert (kind, submit, label) == ("adaptive", None, "adaptive")
    kind, submit, label = spec_serve.resolve_spec_schedule(
        "dflash", env_value="fixed:5")
    assert (kind, submit, label) == ("fixed", 5, "fixed:5")


@pytest.mark.parametrize("value", ["fixed", "fixed:", "fixed:0", "fixed:-1",
                                   "fixed:x", "banana", "conf:0.5"])
def test_schedule_env_fails_closed_on_malformed_values(value):
    with pytest.raises(DrafterConfigError, match="speculative schedule"):
        spec_serve.resolve_spec_schedule("dspark", env_value=value)


def test_spec_generation_result_serves_dspark_at_fixed_three(monkeypatch):
    from moespresso.runtime.deepseek_v4.spec_decode import FixedSubmitDrafter

    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    inner = _FakeDrafter()
    served = ServedDrafter(
        family="dspark", sidecar_dir=Path("/side"), drafter=inner, tap="TAP")
    seen = {}

    def fake_run(model, **kwargs):
        seen.update(kwargs)
        kwargs["on_commit"]([5])
        return _generation([5], rounds=1, proposed=3, accepted=1,
                           mean_accepted_length=2.0, plain_fallbacks=0,
                           submit_length_counts={3: 1})

    result = spec_generation_result(
        "MODEL", _DecodeTokenizer(), served, [1, 2, 3],
        max_tokens=4, temperature=0.0, spec_generate_fn=fake_run)

    assert isinstance(seen["drafter"], FixedSubmitDrafter)
    assert seen["drafter"].block_size == 3
    assert seen["adaptive_cap"] is False
    assert seen["confidence_threshold"] == 0.0
    assert result.speculative["schedule"] == "fixed:3"


class _ResumableFakeDrafter(_FakeDrafter):
    def __init__(self):
        self.imported = []
        self.exported = []
        self.export_frontier_delta = 0
        self.export_raises = False

    def state_frontier(self, state):
        return state.frontier

    def state_nbytes(self, state):
        return 16

    def import_state(self, capsule):
        self.imported.append(capsule)
        return SimpleNamespace(frontier=capsule.frontier)

    def export_state(self, state):
        self.exported.append(state)
        if self.export_raises:
            raise RuntimeError("export failed")
        return SimpleNamespace(
            frontier=state.frontier + self.export_frontier_delta,
            nbytes=16,
        )


def _spec_companion(*, artifact_id="art:side", frontier=3):
    return SpecCacheCompanion(
        family="dspark",
        artifact_id=artifact_id,
        schedule="fixed:3",
        frontier=frontier,
        capsule=SimpleNamespace(frontier=frontier, nbytes=16),
    )


def test_spec_cache_producer_rail_matches_companion_identity(monkeypatch):
    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    served = ServedDrafter(
        family="dspark",
        sidecar_dir=Path("/side"),
        drafter=_ResumableFakeDrafter(),
        tap="TAP",
        artifact_id="art:side",
    )
    expected = spec_serve.spec_cache_producer_rail(served)

    assert expected == _spec_companion().producer_rail
    assert expected == (
        "spec",
        spec_serve.SPEC_CACHE_SCHEMA,
        spec_serve.SPEC_PRODUCER_LATTICE,
        "dspark",
        "art:side",
        "fixed:3",
    )


@pytest.mark.parametrize(
    "served",
    [
        ServedDrafter(
            family="dflash",
            sidecar_dir=Path("/side"),
            drafter=_ResumableFakeDrafter(),
            tap="TAP",
            artifact_id="art:side",
        ),
        ServedDrafter(
            family="dspark",
            sidecar_dir=Path("/side"),
            drafter=_FakeDrafter(),
            tap="TAP",
            artifact_id="art:side",
        ),
        ServedDrafter(
            family="dspark",
            sidecar_dir=Path("/side"),
            drafter=_ResumableFakeDrafter(),
            tap="TAP",
            artifact_id=None,
        ),
    ],
    ids=("other_family", "non_resumable", "missing_artifact"),
)
def test_spec_cache_producer_rail_requires_identified_resumable_dspark(served):
    assert spec_serve.spec_cache_producer_rail(
        served, schedule="fixed:3"
    ) is None


def test_fixed_schedule_resumes_and_prepares_candidate_through_raw_dspark_owner(
    monkeypatch,
):
    from moespresso.runtime.deepseek_v4.spec_decode import FixedSubmitDrafter

    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    raw = _ResumableFakeDrafter()
    served = ServedDrafter(
        family="dspark",
        sidecar_dir=Path("/side"),
        drafter=raw,
        tap="TAP",
        artifact_id="art:side",
    )
    target_cache = [SimpleNamespace(offset=3)]
    continuation = SpecContinuation(
        target_cache=target_cache,
        companion=_spec_companion(),
        prefix_offset=3,
    )
    seen = {}
    continuation_events = []

    def prefill_callback(event):
        return None

    def fake_run(model, **kwargs):
        seen.update(kwargs)
        assert continuation_events == [("ready", 1)]
        assert isinstance(kwargs["drafter"], FixedSubmitDrafter)
        assert not hasattr(kwargs["drafter"], "export_state")
        assert kwargs["state_owner"] is raw
        assert kwargs["target_cache"] is target_cache
        assert kwargs["prompt_offset"] == 3
        assert kwargs["drafter_state"].frontier == 3
        kwargs["target_cache"][0].offset = 5
        kwargs["drafter_state"].frontier = 5
        kwargs["on_commit"]([7])
        generation = _generation([7])
        generation.target_cache = kwargs["target_cache"]
        generation.drafter_state = kwargs["drafter_state"]
        generation.frontier = 5
        return generation

    result = spec_generation_result(
        "MODEL",
        _DecodeTokenizer(),
        served,
        [4, 5],
        max_tokens=1,
        temperature=0.0,
        continuation=continuation,
        continuation_ready_callback=lambda: continuation_events.append(
            ("ready", len(raw.imported))
        ),
        prefill_plan=[1],
        prefill_progress_callback=prefill_callback,
        prefill_progress_frontiers=[4],
        spec_generate_fn=fake_run,
    )

    assert raw.imported == [continuation.companion.capsule]
    assert continuation_events == [("ready", 1)]
    assert seen["prefill_plan"] == [1]
    assert seen["prefill_progress_callback"] is prefill_callback
    assert seen["prefill_progress_frontiers"] == [4]
    assert raw.exported == [seen["drafter_state"]]
    assert result.prompt_tokens == 5
    assert result.cached_tokens == 3
    assert result.prompt_cache is None
    assert result.speculative_prompt_cache is target_cache
    assert result.cache_frontier == 5
    assert result.cache_companion.frontier == 5
    assert result.cache_companion.artifact_id == "art:side"
    assert result.cache_companion.producer_rail == (
        "spec",
        spec_serve.SPEC_CACHE_SCHEMA,
        spec_serve.SPEC_PRODUCER_LATTICE,
        "dspark",
        "art:side",
        "fixed:3",
    )
    assert result.speculative["cache_publication"] == {"status": "ready"}


def test_resume_provenance_mismatch_fails_before_generation(monkeypatch):
    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    raw = _ResumableFakeDrafter()
    served = ServedDrafter(
        family="dspark",
        sidecar_dir=Path("/side"),
        drafter=raw,
        tap="TAP",
        artifact_id="art:side",
    )
    continuation = SpecContinuation(
        target_cache=[SimpleNamespace(offset=3)],
        companion=_spec_companion(artifact_id="art:other"),
        prefix_offset=3,
    )

    def fail_run(*_args, **_kwargs):
        raise AssertionError("invalid provenance must fail before generation")

    continuation_events = []

    with pytest.raises(SpecContinuationError, match="sidecar artifact"):
        spec_generation_result(
            "MODEL",
            _DecodeTokenizer(),
            served,
            [4],
            max_tokens=1,
            temperature=0.0,
            continuation=continuation,
            continuation_ready_callback=lambda: continuation_events.append(
                "ready"
            ),
            spec_generate_fn=fail_run,
        )
    assert raw.imported == []
    assert continuation_events == []


def test_exact_resume_without_suffix_fails_before_state_import(monkeypatch):
    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    raw = _ResumableFakeDrafter()
    served = ServedDrafter(
        family="dspark",
        sidecar_dir=Path("/side"),
        drafter=raw,
        tap="TAP",
        artifact_id="art:side",
    )
    continuation = SpecContinuation(
        target_cache=[SimpleNamespace(offset=3)],
        companion=_spec_companion(),
        prefix_offset=3,
    )

    with pytest.raises(SpecContinuationError, match="continuation logits"):
        spec_generation_result(
            "MODEL",
            _DecodeTokenizer(),
            served,
            [],
            max_tokens=1,
            temperature=0.0,
            continuation=continuation,
            spec_generate_fn=lambda *_args, **_kwargs: None,
        )
    assert raw.imported == []


def test_candidate_frontier_mismatch_preserves_response_without_cache(
    monkeypatch,
):
    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    raw = _ResumableFakeDrafter()
    served = ServedDrafter(
        family="dspark",
        sidecar_dir=Path("/side"),
        drafter=raw,
        tap="TAP",
        artifact_id="art:side",
    )

    def fake_run(model, **kwargs):
        assert kwargs["state_owner"] is raw
        kwargs["on_commit"]([7])
        generation = _generation([7])
        generation.target_cache = [SimpleNamespace(offset=3)]
        generation.drafter_state = SimpleNamespace(frontier=4)
        generation.frontier = 4
        return generation

    result = spec_generation_result(
        "MODEL",
        _DecodeTokenizer(),
        served,
        [1, 2, 3],
        max_tokens=1,
        temperature=0.0,
        spec_generate_fn=fake_run,
    )

    assert result.text == "7"
    assert result.generated_token_ids == (7,)
    assert result.prompt_cache is None
    assert result.speculative_prompt_cache is None
    assert result.cache_frontier is None
    assert result.cache_companion is None
    assert result.speculative["cache_publication"] == {
        "status": "skipped",
        "reason": "target_frontier_mismatch",
    }
    assert raw.exported == []


def test_capsule_frontier_mismatch_skips_candidate_after_generation(monkeypatch):
    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    raw = _ResumableFakeDrafter()
    raw.export_frontier_delta = 1
    served = ServedDrafter(
        family="dspark",
        sidecar_dir=Path("/side"),
        drafter=raw,
        tap="TAP",
        artifact_id="art:side",
    )

    target_cache = [SimpleNamespace(offset=3)]

    def fake_run(model, **kwargs):
        kwargs["on_commit"]([7])
        generation = _generation([7])
        generation.target_cache = target_cache
        generation.drafter_state = SimpleNamespace(frontier=3)
        generation.frontier = 3
        return generation

    result = spec_generation_result(
        "MODEL",
        _DecodeTokenizer(),
        served,
        [1, 2, 3],
        max_tokens=1,
        temperature=0.0,
        spec_generate_fn=fake_run,
    )

    assert result.text == "7"
    assert result.speculative_prompt_cache is target_cache
    assert result.cache_frontier == 3
    assert result.cache_companion is None
    assert result.speculative["cache_publication"] == {
        "status": "skipped",
        "reason": "capsule_frontier_mismatch",
    }
    assert len(raw.exported) == 1


@pytest.mark.parametrize(
    "failure,reason",
    [
        ("missing_state", "generation_state_missing"),
        ("invalid_state", "drafter_frontier_invalid"),
        ("mismatched_state", "drafter_frontier_mismatch"),
        ("export", "drafter_export_failed"),
    ],
)
def test_drafter_candidate_failure_retains_valid_target(
    monkeypatch, failure, reason
):
    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    raw = _ResumableFakeDrafter()
    raw.export_raises = failure == "export"
    served = ServedDrafter(
        family="dspark",
        sidecar_dir=Path("/side"),
        drafter=raw,
        tap="TAP",
        artifact_id="art:side",
    )
    target_cache = [SimpleNamespace(offset=3)]
    state = SimpleNamespace(frontier=3)
    if failure == "missing_state":
        state = None
    elif failure == "invalid_state":
        state = SimpleNamespace(frontier=object())
    elif failure == "mismatched_state":
        state = SimpleNamespace(frontier=2)

    def fake_run(model, **kwargs):
        kwargs["on_commit"]([7])
        generation = _generation([7])
        generation.target_cache = target_cache
        generation.drafter_state = state
        generation.frontier = 3
        return generation

    result = spec_generation_result(
        "MODEL",
        _DecodeTokenizer(),
        served,
        [1, 2, 3],
        max_tokens=1,
        temperature=0.0,
        spec_generate_fn=fake_run,
    )

    assert result.text == "7"
    assert result.speculative_prompt_cache is target_cache
    assert result.cache_frontier == 3
    assert result.cache_companion is None
    assert result.speculative["cache_publication"] == {
        "status": "skipped",
        "reason": reason,
    }


def test_missing_sidecar_identity_rejects_otherwise_valid_target(monkeypatch):
    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    raw = _ResumableFakeDrafter()
    served = ServedDrafter(
        family="dspark",
        sidecar_dir=Path("/side"),
        drafter=raw,
        tap="TAP",
        artifact_id=None,
    )

    def fake_run(model, **kwargs):
        kwargs["on_commit"]([7])
        generation = _generation([7])
        generation.target_cache = [SimpleNamespace(offset=3)]
        generation.drafter_state = SimpleNamespace(frontier=3)
        generation.frontier = 3
        return generation

    result = spec_generation_result(
        "MODEL",
        _DecodeTokenizer(),
        served,
        [1, 2, 3],
        max_tokens=1,
        temperature=0.0,
        spec_generate_fn=fake_run,
    )

    assert result.text == "7"
    assert result.speculative_prompt_cache is None
    assert result.cache_frontier is None
    assert result.cache_companion is None
    assert result.speculative["cache_publication"] == {
        "status": "skipped",
        "reason": "missing_sidecar_artifact",
    }
    assert raw.exported == []


@pytest.mark.parametrize(
    "frontier,reason",
    [
        (2, "frontier_before_prompt"),
        (5, "frontier_beyond_public_tokens"),
    ],
)
def test_candidate_rejects_frontier_outside_public_token_span(monkeypatch, frontier, reason):
    monkeypatch.delenv(spec_serve.SPEC_SCHEDULE_ENV, raising=False)
    raw = _ResumableFakeDrafter()
    served = ServedDrafter(
        family="dspark",
        sidecar_dir=Path("/side"),
        drafter=raw,
        tap="TAP",
        artifact_id="art:side",
    )

    def fake_run(model, **kwargs):
        kwargs["on_commit"]([7])
        generation = _generation([7])
        generation.target_cache = [SimpleNamespace(offset=frontier)]
        generation.drafter_state = SimpleNamespace(frontier=frontier)
        generation.frontier = frontier
        return generation

    result = spec_generation_result(
        "MODEL",
        _DecodeTokenizer(),
        served,
        [1, 2, 3],
        max_tokens=1,
        temperature=0.0,
        spec_generate_fn=fake_run,
    )

    assert result.prompt_cache is None
    assert result.speculative_prompt_cache is None
    assert result.cache_frontier is None
    assert result.cache_companion is None
    assert result.speculative["cache_publication"] == {
        "status": "skipped",
        "reason": reason,
    }
    assert raw.exported == []


def test_fixed_submit_drafter_truncates_proposals_only():
    import mlx.core as mx

    from moespresso.runtime.deepseek_v4.spec_decode import (
        DraftProposal,
        FixedSubmitDrafter,
    )

    class _FiveTokenDrafter(_FakeDrafter):
        def make_state(self):
            return "STATE"

        def ingest(self, state, rows, positions, token_ids):
            return None

        def draft(self, state, anchor_token, anchor_pos, temperature):
            return DraftProposal(
                tokens=mx.arange(5)[None],
                logits=mx.zeros((1, 5, 7)),
                confidence=mx.zeros((1, 5)),
            )

    capped = FixedSubmitDrafter(_FiveTokenDrafter(), 3)
    assert capped.block_size == 3
    assert capped.tap_layer_ids == (40, 41, 42)
    proposal = capped.draft("STATE", 1, 10, 0.0)
    assert proposal.tokens.shape == (1, 3)
    assert proposal.logits.shape == (1, 3, 7)
    assert proposal.confidence.shape == (1, 3)

    with pytest.raises(ValueError, match="submit length"):
        FixedSubmitDrafter(_FiveTokenDrafter(), 0)
