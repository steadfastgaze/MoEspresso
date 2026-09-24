from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import mlx.core as mx
import numpy as np
import pytest

from moespresso.runtime.generation import ContextLimitError, GenerationResult
from moespresso.runtime.http import (
    PackageRequestContract,
    build_cache_generator,
    chat_completion,
)
from moespresso.runtime.kv_policy import KVPolicy, LIVE_KV_Q8, LIVE_KV_RAW
from moespresso.runtime.pooled_decode_session import PooledDecodeSession
from moespresso.runtime.pooled_moe import pooled_state_scope
from moespresso.runtime.qwen4.generation import (
    QWEN4_GENERATION_ADAPTER,
    Qwen4RequestGenerator,
    _limited_free_cache,
    _prefill_memory_policy,
    generate_qwen4_with_metadata,
)
from moespresso.runtime.serve import generate_with_metadata


@dataclass
class _State:
    frontier: int = 0


@dataclass
class _Candidate:
    logits: mx.array
    token: int
    consumed: bool = False


class _SerialLane:
    def __init__(self, model) -> None:
        self.model = model
        self.closed = False
        self.pending = False

    def begin_step(self, input_ids):
        token = int(np.asarray(input_ids)[0, 0])
        self.model.serial_begins.append(token)
        self.pending = True
        return _Candidate(self.model.logits([token]), token)

    def finish_step(self, step, *, evaluated):
        assert not step.consumed
        mx.eval(*evaluated)
        step.consumed = True
        self.pending = False
        self.model.committed.append(step.token)
        self.model.serial_finishes += 1

    def close(self):
        if not self.closed:
            if self.pending:
                self.model.abandoned_serial_pending += 1
            self.closed = True
            self.model.closed_serial_lanes += 1


class _PipelinedSerialLane(_SerialLane):
    def prime_pipelined_step(self, step, *, evaluated, queued):
        assert self.pending
        mx.async_eval(*evaluated, *queued, step.logits)

    def chain_pipelined_step(self, step, input_ids):
        assert self.pending
        assert not step.consumed
        token = int(np.asarray(input_ids)[0, 0])
        self.model.serial_begins.append(token)
        return _Candidate(self.model.logits([token]), token)

    def finish_pipelined_transition(self, step, next_step, *, evaluated, queued):
        assert self.pending
        assert not step.consumed
        mx.async_eval(*evaluated, *queued, next_step.logits)
        mx.eval(*evaluated)
        step.consumed = True
        self.model.committed.append(step.token)
        self.model.serial_finishes += 1

    def finish_terminal_pipelined_step(self, step, *, evaluated):
        self.finish_step(step, evaluated=evaluated)


class _ScopedPipelinedSerialLane(_PipelinedSerialLane):
    """Exercise the real per-owner state-scope decorator contract."""

    def __init__(self, model) -> None:
        super().__init__(model)
        self._model = model
        self._identity = object()

    @pooled_state_scope
    def begin_step(self, input_ids):
        return super().begin_step(input_ids)

    @pooled_state_scope
    def prime_pipelined_step(self, step, *, evaluated, queued):
        return super().prime_pipelined_step(step, evaluated=evaluated, queued=queued)

    @pooled_state_scope
    def chain_pipelined_step(self, step, input_ids):
        return super().chain_pipelined_step(step, input_ids)

    @pooled_state_scope
    def finish_pipelined_transition(self, step, next_step, *, evaluated, queued):
        return super().finish_pipelined_transition(
            step,
            next_step,
            evaluated=evaluated,
            queued=queued,
        )

    @pooled_state_scope
    def finish_step(self, step, *, evaluated):
        return super().finish_step(step, evaluated=evaluated)

    @pooled_state_scope
    def finish_terminal_pipelined_step(self, step, *, evaluated):
        return super().finish_terminal_pipelined_step(step, evaluated=evaluated)

    @pooled_state_scope
    def close(self):
        return super().close()


class _Coordinator:
    def __init__(self, model) -> None:
        self.model = model
        self.state = _State()
        self.closed = False

    def forward_chunk(self, input_ids, **_kwargs):
        tokens = [int(token) for token in np.asarray(input_ids)[0]]
        self.model.prefill_chunks.append(tuple(tokens))
        self.state.frontier += len(tokens)
        return self.model.logits(tokens)

    def commit(self, candidate, keep_tokens):
        assert keep_tokens == 1
        assert not candidate.consumed
        candidate.consumed = True
        self.state.frontier += 1
        self.model.committed.append(candidate.token)
        return self.state

    def propose(self, input_ids, **_kwargs):
        return self.model.propose(self, input_ids)

    def try_enter_plain_serial_lane(self):
        if not self.model.enable_serial:
            return None
        assert not self.closed
        self.closed = True
        self.model.closed_coordinators += 1
        self.model.serial_lane_entries += 1
        lane_type = _PipelinedSerialLane if self.model.enable_pipeline else _SerialLane
        return lane_type(self.model)

    def close(self):
        if not self.closed:
            self.closed = True
            self.model.closed_coordinators += 1


class _ScopedCoordinator(_Coordinator):
    """Use a distinct owner from its serial lane during request teardown."""

    def __init__(self, model) -> None:
        super().__init__(model)
        self._model = model
        self._identity = object()

    @pooled_state_scope
    def forward_chunk(self, input_ids, **kwargs):
        return super().forward_chunk(input_ids, **kwargs)

    @pooled_state_scope
    def try_enter_plain_serial_lane(self):
        if not self.model.enable_serial:
            return None
        assert not self.closed
        self.closed = True
        self.model.closed_coordinators += 1
        self.model.serial_lane_entries += 1
        return _ScopedPipelinedSerialLane(self.model)

    @pooled_state_scope
    def close(self):
        return super().close()


class _Model:
    def __init__(
        self,
        transitions=None,
        *,
        limit=64,
        stops=(7, 9),
        enable_serial=False,
        enable_pipeline=False,
        bounded=False,
        scoped_owners=False,
    ) -> None:
        self._moespresso_generation_adapter = QWEN4_GENERATION_ADAPTER
        self._moespresso_qwen4_stop_ids = frozenset(stops)
        self._moespresso_qwen4_kvarn_context_tokens = limit
        self.transitions = transitions or {1: 2, 2: 3, 3: 7}
        self.prefill_chunks = []
        self.committed = []
        self.closed_coordinators = 0
        self.created_coordinators = 0
        self.enable_serial = enable_serial
        self.enable_pipeline = enable_pipeline
        self._moespresso_pooled_decode_bounded = bounded
        self.scoped_owners = scoped_owners
        if scoped_owners:
            self._moespresso_pooled_decode_session = PooledDecodeSession()
        self.serial_lane_entries = 0
        self.serial_begins = []
        self.serial_finishes = 0
        self.closed_serial_lanes = 0
        self.abandoned_serial_pending = 0
        self.closed = False

    def logits(self, tokens):
        rows = []
        for token in tokens:
            values = np.full((12,), -20.0, dtype=np.float32)
            values[self.transitions[int(token)]] = 20.0
            rows.append(values)
        return mx.array(np.asarray(rows)[None])

    def new_coordinator(self, batch_size):
        assert batch_size == 1
        self.created_coordinators += 1
        return _ScopedCoordinator(self) if self.scoped_owners else _Coordinator(self)

    def propose(self, coordinator, input_ids):
        token = int(np.asarray(input_ids)[0, 0])
        return _Candidate(self.logits([token]), token)

    def close(self):
        self.closed = True


class _Detokenizer:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.text = ""
        self.offset = 0

    def add_token(self, token):
        self.text += self.tokenizer.decode([int(token)])

    def finalize(self):
        pass

    @property
    def last_segment(self):
        segment = self.text[self.offset :]
        self.offset = len(self.text)
        return segment


class _Tokenizer:
    bos_token = None
    eos_token_ids = {7, 9}

    def encode(self, prompt, *, add_special_tokens):
        assert add_special_tokens
        return [int(part) for part in prompt.split()]

    def decode(self, tokens):
        return "".join(chr(ord("a") + int(token)) for token in tokens)

    @property
    def detokenizer(self):
        return _Detokenizer(self)


def _greedy_factory(**_kwargs):
    return lambda logprobs: mx.argmax(logprobs, axis=-1)


def _run(model, *, prefill_step_size):
    return generate_qwen4_with_metadata(
        model,
        _Tokenizer(),
        [1, 1, 1],
        max_tokens=8,
        temperature=0.0,
        prefill_step_size=prefill_step_size,
        sampler_factory=_greedy_factory,
    )


def test_full_prefill_and_chunked_prefill_generate_identical_public_tokens() -> None:
    full_model = _Model()
    chunked_model = _Model()

    full = _run(full_model, prefill_step_size=64)
    chunked = _run(chunked_model, prefill_step_size=2)

    assert full.generated_token_ids == chunked.generated_token_ids == (2, 3, 7)
    assert full.finish_reason == chunked.finish_reason == "stop"
    assert full_model.prefill_chunks == [(1, 1, 1)]
    assert chunked_model.prefill_chunks == [(1, 1), (1,)]
    assert full_model.committed == chunked_model.committed == [2, 3]
    assert full_model.closed_coordinators == chunked_model.closed_coordinators == 1


def test_constrained_prefill_policy_preserves_unconstrained_and_short_requests():
    kwargs = dict(
        capacity=223,
        prompt_rows=2048,
        recommended_bytes=25 << 30,
        active_bytes=20 << 30,
        available_bytes=5 << 30,
    )
    assert _prefill_memory_policy(**kwargs) == (512, 2 << 30)
    assert _prefill_memory_policy(**{**kwargs, "capacity": 512}) is None
    assert _prefill_memory_policy(**{**kwargs, "prompt_rows": 512}) is None
    assert _prefill_memory_policy(**{**kwargs, "active_bytes": 17 << 30}) is None
    assert _prefill_memory_policy(**{**kwargs, "available_bytes": 2 << 30}) == (512, 0)


def test_constrained_prefill_splits_checkpoint_plan_without_changing_tokens(monkeypatch):
    import moespresso.runtime.qwen4.generation as generation

    engaged = []

    @contextmanager
    def cache_scope(limit):
        engaged.append(limit)
        yield

    monkeypatch.setattr(generation, "_resolve_prefill_memory_policy", lambda *_: (2, 1 << 30))
    monkeypatch.setattr(generation, "_limited_free_cache", cache_scope)
    model = _Model()
    progress = []
    result = generate_qwen4_with_metadata(
        model,
        _Tokenizer(),
        [1, 1, 1],
        max_tokens=8,
        temperature=0.0,
        prefill_step_size=3,
        prefill_plan=[3],
        prompt_progress_callback=lambda done, total: progress.append((done, total)),
        sampler_factory=_greedy_factory,
    )

    assert engaged == [1 << 30]
    assert model.prefill_chunks == [(1, 1), (1,)]
    assert progress == [(0, 3), (2, 3), (3, 3)]
    assert result.generated_token_ids == (2, 3, 7)


def test_constrained_prefill_restores_cache_limit_after_error(monkeypatch):
    import moespresso.runtime.qwen4.generation as generation

    state = {"limit": 5 << 30, "clears": 0}

    def set_limit(value):
        old = state["limit"]
        state["limit"] = value
        return old

    def clear_cache():
        state["clears"] += 1

    monkeypatch.setattr(generation.mx, "set_cache_limit", set_limit)
    monkeypatch.setattr(generation.mx, "clear_cache", clear_cache)
    with pytest.raises(RuntimeError, match="prefill failed"):
        with _limited_free_cache(2 << 30):
            assert state["limit"] == 2 << 30
            raise RuntimeError("prefill failed")

    assert state == {"limit": 5 << 30, "clears": 2}


def test_plain_serial_generation_samples_inside_the_state_boundary() -> None:
    model = _Model(enable_serial=True)

    result = _run(model, prefill_step_size=2)

    assert result.generated_token_ids == (2, 3, 7)
    assert result.finish_reason == "stop"
    assert model.serial_lane_entries == 1
    assert model.serial_begins == [2, 3]
    assert model.serial_finishes == 2
    assert model.committed == [2, 3]
    assert model.closed_coordinators == 1
    assert model.closed_serial_lanes == 1


def test_plain_serial_lazy_pipeline_keeps_one_private_target_row() -> None:
    model = _Model(
        transitions={1: 2, 2: 3, 3: 7, 7: 9},
        enable_serial=True,
        enable_pipeline=True,
    )

    result = _run(model, prefill_step_size=2)

    assert result.generated_token_ids == (2, 3, 7)
    assert result.finish_reason == "stop"
    assert model.serial_lane_entries == 1
    assert model.serial_begins == [2, 3, 7]
    assert model.serial_finishes == 2
    assert model.committed == [2, 3]
    assert model.abandoned_serial_pending == 1
    assert model.closed_coordinators == 1
    assert model.closed_serial_lanes == 1


def test_bounded_plain_serial_pipeline_is_enabled_when_structurally_eligible() -> None:
    transitions = {1: 2, 2: 3, 3: 7, 7: 9}
    model = _Model(
        transitions=transitions,
        enable_serial=True,
        enable_pipeline=True,
        bounded=True,
    )

    pipelined = _run(model, prefill_step_size=2)
    assert pipelined.generated_token_ids == (2, 3, 7)
    assert model.serial_begins == [2, 3, 7]
    assert model.committed == [2, 3]


def test_bounded_pipeline_releases_lane_owner_before_coordinator_close(
) -> None:
    model = _Model(
        transitions={1: 2, 2: 3, 3: 7, 7: 9},
        enable_serial=True,
        enable_pipeline=True,
        bounded=True,
        scoped_owners=True,
    )

    result = _run(model, prefill_step_size=2)

    assert result.generated_token_ids == (2, 3, 7)
    assert model.closed_serial_lanes == model.closed_coordinators == 1
    assert not model._moespresso_pooled_decode_session.active


def test_plain_serial_lazy_pipeline_declines_presence_processors() -> None:
    model = _Model(enable_serial=True, enable_pipeline=True)

    result = generate_qwen4_with_metadata(
        model,
        _Tokenizer(),
        [1],
        max_tokens=8,
        temperature=0.0,
        presence_penalty=1.5,
        sampler_factory=_greedy_factory,
        logits_processors_factory=lambda **_kwargs: [lambda _history, logits: logits],
    )

    assert result.generated_token_ids == (2, 3, 7)
    assert model.serial_begins == [2, 3]
    assert model.abandoned_serial_pending == 0


def test_serve_generation_dispatches_to_qwen4_greedy_adapter() -> None:
    model = _Model()
    result = generate_with_metadata(
        model,
        _Tokenizer(),
        [1],
        max_tokens=4,
        temperature=0.0,
        sampler_factory=_greedy_factory,
    )

    assert result.generated_token_ids == (2, 3, 7)
    assert result.text == "cd"
    assert result.prompt_cache is None


def test_sampling_factory_receives_cli_and_server_sampling_values() -> None:
    seen = {}

    def factory(**kwargs):
        seen.update(kwargs)
        return lambda _logprobs: mx.array([5], dtype=mx.int64)

    result = generate_qwen4_with_metadata(
        _Model(transitions={1: 2}),
        _Tokenizer(),
        [1],
        max_tokens=1,
        temperature=0.35,
        top_p=0.81,
        top_k=5,
        min_p=0.04,
        sampler_factory=factory,
    )

    assert seen == {"temp": 0.35, "top_p": 0.81, "top_k": 5, "min_p": 0.04}
    assert result.generated_token_ids == (5,)
    assert result.finish_reason == "length"


@pytest.mark.parametrize("stop_id", [7, 9])
def test_either_released_eos_id_stops_generation(stop_id) -> None:
    result = generate_qwen4_with_metadata(
        _Model(transitions={1: stop_id}),
        _Tokenizer(),
        [1],
        max_tokens=4,
        temperature=0.0,
        sampler_factory=_greedy_factory,
    )

    assert result.generated_token_ids == (stop_id,)
    assert result.finish_reason == "stop"


@pytest.mark.parametrize("enable_serial", [False, True])
def test_response_cancellation_closes_request_state(enable_serial: bool) -> None:
    model = _Model(enable_serial=enable_serial)
    responses = []
    result = generate_qwen4_with_metadata(
        model,
        _Tokenizer(),
        [1],
        max_tokens=8,
        temperature=0.0,
        sampler_factory=_greedy_factory,
        response_callback=lambda _step, response: responses.append(response.token),
        response_stop_callback=lambda: True,
    )

    assert responses == [2]
    assert result.generated_token_ids == (2,)
    assert result.finish_reason == "stop"
    assert model.committed == []
    assert model.closed_coordinators == 1
    assert model.closed_serial_lanes == int(enable_serial)


@pytest.mark.parametrize("enable_serial", [False, True])
def test_callback_error_still_closes_request_state(enable_serial: bool) -> None:
    model = _Model(enable_serial=enable_serial)

    def cancel(_step, _response):
        raise RuntimeError("client disconnected")

    with pytest.raises(RuntimeError, match="client disconnected"):
        generate_qwen4_with_metadata(
            model,
            _Tokenizer(),
            [1],
            max_tokens=8,
            temperature=0.0,
            sampler_factory=_greedy_factory,
            response_callback=cancel,
        )

    assert model.closed_coordinators == 1
    assert model.closed_serial_lanes == int(enable_serial)


def test_serial_sampling_error_abandons_pending_request_state() -> None:
    model = _Model(enable_serial=True)
    calls = 0

    def factory(**_kwargs):
        def sample(logprobs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("sampling failed")
            return mx.argmax(logprobs, axis=-1)

        return sample

    with pytest.raises(RuntimeError, match="sampling failed"):
        generate_qwen4_with_metadata(
            model,
            _Tokenizer(),
            [1],
            max_tokens=8,
            temperature=0.0,
            sampler_factory=factory,
        )

    assert model.serial_begins == [2]
    assert model.serial_finishes == 0
    assert model.abandoned_serial_pending == 1
    assert model.closed_serial_lanes == 1
    assert model.closed_coordinators == 1


def test_context_limit_refuses_before_allocating_composite_state() -> None:
    model = _Model(limit=3)
    with pytest.raises(ContextLimitError, match="served context limit of 3"):
        generate_qwen4_with_metadata(
            model,
            _Tokenizer(),
            [1, 1],
            max_tokens=2,
            temperature=0.0,
            sampler_factory=_greedy_factory,
        )

    assert model.created_coordinators == 0


def test_generic_q8_kv_policy_refuses_before_allocating_composite_state() -> None:
    model = _Model()
    with pytest.raises(ValueError, match="owns its K4/V4 live-cache format"):
        generate_qwen4_with_metadata(
            model,
            _Tokenizer(),
            [1],
            max_tokens=1,
            kv_policy=KVPolicy(live_kv_format=LIVE_KV_Q8),
            sampler_factory=_greedy_factory,
        )

    assert model.created_coordinators == 0


def test_speculative_ready_callback_is_not_silently_ignored() -> None:
    with pytest.raises(ValueError, match="spec_continuation_ready_callback"):
        generate_with_metadata(
            _Model(),
            _Tokenizer(),
            [1],
            max_tokens=1,
            spec_continuation_ready_callback=lambda: None,
            sampler_factory=_greedy_factory,
        )


def test_http_dispatch_passes_memory_and_disk_stores_to_qwen_adapter() -> None:
    from moespresso.runtime.prefix_cache import PromptCacheStore

    model = _Model()
    memory = PromptCacheStore(max_size=2)
    disk = object()

    def memory_factory(size, budget):
        assert size == 2 and budget == 4096
        return memory

    generator = build_cache_generator(
        model,
        _Tokenizer(),
        {"architecture": {"config": {"max_position_embeddings": 64}}},
        context_limit=64,
        prompt_cache_size=2,
        prompt_cache_bytes=4096,
        memory_store_factory=memory_factory,
        disk_store=disk,
    )

    assert isinstance(generator, Qwen4RequestGenerator)
    assert generator.disk_store is disk and generator.cache_store is memory
    generator.disk_store = None
    assert generator.cache_stats() == {
        "default_live_kv_format": "qwen_kvarn_k4v4",
        "supported_live_kv_formats": ["qwen_kvarn_k4v4"],
        "entries": 0,
        "bytes": 0,
        "disk": {"enabled": False},
    }
    generator.close()
    assert model.closed


@pytest.mark.parametrize("mode", [None, "frontier"])
def test_qwen4_serve_uses_shared_disk_configuration(monkeypatch, tmp_path, mode) -> None:
    import moespresso.runtime.disk_kv as disk_kv
    import moespresso.runtime.http as http

    manifest = {
        "artifact_id": "pkg:qwen4-test",
        "subject": {"source_root": "qwen-test"},
        "architecture": {
            "family": "qwen4_exp",
            "modality": "text",
            "config": {"max_position_embeddings": 64},
        },
        "tokenizer": {"files": []},
        "tensors": [],
        "files": [],
    }

    class StopAfterLoad(Exception):
        pass

    if mode is None:
        monkeypatch.delenv("MOESPRESSO_DISK_KV", raising=False)
    else:
        monkeypatch.setenv("MOESPRESSO_DISK_KV", mode)
    monkeypatch.setenv("MOESPRESSO_DISK_KV_ROOT", str(tmp_path / "disk"))
    opened = []
    monkeypatch.setattr(http, "_preflight_manifest_for_cli", lambda _path: manifest)
    monkeypatch.setattr(
        disk_kv,
        "open_disk_store",
        lambda config: opened.append(config),
    )
    monkeypatch.setattr(
        http,
        "build_cache_generator",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(StopAfterLoad()),
    )

    with pytest.raises(StopAfterLoad):
        http.serve(
            tmp_path,
            startup_warmup=False,
            load_model_fn=lambda _path: (_Model(), _Tokenizer(), manifest),
        )

    assert len(opened) == 1 and opened[0].enabled
    assert opened[0].root == tmp_path / "disk"


def test_qwen4_serve_refuses_invalid_disk_configuration_before_load(monkeypatch, tmp_path, capsys) -> None:
    import moespresso.runtime.http as http

    manifest = {
        "architecture": {"family": "qwen4_exp"},
    }
    monkeypatch.setenv("MOESPRESSO_DISK_KV", "invalid")
    monkeypatch.setattr(http, "_preflight_manifest_for_cli", lambda _path: manifest)

    result = http.serve(
        tmp_path,
        startup_warmup=False,
        load_model_fn=lambda _path: (_ for _ in ()).throw(
            AssertionError("invalid disk KV configuration must refuse before load")
        ),
    )

    assert result == 2
    assert "FAILED:" in capsys.readouterr().out


def test_qwen4_http_contract_owns_live_cache_policy() -> None:
    import moespresso.runtime.http as http

    seen = {}

    def generate(_prompt, **kwargs):
        seen.update(kwargs)
        return GenerationResult(
            text="",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=0,
            cached_tokens=0,
        )

    response = chat_completion(
        {"messages": [{"role": "user", "content": "1"}], "max_tokens": 1},
        generate,
        tokenizer=_Tokenizer(),
        request_contract=PackageRequestContract(family="qwen4_exp"),
    )
    assert response["choices"][0]["finish_reason"] == "stop"
    assert seen["kv_policy"] == KVPolicy(live_kv_format=LIVE_KV_RAW)

    with pytest.raises(http.RequestError, match="owns its K4/V4 live-cache format"):
        chat_completion(
            {
                "messages": [{"role": "user", "content": "1"}],
                "live_kv_format": "raw",
            },
            generate,
            tokenizer=_Tokenizer(),
            request_contract=PackageRequestContract(family="qwen4_exp"),
        )
