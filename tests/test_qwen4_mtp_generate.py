"""Single-draft generation emits accepted prefixes and releases request state."""

from contextlib import nullcontext
from types import SimpleNamespace

import mlx.core as mx
import pytest

from moespresso.runtime.qwen4 import mtp_generate as module
@pytest.fixture
def loop(monkeypatch):
    observed = SimpleNamespace(
        ingests=[], closed=[], accepted=1, next_token=4, calls=0, close_error=False
    )

    def logits(tokens):
        return mx.array([[[5.0 if j == token else -5.0 for j in range(8)] for token in tokens]])

    class Coordinator:
        state = SimpleNamespace(frontier=0)

        def forward_chunk_with_widened(self, ids):
            self.state.frontier += ids.shape[1]
            return logits([1] * ids.shape[1]), mx.zeros((1, ids.shape[1], 4))

        def close(self):
            observed.closed.append("target")

    class Drafter:
        def __init__(self, loaded):
            pass

        def make_state(self):
            return SimpleNamespace(frontier=0)

        def ingest(self, state, rows, positions, tokens):
            assert positions == list(range(state.frontier, state.frontier + len(tokens)))
            assert rows.shape[1] == len(tokens)
            observed.ingests.append(list(tokens))
            state.frontier += len(tokens)

        def draft(self, state, anchor, frontier, temperature):
            assert state.frontier == frontier
            return SimpleNamespace(tokens=mx.array([[3]]), logits=logits([3]))

        def close_state(self, state):
            observed.closed.append("draft")

    class Verifier:
        def __init__(self, model, **kwargs):
            assert kwargs["shared_projections"]
            assert kwargs["expert_pair_factory"] is module.Qwen4MTPFullResidentExpertPair
            self.expert_pairs = [SimpleNamespace(paired_calls=0, staged_calls=0, shared_experts=0)]

        def verify(self, coordinator, ids, **kwargs):
            observed.calls += 1
            keep = observed.accepted + 1
            coordinator.state.frontier += keep
            emitted = ([3] if observed.accepted else []) + [observed.next_token]
            return SimpleNamespace(
                acceptance=SimpleNamespace(
                    accepted=observed.accepted, emitted=emitted, next_token=observed.next_token
                ),
                logits=logits(emitted),
                widened=mx.zeros((1, keep, 4)),
            )

        def verify_plain(self, coordinator, ids, **kwargs):
            coordinator.state.frontier += 1
            return SimpleNamespace(logits=logits([5]))

        def close(self):
            observed.closed.append("verifier")
            if observed.close_error:
                raise RuntimeError("verifier cleanup failed")

    model = SimpleNamespace(
        new_coordinator=lambda _: Coordinator(),
        _moespresso_qwen4_stop_ids={7},
        _moespresso_qwen4_kvarn_context_tokens=32,
        _moespresso_ssd_streaming_capacity=512,
        _moespresso_pooled_decode_bounded=False,
        _cache_routing_enabled=False,
    )
    tokenizer = SimpleNamespace(decode=lambda ids: str(ids[0]))
    monkeypatch.setattr(module, "Qwen4MTPDrafter", Drafter)
    monkeypatch.setattr(
        module,
        "make_mtp_verifier",
        lambda model, **kwargs: Verifier(model, **kwargs),
    )
    monkeypatch.setattr(module, "wired_limit", lambda *a: nullcontext())
    return model, tokenizer, observed


@pytest.mark.parametrize("accepted,expected", [(0, [1, 4, 4, 4, 5]), (1, [1, 3, 4, 3, 4])])
def test_acceptance_rejection_and_continuation(loop, accepted, expected):
    model, tokenizer, observed = loop
    observed.accepted = accepted
    result = module.generate_mtp(model, tokenizer, [1, 2], None, max_tokens=5)
    assert result["tokens"] == expected
    assert result["acceptance_rate"] == accepted
    assert observed.closed == ["verifier", "target", "draft"]
    assert observed.ingests[0] == [1, 2]


def test_stop_token_omits_further_drafting_and_history_ingest(loop):
    model, tokenizer, observed = loop
    observed.next_token = 7
    result = module.generate_mtp(model, tokenizer, [1, 2], None, max_tokens=8)
    assert result["tokens"] == [1, 3, 7]
    assert observed.calls == 1
    assert observed.ingests == [[1, 2]]


def test_callback_failure_closes_target_and_drafter(loop):
    model, tokenizer, observed = loop

    def fail(*args):
        raise RuntimeError("client disconnected")

    with pytest.raises(RuntimeError, match="client disconnected"):
        module.generate_mtp(model, tokenizer, [1, 2], None, max_tokens=8, response_callback=fail)
    assert observed.closed == ["verifier", "target", "draft"]


def test_cancelled_prefill_closes_state(loop):
    model, tokenizer, observed = loop
    with pytest.raises(InterruptedError):
        module.generate_mtp(model, tokenizer, [1, 2], None, max_tokens=8, cancelled=lambda: True)
    assert observed.closed == ["verifier", "target", "draft"]


@pytest.mark.parametrize("failure", ["verifier", "detokenizer", "stop_ids"])
def test_setup_failure_closes_constructed_request_state(loop, monkeypatch, failure):
    from moespresso.runtime.qwen4 import generation

    model, tokenizer, observed = loop

    def fail(*_args, **_kwargs):
        raise RuntimeError(f"{failure} setup failed")

    if failure == "verifier":
        monkeypatch.setattr(module, "make_mtp_verifier", fail)
        expected = ["draft"]
    elif failure == "detokenizer":
        monkeypatch.setattr(generation, "_detokenizer", fail)
        expected = ["verifier", "draft"]
    else:
        monkeypatch.setattr(generation, "_stop_ids", fail)
        expected = ["verifier", "draft"]
    with pytest.raises(RuntimeError, match=f"{failure} setup failed"):
        module.generate_mtp(model, tokenizer, [1, 2], None, max_tokens=5)
    assert observed.closed == expected


def test_verifier_cleanup_failure_still_closes_target_and_drafter(loop):
    model, tokenizer, observed = loop
    observed.close_error = True
    with pytest.raises(RuntimeError, match="verifier cleanup failed"):
        module.generate_mtp(model, tokenizer, [1, 2], None, max_tokens=1)
    assert observed.closed == ["verifier", "target", "draft"]


def test_terminal_single_token_does_not_draft(loop):
    model, tokenizer, observed = loop
    result = module.generate_mtp(model, tokenizer, [1, 2], None, max_tokens=1)
    assert result["tokens"] == [1]
    assert observed.calls == 0


def test_installed_command_reserves_drafter_memory_and_records_configuration(monkeypatch, tmp_path):
    import json

    from moespresso.core import artifact
    from moespresso.package.qwen4 import mtp_format
    from moespresso.runtime import http, prefix_cache
    from moespresso.runtime.qwen4 import load, mtp_load

    observed = {}
    model = SimpleNamespace(
        _moespresso_ssd_streaming_capacity=512,
        _moespresso_pooled_decode_bounded=False,
        _cache_routing_enabled=False,
        close=lambda: observed.update(closed=True),
    )
    manifest = {"artifact_id": "target", "architecture": {"family": "qwen4_exp"}}
    sidecar = {
        "artifact_id": "draft",
        "graph": {
            "num_key_value_heads": 2,
            "head_dim": 256,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
        },
        "byte_estimate": {"iq2_k_payload_with_input_padding": 1024},
    }

    def load_target(*args, **kwargs):
        observed.update(kwargs)
        return model, object()

    monkeypatch.setattr(artifact, "read_artifact", lambda *a: manifest)
    monkeypatch.setattr(mtp_format, "read_mtp_sidecar_manifest", lambda *a: sidecar)
    monkeypatch.setattr(load, "load_qwen4_iqk_package_model", load_target)
    monkeypatch.setattr(mtp_load, "load_qwen4_mtp_sidecar", lambda *a, **kw: object())
    monkeypatch.setattr(http, "render_prompt", lambda *a, **kw: "rendered")
    monkeypatch.setattr(http, "qwen4_contract_template_kwargs", lambda *a, **kw: {})
    monkeypatch.setattr(prefix_cache, "encode_rendered_prompt", lambda *a: [1])
    monkeypatch.setattr(
        module, "generate_mtp", lambda *a, **kw: {"tokens": [2], "text": "ok", "rounds": []}
    )
    monkeypatch.setattr(module.mx, "device_info", lambda: {"device_name": "test GPU"})
    # Restore process-local command settings after this in-process CLI invocation.
    for name in (
        "MOESPRESSO_DISK_KV",
        "MOESPRESSO_DS4_DRAFTER",
        "MOESPRESSO_SSD_GROWTH_MAX_EXTRA_GB",
        "MOESPRESSO_SSD_MAX_MEMORY_GB",
    ):
        monkeypatch.setenv(name, "original")
    result_path = tmp_path / "result.json"
    assert (
        module.main(
            [
                str(tmp_path),
                "--sidecar",
                str(tmp_path),
                "--prompt",
                "test",
                "--memory-gb",
                "24",
                "--max-context-tokens",
                "512",
                "--json-out",
                str(result_path),
            ]
        )
        == 0
    )
    result = json.loads(result_path.read_text())
    expected = 1024 + 512 * ((2 * 2 * 256 + 128) * 4 + 24)
    assert observed["additional_resident_bytes"] == expected
    assert observed["cache_routing"] == "off"
    assert "cache_routing_factor" not in observed
    assert "cache_routing_protected_routes" not in observed
    assert observed["closed"]
    assert result["drafter_reserved_bytes"] == expected
    assert result["target_capacity_per_layer"] == 512
    assert result["package_artifact"] == "target"
    assert result["sidecar_artifact"] == "draft"


def test_installed_command_refuses_bounded_target(monkeypatch, tmp_path):
    from moespresso.core import artifact
    from moespresso.package.qwen4 import mtp_format
    from moespresso.runtime.qwen4 import load

    manifest = {"artifact_id": "target", "architecture": {"family": "qwen4_exp"}}
    sidecar = {
        "artifact_id": "draft",
        "graph": {
            "num_key_value_heads": 2,
            "head_dim": 256,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
        },
        "byte_estimate": {"iq2_k_payload_with_input_padding": 1024},
    }
    closed = []
    model = SimpleNamespace(
        _moespresso_ssd_streaming_capacity=511,
        _moespresso_pooled_decode_bounded=True,
        _cache_routing_enabled=False,
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(artifact, "read_artifact", lambda *a: manifest)
    monkeypatch.setattr(mtp_format, "read_mtp_sidecar_manifest", lambda *a: sidecar)
    monkeypatch.setattr(load, "load_qwen4_iqk_package_model", lambda *a, **kw: (model, object()))
    with pytest.raises(ValueError, match="full target expert residency"):
        module.main([str(tmp_path), "--sidecar", str(tmp_path), "--prompt", "test"])
    assert closed == [True]


def test_generate_refuses_cache_biased_full_resident_model(loop):
    model, tokenizer, observed = loop
    model._cache_routing_enabled = True
    with pytest.raises(ValueError, match="full target expert residency"):
        module.generate_mtp(model, tokenizer, [1, 2], None, max_tokens=5)
    assert observed.closed == []
