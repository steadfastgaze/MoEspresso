"""End-to-end serve smoke (model-free, runs anywhere).

Builds a tiny real qwen3_5_moe text model, writes it in source safetensors form,
runs the full pipeline (inventory -> probe -> optimize -> write_package with
passthrough), rebuilds it via the manifest-driven backend (build_model), and
asserts the served model produces non-degenerate logits that agree with the
reference model on most top tokens. A mis-assembled expert / dropped structural
tensor / config.json archaeology would all fail here, before any
real run reaches a high-memory parity environment.

Requires the runtime dependencies and is tiny enough to run in seconds.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
pytest.importorskip("jang_tools.turboquant")

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from moespresso.inventory.build import build_inventory  # noqa: E402
from moespresso.optimize.decide import decide  # noqa: E402
from moespresso.package.plan import package_plan_from_decision  # noqa: E402
from moespresso.package.write import write_package  # noqa: E402
from moespresso.probe.build import build_probe_evidence  # noqa: E402
from moespresso.runtime.build import build_model  # noqa: E402

# Tiny text config. full_attention_interval=2 -> layer 0 linear, layer 1 full attn.
def _text_config() -> dict:
    # A fresh dict per test (mlx_lm / our builder may read it back) so no test
    # pollutes another. dims divisible by affine group sizes (128/64/32).
    return {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 128, "num_hidden_layers": 2, "intermediate_size": 256,
        "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 16,
        "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 128,
        "shared_expert_intermediate_size": 128, "decoder_sparse_step": 1,
        "mlp_only_layers": [], "rms_norm_eps": 1e-6, "vocab_size": 256,
        "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
        "max_position_embeddings": 4096,
        "linear_num_value_heads": 4, "linear_num_key_heads": 2,
        "linear_key_head_dim": 32, "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 4, "full_attention_interval": 2,
        "tie_word_embeddings": False,
        "layer_types": ["linear_attention", "full_attention"],
    }


def _arch() -> dict:
    return {"model_type": "qwen3_5_moe", "text_config": _text_config()}


LAYER_TYPES = ["linear_attention", "full_attention"]


def _reference_model():
    import mlx_lm.models.qwen3_5_moe as M
    mx.random.seed(0)  # deterministic init so the correlation assertion is stable
    args = M.ModelArgs(model_type="qwen3_5_moe", text_config=_text_config())
    model = M.Model(args)
    mx.eval(model.parameters())
    return model


def _param_to_source_name(param_path: str) -> str:
    """Inverse of sanitize's rename: language_model.model.* -> model.language_model.*"""
    if param_path.startswith("language_model.model."):
        return "model.language_model." + param_path[len("language_model.model."):]
    if param_path.startswith("language_model.lm_head."):
        return "lm_head." + param_path[len("language_model.lm_head."):]
    return param_path


# The RMSNorm weights mlx_lm's qwen3_5 sanitize shifts by +1.0 (qwen3_5.py:318-324)
# when the checkpoint looks unsanitized (conv1d last dim != 1 OR mtp present). Real
# Qwen stores these norms in a -1.0 "source convention"; sanitize re-adds the +1.0 at
# load. Our fixture must mirror that or it doesn't exercise the real round-trip.
_SANITIZE_SHIFT_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


def _dump_source_safetensors(model, path):
    """Write the model's params in source checkpoint form (inverse-sanitize names,
    fused experts). switch_mlp.{gate,up} -> experts.gate_up_proj; conv1d back to
    [out,1,k]; RMSNorm weights shifted by -1.0 into source convention so that
    load-time sanitize (which fires because conv1d is [out,1,k]) re-adds the +1.0 and
    recovers the model's actual norms. This mirrors the real Qwen checkpoint; storing
    norms as-is here would not survive the sanitize +1.0 shift, which would silently
    mask the convention mismatch instead of exercising it."""
    flat = dict(tree_flatten(model.parameters()))
    tensors = {}
    # group switch_mlp gate/up by layer to refuse them and re-fuse.
    switch = {}  # layer -> {gate,up,down}
    for path_key, arr in flat.items():
        a = np.array(arr, dtype=np.float32)
        if "switch_mlp" in path_key:
            # language_model.model.layers.N.mlp.switch_mlp.{gate,up,down}_proj.weight
            parts = path_key.split(".")
            layer = parts[parts.index("layers") + 1]
            proj = parts[-2]  # gate_proj/up_proj/down_proj
            switch.setdefault(layer, {})[proj] = a
            continue
        name = _param_to_source_name(path_key)
        if name.endswith("linear_attn.conv1d.weight") and a.ndim == 3 and a.shape[2] == 1:
            a = np.ascontiguousarray(np.moveaxis(a, 1, 2))  # [out,k,1] -> [out,1,k] (source)
        if a.ndim == 1 and any(name.endswith(sfx) for sfx in _SANITIZE_SHIFT_NORM_SUFFIXES):
            a = a - 1.0  # source convention; sanitize re-adds +1.0 at load
        tensors[name] = a
    for layer, sm in switch.items():
        base = f"model.language_model.layers.{layer}.mlp.experts"
        # fuse gate over up per expert: [E,moe,hid] + [E,moe,hid] -> [E,2*moe,hid]
        tensors[f"{base}.gate_up_proj"] = np.concatenate(
            [sm["gate_proj"], sm["up_proj"]], axis=1)
        tensors[f"{base}.down_proj"] = sm["down_proj"]
    _write_safetensors(path, tensors)


def _write_safetensors(path, tensors):
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


def _emit_sidecars(manifest, out_dir):
    """Write the jang-compatible sidecars (as convert does) so build_model can serve."""
    from moespresso.package.sidecars import build_sidecars
    config_json, jang_config = build_sidecars(manifest)
    (out_dir / "config.json").write_text(json.dumps(config_json, indent=2))
    (out_dir / "jang_config.json").write_text(json.dumps(jang_config, indent=2))


def _logits(model, ids):
    out = model(mx.array([ids]))
    mx.eval(out)
    return np.array(out[0, -1], dtype=np.float32)


def _plan_from_probe(ev):
    plan, _summary = package_plan_from_decision(decide(ev, target_quality=0.5))
    return plan


def test_manifest_driven_serve_matches_reference(tmp_path):
    ref = _reference_model()
    src = tmp_path / "src"
    src.mkdir()
    arch = _arch()
    _dump_source_safetensors(ref, src / "model-00001.safetensors")
    (src / "config.json").write_text(json.dumps(arch))

    inv = build_inventory(src, layer_types=LAYER_TYPES)
    assert inv["counts"]["passthrough"] > 0  # norms + ssm state carried
    passthrough = [e for e in inv["tensors"] if e["kind"] == "passthrough"]
    ev = build_probe_evidence(inv, src, expert_sample=2, sample_rows=64)
    plan = _plan_from_probe(ev)
    out = tmp_path / "pkg"
    man = write_package(plan, src, arch, out, passthrough=passthrough)
    assert man["status"] == "valid"
    _emit_sidecars(man, out)  # config.json + jang_config.json (what convert writes)

    # Build via the proven jang loader from the package's sidecars (no source files).
    served, _ = build_model(man, out)

    ids = [1, 5, 9, 13, 2, 7]
    ref_logits = _logits(ref, ids)
    got_logits = _logits(served, ids)

    # Non-degenerate: the served logits must vary (not collapsed to one value).
    assert float(np.std(got_logits)) > 1e-3
    assert np.isfinite(got_logits).all()

    # Agreement: quantization perturbs logits, but the served model must track the
    # reference. On a random-init tiny model the lower-ranked tokens are near-ties
    # (top-k overlap is noise), so we assert the robust signals: a high logit
    # correlation and the same argmax. A mis-assembled expert / dropped structural
    # tensor / archaeology gives correlation ~0. This catches it.
    corr = float(np.corrcoef(ref_logits, got_logits)[0, 1])
    assert corr > 0.5, f"served logits barely correlate with reference (r={corr:.3f})"
    assert int(got_logits.argmax()) == int(ref_logits.argmax())


def test_smoke_artifact_builds_and_runs(tmp_path):
    # A reduced-expert smoke package (1 expert/layer) is a mechanical soundness
    # check and carries no quality claim. With 1 of N experts the FFN is gutted, so output is
    # expected to be incoherent. The point is that it converts, builds, and runs a
    # full forward producing finite output without crashing/OOM/shape errors: the
    # cheap real-run safeguard before a full convert. The manifest clamps
    # num_experts to 1 and the graph has exactly the experts on disk.
    from moespresso.package.write import write_package

    ref = _reference_model()
    src = tmp_path / "src"
    src.mkdir()
    arch = _arch()
    _dump_source_safetensors(ref, src / "model-00001.safetensors")
    (src / "config.json").write_text(json.dumps(arch))

    inv = build_inventory(src, layer_types=LAYER_TYPES)
    passthrough = [e for e in inv["tensors"] if e["kind"] == "passthrough"]
    ev = build_probe_evidence(inv, src, expert_sample=2, sample_rows=64)
    plan = _plan_from_probe(ev)
    out = tmp_path / "pkg"
    man = write_package(plan, src, arch, out, passthrough=passthrough, max_experts=1)
    assert man["status"] == "valid"
    assert man["architecture"]["config"]["num_experts"] == 1
    assert man["architecture"]["smoke_max_experts"] == 1
    _emit_sidecars(man, out)

    # Builds from the manifest and runs a full forward -> finite output (not NaN/inf).
    # We do not assert coherence: 1/N experts cannot be coherent by construction.
    served, _ = build_model(man, out)
    got = _logits(served, [1, 5, 9, 13, 2, 7])
    assert np.isfinite(got).all()
    assert float(np.std(got)) > 1e-3  # non-degenerate


def test_ssd_streaming_tiny_package_matches_resident_with_forced_misses(tmp_path):
    ref = _reference_model()
    src = tmp_path / "src"
    src.mkdir()
    arch = _arch()
    _dump_source_safetensors(ref, src / "model-00001.safetensors")
    (src / "config.json").write_text(json.dumps(arch))

    inv = build_inventory(src, layer_types=LAYER_TYPES)
    passthrough = [e for e in inv["tensors"] if e["kind"] == "passthrough"]
    ev = build_probe_evidence(inv, src, expert_sample=2, sample_rows=64)
    plan = _plan_from_probe(ev)
    out = tmp_path / "pkg"
    man = write_package(plan, src, arch, out, passthrough=passthrough)
    _emit_sidecars(man, out)

    resident, _ = build_model(man, out)
    from moespresso.runtime.ssd_streaming_build import (
        build_ssd_streaming_model,
        ssd_streaming_stats,
    )

    streamed = None
    try:
        streamed, _, installed = build_ssd_streaming_model(out, capacity_per_layer=2)
        assert installed == 2

        ids = [1]  # one token: active set <= top_k, so capacity 2 forces real misses.
        resident_logits = _logits(resident, ids)
        streamed_logits = _logits(streamed, ids)

        corr = float(np.corrcoef(resident_logits, streamed_logits)[0, 1])
        assert corr > 0.999, f"streamed logits diverged from resident (r={corr:.6f})"
        assert int(streamed_logits.argmax()) == int(resident_logits.argmax())
        np.testing.assert_allclose(streamed_logits, resident_logits, rtol=0, atol=3e-3)
        stats = ssd_streaming_stats(streamed)
        assert stats["enabled"] is True
        assert stats["switch_modules"] == 2
        assert stats["expert_misses"] > 0
        assert stats["capacity_per_layer"] == 2
    finally:
        lock = getattr(streamed, "_moespresso_ssd_streaming_lock", None)
        if lock is not None:
            lock.close()


def test_ssd_streaming_identity_capacity_matches_resident_prefill(tmp_path):
    ref = _reference_model()
    src = tmp_path / "src"
    src.mkdir()
    arch = _arch()
    _dump_source_safetensors(ref, src / "model-00001.safetensors")
    (src / "config.json").write_text(json.dumps(arch))

    inv = build_inventory(src, layer_types=LAYER_TYPES)
    passthrough = [e for e in inv["tensors"] if e["kind"] == "passthrough"]
    ev = build_probe_evidence(inv, src, expert_sample=2, sample_rows=64)
    plan = _plan_from_probe(ev)
    out = tmp_path / "pkg"
    man = write_package(plan, src, arch, out, passthrough=passthrough)
    _emit_sidecars(man, out)

    resident, _ = build_model(man, out)
    from moespresso.runtime.ssd_streaming_build import (
        build_ssd_streaming_model,
        ssd_streaming_stats,
    )

    streamed = None
    try:
        streamed, _, installed = build_ssd_streaming_model(out, capacity_per_layer=4)
        assert installed == 2

        ids = [1, 5, 9, 13, 2, 7]
        resident_logits = _logits(resident, ids)
        streamed_logits = _logits(streamed, ids)

        corr = float(np.corrcoef(resident_logits, streamed_logits)[0, 1])
        assert corr > 0.999, f"identity-capacity logits diverged (r={corr:.6f})"
        assert int(streamed_logits.argmax()) == int(resident_logits.argmax())
        np.testing.assert_allclose(streamed_logits, resident_logits, rtol=0, atol=3e-3)
        stats = ssd_streaming_stats(streamed)
        assert stats["capacity_per_layer"] == 4
        assert stats["capacity_budget"] is None  # explicit capacity, no auto budget
    finally:
        lock = getattr(streamed, "_moespresso_ssd_streaming_lock", None)
        if lock is not None:
            lock.close()
