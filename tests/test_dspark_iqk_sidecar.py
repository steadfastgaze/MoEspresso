"""IQ_K experts mode of the DSpark sidecar: staging -> builder -> loader.

Builds a fake DSpark checkpoint plus a fake conversion staging directory
(expert-major ik-wire cells and the converter's inventory), then exercises
the iqk build/load round trip at the decode kernels' real row widths with a
small expert count: stored bytes are the relayout of the staged wire, the
loader installs the trunk IQ_K switch class per stage, the installed forward
matches a plain-ops reference, and the drafter-side engagement report reads
the stage counters. Fail-closed coverage: unknown and relayout-less members,
missing inventory, digest and byte-count mismatches, the `ik_wire` layout
refusal, and a truncated blocks tensor. The default mxfp4 mode is built from
the same source to prove the two arms differ.
"""

import hashlib
import json
import re
import shutil
from dataclasses import asdict

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map

from jang_tools.dsv4.mlx_model import Model, ModelArgs
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

from moespresso.core.artifact import compute_artifact_id
from moespresso.package.deepseek_v4.dspark_sidecar import (
    SIDECAR_MANIFEST_NAME,
    DSparkSidecarError,
    build_dspark_sidecar,
    iqk_blocks_name,
)
from moespresso.package.iqk_format import iqk_geometry
from moespresso.package.iqk_relayout import decode_rows, pack_rows
from moespresso.runtime.deepseek_v4.dspark_load import (
    dspark_iqk_engagement,
    load_dspark_sidecar,
)
from moespresso.runtime.deepseek_v4.dspark_model import DSparkArgs, DSparkDraftModel

VOCAB = 97
NOISE = VOCAB - 1
N_MTP = 2
BLOCK = 5
TARGET_IDS = (0, 1)
MARKOV_RANK = 8
NUM_LAYERS = 2
# The decode kernels are sized per input width (2048 and 4096), so the
# routed widths are real; the expert count stays small.
HIDDEN = 2048
MOE_INTER = 2048
N_EXPERTS = 2
MEMBER = "iq2_ks"
SWIGLU_LIMIT = 10.0

_W123 = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
_F32_SOURCE = re.compile(
    r"(hc_(attn|ffn|head)_(fn|base|scale)|self_attn\.attn_sink"
    r"|confidence_head\.proj\.weight)$"
)
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_PROJ_FILE = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}


def tiny_model_args() -> ModelArgs:
    return ModelArgs(
        model_type="deepseek_v4",
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        qk_rope_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        n_routed_experts=N_EXPERTS,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=MOE_INTER,
        num_hash_layers=0,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        swiglu_limit=SWIGLU_LIMIT,
        hc_mult=2,
        hc_sinkhorn_iters=5,
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=4096,
        sliding_window=8,
        rms_norm_eps=1e-6,
        compress_ratios=[0, 0],
    )


def tiny_dspark_args() -> DSparkArgs:
    return DSparkArgs(
        model_args=tiny_model_args(),
        n_mtp_layers=N_MTP,
        block_size=BLOCK,
        noise_token_id=NOISE,
        target_layer_ids=TARGET_IDS,
        markov_rank=MARKOV_RANK,
    )


def tiny_source_config() -> dict:
    config = asdict(tiny_model_args())
    config.update(
        n_mtp_layers=N_MTP,
        dspark_block_size=BLOCK,
        dspark_noise_token_id=NOISE,
        dspark_target_layer_ids=list(TARGET_IDS),
        dspark_markov_rank=MARKOV_RANK,
    )
    return config


def randomize(module, scale: float = 0.02, seed: int = 0) -> None:
    def _rand(a):
        nonlocal seed
        seed += 1
        return mx.random.normal(a.shape, key=mx.random.key(seed)) * scale

    module.update(tree_map(_rand, module.parameters()))


def write_source(tmp_path, seed: int = 0):
    """A raw-named two-shard snapshot; routed experts are bf16 placeholders.

    The iqk build keeps the source-completeness contract (every expert
    tensor must resolve) while never reading the expert bytes, so the
    placeholders only need the right names and shapes.
    """
    from mlx.utils import tree_flatten

    model = DSparkDraftModel(tiny_dspark_args())
    randomize(model, seed=seed)
    mx.eval(model.parameters())

    raw: dict[str, mx.array] = {}
    for name, arr in tree_flatten(model.parameters()):
        stage, rest = re.match(r"blocks\.(\d+)\.(.+)", name).groups()
        if rest.startswith("mlp.switch_mlp."):
            proj = rest.split(".")[2]
            for e in range(arr.shape[0]):
                raw[f"mtp.{stage}.ffn.experts.{e}.{_W123[proj]}.weight"] = (
                    arr[e].astype(mx.bfloat16))
            continue
        if rest.startswith("self_attn."):
            raw_rest = "attn." + rest[len("self_attn."):]
        elif rest == "input_layernorm.weight":
            raw_rest = "attn_norm.weight"
        elif rest == "post_attention_layernorm.weight":
            raw_rest = "ffn_norm.weight"
        elif rest.startswith("mlp.shared_experts."):
            proj = rest.split(".")[2]
            raw_rest = f"ffn.shared_experts.{_W123[proj]}.weight"
        elif rest.startswith("mlp.gate."):
            raw_rest = "ffn.gate." + rest[len("mlp.gate."):]
        else:
            raw_rest = rest
        dtype = mx.float32 if _F32_SOURCE.search(rest) else mx.bfloat16
        raw[f"mtp.{stage}.{raw_rest}"] = arr.astype(dtype)

    src = tmp_path / "src"
    src.mkdir()
    names = sorted(raw)
    shard_names = [
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
    ]
    half = len(names) // 2
    weight_map = {}
    for shard, members in zip(shard_names, (names[:half], names[half:])):
        mx.save_safetensors(str(src / shard), {n: raw[n] for n in members})
        weight_map.update({n: shard for n in members})
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}))
    (src / "config.json").write_text(json.dumps(tiny_source_config()))
    return src


def _wire(member: str, rows: int, in_features: int, seed: int) -> np.ndarray:
    """Random ik wire rows at serving scale magnitudes.

    Every non-scale field of the member is dense over its byte space, so
    random bytes are valid wire content; only the fp16 scales are drawn so
    a decoded weight stays small and the SwiGLU forward stays inside fp16.
    """
    rng = np.random.default_rng(seed)
    row_bytes = iqk_geometry(member).bytes_per_row(in_features)
    out = rng.integers(0, 256, size=(rows, row_bytes), dtype=np.uint8)
    nblocks = in_features // 256
    slots = [0] if member == "iq2_ks" else [b * 76 for b in range(nblocks)]
    scales = (rng.standard_normal(len(slots) * rows).astype(np.float32)
              * 0.004).astype(np.float16)
    raw = scales.view(np.uint8).reshape(rows, len(slots), 2)
    for i, slot in enumerate(slots):
        out[:, slot:slot + 2] = raw[:, i]
    return out


def _cell_features(projection: str) -> tuple[int, int]:
    if projection == "down_proj":
        return MOE_INTER, HIDDEN
    return HIDDEN, MOE_INTER


def write_staging(tmp_path, member: str = MEMBER, seed: int = 0):
    """A staging directory in the converter's own shape, plus its wires."""
    staging = tmp_path / f"staging_{member}"
    (staging / "tensors").mkdir(parents=True)
    files = {}
    wires = {}
    for stage in range(N_MTP):
        layer_index = NUM_LAYERS + stage
        for i, projection in enumerate(_PROJECTIONS):
            in_features, out_features = _cell_features(projection)
            wire = _wire(member, N_EXPERTS * out_features, in_features,
                         seed=seed + 100 * stage + 10 * i)
            name = f"layer{layer_index:02d}_{_PROJ_FILE[projection]}.{member}"
            payload = wire.tobytes()
            (staging / "tensors" / name).write_bytes(payload)
            row_bytes = wire.shape[1]
            files[name] = {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "member": member,
                "bpw": row_bytes * 8.0 / in_features,
                "bytes_per_expert": out_features * row_bytes,
                "layer_index": layer_index,
                "projection": _PROJ_FILE[projection],
            }
            wires[(stage, projection)] = wire.reshape(
                N_EXPERTS, out_features, row_bytes)
    (staging / "inventory.json").write_text(json.dumps({
        "member": member,
        "units": N_MTP * len(_PROJECTIONS),
        "files": files,
        "routed_bytes": sum(row["bytes"] for row in files.values()),
    }))
    return staging, wires


def make_target() -> Model:
    target = Model(tiny_model_args())
    randomize(target, seed=5)
    mx.eval(target.parameters())
    return target


def _reference_weights(wires) -> dict:
    """Float32 reference decode of every staged expert, per stage."""
    out = {}
    for (stage, projection), wire in wires.items():
        in_features, out_features = _cell_features(projection)
        rows = wire.reshape(-1, wire.shape[2])
        packed = pack_rows(MEMBER, rows, in_features)
        out[(stage, projection)] = decode_rows(
            MEMBER, packed, in_features,
        ).reshape(N_EXPERTS, out_features, in_features).astype(np.float32)
    return out


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return write_source(tmp_path_factory.mktemp("dspark_iqk_src"))


@pytest.fixture(scope="module")
def staging(tmp_path_factory):
    return write_staging(tmp_path_factory.mktemp("dspark_iqk_staging"))


@pytest.fixture(scope="module")
def built(source, staging, tmp_path_factory):
    out = tmp_path_factory.mktemp("dspark_iqk_pkg") / "sidecar"
    build_dspark_sidecar(source, out, experts_format="iqk",
                         routed_artifacts=staging[0])
    return out


@pytest.fixture(scope="module")
def loaded(built):
    target = make_target()
    draft, dargs = load_dspark_sidecar(built, target.model.embed, target.lm_head)
    return draft, dargs


class TestBuild:
    def test_stored_blocks_are_the_relayout_of_the_staged_wire(self, built, staging):
        _, wires = staging
        manifest = json.loads((built / SIDECAR_MANIFEST_NAME).read_text())
        for (stage, projection), wire in wires.items():
            name = iqk_blocks_name(stage, projection)
            row = manifest["tensors"][name]
            shard = mx.load(str(built / row["file"]))
            stored = np.asarray(shard[name])
            in_features, out_features = _cell_features(projection)
            want = pack_rows(
                MEMBER, wire.reshape(-1, wire.shape[2]), in_features,
            ).reshape(N_EXPERTS, out_features, -1)
            assert np.array_equal(stored, want), name

    def test_manifest_rows_carry_the_per_tensor_expert_schema(self, built, staging):
        staging_dir, _ = staging
        manifest = json.loads((built / SIDECAR_MANIFEST_NAME).read_text())
        inventory = json.loads((staging_dir / "inventory.json").read_text())
        for stage in range(N_MTP):
            for projection in _PROJECTIONS:
                row = manifest["tensors"][iqk_blocks_name(stage, projection)]
                in_features, out_features = _cell_features(projection)
                row_bytes = iqk_geometry(MEMBER).bytes_per_row(in_features)
                assert row["format"] == "iqk"
                assert row["iqk_codec"] == MEMBER
                assert row["layout"] == "iqk_relayout"
                assert row["num_experts"] == N_EXPERTS
                assert row["in_features"] == in_features
                assert row["out_features"] == out_features
                assert row["stage"] == stage
                assert row["projection"] == projection
                assert row["dtype"] == "uint8"
                assert row["shape"] == [N_EXPERTS, out_features, row_bytes]
                assert row["staging_file"] in inventory["files"]
        prov = manifest["provenance"]
        assert prov["iqk_staging"]["member"] == MEMBER
        assert prov["iqk_staging"]["file_sha256"] == {
            name: row["sha256"] for name, row in inventory["files"].items()
        }
        assert prov["iqk_gates"]["round_trip"] == "every row"
        assert prov["iqk_gates"]["rows_reference_decoded"] > 0

    def test_non_expert_treatment_is_unchanged(self, built):
        manifest = json.loads((built / SIDECAR_MANIFEST_NAME).read_text())
        rows = manifest["tensors"]
        assert rows["blocks.0.self_attn.wq_a.weight"]["format"] == "affine8"
        assert rows["blocks.0.input_layernorm.weight"]["format"] == "passthrough"
        assert rows["blocks.0.mlp.gate.weight"]["format"] == "passthrough"
        assert "blocks.0.mlp.switch_mlp.gate_proj.weight" not in rows


class TestLoadAndServe:
    def test_loader_installs_the_trunk_switch_class_per_stage(self, loaded):
        draft, _ = loaded
        for stage in range(N_MTP):
            switch = draft.blocks[stage].mlp.switch_mlp
            assert type(switch).__name__ == "IqkDeepseekV4SwitchGLU"
            assert switch.members == {p: MEMBER for p in _PROJECTIONS}
            assert switch.layer == NUM_LAYERS + stage
            assert not switch.training
            # The stage's own clamped-SwiGLU module travels into the switch.
            assert switch.activation.swiglu_limit == SWIGLU_LIMIT

    def test_draft_forward_runs_at_block_shape(self, loaded):
        draft, dargs = loaded
        assert dargs.block_size == BLOCK
        state = draft.make_state()
        n = 4
        main_hidden = mx.random.normal((1, n, HIDDEN * len(TARGET_IDS))) * 0.02
        draft.ingest(state, main_hidden, list(range(n)))
        result = draft.draft(state, anchor_token=1, anchor_pos=n)
        mx.eval(result.tokens, result.logits, result.confidence)
        assert result.tokens.shape == (1, BLOCK)
        assert result.logits.shape == (1, BLOCK, VOCAB)
        assert result.confidence.shape == (1, BLOCK)
        assert np.isfinite(np.asarray(result.logits)).all()

    def test_installed_switch_matches_the_reference_decode(self, loaded, staging):
        """One decode-vs-reference sanity through the installed seam.

        Kernel parity over the value space is the kernel repository's own
        covered ground; this pins that the loader wired the staged bytes to
        the right stage and projection.
        """
        draft, _ = loaded
        _, wires = staging
        reference = _reference_weights(wires)
        rng = np.random.default_rng(23)
        tokens, top_k = 2, 2
        x = (rng.standard_normal((tokens, HIDDEN)) * 0.5).astype(np.float16)
        indices = rng.integers(0, N_EXPERTS, size=(tokens, top_k)).astype(np.uint32)
        for stage in range(N_MTP):
            switch = draft.blocks[stage].mlp.switch_mlp
            got = np.asarray(
                switch(mx.array(x), mx.array(indices)), dtype=np.float64)
            assert got.shape == (tokens, top_k, HIDDEN)
            want = np.zeros_like(got)
            for t in range(tokens):
                row = x[t].astype(np.float64)
                for k in range(top_k):
                    e = int(indices[t, k])
                    gate = reference[(stage, "gate_proj")][e].astype(np.float64) @ row
                    up = reference[(stage, "up_proj")][e].astype(np.float64) @ row
                    gate = np.minimum(gate, SWIGLU_LIMIT)
                    up = np.clip(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
                    hidden = (gate / (1.0 + np.exp(-gate))) * up
                    want[t, k] = (
                        reference[(stage, "down_proj")][e].astype(np.float64)
                        @ hidden)
            rel = np.max(np.abs(got - want)) / max(np.max(np.abs(want)), 1e-9)
            assert rel < 5e-3, (stage, rel)

    def test_switch_serves_batched_block_rows(self, loaded):
        draft, _ = loaded
        switch = draft.blocks[0].mlp.switch_mlp
        rng = np.random.default_rng(31)
        x = mx.array(
            (rng.standard_normal((2, BLOCK, HIDDEN)) * 0.5).astype(np.float16))
        indices = mx.array(
            rng.integers(0, N_EXPERTS, size=(2, BLOCK, 2)).astype(np.uint32))
        out = switch(x, indices)
        mx.eval(out)
        assert out.shape == (2, BLOCK, 2, HIDDEN)
        assert np.isfinite(np.asarray(out)).all()

    def test_engagement_reports_the_drafter_stages(self, loaded):
        draft, _ = loaded
        state = draft.make_state()
        main_hidden = mx.random.normal((1, 2, HIDDEN * len(TARGET_IDS))) * 0.02
        draft.ingest(state, main_hidden, [0, 1])
        result = draft.draft(state, anchor_token=1, anchor_pos=2)
        mx.eval(result.tokens)
        report = dspark_iqk_engagement(draft)
        assert report["switch_modules"] == N_MTP
        assert [s["stage"] for s in report["stages"]] == list(range(N_MTP))
        assert [s["layer"] for s in report["stages"]] == [
            NUM_LAYERS + s for s in range(N_MTP)]
        for entry in report["stages"]:
            assert entry["members"] == {p: MEMBER for p in _PROJECTIONS}
        assert report["total_calls"] >= N_MTP
        assert report["gemv_calls"] + report["sorted_prefill_calls"] >= N_MTP


class TestDefaultModeUnchanged:
    def test_default_build_is_the_mxfp4_arm(self, source, tmp_path):
        out = tmp_path / "sidecar_mxfp4"
        manifest = build_dspark_sidecar(source, out)
        formats = {row["format"] for row in manifest["tensors"].values()}
        assert "iqk" not in formats
        expert_row = manifest["tensors"]["blocks.0.mlp.switch_mlp.gate_proj.weight"]
        assert expert_row["format"] == "mxfp4"
        assert "iqk_staging" not in manifest["provenance"]
        target = make_target()
        draft, _ = load_dspark_sidecar(out, target.model.embed, target.lm_head)
        assert isinstance(
            draft.blocks[0].mlp.switch_mlp.gate_proj, QuantizedSwitchLinear)

    def test_default_mode_refuses_routed_artifacts(self, source, staging, tmp_path):
        with pytest.raises(DSparkSidecarError, match="iqk-mode input"):
            build_dspark_sidecar(source, tmp_path / "sidecar",
                                 routed_artifacts=staging[0])


class TestFailClosed:
    def test_unknown_member_is_refused(self, source, tmp_path):
        staging_dir = tmp_path / "staging_bogus"
        staging_dir.mkdir()
        (staging_dir / "inventory.json").write_text(
            json.dumps({"member": "iq9_z", "files": {"x": {}}}))
        with pytest.raises(DSparkSidecarError, match="unknown IQ_K codec"):
            build_dspark_sidecar(source, tmp_path / "sidecar",
                                 experts_format="iqk",
                                 routed_artifacts=staging_dir)

    def test_member_without_a_relayout_is_refused(self, source, tmp_path):
        staging_dir = tmp_path / "staging_iq3k"
        staging_dir.mkdir()
        (staging_dir / "inventory.json").write_text(
            json.dumps({"member": "iq3_k", "files": {"x": {}}}))
        with pytest.raises(DSparkSidecarError, match="no relayout"):
            build_dspark_sidecar(source, tmp_path / "sidecar",
                                 experts_format="iqk",
                                 routed_artifacts=staging_dir)

    def test_missing_inventory_is_refused(self, source, tmp_path):
        staging_dir = tmp_path / "staging_empty"
        (staging_dir / "tensors").mkdir(parents=True)
        with pytest.raises(DSparkSidecarError, match="inventory"):
            build_dspark_sidecar(source, tmp_path / "sidecar",
                                 experts_format="iqk",
                                 routed_artifacts=staging_dir)

    def test_staging_digest_mismatch_is_refused(self, source, staging, tmp_path):
        staging_dir, _ = staging
        copy = tmp_path / "staging_tampered"
        shutil.copytree(staging_dir, copy)
        victim = sorted((copy / "tensors").iterdir())[0]
        with open(victim, "r+b") as f:
            f.seek(64)
            byte = f.read(1)
            f.seek(64)
            f.write(bytes([byte[0] ^ 0xFF]))
        with pytest.raises(DSparkSidecarError, match="sha256"):
            build_dspark_sidecar(source, tmp_path / "sidecar",
                                 experts_format="iqk",
                                 routed_artifacts=copy)

    def test_staging_byte_count_mismatch_is_refused(self, source, staging, tmp_path):
        staging_dir, _ = staging
        copy = tmp_path / "staging_truncated"
        shutil.copytree(staging_dir, copy)
        victim = sorted((copy / "tensors").iterdir())[0]
        with open(victim, "r+b") as f:
            f.truncate(victim.stat().st_size - 7)
        with pytest.raises(DSparkSidecarError, match="B on disk"):
            build_dspark_sidecar(source, tmp_path / "sidecar",
                                 experts_format="iqk",
                                 routed_artifacts=copy)

    def _rehashed(self, built, tmp_path, mutate):
        """Copy the sidecar, apply `mutate(payload)`, and re-seal the hash.

        Re-sealing keeps the artifact-id gate green so the test reaches the
        validation under test instead of the tamper detector.
        """
        copy = tmp_path / "sidecar_copy"
        shutil.copytree(built, copy)
        manifest_path = copy / SIDECAR_MANIFEST_NAME
        payload = json.loads(manifest_path.read_text())
        mutate(payload, copy)
        payload["artifact_id"] = compute_artifact_id(payload)
        manifest_path.write_text(json.dumps(payload))
        return copy

    def test_loader_serves_a_sidecar_written_before_the_name_was_corrected(
            self, built, tmp_path):
        """A sidecar declaring the pre-rename spelling still loads.

        Sidecars already written hold the old value; readers normalize it so
        their bytes do not have to be shipped again.
        """
        name = iqk_blocks_name(0, "gate_proj")

        def mutate(payload, _copy):
            for row in payload["tensors"].values():
                if row.get("layout") == "iqk_relayout":
                    row["layout"] = "ikq_relayout"

        copy = self._rehashed(built, tmp_path, mutate)
        assert json.loads((copy / "dspark_sidecar.json").read_text())[
            "tensors"][name]["layout"] == "ikq_relayout"
        target = make_target()
        draft, _dargs = load_dspark_sidecar(
            copy, target.model.embed, target.lm_head)
        assert draft is not None

    def test_loader_refuses_the_quantizers_wire_layout(self, built, tmp_path):
        name = iqk_blocks_name(0, "gate_proj")

        def mutate(payload, _copy):
            payload["tensors"][name]["layout"] = "ik_wire"

        copy = self._rehashed(built, tmp_path, mutate)
        target = make_target()
        with pytest.raises(DSparkSidecarError, match="iqk_relayout"):
            load_dspark_sidecar(copy, target.model.embed, target.lm_head)

    def test_loader_refuses_a_truncated_blocks_tensor(self, built, tmp_path):
        name = iqk_blocks_name(0, "gate_proj")
        manifest = json.loads((built / SIDECAR_MANIFEST_NAME).read_text())
        shard_name = manifest["tensors"][name]["file"]

        def mutate(payload, copy):
            shard = dict(mx.load(str(copy / shard_name)))
            shard[name] = shard[name][:, : shard[name].shape[1] // 2]
            mx.save_safetensors(str(copy / shard_name), shard,
                                metadata={"format": "mlx"})
            digest = hashlib.sha256((copy / shard_name).read_bytes()).hexdigest()
            payload["provenance"]["file_sha256"][shard_name] = digest

        copy = self._rehashed(built, tmp_path, mutate)
        target = make_target()
        with pytest.raises(DSparkSidecarError, match="shard shape"):
            load_dspark_sidecar(copy, target.model.embed, target.lm_head)

    def test_loader_refuses_a_partial_iqk_row_set(self, built, tmp_path):
        name = iqk_blocks_name(1, "down_proj")

        def mutate(payload, _copy):
            payload["tensors"][name]["format"] = "passthrough"

        copy = self._rehashed(built, tmp_path, mutate)
        target = make_target()
        with pytest.raises(DSparkSidecarError, match="do not cover"):
            load_dspark_sidecar(copy, target.model.embed, target.lm_head)

    def test_iqk_mode_requires_the_staging_directory(self, source, tmp_path):
        with pytest.raises(DSparkSidecarError, match="routed-artifacts"):
            build_dspark_sidecar(source, tmp_path / "sidecar",
                                 experts_format="iqk")
