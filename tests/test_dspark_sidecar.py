"""Synthetic-source tests for the DSpark sidecar builder and loader.

Builds a fake DSpark checkpoint (raw `mtp.*` names, safetensors shards plus an
index) at tiny dimensions, then exercises the full build/load round trip:
quantized module formats, passthrough dtypes, manifest truthfulness, hash
tamper detection, and fail-closed behavior on missing or unexpected source
tensors. The FP4 variant checks the lossless mxfp4 repack against the probe
codec dequantization.
"""

import json
import re
import shutil
import sys
from dataclasses import asdict

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map

from jang_tools.dsv4.mlx_model import Model, ModelArgs
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

from moespresso.package.deepseek_v4.dspark_sidecar import (
    SIDECAR_KIND,
    SIDECAR_MANIFEST_NAME,
    DSparkSidecarError,
    build_dspark_sidecar,
)
from moespresso.probe.deepseek_v4.codec import dequant_fp4_e2m1_ue8m0
from moespresso.runtime.deepseek_v4 import dspark_load as dspark_load_module
from moespresso.runtime.deepseek_v4.dspark_load import load_dspark_sidecar
from moespresso.runtime.deepseek_v4.dspark_model import DSparkArgs, DSparkDraftModel

VOCAB = 97
NOISE = VOCAB - 1
N_MTP = 2
BLOCK = 3
TARGET_IDS = (0, 1)
MARKOV_RANK = 8

_W123 = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
# Stage-relative final names stored as fp32 control tensors in the fake source.
_F32_SOURCE = re.compile(
    r"(hc_(attn|ffn|head)_(fn|base|scale)|self_attn\.attn_sink"
    r"|confidence_head\.proj\.weight)$"
)


def tiny_model_args(**overrides) -> ModelArgs:
    base = dict(
        model_type="deepseek_v4",
        vocab_size=VOCAB,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        qk_rope_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        num_hash_layers=0,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        hc_mult=2,
        hc_sinkhorn_iters=5,
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=4096,
        sliding_window=8,
        rms_norm_eps=1e-6,
        compress_ratios=[0, 0],
    )
    base.update(overrides)
    return ModelArgs(**base)


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


def randomize(module: nn.Module, scale: float = 0.08, seed: int = 0) -> None:
    def _rand(a):
        nonlocal seed
        seed += 1
        return mx.random.normal(a.shape, key=mx.random.key(seed)) * scale

    module.update(tree_map(_rand, module.parameters()))


def raw_source_weights(seed: int = 0, expert_mode: str = "bf16") -> dict:
    """Raw-named source tensors for a random tiny drafter.

    Names invert `sanitize_dspark_weights`: `attn`/`ffn`/`attn_norm`/`ffn_norm`
    plus per-expert `ffn.experts.E.wK` splits, matching the reference
    checkpoint naming. Most tensors are bf16; hyper-connection parameters, the
    attention sink, and the confidence projection are fp32 controls. With
    `expert_mode="fp4"` the routed experts become packed FP4 codes with UE8M0
    per-32 scales, the real checkpoint layout.
    """
    model = DSparkDraftModel(tiny_dspark_args())
    randomize(model, seed=seed)
    mx.eval(model.parameters())

    rng = np.random.default_rng(seed + 17)
    raw: dict[str, mx.array] = {}
    for name, arr in tree_flatten(model.parameters()):
        stage, rest = re.match(r"blocks\.(\d+)\.(.+)", name).groups()
        if rest.startswith("mlp.switch_mlp."):
            proj = rest.split(".")[2]
            wkey = _W123[proj]
            for e in range(arr.shape[0]):
                base = f"mtp.{stage}.ffn.experts.{e}.{wkey}"
                if expert_mode == "fp4":
                    out_dim, in_dim = arr.shape[1], arr.shape[2]
                    packed = rng.integers(
                        0, 256, size=(out_dim, in_dim // 2), dtype=np.uint8
                    ).view(np.int8)
                    scale = rng.integers(
                        120, 134, size=(out_dim, in_dim // 32), dtype=np.uint8
                    )
                    raw[f"{base}.weight"] = mx.array(packed)
                    raw[f"{base}.scale"] = mx.array(scale)
                else:
                    raw[f"{base}.weight"] = arr[e].astype(mx.bfloat16)
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
        if _F32_SOURCE.search(rest):
            raw[f"mtp.{stage}.{raw_rest}"] = arr.astype(mx.float32)
        else:
            raw[f"mtp.{stage}.{raw_rest}"] = arr.astype(mx.bfloat16)
    return raw


def write_source(tmp_path, raw: dict, distractors: bool = True):
    """Write the raw tensors as a two-shard snapshot with an index."""
    src = tmp_path / "src"
    src.mkdir()
    raw = dict(raw)
    if distractors:
        raw["mtp.0.embed.weight"] = mx.zeros((VOCAB, 64), dtype=mx.bfloat16)
        raw[f"mtp.{N_MTP - 1}.head.weight"] = mx.zeros((VOCAB, 64), dtype=mx.bfloat16)
        raw["layers.0.attn.wq_a.weight"] = mx.zeros((32, 64), dtype=mx.bfloat16)

    names = sorted(raw)
    shard_names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    half = len(names) // 2
    shards = {shard_names[0]: names[:half], shard_names[1]: names[half:]}
    weight_map = {}
    for shard, members in shards.items():
        mx.save_safetensors(str(src / shard), {n: raw[n] for n in members})
        weight_map.update({n: shard for n in members})
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    (src / "config.json").write_text(json.dumps(tiny_source_config()))
    return src


def make_target(seed: int = 5) -> Model:
    target = Model(tiny_model_args())
    randomize(target, seed=seed)
    mx.eval(target.parameters())
    return target


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One shared bf16-source build: (source dir, sidecar dir, raw tensors)."""
    tmp = tmp_path_factory.mktemp("dspark_sidecar_bf16")
    raw = raw_source_weights(seed=0, expert_mode="bf16")
    src = write_source(tmp, raw)
    out = tmp / "sidecar"
    build_dspark_sidecar(src, out)
    return src, out, raw


class TestBuildAndLoad:
    def test_round_trip_draft_runs(self, built):
        _, out, _ = built
        target = make_target()
        draft, dargs = load_dspark_sidecar(out, target.model.embed, target.lm_head)
        assert dargs.block_size == BLOCK
        assert dargs.noise_token_id == NOISE

        # Routed experts are mxfp4-quantized switch projections.
        gate_proj = draft.blocks[0].mlp.switch_mlp.gate_proj
        assert isinstance(gate_proj, QuantizedSwitchLinear)
        assert gate_proj.mode == "mxfp4"
        assert gate_proj.scales.dtype == mx.uint8
        # Dense projections are affine 8-bit, group size 32.
        wq_a = draft.blocks[0].self_attn.wq_a
        assert isinstance(wq_a, nn.QuantizedLinear)
        assert wq_a.bits == 8 and wq_a.group_size == 32
        main_proj = draft.blocks[0].main_proj
        assert isinstance(main_proj, nn.QuantizedLinear)
        # Passthrough tensors keep their declared dtypes.
        assert draft.blocks[0].input_layernorm.weight.dtype == mx.bfloat16
        assert draft.blocks[0].hc_attn_fn.dtype == mx.float32
        assert draft.blocks[0].mlp.gate.weight.dtype == mx.bfloat16
        assert not hasattr(draft.blocks[0].mlp.gate, "scales")
        last = draft.blocks[-1]
        assert last.markov_head.markov_w1.weight.dtype == mx.bfloat16
        assert last.confidence_head.proj.weight.dtype == mx.float32

        state = draft.make_state()
        n = 4
        main_hidden = mx.random.normal((1, n, 64 * len(TARGET_IDS)))
        draft.ingest(state, main_hidden, list(range(n)))
        result = draft.draft(state, anchor_token=1, anchor_pos=n)
        mx.eval(result.tokens, result.logits, result.confidence)
        assert result.tokens.shape == (1, BLOCK)
        assert result.logits.shape == (1, BLOCK, VOCAB)
        assert result.confidence.shape == (1, BLOCK)

    def test_manifest_shape(self, built):
        _, out, _ = built
        manifest = json.loads((out / SIDECAR_MANIFEST_NAME).read_text())
        assert manifest["artifact_kind"] == SIDECAR_KIND
        assert manifest["schema_version"]["major"] == 1
        assert manifest["artifact_id"].startswith("art:")
        assert manifest["dspark"]["target_layer_ids"] == list(TARGET_IDS)
        hashes = manifest["provenance"]["file_sha256"]
        assert len(hashes) == N_MTP
        for file_name in hashes:
            assert (out / file_name).exists()
        rows = manifest["tensors"]
        expert_row = rows["blocks.0.mlp.switch_mlp.gate_proj.weight"]
        assert expert_row["format"] == "mxfp4"
        assert expert_row["lossless"] is False
        assert rows["blocks.0.self_attn.wq_a.weight"]["format"] == "affine8"
        norm_row = rows["blocks.0.input_layernorm.weight"]
        assert norm_row["format"] == "passthrough"
        assert norm_row["dtype"] == "bfloat16"
        sink_row = rows["blocks.1.self_attn.attn_sink"]
        assert sink_row["format"] == "passthrough"
        assert sink_row["dtype"] == "float32"
        for name, row in rows.items():
            assert row["file"] in hashes, name

    def test_affine8_dequantizes_close_to_source(self, built):
        _, out, raw = built
        manifest = json.loads((out / SIDECAR_MANIFEST_NAME).read_text())
        row = manifest["tensors"]["blocks.0.self_attn.wq_a.weight"]
        shard = mx.load(str(out / row["file"]))
        deq = mx.dequantize(
            shard["blocks.0.self_attn.wq_a.weight"],
            shard["blocks.0.self_attn.wq_a.scales"],
            shard["blocks.0.self_attn.wq_a.biases"],
            group_size=32, bits=8, mode="affine",
        )
        src = np.array(raw["mtp.0.attn.wq_a.weight"].astype(mx.float32))
        err = np.abs(np.array(deq.astype(mx.float32)) - src)
        assert err.max() < 0.01

    def test_mxfp4_from_bf16_is_lossy_and_recorded(self, built):
        _, out, raw = built
        manifest = json.loads((out / SIDECAR_MANIFEST_NAME).read_text())
        row = manifest["tensors"]["blocks.0.mlp.switch_mlp.gate_proj.weight"]
        assert row["lossless"] is False
        shard = mx.load(str(out / row["file"]))
        deq = mx.dequantize(
            shard["blocks.0.mlp.switch_mlp.gate_proj.weight"][0],
            shard["blocks.0.mlp.switch_mlp.gate_proj.scales"][0],
            None,
            group_size=32, bits=4, mode="mxfp4",
        )
        src = np.array(raw["mtp.0.ffn.experts.0.w1.weight"].astype(mx.float32))
        err = np.abs(np.array(deq.astype(mx.float32)) - src)
        assert err.max() > 0.0
        assert err.max() < 0.2

    def test_fp4_source_repacks_losslessly(self, tmp_path):
        raw = raw_source_weights(seed=3, expert_mode="fp4")
        src = write_source(tmp_path, raw)
        out = tmp_path / "sidecar"
        build_dspark_sidecar(src, out)
        manifest = json.loads((out / SIDECAR_MANIFEST_NAME).read_text())
        for stage in range(N_MTP):
            for proj in ("gate_proj", "down_proj", "up_proj"):
                row = manifest["tensors"][f"blocks.{stage}.mlp.switch_mlp.{proj}.weight"]
                assert row["format"] == "mxfp4"
                assert row["lossless"] is True

        target = make_target()
        draft, _ = load_dspark_sidecar(out, target.model.embed, target.lm_head)
        qsl = draft.blocks[1].mlp.switch_mlp.down_proj
        for e in range(4):
            deq = mx.dequantize(
                qsl.weight[e], qsl.scales[e], None,
                group_size=32, bits=4, mode="mxfp4",
            ).astype(mx.float32)
            ref = dequant_fp4_e2m1_ue8m0(
                np.array(raw[f"mtp.1.ffn.experts.{e}.w2.weight"]),
                np.array(raw[f"mtp.1.ffn.experts.{e}.w2.scale"]),
                out_dtype=np.float32,
            )
            assert np.array_equal(np.array(deq), ref)


class TestFailClosed:
    def test_missing_source_tensor_listed(self, tmp_path):
        raw = raw_source_weights(seed=1)
        del raw["mtp.1.attn.wkv.weight"]
        src = write_source(tmp_path, raw)
        with pytest.raises(DSparkSidecarError, match="blocks.1.self_attn.wkv.weight"):
            build_dspark_sidecar(src, tmp_path / "sidecar")

    def test_missing_expert_listed(self, tmp_path):
        raw = raw_source_weights(seed=2)
        del raw["mtp.0.ffn.experts.2.w1.weight"]
        src = write_source(tmp_path, raw)
        with pytest.raises(DSparkSidecarError, match="expert set mismatch"):
            build_dspark_sidecar(src, tmp_path / "sidecar")

    def test_unexpected_source_tensor_listed(self, tmp_path):
        raw = raw_source_weights(seed=4)
        raw["mtp.0.bogus.weight"] = mx.zeros((4, 4), dtype=mx.bfloat16)
        src = write_source(tmp_path, raw)
        with pytest.raises(DSparkSidecarError, match="blocks.0.bogus.weight"):
            build_dspark_sidecar(src, tmp_path / "sidecar")

    def test_missing_shard_file_listed(self, tmp_path):
        raw = raw_source_weights(seed=6)
        src = write_source(tmp_path, raw)
        victim = "model-00002-of-00002.safetensors"
        (src / victim).unlink()
        with pytest.raises(DSparkSidecarError, match=victim):
            build_dspark_sidecar(src, tmp_path / "sidecar")

    def test_tampered_shard_rejected(self, built, tmp_path):
        _, out, _ = built
        copy = tmp_path / "sidecar_copy"
        shutil.copytree(out, copy)
        manifest = json.loads((copy / SIDECAR_MANIFEST_NAME).read_text())
        victim = sorted(manifest["provenance"]["file_sha256"])[0]
        with open(copy / victim, "r+b") as f:
            f.seek(-16, 2)
            byte = f.read(1)
            f.seek(-16, 2)
            f.write(bytes([byte[0] ^ 0xFF]))
        target = make_target()
        with pytest.raises(DSparkSidecarError, match="hash mismatch"):
            load_dspark_sidecar(copy, target.model.embed, target.lm_head)

    def test_tampered_manifest_rejected(self, built, tmp_path):
        _, out, _ = built
        copy = tmp_path / "sidecar_copy"
        shutil.copytree(out, copy)
        manifest_path = copy / SIDECAR_MANIFEST_NAME
        payload = json.loads(manifest_path.read_text())
        payload["dspark"]["block_size"] = 99
        manifest_path.write_text(json.dumps(payload))
        target = make_target()
        with pytest.raises(DSparkSidecarError, match="hash mismatch"):
            load_dspark_sidecar(copy, target.model.embed, target.lm_head)


# Dimension overrides that make the constructed float32 skeleton dwarf the
# loaded sidecar bytes; at the default tiny dims the whole tree is a few
# hundred KiB and a leak would vanish into allocator noise.
_SCALED_MARGS = dict(
    vocab_size=4096,
    hidden_size=256,
    n_routed_experts=32,
    moe_intermediate_size=256,
)


@pytest.fixture(scope="class")
def scaled_built(tmp_path_factory):
    """A sidecar built at the scaled dims, with the dims left patched in.

    The patch stays active while the class runs so `tiny_dspark_args` and
    `make_target` agree with the built sidecar.
    """
    mp = pytest.MonkeyPatch()
    mod = sys.modules[__name__]
    base_args = tiny_model_args
    mp.setattr(mod, "VOCAB", _SCALED_MARGS["vocab_size"])
    mp.setattr(mod, "NOISE", _SCALED_MARGS["vocab_size"] - 1)
    mp.setattr(
        mod, "tiny_model_args",
        lambda **overrides: base_args(**{**_SCALED_MARGS, **overrides}),
    )
    try:
        tmp = tmp_path_factory.mktemp("dspark_sidecar_scaled")
        raw = raw_source_weights(seed=11, expert_mode="bf16")
        src = write_source(tmp, raw)
        out = tmp / "sidecar"
        build_dspark_sidecar(src, out)
        yield out
    finally:
        mp.undo()


class TestSkeletonStrip:
    """The load path must never be able to materialize the skeleton.

    Constructing the draft model fills the tree with lazy random-init
    float32 (about 72 GiB for the production three-block drafter), and
    `nn.quantize` chains quantization graphs onto those arrays. The strict
    load replaces every leaf before the hydration eval, so the success path
    never evaluated the skeleton; the hazard is the window between
    construction and the strict load, where any evaluation reaching the
    tree allocates the whole skeleton at once. The loader closes the window
    by swapping every constructed leaf for a zero-stride placeholder before
    reading shards.
    """

    def _skeleton_bytes(self) -> int:
        model = DSparkDraftModel(tiny_dspark_args())
        return sum(a.nbytes for _, a in tree_flatten(model.parameters()))

    def test_unstripped_skeleton_eval_allocates(self, scaled_built):
        """Buggy-arm proof: evaluating an unstripped tree allocates the
        full skeleton, so the peak instrument used below can see the leak."""
        skeleton = self._skeleton_bytes()
        assert skeleton > 32 << 20
        model = DSparkDraftModel(tiny_dspark_args())
        mx.synchronize()
        mx.reset_peak_memory()
        base = mx.get_active_memory()
        mx.eval(model.parameters())
        mx.synchronize()
        assert mx.get_peak_memory() - base > skeleton // 2

    def test_load_window_eval_cannot_materialize_skeleton(
        self, scaled_built, monkeypatch,
    ):
        """Evaluating the tree at the load_weights seam stays cheap.

        The probe subclass evaluates the parameter tree after quantization
        and before the strict load, the widest point of the hazard window.
        Without the strip that eval materializes the random-init skeleton
        and its quantization graphs (measured above the full skeleton
        bytes); with the strip the tree holds only placeholders.
        """
        skeleton = self._skeleton_bytes()
        window: dict[str, int] = {}

        class WindowProbe(DSparkDraftModel):
            def load_weights(self, weights, strict=True):
                mx.synchronize()
                mx.reset_peak_memory()
                base = mx.get_active_memory()
                mx.eval(self.parameters())
                mx.synchronize()
                window["delta"] = mx.get_peak_memory() - base
                return super().load_weights(weights, strict=strict)

        monkeypatch.setattr(dspark_load_module, "DSparkDraftModel", WindowProbe)
        target = make_target()
        load_dspark_sidecar(scaled_built, target.model.embed, target.lm_head)
        assert window["delta"] < skeleton // 4

    def test_load_path_peak_stays_at_loaded_bytes(self, scaled_built):
        """Regression pin: the full load never allocates skeleton-scale
        memory. The strict load already replaced every leaf before the
        hydration eval, so this held before the strip too; the pin keeps
        the property under load-path reorderings.
        """
        skeleton = self._skeleton_bytes()
        target = make_target()
        mx.eval(target.parameters())
        mx.synchronize()
        mx.reset_peak_memory()
        base = mx.get_active_memory()
        draft, _ = load_dspark_sidecar(
            scaled_built, target.model.embed, target.lm_head)
        mx.synchronize()
        assert mx.get_peak_memory() - base < skeleton // 2
        loaded = sum(a.nbytes for _, a in tree_flatten(draft.parameters()))
        assert loaded < skeleton // 4


class TestStageCountResolution:
    """The builder must use a declared stage count, never an inference.

    The stage count and `len(dspark_target_layer_ids)` are unrelated
    quantities (drafter depth versus tapped main-stack layers); a source
    that declares the count nowhere is refused rather than guessed at.
    """

    def test_declared_in_top_level_config(self, built):
        _, out, _ = built
        manifest = json.loads((out / SIDECAR_MANIFEST_NAME).read_text())
        assert manifest["dspark"]["n_mtp_layers"] == N_MTP
        assert manifest["provenance"]["n_mtp_layers"] == N_MTP
        assert manifest["provenance"]["n_mtp_layers_source"] == "config.json"

    def test_declared_only_in_inference_config(self, tmp_path):
        raw = raw_source_weights(seed=21)
        src = write_source(tmp_path, raw)
        config = json.loads((src / "config.json").read_text())
        del config["n_mtp_layers"]
        (src / "config.json").write_text(json.dumps(config))
        (src / "inference").mkdir()
        (src / "inference" / "config.json").write_text(
            json.dumps({"n_mtp_layers": N_MTP}))
        out = tmp_path / "sidecar"
        build_dspark_sidecar(src, out)
        manifest = json.loads((out / SIDECAR_MANIFEST_NAME).read_text())
        assert manifest["dspark"]["n_mtp_layers"] == N_MTP
        assert manifest["provenance"]["n_mtp_layers"] == N_MTP
        assert (
            manifest["provenance"]["n_mtp_layers_source"]
            == "inference/config.json"
        )

    def test_declared_nowhere_refuses(self, tmp_path):
        raw = raw_source_weights(seed=22)
        src = write_source(tmp_path, raw)
        config = json.loads((src / "config.json").read_text())
        del config["n_mtp_layers"]
        (src / "config.json").write_text(json.dumps(config))
        with pytest.raises(
            DSparkSidecarError, match=r"config\.json.*inference/config\.json"
        ):
            build_dspark_sidecar(src, tmp_path / "sidecar")

    def test_declared_count_wins_over_target_id_count(self, tmp_path, monkeypatch):
        """A one-stage drafter tapping two target layers builds as one
        stage; the tap count never sets the depth."""
        mod = sys.modules[__name__]
        monkeypatch.setattr(mod, "N_MTP", 1)
        raw = raw_source_weights(seed=23)
        src = write_source(tmp_path, raw)
        config = json.loads((src / "config.json").read_text())
        assert config["n_mtp_layers"] == 1
        assert len(config["dspark_target_layer_ids"]) == 2
        out = tmp_path / "sidecar"
        build_dspark_sidecar(src, out)
        manifest = json.loads((out / SIDECAR_MANIFEST_NAME).read_text())
        assert manifest["dspark"]["n_mtp_layers"] == 1
        assert len(manifest["dspark"]["target_layer_ids"]) == 2
        assert manifest["provenance"]["n_mtp_layers_source"] == "config.json"
