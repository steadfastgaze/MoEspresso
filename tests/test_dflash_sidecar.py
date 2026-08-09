"""Synthetic-source tests for the DFlash sidecar builder and loader.

Builds a fake speculator checkpoint slice (final module-path names, one
safetensors shard, the speculators config.json) at tiny dimensions, then
exercises the full build/load round trip: quantized module formats,
passthrough dtypes, the raw d2t/t2d tables, the verifier-embedding sample
rows and their load-time logged check, manifest truthfulness, hash tamper
detection, and fail-closed behavior on missing, unexpected, or malformed
source tensors.
"""

import json
import shutil
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map

from jang_tools.dsv4.mlx_model import Model, ModelArgs

from moespresso.package.deepseek_v4.dflash_sidecar import (
    EMBED_SAMPLE_ROWS,
    EMBED_SAMPLE_TOKEN_IDS,
    SIDECAR_KIND,
    SIDECAR_MANIFEST_NAME,
    DFlashSidecarError,
    build_dflash_sidecar,
)
from moespresso.runtime.deepseek_v4 import dflash_load as dflash_load_module
from moespresso.runtime.deepseek_v4.dflash_load import load_dflash_sidecar
from moespresso.runtime.deepseek_v4.dflash_model import DFlashArgs, DFlashDraftModel

VOCAB = 97
DRAFT_VOCAB = 64
HIDDEN = 64
HC = 2
WINDOW = 8
FC_IN = 2 * HC * HIDDEN


def tiny_speculator_config() -> dict:
    return {
        "speculators_model_type": "dflash",
        "aux_hidden_state_layer_ids": [1, 2],
        "block_size": 4,
        "mask_token_id": 1,
        "draft_vocab_size": DRAFT_VOCAB,
        "sliding_window_non_causal": False,
        "speculators_config": {
            "algorithm": "dflash",
            "default_proposal_method": "greedy",
            "proposal_methods": [
                {
                    "proposal_type": "greedy",
                    "speculative_tokens": 3,
                    "verifier_accept_k": 1,
                }
            ],
        },
        "transformer_layer_config": {
            "hc_mult": HC,
            "head_dim": 16,
            "hidden_size": HIDDEN,
            "intermediate_size": 32,
            "layer_types": ["sliding_attention", "sliding_attention"],
            "model_type": "llama",
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
            "sliding_window": WINDOW,
            "vocab_size": VOCAB,
        },
    }


def tiny_target_model_args() -> ModelArgs:
    return ModelArgs(
        model_type="deepseek_v4",
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
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
        hc_mult=HC,
        hc_sinkhorn_iters=5,
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=4096,
        sliding_window=8,
        rms_norm_eps=1e-6,
        compress_ratios=[0, 0],
    )


def randomize_floats(module: nn.Module, scale: float = 0.08, seed: int = 0) -> None:
    def _rand(a):
        nonlocal seed
        if a.dtype not in (mx.float32, mx.float16, mx.bfloat16):
            return a
        seed += 1
        return mx.random.normal(a.shape, key=mx.random.key(seed)) * scale

    module.update(tree_map(_rand, module.parameters()))


def raw_source_weights(seed: int = 0) -> dict:
    """Final-named source tensors for a random tiny DFlash drafter.

    The checkpoint stores final module-path names, so the raw names are
    the parameter tree plus the verifier embedding copy. Floats are bf16;
    d2t is int64 and t2d bool, matching the reference layout.
    """
    from moespresso.runtime.deepseek_v4.dflash_model import DFlashDraftModel

    args = DFlashArgs.from_config(tiny_speculator_config())
    model = DFlashDraftModel(args)
    randomize_floats(model, seed=seed)
    mapped = np.arange(DRAFT_VOCAB) + np.arange(DRAFT_VOCAB) // 2
    t2d = np.zeros((VOCAB,), dtype=np.bool_)
    t2d[mapped] = True

    raw: dict[str, mx.array] = {}
    for name, arr in tree_flatten(model.parameters()):
        if name == "d2t":
            raw[name] = mx.arange(DRAFT_VOCAB, dtype=mx.int64) // 2
        elif name == "t2d":
            raw[name] = mx.array(t2d)
        else:
            raw[name] = arr.astype(mx.bfloat16)
    raw["embed_tokens.weight"] = (
        mx.random.normal((VOCAB, HIDDEN), key=mx.random.key(seed + 999)) * 0.05
    ).astype(mx.bfloat16)
    return raw


def write_source(tmp_path, raw: dict, config: dict | None = None):
    src = tmp_path / "src"
    src.mkdir()
    mx.save_safetensors(str(src / "model.safetensors"), dict(raw))
    (src / "config.json").write_text(
        json.dumps(config if config is not None else tiny_speculator_config()))
    return src


def make_target(seed: int = 5) -> Model:
    target = Model(tiny_target_model_args())
    randomize_floats(target, seed=seed)
    mx.eval(target.parameters())
    return target


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One shared build: (source dir, sidecar dir, raw tensors)."""
    tmp = tmp_path_factory.mktemp("dflash_sidecar")
    raw = raw_source_weights(seed=0)
    src = write_source(tmp, raw)
    out = tmp / "sidecar"
    build_dflash_sidecar(src, out)
    return src, out, raw


class TestBuildAndLoad:
    def test_round_trip_draft_runs(self, built, capsys):
        _, out, _ = built
        target = make_target()
        draft, args = load_dflash_sidecar(out, target.model.embed)
        assert args.block_size == 4
        assert draft.block_size == 3
        assert draft.greedy_only is True
        assert draft.tap_layer_ids == (0, 1)
        assert "embed sample delta" in capsys.readouterr().out

        # Projections, the pruned head, and the fc feature projection are
        # affine 8-bit, group size 32, under the default fc format.
        q_proj = draft.layers[0].self_attn.q_proj
        assert isinstance(q_proj, nn.QuantizedLinear)
        assert q_proj.bits == 8 and q_proj.group_size == 32
        assert isinstance(draft.layers[1].mlp.down_proj, nn.QuantizedLinear)
        assert isinstance(draft.lm_head, nn.QuantizedLinear)
        assert isinstance(draft.fc, nn.QuantizedLinear)
        assert draft.fc.bits == 8 and draft.fc.group_size == 32
        # The norms stay unquantized.
        assert draft.hidden_norm.weight.dtype == mx.bfloat16
        assert draft.layers[0].self_attn.k_norm.weight.dtype == mx.bfloat16
        assert draft.d2t.dtype == mx.int64
        assert draft.t2d.dtype == mx.bool_

        state = draft.make_state()
        n = 4
        rows = mx.random.normal((1, n, FC_IN))
        draft.ingest(state, rows, list(range(n)))
        result = draft.draft(state, anchor_token=1, anchor_pos=n)
        mx.eval(result.tokens, result.logits)
        assert result.tokens.shape == (1, 3)
        assert result.logits.shape == (1, 3, DRAFT_VOCAB)
        assert result.confidence is None
        assert int(result.tokens.max()) < VOCAB

    def test_manifest_shape(self, built):
        _, out, _ = built
        manifest = json.loads((out / SIDECAR_MANIFEST_NAME).read_text())
        assert manifest["artifact_kind"] == SIDECAR_KIND
        assert manifest["schema_version"]["major"] == 1
        assert manifest["artifact_id"].startswith("art:")
        assert manifest["dflash"] == {
            "block_size": 4,
            "speculative_tokens": 3,
            "mask_token_id": 1,
            "draft_vocab_size": DRAFT_VOCAB,
            "target_vocab_size": VOCAB,
            "sliding_window": WINDOW,
            "aux_hidden_state_layer_ids": [1, 2],
            "tap_layer_ids": [0, 1],
        }
        hashes = manifest["provenance"]["file_sha256"]
        assert len(hashes) == 1
        rows = manifest["tensors"]
        assert rows["lm_head.weight"]["format"] == "affine8"
        assert rows["layers.0.self_attn.q_proj.weight"]["format"] == "affine8"
        assert rows["layers.1.mlp.gate_proj.weight"]["format"] == "affine8"
        fc_row = rows["fc.weight"]
        assert fc_row["format"] == "affine8"
        assert fc_row["quant"] == {"group_size": 32, "bits": 8, "mode": "affine"}
        assert "fc.scales" in rows and "fc.biases" in rows
        assert rows["hidden_norm.weight"]["format"] == "passthrough"
        assert rows["layers.0.self_attn.k_norm.weight"]["format"] == "passthrough"
        assert rows["d2t"]["dtype"] == "int64"
        assert rows["t2d"]["dtype"] == "bool"
        assert "note" in rows[EMBED_SAMPLE_ROWS]
        # The verifier embedding itself is dropped.
        assert "embed_tokens.weight" not in rows
        for name, row in rows.items():
            assert row["file"] in hashes, name

    def test_embed_sample_rows_match_source(self, built):
        _, out, raw = built
        manifest = json.loads((out / SIDECAR_MANIFEST_NAME).read_text())
        shard = mx.load(str(out / manifest["tensors"][EMBED_SAMPLE_ROWS]["file"]))
        ids = [int(t) for t in shard[EMBED_SAMPLE_TOKEN_IDS]]
        assert ids == [0, 1, VOCAB // 2, VOCAB - 1]
        expected = np.array(
            raw["embed_tokens.weight"].astype(mx.float32))[np.array(ids)]
        got = np.array(shard[EMBED_SAMPLE_ROWS].astype(mx.float32))
        assert np.array_equal(got, expected)

    def test_affine8_dequantizes_close_to_source(self, built):
        _, out, raw = built
        manifest = json.loads((out / SIDECAR_MANIFEST_NAME).read_text())
        row = manifest["tensors"]["lm_head.weight"]
        shard = mx.load(str(out / row["file"]))
        deq = mx.dequantize(
            shard["lm_head.weight"],
            shard["lm_head.scales"],
            shard["lm_head.biases"],
            group_size=32, bits=8, mode="affine",
        )
        src = np.array(raw["lm_head.weight"].astype(mx.float32))
        err = np.abs(np.array(deq.astype(mx.float32)) - src)
        assert err.max() < 0.01

    def test_tables_survive_bitwise(self, built):
        _, out, raw = built
        shard = mx.load(str(out / "model-dflash-00001-of-00001.safetensors"))
        assert mx.array_equal(shard["d2t"], raw["d2t"])
        assert mx.array_equal(shard["t2d"], raw["t2d"])


class TestFcFormatVariant:
    def test_bf16_fc_round_trip(self, tmp_path):
        raw = raw_source_weights(seed=8)
        src = write_source(tmp_path, raw)
        out = tmp_path / "sidecar_fc_bf16"
        payload = build_dflash_sidecar(src, out, fc_format="bf16")
        assert payload["build_options"] == {"fc_format": "bf16"}

        manifest = json.loads((out / SIDECAR_MANIFEST_NAME).read_text())
        assert manifest["build_options"] == {"fc_format": "bf16"}
        rows = manifest["tensors"]
        fc_row = rows["fc.weight"]
        assert fc_row["format"] == "passthrough"
        assert fc_row["dtype"] == "bfloat16"
        assert "fc.scales" not in rows and "fc.biases" not in rows
        # The other projections stay affine 8-bit regardless of fc format.
        assert rows["lm_head.weight"]["format"] == "affine8"
        assert rows["hidden_norm.weight"]["format"] == "passthrough"

        target = make_target()
        draft, _ = load_dflash_sidecar(out, target.model.embed)
        assert isinstance(draft.fc, nn.Linear)
        assert not isinstance(draft.fc, nn.QuantizedLinear)
        assert draft.fc.weight.dtype == mx.bfloat16
        # The passthrough fc survives bitwise.
        assert mx.array_equal(draft.fc.weight, raw["fc.weight"])

        state = draft.make_state()
        n = 4
        rows_in = mx.random.normal((1, n, FC_IN))
        draft.ingest(state, rows_in, list(range(n)))
        result = draft.draft(state, anchor_token=1, anchor_pos=n)
        mx.eval(result.tokens, result.logits)
        assert result.tokens.shape == (1, 3)
        assert int(result.tokens.max()) < VOCAB

    def test_default_fc_format_recorded(self, built):
        _, out, _ = built
        manifest = json.loads((out / SIDECAR_MANIFEST_NAME).read_text())
        assert manifest["build_options"] == {"fc_format": "affine8"}

    def test_default_fc_dequantizes_close_to_source(self, built):
        _, out, raw = built
        shard = mx.load(str(out / "model-dflash-00001-of-00001.safetensors"))
        deq = mx.dequantize(
            shard["fc.weight"], shard["fc.scales"], shard["fc.biases"],
            group_size=32, bits=8, mode="affine",
        )
        src = np.array(raw["fc.weight"].astype(mx.float32))
        err = np.abs(np.array(deq.astype(mx.float32)) - src)
        assert err.max() < 0.01

    def test_unknown_fc_format_rejected(self, tmp_path):
        raw = raw_source_weights(seed=9)
        src = write_source(tmp_path, raw)
        with pytest.raises(DFlashSidecarError, match="fc format"):
            build_dflash_sidecar(src, tmp_path / "sidecar", fc_format="int4")


class TestFailClosed:
    def test_missing_source_tensor_listed(self, tmp_path):
        raw = raw_source_weights(seed=1)
        del raw["layers.1.self_attn.k_proj.weight"]
        src = write_source(tmp_path, raw)
        with pytest.raises(DFlashSidecarError, match="layers.1.self_attn.k_proj"):
            build_dflash_sidecar(src, tmp_path / "sidecar")

    def test_unexpected_source_tensor_rejected(self, tmp_path):
        raw = raw_source_weights(seed=2)
        raw["layers.0.bogus.weight"] = mx.zeros((4, 4), dtype=mx.bfloat16)
        src = write_source(tmp_path, raw)
        with pytest.raises(DFlashSidecarError, match="no conversion rule"):
            build_dflash_sidecar(src, tmp_path / "sidecar")

    def test_out_of_range_layer_rejected(self, tmp_path):
        raw = raw_source_weights(seed=3)
        raw["layers.2.self_attn.q_proj.weight"] = mx.zeros(
            (64, HIDDEN), dtype=mx.bfloat16)
        src = write_source(tmp_path, raw)
        with pytest.raises(DFlashSidecarError, match="layer index"):
            build_dflash_sidecar(src, tmp_path / "sidecar")

    def test_missing_verifier_embedding_rejected(self, tmp_path):
        raw = raw_source_weights(seed=4)
        del raw["embed_tokens.weight"]
        src = write_source(tmp_path, raw)
        with pytest.raises(DFlashSidecarError, match="verifier embedding"):
            build_dflash_sidecar(src, tmp_path / "sidecar")

    def test_non_causal_config_rejected(self, tmp_path):
        raw = raw_source_weights(seed=5)
        config = tiny_speculator_config()
        config["sliding_window_non_causal"] = True
        src = write_source(tmp_path, raw, config=config)
        with pytest.raises(DFlashSidecarError, match="causal"):
            build_dflash_sidecar(src, tmp_path / "sidecar")

    def test_d2t_outside_target_vocab_rejected(self, tmp_path):
        raw = raw_source_weights(seed=6)
        raw["d2t"] = mx.full((DRAFT_VOCAB,), VOCAB, dtype=mx.int64)
        src = write_source(tmp_path, raw)
        with pytest.raises(DFlashSidecarError, match="target vocabulary"):
            build_dflash_sidecar(src, tmp_path / "sidecar")

    def test_output_dir_not_empty_rejected(self, tmp_path):
        raw = raw_source_weights(seed=7)
        src = write_source(tmp_path, raw)
        out = tmp_path / "sidecar"
        out.mkdir()
        (out / "leftover").write_text("x")
        with pytest.raises(DFlashSidecarError, match="not empty"):
            build_dflash_sidecar(src, out)

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
        with pytest.raises(DFlashSidecarError, match="hash mismatch"):
            load_dflash_sidecar(copy, target.model.embed)

    def test_tampered_manifest_rejected(self, built, tmp_path):
        _, out, _ = built
        copy = tmp_path / "sidecar_copy"
        shutil.copytree(out, copy)
        manifest_path = copy / SIDECAR_MANIFEST_NAME
        payload = json.loads(manifest_path.read_text())
        payload["dflash"]["block_size"] = 99
        manifest_path.write_text(json.dumps(payload))
        target = make_target()
        with pytest.raises(DFlashSidecarError, match="hash mismatch"):
            load_dflash_sidecar(copy, target.model.embed)


# Dimension overrides that make the constructed float32 skeleton dwarf the
# loaded sidecar bytes; at the default tiny dims the whole tree is a few
# hundred KiB and a leak would vanish into allocator noise.
_SCALED_HIDDEN = 512
_SCALED_VOCAB = 16384
_SCALED_DRAFT_VOCAB = 8192
_SCALED_INTERMEDIATE = 2048


@pytest.fixture(scope="class")
def scaled_built(tmp_path_factory):
    """A sidecar built at the scaled dims, with the dims left patched in.

    The patch stays active while the class runs so the speculator config
    and `make_target` agree with the built sidecar.
    """
    mp = pytest.MonkeyPatch()
    mod = sys.modules[__name__]
    base_config = tiny_speculator_config

    def scaled_config():
        config = base_config()
        config["transformer_layer_config"]["intermediate_size"] = (
            _SCALED_INTERMEDIATE
        )
        return config

    mp.setattr(mod, "HIDDEN", _SCALED_HIDDEN)
    mp.setattr(mod, "VOCAB", _SCALED_VOCAB)
    mp.setattr(mod, "DRAFT_VOCAB", _SCALED_DRAFT_VOCAB)
    mp.setattr(mod, "tiny_speculator_config", scaled_config)
    try:
        tmp = tmp_path_factory.mktemp("dflash_sidecar_scaled")
        raw = raw_source_weights(seed=11)
        src = write_source(tmp, raw)
        out = tmp / "sidecar"
        build_dflash_sidecar(src, out)
        yield out
    finally:
        mp.undo()


class TestSkeletonStrip:
    """The load path must never be able to materialize the skeleton.

    Constructing the draft model fills the tree with lazy random-init
    float32, and `nn.quantize` chains quantization graphs onto those
    arrays. The strict load replaces every leaf before the hydration eval,
    so the success path never evaluated the skeleton; the hazard is the
    window between construction and the strict load, where any evaluation
    reaching the tree allocates the whole skeleton at once. The loader
    closes the window by swapping every constructed leaf for a zero-stride
    placeholder before reading shards.
    """

    def _skeleton_bytes(self) -> int:
        args = DFlashArgs.from_config(tiny_speculator_config())
        model = DFlashDraftModel(args)
        return sum(a.nbytes for _, a in tree_flatten(model.parameters()))

    def test_unstripped_skeleton_eval_allocates(self, scaled_built):
        """Buggy-arm proof: evaluating an unstripped tree allocates the
        full skeleton, so the peak instrument used below can see the leak."""
        skeleton = self._skeleton_bytes()
        assert skeleton > 32 << 20
        args = DFlashArgs.from_config(tiny_speculator_config())
        model = DFlashDraftModel(args)
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
        and its quantization graphs; with the strip the tree holds only
        placeholders.
        """
        skeleton = self._skeleton_bytes()
        window: dict[str, int] = {}

        class WindowProbe(DFlashDraftModel):
            def load_weights(self, weights, strict=True):
                mx.synchronize()
                mx.reset_peak_memory()
                base = mx.get_active_memory()
                mx.eval(self.parameters())
                mx.synchronize()
                window["delta"] = mx.get_peak_memory() - base
                return super().load_weights(weights, strict=strict)

        monkeypatch.setattr(dflash_load_module, "DFlashDraftModel", WindowProbe)
        target = make_target()
        load_dflash_sidecar(scaled_built, target.model.embed)
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
        draft, _ = load_dflash_sidecar(scaled_built, target.model.embed)
        mx.synchronize()
        assert mx.get_peak_memory() - base < skeleton // 2
        loaded = sum(a.nbytes for _, a in tree_flatten(draft.parameters()))
        assert loaded < skeleton // 2
