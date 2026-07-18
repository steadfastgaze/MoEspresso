from __future__ import annotations

import json
import os
import struct

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("jang_tools.turboquant")

from moespresso.package.kquant_backend import KQuantEncodedWeight  # noqa: E402
from moespresso.package.kquant_backend import KQuantBackendError  # noqa: E402
from moespresso.package.kquant_format import KQUANT_GEOMETRY  # noqa: E402
from moespresso.package.kquant_recipe import KQuantRecipeError  # noqa: E402
from moespresso.package.bundle import component_array  # noqa: E402
from moespresso.inventory.safetensors_header import read_header  # noqa: E402
from moespresso.probe.gguf_parse import GGUF_MAGIC  # noqa: E402
from moespresso.package.deepseek_v4.kquant_package import (  # noqa: E402
    KQUANT_RECIPE_REPORT_NAME,
    PACKAGE_PLAN_NAME,
    build_ds4_kquant_package,
    main,
    preflight_ds4_kquant_package,
)
from moespresso.package.constants import MANIFEST_NAME  # noqa: E402
from moespresso.package.convert import INVENTORY_NAME  # noqa: E402
from moespresso.runtime.expert_index import build_expert_index  # noqa: E402
from moespresso.runtime.verify import verify_package  # noqa: E402


DS4_ARCH = {
    "model_type": "deepseek_v4",
    "hidden_size": 256,
    "num_hidden_layers": 1,
    "n_routed_experts": 2,
    "num_experts_per_tok": 1,
    "num_nextn_predict_layers": 1,
    "head_dim": 64,
    "qk_rope_head_dim": 32,
    "sliding_window": 128,
    "index_topk": 8,
    "compress_rope_theta": 160000,
    "compress_ratios": [1],
    "vocab_size": 512,
}


def _gguf_string(text: str) -> bytes:
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _write_recipe_gguf(path, tensors):
    from moespresso.probe.gguf_parse import GGUF_MAGIC

    header = struct.pack("<IIQQ", GGUF_MAGIC, 3, len(tensors), 0)
    infos = bytearray()
    for offset, (name, type_id, dims) in enumerate(tensors):
        infos += _gguf_string(name)
        infos += struct.pack("<I", len(dims))
        for dim in dims:
            infos += struct.pack("<Q", dim)
        infos += struct.pack("<I", type_id)
        infos += struct.pack("<Q", offset)
    path.write_bytes(header + bytes(infos))


def _write_payload_recipe_gguf(path):
    tensors = [
        (
            "blk.0.ffn_gate_exps.weight",
            16,
            [256, 1, 2],
            np.concatenate([
                np.full((1, 66), 21, dtype=np.uint8).reshape(-1),
                np.full((1, 66), 31, dtype=np.uint8).reshape(-1),
            ]),
        ),
        (
            "blk.0.ffn_up_exps.weight",
            16,
            [256, 1, 2],
            np.concatenate([
                np.full((1, 66), 22, dtype=np.uint8).reshape(-1),
                np.full((1, 66), 32, dtype=np.uint8).reshape(-1),
            ]),
        ),
        (
            "blk.0.ffn_down_exps.weight",
            10,
            [256, 2, 2],
            np.concatenate([
                np.full((2, 84), 23, dtype=np.uint8).reshape(-1),
                np.full((2, 84), 33, dtype=np.uint8).reshape(-1),
            ]),
        ),
        ("blk.0.attn_q_a.weight", 8, [256, 128], np.array([], dtype=np.uint8)),
    ]
    header = struct.pack("<IIQQ", GGUF_MAGIC, 3, len(tensors), 0)
    infos = bytearray()
    offset = 0
    payload = bytearray()
    for name, type_id, dims, data in tensors:
        infos += _gguf_string(name)
        infos += struct.pack("<I", len(dims))
        for dim in dims:
            infos += struct.pack("<Q", dim)
        infos += struct.pack("<I", type_id)
        infos += struct.pack("<Q", offset)
        raw = np.asarray(data, dtype=np.uint8).tobytes()
        payload += raw
        offset += len(raw)
    metadata = header + bytes(infos)
    pad = b"\0" * ((((len(metadata) + 31) // 32) * 32) - len(metadata))
    path.write_bytes(metadata + pad + bytes(payload))


def _write_safetensors(path, tensors):
    header, blob, off = {}, bytearray(), 0
    for name, (dtype, arr) in tensors.items():
        a = np.ascontiguousarray(arr)
        data = a.tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(a.shape),
            "data_offsets": [off, off + len(data)],
        }
        blob += data
        off += len(data)
    hjson = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)


def _write_legacy_imatrix(path, entries):
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(entries)))
        for name, ncall, values in entries:
            data = np.asarray(values, np.float32)
            encoded = name.encode()
            f.write(struct.pack("<i", len(encoded)))
            f.write(encoded)
            f.write(struct.pack("<i", ncall))
            f.write(struct.pack("<i", data.size))
            f.write(data.tobytes())


def _ds4_kquant_source(path):
    path.mkdir()
    tensors = {
        "layers.0.attn.wq_a.weight": (
            "F8_E4M3",
            np.full((128, 256), 0x38, dtype=np.uint8),
        ),
        "layers.0.attn.wq_a.scale": (
            "F8_E8M0",
            np.array([[127, 127]], dtype=np.uint8),
        ),
        "layers.0.ffn.gate.weight": (
            "F32",
            np.arange(512, dtype=np.float32).reshape(2, 256),
        ),
        "layers.0.attn.attn_sink": ("F32", np.arange(64, dtype=np.float32)),
    }
    packed = np.arange(128, dtype=np.uint8).reshape(1, 128).view(np.int8)
    packed_down = np.arange(256, dtype=np.uint8).reshape(2, 128).view(np.int8)
    for expert in (0, 1):
        tensors[f"layers.0.ffn.experts.{expert}.w1.weight"] = ("I8", packed)
        tensors[f"layers.0.ffn.experts.{expert}.w3.weight"] = ("I8", packed)
        tensors[f"layers.0.ffn.experts.{expert}.w2.weight"] = ("I8", packed_down)
        tensors[f"layers.0.ffn.experts.{expert}.w1.scale"] = (
            "F8_E8M0",
            np.full((1, 8), 127, dtype=np.uint8),
        )
        tensors[f"layers.0.ffn.experts.{expert}.w3.scale"] = (
            "F8_E8M0",
            np.full((1, 8), 127, dtype=np.uint8),
        )
        tensors[f"layers.0.ffn.experts.{expert}.w2.scale"] = (
            "F8_E8M0",
            np.full((2, 8), 127, dtype=np.uint8),
        )
    _write_safetensors(path / "model-00001.safetensors", tensors)
    (path / "config.json").write_text(json.dumps(DS4_ARCH), encoding="utf-8")


def _recipe(path, *, include_up=True):
    tensors = [
        ("blk.0.ffn_gate_exps.weight", 16, [256, 1]),
        ("blk.0.ffn_down_exps.weight", 10, [256, 2]),
        ("blk.0.attn_q_a.weight", 8, [256, 128]),
    ]
    if include_up:
        tensors.insert(1, ("blk.0.ffn_up_exps.weight", 16, [256, 1]))
    _write_recipe_gguf(path, tensors)


def _imatrix(path):
    _write_legacy_imatrix(path, [
        ("blk.0.ffn_gate_exps.weight", 1, np.ones(256, dtype=np.float32)),
        ("blk.0.ffn_up_exps.weight", 1, np.ones(256, dtype=np.float32)),
        ("blk.0.ffn_down_exps.weight", 1, np.ones(256, dtype=np.float32)),
    ])


def test_build_ds4_kquant_package_writes_recipe_driven_package(tmp_path):
    src = tmp_path / "src"
    _ds4_kquant_source(src)
    recipe = tmp_path / "recipe.gguf"
    imatrix = tmp_path / "imatrix.dat"
    out = tmp_path / "pkg"
    _recipe(recipe)
    _imatrix(imatrix)
    calls = []

    def fake_encoder(weight, target, imatrix_vectors):
        calls.append((weight.shape, target.codec, target.imatrix_key, sorted(imatrix_vectors)))
        geometry = KQUANT_GEOMETRY[target.codec]
        blocks = (
            weight.shape[1] + geometry.weights_per_block - 1
        ) // geometry.weights_per_block
        return KQuantEncodedWeight(
            codec=target.codec,
            weight=np.full(
                (weight.shape[0], blocks * geometry.bytes_per_block),
                len(calls),
                dtype=np.uint8,
            ),
            scales=np.zeros((1,), dtype=np.uint8),
        )

    manifest = build_ds4_kquant_package(
        src,
        out,
        gguf_recipe_path=recipe,
        imatrix_path=imatrix,
        shard_size_gb=0.0,
        kquant_encoder=fake_encoder,
    )

    assert manifest["status"] == "valid"
    assert [v for v in verify_package(manifest, out) if v.blocking] == []
    assert (out / INVENTORY_NAME).is_file()
    assert (out / PACKAGE_PLAN_NAME).is_file()
    assert (out / MANIFEST_NAME).is_file()
    assert (out / "config.json").is_file()
    assert (out / "jang_config.json").is_file()
    report = json.loads((out / KQUANT_RECIPE_REPORT_NAME).read_text())
    assert report["recipe"]["expert_targets"] == 3
    assert report["recipe"]["expert_codec_counts"] == {"iq2_xxs": 2, "q2_k": 1}
    assert report["recipe"]["dense"] == {
        "targets": 1,
        "codec_counts": {"q8_0": 1},
        "role_counts": {"attn.wq_a": 1},
    }
    assert report["package_size_bytes"] > 0
    assert len(calls) == 7
    assert {call[1] for call in calls} == {"iq2_xxs", "q2_k", "q8_0"}
    assert calls[0][:3] == (
        (128, 256),
        "q8_0",
        "blk.0.attn_q_a.weight",
    )

    formats = {(t["source_name"], t["format"]) for t in manifest["tensors"]}
    assert ("layers.0.ffn.experts.gate", "kquant") in formats
    assert ("layers.0.attn.wq_a.weight", "kquant") in formats
    assert ("layers.0.ffn.gate.weight", "fp16") in formats
    assert "kquant_dequant" in manifest["required_ops"]
    assert "fp16_passthrough" in manifest["required_ops"]
    assert "raw_dtype_passthrough" in manifest["required_ops"]
    gate_header = next(
        header["layers.0.ffn.gate.weight"]
        for shard in sorted(out.glob("model-*.safetensors"))
        for header in [read_header(shard)]
        if "layers.0.ffn.gate.weight" in header
    )
    assert gate_header["dtype"] == "F16"


def test_preflight_ds4_kquant_package_reports_real_fit_without_encoding(tmp_path):
    src = tmp_path / "src"
    _ds4_kquant_source(src)
    recipe = tmp_path / "recipe.gguf"
    imatrix = tmp_path / "imatrix.dat"
    _recipe(recipe)
    _imatrix(imatrix)

    report = preflight_ds4_kquant_package(
        src,
        gguf_recipe_path=recipe,
        imatrix_path=imatrix,
    )

    assert report["source"]["family"] == "deepseek_v4_flash"
    assert report["source"]["inventory_status"] == "valid"
    assert report["source"]["expert_layer_first"] == 0
    assert report["source"]["expert_layer_last"] == 0
    assert report["recipe"]["expert_targets"] == 3
    assert report["recipe"]["expert_codec_counts"] == {"iq2_xxs": 2, "q2_k": 1}
    assert report["recipe"]["mode"] == {
        "mode": "faithful_recipe",
        "faithful_ds4c_recipe": True,
    }
    assert report["recipe"]["dense"] == {
        "targets": 1,
        "codec_counts": {"q8_0": 1},
        "role_counts": {"attn.wq_a": 1},
    }
    assert report["fit"]["status"] == "valid"
    assert report["fit"]["imatrix_lengths"] == {"256": 3}
    assert report["fit"]["target_shapes"] == [
        {"projection": "down", "codec": "q2_k", "shape": [2, 256], "count": 1},
        {"projection": "gate", "codec": "iq2_xxs", "shape": [1, 256], "count": 1},
        {"projection": "up", "codec": "iq2_xxs", "shape": [1, 256], "count": 1},
    ]
    assert report["fit"]["dense"] == {
        "format_counts": {"kquant": 1},
        "kquant_targets": 1,
        "target_shapes": [
            {"role": "attn.wq_a", "codec": "q8_0", "shape": [128, 256], "count": 1},
        ],
        "imatrix_lengths": {},
    }
    assert report["manual_q1"] == {"status": "not_run"}


def test_fast_diagnostic_preflight_overrides_gate_up_to_q2k(tmp_path):
    src = tmp_path / "src"
    _ds4_kquant_source(src)
    recipe = tmp_path / "recipe.gguf"
    imatrix = tmp_path / "imatrix.dat"
    _recipe(recipe)
    _imatrix(imatrix)

    report = preflight_ds4_kquant_package(
        src,
        gguf_recipe_path=recipe,
        imatrix_path=imatrix,
        fast_diagnostic=True,
    )

    assert report["recipe"]["expert_codec_counts"] == {"q2_k": 3}
    assert report["fit"]["target_shapes"] == [
        {"projection": "down", "codec": "q2_k", "shape": [2, 256], "count": 1},
        {"projection": "gate", "codec": "q2_k", "shape": [1, 256], "count": 1},
        {"projection": "up", "codec": "q2_k", "shape": [1, 256], "count": 1},
    ]
    assert report["recipe"]["mode"]["mode"] == "fast_diagnostic"
    assert report["recipe"]["mode"]["faithful_ds4c_recipe"] is False


def test_fast_diagnostic_keep_iquants_keeps_iq_codecs(tmp_path):
    # The loud opt-out keeps iq* inside a fast-diagnostic build: gate/up stay
    # iq2_xxs (the slow CPU-only path), only down was already q2_k in the recipe.
    src = tmp_path / "src"
    _ds4_kquant_source(src)
    recipe = tmp_path / "recipe.gguf"
    imatrix = tmp_path / "imatrix.dat"
    _recipe(recipe)
    _imatrix(imatrix)

    report = preflight_ds4_kquant_package(
        src,
        gguf_recipe_path=recipe,
        imatrix_path=imatrix,
        fast_diagnostic=True,
        keep_iquants=True,
    )

    assert report["recipe"]["expert_codec_counts"] == {"iq2_xxs": 2, "q2_k": 1}
    # The report mode must NOT claim the iq*->q2_k override ran, because it did not.
    mode = report["recipe"]["mode"]
    assert mode["mode"] == "fast_diagnostic"
    assert mode["routed_iquant_override"]["applied"] is False
    assert mode["faithful_ds4c_recipe"] is True


def test_fast_diagnostic_mode_marks_override_applied_when_swapping(tmp_path):
    # The normal fast path (no keep_iquants) must report the override as applied.
    src = tmp_path / "src"
    _ds4_kquant_source(src)
    recipe = tmp_path / "recipe.gguf"
    imatrix = tmp_path / "imatrix.dat"
    _recipe(recipe)
    _imatrix(imatrix)

    report = preflight_ds4_kquant_package(
        src,
        gguf_recipe_path=recipe,
        imatrix_path=imatrix,
        fast_diagnostic=True,
    )

    assert report["recipe"]["expert_codec_counts"] == {"q2_k": 3}
    mode = report["recipe"]["mode"]
    assert mode["routed_iquant_override"]["applied"] is True
    assert mode["routed_iquant_override"]["to"] == "q2_k"
    assert mode["faithful_ds4c_recipe"] is False


def test_cli_force_iquant_requires_fast_diagnostic(tmp_path):
    # The scary flag is meaningless (and rejected) outside a fast-diagnostic build.
    src = tmp_path / "src"
    _ds4_kquant_source(src)
    recipe = tmp_path / "recipe.gguf"
    imatrix = tmp_path / "imatrix.dat"
    _recipe(recipe)
    _imatrix(imatrix)

    with pytest.raises(SystemExit):
        main([
            str(src), str(tmp_path / "out"),
            "--gguf-recipe", str(recipe),
            "--imatrix", str(imatrix),
            "--preflight-only",
            "--force-very-slow-cpu-iquant-encode",
        ])


def test_ds4_kquant_package_cli_preflight_writes_report_only(tmp_path, capsys):
    src = tmp_path / "src"
    _ds4_kquant_source(src)
    recipe = tmp_path / "recipe.gguf"
    imatrix = tmp_path / "imatrix.dat"
    out = tmp_path / "preflight"
    _recipe(recipe)
    _imatrix(imatrix)

    rc = main([
        str(src),
        str(out),
        "--gguf-recipe",
        str(recipe),
        "--imatrix",
        str(imatrix),
        "--preflight-only",
    ])

    assert rc == 0
    printed = capsys.readouterr().out
    assert "Preflight:" in printed
    assert "targets=3" in printed
    assert (out / KQUANT_RECIPE_REPORT_NAME).is_file()
    assert not (out / INVENTORY_NAME).exists()
    assert not (out / PACKAGE_PLAN_NAME).exists()
    assert not (out / MANIFEST_NAME).exists()
    report = json.loads((out / KQUANT_RECIPE_REPORT_NAME).read_text())
    assert report["fit"]["status"] == "valid"
    assert report["fit"]["dense"]["kquant_targets"] == 1
    assert report["manual_q1"] == {"status": "not_run"}


def test_vendored_ds4_hotlist_vector_is_a_full_ranked_payload():
    from moespresso.package.deepseek_v4.hotlist_vector import (
        load_vendored_expert_hotlist,
    )

    payload = load_vendored_expert_hotlist()
    assert payload["kind"] == "expert_hotlist"
    source = payload["source"]
    assert source["imatrix_name"] == "tarruda_nonproduct_deepseek_v4_imatrix.gguf"
    assert source["use"] == "cold-start expert prewarm ranking only"
    assert len(source["imatrix_sha256"]) == 64
    layers = payload["layers"]
    assert sorted(map(int, layers)) == list(range(43))
    for ranked in layers.values():
        assert len(ranked) == 256
        counts = list(ranked.values())
        assert counts == sorted(counts, reverse=True)
        assert counts[-1] > 0
        assert all(0 <= int(expert) < 256 for expert in ranked)


def test_ds4_build_falls_back_to_vendored_hotlist(tmp_path, monkeypatch, capsys):
    """The legacy .dat build imatrix carries no expert counts, so the builder
    falls back to the vendored ranking. On this one-layer smoke package the
    vendored 43-layer payload cannot align, so the unmocked build skips the
    artifact loudly; with the payload writer intercepted, the report records
    the vendored layers."""
    src = tmp_path / "src"
    _ds4_kquant_source(src)
    recipe = tmp_path / "recipe.gguf"
    imatrix = tmp_path / "imatrix.dat"
    _recipe(recipe)
    _imatrix(imatrix)

    def encoder(weight, target, imatrix_vectors):
        geometry = KQUANT_GEOMETRY[target.codec]
        blocks = (
            weight.shape[1] + geometry.weights_per_block - 1
        ) // geometry.weights_per_block
        return KQuantEncodedWeight(
            codec=target.codec,
            weight=np.zeros(
                (weight.shape[0], blocks * geometry.bytes_per_block),
                dtype=np.uint8,
            ),
            scales=np.zeros((1,), dtype=np.uint8),
        )

    out = tmp_path / "pkg"
    build_ds4_kquant_package(
        src,
        out,
        gguf_recipe_path=recipe,
        imatrix_path=imatrix,
        shard_size_gb=0.0,
        kquant_encoder=encoder,
    )
    assert "[hotlist] SKIPPED" in capsys.readouterr().out
    report = json.loads((out / KQUANT_RECIPE_REPORT_NAME).read_text())
    assert report["expert_hotlist_layers"] == 0
    assert not (out / "expert_hotlist.json").exists()

    import moespresso.package.hotlist as hl

    seen = {}

    def fake_payload_write(out_dir, payload):
        seen["imatrix_name"] = payload["source"]["imatrix_name"]
        return 43

    monkeypatch.setattr(
        hl, "write_package_expert_hotlist_from_payload", fake_payload_write)
    out2 = tmp_path / "pkg2"
    build_ds4_kquant_package(
        src,
        out2,
        gguf_recipe_path=recipe,
        imatrix_path=imatrix,
        shard_size_gb=0.0,
        kquant_encoder=encoder,
    )
    assert seen["imatrix_name"] == "tarruda_nonproduct_deepseek_v4_imatrix.gguf"
    report2 = json.loads((out2 / KQUANT_RECIPE_REPORT_NAME).read_text())
    assert report2["expert_hotlist_layers"] == 43


def test_fast_diagnostic_build_is_labeled_and_keeps_default_path_unchanged(tmp_path):
    src = tmp_path / "src"
    _ds4_kquant_source(src)
    recipe = tmp_path / "recipe.gguf"
    imatrix = tmp_path / "imatrix.dat"
    faithful_out = tmp_path / "faithful"
    fast_out = tmp_path / "fast"
    _recipe(recipe)
    _imatrix(imatrix)
    faithful_calls = []
    fast_calls = []

    def fake_encoder(calls):
        def encode(weight, target, imatrix_vectors):
            calls.append((target.projection if hasattr(target, "projection") else None,
                          target.codec))
            geometry = KQUANT_GEOMETRY[target.codec]
            blocks = (
                weight.shape[1] + geometry.weights_per_block - 1
            ) // geometry.weights_per_block
            return KQuantEncodedWeight(
                codec=target.codec,
                weight=np.full(
                    (weight.shape[0], blocks * geometry.bytes_per_block),
                    len(calls),
                    dtype=np.uint8,
                ),
                scales=np.zeros((1,), dtype=np.uint8),
            )
        return encode

    faithful_manifest = build_ds4_kquant_package(
        src,
        faithful_out,
        gguf_recipe_path=recipe,
        imatrix_path=imatrix,
        shard_size_gb=0.0,
        kquant_encoder=fake_encoder(faithful_calls),
    )
    fast_manifest = build_ds4_kquant_package(
        src,
        fast_out,
        gguf_recipe_path=recipe,
        imatrix_path=imatrix,
        shard_size_gb=0.0,
        kquant_encoder=fake_encoder(fast_calls),
        fast_diagnostic=True,
    )

    assert faithful_manifest["status"] == "valid"
    assert fast_manifest["status"] == "valid"
    assert [v for v in verify_package(fast_manifest, fast_out) if v.blocking] == []
    assert ("gate", "iq2_xxs") in faithful_calls
    assert ("up", "iq2_xxs") in faithful_calls
    assert ("gate", "q2_k") in fast_calls
    assert ("up", "q2_k") in fast_calls
    assert fast_manifest["provenance"]["diagnostic"]["mode"] == "fast_diagnostic"
    assert "diagnostic" not in faithful_manifest["provenance"]
    package_plan = json.loads((fast_out / PACKAGE_PLAN_NAME).read_text())
    assert package_plan["source_constraints"]["diagnostic"]["mode"] == "fast_diagnostic"
    report = json.loads((fast_out / KQUANT_RECIPE_REPORT_NAME).read_text())
    assert report["recipe"]["expert_codec_counts"] == {"q2_k": 3}
    assert report["recipe"]["mode"]["mode"] == "fast_diagnostic"
    idx = build_expert_index(fast_out)
    assert idx.geometry(layer=0, projection="gate_proj").kquant_codec == "q2_k"
    assert idx.geometry(layer=0, projection="up_proj").kquant_codec == "q2_k"


def test_ds4_kquant_force_dry_run_reports_matched_tensors(tmp_path):
    src = tmp_path / "src"
    _ds4_kquant_source(src)
    recipe = tmp_path / "recipe.gguf"
    imatrix = tmp_path / "imatrix.dat"
    out = tmp_path / "out"
    _recipe(recipe)
    _imatrix(imatrix)

    plan = build_ds4_kquant_package(
        src,
        out,
        gguf_recipe_path=recipe,
        imatrix_path=imatrix,
        force_format=["*ffn_gate_exps.weight=tq2"],
        force_format_dry_run=True,
    )

    gate = next(
        row for row in plan["allocation"]
        if row.get("gguf_tensor") == "blk.0.ffn_gate_exps.weight"
    )
    assert gate["format"] == "kquant"
    assert "forced_format" not in gate
    assert plan["force_override_preview"]["matched"][0]["before"] == "kquant:iq2_xxs"
    assert plan["force_override_preview"]["matched"][0]["after"] == "tq2"
    report = json.loads((out / KQUANT_RECIPE_REPORT_NAME).read_text())
    assert report["matched"][0]["gguf_tensor"] == "blk.0.ffn_gate_exps.weight"
    assert report["matched"][0]["before"] == "kquant:iq2_xxs"
    assert report["matched"][0]["after"] == "tq2"
    assert (out / PACKAGE_PLAN_NAME).is_file()
    assert not (out / MANIFEST_NAME).exists()


def test_kquant_cache_reuses_unchanged_package_encodes(tmp_path):
    src = tmp_path / "src"
    _ds4_kquant_source(src)
    recipe = tmp_path / "recipe.gguf"
    imatrix = tmp_path / "imatrix.dat"
    cache = tmp_path / "cache"
    first_out = tmp_path / "first"
    second_out = tmp_path / "second"
    _recipe(recipe)
    _imatrix(imatrix)
    first_calls = []
    second_calls = []

    def fake_encoder(calls):
        def encode(weight, target, imatrix_vectors):
            calls.append(target.codec)
            geometry = KQUANT_GEOMETRY[target.codec]
            blocks = (
                weight.shape[1] + geometry.weights_per_block - 1
            ) // geometry.weights_per_block
            return KQuantEncodedWeight(
                codec=target.codec,
                weight=np.full(
                    (weight.shape[0], blocks * geometry.bytes_per_block),
                    len(calls),
                    dtype=np.uint8,
                ),
                scales=np.zeros((1,), dtype=np.uint8),
            )
        return encode

    build_ds4_kquant_package(
        src,
        first_out,
        gguf_recipe_path=recipe,
        imatrix_path=imatrix,
        shard_size_gb=0.0,
        kquant_encoder=fake_encoder(first_calls),
        kquant_cache_dir=cache,
    )
    manifest = build_ds4_kquant_package(
        src,
        second_out,
        gguf_recipe_path=recipe,
        imatrix_path=imatrix,
        shard_size_gb=0.0,
        kquant_encoder=fake_encoder(second_calls),
        kquant_cache_dir=cache,
    )

    assert first_calls == ["q8_0", "iq2_xxs", "iq2_xxs", "q2_k"] * 1 + [
        "iq2_xxs", "iq2_xxs", "q2_k",
    ]
    assert second_calls == []
    assert [v for v in verify_package(manifest, second_out) if v.blocking] == []
    report = json.loads((second_out / KQUANT_RECIPE_REPORT_NAME).read_text())
    assert "path" not in report["kquant_cache"]
    assert str(cache) not in (second_out / KQUANT_RECIPE_REPORT_NAME).read_text()
    assert report["kquant_cache"]["hits"] == 7
    assert report["kquant_cache"]["writes"] == 0


def test_kquant_cache_reencodes_changed_codec_only(tmp_path):
    src = tmp_path / "src"
    _ds4_kquant_source(src)
    recipe = tmp_path / "recipe.gguf"
    changed_recipe = tmp_path / "recipe_gate_q2.gguf"
    imatrix = tmp_path / "imatrix.dat"
    cache = tmp_path / "cache"
    first_out = tmp_path / "first"
    second_out = tmp_path / "second"
    _recipe(recipe)
    _write_recipe_gguf(changed_recipe, [
        ("blk.0.ffn_gate_exps.weight", 10, [256, 1]),
        ("blk.0.ffn_up_exps.weight", 16, [256, 1]),
        ("blk.0.ffn_down_exps.weight", 10, [256, 2]),
        ("blk.0.attn_q_a.weight", 8, [256, 128]),
    ])
    _imatrix(imatrix)
    first_calls = []
    second_calls = []

    def fake_encoder(calls):
        def encode(weight, target, imatrix_vectors):
            calls.append((
                target.projection if hasattr(target, "projection") else "dense",
                target.codec,
            ))
            geometry = KQUANT_GEOMETRY[target.codec]
            blocks = (
                weight.shape[1] + geometry.weights_per_block - 1
            ) // geometry.weights_per_block
            return KQuantEncodedWeight(
                codec=target.codec,
                weight=np.full(
                    (weight.shape[0], blocks * geometry.bytes_per_block),
                    len(calls),
                    dtype=np.uint8,
                ),
                scales=np.zeros((1,), dtype=np.uint8),
            )
        return encode

    build_ds4_kquant_package(
        src,
        first_out,
        gguf_recipe_path=recipe,
        imatrix_path=imatrix,
        shard_size_gb=0.0,
        kquant_encoder=fake_encoder(first_calls),
        kquant_cache_dir=cache,
    )
    manifest = build_ds4_kquant_package(
        src,
        second_out,
        gguf_recipe_path=changed_recipe,
        imatrix_path=imatrix,
        shard_size_gb=0.0,
        kquant_encoder=fake_encoder(second_calls),
        kquant_cache_dir=cache,
    )

    assert first_calls
    assert second_calls == [("gate", "q2_k"), ("gate", "q2_k")]
    assert [v for v in verify_package(manifest, second_out) if v.blocking] == []
    report = json.loads((second_out / KQUANT_RECIPE_REPORT_NAME).read_text())
    assert report["kquant_cache"]["hits"] == 5
    assert report["kquant_cache"]["writes"] == 2


def test_build_ds4_kquant_package_can_copy_gguf_expert_bytes(tmp_path):
    src = tmp_path / "src"
    _ds4_kquant_source(src)
    recipe = tmp_path / "recipe.gguf"
    imatrix = tmp_path / "imatrix.dat"
    out = tmp_path / "pkg"
    _write_payload_recipe_gguf(recipe)
    _imatrix(imatrix)
    calls = []

    def fake_encoder(weight, target, imatrix_vectors):
        calls.append(target.codec)
        geometry = KQUANT_GEOMETRY[target.codec]
        blocks = (
            weight.shape[1] + geometry.weights_per_block - 1
        ) // geometry.weights_per_block
        return KQuantEncodedWeight(
            codec=target.codec,
            weight=np.full(
                (weight.shape[0], blocks * geometry.bytes_per_block),
                99,
                dtype=np.uint8,
            ),
            scales=np.zeros((1,), dtype=np.uint8),
        )

    manifest = build_ds4_kquant_package(
        src,
        out,
        gguf_recipe_path=recipe,
        imatrix_path=imatrix,
        shard_size_gb=0.0,
        kquant_encoder=fake_encoder,
        copy_gguf_expert_bytes=True,
    )

    assert manifest["status"] == "valid"
    assert calls == ["q8_0"]
    assert [v for v in verify_package(manifest, out) if v.blocking] == []
    report = json.loads((out / KQUANT_RECIPE_REPORT_NAME).read_text())
    assert report["recipe"]["expert_byte_source"]["mode"] == "gguf_bytes"
    from safetensors.numpy import load_file

    arrays = {}
    for file in manifest["files"]:
        arrays.update(load_file(str(out / file["path"])))
    bundle = arrays["layers.0.ffn.experts.tq_bundle"]
    idx = build_expert_index(out)
    gate = component_array(
        bundle,
        idx.row_components(layer=0)[("gate_proj", "weight")],
    )
    down = component_array(
        bundle,
        idx.row_components(layer=0)[("down_proj", "weight")],
    )
    np.testing.assert_array_equal(gate[1], np.full((1, 66), 31, dtype=np.uint8))
    np.testing.assert_array_equal(down[0], np.full((2, 84), 23, dtype=np.uint8))


def test_build_ds4_kquant_package_reuses_one_gguf_expert_reader(tmp_path, monkeypatch):
    from moespresso.package.deepseek_v4 import kquant_package

    src = tmp_path / "src"
    _ds4_kquant_source(src)
    recipe = tmp_path / "recipe.gguf"
    imatrix = tmp_path / "imatrix.dat"
    out = tmp_path / "pkg"
    _write_payload_recipe_gguf(recipe)
    _imatrix(imatrix)
    init_paths = []

    class FakeReader:
        def __init__(self, path):
            init_paths.append(path)

        def load_expert_weight(self, target, *, expert_index):
            rows = {"gate": 1, "up": 1, "down": 2}[target.projection]
            return KQuantEncodedWeight(
                codec=target.codec,
                weight=np.full(
                    (rows, KQUANT_GEOMETRY[target.codec].bytes_per_block),
                    7,
                    dtype=np.uint8,
                ),
                scales=np.zeros((1,), dtype=np.uint8),
            )

    monkeypatch.setattr(kquant_package, "GGUFKQuantExpertReader", FakeReader)

    build_ds4_kquant_package(
        src,
        out,
        gguf_recipe_path=recipe,
        imatrix_path=imatrix,
        shard_size_gb=0.0,
        kquant_encoder=lambda weight, target, _imatrix: KQuantEncodedWeight(
            codec=target.codec,
            weight=np.zeros(
                (
                    weight.shape[0],
                    KQUANT_GEOMETRY[target.codec].bytes_per_block,
                ),
                dtype=np.uint8,
            ),
            scales=np.zeros((1,), dtype=np.uint8),
        ),
        copy_gguf_expert_bytes=True,
    )

    assert init_paths == [recipe]


def test_build_ds4_kquant_package_rejects_gguf_byte_copy_after_codec_override(tmp_path):
    src = tmp_path / "src"
    _ds4_kquant_source(src)
    recipe = tmp_path / "recipe.gguf"
    imatrix = tmp_path / "imatrix.dat"
    out = tmp_path / "pkg"
    _write_payload_recipe_gguf(recipe)
    _imatrix(imatrix)

    with pytest.raises(KQuantRecipeError, match="requires the routed expert codecs"):
        build_ds4_kquant_package(
            src,
            out,
            gguf_recipe_path=recipe,
            imatrix_path=imatrix,
            kquant_encoder=lambda *_args: pytest.fail("encoder should not run"),
            fast_diagnostic=True,
            copy_gguf_expert_bytes=True,
        )


def test_ds4_kquant_preflight_fails_closed_on_unmapped_dense_recipe(tmp_path):
    src = tmp_path / "src"
    _ds4_kquant_source(src)
    recipe = tmp_path / "recipe.gguf"
    imatrix = tmp_path / "imatrix.dat"
    _write_recipe_gguf(recipe, [
        ("blk.0.ffn_gate_exps.weight", 16, [256, 1]),
        ("blk.0.ffn_up_exps.weight", 16, [256, 1]),
        ("blk.0.ffn_down_exps.weight", 10, [256, 2]),
        ("blk.0.attn_q_b.weight", 8, [256, 128]),
    ])
    _imatrix(imatrix)

    with pytest.raises(KQuantRecipeError, match="blk.0.attn_q_b.weight"):
        preflight_ds4_kquant_package(
            src,
            gguf_recipe_path=recipe,
            imatrix_path=imatrix,
        )


def test_build_ds4_kquant_package_fails_closed_on_missing_recipe_projection(tmp_path):
    src = tmp_path / "src"
    _ds4_kquant_source(src)
    recipe = tmp_path / "recipe.gguf"
    imatrix = tmp_path / "imatrix.dat"
    out = tmp_path / "pkg"
    _recipe(recipe, include_up=False)
    _imatrix(imatrix)

    with pytest.raises(KQuantRecipeError, match="blk.0.ffn_up_exps.weight"):
        build_ds4_kquant_package(
            src,
            out,
            gguf_recipe_path=recipe,
            imatrix_path=imatrix,
            kquant_encoder=lambda *_args: pytest.fail("encoder should not run"),
        )


def test_build_ds4_kquant_package_checks_backend_before_writing_artifacts(
    tmp_path,
    monkeypatch,
):
    from moespresso.package.deepseek_v4 import kquant_package

    src = tmp_path / "src"
    _ds4_kquant_source(src)
    recipe = tmp_path / "recipe.gguf"
    imatrix = tmp_path / "imatrix.dat"
    out = tmp_path / "pkg"
    _recipe(recipe)
    _imatrix(imatrix)

    def fail_backend():
        raise KQuantBackendError("mlx-kquant backend unavailable")

    monkeypatch.setattr(kquant_package, "check_kquant_backend_available", fail_backend)

    with pytest.raises(KQuantBackendError, match="backend unavailable"):
        build_ds4_kquant_package(
            src,
            out,
            gguf_recipe_path=recipe,
            imatrix_path=imatrix,
        )

    assert not (out / INVENTORY_NAME).exists()
    assert not (out / PACKAGE_PLAN_NAME).exists()
    assert not (out / MANIFEST_NAME).exists()
    assert list(out.glob("model-*.safetensors")) == []


@pytest.mark.skipif(
    os.environ.get("MOESPRESSO_RUN_KQUANT_REAL_BACKEND") != "1",
    reason="explicit real mlx-kquant backend diagnostic only",
)
def test_build_ds4_kquant_package_with_real_backend_on_tiny_source(tmp_path):
    src = tmp_path / "src"
    _ds4_kquant_source(src)
    recipe = tmp_path / "recipe.gguf"
    imatrix = tmp_path / "imatrix.dat"
    out = tmp_path / "pkg"
    _recipe(recipe)
    _imatrix(imatrix)

    manifest = build_ds4_kquant_package(
        src,
        out,
        gguf_recipe_path=recipe,
        imatrix_path=imatrix,
        shard_size_gb=0.0,
    )

    assert manifest["status"] == "valid"
    assert [v for v in verify_package(manifest, out) if v.blocking] == []
    from safetensors.numpy import load_file

    arrays = {}
    for file in manifest["files"]:
        arrays.update(load_file(str(out / file["path"])))
    assert arrays["layers.0.attn.wq_a.weight"].dtype == np.uint8
    assert arrays["layers.0.attn.wq_a.scales"].tolist() == [0]
    assert "layers.0.attn.wq_a.biases" not in arrays
    assert arrays["layers.0.ffn.gate.weight"].dtype == np.float32
    assert "layers.0.ffn.experts.tq_bundle" in arrays
    report = json.loads((out / KQUANT_RECIPE_REPORT_NAME).read_text())
    assert report["fit"]["dense"]["kquant_targets"] == 1
    assert report["package_size_bytes"] > 0


def test_conservative_dense_mxfp8_allocation_is_not_labeled_lossless():
    from moespresso.package.deepseek_v4.kquant_package import (
        _conservative_dense_allocation,
    )

    entry = {
        "source_name": "layers.0.attn.wo_a.weight",
        "role": "attn_out_a",
        "layer_index": 0,
        "dtype": "F8_E4M3",
    }
    alloc = _conservative_dense_allocation(
        entry, {"layers.0.attn.wo_a.scale"})
    assert alloc["format"] == "mxfp8"
    assert alloc["source_codec"] == "fp8_e4m3_ue8m0"
    # The writer re-encodes through `mx.quantize(mode="mxfp8")`, which
    # re-derives each group scale from amax and clips some group maxima
    # relative to the source e8m0 block scale, so the repack is lossy and
    # the manifest must not claim otherwise.
    assert alloc["lossless"] is False

    plain = _conservative_dense_allocation(
        {
            "source_name": "layers.0.attn.wo_b.weight",
            "role": "attn_out_b",
            "layer_index": 0,
            "dtype": "BF16",
        },
        set(),
    )
    assert plain["format"] == "affine"
    assert "lossless" not in plain
