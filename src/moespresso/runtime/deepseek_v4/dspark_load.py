"""Loader for the DSpark draft-model sidecar package.

Reads the sidecar directory written by
`moespresso.package.deepseek_v4.dspark_sidecar`: validates the content-hashed
manifest (kind, schema version, per-file sha256), rebuilds the model arguments
from the stored source config, quantizes the draft module tree to match the
manifest's per-tensor formats, and performs a strict `load_weights`. Tensor
names were resolved once at build time; nothing here parses names beyond
locating each tensor's owning module. Every mismatch is an error; there is no
silent fallback.

A manifest whose routed experts carry the `iqk` format loads them through the
fully resident switch implementation in
`runtime/deepseek_v4/iqk_experts.py`. DSpark keeps its small expert stacks
resident; the target graph instead uses the pooled IQ_K path, including its
full-capacity subcase. The stored expert rows are opaque payloads rather than
draft-tree parameters: after the skeleton strip and before the strict load,
each stage's `mlp.switch_mlp` is replaced with a switch built from
`IqkSwitchLinear` projections, fed by splitting the stored
`[num_experts, out_features, row_bytes]` blocks into the relayout streams. The
stream arrays then stand in for the blocks tensors in the strict load, so
`load_weights` still covers every parameter of the final tree, and every
blocks tensor is checked against its manifest row (presence, shape, dtype, and
the member's own row geometry) before it is consumed. Only the `iqk_relayout`
layout serves; the quantizer's own `ik_wire` layout is refused by name because
the kernels would decode it into the wrong weights rather than fail.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from jang_tools.dsv4.mlx_model import ModelArgs

from moespresso.core.artifact import compute_artifact_id
from moespresso.package.deepseek_v4.dspark_sidecar import (
    KNOWN_FORMATS,
    FORMAT_AFFINE8,
    FORMAT_IQK,
    FORMAT_MXFP4,
    IQK_BLOCKS_COMPONENT,
    IQK_PROJECTIONS,
    SIDECAR_KIND,
    SIDECAR_MANIFEST_NAME,
    SIDECAR_SCHEMA_MAJOR,
    DSparkSidecarError,
    iqk_blocks_name,
    iqk_cell_features,
)
from moespresso.package.iqk_format import (
    IQK_LAYOUT_IQK_RELAYOUT,
    IQKFormatError,
    normalize_iqk_layout,
)
from moespresso.package.iqk_relayout import relayout_row_bytes, split_streams
from moespresso.runtime.deepseek_v4.dspark_model import DSparkArgs, DSparkDraftModel
from moespresso.runtime.deepseek_v4.spec_decode import strip_draft_skeleton


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mx_dtype_name(dtype) -> str:
    return str(dtype).split(".")[-1]


def read_sidecar_manifest(sidecar_dir: Path, verify_files: bool = True) -> dict:
    """Read and validate the sidecar manifest, failing closed on any mismatch."""
    manifest_path = Path(sidecar_dir) / SIDECAR_MANIFEST_NAME
    if not manifest_path.exists():
        raise DSparkSidecarError(f"missing sidecar manifest {manifest_path}")
    payload = json.loads(manifest_path.read_text())

    kind = payload.get("artifact_kind")
    if kind != SIDECAR_KIND:
        raise DSparkSidecarError(f"unknown sidecar artifact kind {kind!r}")
    major = (payload.get("schema_version") or {}).get("major")
    if major != SIDECAR_SCHEMA_MAJOR:
        raise DSparkSidecarError(
            f"unsupported sidecar schema major {major!r} "
            f"(this build: {SIDECAR_SCHEMA_MAJOR})")
    stored_id = payload.get("artifact_id")
    computed_id = compute_artifact_id(payload)
    if stored_id != computed_id:
        raise DSparkSidecarError(
            f"sidecar manifest hash mismatch: stored {stored_id} != computed {computed_id}")

    tensors = payload.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise DSparkSidecarError("sidecar manifest has no tensor table")
    file_sha256 = (payload.get("provenance") or {}).get("file_sha256")
    if not isinstance(file_sha256, dict) or not file_sha256:
        raise DSparkSidecarError("sidecar manifest has no file hashes")
    for name, row in tensors.items():
        fmt = row.get("format")
        if fmt not in KNOWN_FORMATS:
            raise DSparkSidecarError(f"{name}: unknown tensor format {fmt!r}")
        if row.get("file") not in file_sha256:
            raise DSparkSidecarError(f"{name}: file {row.get('file')!r} is not hashed")

    if verify_files:
        for file_name, expected in sorted(file_sha256.items()):
            shard_path = Path(sidecar_dir) / file_name
            if not shard_path.exists():
                raise DSparkSidecarError(f"missing sidecar shard {shard_path}")
            actual = _sha256_file(shard_path)
            if actual != expected:
                raise DSparkSidecarError(
                    f"sidecar shard {file_name} hash mismatch: "
                    f"{actual} != recorded {expected}")
    return payload


def _module_formats(manifest: dict) -> dict[str, dict]:
    """Map each quantized module path to its quantization parameters."""
    formats = manifest.get("formats") or {}
    module_params: dict[str, dict] = {}
    for name, row in manifest["tensors"].items():
        fmt = row["format"]
        if fmt not in (FORMAT_MXFP4, FORMAT_AFFINE8):
            continue
        params = formats.get(fmt) or row.get("quant")
        if not params:
            raise DSparkSidecarError(f"{name}: quantized format {fmt} has no parameters")
        module = name.rsplit(".", 1)[0]
        known = module_params.setdefault(module, dict(params))
        if known != dict(params):
            raise DSparkSidecarError(
                f"{module}: conflicting quantization parameters across tensors")
    return module_params


def _iqk_expert_rows(manifest: dict, dargs: DSparkArgs) -> dict[int, dict]:
    """Validate and group the manifest's IQ_K expert rows by stage.

    Returns `{stage: {projection: (tensor name, manifest row)}}`, empty when
    the manifest carries no IQ_K rows. Anything short of a complete,
    geometry-consistent relayout set is an error: partial coverage, a row
    whose declared stage or projection disagrees with its name, a layout
    other than the decode kernels' own, an unknown member, or a blocks shape
    that is not the member's row geometry.
    """
    margs = dargs.model_args
    rows = {
        name: row for name, row in manifest["tensors"].items()
        if row.get("format") == FORMAT_IQK
    }
    if not rows:
        return {}
    want = {
        iqk_blocks_name(stage, projection): (stage, projection)
        for stage in range(dargs.n_mtp_layers)
        for projection in IQK_PROJECTIONS
    }
    missing = sorted(set(want) - set(rows))
    unexpected = sorted(set(rows) - set(want))
    if missing or unexpected:
        raise DSparkSidecarError(
            "IQ_K expert rows do not cover the switch seams; "
            f"missing={missing[:8]} unexpected={unexpected[:8]}")

    grouped: dict[int, dict] = {}
    for name, row in rows.items():
        stage, projection = want[name]
        if int(row.get("stage", -1)) != stage or row.get("projection") != projection:
            raise DSparkSidecarError(
                f"{name}: declared stage/projection "
                f"({row.get('stage')!r}, {row.get('projection')!r}) is not "
                f"({stage}, {projection})")
        layout = normalize_iqk_layout(row.get("layout"))
        if layout != IQK_LAYOUT_IQK_RELAYOUT:
            raise DSparkSidecarError(
                f"{name}: layout {layout!r} is not "
                f"{IQK_LAYOUT_IQK_RELAYOUT!r}; the decode kernels read the "
                "relayout and would decode the quantizer's own wire into the "
                "wrong weights rather than fail")
        member = row.get("iqk_codec")
        in_features, out_features = iqk_cell_features(margs, projection)
        if (int(row.get("in_features", -1)) != in_features
                or int(row.get("out_features", -1)) != out_features
                or int(row.get("num_experts", -1)) != int(margs.n_routed_experts)):
            raise DSparkSidecarError(
                f"{name}: recorded geometry "
                f"[{row.get('num_experts')}, {row.get('out_features')}, "
                f"in {row.get('in_features')}] is not the model's "
                f"[{margs.n_routed_experts}, {out_features}, in {in_features}]")
        try:
            row_bytes = relayout_row_bytes(member, in_features)
        except IQKFormatError as e:
            raise DSparkSidecarError(f"{name}: {e}") from e
        want_shape = [int(margs.n_routed_experts), out_features, row_bytes]
        if list(row.get("shape") or []) != want_shape:
            raise DSparkSidecarError(
                f"{name}: blocks shape {row.get('shape')} is not the "
                f"{member} row geometry {want_shape}")
        if row.get("dtype") != "uint8":
            raise DSparkSidecarError(
                f"{name}: blocks dtype {row.get('dtype')!r} is not uint8")
        grouped.setdefault(stage, {})[projection] = (name, row)
    return grouped


def _install_iqk_experts(model, dargs: DSparkArgs, grouped: dict, weights: dict) -> None:
    """Swap each stage's switch seam for the resident IQ_K switch class.

    Consumes the stored blocks tensors from `weights` and puts the relayout
    streams they split into in their place, under the installed modules'
    parameter paths, so the following strict `load_weights` covers every
    parameter of the final tree. The stage's own activation module travels
    into the new switch, and the layer identity is
    `num_hidden_layers + stage`, the target graph's convention for the draft
    stages.
    """
    from mlx_iqk.nn import IqkSwitchLinear

    from moespresso.runtime.deepseek_v4.iqk_experts import iqk_switch_class

    margs = dargs.model_args
    switch_cls = iqk_switch_class()
    for stage in sorted(grouped):
        block = model.blocks[stage]
        stock = block.mlp.switch_mlp
        activation = getattr(stock, "activation", None)
        if activation is None:
            raise DSparkSidecarError(
                f"stage {stage}: the switch seam carries no activation "
                "module; the clamped SwiGLU contract lives there and is not "
                "reconstructed here")
        projections = {}
        for projection in IQK_PROJECTIONS:
            name, row = grouped[stage][projection]
            blocks = weights.pop(name)
            member = row["iqk_codec"]
            module = IqkSwitchLinear(
                member,
                int(row["num_experts"]),
                int(row["out_features"]),
                int(row["in_features"]),
            )
            streams = split_streams(
                member, np.asarray(blocks), int(row["in_features"]))
            streams_mx = {k: mx.array(v) for k, v in streams.items()}
            module.load_streams(streams_mx)
            module.eval()
            prefix = name[: -len(IQK_BLOCKS_COMPONENT)]
            for stream_name, value in streams_mx.items():
                weights[prefix + stream_name] = value
            projections[projection] = module
        switch = switch_cls(
            gate_proj=projections["gate_proj"],
            up_proj=projections["up_proj"],
            down_proj=projections["down_proj"],
            activation=activation,
            layer=int(margs.num_hidden_layers) + stage,
        )
        switch.eval()
        block.mlp.switch_mlp = switch


def dspark_iqk_engagement(model) -> dict:
    """Drafter-side IQ_K engagement facts.

    Reads `model.blocks`, the drafter's stage list, and reports which stages
    serve through the installed resident IQ_K switch modules plus their summed
    route counters, so a measurement arm can prove that the draft experts ran.
    The target graph's `iqk_engagement` reads its layer list and may report
    pooled switches; the two reports never mix.
    """
    stages = []
    for stage, block in enumerate(getattr(model, "blocks", None) or []):
        switch = getattr(getattr(block, "mlp", None), "switch_mlp", None)
        if switch is not None and type(switch).__name__ == "IqkDeepseekV4SwitchGLU":
            stages.append((stage, switch))
    switches = [switch for _, switch in stages]
    return {
        "stages": [
            {"stage": stage, "layer": switch.layer,
             "members": dict(switch.members)}
            for stage, switch in stages
        ],
        "switch_modules": len(switches),
        "total_calls": sum(s.total_calls for s in switches),
        "gemv_calls": sum(s.gemv_calls for s in switches),
        "gemv_pairs": sum(s.gemv_pairs for s in switches),
        "sorted_prefill_calls": sum(s.sorted_prefill_calls for s in switches),
        "sorted_prefill_pairs": sum(s.sorted_prefill_pairs for s in switches),
    }


def load_dspark_sidecar(
    sidecar_dir, embed, lm_head,
) -> tuple[DSparkDraftModel, DSparkArgs]:
    """Load the draft model from a sidecar directory.

    `embed` and `lm_head` are the target model's frozen modules; the sidecar
    does not carry them.
    """
    sidecar_dir = Path(sidecar_dir)
    manifest = read_sidecar_manifest(sidecar_dir)

    margs = ModelArgs.from_dict(manifest["source_config"])
    d = manifest["dspark"]
    dargs = DSparkArgs(
        model_args=margs,
        n_mtp_layers=int(d["n_mtp_layers"]),
        block_size=int(d["block_size"]),
        noise_token_id=int(d["noise_token_id"]),
        target_layer_ids=tuple(int(i) for i in d["target_layer_ids"]),
        markov_rank=int(d["markov_rank"]),
    )
    iqk_rows = _iqk_expert_rows(manifest, dargs)
    model = DSparkDraftModel(dargs, embed=embed, lm_head=lm_head)

    module_params = _module_formats(manifest)
    if module_params:
        nn.quantize(
            model,
            class_predicate=lambda path, m: dict(module_params[path])
            if path in module_params else False,
        )
    # Drop the constructed skeleton before any shard is read: the module
    # tree still references lazy random-init arrays (and the quantization
    # graphs built over them), and an evaluation reaching the tree before
    # the strict load would allocate the full float32 skeleton.
    strip_draft_skeleton(model)

    rows = manifest["tensors"]
    weights: dict[str, mx.array] = {}
    for file_name in sorted(manifest["provenance"]["file_sha256"]):
        part = mx.load(str(sidecar_dir / file_name))
        overlap = set(part) & set(weights)
        if overlap:
            raise DSparkSidecarError(
                f"tensors repeated across sidecar shards: {sorted(overlap)[:8]}")
        weights.update(part)

    missing = sorted(set(rows) - set(weights))
    unexpected = sorted(set(weights) - set(rows))
    if missing or unexpected:
        raise DSparkSidecarError(
            f"sidecar shards do not match the manifest tensor table; "
            f"missing={missing[:8]} unexpected={unexpected[:8]}")
    for name, arr in weights.items():
        row = rows[name]
        if list(arr.shape) != list(row["shape"]):
            raise DSparkSidecarError(
                f"{name}: shard shape {list(arr.shape)} != manifest {row['shape']}")
        if _mx_dtype_name(arr.dtype) != row["dtype"]:
            raise DSparkSidecarError(
                f"{name}: shard dtype {_mx_dtype_name(arr.dtype)} != "
                f"manifest {row['dtype']}")

    if iqk_rows:
        _install_iqk_experts(model, dargs, iqk_rows, weights)

    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model, dargs
