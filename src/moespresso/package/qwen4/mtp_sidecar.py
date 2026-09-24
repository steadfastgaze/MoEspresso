"""Price an additive IQ2_K Qwen MTP sidecar without changing target payloads."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from moespresso.core.artifact import artifact_producer, make_artifact
from moespresso.inventory.qwen4.mtp import inspect_qwen4_mtp_source
from moespresso.package.iqk_format import IQK_LAYOUT_IQK_RELAYOUT
from moespresso.package.qwen4.mtp_format import MTP_CODEC, MTPProjectionPlan, mtp_iq2_projection_plan


def _source_binding(source_dir: Path, target: dict) -> dict:
    from moespresso.inventory.qwen4.source_identity import qwen4_hf_snapshot_source_identity
    from moespresso.package.manifest import file_identity

    subject = target["subject"]
    teacher = {
        "model_id": subject.get("source_root"),
        "revision": subject.get("revision"),
        "config_sha256": file_identity(source_dir / "config.json")["sha256"],
        "index_sha256": file_identity(source_dir / "model.safetensors.index.json")["sha256"],
    }
    identity = qwen4_hf_snapshot_source_identity(source_dir, teacher_source_identity=teacher)
    if identity["snapshot_identity_sha256"] != subject.get("snapshot_identity_sha256"):
        raise ValueError("MTP source snapshot does not match the target package")
    return identity


def plan_qwen4_mtp_sidecar(
    source_dir: Path, target_package: Path, *, allow_uncalibrated: bool = False,
) -> dict:
    """Bind the source and IQ2 allocation to one unchanged target manifest.

    The initial builder supports the codec's unweighted objective only, with
    explicit opt-in. It does not label that objective calibrated.
    """
    from moespresso.core.artifact import read_artifact
    from moespresso.package.constants import MANIFEST_NAME
    from moespresso.package.qwen4.mtp_format import MTP_GRAPH_CONTRACT, MTP_SIDECAR_SCHEMA, mtp_graph_config

    if not allow_uncalibrated:
        raise ValueError("MTP build requires explicit uncalibrated opt-in")
    source_dir, target_package = Path(source_dir), Path(target_package)
    target = read_artifact(target_package / MANIFEST_NAME)
    if target["artifact_kind"] != "package_manifest" or target["status"] != "valid":
        raise ValueError("MTP requires a valid target package manifest")
    if target.get("architecture", {}).get("family") != "qwen4_exp":
        raise ValueError("MTP requires a Qwen target package")
    inventory = preflight_qwen4_mtp_sidecar(source_dir)
    config = inventory["config"]
    if (
        config.get("mtp_num_hidden_layers") != 1
        or config.get("mtp_use_dedicated_embeddings") is not False
        or config.get("tie_word_embeddings") is not False
        or config.get("mtp", {}).get("layer_types") != ["full_attention"]
    ):
        raise ValueError("MTP source does not declare the shared-head single-QSA contract")
    graph = mtp_graph_config(config)
    if graph != mtp_graph_config(target["architecture"]["config"]):
        raise ValueError("MTP source geometry differs from the target")
    source = _source_binding(source_dir, target)
    return make_artifact(
        "package_plan", subject={"family": "qwen4_exp", "role": "mtp_sidecar"},
        producer=artifact_producer("moespresso.package.qwen4.mtp_sidecar"), status="valid",
        sidecar_schema=MTP_SIDECAR_SCHEMA, graph_contract=MTP_GRAPH_CONTRACT,
        target_artifact_id=target["artifact_id"], source_identity=source,
        source_inventory_id=inventory["artifact_id"], graph=graph,
        quantization={"codec": MTP_CODEC, "objective": "ik_unweighted", "calibrated": False},
        tensors=inventory["tensors"], byte_estimate=inventory["byte_estimate"],
    )


def preflight_qwen4_mtp_sidecar(source_dir: Path) -> dict:
    """Resolve source ownership and exact packed bytes before reading weights.

    Counts include input and output padding for existing IQ2_K kernels.
    Activation scratch and draft KV are not included.
    This inventory is not a calibrated quantization decision or a built sidecar.
    """
    config, tensors = inspect_qwen4_mtp_source(source_dir)
    records = []
    source_bytes = packed_bytes = 0
    for tensor in tensors:
        header = tensor.header
        record = asdict(tensor)
        record["header"]["shape"] = list(header.shape)
        source_bytes += header.end - header.begin
        if tensor.kind == "passthrough":
            size = header.end - header.begin
            record.update(format="raw_dtype_passthrough", encoded_bytes=size)
        else:
            plan = mtp_iq2_projection_plan(header.shape)
            size = plan.encoded_bytes
            record.update(
                format="iqk",
                codec=MTP_CODEC,
                layout=IQK_LAYOUT_IQK_RELAYOUT,
                logical_in_features=plan.in_features,
                stored_in_features=sum(part.stored_width for part in plan.slices),
                projection_plan=asdict(plan),
                encoded_bytes=size,
            )
        packed_bytes += size
        records.append(record)
    return make_artifact(
        "source_inventory",
        subject={"family": "qwen4_exp", "role": "mtp_sidecar", "source_format": "hf_safetensors"},
        producer=artifact_producer("moespresso.package.qwen4.mtp_sidecar"),
        status="valid",
        inventory_schema="qwen4_mtp_source_v1",
        target_payload_mutation=False,
        calibration="not_performed",
        config=config,
        tensors=records,
        byte_estimate={
            "source_payload": source_bytes,
            "iq2_k_payload_with_input_padding": packed_bytes,
            "includes_runtime_output_padding": True,
            "includes_runtime_kv_and_scratch": False,
        },
    )


def encode_mtp_iq2_projection(
    weights,
    plan: MTPProjectionPlan,
    importance=None,
    *,
    allow_uncalibrated: bool = False,
) -> list[dict]:
    """Encode one planned projection using the existing IQ2_K codec and layout.

    Missing importance requires an explicit uncalibrated opt-in. This helper
    does not record calibration provenance; the package plan owns that fact.
    """
    import numpy as np
    from mlx_iqk.codec import quantize
    from mlx_iqk.format import component_shapes, pack

    values = np.asarray(weights, dtype=np.float32)
    if mtp_iq2_projection_plan(tuple(values.shape)) != plan:
        raise ValueError("MTP weights disagree with the projection plan")
    if not np.isfinite(values).all():
        raise ValueError("MTP weights must be finite")
    if importance is None:
        if not allow_uncalibrated:
            raise ValueError("MTP encoding requires importance or explicit uncalibrated opt-in")
    else:
        importance = np.asarray(importance, dtype=np.float32)
        if (
            importance.shape != (plan.in_features,)
            or not np.isfinite(importance).all()
            or (importance < 0).any()
            or not (importance > 0).any()
        ):
            raise ValueError("MTP importance must be a finite nonnegative input-width vector")
    values = values.reshape(plan.num_experts, plan.out_features, plan.in_features)
    encoded = []
    for part in plan.slices:
        padded = np.zeros(
            (plan.num_experts, plan.stored_out_features, part.stored_width), dtype=np.float32,
        )
        padded[:, :plan.out_features, :part.end - part.begin] = values[:, :, part.begin:part.end]
        imatrix = None
        if importance is not None:
            imatrix = np.zeros(part.stored_width, dtype=np.float32)
            imatrix[:part.end - part.begin] = importance[part.begin:part.end]
        wire = quantize(MTP_CODEC, padded.reshape(-1, part.stored_width), imatrix)
        packed = pack(MTP_CODEC, wire, part.stored_width)
        shapes = component_shapes(
            MTP_CODEC, plan.num_experts, plan.stored_out_features, part.stored_width,
        )
        encoded.append({name: value.reshape(shapes[name]) for name, value in packed.items()})
    return encoded


def _write_iq2_source_tensor(source_dir, header, tensor_index, writer):
    """Stream row bands into existing component-file and shard writers."""
    from contextlib import ExitStack
    import tempfile

    import numpy as np
    from mlx_iqk.format import component_dtypes, component_shapes

    from moespresso.probe.weight_io import iter_row_chunks

    plan = mtp_iq2_projection_plan(header.shape)
    rows, columns = header.shape[-2:]
    entries = []
    with tempfile.TemporaryDirectory(prefix="mtp-components-", dir=writer.out_dir) as temporary:
        with ExitStack() as stack:
            streams = []
            handles = []
            for index, part in enumerate(plan.slices):
                shapes = component_shapes(MTP_CODEC, plan.num_experts, plan.stored_out_features, part.stored_width)
                group, outputs = {}, {}
                for component, shape in shapes.items():
                    dtype = component_dtypes(MTP_CODEC)[component]
                    key = f"p{tensor_index}.s{index}.{component}"
                    path = Path(temporary) / f"s{index}-{component}.bin"
                    group[key] = {
                        "path": path, "dtype": {"uint32": "U32", "uint8": "U8", "float16": "F16"}[dtype.name],
                        "shape": shape, "nbytes": int(np.prod(shape)) * dtype.itemsize,
                    }
                    outputs[component] = stack.enter_context(path.open("wb"))
                    entries.append({"key": key, "slice": index, "component": component,
                                    "shape": list(shape), "dtype": group[key]["dtype"]})
                streams.append(group)
                handles.append(outputs)
            for expert in range(plan.num_experts):
                start = header.begin + expert * rows * columns * 2
                matrix = replace(header, shape=(rows, columns), begin=start, end=start + rows * columns * 2)
                for _start_row, band in iter_row_chunks(source_dir, matrix, 128 * columns * 4):
                    band_plan = mtp_iq2_projection_plan(tuple(band.shape))
                    packed = encode_mtp_iq2_projection(band, band_plan, allow_uncalibrated=True)
                    for outputs, part in zip(handles, packed, strict=True):
                        for component, values in part.items():
                            outputs[component].write(memoryview(np.ascontiguousarray(values)).cast("B"))
        keyed = {name: info for group in streams for name, info in group.items()}
        for info in keyed.values():
            if info["path"].stat().st_size != info["nbytes"]:
                raise ValueError(f"MTP encoded component byte count disagrees: {info['path'].name}")
        shard = writer.add_streamed_group_from_files(keyed)
    return [{**entry, "file": shard} for entry in entries]


def build_qwen4_mtp_sidecar(
    source_dir: Path, target_package: Path, output_dir: Path, *, plan: dict, progress=None,
) -> dict:
    """Construct the planned external sidecar and publish its manifest last."""
    from datetime import datetime, timezone

    from moespresso.core.artifact import compute_artifact_id, write_artifact
    from moespresso.inventory.safetensors_header import TensorHeader
    from moespresso.package.manifest import file_identity
    from moespresso.package.qwen4.mtp_format import MTP_SIDECAR_NAME
    from moespresso.package.write import _BF16Codes, _ShardWriter
    from moespresso.probe.weight_io import load_full_raw

    source_dir, target_package, output_dir = map(Path, (source_dir, target_package, output_dir))
    destination = output_dir.resolve()
    for root in (source_dir.resolve(), target_package.resolve()):
        if destination == root or root in destination.parents:
            raise ValueError("MTP sidecar output must be outside the source and target packages")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("MTP sidecar output must be a new or empty directory")
    expected = plan_qwen4_mtp_sidecar(source_dir, target_package, allow_uncalibrated=True)
    if plan.get("artifact_id") != compute_artifact_id(plan) or plan["artifact_id"] != expected["artifact_id"]:
        raise ValueError("MTP package plan changed or no longer matches the source and target")
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = _ShardWriter(output_dir, cap_bytes=32 << 20)
    records = []
    for index, tensor in enumerate(plan["tensors"]):
        header = TensorHeader(**{**tensor["header"], "shape": tuple(tensor["header"]["shape"])})
        if tensor["kind"] == "passthrough":
            import numpy as np

            key = f"p{index}.value"
            raw = load_full_raw(source_dir, header)
            if not np.isfinite((raw.astype(np.uint32) << 16).view(np.float32)).all():
                raise ValueError(f"MTP structural tensor is nonfinite: {header.name}")
            values = _BF16Codes(raw)
            shard = writer.add_group({key: values})
            payloads = [{"key": key, "file": shard, "shape": list(header.shape), "dtype": "BF16"}]
        else:
            payloads = _write_iq2_source_tensor(source_dir, header, index, writer)
        record = {key: value for key, value in tensor.items() if key != "header"}
        record.update(source_name=header.name, logical_shape=list(header.shape), payloads=payloads)
        records.append(record)
        if progress is not None:
            progress({"tensor": header.name, "completed": index + 1, "total": len(plan["tensors"])})
    renamed = writer.finalize()
    for record in records:
        for payload in record["payloads"]:
            payload["file"] = renamed[payload["file"]]
    manifest = make_artifact(
        "package_manifest", subject=plan["subject"],
        producer=artifact_producer("moespresso.package.qwen4.mtp_sidecar"), status="valid",
        sidecar_schema=plan["sidecar_schema"], graph_contract=plan["graph_contract"],
        target_artifact_id=plan["target_artifact_id"], graph=plan["graph"],
        source_identity=plan["source_identity"], source_plan_id=plan["artifact_id"],
        quantization=plan["quantization"], byte_estimate=plan["byte_estimate"],
        files=[file_identity(output_dir / name) for name in sorted(renamed.values())], tensors=records,
    )
    stamp = datetime.now(timezone.utc).isoformat()
    write_artifact(output_dir / "package_plan.json", plan, created_at=stamp)
    pending = output_dir / (MTP_SIDECAR_NAME + ".tmp")
    write_artifact(pending, manifest, created_at=stamp)
    pending.rename(output_dir / MTP_SIDECAR_NAME)
    return manifest


def verify_qwen4_mtp_sidecar(sidecar_dir: Path, target: dict) -> dict:
    """Hash sidecar payloads separately from loading or serving."""
    from moespresso.core.paths import resolve_artifact_file
    from moespresso.package.manifest import file_identity
    from moespresso.package.qwen4.mtp_format import read_mtp_sidecar_manifest

    manifest = read_mtp_sidecar_manifest(sidecar_dir, target)
    for record in manifest["files"]:
        path = resolve_artifact_file(sidecar_dir, record["path"])
        if file_identity(path) != record:
            raise ValueError(f"MTP file failed content verification: {record['path']}")
    return manifest
