"""Rebuild a DeepSeek-V4 IQ_K package onto the decode kernels' wire layout.

The routed bytes of an IQ_K package are the quantizer's own super-block
stream (`ik_wire`). The serving kernels read a k-contiguous relayout of the
same bits (`iqk_relayout`). This step rearranges what is already stored: no
encoder runs, no source checkpoint is read, and no reconstructed value
changes. A relayout row is the same width as the wire row it replaces, so
every bundle shape, component offset, and row stride stays where it was and
only the bytes inside each `blocks` component move.

Two gates run over the rewritten bytes, and they answer different questions.
The kernel repository's own suite proves the transform is right in general,
over the whole value space of both members and over the random byte space.
What this step has to prove is that *these* bytes came through it:

- **Round trip, every row.** Each rewritten row is turned back into wire
  bytes and compared with the wire bytes it came from. A byte that moved to
  the wrong place, a truncated read, or a short write fails here, on 100
  percent of the package rather than on a sample. Integer work only, so it
  is affordable over the whole routed stack.
- **Reference decode, sampled rows.** A deterministic sample of rows per
  bundle is decoded twice: the wire bytes through ik's own CPU dequantizer
  and the rewritten bytes through the relayout's reference decode, compared
  as raw fp16 bit patterns. This is the end-to-end statement that the served
  bytes reconstruct the same weights.

The rewrite is per shard and atomic: a shard is written to a temporary file
and renamed only after both gates pass on it. A shard whose bundles already
declare the relayout is skipped, so an interrupted run resumes from the
shards' own recorded layout rather than from a journal.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from moespresso.core.artifact import read_artifact, write_artifact
from moespresso.package.bundle import (
    IQK_CODEC,
    METADATA_KEY,
    decode_bundle_metadata,
    encode_bundle_metadata,
)
from moespresso.package.constants import MANIFEST_NAME
from moespresso.package.iqk_format import (
    IQK_LAYOUT_IK_WIRE,
    IQK_LAYOUT_IQK_RELAYOUT,
)
from moespresso.package.iqk_relayout import (
    check_relayout_member,
    decode_rows,
    pack_rows,
    unpack_rows,
    wire_group_rows,
)
from moespresso.package.manifest import PACKAGE_FORMAT, file_identity

PACKAGE_PLAN_NAME = "package_plan.json"
IQK_REPORT_NAME = "iqk_package_report.json"
RELAYOUT_REPORT_NAME = "iqk_relayout_report.json"
DEFAULT_SAMPLE_ROWS = 64


class IQKRelayoutError(ValueError):
    """The package, a shard, or a rewritten row is not what the relayout needs."""


# --------------------------------------------------------------------------
# Shard IO


def read_shard_header(path: Path) -> tuple[dict, int]:
    """The safetensors header of one shard plus the byte where data starts."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise IQKRelayoutError(f"{path}: too short to hold a safetensors header")
        length = struct.unpack("<Q", raw)[0]
        header = json.loads(f.read(length))
    return header, 8 + length


def bundle_projections(header: dict) -> dict[int, dict]:
    """Per-layer bundle geometry a shard declares, or `{}` when it has none."""
    metadata = header.get("__metadata__") or {}
    if METADATA_KEY not in metadata:
        return {}
    return decode_bundle_metadata(metadata[METADATA_KEY])


def _single_bundle_tensor(header: dict) -> str:
    """The one bundle tensor of a streamed bundle shard, or refuse."""
    names = [name for name in header if name != "__metadata__"]
    if len(names) != 1:
        raise IQKRelayoutError(
            f"a bundle shard carrying {len(names)} tensors is not the streamed "
            "one-bundle-per-shard form this step rewrites")
    return names[0]


def _iqk_spans(geometry: dict) -> list[dict]:
    """Every IQ_K projection of one layer bundle, as row spans."""
    spans = []
    for projection, params in sorted((geometry.get("projections") or {}).items()):
        if params.get("codec") != IQK_CODEC:
            continue
        blocks = params["blocks"]
        out_features, bytes_per_row = blocks["shape"]
        spans.append({
            "projection": projection,
            "codec": check_relayout_member(params["iqk_codec"]),
            "layout": params.get("layout"),
            "offset": int(blocks["offset"]),
            "nbytes": int(blocks["nbytes"]),
            "out_features": int(out_features),
            "bytes_per_row": int(bytes_per_row),
            "in_features": int(params["in_features"]),
        })
    return spans


def _sample_row_indices(count: int, sample: int, salt: int,
                        group: int = 1) -> np.ndarray:
    """A deterministic row sample: the edges, plus a content-seeded spread.

    ``group`` is the member's wire row group. A grouped member's wire is
    addressable in whole groups only, so each sampled row expands to its
    complete group and the returned indices slice the wire at group
    boundaries. The relayout side stays per-row either way.
    """
    if sample <= 0:
        return np.empty(0, dtype=np.int64)
    if sample >= count:
        return np.arange(count, dtype=np.int64)
    edges = [0, 1, count // 2, count - 2, count - 1]
    rng = np.random.default_rng(salt)
    drawn = rng.choice(count, size=sample, replace=False)
    picks = np.unique(np.concatenate([np.array(edges, dtype=np.int64), drawn]))
    if group <= 1:
        return picks
    starts = np.unique(picks // group) * group
    return (starts[:, None] + np.arange(group, dtype=np.int64)).reshape(-1)


# --------------------------------------------------------------------------
# The per-shard rewrite


def relayout_shard(
    src: Path,
    dst: Path,
    *,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    roundtrip: bool = True,
) -> dict:
    """Rewrite one bundle shard onto the relayout. Returns its gate report."""
    from mlx_iqk import codec as iqk_codec

    src, dst = Path(src), Path(dst)
    header, data_start = read_shard_header(src)
    layers = bundle_projections(header)
    if not layers:
        raise IQKRelayoutError(f"{src.name}: carries no expert bundles")
    if len(layers) != 1:
        raise IQKRelayoutError(
            f"{src.name}: carries {len(layers)} layer bundles; this step rewrites "
            "the streamed one-bundle-per-shard form")
    (layer, geometry), = layers.items()
    key = _single_bundle_tensor(header)
    spans = _iqk_spans(geometry)
    if not spans:
        raise IQKRelayoutError(f"{src.name}: layer {layer} has no IQ_K projections")
    stale = [s["projection"] for s in spans if s["layout"] != IQK_LAYOUT_IK_WIRE]
    if stale:
        raise IQKRelayoutError(
            f"{src.name}: projection(s) {stale} are not on {IQK_LAYOUT_IK_WIRE!r}")

    num_experts = int(geometry["num_experts"])
    row_bytes = int(geometry["row_bytes"])
    entry = header[key]
    if entry["dtype"] != "U8" or entry["shape"] != [num_experts, row_bytes]:
        raise IQKRelayoutError(
            f"{src.name}: bundle tensor is {entry['dtype']}{entry['shape']}, not "
            f"U8[{num_experts}, {row_bytes}]")

    out_geometry = json.loads(json.dumps(geometry))
    for span in spans:
        out_geometry["projections"][span["projection"]]["layout"] = (
            IQK_LAYOUT_IQK_RELAYOUT)
    out_header = {
        "__metadata__": {
            "format": PACKAGE_FORMAT,
            METADATA_KEY: encode_bundle_metadata({layer: out_geometry}),
        },
        key: {
            "dtype": "U8",
            "shape": [num_experts, row_bytes],
            "data_offsets": [0, num_experts * row_bytes],
        },
    }
    blob = json.dumps(out_header, sort_keys=True, separators=(",", ":")).encode("utf-8")

    rows_checked = 0
    sampled = 0
    tmp = dst.with_name(dst.name + ".relayout.tmp")
    started = time.perf_counter()
    try:
        with open(src, "rb") as fin, open(tmp, "wb") as fout:
            fout.write(struct.pack("<Q", len(blob)))
            fout.write(blob)
            for expert in range(num_experts):
                fin.seek(data_start + expert * row_bytes)
                raw = fin.read(row_bytes)
                if len(raw) != row_bytes:
                    raise IQKRelayoutError(
                        f"{src.name}: expert {expert} short read "
                        f"({len(raw)} of {row_bytes} B)")
                row = bytearray(raw)
                for span in spans:
                    wire = np.frombuffer(
                        raw, dtype=np.uint8, count=span["nbytes"], offset=span["offset"]
                    ).reshape(span["out_features"], span["bytes_per_row"])
                    packed = pack_rows(span["codec"], wire, span["in_features"])
                    if roundtrip:
                        back = unpack_rows(span["codec"], packed, span["in_features"])
                        if not np.array_equal(back, wire):
                            raise IQKRelayoutError(
                                f"{src.name}: layer {layer} {span['projection']} "
                                f"expert {expert} does not round trip; the "
                                "rearranged bytes are not the stored bytes")
                        rows_checked += span["out_features"]
                    picks = _sample_row_indices(
                        span["out_features"], sample_rows,
                        salt=(layer * 1_000_003 + expert * 1009
                              + len(span["projection"])),
                        group=wire_group_rows(span["codec"]))
                    if picks.size:
                        want = iqk_codec.dequantize(
                            span["codec"], np.ascontiguousarray(wire[picks]),
                            span["in_features"]).astype(np.float16)
                        got = decode_rows(
                            span["codec"], np.ascontiguousarray(packed[picks]),
                            span["in_features"]).astype(np.float16)
                        bad = int(np.count_nonzero(
                            want.view(np.uint16) != got.view(np.uint16)))
                        if bad:
                            raise IQKRelayoutError(
                                f"{src.name}: layer {layer} {span['projection']} "
                                f"expert {expert} decodes to {bad} differing fp16 "
                                "values through the two references")
                        sampled += int(picks.size)
                    row[span["offset"]:span["offset"] + span["nbytes"]] = (
                        packed.tobytes())
                fout.write(bytes(row))
        os.replace(tmp, dst)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return {
        "shard": dst.name,
        "layer": int(layer),
        "experts": num_experts,
        "projections": [s["projection"] for s in spans],
        "members": sorted({s["codec"] for s in spans}),
        "rows_round_tripped": rows_checked,
        "rows_reference_decoded": sampled,
        "seconds": round(time.perf_counter() - started, 3),
    }


def _relayout_shard_task(args: tuple) -> dict:
    src, dst, sample_rows, roundtrip = args
    return relayout_shard(Path(src), Path(dst),
                          sample_rows=sample_rows, roundtrip=roundtrip)


# --------------------------------------------------------------------------
# The artifacts


def _relayout_plan(plan: dict) -> tuple[dict, int]:
    """The same plan with every IQ_K allocation moved onto the relayout."""
    out = json.loads(json.dumps(plan))
    moved = 0
    for alloc in out.get("allocation", []):
        if alloc.get("format") != IQK_CODEC:
            continue
        check_relayout_member(alloc.get("iqk_codec") or alloc.get("codec"))
        if alloc.get("layout") != IQK_LAYOUT_IK_WIRE:
            raise IQKRelayoutError(
                f"{alloc.get('source_name')} {alloc.get('projection')}: plan layout "
                f"{alloc.get('layout')!r} is not {IQK_LAYOUT_IK_WIRE!r}")
        alloc["layout"] = IQK_LAYOUT_IQK_RELAYOUT
        moved += 1
    if not moved:
        raise IQKRelayoutError("package plan carries no IQ_K allocation")
    out.pop("artifact_id", None)
    out.pop("created_at", None)
    return out, moved


def _relayout_manifest(manifest: dict, plan_id: str, files: list[dict]) -> tuple[dict, int]:
    """The same manifest on the relayout, against the rewritten files."""
    out = json.loads(json.dumps(manifest))
    moved = 0
    for tensor in out.get("tensors", []):
        if tensor.get("format") != IQK_CODEC:
            continue
        params = tensor.setdefault("format_params", {})
        if params.get("layout") != IQK_LAYOUT_IK_WIRE:
            raise IQKRelayoutError(
                f"{tensor.get('source_name')}: manifest layout "
                f"{params.get('layout')!r} is not {IQK_LAYOUT_IK_WIRE!r}")
        params["layout"] = IQK_LAYOUT_IQK_RELAYOUT
        moved += 1
    if not moved:
        raise IQKRelayoutError("package manifest declares no IQ_K tensors")
    out["files"] = sorted(files, key=lambda f: f["path"])
    out.setdefault("provenance", {})["source_plan_id"] = plan_id
    out.pop("artifact_id", None)
    out.pop("created_at", None)
    return out, moved


def _rewrite_sidecars(package_dir: Path, manifest: dict, seed: int) -> bool:
    """Regenerate the jang-compatible sidecars from the rewritten manifest.

    `moespresso verify` re-derives them and compares, so they are written
    again from the manifest that now ships rather than left as the ones the
    previous manifest produced. A package that carries no sidecars keeps
    none.
    """
    config_path = package_dir / "config.json"
    jang_path = package_dir / "jang_config.json"
    if not (config_path.is_file() and jang_path.is_file()):
        return False
    from moespresso.package.sidecars import build_sidecars

    config_json, jang_config = build_sidecars(manifest, seed=seed)
    config_path.write_text(json.dumps(config_json, indent=2))
    jang_path.write_text(json.dumps(jang_config, indent=2))
    return True


def _sidecar_seed(package_dir: Path) -> int:
    jang = package_dir / "jang_config.json"
    if jang.is_file():
        value = json.loads(jang.read_text()).get("mxtq_seed")
        if isinstance(value, int):
            return value
    return 42


# --------------------------------------------------------------------------
# The rebuild


def relayout_package(
    package_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    workers: int = 4,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    roundtrip: bool = True,
    verbose: bool = False,
) -> dict:
    """Move a built IQ_K package's routed bundles onto the relayout."""
    package_dir = Path(package_dir)
    out_dir = Path(output_dir) if output_dir is not None else package_dir
    in_place = out_dir == package_dir
    log = (lambda msg: print(msg, flush=True)) if verbose else (lambda msg: None)

    manifest = read_artifact(package_dir / MANIFEST_NAME)
    plan = read_artifact(package_dir / PACKAGE_PLAN_NAME)
    if manifest.get("architecture", {}).get("family") != "deepseek_v4_flash":
        raise IQKRelayoutError(f"{package_dir} is not a DeepSeek V4 package")

    declared = sorted({
        f["path"] for f in manifest.get("files", [])
    })
    bundle_shards: list[str] = []
    plain_shards: list[str] = []
    for name in declared:
        header, _ = read_shard_header(package_dir / name)
        layers = bundle_projections(header)
        if any(_iqk_spans(geo) for geo in layers.values()):
            bundle_shards.append(name)
        else:
            plain_shards.append(name)
    if not bundle_shards:
        raise IQKRelayoutError(f"{package_dir}: no shard carries IQ_K bundles")
    log(f"[1/4] {len(bundle_shards)} IQ_K bundle shard(s), "
        f"{len(plain_shards)} other shard(s)")

    if not in_place:
        out_dir.mkdir(parents=True, exist_ok=True)
        for entry in sorted(package_dir.iterdir()):
            if entry.name in bundle_shards or not entry.is_file():
                continue
            shutil.copy2(entry, out_dir / entry.name)
        log(f"[2/4] copied {len(plain_shards)} shard(s) and the package furniture "
            f"to {out_dir}")
    else:
        log("[2/4] rewriting in place; each shard is renamed over its own only "
            "after both gates pass")

    tasks = [
        (str(package_dir / name), str(out_dir / name), sample_rows, roundtrip)
        for name in bundle_shards
    ]
    started = time.perf_counter()
    reports: list[dict] = []
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for report in pool.map(_relayout_shard_task, tasks):
                reports.append(report)
                log(f"  layer {report['layer']:>2}: {report['rows_round_tripped']} rows "
                    f"round-tripped, {report['rows_reference_decoded']} decoded, "
                    f"{report['seconds']}s")
    else:
        for task in tasks:
            report = _relayout_shard_task(task)
            reports.append(report)
            log(f"  layer {report['layer']:>2}: {report['rows_round_tripped']} rows "
                f"round-tripped, {report['rows_reference_decoded']} decoded, "
                f"{report['seconds']}s")
    rewrite_seconds = time.perf_counter() - started
    log(f"[3/4] {len(reports)} shard(s) rewritten in {rewrite_seconds:.1f}s")

    new_plan, plan_moved = _relayout_plan(plan)
    plan_id = write_artifact(out_dir / PACKAGE_PLAN_NAME, new_plan,
                             created_at=plan.get("created_at"))
    files = [file_identity(out_dir / name) for name in declared]
    new_manifest, tensors_moved = _relayout_manifest(manifest, plan_id, files)
    manifest_id = write_artifact(out_dir / MANIFEST_NAME, new_manifest,
                                 created_at=manifest.get("created_at"))
    _rewrite_sidecars(out_dir, read_artifact(out_dir / MANIFEST_NAME),
                      _sidecar_seed(package_dir))
    log(f"[4/4] plan {plan_id}\n      manifest {manifest_id}")

    report = {
        "package": str(out_dir),
        "source_package": str(package_dir),
        "in_place": in_place,
        "from_layout": IQK_LAYOUT_IK_WIRE,
        "to_layout": IQK_LAYOUT_IQK_RELAYOUT,
        "plan_id": {"before": plan.get("artifact_id"), "after": plan_id},
        "manifest_id": {"before": manifest.get("artifact_id"), "after": manifest_id},
        "allocations_moved": plan_moved,
        "manifest_tensors_moved": tensors_moved,
        "gates": {
            "round_trip": "every row" if roundtrip else "off",
            "reference_decode_sample_rows": sample_rows,
            "rows_round_tripped": sum(r["rows_round_tripped"] for r in reports),
            "rows_reference_decoded": sum(r["rows_reference_decoded"] for r in reports),
        },
        "shards": reports,
        "rewrite_seconds": round(rewrite_seconds, 1),
    }
    (out_dir / RELAYOUT_REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True))
    build_report = out_dir / IQK_REPORT_NAME
    if build_report.is_file():
        payload = json.loads(build_report.read_text())
        payload["wire_layout"] = IQK_LAYOUT_IQK_RELAYOUT
        payload["package_plan_id"] = plan_id
        payload["package_manifest_id"] = manifest_id
        payload["relayout"] = {"report": RELAYOUT_REPORT_NAME}
        build_report.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Move a built DeepSeek-V4 IQ_K package's routed bundles from "
                    "the quantizer's wire layout onto the decode kernels' relayout")
    parser.add_argument("package_dir", help="Built IQ_K package directory")
    parser.add_argument(
        "--output", default=None,
        help="Write the relayout package here instead of rewriting in place. "
             "The rewrite is the same size as the package it reads, so a copy "
             "needs the package's own footprint free again")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS,
        help="Rows per expert per projection decoded through both references")
    parser.add_argument(
        "--no-round-trip", action="store_true",
        help="Skip the every-row round-trip gate. The gate is what proves this "
             "package's own bytes survived the rearrangement")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = relayout_package(
            args.package_dir,
            output_dir=args.output,
            workers=args.workers,
            sample_rows=args.sample_rows,
            roundtrip=not args.no_round_trip,
            verbose=True,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", flush=True)
        return 2
    print(json.dumps({k: v for k, v in report.items() if k != "shards"},
                     indent=2, sort_keys=True), flush=True)
    return 0
