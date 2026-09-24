"""Qwen4 converted IQ_K artifact publication and resume contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gc
import weakref

import numpy as np
import pytest

from moespresso.package.iqk_artifacts import IQKConvertedArtifacts
from moespresso.package.iqk_format import iqk_geometry
from moespresso.package.iqk_format import IQK_LAYOUT_IQK_RELAYOUT
from moespresso.package.iqk_relayout import relayout_implementation_identity
from moespresso.package.iqk_relayout import pack_rows, unpack_rows
from moespresso.package.qwen4.iqk_convert import (
    CELL_STATE_SCHEMA,
    CONVERSION_INVENTORY_SCHEMA,
    IQKCellSpec,
    Qwen4IQKConversionError,
    ReleasedExpertLayerSource,
    RUN_CONTRACT_SCHEMA,
    STATE_DIR_NAME,
    TeacherLayerMoments,
    convert_qwen4_iqk_artifacts,
    cpu_iqk_encoder,
    seed_qwen4_iqk_conversion_cache,
)
from moespresso.inventory.safetensors_header import TensorHeader
from moespresso.package.qwen4.iqk_package import (
    ZERO_COUNT_MEAN_POLICY,
    _conversion_identity,
    zero_count_mean_policy_contract,
)


PROJECTIONS = ("gate", "up", "down")


@pytest.mark.parametrize("member", ("iq1_s_r4", "iq2_ks", "iq2_k", "iq3_k"))
def test_pinned_cpu_codec_relayout_round_trips_without_mlx_arrays(member: str) -> None:
    weights = np.arange(1024, dtype=np.float32).reshape(4, 256) / 1000
    steering = np.ones(256, dtype=np.float32)

    wire = cpu_iqk_encoder(member, weights, steering)
    blocks = pack_rows(member, wire, 256)

    assert np.array_equal(unpack_rows(member, blocks, 256), wire)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cells(num_experts: int = 2) -> tuple[IQKCellSpec, ...]:
    codecs = {"gate": "iq2_ks", "up": "iq2_k", "down": "iq2_ks"}
    out = []
    for projection in PROJECTIONS:
        codec = codecs[projection]
        logical = (2, 256) if projection in {"gate", "up"} else (4, 128)
        stored = logical if projection != "down" else (4, 256)
        row_bytes = iqk_geometry(codec).bytes_per_row(stored[1])
        bytes_per_expert = stored[0] * row_bytes
        out.append(
            IQKCellSpec(
                layer_index=2,
                projection=projection,
                codec=codec,
                logical_shape=logical,
                stored_shape=stored,
                zero_padding=stored[1] - logical[1],
                source_name=f"source.{projection}",
                surface_cell_identity=f"{PROJECTIONS.index(projection) + 1:064x}",
                surface_run_contract_identity="4" * 64,
                row_bytes=row_bytes,
                bytes_per_expert=bytes_per_expert,
                size_bytes=num_experts * bytes_per_expert,
            )
        )
    return tuple(out)


def _iq3_cells(num_experts: int = 2) -> tuple[IQKCellSpec, ...]:
    geometry = iqk_geometry("iq3_k")
    return tuple(
        replace(
            cell,
            codec="iq3_k",
            row_bytes=geometry.bytes_per_row(cell.stored_shape[1]),
            bytes_per_expert=cell.stored_shape[0]
            * geometry.bytes_per_row(cell.stored_shape[1]),
            size_bytes=num_experts
            * cell.stored_shape[0]
            * geometry.bytes_per_row(cell.stored_shape[1]),
        )
        for cell in _cells(num_experts)
    )


def _run_contract(
    cells: tuple[IQKCellSpec, ...],
    *,
    salt: str = "a",
    surface_content: str = "d",
    zero_count_policy: dict | None = None,
) -> dict:
    surface_run_identity = cells[0].surface_run_contract_identity
    assert all(cell.surface_run_contract_identity == surface_run_identity for cell in cells)
    body = {
        "schema": RUN_CONTRACT_SCHEMA,
        "source_identity_sha256": salt * 64,
        "allocation_decision_id": "decision:" + "b" * 64,
        "teacher_identity_sha256": "c" * 64,
        "surface_content_sha256": surface_content * 64,
        "surface_run_contract_identity": surface_run_identity,
        "encoder": {
            "package": "synthetic",
            "version": "1",
            "source_layout": "ik_wire",
            "published_layout": IQK_LAYOUT_IQK_RELAYOUT,
        },
        "relayout_implementation": relayout_implementation_identity(),
        "num_experts": 2,
        "worker_policy": {"requested": 1, "maximum": 8},
        "steering": "ordinary_route_active_per_expert_sum2_div_count",
        **({"zero_count_policy": zero_count_policy} if zero_count_policy is not None else {}),
        "cells": [cell.portable_record() for cell in cells],
    }
    return {**body, "identity_sha256": _canonical_sha256(body)}


def _inventory_base(run_contract: dict) -> dict:
    inventory = {
        "schema": CONVERSION_INVENTORY_SCHEMA,
        "source_identity": {"snapshot_identity_sha256": "a" * 64},
        "allocation_decision_id": "decision:" + "b" * 64,
        "teacher_identity": {"identity_sha256": "c" * 64},
        "surface_identity": {
            "content_sha256": run_contract["surface_content_sha256"],
            "run_contract_identity": run_contract["surface_run_contract_identity"],
        },
        "conversion_run_contract": run_contract,
    }
    if "zero_count_policy" in run_contract:
        inventory["zero_count_policy"] = run_contract["zero_count_policy"]
    return inventory


def _specialized_cells() -> tuple[IQKCellSpec, ...]:
    return tuple(
        replace(
            cell,
            zero_count_experts=(0,),
            zero_count_fallback_policy=ZERO_COUNT_MEAN_POLICY,
        )
        for cell in _cells()
    )


class _Source:
    def __init__(self, reads: list[int]):
        self.reads = reads
        self.closed = False

    def read(self, expert: int):
        self.reads.append(expert)
        gate = np.full((2, 256), expert + 1, dtype=np.float32)
        up = np.full((2, 256), expert + 3, dtype=np.float32)
        down = np.full((4, 128), expert + 5, dtype=np.float32)
        return gate, up, down

    def close(self) -> None:
        self.closed = True


def _moments(*, zero: bool = False) -> TeacherLayerMoments:
    counts = np.asarray([0 if zero else 2, 4], dtype=np.uint64)
    gate = np.vstack(
        [
            np.full(256, float(counts[0]) * 3, dtype=np.float64),
            np.full(256, float(counts[1]) * 7, dtype=np.float64),
        ]
    )
    down = np.vstack(
        [
            np.full(128, float(counts[0]) * 5, dtype=np.float64),
            np.full(128, float(counts[1]) * 11, dtype=np.float64),
        ]
    )
    return TeacherLayerMoments(2, gate, down, counts, "e" * 64)


class _Encoder:
    def __init__(self):
        self.calls = []

    def __call__(self, member, weights, steering):
        self.calls.append(
            {
                "member": member,
                "weights": weights.copy(),
                "steering": steering.copy(),
            }
        )
        row_bytes = iqk_geometry(member).bytes_per_row(weights.shape[1])
        marker = int(np.rint(weights[0, 0] + steering[0])) % 251
        return np.full((weights.shape[0], row_bytes), marker, dtype=np.uint8)


def _convert(
    out: Path,
    cells: tuple[IQKCellSpec, ...],
    contract: dict,
    encoder,
    reads: list[int],
    *,
    moments=None,
    source_opens: list[int] | None = None,
):
    sources = []

    def source_factory(layer):
        if source_opens is not None:
            source_opens.append(layer)
        source = _Source(reads)
        sources.append(source)
        return source

    payload = convert_qwen4_iqk_artifacts(
        out,
        cells=cells,
        run_contract=contract,
        inventory_base=_inventory_base(contract),
        moment_provider=lambda _layer: _moments() if moments is None else moments,
        source_factory=source_factory,
        encoder=encoder,
        workers=1,
        num_experts=2,
    )
    assert all(source.closed for source in sources)
    return payload


def test_conversion_streams_mixed_cells_and_zero_pads_down(tmp_path: Path) -> None:
    cells = _cells()
    contract = _run_contract(cells)
    encoder = _Encoder()
    reads = []

    inventory = _convert(tmp_path / "converted", cells, contract, encoder, reads)

    assert reads == [0, 1]
    assert [call["member"] for call in encoder.calls] == [
        "iq2_ks",
        "iq2_k",
        "iq2_ks",
    ] * 2
    np.testing.assert_array_equal(encoder.calls[0]["steering"], 3.0)
    np.testing.assert_array_equal(encoder.calls[1]["steering"], 3.0)
    np.testing.assert_array_equal(encoder.calls[2]["steering"][:128], 5.0)
    np.testing.assert_array_equal(encoder.calls[2]["steering"][128:], 0.0)
    np.testing.assert_array_equal(encoder.calls[2]["weights"][:, 128:], 0.0)
    assert inventory["schema"] == CONVERSION_INVENTORY_SCHEMA
    assert inventory["summary"]["member_counts"] == {"iq2_k": 1, "iq2_ks": 2}
    assert inventory["summary"]["output_bytes"] == sum(cell.size_bytes for cell in cells)
    assert all("/" not in record["name"] for record in inventory["files"])

    members = {2: {cell.projection: cell.codec for cell in cells}}
    shapes = {2: {cell.projection: cell.stored_shape for cell in cells}}
    artifacts = IQKConvertedArtifacts(
        tmp_path / "converted",
        members,
        shapes,
        2,
        layout=IQK_LAYOUT_IQK_RELAYOUT,
    )
    try:
        report = artifacts.verify_digests(tmp_path / "converted" / "inventory.json")
        assert report["files_checked"] == 3
        assert report["bytes_checked"] == inventory["summary"]["output_bytes"]
        identity = _conversion_identity(
            tmp_path / "converted" / "inventory.json",
            artifacts=artifacts,
            source_identity={"snapshot_identity_sha256": "a" * 64},
            decision={
                "artifact_id": "decision:" + "b" * 64,
                "teacher_identity": {"identity_sha256": "c" * 64},
                "surface_identity": {
                    "content_sha256": "d" * 64,
                    "run_contract_identity": "4" * 64,
                },
            },
        )
        assert identity["layout"] == IQK_LAYOUT_IQK_RELAYOUT
    finally:
        artifacts.close()


def test_conversion_accepts_iq3_k_cells_and_pads_down_rows(tmp_path: Path) -> None:
    cells = _iq3_cells()
    contract = _run_contract(cells)
    encoder = _Encoder()
    reads = []

    inventory = _convert(tmp_path / "converted", cells, contract, encoder, reads)

    assert reads == [0, 1]
    assert [call["member"] for call in encoder.calls] == ["iq3_k"] * 6
    assert inventory["summary"]["member_counts"] == {"iq3_k": 3}
    assert inventory["summary"]["output_bytes"] == sum(cell.size_bytes for cell in cells)
    down_calls = [call for call in encoder.calls if call["weights"].shape == (4, 256)]
    assert len(down_calls) == 2
    np.testing.assert_array_equal(down_calls[0]["weights"][:, 128:], 0.0)
    np.testing.assert_array_equal(down_calls[0]["steering"][128:], 0.0)


def test_conversion_exactly_resumes_prepared_and_published_cells(tmp_path: Path) -> None:
    cells = _cells()
    contract = _run_contract(cells)
    output = tmp_path / "converted"
    _convert(output, cells, contract, _Encoder(), [])
    inventory = output / "inventory.json"
    inventory.unlink()

    gate = cells[0]
    final = output / gate.name
    staged = output / STATE_DIR_NAME / "staging" / f"{gate.name}.partial"
    final.replace(staged)

    def forbidden_encoder(*_args):
        raise AssertionError("resume encoded an already prepared layer")

    reads = []
    resumed = _convert(output, cells, contract, forbidden_encoder, reads)

    assert reads == []
    assert resumed["summary"]["cells"] == 3
    assert final.is_file()
    assert not staged.exists()


def test_completed_inventory_fast_return_skips_moments_and_source(
    tmp_path: Path,
) -> None:
    cells = _cells()
    contract = _run_contract(cells)
    output = tmp_path / "converted"
    original = _convert(output, cells, contract, _Encoder(), [])

    def forbidden_moments(_layer):
        raise AssertionError("completed inventory requested moments")

    def forbidden_source(_layer):
        raise AssertionError("completed inventory opened a source")

    resumed = convert_qwen4_iqk_artifacts(
        output,
        cells=cells,
        run_contract=contract,
        inventory_base=_inventory_base(contract),
        moment_provider=forbidden_moments,
        source_factory=forbidden_source,
        encoder=_Encoder(),
        workers=1,
        num_experts=2,
    )

    assert resumed == original


def test_conversion_cache_seed_reuses_only_identical_cells(tmp_path: Path) -> None:
    donor_cells = _cells()
    donor_contract = _run_contract(donor_cells)
    donor = tmp_path / "donor"
    _convert(donor, donor_cells, donor_contract, _Encoder(), [])

    gate = donor_cells[0]
    geometry = iqk_geometry("iq2_k")
    row_bytes = geometry.bytes_per_row(gate.stored_shape[1])
    target_gate = replace(
        gate,
        codec="iq2_k",
        row_bytes=row_bytes,
        bytes_per_expert=gate.stored_shape[0] * row_bytes,
        size_bytes=2 * gate.stored_shape[0] * row_bytes,
    )
    target_cells = tuple(
        replace(
            cell,
            surface_cell_identity=f"{index + 20:064x}",
            surface_run_contract_identity="5" * 64,
        )
        for index, cell in enumerate((target_gate, *donor_cells[1:]))
    )
    target_contract = _run_contract(target_cells, surface_content="e")
    output = tmp_path / "target"
    report = seed_qwen4_iqk_conversion_cache(
        output,
        cells=target_cells,
        run_contract=target_contract,
        inventory_base=_inventory_base(target_contract),
        reuse_from=(donor,),
    )

    assert report["reused_cells"] == [donor_cells[1].name, donor_cells[2].name]
    assert report["missing_cells"] == [target_gate.name]
    assert report["reused_bytes"] == sum(cell.size_bytes for cell in donor_cells[1:])
    assert report["storage"]["remaining_output_bytes"] == target_gate.size_bytes
    for cell in donor_cells[1:]:
        assert (output / cell.name).stat().st_ino == (donor / cell.name).stat().st_ino

    encoder = _Encoder()
    reads = []
    inventory = _convert(
        output,
        target_cells,
        target_contract,
        encoder,
        reads,
    )

    assert reads == [0, 1]
    assert [call["member"] for call in encoder.calls] == ["iq2_k", "iq2_k"]
    assert inventory["summary"]["cells"] == 3
    assert inventory["summary"]["member_counts"] == {"iq2_k": 2, "iq2_ks": 1}


def test_conversion_cache_seed_rejects_teacher_drift_before_output(
    tmp_path: Path,
) -> None:
    cells = _cells()
    contract = _run_contract(cells)
    donor = tmp_path / "donor"
    _convert(donor, cells, contract, _Encoder(), [])
    output = tmp_path / "target"
    inventory_base = _inventory_base(contract)
    inventory_base["teacher_identity"] = {"identity_sha256": "f" * 64}

    with pytest.raises(Qwen4IQKConversionError, match="teacher_identity"):
        seed_qwen4_iqk_conversion_cache(
            output,
            cells=cells,
            run_contract=contract,
            inventory_base=inventory_base,
            reuse_from=(donor,),
        )

    assert not output.exists()


def test_multilayer_conversion_streams_one_moment_layer_at_a_time(
    tmp_path: Path,
) -> None:
    first = _cells()
    second = tuple(
        replace(
            cell,
            layer_index=3,
            source_name=f"source.layer3.{cell.projection}",
            surface_cell_identity=f"{index + 10:064x}",
        )
        for index, cell in enumerate(first)
    )
    cells = first + second
    contract = _run_contract(cells)
    events = []
    previous_moments = None

    def moments_provider(layer):
        nonlocal previous_moments
        if previous_moments is not None:
            gc.collect()
            assert previous_moments() is None
        moments = replace(_moments(), layer_index=layer)
        previous_moments = weakref.ref(moments)
        events.append(("moments", layer))
        return moments

    class OrderedSource(_Source):
        def __init__(self, layer):
            super().__init__([])
            self.layer = layer

        def close(self):
            events.append(("close", self.layer))
            super().close()

    def source_factory(layer):
        events.append(("source", layer))
        return OrderedSource(layer)

    inventory = convert_qwen4_iqk_artifacts(
        tmp_path / "converted",
        cells=cells,
        run_contract=contract,
        inventory_base=_inventory_base(contract),
        moment_provider=moments_provider,
        source_factory=source_factory,
        encoder=_Encoder(),
        workers=1,
        num_experts=2,
    )

    assert inventory["summary"]["layers"] == 2
    assert events == [
        ("moments", 2),
        ("source", 2),
        ("close", 2),
        ("moments", 3),
        ("source", 3),
        ("close", 3),
    ]


def test_conversion_refuses_same_size_published_corruption(tmp_path: Path) -> None:
    cells = _cells()
    contract = _run_contract(cells)
    output = tmp_path / "converted"
    _convert(output, cells, contract, _Encoder(), [])
    (output / "inventory.json").unlink()
    path = output / cells[0].name
    with open(path, "r+b") as payload:
        payload.seek(0)
        payload.write(b"\xff")

    with pytest.raises(Qwen4IQKConversionError, match="sha256"):
        _convert(output, cells, contract, _Encoder(), [])


def test_conversion_refuses_zero_count_before_source_read(tmp_path: Path) -> None:
    cells = _cells()
    contract = _run_contract(cells)
    reads = []

    with pytest.raises(Qwen4IQKConversionError, match="zero counts|no route-active"):
        _convert(
            tmp_path / "converted",
            cells,
            contract,
            _Encoder(),
            reads,
            moments=_moments(zero=True),
        )

    assert reads == []


def test_zero_count_steering_is_unweighted_mean_of_normalized_active_rows() -> None:
    counts = np.asarray([0, 2, 6], dtype=np.uint64)
    gate = np.asarray(
        [
            [0.0, 0.0],
            [6.0, 10.0],
            [54.0, 66.0],
        ],
        dtype=np.float64,
    )
    down = np.asarray(
        [
            [0.0, 0.0],
            [8.0, 12.0],
            [60.0, 72.0],
        ],
        dtype=np.float64,
    )
    moments = TeacherLayerMoments(
        0,
        gate,
        down,
        counts,
        "e" * 64,
        zero_count_experts=(0,),
        zero_count_fallback_policy=ZERO_COUNT_MEAN_POLICY,
    ).precompute_zero_count_fallbacks()

    np.testing.assert_array_equal(moments.steering("gate", 0), [6.0, 8.0])
    np.testing.assert_array_equal(moments.steering("up", 0), [6.0, 8.0])
    np.testing.assert_array_equal(moments.steering("down", 0), [7.0, 9.0])
    assert not np.array_equal(
        moments.steering("gate", 0),
        np.sum(gate, axis=0) / np.sum(counts),
    )


def test_specialized_conversion_precomputes_and_records_zero_count_policy(
    tmp_path: Path,
) -> None:
    cells = _specialized_cells()
    policy = zero_count_mean_policy_contract(((2, 0),))
    contract = _run_contract(cells, zero_count_policy=policy)
    moments = replace(
        _moments(zero=True),
        zero_count_experts=(0,),
        zero_count_fallback_policy=ZERO_COUNT_MEAN_POLICY,
    )

    inventory = _convert(
        tmp_path / "converted",
        cells,
        contract,
        _Encoder(),
        [],
        moments=moments,
    )

    assert inventory["zero_count_policy"] == policy
    assert inventory["summary"]["zero_count_fallback_pairs"] == [[2, 0]]
    assert inventory["summary"]["zero_count_fallback_policy"] == ZERO_COUNT_MEAN_POLICY
    assert all(record["zero_count_experts"] == [0] for record in inventory["files"])


def test_invalid_zero_count_mean_fails_before_source_or_layer_staging(
    tmp_path: Path,
) -> None:
    cells = _specialized_cells()
    policy = zero_count_mean_policy_contract(((2, 0),))
    contract = _run_contract(cells, zero_count_policy=policy)
    moments = replace(
        _moments(zero=True),
        gate_up_sum2=np.zeros((2, 256), dtype=np.float64),
        zero_count_experts=(0,),
        zero_count_fallback_policy=ZERO_COUNT_MEAN_POLICY,
    )
    output = tmp_path / "converted"
    reads = []
    source_opens = []

    with pytest.raises(Qwen4IQKConversionError, match="active-row mean"):
        _convert(
            output,
            cells,
            contract,
            _Encoder(),
            reads,
            moments=moments,
            source_opens=source_opens,
        )

    assert reads == []
    assert source_opens == []
    assert output.is_dir()
    state = output / STATE_DIR_NAME
    assert list((state / "staging").iterdir()) == []
    assert list((state / "cells").iterdir()) == []
    assert list((state / "layers").iterdir()) == []


def test_conversion_refuses_run_contract_and_cell_state_drift(tmp_path: Path) -> None:
    cells = _cells()
    contract = _run_contract(cells)
    output = tmp_path / "converted"
    _convert(output, cells, contract, _Encoder(), [])
    (output / "inventory.json").unlink()

    drifted = _run_contract(cells, salt="f")
    with pytest.raises(Qwen4IQKConversionError, match="another run contract"):
        _convert(output, cells, drifted, _Encoder(), [])

    descriptor = output / STATE_DIR_NAME / "cells" / "layer02-gate.json"
    payload = json.loads(descriptor.read_text())
    assert payload["schema"] == CELL_STATE_SCHEMA
    payload["file"]["codec"] = "iq2_k"
    descriptor.write_text(json.dumps(payload))
    with pytest.raises(Qwen4IQKConversionError, match="codec drifted"):
        _convert(output, cells, contract, _Encoder(), [])


def test_released_source_publishes_descriptors_before_worker_contention(
    tmp_path: Path,
) -> None:
    (tmp_path / "gate.safetensors").write_bytes(b"")
    (tmp_path / "down.safetensors").write_bytes(b"")
    gate = TensorHeader("gate", (2, 2, 2), "BF16", "gate.safetensors", 0, 0, 16)
    down = TensorHeader("down", (2, 2, 2), "BF16", "down.safetensors", 0, 0, 16)

    source = ReleasedExpertLayerSource(tmp_path, gate, down)
    try:
        assert set(source._descriptors) == {"gate.safetensors", "down.safetensors"}
        expected = {
            "gate.safetensors": source._descriptors["gate.safetensors"],
            "down.safetensors": source._descriptors["down.safetensors"],
        }
        headers = [gate, down] * 128
        with ThreadPoolExecutor(max_workers=8) as pool:
            descriptors = list(pool.map(source._fd, headers))
        assert descriptors == [expected[header.shard] for header in headers]
        assert set(source._descriptors) == set(expected)
    finally:
        source.close()
    assert source._descriptors == {}


def test_conversion_refuses_stale_state_without_run_contract(tmp_path: Path) -> None:
    cells = _cells()
    contract = _run_contract(cells)
    output = tmp_path / "converted"
    stale = output / STATE_DIR_NAME / "staging" / "stale.partial"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"incomplete")

    with pytest.raises(Qwen4IQKConversionError, match="nonempty.*no run contract"):
        _convert(output, cells, contract, _Encoder(), [])
