"""Cold-start expert hotlist baked into the package from imatrix counts.

The calibration imatrix (already a mandatory convert input with recorded
provenance) carries per-layer routed-expert usage counters (~millions of
routed calibration tokens per layer). Measured against real request demand:
seeding capacity-70 from these counts captures a median 0.40
of a request's expert-demand mass vs 0.27 for arbitrary seeding, with zero
run history. A runtime-saved demand hotlist captures ~0.60 and takes
precedence when present (the layered seeding design); this artifact is the
cold-start floor before any request history exists.

The emitted `expert_hotlist.json` uses the same schema as the saved-demand
hotlists, so `ssd_streaming_build.load_expert_hotlist` consumes either
interchangeably (it caps installed priors so neither can dominate live
traffic; see prior_cap there).

Fail-closed alignment: imatrix counts are keyed by GGUF block index; the
package's routed layers are keyed by model layer index. These coincide on
every artifact observed so far, but a mismatch would silently seed the wrong
layers' experts, so the builder requires exact layer-set equality with the
package's expert index and emits nothing (with a loud reason) otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from moespresso.probe.calibration import imatrix_expert_counts
from moespresso.runtime.expert_index import build_expert_index

HOTLIST_NAME = "expert_hotlist.json"


class HotlistAlignmentError(ValueError):
    """imatrix expert counts do not align with the package's routed layers."""


def build_package_expert_hotlist(
    counts: dict[int, np.ndarray],
    *,
    layers_indexed: tuple[int, ...],
    num_experts: int,
    source: dict | None = None,
) -> dict:
    """Pure: ranked per-layer expert counts -> the expert_hotlist payload.

    Raises HotlistAlignmentError unless the count layers exactly equal the
    package's routed layers and every layer has at least `num_experts`
    counters (a smoke package keeps the first N experts, so counts[:N] are
    the right counters for it).
    """
    if set(counts) != set(layers_indexed):
        raise HotlistAlignmentError(
            f"imatrix count layers {_span(set(counts))} != package routed "
            f"layers {_span(set(layers_indexed))}: refusing to emit a "
            f"possibly misaligned hotlist")
    layers: dict[str, dict[str, int]] = {}
    for layer in sorted(layers_indexed):
        c = counts[layer]
        if len(c) < num_experts:
            raise HotlistAlignmentError(
                f"layer {layer}: {len(c)} expert counters < package "
                f"num_experts {num_experts}")
        c = c[:num_experts]
        layers[str(layer)] = {
            str(expert): int(c[expert])
            for expert in np.argsort(c)[::-1].tolist()
            if c[expert] > 0
        }
    return {
        "version": 1,
        "kind": "expert_hotlist",
        "source": {"kind": "gguf_imatrix_counts", **(source or {})},
        "layers": layers,
    }


def _span(layers: set[int]) -> str:
    return f"[{min(layers)}..{max(layers)}] (n={len(layers)})" if layers else "[]"


def write_package_expert_hotlist(
    package_dir: str | Path,
    imatrix_path: str | Path,
    *,
    imatrix_identity: dict | None = None,
) -> int:
    """Read imatrix counts, align against the package index, write the hotlist.

    Returns the number of layers written; 0 (nothing written) when the
    imatrix has no expert counts (dense model) or the package has no routed
    experts. Alignment failures raise; the convert caller logs and proceeds
    without a hotlist rather than shipping a wrong one.
    """
    package_dir = Path(package_dir)
    counts = imatrix_expert_counts(imatrix_path)
    if not counts:
        return 0
    try:
        index = build_expert_index(package_dir)
    except ValueError:
        return 0  # dense package: no routed experts, no hotlist
    source = {"imatrix_name": Path(imatrix_path).name}
    if imatrix_identity:
        source["imatrix_sha256"] = imatrix_identity.get("sha256")
    payload = build_package_expert_hotlist(
        counts,
        layers_indexed=index.layers_indexed(),
        num_experts=index.num_experts,
        source=source,
    )
    (package_dir / HOTLIST_NAME).write_text(json.dumps(payload))
    return len(payload["layers"])


def write_package_expert_hotlist_from_payload(
    package_dir: str | Path,
    payload: dict,
) -> int:
    """Write a pre-extracted hotlist payload, re-validated against the package.

    For model families whose build imatrix carries no per-expert counts (the
    legacy .dat format stores a single call counter per tensor), the ranking
    is extracted once from a counts-bearing imatrix and committed as package
    data; this writer installs that payload. The payload's per-layer counts
    are rebuilt into count vectors and pushed through the same alignment
    contract as the imatrix path (exact layer-set equality, expert slicing
    for smoke packages), so a payload extracted from a different layer layout
    raises instead of seeding the wrong layers. Returns the number of layers
    written; 0 when the payload is empty or the package has no routed experts.
    """
    package_dir = Path(package_dir)
    layers = payload.get("layers") or {}
    if not layers:
        return 0
    try:
        index = build_expert_index(package_dir)
    except ValueError:
        return 0  # dense package: no routed experts, no hotlist
    counts: dict[int, np.ndarray] = {}
    for key, ranked in layers.items():
        size = index.num_experts
        if ranked:
            size = max(size, max(int(e) for e in ranked) + 1)
        c = np.zeros(size, dtype=np.float32)
        for expert, count in ranked.items():
            c[int(expert)] = float(count)
        counts[int(key)] = c
    out = build_package_expert_hotlist(
        counts,
        layers_indexed=index.layers_indexed(),
        num_experts=index.num_experts,
        source=payload.get("source"),
    )
    (package_dir / HOTLIST_NAME).write_text(json.dumps(out))
    return len(out["layers"])
