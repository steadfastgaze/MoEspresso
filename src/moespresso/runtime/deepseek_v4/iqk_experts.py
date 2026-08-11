"""Fully resident IQ_K switches for DeepSeek-V4 reference and draft graphs.

Reads each layer's expert bundle through the expert index, splits every
projection's opaque `blocks` component into the relayout streams the decode
kernels take, and swaps the layer's switch module for
:class:`IqkDeepseekV4SwitchGLU`. Routing, the shared expert, hyper
connections, and the weighted sum stay on the graph's own MoE block, and the
layer's own clamped-SwiGLU activation module travels into the new switch
unchanged.

The target package loader installs IQ_K experts through the codec-aware pooled
path. A pool that holds all experts is its optimized full-resident subcase;
smaller pools use the same graph seam with on-demand loads. This module keeps
the standalone resident implementation used by DSpark sidecars and explicit
reference installs. Both implementations consume the same relayout streams
and kernel contract.

Only `iqk_relayout` bundles serve. A package still on the quantizer's own
`ik_wire` layout is refused by name: the kernels read a k-contiguous stream
and would decode the wire byte-for-byte into the wrong weights rather than
fail, so the layout is a gate and not a hint.

The per-(layer, projection) member is whatever the allocation assigned, so
one layer mixing members across its projections (`IQ2_KS`, `IQ2_K`,
`IQ1_S_R4`) is a normal layer here and not a special case.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from moespresso.package.bundle import IQK_CODEC
from moespresso.package.iqk_format import (
    IQK_LAYOUT_IQK_RELAYOUT,
    normalize_iqk_layout,
)
from moespresso.package.iqk_relayout import split_streams
from moespresso.runtime.expert_index import ExpertIndex

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

# Routed token-expert pairs at or above which the sorted dequantized route
# serves. Below it the fused GEMV streams the packed bytes instead.
#
# The two routes cost very different things at this geometry. The GEMV reads
# one expert row block per pair, 2.30 MB at both routed shapes. The
# dequantized route pays a fixed 4.9 GB per projection to materialize all 256
# experts as fp16 and then reads the selected ones back, about 9.2 GB in
# total whatever the pair count is. They cross near four thousand pairs,
# which is where this default sits. The stock switch seam crosses at 64
# pairs, a number set by an eight-expert layer; at 256 experts that would put
# every short prompt on a route that decodes 250 experts nothing routed to.
_DEFAULT_SORTED_PREFILL_MIN_PAIRS = 4096
_SORT_PAIRS_ENV = "MOESPRESSO_DSV4_IQK_SORT_PAIRS"

# Mid-token commit cadence shared by the standalone resident switch and the
# full-resident pooled IQ_K route. Both otherwise build the whole token graph
# before the generator's commit, leaving host graph construction serial with
# GPU work. A cadence of k submits one async evaluation after every kth routed
# layer. The bounded ring path already kicks each block and needs no additional
# commit. Default 4; 0 disables the cadence.
_DECODE_FLUSH_ENV = "MOESPRESSO_DSV4_IQK_DECODE_FLUSH_LAYERS"
_DECODE_FLUSH_DEFAULT = 4

# Out-feature split of the sorted route's dequantized projections. The
# unsplit route materializes one full fp16 expert stack per projection
# (4.0 GiB at both routed geometries), and MLX allocates kernel outputs at
# encode while freeing them at command-buffer completion, so all three
# stacks of a layer are in flight at once: a measured 13.07 GiB transient
# per sorted call at a 3,844-token prompt, most of the per-request
# working set. Splitting each projection into `parts` out-row ranges runs
# the same dot products per output element (served text bit-identical at
# every measured setting) while capping each buffer at 4.0/parts GiB.
# Served A/B at the 3,844-token anchor: parts 16 cuts the request working
# set 22.19 to 14.17 GiB with decode unchanged (26.58-26.61 against a
# 26.58 baseline) and prefill inside the run-to-run band (255.5-257.7
# against 258.1-264.3 across same-session arms); parts 32 reaches the
# serve path's no-dequant floor (11.64 GiB) but prices prefill at -5.6
# percent, so it stays opt-in. Default 16; 1 is the kill switch (unsplit).
_SORT_NSPLIT_ENV = "MOESPRESSO_DSV4_IQK_SORT_NSPLIT"
_SORT_NSPLIT_DEFAULT = 16

# Verify-shape commit for the speculative multi-token verify forward. A
# spec round's verify forward carries 2..6 rows, so the single-token
# decode commit above never fires inside a round and the whole verify
# graph is built host-serial before the acceptance eval; the serial build
# measures ~12.8 ms per round at a 3,844-token anchor prompt. Wrapped
# blocks therefore also commit multi-token forwards of at most
# `_VERIFY_FLUSH_MAX_ROWS` rows (prefill chunks are far wider and never
# commit). The commit is scheduling only: outputs are exactly the wrapped
# block's. A served A/B at the anchor measured it worth 10.8 ms of round
# wall median at a fixed five-token draft schedule (117.6 against 128.4)
# with token identity on every arm, so the commit is structural on the
# wrapper; cadence 0 remains the kill switch for the whole wrapper.
_VERIFY_FLUSH_MAX_ROWS = 8


class IqkInstallError(RuntimeError):
    pass


def sorted_prefill_min_pairs() -> int:
    """Pair count at which the sorted dequantized route takes over.

    Read per call so a study can move it without a reinstall. A value that is
    not a positive integer is refused rather than silently ignored.
    """
    raw = os.environ.get(_SORT_PAIRS_ENV)
    if raw is None or raw == "":
        return _DEFAULT_SORTED_PREFILL_MIN_PAIRS
    try:
        value = int(raw)
    except ValueError as exc:
        raise IqkInstallError(
            f"{_SORT_PAIRS_ENV}={raw!r} is not an integer") from exc
    if value < 1:
        raise IqkInstallError(f"{_SORT_PAIRS_ENV}={raw!r} is not positive")
    return value


# Process default for the sorted-route split, raised to 32 by the
# bundled-drafter capacity policy when the drafter fits only at the
# parts-32 working-set floor (drafter_policy). The environment variable
# always wins over the process default.
_SORT_NSPLIT_PROCESS_DEFAULT = _SORT_NSPLIT_DEFAULT


def _validated_nsplit(value: int, source: str) -> int:
    if value < 1 or value & (value - 1):
        raise IqkInstallError(
            f"{source}={value!r} is not a positive power of two")
    return value


def set_sorted_prefill_default(parts: int) -> None:
    """Set the process default for the sorted-route split parts.

    The capacity policy raises the default to 32 when the bundled drafter
    fits only at the parts-32 working-set floor; an explicit
    ``MOESPRESSO_DSV4_IQK_SORT_NSPLIT`` still wins on every read.
    """
    global _SORT_NSPLIT_PROCESS_DEFAULT
    _SORT_NSPLIT_PROCESS_DEFAULT = _validated_nsplit(
        int(parts), "sorted prefill default")


def sorted_prefill_nsplit() -> int:
    """Out-feature parts per projection on the sorted route; 1 is unsplit.

    Read per call so a study can move it without a reinstall. Only positive
    powers of two divide the routed out widths into range-kernel rows, so
    anything else is refused rather than silently ignored.
    """
    raw = os.environ.get(_SORT_NSPLIT_ENV)
    if raw is None or raw == "":
        return _SORT_NSPLIT_PROCESS_DEFAULT
    try:
        value = int(raw)
    except ValueError as exc:
        raise IqkInstallError(
            f"{_SORT_NSPLIT_ENV}={raw!r} is not an integer") from exc
    return _validated_nsplit(value, f"{_SORT_NSPLIT_ENV}")


def iqk_decode_flush_layers() -> int:
    """Mid-token commit cadence for the IQ_K decode path; 0 disables.

    A value that is not a non-negative integer is refused rather than
    silently ignored.
    """
    raw = os.environ.get(_DECODE_FLUSH_ENV)
    if raw is None or raw == "":
        return _DECODE_FLUSH_DEFAULT
    try:
        value = int(raw)
    except ValueError as exc:
        raise IqkInstallError(
            f"{_DECODE_FLUSH_ENV}={raw!r} is not an integer") from exc
    if value < 0:
        raise IqkInstallError(f"{_DECODE_FLUSH_ENV}={raw!r} is negative")
    return value


# --------------------------------------------------------------------------
# The switch module


_SWITCH_CLS = None


def iqk_switch_class():
    """Build the switch class lazily so importing this module stays light.

    Public because the drafter loader installs the same switch on the draft
    stages: one class means one seam contract and one set of route counters
    for the trunk and the draft tree.
    """
    global _SWITCH_CLS
    if _SWITCH_CLS is not None:
        return _SWITCH_CLS

    import mlx.core as mx
    from mlx import nn

    class IqkDeepseekV4SwitchGLU(nn.Module):
        """SwitchGLU seam over three IQ_K projections.

        The forward matches `mlx_lm.models.switch_layers.SwitchGLU.__call__`:
        it takes router indices `[..., top_k]` and returns
        `[..., top_k, out_features]`, so the graph's own MoE block consumes
        it unchanged. Two differences from the stock seam, both of them
        consequences of a 256-expert layer:

        - the sorted route engages on the pair count this geometry actually
          crosses at, not on the stock eight-expert number;
        - the sorted route meets the decoded weights at fp16. Meeting
          bfloat16 activations against fp16 weights promotes the matmul to
          float32, which materializes a float32 copy of the whole decoded
          expert stack per projection.

        The sorted route's gathered row axis is materialized by a reshape of
        a contiguous gather. MLX's sorted `gather_mm` reads a stale
        leading-dimension stride from an axis-inserted view and returns the
        first row's data for every later row.
        """

        def __init__(self, *, gate_proj, up_proj, down_proj, activation, layer):
            super().__init__()
            self.gate_proj = gate_proj
            self.up_proj = up_proj
            self.down_proj = down_proj
            self.activation = activation
            self.layer = int(layer)
            self.input_dims = int(gate_proj.in_features)
            self.hidden_dims = int(gate_proj.out_features)
            self.members = {
                "gate_proj": gate_proj.member,
                "up_proj": up_proj.member,
                "down_proj": down_proj.member,
            }
            # Engagement counters, read by the served-path probes.
            self.total_calls = 0
            self.gemv_calls = 0
            self.sorted_prefill_calls = 0
            self.gemv_pairs = 0
            self.sorted_prefill_pairs = 0
            self.sorted_nsplit_calls = 0
            self.sorted_nsplit_parts = 0

        def __call__(self, x, indices) -> mx.array:
            pairs = int(indices.size)
            self.total_calls += 1
            if pairs < sorted_prefill_min_pairs():
                self.gemv_calls += 1
                self.gemv_pairs += pairs
                return self._call_gemv(x, indices)
            self.sorted_prefill_calls += 1
            self.sorted_prefill_pairs += pairs
            return self._call_sorted(x, indices)

        def _call_gemv(self, x, indices) -> mx.array:
            """Stream the packed bytes; every pair reads its own expert rows."""
            up = self.up_proj.gemv(x, indices)
            gate = self.gate_proj.gemv(x, indices)
            act = self.activation(up, gate)
            return self.down_proj.gemv(act, indices).squeeze(-2)

        def _call_sorted(self, x, indices) -> mx.array:
            """Sort the pairs once, then read each expert's weights once."""
            top_k = int(indices.shape[-1])
            flat = indices.reshape(-1)
            order = mx.argsort(flat)
            sorted_ids = flat[order]
            rows = int(flat.size)
            gathered = x.reshape(-1, self.input_dims)[order // top_k]
            xg = gathered.reshape(rows, 1, self.input_dims).astype(mx.float16)

            parts = sorted_prefill_nsplit()
            if parts > 1:
                self.sorted_nsplit_calls += 1
                self.sorted_nsplit_parts = parts
                up = self._sorted_split(self.up_proj, xg, sorted_ids, parts)
                gate = self._sorted_split(self.gate_proj, xg, sorted_ids,
                                          parts)
                act = self.activation(up, gate)
                down = self._sorted_split(self.down_proj, act, sorted_ids,
                                          parts)
            else:
                up = self.up_proj(xg, sorted_ids, sorted_indices=True)
                gate = self.gate_proj(xg, sorted_ids, sorted_indices=True)
                act = self.activation(up, gate)
                down = self.down_proj(act, sorted_ids, sorted_indices=True)
            out = down.reshape(rows, -1)[mx.argsort(order)]
            return mx.unflatten(out, 0, tuple(indices.shape))

        @staticmethod
        def _sorted_split(proj, operand, sorted_ids, parts: int) -> mx.array:
            """The sorted projection as ``parts`` out-row-range matmuls.

            Each output element is the same single dot product the unsplit
            call computes; the ranges only cap the size of the fp16 dequant
            temporaries in flight. Range outputs concatenate along the
            feature axis in range order.
            """
            step = proj.out_features // parts
            return mx.concatenate(
                [proj.sorted_matmul_range(operand, sorted_ids, i * step, step)
                 for i in range(parts)],
                axis=-1)

    _SWITCH_CLS = IqkDeepseekV4SwitchGLU
    return _SWITCH_CLS


# --------------------------------------------------------------------------
# The mid-token decode commit


_FLUSH_CLS = None


def _flush_cls():
    """Build the flush wrapper lazily so importing this module stays light."""
    global _FLUSH_CLS
    if _FLUSH_CLS is not None:
        return _FLUSH_CLS

    import mlx.core as mx
    from mlx import nn

    class IqkDecodeFlushMoEBlock(nn.Module):
        """MoE block wrapper that commits the queued graph after its output.

        Delegates every call to the wrapped block and, on single-token
        decode calls and verify-shaped multi-token calls (at most
        ``_VERIFY_FLUSH_MAX_ROWS`` rows, the speculative verify width),
        hands the block output to ``mx.async_eval``. The commit is
        asynchronous and schedules everything queued so far, so the
        output values are exactly the wrapped block's; only the encode
        schedule changes. Prefill forwards pass through untouched.
        ``iqk_decode_flush_calls`` and ``iqk_verify_flush_calls`` count
        the commits as engagement evidence. The pass-through properties
        keep the wrapped block's introspection seams (`switch_mlp`,
        `gate`, `shared_experts`) readable through the wrapper.
        """

        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.iqk_decode_flush_calls = 0
            self.iqk_verify_flush_calls = 0

        @property
        def switch_mlp(self):
            return self.inner.switch_mlp

        @property
        def gate(self):
            return self.inner.gate

        @property
        def shared_experts(self):
            return self.inner.shared_experts

        def __call__(self, x, input_ids=None):
            y = self.inner(x, input_ids=input_ids)
            rows = 1
            for dim in x.shape[:-1]:
                rows *= int(dim)
            if rows == 1:
                mx.async_eval(y)
                self.iqk_decode_flush_calls += 1
            elif rows <= _VERIFY_FLUSH_MAX_ROWS:
                mx.async_eval(y)
                self.iqk_verify_flush_calls += 1
            return y

    _FLUSH_CLS = IqkDecodeFlushMoEBlock
    return _FLUSH_CLS


def install_iqk_decode_flush(model) -> int:
    """Wrap every Nth IQ_K MoE block with the mid-token commit.

    Cadence N comes from ``MOESPRESSO_DSV4_IQK_DECODE_FLUSH_LAYERS``,
    default 4; 0 wraps nothing, which leaves the decode graph committed
    only by the generator. Only layers whose switch seam is the installed
    IQ_K switch consume an ordinal, and blocks at positions where
    ``(ordinal + 1) % N == 0`` are wrapped, so the cadence paces exactly
    the layers whose decode dispatches it commits. Returns the number of
    blocks wrapped; every wrapped block is put in eval mode at install.
    """
    existing = getattr(model, "_moespresso_dsv4_iqk_decode_flush", None)
    if existing is not None:
        return int(existing.get("wrapped_blocks", 0))
    cadence = iqk_decode_flush_layers()
    if cadence < 1:
        return 0
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return 0
    flush_cls = _flush_cls()
    wrapped = 0
    ordinal = 0
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        switch = getattr(mlp, "switch_mlp", None)
        if switch is None or type(switch).__name__ != "IqkDeepseekV4SwitchGLU":
            continue
        if (ordinal + 1) % cadence == 0:
            block = flush_cls(mlp)
            block.eval()
            layer.mlp = block
            wrapped += 1
        ordinal += 1
    object.__setattr__(model, "_moespresso_dsv4_iqk_decode_flush", {
        "cadence": cadence,
        "wrapped_blocks": wrapped,
    })
    return wrapped


def iqk_decode_flush_blocks(model) -> list:
    """Every installed flush wrapper, in layer order."""
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "model", None), "layers", None)
    return [
        layer.mlp for layer in layers or []
        if type(getattr(layer, "mlp", None)).__name__ == "IqkDecodeFlushMoEBlock"
    ]


# --------------------------------------------------------------------------
# Reading the bundle


def _projection_blocks(
    handle,
    index: ExpertIndex,
    layer: int,
    projection: str,
) -> np.ndarray:
    """Every expert's `blocks` component for one projection, stacked.

    Read with `pread` rather than through a mapping. A routed IQ_K stack is
    the whole package, and a mapping held across the install leaves every
    byte of it in the page cache next to the device arrays built from it,
    which doubles the standing footprint at exactly the moment residency is
    at its peak.
    """
    first = index.locate(layer=layer, expert=0, projection=projection,
                         component="blocks")
    num_experts = index.num_experts
    row_bytes = index.row_bytes(layer=layer)
    base = index.locate_row(layer=layer, expert=0).offset
    comp_offset = first.offset - base
    out = np.empty((num_experts, first.nbytes), dtype=np.uint8)
    view = memoryview(out).cast("B")
    for expert in range(num_experts):
        start = base + expert * row_bytes + comp_offset
        got = os.preadv(handle, [view[expert * first.nbytes:
                                      (expert + 1) * first.nbytes]], start)
        if got != first.nbytes:
            raise IqkInstallError(
                f"layer {layer} {projection}: expert {expert} short read "
                f"({got} of {first.nbytes} B)")
    return out.reshape((num_experts, *first.shape))


def _layer_geometry(index: ExpertIndex, layer: int) -> dict:
    """Validated per-projection facts for one IQ_K layer bundle."""
    out = {}
    for projection in PROJECTIONS:
        geometry = index.geometry(layer=layer, projection=projection)
        if geometry.codec != IQK_CODEC:
            raise IqkInstallError(
                f"layer {layer} {projection}: codec {geometry.codec!r} is not "
                f"{IQK_CODEC!r}; the IQ_K installer serves an all-IQ_K layer")
        if normalize_iqk_layout(geometry.layout) != IQK_LAYOUT_IQK_RELAYOUT:
            raise IqkInstallError(
                f"layer {layer} {projection}: bundle layout {geometry.layout!r} "
                f"is not {IQK_LAYOUT_IQK_RELAYOUT!r}; the decode kernels read the "
                "relayout and would decode the quantizer's own wire into the "
                "wrong weights rather than fail. Rebuild the package with "
                "moespresso-ds4-iqk-relayout")
        if geometry.in_features is None or geometry.iqk_codec is None:
            raise IqkInstallError(
                f"layer {layer} {projection}: bundle records no IQ_K member or "
                "input width")
        out[projection] = geometry
    return out


# --------------------------------------------------------------------------
# The install


def manifest_requires_iqk_experts(manifest: dict) -> bool:
    return any(
        tensor.get("format") == IQK_CODEC and tensor.get("kind") == "expert"
        for tensor in manifest.get("tensors", [])
    )


def _layer_switch_seam(model, layer: int):
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or layer >= len(layers):
        raise IqkInstallError(f"model has no layer {layer}")
    mlp = getattr(layers[layer], "mlp", None)
    if mlp is None or not hasattr(mlp, "switch_mlp"):
        raise IqkInstallError(f"layer {layer} has no mlp.switch_mlp seam")
    return mlp


def install_deepseek_v4_iqk_experts(
    model,
    package_dir: str | Path,
    index: ExpertIndex,
) -> int:
    """Install the standalone fully resident IQ_K reference implementation."""
    import mlx.core as mx
    from mlx_iqk.nn import IqkSwitchLinear

    package_dir = Path(package_dir)
    switch_cls = iqk_switch_class()
    installed = 0
    members: dict[str, int] = {}

    for layer in index.layers_indexed():
        geometries = _layer_geometry(index, layer)
        mlp = _layer_switch_seam(model, layer)
        stock = mlp.switch_mlp
        activation = getattr(stock, "activation", None)
        if activation is None:
            raise IqkInstallError(
                f"layer {layer}: the switch seam carries no activation module; "
                "the clamped SwiGLU contract lives there and is not "
                "reconstructed here")
        shard = index.locate_row(layer=layer, expert=0).shard
        handle = os.open(package_dir / shard, os.O_RDONLY)
        try:
            projections = {}
            for projection, geometry in geometries.items():
                module = IqkSwitchLinear(
                    geometry.iqk_codec,
                    index.num_experts,
                    geometry.out_features,
                    geometry.in_features,
                )
                blocks = _projection_blocks(handle, index, layer, projection)
                streams = split_streams(
                    geometry.iqk_codec, blocks, geometry.in_features)
                del blocks
                module.load_streams(
                    {name: mx.array(value) for name, value in streams.items()})
                del streams
                mx.eval(*[getattr(module, name) for name in module.stream_names()])
                module.eval()
                projections[projection] = module
                members[geometry.iqk_codec] = members.get(geometry.iqk_codec, 0) + 1
        finally:
            os.close(handle)
        switch = switch_cls(
            gate_proj=projections["gate_proj"],
            up_proj=projections["up_proj"],
            down_proj=projections["down_proj"],
            activation=activation,
            layer=layer,
        )
        switch.eval()
        mlp.switch_mlp = switch
        mx.clear_cache()
        installed += 1

    object.__setattr__(model, "_moespresso_dsv4_iqk_install", {
        "layers": list(index.layers_indexed()),
        "layers_installed": installed,
        "num_experts": index.num_experts,
        "layout": IQK_LAYOUT_IQK_RELAYOUT,
        "member_counts": dict(sorted(members.items())),
        "sorted_prefill_min_pairs": sorted_prefill_min_pairs(),
        "sorted_prefill_nsplit": sorted_prefill_nsplit(),
    })
    return installed


def iqk_switch_modules(model) -> list:
    """Every resident or pooled IQ_K switch, deepest layer last."""
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "model", None), "layers", None)
    out = []
    for layer in layers or []:
        switch = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
        if switch is not None and (
            switch.__class__.__name__ == "IqkDeepseekV4SwitchGLU"
            or bool(getattr(switch, "_all_iqk", False))
        ):
            out.append(switch)
    return out


def iqk_engagement(model) -> dict:
    """Route counters plus the kernels actually compiled in this process."""
    from moespresso.runtime.deepseek_v4.iqk_decode_kernel import (
        built_dual_gemv_kernels,
    )
    from mlx_iqk.kernels import (
        built_dequant_kernels,
        built_dequant_range_kernels,
        built_gemv_kernels,
    )

    switches = iqk_switch_modules(model)
    flush_blocks = iqk_decode_flush_blocks(model)
    drafter_policy = getattr(model, "_moespresso_ds4_drafter_policy", None)
    policy_mode = (drafter_policy or {}).get("mode")
    policy_decision = (drafter_policy or {}).get("decision")
    return {
        "install": getattr(model, "_moespresso_dsv4_iqk_install", None),
        "switch_modules": len(switches),
        "layers": [s.layer for s in switches],
        "total_calls": sum(s.total_calls for s in switches),
        "gemv_calls": sum(s.gemv_calls for s in switches),
        "gemv_pairs": sum(s.gemv_pairs for s in switches),
        "sorted_prefill_calls": sum(s.sorted_prefill_calls for s in switches),
        "sorted_prefill_pairs": sum(s.sorted_prefill_pairs for s in switches),
        "sorted_nsplit_calls": sum(s.sorted_nsplit_calls for s in switches),
        "sorted_nsplit_parts": max(
            (s.sorted_nsplit_parts for s in switches), default=0),
        "layers_without_a_gemv_call": [
            s.layer for s in switches if s.gemv_calls == 0],
        "decode_flush": getattr(
            model, "_moespresso_dsv4_iqk_decode_flush", None),
        "iqk_decode_flush_calls": (
            sum(b.iqk_decode_flush_calls for b in flush_blocks)
            + sum(int(getattr(s, "iqk_decode_flush_calls", 0)) for s in switches)
        ),
        "iqk_verify_flush_calls": (
            sum(b.iqk_verify_flush_calls for b in flush_blocks)
            + sum(int(getattr(s, "iqk_verify_flush_calls", 0)) for s in switches)
        ),
        "iqk_dual_gemv_calls": sum(
            int(getattr(s, "iqk_dual_gemv_calls", 0)) for s in switches
        ),
        "iqk_dual_gemv_pairs": sum(
            int(getattr(s, "iqk_dual_gemv_pairs", 0)) for s in switches
        ),
        "drafter_policy": drafter_policy,
        "ds4_drafter_policy_auto_on": int(
            policy_mode == "auto" and policy_decision == "on"),
        "ds4_drafter_policy_auto_off": int(
            policy_mode == "auto" and policy_decision == "off"),
        "ds4_drafter_policy_override": int(policy_mode == "override"),
        "built_gemv_kernels": [list(k) for k in built_gemv_kernels()],
        "built_dequant_kernels": [list(k) for k in built_dequant_kernels()],
        "built_dequant_range_kernels": [
            list(k) for k in built_dequant_range_kernels()],
        "built_iqk_dual_gemv_kernels": [
            list(k) for k in built_dual_gemv_kernels()],
        # The same registry as an integer, so the census surfaces that hold
        # only counts can carry it and the phase splitter can subtract it.
        "built_dequant_range_kernel_count": len(built_dequant_range_kernels()),
        "built_iqk_dual_gemv_kernel_count": len(built_dual_gemv_kernels()),
    }
