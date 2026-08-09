"""Dense-tensor IQ_K serving for DeepSeek-V4.

Mirrors the two established install idioms: the manifest-driven module swap
of `runtime/kquant_install.py` (dense tensors bind by `module_weight_key`,
so the swap happens before the shard load and the loader needs no special
casing) and the fail-closed installer discipline of the resident/reference
`runtime/deepseek_v4/iqk_experts.py` implementation (unknown member, unknown
layout, or a geometry mismatch refuses by name at install; a per-call contract
miss delegates to the dequant bridge and is counted, never silent).

The kernel surface is reached through one seam, `_backend_builder`, whose
default implementation consumes the existing `mlx_iqk.nn.IqkSwitchLinear`
at a stack of one expert (a dense tensor is a one-expert stack; the wo_a
grouped projection is a `groups`-expert stack). A member the kernel
repository does not serve refuses at backend build. Tests drive every
route through an injected builder, so the suite passes with the dense
kernel members absent.

The GEMV activation lattice is fp16 (the mlx-iqk contract), which differs
from the q8_0 wire family's bfloat16 lattice; moving the dense side is
math-affecting either way and carries the full quality campaign before any
landing.
"""

from __future__ import annotations

import os
from types import MethodType

import numpy as np

from moespresso.package.iqk_format import (
    IQK_LAYOUT_IQK_RELAYOUT,
    IQK_LAYOUTS,
    iqk_dense_geometry,
    normalize_iqk_layout,
)

# Kill switch for the dense IQ_K decode GEMV family. Default on; `0` routes
# every call through the dequant bridge (counted `delegated`), which is the
# A/B arm switch for the landing campaign and the kill switch after.
_DENSE_QMV_ENV = "MOESPRESSO_DSV4_IQK_DENSE_QMV"

# Dense IQ_K matmul engagement counts by form and site, exported through
# `ssd_streaming_stats` and the speed-stats count keys so served A/B arms
# can prove which composition ran at each seam.
_IQK_DENSE_MATMUL_CALL_COUNTS = {
    "decode_gemv": 0,
    "decode_gemv_wo_b": 0,
    "decode_gemv_lm_head": 0,
    "decode_gemv_w2": 0,
    "prefill_dequant": 0,
    "wo_a_gather": 0,
    "wo_a_bulk": 0,
    "delegated": 0,
}


def iqk_dense_matmul_call_counts() -> dict[str, int]:
    """Return dense IQ_K matmul engagement counts by form."""
    return dict(_IQK_DENSE_MATMUL_CALL_COUNTS)


class IqkDenseInstallError(RuntimeError):
    pass


def iqk_dense_qmv_enabled() -> bool:
    """Kill switch for the dense IQ_K decode GEMV family."""
    return os.environ.get(_DENSE_QMV_ENV, "1") != "0"


# --------------------------------------------------------------------------
# Manifest map


def iqk_dense_weight_map_from_manifest(manifest: dict) -> dict[str, dict]:
    """Return `{module_weight_key: install facts}` for dense IQ_K tensors.

    The key contract mirrors the K-quant module swap: the manifest carries
    the module weight key so loading stays fail-closed and
    architecture-local, and nothing here reparses source tensor names.
    Routed IQ_K expert entries are excluded. The target loader handles them
    through its pooled bundle path; DSpark and explicit reference installs use
    the standalone resident implementation.
    """
    out: dict[str, dict] = {}
    for tensor in manifest.get("tensors", []):
        if tensor.get("format") != "iqk" or tensor.get("kind") == "expert":
            continue
        params = tensor.get("format_params") or {}
        member = params.get("iqk_codec")
        # Raises on routed-only members and unknown names.
        try:
            iqk_dense_geometry(str(member))
        except ValueError as exc:
            raise IqkDenseInstallError(
                f"{tensor.get('source_name')}: {exc}") from exc
        layout = normalize_iqk_layout(params.get("layout"))
        if layout not in IQK_LAYOUTS:
            raise IqkDenseInstallError(
                f"{tensor.get('source_name')}: unknown IQ_K wire layout "
                f"{layout!r}")
        key = tensor.get("module_weight_key")
        if not isinstance(key, str) or not key.endswith(".weight"):
            raise IqkDenseInstallError(
                f"{tensor.get('source_name')}: dense IQ_K manifest entry "
                "must carry module_weight_key ending in '.weight'")
        previous = out.get(key)
        if previous is not None and previous["member"] != member:
            raise IqkDenseInstallError(
                f"{key}: conflicting IQ_K members {previous['member']!r} "
                f"and {member!r}")
        out[key] = {
            "member": str(member),
            "layout": str(layout),
            "source_name": tensor.get("source_name"),
        }
    return out


def manifest_requires_iqk_dense(manifest: dict) -> bool:
    return any(
        tensor.get("format") == "iqk" and tensor.get("kind") != "expert"
        for tensor in manifest.get("tensors", [])
    )


# --------------------------------------------------------------------------
# The backend seam


def _default_backend_builder(member: str, wire_stack, out_features: int,
                             in_features: int):
    """Serving backend over the kernel repository's own stacked module.

    `wire_stack` is uint8 `[num_stacks, out_features, bytes_per_row]`; a
    dense tensor passes one stack, the grouped wo_a passes one per group.
    A member the kernel repository does not serve refuses here by name:
    the caller counts nothing and the install fails closed.
    """
    import mlx.core as mx
    from mlx_iqk import format as iqk_format
    from mlx_iqk.nn import IqkSwitchLinear

    from moespresso.package.iqk_relayout import split_streams

    if member not in iqk_format.MEMBERS:
        raise IqkDenseInstallError(
            f"IQ_K member {member!r} has no serving kernels; the kernel "
            f"repository serves {list(iqk_format.MEMBERS)}")
    stacks = int(wire_stack.shape[0])
    module = IqkSwitchLinear(member, stacks, out_features, in_features)
    streams = split_streams(member, np.asarray(wire_stack), in_features)
    module.load_streams(
        {name: mx.array(value) for name, value in streams.items()})
    mx.eval(*[getattr(module, name) for name in module.stream_names()])
    module.eval()
    return module


# Module-level so a broken environment is one seam wide and tests can
# drive every route without the dense kernel members.
_backend_builder = _default_backend_builder


# --------------------------------------------------------------------------
# The dense module


_DENSE_CLS = None


def _dense_cls():
    """Build the dense module class lazily so importing stays light."""
    global _DENSE_CLS
    if _DENSE_CLS is not None:
        return _DENSE_CLS

    import mlx.core as mx
    from mlx import nn

    class IqkDenseLinear(nn.Module):
        """One dense tensor on IQ_K wire, plain `__call__(x)` seam.

        The `weight` parameter is the packed wire `[out_features,
        bytes_per_row]`, named so the package shard key binds through the
        stock loader. The serving backend builds lazily on first use from
        that wire; decode-shaped calls stream the packed bytes through the
        GEMV, bulk calls dequantize once and take a matmul, and anything
        outside the contract takes the bridge and is counted. Output is
        returned in the activation dtype; the DS4 fp32 seam wrappers own
        the protected seams' contract.
        """

        mode = "iqk_dense"

        def __init__(self, member: str, layout: str, out_features: int,
                     in_features: int):
            super().__init__()
            geometry = iqk_dense_geometry(member)
            if layout != IQK_LAYOUT_IQK_RELAYOUT:
                raise IqkDenseInstallError(
                    f"dense IQ_K layout {layout!r} does not serve; the "
                    "decode kernels read the relayout and would decode the "
                    "quantizer's own wire into the wrong weights rather "
                    "than fail. Rebuild the package with the dense relayout "
                    "step")
            self.member = str(member)
            self.layout = str(layout)
            self.out_features = int(out_features)
            self.in_features = int(in_features)
            self.bytes_per_row = geometry.bytes_per_row(self.in_features)
            self.weight = mx.zeros(
                (self.out_features, self.bytes_per_row), dtype=mx.uint8)
            # Set by the seam installer for per-site engagement evidence.
            self.counter_site = None
            self.freeze()

        # -- backends ----------------------------------------------------

        def _ensure_backend(self):
            backend = getattr(self, "_moespresso_iqk_dense_backend", None)
            if backend is None:
                wire = self.weight
                if tuple(int(v) for v in wire.shape) != (
                        self.out_features, self.bytes_per_row):
                    raise IqkDenseInstallError(
                        f"dense IQ_K wire shape {tuple(wire.shape)} does "
                        f"not match [{self.out_features}, "
                        f"{self.bytes_per_row}] at {self.member}")
                backend = _backend_builder(
                    self.member, wire[None], self.out_features,
                    self.in_features)
                object.__setattr__(
                    self, "_moespresso_iqk_dense_backend", backend)
            return backend

        def _ensure_group_backend(self, groups: int, rank: int):
            backend = getattr(
                self, "_moespresso_iqk_dense_group_backend", None)
            if backend is None:
                if groups * rank != self.out_features:
                    raise IqkDenseInstallError(
                        f"grouped IQ_K projection wants {groups} x {rank} "
                        f"rows but the wire holds {self.out_features}")
                stack = self.weight.reshape(groups, rank, self.bytes_per_row)
                backend = _backend_builder(
                    self.member, stack, rank, self.in_features)
                object.__setattr__(
                    self, "_moespresso_iqk_dense_group_backend", backend)
            return backend

        # -- routes --------------------------------------------------------

        def _count(self, key: str) -> None:
            _IQK_DENSE_MATMUL_CALL_COUNTS[key] += 1
            site = self.counter_site
            if site is not None:
                site_key = f"decode_gemv_{site}"
                if key == "decode_gemv" and site_key in (
                        _IQK_DENSE_MATMUL_CALL_COUNTS):
                    _IQK_DENSE_MATMUL_CALL_COUNTS[site_key] += 1

        def _dequant_matmul(self, backend, x):
            """The bridge: reference-exact fp16 dequant plus a matmul.

            Mirrors the K-quant dequant bridge's role: the always-correct
            composition every other route must match, and the route every
            contract miss falls back to.
            """
            weights = backend.dequantized()[0]
            return mx.matmul(
                x.astype(mx.float32), weights.T.astype(mx.float32),
            ).astype(x.dtype)

        def __call__(self, x):
            rows = 1
            for dim in x.shape[:-1]:
                rows *= int(dim)
            if int(x.shape[-1]) != self.in_features:
                raise IqkDenseInstallError(
                    f"dense IQ_K activation width {int(x.shape[-1])} does "
                    f"not match in_features {self.in_features}")
            # The backend builds (or refuses) before any route is counted,
            # so counters record executed routes only.
            backend = self._ensure_backend()
            if not iqk_dense_qmv_enabled():
                self._count("delegated")
                return self._dequant_matmul(backend, x)
            if rows == 1 and x.dtype in (
                    mx.float32, mx.bfloat16, mx.float16):
                # Decode-shaped route: one activation row streamed through
                # the fused GEMV on the resident packed bytes, expert slot
                # zero of the one-expert stack.
                self._count("decode_gemv")
                row = x.reshape(1, self.in_features)
                index = mx.zeros((1,), dtype=mx.uint32)
                y = backend.gemv(row, index)
                return y.reshape(
                    tuple(int(d) for d in x.shape[:-1]) +
                    (self.out_features,)).astype(x.dtype)
            if rows > 1 and x.dtype in (mx.float32, mx.bfloat16, mx.float16):
                self._count("prefill_dequant")
                return self._dequant_matmul(backend, x)
            self._count("delegated")
            return self._dequant_matmul(backend, x)

    _DENSE_CLS = IqkDenseLinear
    return _DENSE_CLS


# --------------------------------------------------------------------------
# Module install


def _walk_module_path(model, path: str):
    """Resolve `parent, attr` for a dotted module path on the graph."""
    parts = path.split(".")
    obj = model
    for part in parts[:-1]:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj, parts[-1]


def _logical_shape(module) -> tuple[int, int]:
    """Logical `[out, in]` of the skeleton module being replaced.

    An affine QuantizedLinear stores a packed weight, so the reduction
    width comes from its scales; a plain linear carries it directly.
    """
    weight = getattr(module, "weight", None)
    if weight is None:
        raise IqkDenseInstallError(
            f"module {type(module).__name__} carries no weight to size the "
            "IQ_K swap from")
    out_features = int(weight.shape[0])
    scales = getattr(module, "scales", None)
    group_size = getattr(module, "group_size", None)
    if scales is not None and group_size is not None:
        return out_features, int(scales.shape[-1]) * int(group_size)
    return out_features, int(weight.shape[-1])


def install_deepseek_v4_iqk_dense_modules(model, manifest: dict) -> int:
    """Swap manifest-declared dense IQ_K modules onto the graph.

    Runs before the regular-weight load so each module's `weight` key binds
    the packed wire directly. Returns the number of swapped modules.
    """
    weight_map = iqk_dense_weight_map_from_manifest(manifest)
    if not weight_map:
        return 0
    dense_cls = _dense_cls()
    installed = 0
    members: dict[str, int] = {}
    layouts: set[str] = set()
    for key in sorted(weight_map):
        facts = weight_map[key]
        module_path = key[: -len(".weight")]
        parent, attr = _walk_module_path(model, module_path)
        current = getattr(parent, attr)
        out_features, in_features = _logical_shape(current)
        module = dense_cls(
            facts["member"], facts["layout"], out_features, in_features)
        module.eval()
        if parent is not None and str(attr).isdigit():
            parent[int(attr)] = module
        else:
            setattr(parent, attr, module)
        members[facts["member"]] = members.get(facts["member"], 0) + 1
        layouts.add(facts["layout"])
        installed += 1
    object.__setattr__(model, "_moespresso_dsv4_iqk_dense_install", {
        "modules": installed,
        "member_counts": dict(sorted(members.items())),
        "layouts": sorted(layouts),
    })
    return installed


# --------------------------------------------------------------------------
# The DS4 seams


def install_deepseek_v4_iqk_dense_seams(model) -> int:
    """Install the DS4-specific routes over swapped dense IQ_K modules.

    The grouped wo_a projection, the fp32 seam contract at wo_b, and the
    per-site counter labels. Layers whose modules are not IQ_K dense keep
    their stock routes untouched. Returns the number of patched layers.
    """
    import mlx.core as mx
    from mlx import nn

    class _IqkDs4Fp32Dense(nn.Module):
        """The fp32 seam contract over a dense IQ_K wo module.

        A float16 output representation at a quantized dense seam is a
        recorded quality failure; this wrapper casts the protected seams
        to float32 end to end, mirroring the K-quant bridge's contract.
        """

        mode = "iqk_dense"

        def __init__(self, original):
            super().__init__()
            self.original = original
            self.member = getattr(original, "member", None)
            self.freeze()

        def __call__(self, x):
            return self.original(x.astype(mx.float32)).astype(mx.float32)

    def _iqk_grouped_output_projection(self, out):
        wo_a = self.wo_a
        if getattr(wo_a, "mode", None) != "iqk_dense":
            raise IqkDenseInstallError(
                "IQ_K grouped output projection was installed on a "
                "non-IQ_K wo_a")
        bsz, length = out.shape[:2]
        groups = int(self.o_groups)
        rank = int(self.o_lora_rank)
        group_feat = (self.n_heads * self.head_dim) // groups
        if group_feat != wo_a.in_features:
            raise IqkDenseInstallError(
                f"IQ_K wo_a group width {group_feat} does not match the "
                f"wire's in_features {wo_a.in_features}")
        grouped = out.reshape(bsz, length, groups, group_feat)
        rows = int(bsz) * int(length)
        if (
            rows == 1
            and iqk_dense_qmv_enabled()
            and grouped.dtype in (mx.float32, mx.bfloat16, mx.float16)
        ):
            # Gather decode form: all groups as expert slots in one GEMV
            # dispatch. Each group carries its own activation row, so the
            # operand is the down-seam mapping (one row per pair); a
            # broadcast-token operand here would read group 0's row for
            # every group (the pinned IqkSwitchLinear seam defect class).
            # Count after the backend builds: a member the kernel repository
            # does not serve refuses here, and a route counted before that
            # refusal reports engagement no forward ever completed.
            backend = wo_a._ensure_group_backend(groups, rank)
            _IQK_DENSE_MATMUL_CALL_COUNTS["wo_a_gather"] += 1
            x = grouped.reshape(groups, group_feat)
            slots = mx.arange(groups, dtype=mx.uint32)
            y = backend.gemv(x, slots)
            # [groups, 1, rank] -> last-axis concatenation in group order,
            # the same order as the per-group loop.
            y = y.reshape(1, 1, groups * rank).astype(mx.float32)
            return y
        backend = wo_a._ensure_group_backend(groups, rank)
        _IQK_DENSE_MATMUL_CALL_COUNTS["wo_a_bulk"] += 1
        weights = backend.dequantized()  # [groups, rank, group_feat] fp16
        x = grouped.astype(mx.float32).transpose(2, 0, 1, 3)
        # The rhs batch axis must align with the lhs group axis; without
        # the inserted axis the two batch shapes broadcast against each
        # other and every group multiplies every group's weights.
        w = weights.swapaxes(-1, -2).astype(mx.float32)[:, None]
        y = mx.matmul(x, w)
        return y.transpose(1, 2, 0, 3).reshape(bsz, length, groups * rank)

    patched = 0
    layers = getattr(getattr(model, "model", None), "layers", ())
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        layer_patched = False
        wo_a = getattr(attn, "wo_a", None)
        if getattr(wo_a, "mode", None) == "iqk_dense":
            object.__setattr__(
                attn,
                "_grouped_output_projection",
                MethodType(_iqk_grouped_output_projection, attn),
            )
            layer_patched = True
        wo_b = getattr(attn, "wo_b", None)
        if (
            getattr(wo_b, "mode", None) == "iqk_dense"
            and not getattr(wo_b, "_moespresso_dsv4_iqk_fp32_dense", False)
        ):
            object.__setattr__(wo_b, "counter_site", "wo_b")
            wrapped = _IqkDs4Fp32Dense(wo_b)
            object.__setattr__(
                wrapped, "_moespresso_dsv4_iqk_fp32_dense", True)
            object.__setattr__(attn, "wo_b", wrapped)
            layer_patched = True
        shared = getattr(getattr(layer, "mlp", None), "shared_experts", None)
        down = getattr(shared, "down_proj", None)
        if getattr(down, "mode", None) == "iqk_dense":
            object.__setattr__(down, "counter_site", "w2")
        if layer_patched:
            patched += 1
    lm_head = getattr(model, "lm_head", None)
    if getattr(lm_head, "mode", None) == "iqk_dense":
        object.__setattr__(lm_head, "counter_site", "lm_head")
    object.__setattr__(
        model, "_moespresso_dsv4_iqk_dense_seam_layers", patched)
    return patched


def patch_deepseek_v4_iqk_dense_lm_head(model) -> bool:
    """Teach DS4's fp32 logits path about a dense IQ_K lm_head.

    Structurally the K-quant lm_head patch keyed on the IQ_K dense mode:
    generation prefill slices to the newest position before the head (the
    sampler consumes only that row, and the slice skips a [L, vocab] fp32
    matmul per prefill chunk), the surviving decode row takes the dense
    GEMV, and the cacheless scorer paths keep every row on the bridge.
    """
    lm_head = getattr(model, "lm_head", None)
    if getattr(lm_head, "mode", None) != "iqk_dense":
        object.__setattr__(model, "_moespresso_dsv4_iqk_dense_lm_head", False)
        return False
    # The patch is the authoritative site for the head's counter label, so
    # a head that never went through the layer seam installer still counts
    # per site.
    object.__setattr__(lm_head, "counter_site", "lm_head")
    import mlx.core as mx

    cls = type(model)
    original_call = getattr(cls, "_moespresso_original_call", cls.__call__)
    if not hasattr(cls, "_moespresso_original_call"):
        setattr(cls, "_moespresso_original_call", original_call)

    def _iqk_dense_lm_head_call(self, input_ids, cache=None, mask=None):
        if getattr(getattr(self, "lm_head", None), "mode", None) != (
                "iqk_dense"):
            return original_call(self, input_ids, cache=cache, mask=mask)
        h = self.model(input_ids, cache=cache, mask=mask)
        if cache is not None and int(h.shape[1]) > 1:
            h = h[:, -1:, :]
        return self.lm_head(h.astype(mx.float32)).astype(mx.float32)

    setattr(cls, "__call__", _iqk_dense_lm_head_call)
    object.__setattr__(model, "_moespresso_dsv4_iqk_dense_lm_head", True)
    return True


def iqk_dense_engagement(model) -> dict:
    """Install facts plus route counters for the served-path probes."""
    return {
        "install": getattr(model, "_moespresso_dsv4_iqk_dense_install", None),
        "seam_layers": int(getattr(
            model, "_moespresso_dsv4_iqk_dense_seam_layers", 0) or 0),
        "lm_head": bool(getattr(
            model, "_moespresso_dsv4_iqk_dense_lm_head", False)),
        "counts": iqk_dense_matmul_call_counts(),
    }
