"""Sorted K-quant routed-expert path for the resident qwen3_5_moe MoE block.

The stock mlx_lm SwitchGLU runs the top-k routed experts as unsorted gather
matmuls: `gather_qmm` lowers to one vector-matmul per token-expert pair, so
every selected expert's quantized weights are re-read once per assigned token.
On a long prefill that is the dominant cost (the attribution ledger prices the
`switch_experts` stage at roughly two thirds of the 4K prefill).

This module replaces that seam with the DS4-proven sorted route: at prefill the
routed token-expert pairs sort by expert id on device, one combined gate/up GEMM
and one down GEMM run through `mlx_kquant.gather_qmm_sorted` (which derives each
expert's contiguous row range in-kernel from the sorted ids, so every expert's
weights are read once for its whole segment), and the rows scatter back to the
caller's order. On the fused kernel the combined gate/up GEMM and the SwiGLU
collapse into `gather_qmm_sorted_swiglu`, whose epilogue applies the activation on
the float32 accumulators so the wide intermediate is never materialized.

The experts are already fully resident on the resident qwen K-quant path (every
expert stack is installed at build time), so this owns the resident stacks
directly. There is no expert pool, no residency streaming, and no on-device slot
remap: the routed ids are the stack row ids. Decode (single-token) and any small
or off-contract call fall back to the stock unsorted gather, which stays correct
for every package shape.

The sorted route changes the floating-point accumulation order (per-expert
segment GEMMs instead of per-pair matvecs) and, with the fused kernel, skips the
intermediate rounding to the row dtype, so it is math-affecting rather than
bit-identical. The campaign quality ladder judges this route; primitive-level
token identity is not required.

Kill switch: `MOESPRESSO_QWEN_MOE_SORTED=0` restores the stock unsorted SwitchGLU
seam for the whole process. Default on. Eligibility fails closed to the stock
gather on any contract mismatch (non-K-quant projections, mismatched gate/up
codec or geometry, a kernel-less mlx_kquant).
"""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn

# Sorted routed MoE kill switch (family style). Default on; set to 0 to restore
# the stock unsorted SwitchGLU gather for the whole process.
_QWEN_MOE_SORTED = os.environ.get("MOESPRESSO_QWEN_MOE_SORTED", "1") != "0"

# Fused gate/up + SwiGLU: when mlx_kquant ships gather_qmm_sorted_swiglu, the
# combined gate/up GEMM and the elementwise activation collapse into one kernel.
# The fused epilogue skips the intermediate store in the row dtype, giving
# numerical equivalence to the unfused sorted pair without bit identity. Default on;
# MOESPRESSO_QWEN_MOE_SORTED_SWIGLU=0 keeps the unfused gather_qmm_sorted plus the
# activation module.
_QWEN_MOE_SORTED_SWIGLU = os.environ.get("MOESPRESSO_QWEN_MOE_SORTED_SWIGLU", "1") != "0"

# Sorted-prefill row threshold. Below this many routed token-expert pairs the
# per-pair gather kernel wins (decode and short prefills), so the route stays on
# the stock unsorted gather. Mirrors the pooled path's segmented-prefill floor.
_SORTED_PREFILL_MIN_ROWS = 4096


class SortedKQuantSwitchGLUError(RuntimeError):
    pass


def sorted_moe_enabled() -> bool:
    """Whether the sorted routed MoE route is enabled for this process."""
    return _QWEN_MOE_SORTED


class SortedKQuantSwitchGLU(nn.Module):
    """SwitchGLU seam that sorts routed pairs and reads each expert once.

    Wraps the resident combined gate/up K-quant stack and the down stack. The
    forward matches `mlx_lm.models.switch_layers.SwitchGLU.__call__`: it takes
    the router indices `[..., top_k]` and returns `[..., top_k, out_features]`,
    so the stock qwen sparse MoE block (router, shared expert, weighted sum)
    consumes it unchanged.
    """

    def __init__(self, *, gate_up, down_proj, activation, gate_out_features):
        super().__init__()
        # gate_up holds the [gate | up] combined stack (gate rows first). down
        # is the stock resident down projection. Both are mlx_kquant modules
        # carrying uint8 wire bytes, a vestigial scales placeholder, and a
        # kquant_type.
        self.gate_up = gate_up
        self.down_proj = down_proj
        self.activation = activation
        self.gate_out_features = int(gate_out_features)

        self.num_experts = int(gate_up.weight.shape[0])
        self.gate_up_type = gate_up.kquant_type
        self.down_type = down_proj.kquant_type
        # in_features of the gate/up projection is the down projection's output
        # width (the residual-stream hidden). Read it from the down stack's
        # output-row count, which is the model hidden size.
        self.in_features = int(down_proj.weight.shape[1])

        # Engagement counters (exported through the runtime stats surface).
        self.total_calls = 0
        self.prefill_calls = 0
        self.decode_calls = 0
        self.sorted_prefill_calls = 0
        self.fused_swiglu_calls = 0
        self.fallback_calls = 0

        # One-shot fail-closed eligibility verdict (None until first decided).
        self._sorted_ready_cached: bool | None = None

    # ---- eligibility --------------------------------------------------

    def _sorted_ready(self) -> bool:
        ready = self._sorted_ready_cached
        if ready is None:
            ready = self._sorted_eligible()
            self._sorted_ready_cached = ready
        return ready

    def _sorted_eligible(self) -> bool:
        if not _QWEN_MOE_SORTED:
            return False
        try:
            import mlx_kquant as kq
        except ImportError:
            return False
        if getattr(kq, "gather_qmm_sorted", None) is None:
            return False
        # The combined stack must hold exactly 2 * gate_out rows per expert
        # (gate rows first, then up), the gather_qmm_sorted_swiglu w layout.
        if int(self.gate_up.weight.shape[1]) != 2 * self.gate_out_features:
            return False
        if int(self.down_proj.weight.shape[0]) != self.num_experts:
            return False
        return True

    def _fused_swiglu_available(self) -> bool:
        if not _QWEN_MOE_SORTED_SWIGLU:
            return False
        import mlx_kquant as kq

        return getattr(kq, "gather_qmm_sorted_swiglu", None) is not None

    def _swiglu_limit(self) -> float:
        return float(getattr(self.activation, "swiglu_limit", 0.0) or 0.0)

    # ---- forward ------------------------------------------------------

    def __call__(self, x, indices) -> mx.array:
        self.total_calls += 1
        rows = int(indices.size)
        token_layers = 1
        for dim in indices.shape[:-1]:
            token_layers *= int(dim)
        if token_layers == 1:
            self.decode_calls += 1
        else:
            self.prefill_calls += 1

        # The sorted route runs on the same inference math as the stock seam,
        # which itself sorts routed pairs at this scale regardless of training
        # mode. Serving never trains this seam; the served model reports
        # training=True without ever running a backward pass, so gating on it
        # would keep the fast route off the whole served path.
        if rows >= _SORTED_PREFILL_MIN_ROWS and self._sorted_ready():
            self.sorted_prefill_calls += 1
            return self._call_sorted(x, indices)

        self.fallback_calls += 1
        return self._call_unsorted(x, indices)

    def _call_sorted(self, x, indices) -> mx.array:
        """Full-resident sorted-ids prefill with no host synchronization.

        Sort the routed token-expert pairs by expert id on device, run the
        combined gate/up GEMM and the down GEMM through the sorted-ids kernel
        (each expert's weights read once for its whole segment), then unsort to
        the caller's `[..., top_k, out_features]` contract. The routed ids are
        the stack row ids (full residency), so no slot remap is needed.
        """
        import mlx_kquant as kq

        top_k = int(indices.shape[-1])
        flat_idx = indices.reshape(-1)
        order = mx.argsort(flat_idx)
        sorted_ids = flat_idx[order]

        x_tokens = x.reshape(-1, self.in_features)
        # Each routed pair reads the token row it belongs to (order // top_k),
        # placed in sorted-expert order.
        x_g = x_tokens[order // top_k]

        gate_n = self.gate_out_features
        if self._fused_swiglu_available():
            self.fused_swiglu_calls += 1
            x_act = kq.gather_qmm_sorted_swiglu(
                x_g,
                self.gate_up.weight,
                self.gate_up.scales,
                self.gate_up_type,
                sorted_ids,
                gate_n,
                self._swiglu_limit(),
            )
        else:
            combined = kq.gather_qmm_sorted(
                x_g,
                self.gate_up.weight,
                self.gate_up.scales,
                self.gate_up_type,
                sorted_ids,
            )
            # mlx_lm swiglu: activation(up, gate) = silu(gate) * up. The combined
            # stack is [gate | up], so gate is the low half, up the high half.
            x_act = self.activation(combined[..., gate_n:], combined[..., :gate_n])

        down = kq.gather_qmm_sorted(
            x_act,
            self.down_proj.weight,
            self.down_proj.scales,
            self.down_type,
            sorted_ids,
        )
        out = down[mx.argsort(order)]
        return mx.unflatten(out, 0, indices.shape)

    def _call_unsorted(self, x, indices) -> mx.array:
        """Stock unsorted gather over the combined gate/up and down stacks.

        The correctness fallback for decode, short prefills, and any off-contract
        call. It mirrors the stock SwitchGLU forward exactly (expand, gather
        gate/up in one combined matmul, activation, down gather), so the routed
        math is the stock per-pair gather kernel.
        """
        import mlx_kquant as kq

        x = mx.expand_dims(x, (-2, -3))
        combined = kq.gather_qmm(
            x,
            self.gate_up.weight,
            self.gate_up.scales,
            self.gate_up_type,
            rhs_indices=indices,
            transpose=True,
            sorted_indices=False,
        )
        gate_n = self.gate_out_features
        x_act = self.activation(combined[..., gate_n:], combined[..., :gate_n])
        out = kq.gather_qmm(
            x_act,
            self.down_proj.weight,
            self.down_proj.scales,
            self.down_type,
            rhs_indices=indices,
            transpose=True,
            sorted_indices=False,
        )
        return out.squeeze(-2)


class _CombinedGateUpKQuant(nn.Module):
    """Holder for the combined [gate | up] K-quant expert stack.

    Carries the uint8 wire bytes, the vestigial scales placeholder, and the
    codec name, so it duck-types with the mlx_kquant switch modules the sorted
    kernels consume. It is never called directly; `SortedKQuantSwitchGLU` reads
    its `weight`/`scales`/`kquant_type`.
    """

    def __init__(self, *, weight, scales, kquant_type):
        super().__init__()
        self.mode = "kquant"
        self.kquant_type = kquant_type
        self.weight = weight
        self.scales = scales
        self.freeze()


def _combine_gate_up_stack(gate_proj, up_proj):
    """Build the combined [gate | up] K-quant stack from two resident stacks.

    The gather_qmm_sorted_swiglu / gather_qmm_sorted w layout is
    (n_experts, 2 * gate_out, bytes_per_row) with each expert's gate rows first
    and its up rows second. The two resident stacks are
    (n_experts, gate_out, bytes_per_row) each with identical geometry, so the
    combined stack concatenates them along the output-row axis. Returns the
    concatenated uint8 array; the caller drops the two singles so residency is
    net neutral.
    """
    gate_w = gate_proj.weight
    up_w = up_proj.weight
    if gate_w.shape != up_w.shape:
        raise SortedKQuantSwitchGLUError(
            f"cannot combine gate/up: shapes differ gate={gate_w.shape} up={up_w.shape}"
        )
    if getattr(gate_proj, "kquant_type", None) != getattr(up_proj, "kquant_type", None):
        raise SortedKQuantSwitchGLUError(
            f"cannot combine gate/up: codecs differ gate={gate_proj.kquant_type!r} "
            f"up={up_proj.kquant_type!r}"
        )
    combined = mx.concatenate([gate_w, up_w], axis=1)
    mx.eval(combined)
    return combined


def _iter_switch_mlps(model):
    """Yield (mlp, switch_mlp) for every layer that carries a routed switch_mlp."""
    for path in (
        ("language_model", "model", "layers"),
        ("model", "layers"),
        ("layers",),
    ):
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            layers = obj
            break
    else:
        return
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        sw = getattr(mlp, "switch_mlp", None)
        if sw is None:
            continue
        yield mlp, sw


def install_sorted_kquant_switchglus(model) -> int:
    """Swap resident K-quant SwitchGLU seams for the sorted routed route.

    Walks every routed layer, builds the combined gate/up stack from the two
    resident K-quant stacks (dropping the singles so residency is net neutral),
    and replaces `switch_mlp` with a `SortedKQuantSwitchGLU`. A layer whose
    gate/up projections are not combinable K-quant stacks is left on its stock
    seam (fail-closed): the sorted route only claims layers it can serve. Returns
    the number of layers swapped. A no-op when the kill switch is off.
    """
    if not _QWEN_MOE_SORTED:
        return 0
    installed = 0
    for mlp, sw in _iter_switch_mlps(model):
        gate = getattr(sw, "gate_proj", None)
        up = getattr(sw, "up_proj", None)
        down = getattr(sw, "down_proj", None)
        if gate is None or up is None or down is None:
            continue
        if not (_is_kquant(gate) and _is_kquant(up) and _is_kquant(down)):
            continue
        try:
            combined_weight = _combine_gate_up_stack(gate, up)
        except SortedKQuantSwitchGLUError:
            continue
        gate_out_features = int(gate.weight.shape[1])
        combined = _CombinedGateUpKQuant(
            weight=combined_weight,
            scales=gate.scales,
            kquant_type=gate.kquant_type,
        )
        setattr(
            mlp,
            "switch_mlp",
            SortedKQuantSwitchGLU(
                gate_up=combined,
                down_proj=down,
                activation=sw.activation,
                gate_out_features=gate_out_features,
            ),
        )
        installed += 1
    return installed


def _is_kquant(module) -> bool:
    return (
        getattr(module, "mode", None) == "kquant"
        and getattr(module, "kquant_type", None) is not None
        and getattr(module, "weight", None) is not None
    )
