"""Explicit synthetic allocations for manifest and package-boundary tests."""

from moespresso.core.artifact import make_artifact

AFFINE_BITS = (4,)
EXPERT_BITS = (4,)


def synthetic_decision(evidence):
    """Build fixed test allocations without invoking conversion or calibration."""
    allocation = []
    for unit in evidence["units"]:
        row = {
            key: value
            for key, value in unit.items()
            if key
            in {"source_name", "role", "kind", "layer_index", "projection", "n_experts", "shape"}
        }
        if row["kind"] == "expert":
            row.update(
                format="mxfp4", codec="mxfp4", bits=4, source_codec="fp4_e2m1_ue8m0", lossless=True
            )
        elif row["role"] in {"moe.router_gate", "moe.shared_expert_gate"}:
            row.update(kind="fp16_passthrough", format="fp16", bits=16)
        else:
            row.update(format="affine", bits=4, group_size=32)
        allocation.append(row)
    return make_artifact(
        "optimizer_decision",
        evidence["subject"],
        {"name": "synthetic_fixture", "version": "1"},
        status="valid",
        allocation=allocation,
        inputs=[evidence["artifact_id"]],
        constraints={},
        achieved={},
        required_features=evidence.get("required_features", []),
    )


def resident_mxfp4(n_experts, in_features, out_features):
    """A small resident projection using the same bytes as pooled-MXFP4 tests."""
    import mlx.core as mx
    import mlx.nn as nn

    class Projection(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.num_experts = n_experts
            self.bits = 4
            self.codec = "mxfp4"
            weight = mx.random.normal((n_experts, out_features, in_features)) * 0.02
            self.packed, self.scales = mx.quantize(weight, group_size=32, bits=4, mode="mxfp4")
            mx.eval(self.packed, self.scales)

        def __call__(self, x, indices, *, sorted_indices=False):
            return mx.gather_qmm(
                x,
                self.packed,
                self.scales,
                None,
                rhs_indices=indices,
                transpose=True,
                group_size=32,
                bits=4,
                mode="mxfp4",
                sorted_indices=sorted_indices,
            )

    return Projection()
