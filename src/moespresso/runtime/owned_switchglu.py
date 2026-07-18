"""MoEspresso-owned SwitchGLU forward.

Jang patches mlx_lm's SwitchGLU.__call__ at the class level for a fast fused
gate+up path. That fused kernel has one bit-width parameter for both gate and up,
so it is not valid for packages whose routed experts deliberately use different
gate/up bit-widths. This tiny wrapper keeps the same module attributes but owns
the forward as a distinct class, so mixed-bit layers can use each projection's
own runtime contract.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class OwnedSwitchGLU(nn.Module):
    """SwitchGLU forward that cannot be affected by Jang's class monkeypatch."""

    def __init__(self, *, gate_proj, up_proj, down_proj, activation):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)

        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)
