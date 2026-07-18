"""Packed-size formulas for the two quantized weight formats.

Pure arithmetic: bytes on disk for a tensor at a given bit-width. The optimizer
uses these to price each bit-up against the size budget. Format-neutral: no
names, no model.
"""

from __future__ import annotations


def tq_expert_bytes(n_experts: int, rows: int, cols: int, bits: int) -> int:
    """Packed size for a stacked expert tensor under TurboQuant.

    Per expert: rows * ceil(cols*bits/32)*4 (packed words) + rows*2 (fp16 norms).
    Plus 1 byte for the per-tensor tq_bits scalar.
    """
    packed_per_row = ((cols * bits + 31) // 32) * 4
    per_expert = rows * packed_per_row + rows * 2
    return n_experts * per_expert + 1


def mxfp4_expert_bytes(n_experts: int, rows: int, cols: int) -> int:
    """Packed size for a routed expert tensor stored as source mxfp4.

    Per expert: rows * ceil(cols/8)*4 packed bytes plus one uint8 UE8M0 scale
    per 32 input values. The format has no norms, seed, or affine biases.
    """
    packed_words = (cols + 7) // 8
    scale_cols = (cols + 31) // 32
    per_expert = rows * (packed_words * 4 + scale_cols)
    return n_experts * per_expert


def mx_float_bytes(rows: int, cols: int, bits: int) -> int:
    """Packed size for an MLX MX float quantized dense tensor.

    ``mxfp4`` and ``mxfp8`` use fixed group size 32, one uint8 UE8M0 scale per
    group, and no affine bias side table.
    """
    n_elements = rows * cols
    weight_bytes = (n_elements * bits + 7) // 8
    scale_bytes = rows * ((cols + 31) // 32)
    return weight_bytes + scale_bytes


def affine_bytes(rows: int, cols: int, bits: int, group_size: int) -> int:
    """Packed size for a standard MLX affine-quantized tensor."""
    n_elements = rows * cols
    weight_bytes = (n_elements * bits + 7) // 8
    n_groups = n_elements // group_size
    overhead = n_groups * 4  # fp16 scale + fp16 bias per group, 2*2 bytes
    return weight_bytes + overhead
