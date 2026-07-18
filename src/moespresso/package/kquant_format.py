"""K-quant codec geometry shared by package writers and runtime loaders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KQuantCodecGeometry:
    group_size: int
    bits: int
    bytes_per_block: int
    weights_per_block: int


KQUANT_GEOMETRY = {
    "q8_0": KQuantCodecGeometry(32, 8, 34, 32),
    "q4_0": KQuantCodecGeometry(32, 4, 18, 32),
    "q4_1": KQuantCodecGeometry(32, 4, 20, 32),
    "q5_0": KQuantCodecGeometry(32, 5, 22, 32),
    "q5_1": KQuantCodecGeometry(32, 5, 24, 32),
    "q2_k": KQuantCodecGeometry(256, 2, 84, 256),
    "q3_k": KQuantCodecGeometry(256, 3, 110, 256),
    "q4_k": KQuantCodecGeometry(256, 4, 144, 256),
    "q5_k": KQuantCodecGeometry(256, 5, 176, 256),
    "q6_k": KQuantCodecGeometry(256, 6, 210, 256),
    "iq4_nl": KQuantCodecGeometry(32, 4, 18, 32),
    "iq4_xs": KQuantCodecGeometry(256, 4, 136, 256),
    "iq3_s": KQuantCodecGeometry(256, 3, 110, 256),
    "iq3_xxs": KQuantCodecGeometry(256, 3, 98, 256),
    "iq2_xxs": KQuantCodecGeometry(256, 2, 66, 256),
    "iq2_xs": KQuantCodecGeometry(256, 2, 74, 256),
    "iq2_s": KQuantCodecGeometry(256, 2, 82, 256),
    "iq1_s": KQuantCodecGeometry(256, 1, 50, 256),
    "iq1_m": KQuantCodecGeometry(256, 1, 56, 256),
}

IMATRIX_STEERED_CODECS = frozenset({
    "q2_k", "q3_k", "q4_k", "q5_k", "q6_k",
    "iq4_nl", "iq4_xs", "iq3_s", "iq3_xxs",
    "iq2_xxs", "iq2_xs", "iq2_s", "iq1_s", "iq1_m",
})
IMATRIX_REQUIRED_CODECS = frozenset({
    "q2_k", "q3_k", "q4_k", "q5_k", "q6_k",
    "iq2_xxs", "iq2_xs", "iq1_s",
})

# Codecs whose encode has a real Metal (GPU) kernel in mlx-kquant. The K-quant
# superblock and legacy block codecs encode on either stream; the nine `iq*`
# codecs have no GPU encoder (ggml ships none) and mlx-kquant force-pins them to
# the CPU stream, where a single-threaded scalar encode is ~30-75x slower per
# tensor (measured: q2_k encode 18.6 ms GPU vs 557 ms CPU; iq2_xxs 1434 ms CPU).
# Routing a GPU-capable codec's encode to the GPU stream is bit-identical to the
# CPU encode and is the difference between a ~10 min and a multi-hour build.
GPU_ENCODE_CODECS = frozenset({
    "q2_k", "q3_k", "q4_k", "q5_k", "q6_k",
    "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
})


def encode_stream_for_codec(codec: str) -> str:
    """Stream a codec's encode should run on."""
    return "gpu" if codec in GPU_ENCODE_CODECS else "cpu"
