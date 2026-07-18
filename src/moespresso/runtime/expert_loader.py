"""Direct expert byte-range loader for SSD-streamed MoE.

Loads one expert's packed/norms bytes from a package shard via `pread` (using the
expert byte-offset index, expert_index.py) directly into an `mx.array` buffer,
without faulting the whole stacked tensor and without materializing a Python
`bytes` payload.

No TQ dequant: the bytes stay packed uint32 and jang's kernel runs them. This
module returns a fresh per-expert array for proof/tests; the product miss path
will use the same `pread_into` primitive to fill persistent pool slots.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from moespresso.runtime.expert_index import ExpertIndex
from moespresso.runtime.pread_into import pread_into

_ST_TO_MX = {"U32": mx.uint32, "F16": mx.float16,
             "F32": mx.float32, "U8": mx.uint8}


def load_expert(package_dir: str | Path, index: ExpertIndex, *, layer: int,
                expert: int, projection: str, component: str) -> mx.array:
    """pread one expert's bytes directly into a fresh mx.array.

    The byte range, shard, shape, and dtype all come from the index; callers provide
    only package root + logical expert coordinates. This keeps shard ownership at
    the index boundary and avoids silent wrong-file reads when components straddle
    shards.
    """
    br = index.locate(layer=layer, expert=expert, projection=projection,
                      component=component)
    mx_dtype = _ST_TO_MX.get(br.dtype)
    if mx_dtype is None:
        raise ValueError(f"unsupported dtype {br.dtype!r}")
    arr = mx.zeros(br.shape, dtype=mx_dtype)
    mx.eval(arr)
    pread_into(arr, Path(package_dir) / br.shard,
               file_offset=br.offset, nbytes=br.nbytes)
    mx.eval(arr)
    return arr
