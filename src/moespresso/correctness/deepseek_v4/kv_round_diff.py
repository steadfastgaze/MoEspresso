"""Compare DS4 KVrope dumps against the DS4 FP8 KV round trip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from moespresso.correctness.deepseek_v4.compressor_diff import HEAD_DIM, _fp8_kv_round
from moespresso.correctness.deepseek_v4.rope_diff import _metrics


def _layers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _read_pos_dump(prefix: Path, name: str, layer: int, pos: int) -> np.ndarray:
    path = Path(f"{prefix}_{name}-{layer}_pos{pos}.bin")
    if not path.is_file():
        raise FileNotFoundError(f"missing DS4 dump: {path}")
    return np.fromfile(path, dtype=np.float32)


def compare_kv_round_dumps(
    *,
    dump_prefix: Path,
    layers: list[int],
    pos: int,
) -> dict:
    rows = []
    for layer_id in layers:
        kvrope = _read_pos_dump(dump_prefix, "KVrope", layer_id, pos)
        kvcur = _read_pos_dump(dump_prefix, "KVcur", layer_id, pos)
        if kvrope.size != kvcur.size or kvrope.size % HEAD_DIM:
            raise ValueError(
                f"bad KV dump sizes for layer {layer_id}: "
                f"KVrope={kvrope.size} KVcur={kvcur.size}"
            )
        kvrope_rows = kvrope.reshape(-1, HEAD_DIM)
        kvcur_rows = kvcur.reshape(-1, HEAD_DIM)
        rounded = _fp8_kv_round(kvrope_rows)
        rows.append({
            "layer": int(layer_id),
            "rows": int(kvrope_rows.shape[0]),
            "variants": {
                "no_round": _metrics(kvrope_rows.reshape(-1), kvcur_rows.reshape(-1)),
                "fp8": _metrics(rounded.reshape(-1), kvcur_rows.reshape(-1)),
            },
        })
    return {
        "dump_prefix": str(dump_prefix),
        "pos": int(pos),
        "layers": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-prefix", required=True, type=Path)
    parser.add_argument("--layers", default="0,2", type=_layers)
    parser.add_argument("--pos", required=True, type=int)
    args = parser.parse_args(argv)
    result = compare_kv_round_dumps(
        dump_prefix=args.dump_prefix,
        layers=args.layers,
        pos=args.pos,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
