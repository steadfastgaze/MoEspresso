"""Compare DS4 HC split dumps against the served MLX HC split implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from moespresso.correctness.deepseek_v4.rope_diff import _metrics
from moespresso.runtime.serve import load_served_model


N_HC = 4
MIX_HC = (2 + N_HC) * N_HC


def _layers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _families(value: str) -> list[str]:
    families = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(families) - {"attn", "ffn"})
    if unknown:
        raise ValueError(f"unknown HC family: {', '.join(unknown)}")
    return families


def _read_pos_dump(prefix: Path, name: str, layer: int, pos: int) -> np.ndarray:
    path = Path(f"{prefix}_{name}-{layer}_pos{pos}.bin")
    if not path.is_file():
        raise FileNotFoundError(f"missing DS4 dump: {path}")
    return np.fromfile(path, dtype=np.float32)


def _layer_at(model, layer: int):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("loaded model has no model.layers")
    return layers[layer]


def _as_numpy(value) -> np.ndarray:
    import mlx.core as mx

    mx.eval(value)
    return np.asarray(value.astype(mx.float32))


def _hc_split_from_mlx(mixes: np.ndarray, scale, base) -> tuple[np.ndarray, ...]:
    import mlx.core as mx

    from jang_tools.dsv4.mlx_model import hc_split_sinkhorn

    pre, post, comb = hc_split_sinkhorn(
        mx.array(mixes.reshape(1, MIX_HC), dtype=mx.float32),
        scale,
        base,
        N_HC,
        iters=20,
        eps=1e-6,
    )
    return (
        _as_numpy(pre).reshape(N_HC),
        _as_numpy(post).reshape(N_HC),
        _as_numpy(comb).reshape(N_HC * N_HC),
    )


def compare_hc_split_dumps(
    *,
    package_dir: Path,
    dump_prefix: Path,
    layers: list[int],
    pos: int,
    families: list[str],
) -> dict:
    model, _tokenizer, manifest = load_served_model(package_dir)
    rows = []
    for layer_id in layers:
        layer = _layer_at(model, layer_id)
        family_rows = {}
        for family in families:
            prefix = f"hc_{family}_pre"
            mixes = _read_pos_dump(dump_prefix, f"{prefix}_mixes", layer_id, pos)
            if mixes.size != MIX_HC:
                raise ValueError(
                    f"bad {prefix}_mixes size for layer {layer_id}: {mixes.size}"
                )
            scale = getattr(layer, f"hc_{family}_scale")
            base = getattr(layer, f"hc_{family}_base")
            got_pre, got_post, got_comb = _hc_split_from_mlx(mixes, scale, base)
            ref_pre = _read_pos_dump(dump_prefix, f"{prefix}_weights", layer_id, pos)
            ref_post = _read_pos_dump(
                dump_prefix,
                f"{prefix}_post_weights",
                layer_id,
                pos,
            )
            ref_comb = _read_pos_dump(dump_prefix, f"{prefix}_comb", layer_id, pos)
            if ref_pre.size != N_HC:
                raise ValueError(
                    f"bad {prefix}_weights size for layer {layer_id}: {ref_pre.size}"
                )
            if ref_post.size != N_HC:
                raise ValueError(
                    f"bad {prefix}_post_weights size for layer {layer_id}: "
                    f"{ref_post.size}"
                )
            if ref_comb.size != N_HC * N_HC:
                raise ValueError(
                    f"bad {prefix}_comb size for layer {layer_id}: {ref_comb.size}"
                )
            family_rows[family] = {
                "pre": _metrics(got_pre, ref_pre),
                "post": _metrics(got_post, ref_post),
                "comb": _metrics(got_comb, ref_comb),
                "comb_transposed": _metrics(
                    got_comb.reshape(N_HC, N_HC).T.reshape(N_HC * N_HC),
                    ref_comb,
                ),
            }
        rows.append({
            "layer": int(layer_id),
            "families": family_rows,
        })
    return {
        "package_dir": str(package_dir),
        "package_manifest_id": manifest.get("artifact_id"),
        "dump_prefix": str(dump_prefix),
        "pos": int(pos),
        "layers": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--dump-prefix", required=True, type=Path)
    parser.add_argument("--layers", default="0,2", type=_layers)
    parser.add_argument("--pos", required=True, type=int)
    parser.add_argument("--families", default="attn,ffn", type=_families)
    args = parser.parse_args(argv)
    result = compare_hc_split_dumps(
        package_dir=args.package,
        dump_prefix=args.dump_prefix,
        layers=args.layers,
        pos=args.pos,
        families=args.families,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
