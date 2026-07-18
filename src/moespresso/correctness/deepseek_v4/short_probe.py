"""Manual short-prompt DS4 serve-graph probes."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from moespresso.correctness.deepseek_v4.quality import (
    DEEPSEEK_V4_Q1_TOP_LOGPROBS,
    DS4_TEST_VECTOR_FIXTURE_ROOT,
    _read_json,
    _single_token_id_from_bytes,
)
from moespresso.runtime.deepseek_v4.renderer import DEEPSEEK_V4_PROMPT_RENDERER
from moespresso.runtime.http import render_prompt
from moespresso.runtime.serve import generate_with_metadata, load_served_model

_DEFAULT_PROMPTS = ("short_code_completion", "short_reasoning_plain")
SOURCE_ENV = "MOESPRESSO_DEEPSEEK_V4_SOURCE"
GGUF_RECIPE_ENV = "MOESPRESSO_DS4_KQUANT_GGUF_RECIPE"


def _path_from_env(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


_DEFAULT_SOURCE = _path_from_env(SOURCE_ENV)
_DEFAULT_GGUF = _path_from_env(GGUF_RECIPE_ENV)
_SOURCE_ATTENTION_PROJECTIONS = (
    "wq_a",
    "wq_b",
    "wkv",
    "wo_a",
    "wo_b",
)
_GGUF_ATTENTION_PROJECTIONS = {
    "wq_a": "attn_q_a",
    "wq_b": "attn_q_b",
    "wkv": "attn_kv",
    "wo_a": "attn_output_a",
    "wo_b": "attn_output_b",
}


def _validate_probe_input_paths(
    *,
    embedding_arm: str = "default",
    attention_projection_arm: str = "default",
    ffn_routed_arm: str = "default",
    source_dir: Path | None,
    gguf_path: Path | None,
) -> None:
    source_needed = (
        embedding_arm == "source_prompt_rows"
        or attention_projection_arm in {"source_layer0_all", "source_layers0_3_all"}
    )
    gguf_needed = (
        attention_projection_arm == "ds4_gguf_layer0_all"
        or ffn_routed_arm == "ds4_gguf_layer1"
    )
    if source_needed and source_dir is None:
        raise ValueError(f"--source-dir is required unless ${SOURCE_ENV} is set")
    if gguf_needed and gguf_path is None:
        raise ValueError(f"--gguf-path is required unless ${GGUF_RECIPE_ENV} is set")


def _prompt_specs(fixture_root: Path, prompt_ids: Iterable[str]) -> list[dict]:
    wanted = set(prompt_ids)
    manifest = _read_json(fixture_root / "manifest.json")
    specs = [row for row in manifest["prompts"] if row["id"] in wanted]
    found = {row["id"] for row in specs}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"unknown DS4 prompt id(s): {missing}")
    return specs


def _official_first_token_id(tokenizer, official_step: dict) -> tuple[int | None, str]:
    return _single_token_id_from_bytes(tokenizer, official_step["token"]["bytes"])


def _top_rank(top_logprobs: tuple[dict, ...], token_id: int | None) -> int | None:
    if token_id is None:
        return None
    ids = [int(item["token_id"]) for item in top_logprobs]
    return ids.index(int(token_id)) if int(token_id) in ids else None


def _patch_cache_arm(arm: str) -> None:
    if arm == "required_cache":
        return
    if arm != "upstream_plain_kvcache":
        raise ValueError(f"unknown cache arm: {arm}")

    import moespresso.runtime.deepseek_v4.model as dsv4_model

    def _leave_upstream_make_cache(model) -> None:
        object.__setattr__(
            model,
            "_moespresso_dsv4_attention_cache_contract",
            "upstream_plain_kvcache_ab",
        )

    dsv4_model._patch_deepseek_v4_required_attention_cache = _leave_upstream_make_cache


def _patch_attention_arm(arm: str) -> None:
    if arm == "default":
        return
    if arm != "exact":
        raise ValueError(f"unknown attention arm: {arm}")

    import mlx.core as mx
    import jang_tools.dsv4.mlx_model as dsv4_model

    def _apply_mask(scores, mask):
        if mask is None:
            return scores
        if isinstance(mask, str) and mask == "causal":
            queries = scores.shape[-2]
            keys = scores.shape[-1]
            q_pos = mx.arange(queries) + (keys - queries)
            k_pos = mx.arange(keys)
            visible = k_pos[None, :] <= q_pos[:, None]
            return mx.where(visible[None, None], scores, -mx.inf)
        if getattr(mask, "dtype", None) == mx.bool_:
            return mx.where(mask, scores, -mx.inf)
        return scores + mask

    def _exact_attention(queries, keys, values, cache, scale, mask, sinks=None):
        del cache
        dtype = queries.dtype
        q = queries.astype(mx.float16).astype(mx.float32)
        k = keys.astype(mx.float16).astype(mx.float32)
        v = values.astype(mx.float16).astype(mx.float32)
        scores = (q @ k.swapaxes(-1, -2)) * scale
        scores = _apply_mask(scores, mask)
        if sinks is None:
            return mx.softmax(scores, axis=-1, precise=True) @ v

        sink = sinks.astype(mx.float16).astype(mx.float32).reshape(
            (1, sinks.shape[0], 1, 1)
        )
        max_score = mx.maximum(mx.max(scores, axis=-1, keepdims=True), sink)
        weights = mx.exp(scores - max_score)
        denom = mx.sum(weights, axis=-1, keepdims=True) + mx.exp(sink - max_score)
        return ((weights / denom) @ v).astype(dtype)

    dsv4_model.scaled_dot_product_attention = _exact_attention


def _load_source_embedding_rows(source_dir: Path, tokens: list[int]):
    from moespresso.probe.weight_io import (
        _bytes_to_float32,
        load_2d_rows_raw,
        scan_offsets,
    )

    unique = np.array(sorted(set(int(token) for token in tokens)), dtype=np.int64)
    header = scan_offsets(source_dir)["embed.weight"]
    raw = load_2d_rows_raw(source_dir, header, unique)
    rows = _bytes_to_float32(raw.tobytes(), header.dtype, raw.shape)
    return unique, rows


def _patch_prompt_embedding_rows(
    model,
    source_dir: Path | None,
    tokens: list[int],
) -> None:
    _validate_probe_input_paths(
        embedding_arm="source_prompt_rows",
        source_dir=source_dir,
        gguf_path=None,
    )
    assert source_dir is not None
    import mlx.core as mx
    import mlx.nn as nn

    model_body = getattr(model, "model", model)
    original = getattr(model_body, "embed", None)
    if original is None:
        raise ValueError("model has no embed module to patch")
    unique, rows = _load_source_embedding_rows(source_dir, tokens)

    class _PromptSourceEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.original = original
            self.tokens = mx.array(unique, dtype=mx.int32)
            self.rows = mx.array(rows, dtype=mx.float32)

        def __call__(self, input_ids):
            out = self.original(input_ids)
            patched = out.astype(mx.float32)
            for token, row in zip(self.tokens, self.rows, strict=True):
                patched = mx.where(
                    (input_ids == token)[..., None],
                    row,
                    patched,
                )
            return patched

    model_body.embed = _PromptSourceEmbedding()
    mx.eval(model_body.embed.rows)


def _source_attention_linear(source_dir: Path, layer: int, projection: str):
    import mlx.core as mx
    import mlx.nn as nn

    from moespresso.probe.deepseek_v4.codec import load_dequantized_fp8
    from moespresso.probe.weight_io import scan_offsets

    catalog = scan_offsets(source_dir)
    prefix = f"layers.{layer}.attn.{projection}"
    weight_name = f"{prefix}.weight"
    scale_name = f"{prefix}.scale"
    if weight_name not in catalog or scale_name not in catalog:
        raise ValueError(f"missing source FP8 tensor or scale for {prefix}")
    weight = load_dequantized_fp8(
        source_dir,
        catalog[weight_name],
        catalog[scale_name],
        out_dtype=np.float16,
    )
    out_dim, in_dim = weight.shape
    linear = nn.Linear(int(in_dim), int(out_dim), bias=False)
    linear.weight = mx.array(weight, dtype=mx.float16)
    mx.eval(linear.weight)
    return linear


def _read_gguf_header(path: Path):
    from moespresso.probe.gguf_parse import GGUFBufferParser

    parser = GGUFBufferParser()
    with open(path, "rb") as f:
        while not parser.is_complete():
            chunk = f.read(1 << 20)
            if not chunk:
                parser.try_parse()
                break
            parser.feed(chunk)
            parser.try_parse()
    if parser.header is None or not parser.is_complete():
        raise ValueError(f"failed to parse GGUF tensor directory from {path}")
    data_offset = ((parser.total_consumed() + 31) // 32) * 32
    return parser, data_offset


def _load_gguf_q8_0(path: Path, tensor_name: str, *, out_dtype=np.float16) -> np.ndarray:
    from moespresso.probe.gguf_parse import TENSOR_TYPE_NAMES

    parser, data_offset = _read_gguf_header(path)
    tensors = {ti.name: ti for ti in parser.tensor_infos}
    tensor = tensors.get(tensor_name)
    if tensor is None:
        raise ValueError(f"missing GGUF tensor {tensor_name!r}")
    type_name = TENSOR_TYPE_NAMES.get(tensor.type_id)
    if type_name != "Q8_0":
        raise ValueError(f"{tensor_name}: expected Q8_0, got {type_name!r}")
    if tensor.n_dimensions != 2:
        raise ValueError(f"{tensor_name}: expected 2D, got {tensor.n_dimensions}D")

    in_dim, out_dim = (int(dim) for dim in tensor.dimensions)
    blocks = (in_dim + 31) // 32
    row_bytes = blocks * 34
    nbytes = out_dim * row_bytes
    with open(path, "rb") as f:
        f.seek(data_offset + tensor.offset)
        raw = f.read(nbytes)
    if len(raw) != nbytes:
        raise ValueError(f"{tensor_name}: short read {len(raw)} of {nbytes} bytes")

    block_dtype = np.dtype([("scale", "<f2"), ("qs", "i1", (32,))])
    packed = np.frombuffer(raw, dtype=block_dtype).reshape(out_dim, blocks)
    values = (
        packed["qs"].astype(np.float32)
        * packed["scale"].astype(np.float32)[..., None]
    ).reshape(out_dim, blocks * 32)[:, :in_dim]
    return values.astype(out_dtype)


def _gguf_kquant_storage_shape(tensor, codec: str) -> tuple[int, int, int]:
    from moespresso.package.kquant_format import KQUANT_GEOMETRY

    geometry = KQUANT_GEOMETRY.get(codec)
    if geometry is None:
        raise ValueError(f"{tensor.name}: unsupported K-quant codec {codec!r}")
    if tensor.n_dimensions != 3:
        raise ValueError(
            f"{tensor.name}: expected stacked expert tensor, "
            f"got {tensor.n_dimensions}D")
    in_features, out_features, experts = (int(dim) for dim in tensor.dimensions)
    if in_features % geometry.weights_per_block:
        raise ValueError(
            f"{tensor.name}: in_features {in_features} is not divisible by "
            f"{geometry.weights_per_block} for {codec}")
    packed_cols = in_features // geometry.weights_per_block * geometry.bytes_per_block
    return experts, out_features, packed_cols


def _load_gguf_kquant_tensor(path: Path, tensor_name: str):
    import mlx.core as mx

    from moespresso.package.kquant_recipe import GGUF_TO_KQUANT_CODEC
    from moespresso.probe.gguf_parse import TENSOR_TYPE_NAMES

    parser, data_offset = _read_gguf_header(path)
    tensors = {ti.name: ti for ti in parser.tensor_infos}
    tensor = tensors.get(tensor_name)
    if tensor is None:
        raise ValueError(f"missing GGUF tensor {tensor_name!r}")
    type_name = TENSOR_TYPE_NAMES.get(tensor.type_id)
    codec = GGUF_TO_KQUANT_CODEC.get(type_name)
    if codec is None:
        raise ValueError(
            f"{tensor_name}: expected K-quant tensor, got {type_name!r}")
    shape = _gguf_kquant_storage_shape(tensor, codec)
    nbytes = int(np.prod(shape, dtype=np.int64))
    with open(path, "rb") as f:
        f.seek(data_offset + tensor.offset)
        raw = f.read(nbytes)
    if len(raw) != nbytes:
        raise ValueError(f"{tensor_name}: short read {len(raw)} of {nbytes} bytes")
    weight = mx.array(np.frombuffer(raw, dtype=np.uint8).reshape(shape))
    scales = mx.zeros((1,), dtype=mx.uint8)
    return weight, scales, codec


def _gguf_attention_linear(gguf_path: Path, layer: int, projection: str):
    import mlx.core as mx
    import mlx.nn as nn

    gguf_projection = _GGUF_ATTENTION_PROJECTIONS[projection]
    weight = _load_gguf_q8_0(
        gguf_path,
        f"blk.{layer}.{gguf_projection}.weight",
        out_dtype=np.float16,
    )
    out_dim, in_dim = weight.shape
    linear = nn.Linear(int(in_dim), int(out_dim), bias=False)
    linear.weight = mx.array(weight, dtype=mx.float16)
    mx.eval(linear.weight)
    return linear


def _patch_attention_projection_arm(
    model,
    arm: str,
    source_dir: Path | None,
    gguf_path: Path | None,
) -> None:
    _validate_probe_input_paths(
        attention_projection_arm=arm,
        source_dir=source_dir,
        gguf_path=gguf_path,
    )
    if arm == "default":
        return
    if arm in {"source_layer0_all", "source_layers0_3_all"}:
        assert source_dir is not None
        layers = range(1)
        if arm == "source_layers0_3_all":
            layers = range(4)

        def load_linear(layer: int, projection: str):
            return _source_attention_linear(source_dir, layer, projection)

    elif arm == "ds4_gguf_layer0_all":
        assert gguf_path is not None
        layers = range(1)

        def load_linear(layer: int, projection: str):
            return _gguf_attention_linear(gguf_path, layer, projection)

    else:
        raise ValueError(f"unknown attention projection arm: {arm}")

    model_body = getattr(model, "model", model)
    for layer in layers:
        try:
            attention = model_body.layers[layer].self_attn
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError(
                f"model has no layer-{layer} self_attn module to patch"
            ) from exc
        for projection in _SOURCE_ATTENTION_PROJECTIONS:
            setattr(
                attention,
                projection,
                load_linear(layer, projection),
            )


def _make_gguf_routed_switch(gguf_path: Path, layer: int, activation):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx_kquant as kq

    def load_projection(projection: str):
        return _load_gguf_kquant_tensor(
            gguf_path,
            f"blk.{layer}.ffn_{projection}_exps.weight",
        )

    class _GGUFRoutedSwitch(nn.Module):
        def __init__(self):
            super().__init__()
            self.mode = "diagnostic_gguf_routed"
            self.layer = int(layer)
            self.activation = activation
            self.gate_weight, self.gate_scales, self.gate_codec = load_projection(
                "gate")
            self.up_weight, self.up_scales, self.up_codec = load_projection("up")
            self.down_weight, self.down_scales, self.down_codec = load_projection(
                "down")
            self.block_exit_kick_calls = 0
            self.decode_moe_block_calls = 0
            self.decode_moe_block_seconds = 0.0
            self.overlap_shared_eval_calls = 0
            self.overlap_shared_eval_seconds = 0.0
            self.overlap_prefill_no_eval_calls = 0

        def begin_projection_load(self, indices):
            del indices
            return None

        def _gather(self, x, weight, scales, codec, indices, *, sorted_indices):
            return kq.gather_qmm(
                x,
                weight,
                scales,
                codec,
                rhs_indices=indices,
                transpose=True,
                sorted_indices=sorted_indices,
            )

        def __call__(self, x, indices, *, load_ticket=None):
            del load_ticket
            from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

            x = mx.expand_dims(x, (-2, -3))
            do_sort = indices.size >= 64
            idx = indices
            inv_order = None
            if do_sort:
                x, idx, inv_order = _gather_sort(x, indices)
            if self.training:
                idx = mx.stop_gradient(idx)
            idx = idx.astype(mx.uint32)

            x_up = self._gather(
                x,
                self.up_weight,
                self.up_scales,
                self.up_codec,
                idx,
                sorted_indices=do_sort,
            )
            x_gate = self._gather(
                x,
                self.gate_weight,
                self.gate_scales,
                self.gate_codec,
                idx,
                sorted_indices=do_sort,
            )
            out = self._gather(
                self.activation(x_up, x_gate),
                self.down_weight,
                self.down_scales,
                self.down_codec,
                idx,
                sorted_indices=do_sort,
            )
            if do_sort:
                out = _scatter_unsort(out, inv_order, indices.shape)
            return out.squeeze(-2)

    return _GGUFRoutedSwitch()


def _patch_ffn_routed_arm(model, arm: str, gguf_path: Path | None) -> None:
    _validate_probe_input_paths(
        ffn_routed_arm=arm,
        source_dir=None,
        gguf_path=gguf_path,
    )
    if arm == "default":
        return
    if arm == "ds4_gguf_layer1":
        assert gguf_path is not None
        layers = (1,)
    else:
        raise ValueError(f"unknown FFN routed arm: {arm}")

    model_body = getattr(model, "model", model)
    for layer in layers:
        try:
            mlp = model_body.layers[layer].mlp
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError(f"model has no layer-{layer} mlp module to patch") from exc
        switch = getattr(mlp, "switch_mlp", None)
        activation = getattr(switch, "activation", None)
        if activation is None:
            raise ValueError(f"layer-{layer} switch_mlp has no activation")
        setattr(mlp, "switch_mlp", _make_gguf_routed_switch(
            gguf_path,
            layer,
            activation,
        ))


def run_probe(
    package_dir: Path,
    *,
    fixture_root: Path,
    prompt_ids: Iterable[str],
    cache_arm: str,
    attention_arm: str,
    embedding_arm: str,
    attention_projection_arm: str,
    ffn_routed_arm: str,
    source_dir: Path | None,
    gguf_path: Path | None,
) -> dict:
    _patch_cache_arm(cache_arm)
    _patch_attention_arm(attention_arm)
    model, tokenizer, manifest = load_served_model(package_dir)
    _patch_attention_projection_arm(
        model, attention_projection_arm, source_dir, gguf_path)
    _patch_ffn_routed_arm(model, ffn_routed_arm, gguf_path)
    rows = []
    for prompt_spec in _prompt_specs(fixture_root, prompt_ids):
        prompt_text = (fixture_root / prompt_spec["prompt_file"]).read_text(
            encoding="utf-8"
        )
        official = _read_json(fixture_root / prompt_spec["official_file"])
        official_step = official["steps"][0]
        expected_id, expected_text = _official_first_token_id(tokenizer, official_step)
        rendered = render_prompt(
            [{"role": "user", "content": prompt_text}],
            tokenizer,
            template_kwargs={"enable_thinking": False},
            prompt_renderer=DEEPSEEK_V4_PROMPT_RENDERER,
        )
        if embedding_arm == "source_prompt_rows":
            _patch_prompt_embedding_rows(model, source_dir, tokenizer.encode(rendered))
        elif embedding_arm != "default":
            raise ValueError(f"unknown embedding arm: {embedding_arm}")
        result = generate_with_metadata(
            model,
            tokenizer,
            rendered,
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            top_logprobs=DEEPSEEK_V4_Q1_TOP_LOGPROBS,
        )
        selected_id = (
            int(result.generated_token_ids[0]) if result.generated_token_ids else None
        )
        top20 = result.top_logprobs[0] if result.top_logprobs else ()
        rows.append({
            "id": prompt_spec["id"],
            "prompt_tokens": result.prompt_tokens,
            "candidate_selected": {
                "token_id": selected_id,
                "text": tokenizer.decode([selected_id]) if selected_id is not None else "",
            },
            "official_selected": {
                "token_id": expected_id,
                "text": expected_text,
            },
            "official_rank": _top_rank(top20, expected_id),
            "top20_token_ids": [int(item["token_id"]) for item in top20],
        })
    return {
        "package_dir": str(package_dir),
        "package_manifest_id": manifest.get("artifact_id"),
        "cache_arm": cache_arm,
        "attention_arm": attention_arm,
        "embedding_arm": embedding_arm,
        "attention_projection_arm": attention_projection_arm,
        "ffn_routed_arm": ffn_routed_arm,
        "prompts": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=DS4_TEST_VECTOR_FIXTURE_ROOT,
    )
    parser.add_argument(
        "--prompt-id",
        action="append",
        dest="prompt_ids",
        default=None,
        help="Prompt id to run. Defaults to the two failing short Q1 prompts.",
    )
    parser.add_argument(
        "--cache-arm",
        choices=("required_cache", "upstream_plain_kvcache"),
        required=True,
    )
    parser.add_argument(
        "--attention-arm",
        choices=("default", "exact"),
        default="default",
    )
    parser.add_argument(
        "--embedding-arm",
        choices=("default", "source_prompt_rows"),
        default="default",
    )
    parser.add_argument(
        "--attention-projection-arm",
        choices=(
            "default",
            "source_layer0_all",
            "source_layers0_3_all",
            "ds4_gguf_layer0_all",
        ),
        default="default",
    )
    parser.add_argument(
        "--ffn-routed-arm",
        choices=("default", "ds4_gguf_layer1"),
        default="default",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=_DEFAULT_SOURCE,
        help=f"DS4 HF source checkpoint. Defaults to ${SOURCE_ENV}.",
    )
    parser.add_argument(
        "--gguf-path",
        type=Path,
        default=_DEFAULT_GGUF,
        help=f"DS4 GGUF recipe path. Defaults to ${GGUF_RECIPE_ENV}.",
    )
    args = parser.parse_args(argv)
    try:
        _validate_probe_input_paths(
            embedding_arm=args.embedding_arm,
            attention_projection_arm=args.attention_projection_arm,
            ffn_routed_arm=args.ffn_routed_arm,
            source_dir=args.source_dir,
            gguf_path=args.gguf_path,
        )
    except ValueError as exc:
        parser.error(str(exc))
    result = run_probe(
        args.package,
        fixture_root=args.fixture_root,
        prompt_ids=args.prompt_ids or _DEFAULT_PROMPTS,
        cache_arm=args.cache_arm,
        attention_arm=args.attention_arm,
        embedding_arm=args.embedding_arm,
        attention_projection_arm=args.attention_projection_arm,
        ffn_routed_arm=args.ffn_routed_arm,
        source_dir=args.source_dir,
        gguf_path=args.gguf_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
