"""Remote Hugging Face model inspector.

It intentionally uses only stdlib HTTP and Range requests: inspecting a remote
repository should not require downloading the model or adding a Hugging Face SDK
dependency.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

from moespresso.probe.gguf_parse import GGUFBufferParser, GGUFMetadata, TENSOR_TYPE_NAMES

CHUNK_SIZE = 2 * 1024 * 1024
MAX_TOTAL = 50 * 1024 * 1024
MAX_FETCHES_PER_VALUE = 30

CONFIG_DISPLAY_FIELDS = [
    "architectures",
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "vocab_size",
    "max_position_embeddings",
    "torch_dtype",
    "num_experts",
    "num_experts_per_tok",
    "head_dim",
    "tie_word_embeddings",
]


def _normalize_url(url: str) -> str:
    return url.replace("/blob/", "/resolve/")


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if "huggingface" not in (parsed.hostname or ""):
        raise ValueError(f"Not a Hugging Face URL: {url}")


def _parse_repo_id(url: str) -> str:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.rstrip("/").split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Cannot extract repo_id from URL: {url}")
    return f"{parts[0]}/{parts[1]}"


def _is_gguf_url(url: str) -> bool:
    return urlparse(url).path.endswith(".gguf")


def _fetch_bytes(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _fetch_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url)
    req.add_header("Range", f"bytes={start}-{end}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status == 200:
                raise ValueError("Server does not support Range requests")
            if resp.status == 206:
                return resp.read()
            raise ValueError(f"Unexpected HTTP status: {resp.status}")
    except urllib.error.HTTPError as e:
        if e.code == 416:
            raise ValueError(
                "Requested range not satisfiable; file may be smaller than expected"
            ) from e
        raise


def _fetch_json(url: str, timeout: int = 30) -> dict | None:
    try:
        return json.loads(_fetch_bytes(url, timeout=timeout))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def _fetch_safetensors_header(base_url: str, shard: str) -> tuple[str, dict]:
    url = f"{base_url}/resolve/main/{shard}"
    first8 = _fetch_range(url, 0, 7)
    header_size = struct.unpack("<Q", first8)[0]
    header_bytes = _fetch_range(url, 8, 8 + header_size - 1)
    return shard, json.loads(header_bytes)


def _format_ranges(indices: set[int]) -> str:
    if not indices:
        return ""

    sorted_idx = sorted(indices)
    parts: list[str] = []
    start = prev = sorted_idx[0]
    for i in sorted_idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = i
    parts.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(parts)


def _numeric_template(name: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    template_parts: list[str] = []
    idx_values: list[int] = []
    for part in name.split("."):
        if part.isdigit():
            template_parts.append("{}")
            idx_values.append(int(part))
        else:
            template_parts.append(part)
    return tuple(template_parts), tuple(idx_values)


def _compress_tensors(all_tensors: list[tuple[str, list[int], str]]) -> list[str]:
    groups: dict[tuple[tuple[str, ...], str, str], set[tuple[int, ...]]] = defaultdict(set)

    for name, shape, dtype in all_tensors:
        template, idx_values = _numeric_template(name)
        shape_str = ", ".join(str(d) for d in shape)
        groups[(template, shape_str, dtype)].add(idx_values)

    lines: list[str] = []
    for (template, shape_str, dtype), tuples in sorted(groups.items()):
        num_positions = len(next(iter(tuples))) if tuples else 0
        if num_positions == 0:
            lines.append(f"    {'.'.join(template)}: [{shape_str}]  {dtype}")
            continue

        unique_per_pos = [set(t[i] for t in tuples) for i in range(num_positions)]
        expected = 1
        for values in unique_per_pos:
            expected *= len(values)

        if len(tuples) == expected:
            formatted = [f"[{_format_ranges(values)}]" for values in unique_per_pos]
            lines.append(
                f"    {_fill_template(template, formatted)}: [{shape_str}]  {dtype}"
            )
            continue

        for values in sorted(tuples):
            lines.append(
                f"    {_fill_template(template, [str(v) for v in values])}: "
                f"[{shape_str}]  {dtype}"
            )

    return lines


def _fill_template(template: Iterable[str], values: list[str]) -> str:
    filled: list[str] = []
    value_index = 0
    for part in template:
        if part == "{}":
            filled.append(values[value_index])
            value_index += 1
        else:
            filled.append(part)
    return ".".join(filled)


def _format_size(n: int) -> str:
    if n >= 1_000_000_000_000:
        return f"{n / 1_000_000_000_000:.1f} TB"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f} GB"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    return f"{n:,} bytes"


def _format_metadata(kv_pairs: list) -> str:
    lines: list[str] = []
    current_prefix: str | None = None

    for kv in kv_pairs:
        prefix = kv.key.split(".")[0]
        if prefix != current_prefix:
            current_prefix = prefix
            lines.append(f"\n  [{prefix}]")

        val = kv.value
        if kv.value_type == 9 and isinstance(val, list):
            lines.append(f"    {kv.key}: {len(val)} items")
        elif isinstance(val, str) and len(val) > 200:
            lines.append(f"    {kv.key}: ({len(val):,} chars)")
        else:
            lines.append(f"    {kv.key}: {val}")

    return "\n".join(lines)


def _format_tensors(tensor_infos: list) -> str:
    lines: list[str] = []
    for ti in tensor_infos:
        type_name = TENSOR_TYPE_NAMES.get(ti.type_id, f"unknown({ti.type_id})")
        shape = ", ".join(str(d) for d in ti.dimensions)
        lines.append(f"    {ti.name}: [{shape}]  {type_name}  offset=0x{ti.offset:X}")
    return "\n".join(lines)


def _print_gguf_results(parser: GGUFBufferParser) -> None:
    header = parser.header
    if header is None:
        raise ValueError("GGUF header was not parsed")

    print(
        f"GGUF v{header.version}  |  {header.tensor_count} tensors  |  "
        f"{header.metadata_kv_count} metadata KV pairs"
    )
    print(f"Fetched {parser.total_consumed():,} bytes")
    print(_format_metadata(parser.kv_pairs))

    if parser.tensor_infos:
        print("\n  [tensors]")
        print(_format_tensors(parser.tensor_infos))


def _read_remote_gguf_parser(url: str) -> GGUFBufferParser:
    parser = GGUFBufferParser()
    offset = 0
    fetches_since_progress = 0

    while not parser.is_complete():
        if offset >= MAX_TOTAL:
            raise ValueError(f"GGUF metadata exceeds {MAX_TOTAL // (1024 * 1024)}MB limit")

        chunk = _fetch_range(url, offset, offset + CHUNK_SIZE - 1)
        if not chunk:
            break
        parser.feed(chunk)
        offset += len(chunk)
        fetches_since_progress += 1

        prev_kv = len(parser.kv_pairs)
        prev_ti = len(parser.tensor_infos)
        parser.try_parse()

        if len(parser.kv_pairs) > prev_kv or len(parser.tensor_infos) > prev_ti:
            fetches_since_progress = 0
        elif fetches_since_progress > MAX_FETCHES_PER_VALUE:
            raise ValueError(
                f"Exceeded {MAX_FETCHES_PER_VALUE} chunk fetches without progress"
            )

    if parser.header is None:
        raise ValueError(f"Failed to parse GGUF header from {url}")
    if not parser.is_complete():
        raise ValueError(
            f"Truncated GGUF metadata in {url}: parsed "
            f"{len(parser.kv_pairs)}/{parser.header.metadata_kv_count} metadata "
            f"pairs and {len(parser.tensor_infos)}/{parser.header.tensor_count} "
            "tensor infos"
        )
    return parser


def read_remote_gguf_metadata(url: str) -> GGUFMetadata:
    """Read a Hugging Face GGUF tensor directory via HTTP Range requests."""
    _validate_url(url)
    if not _is_gguf_url(url):
        raise ValueError(f"Not a GGUF URL: {url}")
    parser = _read_remote_gguf_parser(_normalize_url(url))
    return GGUFMetadata(
        header=parser.header,
        kv_pairs=list(parser.kv_pairs),
        tensor_infos=list(parser.tensor_infos),
    )


def inspect_gguf(url: str) -> None:
    parser = _read_remote_gguf_parser(url)
    _print_gguf_results(parser)


def inspect_safetensors_repo(repo_id: str) -> None:
    base_url = f"https://huggingface.co/{repo_id}"

    config = _fetch_json(f"{base_url}/raw/main/config.json")
    if config is None:
        raise ValueError(f"No config.json found in {repo_id}")

    index = _fetch_json(f"{base_url}/raw/main/model.safetensors.index.json")
    if index is not None:
        total_size = index.get("metadata", {}).get("total_size", 0)
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        total_size = 0
        shard_files = ["model.safetensors"]

    shard_headers: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_fetch_safetensors_header, base_url, s): s for s in shard_files}
        for future in as_completed(futures):
            shard, header = future.result()
            shard_headers[shard] = header

    all_tensors: list[tuple[str, list[int], str]] = []
    shard_tensor_counts: dict[str, int] = {}
    dtype_counts: dict[str, int] = defaultdict(int)

    for shard in shard_files:
        header = shard_headers[shard]
        tensors_in_shard = {k: v for k, v in header.items() if k != "__metadata__"}
        shard_tensor_counts[shard] = len(tensors_in_shard)
        for name, info in tensors_in_shard.items():
            dtype = info.get("dtype", "UNKNOWN")
            shape = info.get("shape", [])
            all_tensors.append((name, shape, dtype))
            dtype_counts[dtype] += 1

    if total_size == 0:
        total_size = _size_from_safetensors_headers(shard_headers.values())

    all_tensors.sort(key=lambda t: t[0])
    num_shards = len(shard_files)
    shard_word = "shards" if num_shards != 1 else "shard"
    print(
        f"Safetensors  |  {repo_id}  |  {num_shards} {shard_word}  |  "
        f"{len(all_tensors)} tensors  |  {_format_size(total_size)}"
    )

    print("\n  [config]")
    for field in CONFIG_DISPLAY_FIELDS:
        if field in config:
            value = config[field]
            if isinstance(value, list):
                value = json.dumps(value)
            print(f"    {field}: {value}")

    print("\n  [shards]")
    for shard in shard_files:
        print(f"    {shard}: {shard_tensor_counts[shard]} tensors")

    print("\n  [dtype breakdown]")
    for dtype, count in sorted(dtype_counts.items(), key=lambda x: -x[1]):
        print(f"    {dtype}: {count} tensors")

    print(f"\n  [tensors]  ({len(all_tensors)} total)")
    for line in _compress_tensors(all_tensors):
        print(line)


def _size_from_safetensors_headers(headers: Iterable[dict]) -> int:
    total_size = 0
    for header in headers:
        for info in header.values():
            if isinstance(info, dict) and "data_offsets" in info:
                total_size = max(total_size, max(info["data_offsets"]))
    return total_size


def inspect_url(url: str) -> None:
    _validate_url(url)

    if _is_gguf_url(url):
        inspect_gguf(_normalize_url(url))
        return

    inspect_safetensors_repo(_parse_repo_id(url))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Hugging Face model files remotely via HTTP Range requests.",
    )
    parser.add_argument("url", help="Hugging Face model file or repo URL")
    args = parser.parse_args()

    try:
        inspect_url(args.url)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Network error: {e.reason}", file=sys.stderr)
        sys.exit(1)
