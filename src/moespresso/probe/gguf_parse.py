"""Progressive GGUF binary parser: operates on a growing bytearray buffer.

Stdlib only. Used by probe/calibration.py to read GGUF imatrix headers without
faulting the whole (large) tensor-data section into memory.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian

MAX_KV_COUNT = 100_000
MAX_ARRAY_LEN = 1_000_000

VALUE_TYPE_FORMATS: dict[int, tuple[str, int]] = {
    0: ("<B", 1),   # UINT8
    1: ("<b", 1),   # INT8
    2: ("<H", 2),   # UINT16
    3: ("<h", 2),   # INT16
    4: ("<I", 4),   # UINT32
    5: ("<i", 4),   # INT32
    6: ("<f", 4),   # FLOAT32
    7: ("<B", 1),   # BOOL (stored as uint8)
    10: ("<Q", 8),  # UINT64
    11: ("<q", 8),  # INT64
    12: ("<d", 8),  # FLOAT64
}

VALUE_TYPE_NAMES: dict[int, str] = {
    0: "UINT8", 1: "INT8", 2: "UINT16", 3: "INT16",
    4: "UINT32", 5: "INT32", 6: "FLOAT32", 7: "BOOL",
    8: "STRING", 9: "ARRAY", 10: "UINT64", 11: "INT64", 12: "FLOAT64",
}

TENSOR_TYPE_NAMES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16",
}


@dataclass
class GGUFHeader:
    magic: int
    version: int
    tensor_count: int
    metadata_kv_count: int


@dataclass
class GGUFKeyValue:
    key: str
    value_type: int
    value: str | int | float | bool | list[object]


@dataclass
class GGUFTensorInfo:
    name: str
    n_dimensions: int
    dimensions: list[int]
    type_id: int
    offset: int


@dataclass
class GGUFMetadata:
    header: GGUFHeader
    kv_pairs: list[GGUFKeyValue] = field(default_factory=list)
    tensor_infos: list[GGUFTensorInfo] = field(default_factory=list)


class GGUFBufferParser:
    """Progressive GGUF parser fed chunk-by-chunk."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._pos = 0  # how far parsing has progressed
        self._header: GGUFHeader | None = None
        self._kv_pairs: list[GGUFKeyValue] = []
        self._tensor_infos: list[GGUFTensorInfo] = []

    @property
    def header(self) -> GGUFHeader | None:
        return self._header

    @property
    def kv_pairs(self) -> list[GGUFKeyValue]:
        return self._kv_pairs

    @property
    def tensor_infos(self) -> list[GGUFTensorInfo]:
        return self._tensor_infos

    @property
    def metadata(self) -> dict[str, object]:
        return {kv.key: kv.value for kv in self._kv_pairs}

    def feed(self, data: bytes) -> None:
        self._buf.extend(data)

    def _available(self) -> int:
        return len(self._buf) - self._pos

    def _peek(self, n: int) -> bytes | None:
        if self._available() < n:
            return None
        return bytes(self._buf[self._pos : self._pos + n])

    def is_complete(self) -> bool:
        if self._header is None:
            return False
        if len(self._kv_pairs) < self._header.metadata_kv_count:
            return False
        if len(self._tensor_infos) < self._header.tensor_count:
            return False
        return True

    def needs_more_data(self) -> bool:
        return not self.is_complete() and self._pos >= len(self._buf)

    def try_parse(self) -> None:
        """Parse as many complete records as the buffer allows."""
        if self._header is None:
            self._try_parse_header()
            if self._header is None:
                return

        while len(self._kv_pairs) < self._header.metadata_kv_count:
            kv = self._try_parse_kv()
            if kv is None:
                return
            self._kv_pairs.append(kv)

        while len(self._tensor_infos) < self._header.tensor_count:
            ti = self._try_parse_tensor_info()
            if ti is None:
                return
            self._tensor_infos.append(ti)

    def _try_parse_header(self) -> None:
        if self._available() < 24:
            return

        magic = struct.unpack_from("<I", self._buf, self._pos)[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: 0x{magic:08X}")
        self._pos += 4

        version = struct.unpack_from("<I", self._buf, self._pos)[0]
        if version < 2:
            raise ValueError(f"Unsupported GGUF version: {version}")
        self._pos += 4

        tensor_count = struct.unpack_from("<Q", self._buf, self._pos)[0]
        self._pos += 8

        kv_count = struct.unpack_from("<Q", self._buf, self._pos)[0]
        if kv_count > MAX_KV_COUNT:
            raise ValueError(f"GGUF metadata KV count ({kv_count}) exceeds safety limit ({MAX_KV_COUNT})")
        self._pos += 8

        self._header = GGUFHeader(
            magic=magic,
            version=version,
            tensor_count=tensor_count,
            metadata_kv_count=kv_count,
        )

    def _try_read_string(self) -> tuple[str, int] | None:
        """Try to read a string. Returns (value, byte_size) or None if incomplete."""
        if self._available() < 8:
            return None
        length = struct.unpack_from("<Q", self._buf, self._pos)[0]
        total = 8 + length
        if self._available() < total:
            return None
        s = self._buf[self._pos + 8 : self._pos + total].decode("utf-8")
        return s, total

    def _try_read_value(self, vtype: int) -> tuple[object, int] | None:
        """Try to read a value of the given type. Advances _pos on success.

        Returns (value, byte_size) or None if insufficient data (pos unchanged).
        """
        if vtype in VALUE_TYPE_FORMATS:
            fmt, size = VALUE_TYPE_FORMATS[vtype]
            if self._available() < size:
                return None
            val = struct.unpack_from(fmt, self._buf, self._pos)[0]
            self._pos += size
            if vtype == 7:  # BOOL
                val = bool(val)
            return val, size

        if vtype == 8:  # STRING
            result = self._try_read_string()
            if result is None:
                return None
            _, sz = result
            self._pos += sz
            return result

        if vtype == 9:  # ARRAY
            if self._available() < 12:
                return None
            elem_type = struct.unpack_from("<I", self._buf, self._pos)[0]
            arr_len = struct.unpack_from("<Q", self._buf, self._pos + 4)[0]
            if arr_len > MAX_ARRAY_LEN:
                raise ValueError(f"GGUF array length ({arr_len}) exceeds safety limit ({MAX_ARRAY_LEN})")

            consumed = 12
            saved_pos = self._pos
            self._pos += 12

            items: list[object] = []
            for _ in range(arr_len):
                result = self._try_read_value(elem_type)
                if result is None:
                    self._pos = saved_pos
                    return None
                val, sz = result
                items.append(val)
                consumed += sz

            return items, consumed

        raise ValueError(f"Unsupported GGUF value type: {vtype}")

    def _try_parse_kv(self) -> GGUFKeyValue | None:
        saved_pos = self._pos

        key_result = self._try_read_string()
        if key_result is None:
            return None
        key, key_size = key_result
        self._pos += key_size

        if self._available() < 4:
            self._pos = saved_pos
            return None
        vtype = struct.unpack_from("<I", self._buf, self._pos)[0]
        self._pos += 4

        value_result = self._try_read_value(vtype)
        if value_result is None:
            self._pos = saved_pos
            return None
        value, _value_size = value_result

        return GGUFKeyValue(key=key, value_type=vtype, value=value)

    def _try_parse_tensor_info(self) -> GGUFTensorInfo | None:
        saved_pos = self._pos

        name_result = self._try_read_string()
        if name_result is None:
            return None
        name, name_size = name_result
        self._pos += name_size

        if self._available() < 4:
            self._pos = saved_pos
            return None
        n_dims = struct.unpack_from("<I", self._buf, self._pos)[0]
        self._pos += 4

        dims_size = 8 * n_dims
        if self._available() < dims_size + 4 + 8:
            self._pos = saved_pos
            return None

        dims = list(struct.unpack_from(f"<{n_dims}Q", self._buf, self._pos))
        self._pos += dims_size

        type_id = struct.unpack_from("<I", self._buf, self._pos)[0]
        self._pos += 4

        offset = struct.unpack_from("<Q", self._buf, self._pos)[0]
        self._pos += 8

        return GGUFTensorInfo(
            name=name, n_dimensions=n_dims, dimensions=dims, type_id=type_id, offset=offset
        )

    def total_consumed(self) -> int:
        return self._pos


def read_gguf_metadata(
    path: str | Path,
    *,
    chunk_bytes: int = 1 << 20,
) -> GGUFMetadata:
    """Read a GGUF file's metadata and tensor infos without reading tensor data."""
    path = Path(path)
    parser = GGUFBufferParser()
    with open(path, "rb") as f:
        while not parser.is_complete():
            chunk = f.read(chunk_bytes)
            if not chunk:
                parser.try_parse()
                break
            parser.feed(chunk)
            parser.try_parse()
    if parser.header is None:
        raise ValueError(f"Failed to parse GGUF header from {path}")
    if not parser.is_complete():
        raise ValueError(
            f"Truncated GGUF metadata in {path}: parsed "
            f"{len(parser.kv_pairs)}/{parser.header.metadata_kv_count} metadata "
            f"pairs and {len(parser.tensor_infos)}/{parser.header.tensor_count} "
            "tensor infos"
        )
    return GGUFMetadata(
        header=parser.header,
        kv_pairs=list(parser.kv_pairs),
        tensor_infos=list(parser.tensor_infos),
    )
