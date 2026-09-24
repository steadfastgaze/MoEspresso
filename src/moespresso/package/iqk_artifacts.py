"""Validated converted IQ_K routed-expert cell files."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from moespresso.package.iqk_format import (
    IQK_LAYOUT_IK_WIRE,
    iqk_geometry,
    validate_iqk_layout,
)


PROJECTIONS = ("gate", "up", "down")


class IQKArtifactError(ValueError):
    """A converted IQ_K cell set is missing or byte-incompatible."""


def conversion_inventory_files(
    path: str | Path,
    *,
    error_type: type[ValueError] = IQKArtifactError,
) -> dict[str, dict]:
    """Return recorded converted file sizes and hashes by canonical basename."""
    data = json.loads(Path(path).read_text())
    entries = data.get("files") or data.get("artifacts") or data
    out: dict[str, dict] = {}
    if isinstance(entries, dict):
        items = entries.items()
    elif isinstance(entries, list):
        items = [
            (entry.get("name") or entry.get("file") or entry.get("path"), entry)
            for entry in entries
            if isinstance(entry, dict)
        ]
    else:
        raise error_type(f"{path}: unreadable conversion inventory")
    for name, entry in items:
        if name is None or not isinstance(entry, dict):
            continue
        basename = Path(str(name)).name
        if basename in out:
            raise error_type(f"{path}: duplicate converted artifact {basename}")
        size = entry.get("size_bytes", entry.get("size", entry.get("bytes")))
        digest = entry.get("sha256")
        if size is None or digest is None:
            continue
        out[basename] = {"size_bytes": int(size), "sha256": str(digest)}
    if not out:
        raise error_type(f"{path}: conversion inventory records no digests")
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def conversion_inventory_sha256(path: str | Path) -> str:
    """Hash a conversion inventory by canonical JSON content."""
    payload = json.loads(Path(path).read_text())
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class IQKConvertedArtifacts:
    """Read a headerless, row-major converted IQ_K expert cell set."""

    def __init__(
        self,
        root: str | Path,
        members: dict[int, dict[str, str]],
        shapes: dict[int, dict[str, tuple[int, int]]],
        num_experts: int,
        *,
        layout: str = IQK_LAYOUT_IK_WIRE,
        max_open: int = 6,
        error_type: type[ValueError] = IQKArtifactError,
    ):
        self.root = Path(root)
        self.members = {int(key): dict(value) for key, value in members.items()}
        self.num_experts = int(num_experts)
        self.layout = validate_iqk_layout(layout)
        self._max_open = int(max_open)
        self._error_type = error_type
        self._fds: dict[tuple[int, str], int] = {}
        self.cells: dict[tuple[int, str], dict] = {}
        total = 0
        for layer in sorted(self.members):
            for projection in PROJECTIONS:
                codec = self.members[layer][projection]
                geometry = iqk_geometry(codec)
                out_features, in_features = shapes[layer][projection]
                bytes_per_row = geometry.bytes_per_row(in_features)
                bytes_per_expert = out_features * bytes_per_row
                path = self.root / f"layer{layer:02d}_{projection}.{codec}"
                if not path.exists():
                    raise self._error(f"converted artifact missing: {path}")
                if path.is_symlink() or not path.is_file():
                    raise self._error(
                        f"converted artifact is not a regular file: {path}"
                    )
                size = path.stat().st_size
                expected = bytes_per_expert * self.num_experts
                if size != expected:
                    raise self._error(
                        f"{path}: size {size} != {expected} "
                        f"({self.num_experts} experts x {out_features} rows x "
                        f"{bytes_per_row} B at {codec})"
                    )
                total += size
                self.cells[(layer, projection)] = {
                    "path": path,
                    "codec": codec,
                    "out_features": int(out_features),
                    "in_features": int(in_features),
                    "bytes_per_row": int(bytes_per_row),
                    "bytes_per_expert": int(bytes_per_expert),
                    "size_bytes": int(size),
                    "bpw": geometry.bpw(in_features),
                    "layout": self.layout,
                }
        self.total_bytes = total

    def _error(self, message: str) -> ValueError:
        return self._error_type(message)

    def layers(self) -> list[int]:
        return sorted(self.members)

    def _fd(self, layer: int, projection: str) -> int:
        key = (layer, projection)
        descriptor = self._fds.get(key)
        if descriptor is None:
            if len(self._fds) >= self._max_open:
                oldest = next(iter(self._fds))
                os.close(self._fds.pop(oldest))
            descriptor = os.open(self.cells[key]["path"], os.O_RDONLY)
            self._fds[key] = descriptor
        return descriptor

    def expert_blocks(self, layer: int, expert_index: int, projection: str) -> np.ndarray:
        """Return one expert cell as ``[out_features, bytes_per_row]`` bytes."""
        cell = self.cells.get((int(layer), str(projection)))
        if cell is None:
            raise self._error(f"no converted cell for layer {layer} {projection}")
        if not 0 <= int(expert_index) < self.num_experts:
            raise self._error(
                f"expert {expert_index} outside [0, {self.num_experts})"
            )
        byte_count = cell["bytes_per_expert"]
        offset = int(expert_index) * byte_count
        raw = os.pread(self._fd(int(layer), str(projection)), byte_count, offset)
        if len(raw) != byte_count:
            raise self._error(
                f"{cell['path']}: short read of {len(raw)} B at offset {offset}"
            )
        return np.frombuffer(raw, dtype=np.uint8).reshape(
            cell["out_features"], cell["bytes_per_row"]
        )

    def verify_digests(self, inventory_path: str | Path) -> dict:
        """Require every cell to reproduce its conversion inventory digest."""
        recorded = conversion_inventory_files(
            inventory_path,
            error_type=self._error_type,
        )
        checked = 0
        for cell in self.cells.values():
            name = cell["path"].name
            wanted = recorded.get(name)
            if wanted is None:
                raise self._error(f"{inventory_path}: no recorded digest for {name}")
            if int(wanted["size_bytes"]) != cell["size_bytes"]:
                raise self._error(
                    f"{name}: size {cell['size_bytes']} != recorded "
                    f"{wanted['size_bytes']}"
                )
            actual = sha256_file(cell["path"])
            if actual != wanted["sha256"]:
                raise self._error(
                    f"{name}: sha256 {actual} != recorded {wanted['sha256']}"
                )
            checked += 1
        return {
            "files_checked": checked,
            "bytes_checked": self.total_bytes,
            "inventory_sha256": conversion_inventory_sha256(inventory_path),
        }

    def identity(self) -> dict:
        by_codec: dict[str, int] = {}
        geometry = []
        for (layer, projection), cell in sorted(self.cells.items()):
            by_codec[cell["codec"]] = by_codec.get(cell["codec"], 0) + 1
            geometry.append(
                {
                    "layer_index": layer,
                    "projection": projection,
                    "codec": cell["codec"],
                    "out_features": cell["out_features"],
                    "in_features": cell["in_features"],
                    "bytes_per_row": cell["bytes_per_row"],
                    "bytes_per_expert": cell["bytes_per_expert"],
                    "size_bytes": cell["size_bytes"],
                }
            )
        return {
            "cells": len(self.cells),
            "layers": len(self.members),
            "num_experts": self.num_experts,
            "routed_bytes": self.total_bytes,
            "layout": self.layout,
            "member_counts": dict(sorted(by_codec.items())),
            "cell_geometry": geometry,
        }

    def close(self) -> None:
        for descriptor in self._fds.values():
            os.close(descriptor)
        self._fds.clear()
