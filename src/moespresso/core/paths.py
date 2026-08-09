"""Fail-closed resolution for file names declared by artifacts."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath


class UnsafeArtifactPathError(ValueError):
    """An artifact-declared file name does not stay within its root."""


def resolve_artifact_file(root: str | Path, value: object) -> Path:
    """Resolve one flat artifact file name below ``root``.

    Package manifests use file names, not paths. Rejecting separators also
    keeps the contract independent of the host platform and prevents a
    manifest from turning a package operation into a read or write elsewhere.
    Existing symlinks are resolved before the containment check.
    """
    if not isinstance(value, str) or not value:
        raise UnsafeArtifactPathError(
            f"artifact file name must be a non-empty string, got {value!r}")

    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    normalized = value.replace("\\", "/")
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or "/" in normalized
        or value in (".", "..")
    ):
        raise UnsafeArtifactPathError(
            f"artifact file name {value!r} is not a root-relative file name")

    try:
        resolved_root = Path(root).resolve()
        resolved = (Path(root) / value).resolve()
        resolved.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise UnsafeArtifactPathError(
            f"artifact file name {value!r} escapes root {Path(root)}") from exc
    return resolved
