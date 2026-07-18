"""Verify a package against its ``package_manifest`` before loading weights.

PURE (no mlx/jang): the on-demand "never trust, verify" gate. Verification is
fail-closed over the manifest contract and content identity, every file identity
the manifest declares, and every declared tensor key. Runtime compatibility
sidecars have a separate semantic check because package writers legitimately run
``verify_package`` before those generated files exist.

Both entry points return :class:`Validation` entries (empty means clean). The CLI
treats every blocking entry as a hard stop.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path, PurePosixPath, PureWindowsPath

from moespresso.core.artifact import (
    ArtifactError,
    Validation,
    compute_artifact_id,
    validate_base,
)
from moespresso.inventory.safetensors_header import read_header
from moespresso.package.manifest import PACKAGE_FORMAT, PACKAGE_FORMAT_VERSION


class PackageVerificationError(Exception):
    """The package failed manifest verification; refusing to proceed."""


# Per-format the suffixes a tensor's key_prefix expands to on disk. Mirrors the
# writer (package/write.py) and the manifest's declared formats. A layer's three
# tq entries share one key_prefix (``...switch_mlp.experts``) and all expand to
# the same per-layer bundle tensor.
_KEY_SUFFIXES = {
    "tq": ("tq_bundle",),
    "mxfp4": ("tq_bundle",),
    "kquant": ("tq_bundle",),
    "mxfp8": ("weight", "scales"),
    "affine": ("weight", "scales", "biases"),
    "fp16": (None,),  # the prefix itself is the key
    "f32_passthrough": (None,),
    "raw_dtype_passthrough": (None,),
}

_SIDECAR_NAMES = ("config.json", "jang_config.json")


def _validation(code: str, message: str, *, path: str = "", **fields) -> Validation:
    return Validation(
        "error",
        code,
        message,
        path=path,
        phase="runtime",
        blocking=True,
        **fields,
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _declared_path(
    package_dir: Path,
    value: object,
    *,
    manifest_path: str,
    out: list[Validation],
) -> Path | None:
    """Resolve a manifest path while refusing absolute/traversal/symlink escapes."""
    if not isinstance(value, str) or not value:
        out.append(_validation(
            "runtime.invalid_declared_path",
            f"declared path must be a non-empty string, got {value!r}",
            path=manifest_path,
        ))
        return None

    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    normalized_parts = value.replace("\\", "/").split("/")
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part == ".." for part in normalized_parts)
    ):
        out.append(_validation(
            "runtime.unsafe_declared_path",
            f"declared path {value!r} escapes the package root",
            path=manifest_path,
        ))
        return None

    try:
        root = package_dir.resolve()
        resolved = (package_dir / value).resolve()
    except (OSError, RuntimeError) as exc:
        out.append(_validation(
            "runtime.invalid_declared_path",
            f"could not resolve declared path {value!r}: {exc}",
            path=manifest_path,
        ))
        return None
    try:
        resolved.relative_to(root)
    except ValueError:
        out.append(_validation(
            "runtime.unsafe_declared_path",
            f"declared path {value!r} resolves outside the package root",
            path=manifest_path,
        ))
        return None
    return resolved


def _manifest_issues(manifest: dict) -> list[Validation]:
    out: list[Validation] = []
    try:
        out.extend(validate_base(manifest))
    except (ArtifactError, AttributeError, TypeError, ValueError) as exc:
        out.append(_validation(
            "runtime.manifest_contract",
            f"manifest violates the artifact base contract: {exc}",
            path="/",
        ))

    if manifest.get("artifact_kind") != "package_manifest":
        out.append(_validation(
            "runtime.wrong_artifact_kind",
            "verification requires a package_manifest artifact",
            path="/artifact_kind",
            expected="package_manifest",
            actual=manifest.get("artifact_kind"),
        ))

    try:
        actual_id = compute_artifact_id(manifest)
    except (ArtifactError, AttributeError, TypeError, ValueError) as exc:
        out.append(_validation(
            "runtime.manifest_content_invalid",
            f"manifest content cannot be hashed canonically: {exc}",
            path="/artifact_id",
        ))
    else:
        if manifest.get("artifact_id") != actual_id:
            out.append(_validation(
                "runtime.manifest_id_mismatch",
                "manifest artifact_id does not match its canonical content",
                path="/artifact_id",
                expected=actual_id,
                actual=manifest.get("artifact_id"),
            ))

    if manifest.get("status") != "valid":
        out.append(_validation(
            "runtime.manifest_not_valid",
            f"manifest status is {manifest.get('status')!r}, not 'valid'",
            path="/status",
            expected="valid",
            actual=manifest.get("status"),
        ))

    if manifest.get("package_format") != PACKAGE_FORMAT:
        out.append(_validation(
            "runtime.package_format_mismatch",
            f"package format {manifest.get('package_format')!r} is not supported",
            path="/package_format",
            expected=PACKAGE_FORMAT,
            actual=manifest.get("package_format"),
        ))
    if manifest.get("package_format_version") != PACKAGE_FORMAT_VERSION:
        out.append(_validation(
            "runtime.package_format_version_mismatch",
            "package format version is not supported",
            path="/package_format_version",
            expected=PACKAGE_FORMAT_VERSION,
            actual=manifest.get("package_format_version"),
        ))

    for key, expected_type in (
        ("architecture", dict),
        ("tensors", list),
        ("required_ops", list),
        ("files", list),
        ("tokenizer", dict),
    ):
        if not isinstance(manifest.get(key), expected_type):
            out.append(_validation(
                "runtime.invalid_manifest_field",
                f"manifest field {key!r} must be a {expected_type.__name__}",
                path=f"/{key}",
            ))

    embedded = manifest.get("validation", [])
    if not isinstance(embedded, list):
        out.append(_validation(
            "runtime.invalid_manifest_validation",
            "manifest validation must be a list",
            path="/validation",
        ))
    else:
        for index, entry in enumerate(embedded):
            if not isinstance(entry, dict):
                out.append(_validation(
                    "runtime.invalid_manifest_validation",
                    f"manifest validation entry {index} must be an object",
                    path=f"/validation/{index}",
                ))
                continue
            if not entry.get("blocking"):
                continue
            out.append(Validation(
                severity=str(entry.get("severity", "error")),
                code=str(entry.get("code", "runtime.embedded_blocking_validation")),
                message=str(entry.get(
                    "message", "manifest contains an embedded blocking validation")),
                path=str(entry.get("path", f"/validation/{index}")),
                phase=str(entry.get("phase", "package")),
                blocking=True,
                expected=entry.get("expected"),
                actual=entry.get("actual"),
            ))
    return out


def _identity_groups(manifest: dict) -> list[tuple[str, object]]:
    groups: list[tuple[str, object]] = []
    files = manifest.get("files", [])
    if isinstance(files, list):
        for index, identity in enumerate(files):
            groups.append((f"/files/{index}", identity))
    tokenizer = manifest.get("tokenizer", {})
    if isinstance(tokenizer, dict):
        tokenizer_files = tokenizer.get("files", [])
        if isinstance(tokenizer_files, list):
            for index, identity in enumerate(tokenizer_files):
                groups.append((f"/tokenizer/files/{index}", identity))
        else:
            groups.append(("/tokenizer/files", tokenizer_files))
    elif tokenizer is not None:
        groups.append(("/tokenizer", tokenizer))
    if "agentic_profile" in manifest:
        groups.append(("/agentic_profile", manifest.get("agentic_profile")))
    return groups


def _verify_identity(
    package_dir: Path,
    manifest_path: str,
    identity: object,
) -> tuple[list[Validation], str | None]:
    out: list[Validation] = []
    if not isinstance(identity, dict):
        out.append(_validation(
            "runtime.invalid_file_identity",
            "declared file identity must be an object",
            path=manifest_path,
        ))
        return out, None

    declared = identity.get("path")
    path = _declared_path(
        package_dir,
        declared,
        manifest_path=f"{manifest_path}/path",
        out=out,
    )
    if path is None:
        return out, declared if isinstance(declared, str) else None

    size_expected = identity.get("size_bytes")
    digest_expected = identity.get("sha256")
    if (
        not isinstance(size_expected, int)
        or isinstance(size_expected, bool)
        or size_expected < 0
    ):
        out.append(_validation(
            "runtime.invalid_file_identity",
            f"{declared} has invalid size_bytes {size_expected!r}",
            path=f"{manifest_path}/size_bytes",
        ))
    if (
        not isinstance(digest_expected, str)
        or len(digest_expected) != 64
        or any(c not in "0123456789abcdef" for c in digest_expected)
    ):
        out.append(_validation(
            "runtime.invalid_file_identity",
            f"{declared} has an invalid sha256 digest",
            path=f"{manifest_path}/sha256",
        ))

    if not path.is_file():
        out.append(_validation(
            "runtime.missing_file",
            f"declared file {declared} not found",
            path=f"/{declared}",
        ))
        return out, declared

    try:
        size = path.stat().st_size
    except OSError as exc:
        out.append(_validation(
            "runtime.file_read_error",
            f"could not stat declared file {declared}: {exc}",
            path=f"/{declared}",
        ))
        return out, declared
    if isinstance(size_expected, int) and not isinstance(size_expected, bool):
        if size != size_expected:
            out.append(_validation(
                "runtime.size_mismatch",
                f"{declared} size {size} != declared {size_expected}",
                path=f"/{declared}",
                expected=size_expected,
                actual=size,
            ))
            return out, declared

    if isinstance(digest_expected, str) and len(digest_expected) == 64:
        try:
            digest = _sha256(path)
        except OSError as exc:
            out.append(_validation(
                "runtime.file_read_error",
                f"could not hash declared file {declared}: {exc}",
                path=f"/{declared}",
            ))
        else:
            if digest != digest_expected:
                out.append(_validation(
                    "runtime.sha256_mismatch",
                    f"{declared} content hash differs from manifest",
                    path=f"/{declared}",
                    expected=digest_expected,
                    actual=digest,
                ))
    return out, declared


def expected_keys(tensor: dict) -> list[str]:
    """The on-disk safetensors keys a manifest tensor entry declares."""
    prefix = tensor["key_prefix"]
    fmt = tensor["format"]
    if fmt not in _KEY_SUFFIXES:
        raise PackageVerificationError(f"unsupported tensor format {fmt!r}")
    if fmt in {"mxfp4", "kquant"} and tensor.get("kind") == "affine":
        suffixes = ("weight", "scales")
    else:
        suffixes = _KEY_SUFFIXES[fmt]
    return [prefix if suffix is None else f"{prefix}.{suffix}" for suffix in suffixes]


def verify_package(manifest: dict, package_dir: Path) -> list[Validation]:
    """Check manifest, declared files, and tensor keys. Empty list means clean."""
    package_dir = Path(package_dir)
    out = _manifest_issues(manifest)

    # Cover every identity-bearing package member, including all sidecars.
    declared_shards: set[str] = set()
    for manifest_path, identity in _identity_groups(manifest):
        issues, declared = _verify_identity(package_dir, manifest_path, identity)
        out.extend(issues)
        if manifest_path.startswith("/files/") and declared is not None:
            declared_shards.add(declared)

    # Every declared tensor key must be present in a manifest-declared shard.
    headers: dict[str, set[str] | None] = {}
    tensors = manifest.get("tensors", [])
    if not isinstance(tensors, list):
        return out
    for index, tensor in enumerate(tensors):
        tensor_path = f"/tensors/{index}"
        if not isinstance(tensor, dict):
            out.append(_validation(
                "runtime.invalid_tensor_entry",
                "manifest tensor entry must be an object",
                path=tensor_path,
            ))
            continue
        shard = tensor.get("shard")
        if shard not in declared_shards:
            out.append(_validation(
                "runtime.undeclared_tensor_shard",
                f"tensor {tensor.get('source_name', index)!r} references shard "
                f"{shard!r}, which has no top-level file identity",
                path=f"{tensor_path}/shard",
            ))
            continue
        if shard not in headers:
            shard_path = _declared_path(
                package_dir,
                shard,
                manifest_path=f"{tensor_path}/shard",
                out=out,
            )
            if shard_path is None or not shard_path.is_file():
                headers[shard] = None
            else:
                try:
                    headers[shard] = set(read_header(shard_path)) - {"__metadata__"}
                except (OSError, UnicodeError, ValueError, AttributeError, struct.error) as exc:
                    out.append(_validation(
                        "runtime.invalid_safetensors_header",
                        f"could not read safetensors header from {shard}: {exc}",
                        path=f"/{shard}",
                    ))
                    headers[shard] = None
        present = headers[shard]
        if present is None:
            continue
        if tensor.get("format") not in _KEY_SUFFIXES:
            out.append(_validation(
                "runtime.unsupported_tensor_format",
                f"{tensor.get('source_name', index)} declares unsupported format "
                f"{tensor.get('format')!r}",
                path=f"{tensor_path}/format",
            ))
            continue
        try:
            keys = expected_keys(tensor)
        except (KeyError, TypeError) as exc:
            out.append(_validation(
                "runtime.invalid_tensor_entry",
                f"tensor entry is incomplete: {exc}",
                path=tensor_path,
            ))
            continue
        for key in keys:
            if key not in present:
                out.append(_validation(
                    "runtime.missing_tensor_key",
                    f"{tensor.get('source_name', index)} declares key {key}, "
                    f"absent in {shard}",
                    path=f"/{tensor.get('source_name', index)}",
                ))
    return out


def _read_json_object(path: Path, *, name: str) -> tuple[dict | None, list[Validation]]:
    if not path.is_file():
        return None, [_validation(
            "runtime.missing_sidecar",
            f"generated runtime sidecar {name} is missing",
            path=f"/{name}",
        )]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, [_validation(
            "runtime.invalid_sidecar_json",
            f"could not read {name} as JSON: {exc}",
            path=f"/{name}",
        )]
    if not isinstance(value, dict):
        return None, [_validation(
            "runtime.invalid_sidecar_json",
            f"{name} must contain a JSON object",
            path=f"/{name}",
        )]
    return value, []


def _integer_seed(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _sidecar_seed(
    manifest: dict,
    config: dict,
    jang_config: dict,
) -> tuple[int, list[Validation]]:
    """Infer the generator seed without pretending K-quant manifests pin one.

    TQ tensors do pin their transform seed, so that value is authoritative.
    K-quant and affine-only manifests do not: for them the two generated views
    must agree, and their common seed is used to reproduce the views. If only
    one view still carries a seed, it is used so the missing/tampered field is
    also exposed by the full semantic comparison.
    """
    out: list[Validation] = []
    tensors = manifest.get("tensors", [])
    if not isinstance(tensors, list):
        tensors = []

    def tensor_seed(tensor: dict) -> int | None:
        params = tensor.get("format_params")
        if not isinstance(params, dict):
            return None
        return _integer_seed(params.get("seed"))

    tq_values = {
        seed
        for tensor in tensors
        if isinstance(tensor, dict) and tensor.get("format") == "tq"
        for seed in [tensor_seed(tensor)]
        if seed is not None
    }
    tq_entries = [
        tensor
        for tensor in tensors
        if isinstance(tensor, dict) and tensor.get("format") == "tq"
    ]
    if len(tq_values) != (1 if tq_entries else 0):
        out.append(_validation(
            "runtime.inconsistent_manifest_seed",
            "TQ tensor entries must declare one consistent integer seed",
            path="/tensors",
        ))

    config_raw = config.get("mxtq_seed")
    jang_raw = jang_config.get("mxtq_seed")
    config_seed = _integer_seed(config_raw)
    jang_seed = _integer_seed(jang_raw)
    for name, raw, parsed in (
        ("config.json", config_raw, config_seed),
        ("jang_config.json", jang_raw, jang_seed),
    ):
        if raw is not None and parsed is None:
            out.append(_validation(
                "runtime.invalid_sidecar_seed",
                f"{name} mxtq_seed must be an integer",
                path=f"/{name}/mxtq_seed",
            ))

    manifest_seed = next(iter(tq_values), None) if len(tq_values) == 1 else None
    if manifest_seed is not None:
        for name, actual in (
            ("config.json", config_seed),
            ("jang_config.json", jang_seed),
        ):
            if actual is not None and actual != manifest_seed:
                out.append(_validation(
                    "runtime.sidecar_seed_mismatch",
                    f"{name} mxtq_seed {actual} differs from manifest TQ seed "
                    f"{manifest_seed}",
                    path=f"/{name}/mxtq_seed",
                    expected=manifest_seed,
                    actual=actual,
                ))
        return manifest_seed, out

    if config_seed is not None and jang_seed is not None and config_seed != jang_seed:
        out.append(_validation(
            "runtime.sidecar_seed_mismatch",
            "config.json and jang_config.json declare different mxtq_seed values",
            path="/jang_config.json/mxtq_seed",
            expected=config_seed,
            actual=jang_seed,
        ))
    return config_seed if config_seed is not None else (jang_seed or 42), out


def verify_generated_sidecars(manifest: dict, package_dir: Path) -> list[Validation]:
    """Compare generated runtime sidecars with the manifest-derived semantics.

    This intentionally remains separate from :func:`verify_package`: writers can
    verify shards immediately after writing them, before generating these views.
    """
    package_dir = Path(package_dir)
    actual: dict[str, dict] = {}
    out: list[Validation] = []
    for name in _SIDECAR_NAMES:
        value, issues = _read_json_object(package_dir / name, name=name)
        out.extend(issues)
        if value is not None:
            actual[name] = value
    if len(actual) != len(_SIDECAR_NAMES):
        return out

    seed, seed_issues = _sidecar_seed(
        manifest,
        actual["config.json"],
        actual["jang_config.json"],
    )
    out.extend(seed_issues)
    try:
        from moespresso.package.sidecars import build_sidecars

        config_expected, jang_expected = build_sidecars(manifest, seed=seed)
    except (KeyError, TypeError, ValueError) as exc:
        out.append(_validation(
            "runtime.sidecar_generation_failed",
            f"could not derive runtime sidecars from the manifest: {exc}",
            path="/architecture",
        ))
        return out

    for name, expected in (
        ("config.json", config_expected),
        ("jang_config.json", jang_expected),
    ):
        if actual[name] != expected:
            out.append(_validation(
                "runtime.sidecar_semantic_mismatch",
                f"{name} does not match the manifest-derived runtime view",
                path=f"/{name}",
            ))
    return out
