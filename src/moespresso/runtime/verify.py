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
    read_artifact,
    validate_base,
)
from moespresso.inventory.safetensors_header import read_header
from moespresso.package.manifest import PACKAGE_FORMAT, PACKAGE_FORMAT_VERSION
from moespresso.runtime.deepseek_v4.expert_layout import (
    EXPERT_SELECTION_FILENAME,
    NUM_HASH_LAYERS,
    PER_LAYER_EXPERTS_FEATURE,
    DeepseekV4ExpertLayout,
    DeepseekV4ExpertLayoutError,
    parse_deepseek_v4_expert_layout,
    validate_expert_index_counts,
)
from moespresso.runtime.expert_index import build_expert_index
from moespresso.runtime.qwen4.expert_layout import (
    EXPERT_SELECTION_FILENAME as QWEN4_EXPERT_SELECTION_FILENAME,
    PER_LAYER_EXPERTS_FEATURE as QWEN4_PER_LAYER_EXPERTS_FEATURE,
    Qwen4ExpertLayoutError,
    parse_qwen4_expert_layout,
    validate_expert_index_counts as validate_qwen4_expert_index_counts,
)
from moespresso.runtime.qwen4.ple_contract import (
    Qwen4PLEProviderError,
    derive_qwen4_ple_provider_contract,
    parse_qwen4_ple_component_contract,
    qwen4_ple_contract_mismatches,
)


class PackageVerificationError(Exception):
    """The package failed manifest verification; refusing to proceed."""


# Per-format the suffixes a tensor's key_prefix expands to on disk. Mirrors the
# writer (package/write.py) and the manifest's declared formats. A layer's three
# Routed entries share one key_prefix (``...switch_mlp.experts``) and all expand to
# the same per-layer bundle tensor.
_KEY_SUFFIXES = {
    "mxfp4": ("tq_bundle",),
    "kquant": ("tq_bundle",),
    "iqk": ("tq_bundle",),
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


def _verify_drafter_component(manifest: dict, package_dir: Path) -> list[Validation]:
    """Identity checks for the declared draft-model component.

    The component is all-or-nothing. With every declared file present, each
    identity is hashed like any other declared file. With every declared file
    absent, a component marked ``optional`` reads as a clean non-blocking
    entry: the distribution shipped without the drafter and the package
    serves plain. A partially present component is corruption and blocks, as
    does any absence from a component not marked optional.
    """
    drafter = manifest.get("drafter")
    if drafter is None:
        return []
    out: list[Validation] = []
    if not isinstance(drafter, dict):
        out.append(_validation(
            "runtime.invalid_drafter_component",
            "manifest drafter component must be an object",
            path="/drafter",
        ))
        return out
    files = drafter.get("files")
    if not isinstance(files, list) or not files:
        out.append(_validation(
            "runtime.invalid_drafter_component",
            "manifest drafter component declares no files",
            path="/drafter/files",
        ))
        return out

    present: list[bool] = []
    names: list[str] = []
    for index, identity in enumerate(files):
        declared = identity.get("path") if isinstance(identity, dict) else None
        path = _declared_path(
            package_dir,
            declared,
            manifest_path=f"/drafter/files/{index}/path",
            out=out,
        )
        if path is None:
            return out
        names.append(str(declared))
        present.append(path.is_file())

    if not any(present):
        if drafter.get("optional") is True:
            out.append(Validation(
                "info",
                "runtime.drafter_component_absent",
                f"optional drafter component is absent ({len(names)} declared "
                "file(s) not present); the package serves without a drafter",
                path="/drafter",
                phase="runtime",
                blocking=False,
            ))
            return out
        out.append(_validation(
            "runtime.missing_drafter_component",
            "drafter component is declared without optional and none of its "
            "files are present",
            path="/drafter/files",
        ))
        return out
    if not all(present):
        missing = [name for name, here in zip(names, present) if not here]
        out.append(_validation(
            "runtime.partial_drafter_component",
            "drafter component is partially present; missing: "
            + ", ".join(sorted(missing)),
            path="/drafter/files",
            expected=sorted(names),
            actual=sorted(name for name, here in zip(names, present) if here),
        ))
    for index, identity in enumerate(files):
        if not present[index]:
            continue
        issues, _ = _verify_identity(
            package_dir, f"/drafter/files/{index}", identity)
        out.extend(issues)
    return out


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


def _verify_deepseek_v4_expert_layout(
    manifest: dict,
    package_dir: Path,
) -> list[Validation]:
    """Authenticate the compact selection artifact and its bundle geometry."""
    try:
        layout = parse_deepseek_v4_expert_layout(manifest)
    except DeepseekV4ExpertLayoutError as exc:
        return [_validation(
            "runtime.invalid_deepseek_v4_expert_layout",
            f"invalid DeepSeek V4 compact expert layout: {exc}",
            path="/expert_layout/per_layer_experts",
        )]
    if layout is None:
        return []

    out: list[Validation] = []
    files = manifest.get("files")
    declarations = [
        entry for entry in files
        if isinstance(entry, dict)
        and entry.get("path") == EXPERT_SELECTION_FILENAME
    ] if isinstance(files, list) else []
    if len(declarations) != 1:
        out.append(_validation(
            "runtime.expert_selection_identity_missing",
            f"compact packages must declare exactly one {EXPERT_SELECTION_FILENAME} "
            "file identity",
            path="/files",
            expected=1,
            actual=len(declarations),
        ))

    selection_path = _declared_path(
        package_dir,
        EXPERT_SELECTION_FILENAME,
        manifest_path="/expert_layout/per_layer_experts/source_selection_artifact_id",
        out=out,
    )
    selection = None
    if selection_path is not None:
        if not selection_path.is_file():
            out.append(_validation(
                "runtime.missing_expert_selection",
                f"compact package is missing {EXPERT_SELECTION_FILENAME}",
                path=f"/{EXPERT_SELECTION_FILENAME}",
            ))
        else:
            try:
                selection = read_artifact(selection_path)
            except (ArtifactError, OSError, UnicodeError, ValueError, TypeError) as exc:
                out.append(_validation(
                    "runtime.invalid_expert_selection",
                    f"could not authenticate {EXPERT_SELECTION_FILENAME}: {exc}",
                    path=f"/{EXPERT_SELECTION_FILENAME}",
                ))

    if selection is not None:
        if selection.get("artifact_kind") != "deepseek_v4_expert_selection":
            out.append(_validation(
                "runtime.wrong_expert_selection_kind",
                f"{EXPERT_SELECTION_FILENAME} has the wrong artifact kind",
                path=f"/{EXPERT_SELECTION_FILENAME}/artifact_kind",
                expected="deepseek_v4_expert_selection",
                actual=selection.get("artifact_kind"),
            ))
        if selection.get("status") != "valid":
            out.append(_validation(
                "runtime.expert_selection_not_valid",
                f"{EXPERT_SELECTION_FILENAME} must have status 'valid'",
                path=f"/{EXPERT_SELECTION_FILENAME}/status",
                expected="valid",
                actual=selection.get("status"),
            ))
        if selection.get("artifact_id") != layout.source_selection_artifact_id:
            out.append(_validation(
                "runtime.expert_selection_id_mismatch",
                "embedded expert selection id does not match the shipped artifact",
                path=(
                    "/expert_layout/per_layer_experts/"
                    "source_selection_artifact_id"
                ),
                expected=layout.source_selection_artifact_id,
                actual=selection.get("artifact_id"),
            ))
        selection_features = selection.get("required_features")
        if (
            not isinstance(selection_features, list)
            or PER_LAYER_EXPERTS_FEATURE not in selection_features
        ):
            out.append(_validation(
                "runtime.expert_selection_feature_missing",
                f"{EXPERT_SELECTION_FILENAME} does not require "
                f"{PER_LAYER_EXPERTS_FEATURE!r}",
                path=f"/{EXPERT_SELECTION_FILENAME}/required_features",
            ))
        expected_payload = layout.selection_payload()
        for key, expected in expected_payload.items():
            actual = selection.get(key)
            if actual != expected:
                out.append(_validation(
                    "runtime.expert_selection_payload_mismatch",
                    f"embedded expert layout field {key!r} does not match "
                    f"{EXPERT_SELECTION_FILENAME}",
                    path=f"/expert_layout/per_layer_experts/{key}",
                    expected=expected,
                    actual=actual,
                ))

    try:
        index = build_expert_index(package_dir)
        validate_expert_index_counts(layout, index)
    except (
        DeepseekV4ExpertLayoutError,
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        struct.error,
    ) as exc:
        out.append(_validation(
            "runtime.expert_layout_bundle_mismatch",
            f"compact expert layout does not match routed bundle headers: {exc}",
            path="/expert_layout/per_layer_experts/layers",
        ))
    out.extend(_verify_deepseek_v4_router_headers(manifest, package_dir, layout))
    return out


def _verify_qwen4_expert_layout(
    manifest: dict,
    package_dir: Path,
) -> list[Validation]:
    """Authenticate a Qwen compact selection and its bundle counts."""

    try:
        layout = parse_qwen4_expert_layout(manifest)
    except Qwen4ExpertLayoutError as exc:
        return [
            _validation(
                "runtime.invalid_qwen4_expert_layout",
                f"invalid Qwen4 compact expert layout: {exc}",
                path="/expert_layout/per_layer_experts",
            )
        ]
    if layout is None:
        return []

    out: list[Validation] = []
    files = manifest.get("files")
    declarations = (
        [
            entry
            for entry in files
            if isinstance(entry, dict)
            and entry.get("path") == QWEN4_EXPERT_SELECTION_FILENAME
        ]
        if isinstance(files, list)
        else []
    )
    if len(declarations) != 1:
        out.append(
            _validation(
                "runtime.qwen4_expert_selection_identity_missing",
                f"compact packages must declare exactly one "
                f"{QWEN4_EXPERT_SELECTION_FILENAME} file identity",
                path="/files",
                expected=1,
                actual=len(declarations),
            )
        )

    selection_path = _declared_path(
        package_dir,
        QWEN4_EXPERT_SELECTION_FILENAME,
        manifest_path=(
            "/expert_layout/per_layer_experts/source_selection_artifact_id"
        ),
        out=out,
    )
    selection = None
    if selection_path is not None:
        if not selection_path.is_file():
            out.append(
                _validation(
                    "runtime.missing_qwen4_expert_selection",
                    f"compact package is missing {QWEN4_EXPERT_SELECTION_FILENAME}",
                    path=f"/{QWEN4_EXPERT_SELECTION_FILENAME}",
                )
            )
        else:
            try:
                selection = read_artifact(selection_path)
            except (ArtifactError, OSError, UnicodeError, ValueError, TypeError) as exc:
                out.append(
                    _validation(
                        "runtime.invalid_qwen4_expert_selection",
                        f"could not authenticate {QWEN4_EXPERT_SELECTION_FILENAME}: {exc}",
                        path=f"/{QWEN4_EXPERT_SELECTION_FILENAME}",
                    )
                )

    if selection is not None:
        if selection.get("artifact_kind") != "qwen4_expert_selection":
            out.append(
                _validation(
                    "runtime.wrong_qwen4_expert_selection_kind",
                    f"{QWEN4_EXPERT_SELECTION_FILENAME} has the wrong artifact kind",
                    path=f"/{QWEN4_EXPERT_SELECTION_FILENAME}/artifact_kind",
                    expected="qwen4_expert_selection",
                    actual=selection.get("artifact_kind"),
                )
            )
        if selection.get("status") != "valid":
            out.append(
                _validation(
                    "runtime.qwen4_expert_selection_not_valid",
                    f"{QWEN4_EXPERT_SELECTION_FILENAME} must have status 'valid'",
                    path=f"/{QWEN4_EXPERT_SELECTION_FILENAME}/status",
                    expected="valid",
                    actual=selection.get("status"),
                )
            )
        if selection.get("artifact_id") != layout.source_selection_artifact_id:
            out.append(
                _validation(
                    "runtime.qwen4_expert_selection_id_mismatch",
                    "embedded Qwen expert selection id does not match the shipped artifact",
                    path=(
                        "/expert_layout/per_layer_experts/"
                        "source_selection_artifact_id"
                    ),
                    expected=layout.source_selection_artifact_id,
                    actual=selection.get("artifact_id"),
                )
            )
        selection_features = selection.get("required_features")
        if (
            not isinstance(selection_features, list)
            or QWEN4_PER_LAYER_EXPERTS_FEATURE not in selection_features
        ):
            out.append(
                _validation(
                    "runtime.qwen4_expert_selection_feature_missing",
                    f"{QWEN4_EXPERT_SELECTION_FILENAME} does not require "
                    f"{QWEN4_PER_LAYER_EXPERTS_FEATURE!r}",
                    path=f"/{QWEN4_EXPERT_SELECTION_FILENAME}/required_features",
                )
            )
        expected_payload = layout.selection_payload()
        for key, expected in expected_payload.items():
            actual = selection.get(key)
            if actual != expected:
                out.append(
                    _validation(
                        "runtime.qwen4_expert_selection_payload_mismatch",
                        f"embedded Qwen expert layout field {key!r} does not "
                        f"match {QWEN4_EXPERT_SELECTION_FILENAME}",
                        path=f"/expert_layout/per_layer_experts/{key}",
                        expected=expected,
                        actual=actual,
                    )
                )

    try:
        index = build_expert_index(package_dir)
        validate_qwen4_expert_index_counts(layout, index)
    except (
        Qwen4ExpertLayoutError,
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        struct.error,
    ) as exc:
        out.append(
            _validation(
                "runtime.qwen4_expert_layout_bundle_mismatch",
                f"Qwen compact expert layout does not match bundle headers: {exc}",
                path="/expert_layout/per_layer_experts/layers",
            )
        )
    return out


def _verify_deepseek_v4_router_headers(
    manifest: dict,
    package_dir: Path,
    layout: DeepseekV4ExpertLayout,
) -> list[Validation]:
    """Check router axis zero from manifest-declared shard and tensor keys."""
    router_roles = {"moe.router_gate", "moe.router_bias"}
    entries: dict[tuple[int, str], tuple[int, dict]] = {}
    out: list[Validation] = []
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list):
        return [_validation(
            "runtime.invalid_router_manifest",
            "compact package manifest tensors must be a list",
            path="/tensors",
        )]
    for tensor_index, tensor in enumerate(tensors):
        if not isinstance(tensor, dict) or tensor.get("role") not in router_roles:
            continue
        role = tensor["role"]
        layer = tensor.get("layer_index")
        if (
            not isinstance(layer, int)
            or isinstance(layer, bool)
            or layer not in layout.layers
        ):
            out.append(_validation(
                "runtime.invalid_router_manifest",
                f"router tensor {tensor_index} has an invalid compact layer",
                path=f"/tensors/{tensor_index}/layer_index",
                actual=layer,
            ))
            continue
        entry_key = (layer, role)
        if entry_key in entries:
            out.append(_validation(
                "runtime.duplicate_router_tensor",
                f"compact layer {layer} declares duplicate {role} tensors",
                path=f"/tensors/{tensor_index}",
            ))
            continue
        entries[entry_key] = (tensor_index, tensor)

    for layer in sorted(layout.layers):
        # Score routing cannot be reconstructed without both learned tensors.
        # Reduced manifests may omit hash-router passthrough entries; their graph
        # state is checked post-hydration. Any declared hash gate is checked at
        # source width below.
        required_roles = (
            ["moe.router_gate", "moe.router_bias"]
            if layer >= NUM_HASH_LAYERS
            else []
        )
        for role in required_roles:
            if (layer, role) not in entries:
                out.append(_validation(
                    "runtime.missing_router_tensor",
                    f"compact layer {layer} has no manifest {role} tensor",
                    path="/tensors",
                ))

    headers: dict[str, dict | None] = {}
    for (layer, role), (tensor_index, tensor) in sorted(entries.items()):
        shard = tensor.get("shard")
        key = tensor.get("key_prefix")
        tensor_path = f"/tensors/{tensor_index}"
        if not isinstance(shard, str) or not isinstance(key, str) or not key:
            out.append(_validation(
                "runtime.invalid_router_manifest",
                f"compact layer {layer} {role} has no shard/key location",
                path=tensor_path,
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
                    headers[shard] = read_header(shard_path)
                except (
                    AttributeError,
                    OSError,
                    TypeError,
                    UnicodeError,
                    ValueError,
                    struct.error,
                ) as exc:
                    out.append(_validation(
                        "runtime.invalid_router_header",
                        f"could not read router shard {shard}: {exc}",
                        path=f"/{shard}",
                    ))
                    headers[shard] = None
        header = headers[shard]
        if header is None:
            continue
        meta = header.get(key)
        if not isinstance(meta, dict):
            out.append(_validation(
                "runtime.missing_router_tensor_key",
                f"compact layer {layer} {role} key {key!r} is absent from {shard}",
                path=tensor_path,
            ))
            continue
        shape = meta.get("shape")
        expected_rank = 2 if role == "moe.router_gate" else 1
        expected_width = layout.layers[layer].num_experts
        if (
            not isinstance(shape, list)
            or len(shape) != expected_rank
            or not isinstance(shape[0], int)
            or isinstance(shape[0], bool)
            or shape[0] != expected_width
        ):
            out.append(_validation(
                "runtime.router_header_width_mismatch",
                f"compact layer {layer} {role} header shape {shape!r} does not "
                f"declare axis-0 width {expected_width}",
                path=f"/{shard}/{key}",
                expected=expected_width,
                actual=shape[0] if isinstance(shape, list) and shape else None,
            ))
    return out


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


def _is_qwen4_architecture(architecture: object) -> bool:
    if not isinstance(architecture, dict):
        return False
    config = architecture.get("config")
    config_type = config.get("model_type") if isinstance(config, dict) else None
    return any(
        value in {"qwen4_exp", "qwen4_exp_text"}
        for value in (architecture.get("family"), architecture.get("text_model_type"), config_type)
    )


def _verify_qwen4_ple_provider(
    manifest: dict,
    package_dir: Path,
) -> list[Validation]:
    """Bind repeated PLE metadata to independently derived architecture facts."""
    architecture = manifest.get("architecture")
    component = manifest.get("ple_provider")
    if not _is_qwen4_architecture(architecture):
        if component is None:
            return []
        return [_validation(
            "runtime.unexpected_qwen4_ple_provider",
            "ple_provider is declared by a package that is not Qwen4-Exp",
            path="/ple_provider",
        )]
    try:
        expected = derive_qwen4_ple_provider_contract(architecture)
    except Qwen4PLEProviderError as exc:
        return [_validation(
            "runtime.invalid_qwen4_ple_architecture",
            f"could not derive the PLE provider contract: {exc}",
            path="/architecture/config",
        )]
    if expected is None:
        if component is None:
            return []
        return [_validation(
            "runtime.unexpected_qwen4_ple_provider",
            "ple_provider is declared but the architecture has no PLE layer",
            path="/ple_provider",
        )]
    if component is None:
        return [_validation(
            "runtime.missing_qwen4_ple_provider",
            "the Qwen4 architecture declares PLE but the package has no PLE provider",
            path="/ple_provider",
        )]
    try:
        actual = parse_qwen4_ple_component_contract(component)
    except Qwen4PLEProviderError as exc:
        return [_validation(
            "runtime.invalid_qwen4_ple_provider",
            f"PLE provider metadata is invalid: {exc}",
            path="/ple_provider",
        )]
    mismatches = qwen4_ple_contract_mismatches(actual, expected)
    if mismatches:
        return [_validation(
            "runtime.qwen4_ple_contract_mismatch",
            "PLE provider does not match the package architecture: " + ", ".join(mismatches),
            path="/ple_provider",
            expected={name: getattr(expected, name) for name in mismatches},
            actual={name: getattr(actual, name) for name in mismatches},
        )]

    out: list[Validation] = []
    files = manifest.get("files")
    identities = {}
    if isinstance(files, list):
        for identity in files:
            if isinstance(identity, dict) and isinstance(identity.get("path"), str):
                identities[identity["path"]] = identity
    shards = component.get("shards")
    assert isinstance(shards, list)
    seen_paths = set()
    expected_bytes = actual.rows_per_shard * actual.row_bytes
    shard_fields = {"index", "path", "row_start", "row_count"}
    for index, shard in enumerate(shards):
        path = f"/ple_provider/shards/{index}"
        if not isinstance(shard, dict) or set(shard) != shard_fields:
            out.append(_validation(
                "runtime.invalid_qwen4_ple_shard",
                "PLE shards must declare only index, path, row_start, and row_count",
                path=path,
            ))
            continue
        expected_start = index * actual.rows_per_shard
        if (
            shard.get("index") != index
            or shard.get("row_start") != expected_start
            or shard.get("row_count") != actual.rows_per_shard
        ):
            out.append(_validation(
                "runtime.invalid_qwen4_ple_shard",
                "PLE shards must partition padded rows uniformly and in order",
                path=path,
            ))
        declared = shard.get("path")
        if isinstance(declared, str):
            posix = PurePosixPath(declared)
            windows = PureWindowsPath(declared)
            if (
                "\\" in declared
                or posix.is_absolute()
                or windows.is_absolute()
                or bool(windows.drive)
                or any(part in {"", ".", ".."} for part in declared.split("/"))
            ):
                out.append(_validation(
                    "runtime.invalid_qwen4_ple_shard",
                    "PLE shard path must be canonical and package-relative",
                    path=f"{path}/path",
                ))
        _declared_path(
            package_dir,
            declared,
            manifest_path=f"{path}/path",
            out=out,
        )
        if not isinstance(declared, str) or declared in seen_paths:
            out.append(_validation(
                "runtime.invalid_qwen4_ple_shard",
                "PLE shard paths must be non-empty and unique",
                path=f"{path}/path",
            ))
            continue
        seen_paths.add(declared)
        identity = identities.get(declared)
        if not isinstance(identity, dict):
            out.append(_validation(
                "runtime.undeclared_qwen4_ple_shard",
                f"PLE shard {declared!r} has no top-level file identity",
                path=f"{path}/path",
            ))
        elif identity.get("size_bytes") != expected_bytes:
            out.append(_validation(
                "runtime.qwen4_ple_shard_size_mismatch",
                f"PLE shard {declared!r} identity has the wrong byte size",
                path=f"{path}/path",
                expected=expected_bytes,
                actual=identity.get("size_bytes"),
            ))
    return out


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

    # The bundled drafter carries its own all-or-nothing presence contract.
    out.extend(_verify_drafter_component(manifest, package_dir))

    # Qwen4 PLE row geometry and hashing are owned by the architecture. The
    # component only declares where those independently derived rows live.
    out.extend(_verify_qwen4_ple_provider(manifest, package_dir))

    # Compact packages bind one family-specific selection artifact to the
    # per-layer expert counts encoded in bundle headers.
    architecture = manifest.get("architecture")
    family = architecture.get("family") if isinstance(architecture, dict) else None
    if family == "deepseek_v4_flash":
        out.extend(_verify_deepseek_v4_expert_layout(manifest, package_dir))
    elif family == "qwen4_exp":
        out.extend(_verify_qwen4_expert_layout(manifest, package_dir))

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
    """Resolve the common seed field used to reproduce generated sidecars.

    Both views must agree. If one omits the field, semantic comparison against
    the regenerated sidecars exposes the omission.
    """
    out: list[Validation] = []
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
