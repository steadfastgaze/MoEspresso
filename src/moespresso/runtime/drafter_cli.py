"""External speculative-drafter discovery for user-facing commands.

The runtime retains family-specific loaders for DSpark, DFlash, and MTP.
The public external-sidecar surface currently accepts DSpark only. Discovery
uses the sidecar root's manifest name and artifact kind, so callers do not
need to know the internal environment-variable syntax.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExternalDrafter:
    """A recognized sidecar root and its drafter family."""

    family: str
    root: Path
    manifest_path: Path
    artifact_kind: str


class ExternalDrafterError(ValueError):
    """An external sidecar root cannot be selected by the public CLI."""


_SIDECAR_CONTRACTS = {
    "dspark_sidecar.json": ("dspark", "deepseek_v4_dspark_sidecar"),
    "dflash_sidecar.json": ("dflash", "deepseek_v4_dflash_sidecar"),
    "mtp_sidecar.json": ("mtp", "deepseek_v4_mtp_sidecar"),
}

# The drafter consumes target hidden states at these geometry-sensitive seams.
# Quantization and retained-expert layout do not affect this compatibility
# contract because the target runtime restores source-width router identities.
_DSPARK_TARGET_CONFIG_KEYS = (
    "hidden_size",
    "vocab_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "q_lora_rank",
    "qk_rope_head_dim",
    "o_lora_rank",
    "o_groups",
    "n_routed_experts",
    "moe_intermediate_size",
    "num_experts_per_tok",
    "dspark_block_size",
    "dspark_noise_token_id",
    "dspark_target_layer_ids",
    "dspark_markov_rank",
)
_DSPARK_MANIFEST_CONFIG_FIELDS = {
    "block_size": "dspark_block_size",
    "noise_token_id": "dspark_noise_token_id",
    "target_layer_ids": "dspark_target_layer_ids",
    "markov_rank": "dspark_markov_rank",
}
_DSPARK_FORMATS = frozenset({"mxfp4", "affine8", "passthrough", "iqk"})


def detect_external_drafter(root: str | Path) -> ExternalDrafter:
    """Recognize one sidecar manifest at ``root`` and accept DSpark only."""
    sidecar_root = Path(root).expanduser()
    if not sidecar_root.is_dir():
        raise ExternalDrafterError(
            f"external drafter directory not found: {sidecar_root}"
        )

    matches = [
        (sidecar_root / name, family, kind)
        for name, (family, kind) in _SIDECAR_CONTRACTS.items()
        if (sidecar_root / name).is_file()
    ]
    if not matches:
        names = ", ".join(sorted(_SIDECAR_CONTRACTS))
        raise ExternalDrafterError(
            f"{sidecar_root} has no recognized drafter manifest; expected "
            f"exactly one of {names}"
        )
    if len(matches) != 1:
        names = ", ".join(path.name for path, _family, _kind in matches)
        raise ExternalDrafterError(
            f"{sidecar_root} has multiple drafter manifests: {names}"
        )

    manifest_path, family, expected_kind = matches[0]
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExternalDrafterError(
            f"cannot read external drafter manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ExternalDrafterError(
            f"external drafter manifest {manifest_path} must contain a JSON object"
        )
    actual_kind = payload.get("artifact_kind")
    if actual_kind != expected_kind:
        raise ExternalDrafterError(
            f"{manifest_path.name} has artifact kind {actual_kind!r}; "
            f"expected {expected_kind!r}"
        )
    if family != "dspark":
        raise ExternalDrafterError(
            f"{family} sidecars are recognized but are not supported by the "
            "external CLI in this release; --drafter currently accepts DSpark"
        )
    return ExternalDrafter(
        family=family,
        root=sidecar_root,
        manifest_path=manifest_path,
        artifact_kind=expected_kind,
    )


def verify_external_dspark(
    package_manifest: dict, external: ExternalDrafter
) -> dict:
    """Verify DSpark bytes and target geometry without constructing a model."""
    if external.family != "dspark":
        raise ExternalDrafterError(
            f"external drafter family {external.family!r} is not DSpark"
        )

    # This verifier mirrors the runtime loader's artifact-id, containment, and
    # per-file SHA checks without importing MLX.
    from moespresso.package.deepseek_v4.dspark_bundle import (
        DSparkBundleError,
        read_valid_sidecar_manifest,
    )

    try:
        sidecar_manifest = read_valid_sidecar_manifest(external.root)
    except (DSparkBundleError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExternalDrafterError(str(exc)) from exc

    architecture = package_manifest.get("architecture") or {}
    if architecture.get("family") != "deepseek_v4_flash":
        raise ExternalDrafterError(
            "external DSpark sidecars require a DeepSeek-V4-Flash package"
        )
    target_config = architecture.get("config") or {}
    source_config = sidecar_manifest.get("source_config") or {}
    mismatches = [
        key
        for key in _DSPARK_TARGET_CONFIG_KEYS
        if key not in target_config
        or key not in source_config
        or target_config.get(key) != source_config.get(key)
    ]
    if mismatches:
        raise ExternalDrafterError(
            "external DSpark sidecar is incompatible with the package config: "
            + ", ".join(mismatches)
        )

    if (sidecar_manifest.get("subject") or {}).get("family") != (
        "deepseek_v4_flash_dspark"
    ):
        raise ExternalDrafterError(
            "external DSpark sidecar has an invalid subject family"
        )
    dspark = sidecar_manifest.get("dspark") or {}
    dspark_mismatches = [
        field
        for field, config_key in _DSPARK_MANIFEST_CONFIG_FIELDS.items()
        if field not in dspark or dspark.get(field) != source_config.get(config_key)
    ]
    try:
        stage_count = int(dspark.get("n_mtp_layers", 0))
    except (TypeError, ValueError):
        stage_count = 0
    if dspark_mismatches or stage_count < 1:
        details = list(dspark_mismatches)
        if stage_count < 1:
            details.append("n_mtp_layers")
        raise ExternalDrafterError(
            "external DSpark sidecar has an inconsistent DSpark contract: "
            + ", ".join(details)
        )

    tensors = sidecar_manifest.get("tensors")
    file_hashes = (sidecar_manifest.get("provenance") or {}).get("file_sha256")
    if not isinstance(tensors, dict) or not tensors:
        raise ExternalDrafterError("external DSpark sidecar has no tensor table")
    invalid_tensors = [
        name
        for name, row in tensors.items()
        if not isinstance(row, dict)
        or row.get("format") not in _DSPARK_FORMATS
        or row.get("file") not in file_hashes
    ]
    if invalid_tensors:
        raise ExternalDrafterError(
            "external DSpark sidecar has invalid tensor rows: "
            + ", ".join(sorted(invalid_tensors)[:8])
        )

    package_source = str(
        (package_manifest.get("subject") or {}).get("source_root") or ""
    ).rstrip("/").rsplit("/", 1)[-1]
    sidecar_source = str(
        (sidecar_manifest.get("provenance") or {}).get("source_snapshot") or ""
    ).rstrip("/").rsplit("/", 1)[-1]
    if not package_source or not sidecar_source:
        raise ExternalDrafterError(
            "external DSpark sidecar or package has no source checkpoint identity"
        )
    if package_source != sidecar_source:
        raise ExternalDrafterError(
            "external DSpark sidecar source does not match the package: "
            f"{sidecar_source!r} != {package_source!r}"
        )
    return sidecar_manifest


def add_external_drafter_argument(parser) -> None:
    """Add the shared DSpark sidecar option to a user-facing parser."""
    parser.add_argument(
        "--drafter",
        metavar="PATH",
        help="Use an external DSpark sidecar directory. This option overrides "
        "MOESPRESSO_DS4_DRAFTER and bundled automatic selection.",
    )


def parse_external_drafter_argument(parser, value: str | None) -> Path | None:
    """Return the validated DSpark root or report a normal CLI usage error."""
    if value is None:
        return None
    try:
        return detect_external_drafter(value).root
    except ExternalDrafterError as exc:
        parser.error(str(exc))
    raise AssertionError("argparse.error must terminate")
