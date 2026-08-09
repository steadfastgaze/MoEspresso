"""Bundle a DSpark drafter sidecar into a built DeepSeek-V4 package.

Produces a new package directory whose files are hard links to the source
package's shards and furniture plus the sidecar's shards and manifest, with
the package manifest extended by a declared ``drafter`` component. The
component carries the identity (path, size, sha256) of every sidecar file,
the sidecar's own artifact id, and the source package's manifest id, so
``moespresso-verify`` covers the drafter bytes and the runtime can resolve
the bundled drafter from the manifest alone.

The component is declared ``optional``: a distribution of the same package
without the drafter files verifies clean and serves without a drafter (the
loader counts the absent component and stays plain). A partially present
component is corruption and fails verification. Removing the drafter from a
bundled package is therefore deleting its declared files, never a rebuild.

Hard links keep the bundled package byte-identical to its inputs at zero
disk cost on the same filesystem; a link that cannot be created falls back
to a copy and the report records which files copied. The package manifest,
any legacy sidecar manifest that needs layout canonicalization, the regenerated
compatibility sidecars, and the bundle report are written fresh. A fresh file
is never written through a hard link because that would corrupt the source.

    uv run --locked moespresso-ds4-dspark-bundle <package> <sidecar> \
        --output <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from moespresso.core.artifact import compute_artifact_id, read_artifact, write_artifact
from moespresso.package.bundle import IQK_CODEC
from moespresso.package.constants import MANIFEST_NAME
from moespresso.package.iqk_format import (
    IQKFormatError,
    normalize_iqk_layout,
    validate_iqk_layout,
)
from moespresso.package.manifest import file_identity

# Restates the sidecar contract constants from
# moespresso.package.deepseek_v4.dspark_sidecar. That builder module imports
# mlx at module scope; this bundler stays importable without it. The
# equalities are pinned by test.
SIDECAR_MANIFEST_NAME = "dspark_sidecar.json"
SIDECAR_KIND = "deepseek_v4_dspark_sidecar"
SIDECAR_SCHEMA_MAJOR = 1

BUNDLE_REPORT_NAME = "dspark_bundle_report.json"

# Fresh-written outputs; never hard-linked from the source package because a
# write through a linked path would mutate the source's inode.
_REWRITTEN_NAMES = (MANIFEST_NAME, "config.json", "jang_config.json",
                    BUNDLE_REPORT_NAME)


class DSparkBundleError(RuntimeError):
    """A bundle input or output that violates the component contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_valid_sidecar_manifest(sidecar_dir: Path) -> dict:
    """Read the sidecar manifest and gate every shard hash, failing closed.

    Mirrors the runtime loader's contract (runtime/deepseek_v4/dspark_load
    ``read_sidecar_manifest``) without importing mlx: artifact kind, schema
    major, recomputed artifact id, and the sha256 of every hashed file.
    """
    manifest_path = Path(sidecar_dir) / SIDECAR_MANIFEST_NAME
    if not manifest_path.is_file():
        raise DSparkBundleError(f"missing sidecar manifest {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    kind = payload.get("artifact_kind")
    if kind != SIDECAR_KIND:
        raise DSparkBundleError(f"unknown sidecar artifact kind {kind!r}")
    major = (payload.get("schema_version") or {}).get("major")
    if major != SIDECAR_SCHEMA_MAJOR:
        raise DSparkBundleError(
            f"unsupported sidecar schema major {major!r} "
            f"(this build: {SIDECAR_SCHEMA_MAJOR})")
    stored_id = payload.get("artifact_id")
    computed_id = compute_artifact_id(payload)
    if stored_id != computed_id:
        raise DSparkBundleError(
            f"sidecar manifest hash mismatch: stored {stored_id} != "
            f"computed {computed_id}")
    file_sha256 = (payload.get("provenance") or {}).get("file_sha256")
    if not isinstance(file_sha256, dict) or not file_sha256:
        raise DSparkBundleError("sidecar manifest has no file hashes")
    for name, expected in sorted(file_sha256.items()):
        shard = Path(sidecar_dir) / name
        if not shard.is_file():
            raise DSparkBundleError(f"missing sidecar shard {shard}")
        actual = _sha256_file(shard)
        if actual != expected:
            raise DSparkBundleError(
                f"sidecar shard {name} hash mismatch: {actual} != "
                f"recorded {expected}")
    return payload


def drafter_component(
    sidecar_manifest: dict,
    identities: list[dict],
    *,
    source_package_manifest_id: str | None,
) -> dict:
    """The manifest ``drafter`` component for a validated DSpark sidecar."""
    member = ((sidecar_manifest.get("provenance") or {})
              .get("iqk_staging") or {}).get("member")
    return {
        "family": "dspark",
        "optional": True,
        "manifest_path": SIDECAR_MANIFEST_NAME,
        "sidecar_artifact_id": sidecar_manifest.get("artifact_id"),
        "experts_format": member if member else "mxfp4",
        "files": sorted(identities, key=lambda f: f["path"]),
        "provenance": {
            "sidecar_kind": SIDECAR_KIND,
            "source_package_manifest_id": source_package_manifest_id,
        },
    }


def _place(source: Path, dest: Path) -> str:
    """Hard-link ``source`` at ``dest``; fall back to a copy across devices."""
    try:
        os.link(source, dest)
        return "link"
    except OSError:
        shutil.copy2(source, dest)
        return "copy"


def _canonical_iqk_layouts(payload: dict, *, sidecar: bool) -> tuple[dict, bool]:
    """Clone an artifact and canonicalize recorded IQ_K layout spellings."""
    canonical = json.loads(json.dumps(payload))
    table = canonical.get("tensors")
    rows = table.items() if sidecar and isinstance(table, dict) else enumerate(
        table if isinstance(table, list) else [])
    changed = False
    for key, row in rows:
        if not isinstance(row, dict) or row.get("format") != IQK_CODEC:
            continue
        params = row if sidecar else row.get("format_params")
        if not isinstance(params, dict):
            raise DSparkBundleError(f"IQ_K tensor {key!r} has no format parameters")
        recorded = params.get("layout")
        normalized = normalize_iqk_layout(recorded)
        try:
            validate_iqk_layout(normalized)
        except IQKFormatError as exc:
            raise DSparkBundleError(f"IQ_K tensor {key!r}: {exc}") from exc
        if normalized != recorded:
            params["layout"] = normalized
            changed = True
    return canonical, changed


def _rewrite_compat_sidecars(out_dir: Path, manifest: dict, seed: int) -> bool:
    """Regenerate config.json/jang_config.json from the manifest that ships.

    ``moespresso-verify`` re-derives them and compares; a package carrying
    none keeps none (the relayout tool's convention).
    """
    config_path = out_dir / "config.json"
    jang_path = out_dir / "jang_config.json"
    from moespresso.package.sidecars import build_sidecars

    config_json, jang_config = build_sidecars(manifest, seed=seed)
    config_path.write_text(json.dumps(config_json, indent=2))
    jang_path.write_text(json.dumps(jang_config, indent=2))
    return True


def _package_seed(package_dir: Path) -> int:
    jang = package_dir / "jang_config.json"
    if jang.is_file():
        value = json.loads(jang.read_text()).get("mxtq_seed")
        if isinstance(value, int):
            return value
    return 42


def bundle_dspark_drafter(
    package_dir: str | Path,
    sidecar_dir: str | Path,
    output_dir: str | Path,
    *,
    verbose: bool = False,
) -> dict:
    """Assemble ``output_dir`` = package + bundled DSpark drafter component."""
    package_dir = Path(package_dir)
    sidecar_dir = Path(sidecar_dir)
    out_dir = Path(output_dir)
    log = (lambda msg: print(msg, flush=True)) if verbose else (lambda msg: None)
    if out_dir.resolve() == package_dir.resolve():
        raise DSparkBundleError(
            "bundle output must be a new directory, not the source package")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise DSparkBundleError(f"bundle output {out_dir} is not empty")

    source_manifest = read_artifact(package_dir / MANIFEST_NAME)
    if (source_manifest.get("architecture") or {}).get("family") != "deepseek_v4_flash":
        raise DSparkBundleError(f"{package_dir} is not a DeepSeek-V4 package")
    if source_manifest.get("drafter") is not None:
        raise DSparkBundleError(
            f"{package_dir} already declares a drafter component")
    manifest, _ = _canonical_iqk_layouts(
        source_manifest, sidecar=False)

    source_sidecar_manifest = read_valid_sidecar_manifest(sidecar_dir)
    sidecar_manifest, sidecar_changed = _canonical_iqk_layouts(
        source_sidecar_manifest, sidecar=True)
    sidecar_names = [SIDECAR_MANIFEST_NAME] + sorted(
        (sidecar_manifest.get("provenance") or {}).get("file_sha256", {}))
    log(f"[1/4] sidecar {source_sidecar_manifest.get('artifact_id')} validated "
        f"({len(sidecar_names)} file(s))")

    package_names = [
        entry.name for entry in sorted(package_dir.iterdir())
        if entry.is_file() and entry.name not in _REWRITTEN_NAMES
    ]
    collisions = sorted(set(package_names) & set(sidecar_names))
    if collisions:
        raise DSparkBundleError(
            "sidecar file names collide with package files: "
            + ", ".join(collisions))

    out_dir.mkdir(parents=True, exist_ok=True)
    placements: dict[str, str] = {}
    for name in package_names:
        placements[name] = _place(package_dir / name, out_dir / name)
    for name in sidecar_names:
        if name == SIDECAR_MANIFEST_NAME and sidecar_changed:
            rewritten = dict(sidecar_manifest)
            rewritten.pop("artifact_id", None)
            sidecar_id = write_artifact(
                out_dir / name,
                rewritten,
                created_at=source_sidecar_manifest.get("created_at"),
            )
            sidecar_manifest["artifact_id"] = sidecar_id
            placements[name] = "rewrite"
        else:
            placements[name] = _place(sidecar_dir / name, out_dir / name)
    linked = sum(1 for how in placements.values() if how == "link")
    copied = sum(1 for how in placements.values() if how == "copy")
    rewritten = sum(1 for how in placements.values() if how == "rewrite")
    log(f"[2/4] placed {len(placements)} file(s): {linked} linked, "
        f"{copied} copied, {rewritten} rewritten")

    identities = [file_identity(out_dir / name) for name in sidecar_names]
    component = drafter_component(
        sidecar_manifest, identities,
        source_package_manifest_id=source_manifest.get("artifact_id"))
    component["provenance"]["source_sidecar_artifact_id"] = (
        source_sidecar_manifest.get("artifact_id"))

    bundled = json.loads(json.dumps(manifest))
    bundled["drafter"] = component
    bundled.pop("artifact_id", None)
    bundled.pop("created_at", None)
    manifest_id = write_artifact(out_dir / MANIFEST_NAME, bundled,
                                 created_at=manifest.get("created_at"))
    rewrote = False
    if (package_dir / "config.json").is_file() and (
            package_dir / "jang_config.json").is_file():
        rewrote = _rewrite_compat_sidecars(
            out_dir, read_artifact(out_dir / MANIFEST_NAME),
            _package_seed(package_dir))
    log(f"[3/4] manifest {manifest_id}"
        + (" (compat sidecars regenerated)" if rewrote else ""))

    sidecar_bytes = sum(identity["size_bytes"] for identity in identities)
    package_bytes = sum(
        int(f.get("size_bytes", 0)) for f in source_manifest.get("files", []))
    report = {
        "package": str(out_dir),
        "source_package": str(package_dir),
        "source_sidecar": str(sidecar_dir),
        "manifest_id": {"before": source_manifest.get("artifact_id"),
                        "after": manifest_id},
        "sidecar_artifact_id": sidecar_manifest.get("artifact_id"),
        "source_sidecar_artifact_id": source_sidecar_manifest.get("artifact_id"),
        "drafter_files": sidecar_names,
        "placements": {"linked": linked, "copied": copied,
                       "rewritten": rewritten,
                       "by_file": placements},
        "bytes": {
            "package_shards": package_bytes,
            "drafter_files": sidecar_bytes,
            "total_with_drafter": package_bytes + sidecar_bytes,
        },
    }
    (out_dir / BUNDLE_REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True))
    log(f"[4/4] report {out_dir / BUNDLE_REPORT_NAME}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", help="built DeepSeek-V4 package")
    parser.add_argument("sidecar_dir", help="DSpark sidecar directory")
    parser.add_argument("--output", required=True,
                        help="new bundled package directory")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    report = bundle_dspark_drafter(
        args.package_dir, args.sidecar_dir, args.output, verbose=args.verbose)
    print(json.dumps({
        "package": report["package"],
        "manifest_id": report["manifest_id"],
        "bytes": report["bytes"],
        "placements": {"linked": report["placements"]["linked"],
                       "copied": report["placements"]["copied"],
                       "rewritten": report["placements"]["rewritten"]},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
