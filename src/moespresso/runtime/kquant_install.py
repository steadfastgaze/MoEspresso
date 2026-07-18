"""Lazy K-quant module installation from a MoEspresso manifest.

The manifest/package contract is pure, but serving a K-quant package needs one
compute-edge call: swap the constructed MLX modules to `mlx-kquant` module
classes before loading their uint8 wire bytes. Keep that import here and only
behind the explicit K-quant runtime path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from moespresso.package.kquant_format import KQUANT_GEOMETRY


class KQuantInstallError(RuntimeError):
    pass


def kquant_weight_codec_map_from_manifest(manifest: dict) -> dict[str, str]:
    """Return `{<module>.weight: codec}` for K-quant tensors in a manifest.

    The key is the exact contract `mlx_kquant.nn.install_kquant_modules()`
    consumes. Do not derive it from source tensor names at runtime: the DS4 name
    mapper must put the installer key into the manifest so loading stays
    fail-closed and architecture-local.

    Routed expert bundle entries are handled by the pooled expert installer,
    which reads their bundle metadata and installs bundle-backed projections.
    The generic module swap is for tensors that are loaded directly by module
    weight key.
    """
    out: dict[str, str] = {}
    for tensor in manifest.get("tensors", []):
        if tensor.get("format") != "kquant":
            continue
        if tensor.get("kind") == "expert":
            continue
        params = tensor.get("format_params") or {}
        codec = params.get("kquant_codec")
        if codec not in KQUANT_GEOMETRY:
            raise KQuantInstallError(
                f"{tensor.get('source_name')}: unknown K-quant codec {codec!r}")
        key = tensor.get("module_weight_key")
        if not isinstance(key, str) or not key.endswith(".weight"):
            raise KQuantInstallError(
                f"{tensor.get('source_name')}: K-quant manifest entry must carry "
                "module_weight_key ending in '.weight'")
        previous = out.get(key)
        if previous is not None and previous != codec:
            raise KQuantInstallError(
                f"{key}: conflicting K-quant codecs {previous!r} and {codec!r}")
        out[key] = codec
    return out


def _load_installer() -> Callable[[Any, dict[str, str]], int]:
    try:
        from mlx_kquant.nn import install_kquant_modules
    except ImportError as exc:  # pragma: no cover - tested via monkeypatch
        raise KQuantInstallError(
            "mlx-kquant is required to install K-quant modules. Reinstall MoEspresso "
            "so `mlx_kquant` and ABI-compatible `mlx` are available."
        ) from exc
    return install_kquant_modules


def install_manifest_kquant_modules(
    model: Any,
    manifest: dict,
    *,
    installer: Callable[[Any, dict[str, str]], int] | None = None,
) -> int:
    """Install K-quant modules declared by `manifest` onto `model`.

    Returns the number of swapped modules. A manifest with no K-quant tensors is
    a no-op and does not import `mlx_kquant`.
    """
    codec_map = kquant_weight_codec_map_from_manifest(manifest)
    if not codec_map:
        return 0
    installer = installer or _load_installer()
    installed = int(installer(model, codec_map))
    object.__setattr__(model, "_moespresso_kquant_modules_installed", installed)
    object.__setattr__(
        model,
        "_moespresso_kquant_module_weight_keys",
        tuple(sorted(codec_map)),
    )
    return installed
