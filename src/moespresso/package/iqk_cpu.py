"""Load the pinned mlx-iqk CPU codec and format without importing MLX.

The published ``mlx_iqk`` package exports its neural-network surface from
``__init__``.  Importing that package therefore imports MLX even when a build
step needs only the NumPy relayout and ctypes CPU quantizer.  This loader reads
the two pure host modules directly from the pinned distribution and supplies
their absolute intra-package imports only while the modules execute.
"""

from __future__ import annotations

from importlib import util
from importlib.metadata import PackageNotFoundError, distribution
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
from threading import Lock
from types import ModuleType


class IQKCPULoadError(RuntimeError):
    """The pinned CPU-only mlx-iqk modules could not be loaded."""


_LOCK = Lock()
_FORMAT = None
_CODEC = None
_ALIAS_PREFIX = "_moespresso_mlx_iqk_cpu"


def _module_root() -> Path:
    try:
        root = Path(distribution("mlx-iqk").locate_file("mlx_iqk")).resolve()
    except PackageNotFoundError as exc:
        raise IQKCPULoadError("mlx-iqk is not installed") from exc
    required = {name: root / f"{name}.py" for name in ("iq1grid", "format", "codec")}
    missing = [path for path in required.values() if not path.is_file()]
    if missing:
        raise IQKCPULoadError(f"mlx-iqk CPU module is missing: {missing[0]}")
    return root


def _load(alias: str, path: Path):
    spec = util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise IQKCPULoadError(f"could not load mlx-iqk CPU module {path}")
    module = util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(alias, None)
        raise
    return module


def _load_modules() -> tuple[ModuleType, ModuleType]:
    root = _module_root()
    watched = ("mlx_iqk", "mlx_iqk.iq1grid", "mlx_iqk.format")
    saved = {name: sys.modules.get(name) for name in watched}
    stub = ModuleType("mlx_iqk")
    stub.__path__ = [str(root)]
    stub.__package__ = "mlx_iqk"
    stub.__spec__ = ModuleSpec("mlx_iqk", loader=None, is_package=True)
    format_alias = f"{_ALIAS_PREFIX}.format"
    codec_alias = f"{_ALIAS_PREFIX}.codec"
    try:
        sys.modules["mlx_iqk"] = stub
        grid = _load("mlx_iqk.iq1grid", root / "iq1grid.py")
        stub.iq1grid = grid
        format_module = _load(format_alias, root / "format.py")
        sys.modules["mlx_iqk.format"] = format_module
        stub.format = format_module
        codec_module = _load(codec_alias, root / "codec.py")
    except Exception as exc:
        sys.modules.pop(format_alias, None)
        sys.modules.pop(codec_alias, None)
        raise IQKCPULoadError(f"could not load mlx-iqk CPU modules: {exc}") from exc
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return format_module, codec_module


def _modules() -> tuple[ModuleType, ModuleType]:
    global _FORMAT, _CODEC
    if _FORMAT is not None and _CODEC is not None:
        return _FORMAT, _CODEC
    with _LOCK:
        if _FORMAT is None or _CODEC is None:
            _FORMAT, _CODEC = _load_modules()
    return _FORMAT, _CODEC


def iqk_format() -> ModuleType:
    """Return the pure NumPy format module from the pinned distribution."""
    return _modules()[0]


def iqk_codec() -> ModuleType:
    """Return the ctypes CPU codec module from the pinned distribution."""
    return _modules()[1]


__all__ = ["IQKCPULoadError", "iqk_codec", "iqk_format"]
