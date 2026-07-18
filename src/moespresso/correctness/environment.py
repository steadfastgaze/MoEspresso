"""Environment facts recorded in gate evidence.

The installed mlx wheel variant moves knife-edge quality anchors: the
macosx_15_0_arm64 and macosx_26_0_arm64 lattices of the same mlx version
produce different Q1/Q2 realizations on the same package, and the variant
is chosen by the installer at sync time, so a cache-less reinstall can
silently move the lattice. Gate outputs record the installed wheel tag so
an anchor shift is attributable to the environment instead of surfacing
as a quality mystery. The helpers only record this fact; they
never raise, so a gate run on a host with no readable dist-info still
completes.
"""

from __future__ import annotations

from importlib import metadata

MLX_WHEEL_UNKNOWN = "unknown"


def mlx_wheel_tag() -> str:
    """Return the Tag value from the installed mlx dist-info WHEEL file.

    Reads the wheel metadata of the installed ``mlx`` distribution and
    returns its ``Tag:`` line (for example
    ``cp313-cp313-macosx_26_0_arm64``). Multi-tag wheels return the tags
    joined with commas in file order. Returns ``"unknown"`` on any
    failure: mlx not installed, no WHEEL file in the dist-info, or no Tag
    line. Never raises.
    """
    try:
        text = metadata.distribution("mlx").read_text("WHEEL")
        if not text:
            return MLX_WHEEL_UNKNOWN
        tags = []
        for line in text.splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip().lower() == "tag":
                tag = value.strip()
                if tag:
                    tags.append(tag)
        if not tags:
            return MLX_WHEEL_UNKNOWN
        return ",".join(tags)
    except Exception:
        return MLX_WHEEL_UNKNOWN
