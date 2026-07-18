"""RAM-aware chunk sizing, pure, no mlx/jang.

The in-memory affine/fp16 row-band is the knob that can OOM conversion. These pin
the sizing policy: it scales with free RAM and clamps to a floor/ceiling.
"""

from __future__ import annotations

from moespresso.package.write import (
    _CHUNK_CEILING_BYTES,
    _CHUNK_FLOOR_BYTES,
    safe_chunk_bytes,
)


def test_scales_with_free_ram():
    # 10% of 8 GB = 800 MB, but the ceiling caps it.
    assert safe_chunk_bytes(8 * 1024**3) == _CHUNK_CEILING_BYTES
    # 10% of 2 GB = ~205 MB, between floor and ceiling -> used as-is.
    assert safe_chunk_bytes(2 * 1024**3) == int(2 * 1024**3 * 0.10)


def test_low_free_ram_clamps_to_floor():
    # With ~100 MB free: 10% = 10 MB < floor -> floor (small, safe band).
    assert safe_chunk_bytes(100 * 1024**2) == _CHUNK_FLOOR_BYTES
    # Even with ~0 free, never below the floor (no degenerate 1-byte reads).
    assert safe_chunk_bytes(0) == _CHUNK_FLOOR_BYTES


def test_never_exceeds_ceiling():
    assert safe_chunk_bytes(1024**4) == _CHUNK_CEILING_BYTES  # 1 TB free
    assert safe_chunk_bytes(10**15) == _CHUNK_CEILING_BYTES


def test_custom_fraction_and_bounds():
    # explicit fraction/floor/ceiling are honored.
    assert safe_chunk_bytes(1000, fraction=0.5, floor=100, ceiling=400) == 400
    assert safe_chunk_bytes(1000, fraction=0.2, floor=100, ceiling=400) == 200
    assert safe_chunk_bytes(100, fraction=0.2, floor=100, ceiling=400) == 100


def test_monotonic_in_free_ram():
    # more free RAM never yields a smaller band (within the clamps).
    sizes = [safe_chunk_bytes(b) for b in
             (64 * 1024**2, 512 * 1024**2, 2 * 1024**3, 16 * 1024**3)]
    assert sizes == sorted(sizes)


def test_autosize_returns_a_bounded_value():
    # whatever the host reports, the auto value sits within [floor, ceiling].
    from moespresso.package.write import _autosize_chunk_bytes
    v = _autosize_chunk_bytes()
    assert _CHUNK_FLOOR_BYTES <= v <= _CHUNK_CEILING_BYTES
