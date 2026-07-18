"""Monotone q-envelope: pure, no model.

The probe measures q[t,c] on sampled rows, so a higher-bit encoding can occasionally
score slightly below a lower-bit one (sampling noise). A quality/risk-per-byte greedy
would exploit that to justify fewer bits. Before optimization, each unit's q-table is
projected onto a monotone envelope ("more bits never yields less q") so the exploitation
is impossible. A genuine inversion beyond the measurement-noise band is reported
(invalidation: re-measure), never silently used.

Three q-table shapes occur (see decide.py): expert `{bits: q}`, affine
`{(bits, gs): q}`, and dense MX `{(format, bits, gs): q}`. For affine and
dense MX, monotonicity is enforced in `bits` independently per group_size (the
bits ladder is the precision axis; group size is a separate knob).
"""

from __future__ import annotations


def _bits_of(key) -> int:
    """Bit-width from a q-table key."""
    if not isinstance(key, tuple):
        return key
    if len(key) == 2:
        return key[0]
    if len(key) == 3:
        return key[1]
    raise ValueError(f"unsupported q-table key shape: {key!r}")


def _group_of(key):
    """Group a key shares its bit-ladder with."""
    if not isinstance(key, tuple):
        return None
    if len(key) == 2:
        return key[1]
    if len(key) == 3:
        return key[2]
    raise ValueError(f"unsupported q-table key shape: {key!r}")


def monotone_envelope_by_bits(quality: dict) -> dict:
    """Return a copy of `quality` made non-decreasing in bits (per group_size).

    Within each group (fixed gs, or the single expert group), walk bits ascending and
    replace each q with the running max, so q never drops as bits rise. Values are only
    ever raised, never lowered; an already-monotone table is returned unchanged (==).
    """
    if not quality:
        return {}
    out = dict(quality)
    groups: dict[object, list] = {}
    for key in quality:
        groups.setdefault(_group_of(key), []).append(key)
    for keys in groups.values():
        running = None
        for key in sorted(keys, key=_bits_of):
            q = quality[key]
            running = q if running is None else max(running, q)
            out[key] = running
    return out


def non_monotonic_inversions(quality: dict, noise_band: float = 0.05) -> list[tuple]:
    """Return inversions whose depth exceeds the sampling-noise band.

    Returns (lower_bits, higher_bits, drop) for each adjacent pair (within a group, by
    ascending bits) where the higher-bit q is more than `noise_band` below the lower-bit
    q. Empty == every dip is within the noise band (safe to smooth via the envelope).
    """
    out: list[tuple] = []
    groups: dict[object, list] = {}
    for key in quality:
        groups.setdefault(_group_of(key), []).append(key)
    for keys in groups.values():
        ordered = sorted(keys, key=_bits_of)
        for lo, hi in zip(ordered, ordered[1:]):
            drop = quality[lo] - quality[hi]
            if drop > noise_band:
                out.append((_bits_of(lo), _bits_of(hi), drop))
    return out
