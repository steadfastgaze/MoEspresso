"""Monotone q-envelope: pure-function specs.

q[t,c] is measured on sampled rows, so a higher-bit choice can occasionally measure
slightly worse than a lower-bit one (sampling noise). A risk/quality-per-byte greedy
would exploit that to justify fewer bits. The envelope enforces "more bits never gives
less q" before optimization, so that exploitation is impossible. Genuine inversions
beyond the noise band remain visible as invalidation evidence.
"""

from __future__ import annotations

from moespresso.optimize.monotone import (
    monotone_envelope_by_bits,
    non_monotonic_inversions,
)


# --- expert table: {bits: q} ---

def test_envelope_lifts_a_noise_dip_to_the_lower_bit_value():
    # 4-bit measured slightly below 2-bit (noise). Envelope: q(4) >= q(2).
    q = {1: 0.55, 2: 0.88, 4: 0.86}
    env = monotone_envelope_by_bits(q)
    assert env[1] == 0.55
    assert env[2] == 0.88
    assert env[4] == 0.88            # lifted to the running maximum


def test_envelope_is_identity_when_already_monotone():
    q = {1: 0.55, 2: 0.87, 4: 0.99}
    assert monotone_envelope_by_bits(q) == q


def test_envelope_never_lowers_a_value():
    q = {2: 0.90, 3: 0.80, 4: 0.95, 6: 0.93}
    env = monotone_envelope_by_bits(q)
    for b in q:
        assert env[b] >= q[b]
    # non-decreasing across the ladder
    bits = sorted(env)
    assert all(env[bits[i]] <= env[bits[i + 1]] for i in range(len(bits) - 1))


# --- affine table: {(bits, gs): q}, monotone in bits, per fixed gs ---

def test_envelope_handles_affine_bits_gs_keys_per_group_size():
    q = {(2, 64): 0.84, (4, 64): 0.82, (6, 64): 0.97,    # 4@gs64 dips below 2
         (2, 128): 0.80, (4, 128): 0.90, (6, 128): 0.95}
    env = monotone_envelope_by_bits(q)
    assert env[(4, 64)] == 0.84      # lifted within gs=64 ladder
    assert env[(6, 64)] == 0.97
    assert env[(4, 128)] == 0.90     # gs=128 ladder independent, already monotone
    assert env[(2, 128)] == 0.80


# --- dense MX table: {(format, bits, gs): q}, monotone in bits per group size ---

def test_envelope_handles_dense_mx_format_bits_gs_keys_per_group_size():
    q = {
        ("mxfp4", 4, 32): 0.93,
        ("mxfp8", 8, 32): 0.91,      # 8-bit MX measured below 4-bit MX
    }
    env = monotone_envelope_by_bits(q)
    assert env[("mxfp4", 4, 32)] == 0.93
    assert env[("mxfp8", 8, 32)] == 0.93


# --- inversion reporting (the "flag, don't hide" half) ---

def test_small_dip_within_noise_band_is_not_flagged():
    q = {2: 0.88, 4: 0.86}           # 0.02 dip
    assert non_monotonic_inversions(q, noise_band=0.05) == []


def test_large_inversion_beyond_band_is_flagged():
    q = {2: 0.90, 4: 0.60}           # 0.30 genuine inversion
    bad = non_monotonic_inversions(q, noise_band=0.05)
    assert bad and bad[0][:2] == (2, 4)   # (lower_bits, higher_bits, drop)


def test_empty_or_single_entry_tables_are_safe():
    assert monotone_envelope_by_bits({}) == {}
    assert monotone_envelope_by_bits({4: 0.9}) == {4: 0.9}
    assert non_monotonic_inversions({4: 0.9}) == []
