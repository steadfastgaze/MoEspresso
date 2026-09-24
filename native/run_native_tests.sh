#!/bin/sh
# Require the native gate instead of accepting skipped gate tests.
set -eu
cd "$(dirname "$0")/.."
MOESPRESSO_REQUIRE_NATIVE_GATE=1 uv run --locked pytest \
  tests/test_pooled_switchglu.py -k "gate_decode or ring" -q
