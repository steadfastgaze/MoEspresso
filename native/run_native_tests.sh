#!/bin/sh
# The unambiguous native-gate test target:
# builds the extension, then runs the gate tests in REQUIRED mode (missing
# or self-test-failing native build FAILS instead of skipping) plus the ring
# visibility tests. "Green" here means NATIVE-GATE green, not fallback green.
#
# Python selection: honor MOESPRESSO_PYTHON, else prefer
# `uv run --locked` when a uv.lock exists (guarantees the project env on any
# checkout), else fall back to .venv. A wrong interpreter here fails with
# "No Metal device available" from mlx.nn, a false red.
set -e
cd "$(dirname "$0")/.."
if [ -n "$MOESPRESSO_PYTHON" ]; then
  RUN="$MOESPRESSO_PYTHON"
  BUILD="./native/build.sh"
elif command -v uv > /dev/null 2>&1 && [ -f uv.lock ]; then
  RUN="uv run --locked --group native python"
  BUILD="uv run --locked --group native ./native/build.sh"
else
  RUN=".venv/bin/python"
  BUILD="./native/build.sh"
fi
$BUILD
MOESPRESSO_REQUIRE_NATIVE_GATE=1 $RUN -m pytest \
  tests/test_pooled_switchglu.py -k "gate_decode or ring" -q
echo "NATIVE-GATE TARGET: GREEN"
