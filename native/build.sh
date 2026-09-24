#!/bin/sh
# Compatibility entry point for an explicit package rebuild.
set -eu
cd "$(dirname "$0")/.."
exec uv sync --locked --reinstall-package moespresso
