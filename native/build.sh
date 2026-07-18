#!/bin/sh
# Build the MoEspresso native extensions against the venv's MLX.
#
# Portability note: some CommandLineTools installs ship a clang that looks for
# libc++ headers in the TOOLCHAIN copy (CLT/usr/include/c++/v1) which can be
# missing after partial CLT updates, while the SDK always carries a full
# copy. We probe a trivial compile and only add the SDK-libc++ flags when
# the plain toolchain is broken. Reinstalling CLT
# (sudo rm -rf /Library/Developer/CommandLineTools && xcode-select --install)
# fixes it properly; the probe makes the build work either way.
#
# nanobind must match MLX's pin (v2.12.0 for mlx 0.31.x) and the module is
# built NB_DOMAIN mlx so mx.array casters are shared.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
SDK="$(xcrun --show-sdk-path)"
PY="${MOESPRESSO_PYTHON:-$(cd "$ROOT/.." && pwd)/.venv/bin/python}"

EXTRA_CXX_FLAGS=""
if ! printf '#include <cstddef>\nint main(){return 0;}\n' \
    | clang++ -x c++ -std=c++17 -c - -o /dev/null 2>/dev/null; then
  echo "toolchain libc++ headers broken; falling back to the SDK copy"
  EXTRA_CXX_FLAGS="-nostdinc++ -isystem $SDK/usr/include/c++/v1"
fi

build_one() {
  name="$1"
  cd "$ROOT/$name"
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$PY" \
    -DCMAKE_OSX_SYSROOT="$SDK" \
    ${EXTRA_CXX_FLAGS:+-DCMAKE_CXX_FLAGS="$EXTRA_CXX_FLAGS"}
  cmake --build build -j8
  echo "built $name"
}

build_one gate
build_one ds4_moe
