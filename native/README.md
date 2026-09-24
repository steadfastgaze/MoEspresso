# Native extensions with uv

Build from the checkout on arm64 macOS 26.2 or later, with Apple's command-line
tools and Metal SDK installed:

```sh
uv sync --locked
uv run --locked moespresso serve /path/to/model-package
```

The scikit-build-core backend uses CMake to compile the gate extension
and installs it under `moespresso/_native`. Build
dependencies are isolated from the runtime environment. The runtime MLX version
is pinned to the version used to compile the extension because it consumes its
C++ ABI. No separate dependency checkout or runtime archive is required.

uv tracks native sources and CMake configuration, rebuilding the package when
they change. To force a rebuild, use
`uv sync --locked --reinstall-package moespresso`. The compatibility script
`native/build.sh` runs that same command. Imports and server startup never
compile code; the native visibility self-test runs when the runtime first loads
the gate, not during installation. Unset any `MOESPRESSO_NATIVE_DIR` override
before serving to use the installed extension.

The runtime dependency pins remain in `pyproject.toml` and `uv.lock`. The
published mlx-iqk wheel and pinned public mlx-kquant source are resolved by uv;
the latter may need a local build when no compatible cached wheel exists.
Python wheels contain interpreter-specific arm64 extensions and require the
same macOS minimum as MoEspresso. Source installations require the Metal SDK;
installing a compatible prebuilt wheel does not compile the extension.

Run the required native-gate checks with `sh native/run_native_tests.sh`.

After generating an answer, inspect `GET /health`. For bounded
Qwen-architecture serving, `shared_pooled_decode` and `native_gate_loaded`
should be true and `qwen_native_publication_published` should increase.
`native_gate_bound` is request-local and returns to false when the request
finishes. These counters are observational, not an atomic snapshot during
generation.

Model weights are separate from the checkout. Copy the complete package,
including PLE data, the manifest, tokenizer and hotlist. For an SSD-streaming
speed test, use the intended serving SSD and preserve free-space headroom.
