# Distribution

Source archives include `LICENSE-MIT`, `LICENSE-APACHE-2.0` and
`THIRD-PARTY-NOTICES`. The latter covers adapted source, with project-level
attribution and retained license texts.

## Binary releases

A binary release includes:

- The executable and the three source license/notice files.
- Dependency license texts generated for its locked dependencies and target.
- The matching Rust standard-library copyright report and license texts,
  including compiler-builtins runtime support.

Inspect the final archive before publishing.

## Dependency licenses

`about.toml` configures cargo-about and `about.hbs` formats its output:

```sh
cargo install cargo-about --version 0.9.2 --locked --features cli
cargo about generate --locked --offline --fail about.hbs --output-file target/release/DEPENDENCY-NOTICES.txt
cargo about generate --locked --offline --fail --format json --output-file target/release/dependency-notices-audit.json
```

The configuration targets `aarch64-apple-darwin` and includes runtime, build
and development dependencies. It selects packaged texts for dependencies
with split or nested licenses. Review generation errors and the JSON audit
when changing dependencies, target or build features.

## Toolchain licenses

Record `rustc --version --verbose` and `rustc --print sysroot` for the release
build. Copy that toolchain's `COPYRIGHT-library.html`, the referenced license
texts and the complete matching `library/compiler-builtins/LICENSE.txt` into
the binary bundle. Check both the in-tree and dependency sections of the
report; some packaged toolchains omit dependency details. Missing texts must
be obtained from the matching toolchain sources before packaging.
