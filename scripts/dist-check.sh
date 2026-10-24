#!/bin/sh
# Audit source-only distributions; binary notices are a separate release task.
set -eu

cd "$(dirname "$0")/.."
repository_root=$(pwd -P)
distribution_work=$(mktemp -d "${TMPDIR:-/tmp}/moespresso-dist-check.XXXXXXXX")
trap 'rm -rf "$distribution_work"' EXIT HUP INT TERM

rg --files --hidden --no-ignore \
    Cargo.toml Cargo.lock build.rs Makefile AGENTS.md README.md .gitignore \
    LICENSE-MIT LICENSE-APACHE-2.0 THIRD-PARTY-NOTICES about.toml about.hbs \
    src kernels tests docs scripts > "$distribution_work/candidates"
LC_ALL=C sort -u "$distribution_work/candidates" > "$distribution_work/files"
: > "$distribution_work/members"

while IFS= read -r path; do
    case "$path" in
        *[!a-zA-Z0-9_./-]*|../*|*/../*|*/./*|*/.*|*private*|*research*|*credentials*|*captured*)
            printf 'Rejected distribution path: %s\n' "$path" >&2
            exit 1
            ;;
    esac
    case "$path" in
        Cargo.toml|Cargo.lock|build.rs|Makefile|AGENTS.md|README.md|.gitignore|LICENSE-MIT|LICENSE-APACHE-2.0|THIRD-PARTY-NOTICES|about.toml|about.hbs|src/*.rs|src/ane/native.m|src/engine/ane_prefill/pack_q6.metal|src/engine/ane_prefill/padded_unpack.metal|kernels/metal/*.metal|tests/*.rs|docs/*.md|docs/assets/hero.webp|scripts/*.sh) ;;
        *) printf 'Path is outside the source distribution allowlist: %s\n' "$path" >&2; exit 1 ;;
    esac
    if [ ! -f "$path" ] || [ -L "$path" ]; then
        printf 'Distribution members must be regular files: %s\n' "$path" >&2
        exit 1
    fi
    printf '%s\n' "$path" >> "$distribution_work/members"
done < "$distribution_work/files"

for required in Cargo.toml Cargo.lock build.rs src/ane.rs src/ane/native.m src/ane/mil.rs src/engine/ane_prefill/pack_q6.metal src/engine/ane_prefill/padded_unpack.metal Makefile AGENTS.md README.md LICENSE-MIT LICENSE-APACHE-2.0 THIRD-PARTY-NOTICES about.toml about.hbs scripts/dist-check.sh; do
    rg -F -x -- "$required" "$distribution_work/members" > /dev/null || {
        printf 'Required distribution member is missing: %s\n' "$required" >&2
        exit 1
    }
done

mkdir "$distribution_work/source"
COPYFILE_DISABLE=1 tar -cf "$distribution_work/source.tar" -T "$distribution_work/members"
tar -tf "$distribution_work/source.tar" > "$distribution_work/archive-members"
LC_ALL=C sort "$distribution_work/archive-members" > "$distribution_work/archive-members-sorted"
cmp "$distribution_work/members" "$distribution_work/archive-members-sorted"
tar -xf "$distribution_work/source.tar" -C "$distribution_work/source"
for required in Cargo.lock LICENSE-MIT LICENSE-APACHE-2.0 THIRD-PARTY-NOTICES; do
    cmp "$repository_root/$required" "$distribution_work/source/$required"
done

set --
while IFS= read -r path; do
    set -- "$@" "$distribution_work/source/$path"
done < "$distribution_work/members"
# Split literal roots so the audit source does not match its own patterns.
machine_paths='/''Users/|/''home/[^ /]+/|/''private/var/folders/'
if rg -n -- "$machine_paths" "$@"; then
    printf '%s\n' 'Source distribution contains machine-local paths.' >&2
    exit 1
else
    audit_status=$?
    if [ "$audit_status" -ne 1 ]; then
        printf '%s\n' 'Could not inspect source distribution content.' >&2
        exit "$audit_status"
    fi
fi

# An independent target directory prevents a successful local build from
# concealing missing source or embedded Metal files in the archive.
CARGO_TARGET_DIR="$distribution_work/build" cargo build --locked --offline \
    --manifest-path "$distribution_work/source/Cargo.toml" --all-targets
cmp "$repository_root/Cargo.lock" "$distribution_work/source/Cargo.lock"
mkdir -p "$repository_root/target/dist"
cp "$distribution_work/source.tar" "$repository_root/target/dist/moespresso-source.tar"
printf '%s\n' 'Audited source archive: target/dist/moespresso-source.tar'
