"""Build and audit MoEspresso release artifacts.

The test functions exercise the policy without building. Running this file with
``--build`` creates a wheel and source distribution in a temporary directory
and audits their real members and text payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SEGMENTS = {
    "specs_archive",
    "private",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}
FORBIDDEN_SUFFIXES = (".pyc", ".pyo")
FORBIDDEN_NATIVE_SUFFIXES = (".a", ".dylib", ".metallib", ".o", ".so")
HOST_PATH_MARKERS = (
    ("/" + "Users/", "macOS user path"),
    ("/" + "Volumes/", "macOS volume path"),
    ("C:" + "\\Users\\", "Windows user path"),
)
PRIVATE_PACKAGE_PAYLOAD_MARKERS = ("specs_archive/",)
LICENSE_NAMES = {"LICENSE-MIT", "LICENSE-APACHE-2.0", "THIRD-PARTY-NOTICES"}
LICENSE_BYTES = {name: (REPO_ROOT / name).read_bytes() for name in LICENSE_NAMES}
EXPECTED_VERSION = "1.0.0"
EXPECTED_LICENSE_EXPRESSION = "MIT OR Apache-2.0"
EXPECTED_RUNTIME_REQUIREMENTS = {
    "jang",
    "mlx",
    "mlx-kquant",
    "mlx-lm",
    "numpy",
    "psutil",
}
README_BYTES = (REPO_ROOT / "README.md").read_bytes()
PROJECT_METADATA = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
EXPECTED_SURFACE_SUFFIXES = (
    "moespresso/package/deepseek_v4/kquant_package.py",
    "moespresso/correctness/deepseek_v4/quality.py",
    "moespresso/correctness/fixtures/deepseek_v4/test_vectors/official.vec",
    "moespresso/correctness/fixtures/deepseek_v4/test_vectors/official/"
    "short_reasoning_plain.official.json",
    "moespresso/correctness/fixtures/deepseek_v4/q2_official_continuations/prompts.jsonl",
    "moespresso/correctness/ornith/gate.py",
    "moespresso/correctness/ornith/tasks.py",
    "moespresso/correctness/ornith/scoring.py",
    "moespresso/package/qwen/kquant_package.py",
    "moespresso/runtime/qwen/full_attention.py",
    "moespresso/package/templates/qwen3_5_moe.chat_template.jinja",
)
PUBLIC_ENTRY_POINTS = tuple(sorted(PROJECT_METADATA["project"]["scripts"]))
PRIVATE_TEST_CALL = "tasks.load_private_" + "questions()"


def _normalize_member(name: str) -> str:
    parts = list(PurePosixPath(name).parts)
    if (
        len(parts) > 1
        and parts[0].startswith("moespresso-")
        and not parts[0].endswith(".dist-info")
    ):
        parts = parts[1:]
    return "/".join(parts)


def _read_archive(path: Path) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    members[_normalize_member(info.filename)] = archive.read(info)
        return members
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            for info in archive.getmembers():
                if not info.isfile():
                    continue
                extracted = archive.extractfile(info)
                if extracted is not None:
                    members[_normalize_member(info.name)] = extracted.read()
        return members
    raise ValueError(f"unsupported distribution archive: {path}")


def _line_with_marker(text: str, marker: str) -> int:
    return next(
        (line_no for line_no, line in enumerate(text.splitlines(), 1) if marker in line),
        1,
    )


def audit_members(kind: str, members: dict[str, bytes]) -> list[str]:
    """Return release-boundary failures for normalized archive members."""
    failures: list[str] = []
    names = tuple(sorted(members))

    for name in names:
        parts = PurePosixPath(name).parts
        bad_segments = FORBIDDEN_SEGMENTS.intersection(parts)
        if bad_segments:
            failures.append(f"{kind}: forbidden path {name}")
        if name.endswith(FORBIDDEN_SUFFIXES):
            failures.append(f"{kind}: bytecode path {name}")
        if name.endswith(FORBIDDEN_NATIVE_SUFFIXES):
            failures.append(f"{kind}: generated native artifact {name}")
        if any(part == ".env" or part.startswith(".env.") for part in parts):
            failures.append(f"{kind}: environment file {name}")

        data = members[name]
        if b"\x00" in data[:1024]:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for marker, label in HOST_PATH_MARKERS:
            if marker in text:
                line_no = _line_with_marker(text, marker)
                failures.append(f"{kind}: {name}:{line_no} contains {label}")
        if "moespresso" in parts:
            for marker in PRIVATE_PACKAGE_PAYLOAD_MARKERS:
                if marker in text:
                    line_no = _line_with_marker(text, marker)
                    failures.append(
                        f"{kind}: {name}:{line_no} contains private package reference"
                    )

    for license_name in sorted(LICENSE_NAMES):
        matches = [name for name in names if PurePosixPath(name).name == license_name]
        if not matches:
            failures.append(f"{kind}: missing {license_name}")
            continue
        for name in matches:
            if members[name] != LICENSE_BYTES[license_name]:
                failures.append(f"{kind}: {name} does not match the repository copy")

    for suffix in EXPECTED_SURFACE_SUFFIXES:
        if not any(name.endswith(suffix) for name in names):
            failures.append(f"{kind}: missing product surface {suffix}")

    if kind == "wheel":
        metadata_bytes = next(
            (data for name, data in members.items() if name.endswith(".dist-info/METADATA")),
            b"",
        )
        failures.extend(_audit_core_metadata("wheel", "METADATA", metadata_bytes))

        entry_points = next(
            (
                data
                for name, data in members.items()
                if name.endswith(".dist-info/entry_points.txt")
            ),
            b"",
        ).decode("utf-8", errors="replace")
        for command in PUBLIC_ENTRY_POINTS:
            if f"{command} =" not in entry_points:
                failures.append(f"wheel: missing console entry point {command}")

    if kind == "sdist":
        pkg_info = members.get("PKG-INFO", b"")
        failures.extend(_audit_core_metadata("sdist", "PKG-INFO", pkg_info))
        if members.get("README.md") != README_BYTES:
            failures.append("sdist: README.md does not exactly match the public README")
        test_source = next(
            (
                data.decode("utf-8", errors="replace")
                for name, data in members.items()
                if name.endswith("tests/test_ornith_gate.py")
            ),
            "",
        )
        if not test_source:
            failures.append("sdist: missing tests/test_ornith_gate.py")
        elif PRIVATE_TEST_CALL in test_source:
            failures.append("sdist: Ornith tests read ignored private fixtures")

    return failures


def _audit_core_metadata(kind: str, label: str, data: bytes) -> list[str]:
    """Validate the release identity, dual license, and full long description."""
    if not data:
        return [f"{kind}: missing distribution {label}"]
    raw_headers, separator, body = data.partition(b"\n\n")
    if not separator:
        return [f"{kind}: malformed distribution {label}"]
    headers = raw_headers.decode("utf-8", errors="replace").splitlines()

    def values(name: str) -> list[str]:
        prefix = f"{name}: "
        return [line[len(prefix) :] for line in headers if line.startswith(prefix)]

    failures: list[str] = []
    if values("Version") != [EXPECTED_VERSION]:
        failures.append(f"{kind}: {label} version is not {EXPECTED_VERSION}")
    if values("Description-Content-Type") != ["text/markdown"]:
        failures.append(f"{kind}: README is not published as Markdown metadata")
    if values("License-Expression") != [EXPECTED_LICENSE_EXPRESSION]:
        failures.append(f"{kind}: {label} license expression is not {EXPECTED_LICENSE_EXPRESSION}")
    if set(values("License-File")) != LICENSE_NAMES:
        failures.append(f"{kind}: {label} does not declare the exact license files")
    requirements = values("Requires-Dist")
    requirement_names = {
        re.split(r"[\s<>=!~@;\[]", requirement, maxsplit=1)[0].lower()
        for requirement in requirements
    }
    missing_requirements = sorted(EXPECTED_RUNTIME_REQUIREMENTS - requirement_names)
    if missing_requirements:
        failures.append(
            f"{kind}: {label} is missing runtime requirements "
            + ", ".join(missing_requirements)
        )
    if "compute" in values("Provides-Extra") or any(
        "extra == 'compute'" in requirement or 'extra == "compute"' in requirement
        for requirement in requirements
    ):
        failures.append(f"{kind}: {label} still exposes the removed compute extra")
    if body != README_BYTES:
        failures.append(f"{kind}: {label} does not embed the exact public README")
    return failures


def audit_archive(path: Path) -> list[str]:
    kind = "wheel" if path.suffix == ".whl" else "sdist"
    return audit_members(kind, _read_archive(path))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_and_audit() -> tuple[list[Path], list[str]]:
    with tempfile.TemporaryDirectory(prefix="moespresso-dist-check-") as temp_dir:
        out_dir = Path(temp_dir)
        direct_dir = out_dir / "direct"
        rebuilt_dir = out_dir / "from-sdist"
        direct_dir.mkdir()
        rebuilt_dir.mkdir()
        subprocess.run(
            ["uv", "build", "--wheel", "--sdist", "--out-dir", str(direct_dir)],
            cwd=REPO_ROOT,
            check=True,
        )
        direct_wheels = sorted(direct_dir.glob("*.whl"))
        sdists = sorted(direct_dir.glob("*.tar.gz"))
        if len(direct_wheels) != 1 or len(sdists) != 1:
            return [], ["build: expected exactly one direct wheel and one source distribution"]
        subprocess.run(
            [
                "uv",
                "build",
                str(sdists[0]),
                "--wheel",
                "--out-dir",
                str(rebuilt_dir),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        rebuilt_wheels = sorted(rebuilt_dir.glob("*.whl"))
        if len(rebuilt_wheels) != 1:
            return [], ["build: expected exactly one wheel rebuilt from the sdist"]
        archives = [direct_wheels[0], sdists[0], rebuilt_wheels[0]]
        failures = [failure for path in archives for failure in audit_archive(path)]
        if _sha256_file(direct_wheels[0]) != _sha256_file(rebuilt_wheels[0]):
            failures.append("build: wheel rebuilt from sdist is not byte-identical")
        names = [
            Path(direct_wheels[0].name),
            Path(sdists[0].name),
            Path("from-sdist") / rebuilt_wheels[0].name,
        ]
        return names, failures


def _minimal_metadata() -> bytes:
    headers = (
        "Metadata-Version: 2.4\n"
        "Name: moespresso\n"
        f"Version: {EXPECTED_VERSION}\n"
        f"License-Expression: {EXPECTED_LICENSE_EXPRESSION}\n"
        "License-File: LICENSE-APACHE-2.0\n"
        "License-File: LICENSE-MIT\n"
        "License-File: THIRD-PARTY-NOTICES\n"
        "Requires-Dist: jang>=2.5.29\n"
        "Requires-Dist: mlx>=0.31.2\n"
        "Requires-Dist: mlx-kquant@ git+https://example.invalid/mlx-kquant.git\n"
        "Requires-Dist: mlx-lm>=0.31.3\n"
        "Requires-Dist: numpy>=1.26\n"
        "Requires-Dist: psutil>=7.0.0\n"
        "Description-Content-Type: text/markdown\n"
        "\n"
    ).encode()
    return headers + README_BYTES


def _minimal_members(kind: str) -> dict[str, bytes]:
    members = {suffix: b"public\n" for suffix in EXPECTED_SURFACE_SUFFIXES}
    for license_name in LICENSE_NAMES:
        members[license_name] = LICENSE_BYTES[license_name]
    if kind == "wheel":
        entry_points = "\n".join(f"{name} = package:main" for name in PUBLIC_ENTRY_POINTS)
        members["moespresso-1.0.0.dist-info/entry_points.txt"] = entry_points.encode()
        members["moespresso-1.0.0.dist-info/METADATA"] = _minimal_metadata()
    else:
        members["tests/test_ornith_gate.py"] = b"synthetic_fixture = True\n"
        members["PKG-INFO"] = _minimal_metadata()
        members["README.md"] = README_BYTES
    return members


def test_audit_accepts_minimal_public_artifacts():
    assert audit_members("wheel", _minimal_members("wheel")) == []
    assert audit_members("sdist", _minimal_members("sdist")) == []


def test_member_normalization_preserves_wheel_metadata():
    assert _normalize_member("moespresso-1.0/src/moespresso/example.py") == (
        "src/moespresso/example.py"
    )
    assert _normalize_member("moespresso-1.0.dist-info/entry_points.txt") == (
        "moespresso-1.0.dist-info/entry_points.txt"
    )


def test_audit_rejects_modified_third_party_notice():
    members = _minimal_members("wheel")
    members["THIRD-PARTY-NOTICES"] = b"truncated\n"

    failures = audit_members("wheel", members)

    assert any(
        "THIRD-PARTY-NOTICES does not match the repository copy" in failure
        for failure in failures
    )


def test_audit_rejects_private_material_and_host_paths():
    members = _minimal_members("sdist")
    members["specs_archive/notes.md"] = b"private history\n"
    members["moespresso/correctness/fixtures/ornith/private/key.json"] = b"{}\n"
    host_path = "/" + "Users/example/checkpoint.bin"
    members["moespresso/leak.txt"] = host_path.encode()

    failures = audit_members("sdist", members)

    assert any("specs_archive/notes.md" in failure for failure in failures)
    assert any("ornith/private/key.json" in failure for failure in failures)
    assert any("macOS user path" in failure for failure in failures)


def test_audit_rejects_private_references_in_package_payloads():
    members = _minimal_members("wheel")
    members["moespresso/package/agentic_profile.py"] = (
        b'PROVENANCE = "specs_archive/agent/study.md"\n'
    )

    failures = audit_members("wheel", members)

    assert any("private package reference" in failure for failure in failures)


def test_audit_rejects_incomplete_release_metadata():
    members = _minimal_members("wheel")
    metadata_name = "moespresso-1.0.0.dist-info/METADATA"
    members[metadata_name] = members[metadata_name].replace(
        b"License-Expression: MIT OR Apache-2.0\n", b"License-Expression: MIT\n"
    )
    members[metadata_name] = members[metadata_name].replace(README_BYTES, b"truncated\n")
    members["moespresso-1.0.0.dist-info/entry_points.txt"] = b"moespresso-serve = pkg:main\n"

    failures = audit_members("wheel", members)

    assert any("license expression" in failure for failure in failures)
    assert any("exact public README" in failure for failure in failures)
    assert any("missing console entry point moespresso-verify" in failure for failure in failures)


def test_audit_rejects_removed_compute_extra_metadata():
    members = _minimal_members("wheel")
    metadata_name = "moespresso-1.0.0.dist-info/METADATA"
    members[metadata_name] = members[metadata_name].replace(
        b"Requires-Dist: mlx>=0.31.2\n",
        b"Requires-Dist: mlx>=0.31.2; extra == 'compute'\nProvides-Extra: compute\n",
    )

    failures = audit_members("wheel", members)

    assert any("removed compute extra" in failure for failure in failures)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="*", type=Path)
    parser.add_argument("--build", action="store_true", help="build both artifacts first")
    args = parser.parse_args(argv)
    if args.build and args.archives:
        parser.error("--build does not accept archive paths")
    if not args.build and not args.archives:
        parser.error("pass --build or one or more archive paths")

    if args.build:
        archives, failures = _build_and_audit()
    else:
        archives = args.archives
        failures = [failure for path in archives for failure in audit_archive(path)]

    if failures:
        print("Distribution audit failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("Distribution audit passed: " + ", ".join(str(path) for path in archives))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
