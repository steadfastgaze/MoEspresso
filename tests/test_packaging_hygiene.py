from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _local_source_paths(value):
    """Every local filesystem source in a pyproject or lockfile tree.

    ``editable`` carries the flag in a pyproject source table and the path in
    a lockfile entry, so only string values count. The project's own root
    entry (``.``) is not a dependency on another checkout.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"directory", "editable", "path"} and isinstance(item, str):
                if item != ".":
                    yield item
            yield from _local_source_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from _local_source_paths(item)


def test_package_metadata_names_all_tracked_license_files():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["license-files"] == [
        "LICENSE-MIT",
        "LICENSE-APACHE-2.0",
        "THIRD-PARTY-NOTICES",
    ]
    assert all((ROOT / name).is_file() for name in project["license-files"])


def test_build_excludes_private_and_nonrelease_trees():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    build = config["tool"]["scikit-build"]
    excluded = set(build["sdist"]["exclude"])

    assert {
        ".research/**",
        "specs_archive/**",
        "native/*/build/**",
        "**/private/**",
        "**/__pycache__/**",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.so",
        "**/*.metallib",
    } <= excluded
    assert build["wheel"]["packages"] == ["src/moespresso"]
    assert {"**/private/**", "**/__pycache__/**", "**/*.pyc", "**/*.pyo"} <= set(
        build["wheel"]["exclude"]
    )


def test_uv_native_rebuild_inputs_exclude_generated_output():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    keys = config["tool"]["uv"]["cache-keys"]
    inputs = {item["file"] for item in keys if "file" in item}
    assert {
        "pyproject.toml",
        "CMakeLists.txt",
        "native/*/CMakeLists.txt",
        "native/*/*.cpp",
        "native/*/*.h",
    } <= inputs
    for pattern in inputs:
        assert "/" in pattern or not any(character in pattern for character in "*?[")
        assert not any("build" in path.relative_to(ROOT).parts for path in ROOT.glob(pattern))


def test_dependency_resolution_has_no_local_sources():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sources = config.get("tool", {}).get("uv", {}).get("sources", {})

    assert list(_local_source_paths(sources)) == []


def test_public_serve_alias_uses_the_startup_supervisor():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["project"]["scripts"]["moespresso-serve"] == ("moespresso.serve_supervisor:main")


def test_mlx_iqk_is_an_ordinary_published_pinned_requirement():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = [
        item
        for item in config["project"]["dependencies"]
        if item.lower().replace("_", "-").startswith("mlx-iqk")
    ]

    assert requirements == ["mlx-iqk==0.1.3"]


def test_mlx_lm_is_an_ordinary_published_requirement():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = [
        item
        for item in config["project"]["dependencies"]
        if item.lower().replace("_", "-").startswith("mlx-lm")
    ]

    assert len(requirements) == 1
    assert "@" not in requirements[0]
    assert "://" not in requirements[0]


def test_lockfile_has_no_local_sources():
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))

    assert list(_local_source_paths(lock)) == []
