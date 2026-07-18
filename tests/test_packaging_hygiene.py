from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _local_source_paths(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"directory", "path"}:
                yield str(item)
            yield from _local_source_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from _local_source_paths(item)


def test_package_metadata_names_all_tracked_license_files():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    assert project["license-files"] == [
        "LICENSE-MIT",
        "LICENSE-APACHE-2.0",
        "THIRD-PARTY-NOTICES",
    ]
    assert all((ROOT / name).is_file() for name in project["license-files"])


def test_hatch_build_excludes_private_and_nonrelease_trees():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    excluded = set(config["tool"]["hatch"]["build"]["exclude"])

    assert {
        "/specs_archive",
        "/specs_archive/**",
        "/native/gate/build",
        "/native/gate/build/**",
        "/native/ds4_moe/build",
        "/native/ds4_moe/build/**",
        "**/private",
        "**/private/**",
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
    } <= excluded


def test_dependency_resolution_has_no_local_source_overrides():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sources = config.get("tool", {}).get("uv", {}).get("sources", {})

    assert list(_local_source_paths(sources)) == []


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


def test_lockfile_has_no_local_directory_sources():
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))

    assert list(_local_source_paths(lock)) == []
