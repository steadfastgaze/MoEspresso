from __future__ import annotations

import ast
import tomllib
from importlib import metadata as importlib_metadata
from pathlib import Path

import moespresso
from moespresso.core.artifact import artifact_producer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src" / "moespresso"


def _producer_assignments(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [
            target.id
            for target in node.targets
            if isinstance(target, ast.Name) and target.id.endswith("PRODUCER")
        ]
        values.extend((name, node.value) for name in names)
    return values


def _producer_modules():
    """Every module that stamps a top-level producer into artifact provenance.

    Discovered by scanning rather than listed, so a module added later cannot
    bypass the single release-version source behind a hand-maintained tuple.
    """
    modules = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        if "private" in path.relative_to(SOURCE_ROOT).parts:
            continue
        if _producer_assignments(path):
            modules.append(path)
    return modules


def test_imported_version_matches_project_metadata():
    pyproject = REPOSITORY_ROOT / "pyproject.toml"
    project_metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")

    assert project_metadata["project"]["readme"] == "README.md"
    assert moespresso.__version__ == project_metadata["project"]["version"]
    assert importlib_metadata.version("moespresso") == project_metadata["project"]["version"]
    assert f"MoEspresso {moespresso.__version__} requires" in readme


def test_native_build_uses_standard_backend_and_matching_mlx_abi():
    project_metadata = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    backend = project_metadata["build-system"]
    assert backend["build-backend"] == "scikit_build_core.build"
    assert "scikit-build-core==1.0.3" in backend["requires"]
    assert "nanobind==2.12.0" in backend["requires"]
    mlx_build = [item for item in backend["requires"] if item.startswith("mlx==")]
    assert len(mlx_build) == 1
    assert mlx_build[0] in project_metadata["project"]["dependencies"]
    assert "native" not in project_metadata["dependency-groups"]
    assert project_metadata["tool"]["scikit-build"]["editable"]["rebuild"] is False


def test_lock_and_artifact_producers_match_release_version():
    lock = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text(encoding="utf-8"))
    project_packages = [row for row in lock["package"] if row.get("name") == "moespresso"]

    assert len(project_packages) == 1
    assert project_packages[0]["version"] == moespresso.__version__

    producer_modules = _producer_modules()

    assert producer_modules, "no PRODUCER-stamping module found under src/moespresso"
    assert artifact_producer("test.tool") == {
        "tool": "test.tool",
        "version": moespresso.__version__,
    }
    for path in producer_modules:
        for name, value in _producer_assignments(path):
            assert isinstance(value, ast.Call), (
                name,
                str(path.relative_to(REPOSITORY_ROOT)),
            )
            assert isinstance(value.func, ast.Name)
            assert value.func.id == "artifact_producer"
            assert len(value.args) == 1 and not value.keywords
            assert isinstance(value.args[0], ast.Constant)
            assert isinstance(value.args[0].value, str) and value.args[0].value
