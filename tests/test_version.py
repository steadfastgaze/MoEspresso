from __future__ import annotations

import ast
import tomllib
from importlib import metadata as importlib_metadata
from pathlib import Path

import moespresso


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src" / "moespresso"


def _literal_assignments(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            values.append(ast.literal_eval(node.value))
    return values


def _literal_assignment(path: Path, name: str):
    values = _literal_assignments(path, name)
    assert len(values) == 1, f"expected one literal {name} assignment in {path}"
    return values[0]


def _producer_modules():
    """Every module that stamps a top-level PRODUCER dict into artifact provenance.

    Discovered by scanning rather than listed, so a module added later cannot
    keep stamping a stale version behind a hand-maintained tuple.
    """
    modules = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        if "private" in path.relative_to(SOURCE_ROOT).parts:
            continue
        if _literal_assignments(path, "PRODUCER"):
            modules.append(path)
    return modules


def test_imported_version_matches_project_metadata():
    pyproject = REPOSITORY_ROOT / "pyproject.toml"
    project_metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")

    assert project_metadata["project"]["readme"] == "README.md"
    assert moespresso.__version__ == project_metadata["project"]["version"]
    assert importlib_metadata.version("moespresso") == project_metadata["project"]["version"]
    assert project_metadata["build-system"]["requires"] == ["hatchling==1.31.0"]
    assert f"MoEspresso {moespresso.__version__} requires" in readme


def test_lock_and_artifact_producers_match_release_version():
    lock = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text(encoding="utf-8"))
    project_packages = [row for row in lock["package"] if row.get("name") == "moespresso"]

    assert len(project_packages) == 1
    assert project_packages[0]["version"] == moespresso.__version__

    producer_modules = _producer_modules()

    assert producer_modules, "no PRODUCER-stamping module found under src/moespresso"
    for path in producer_modules:
        producer = _literal_assignment(path, "PRODUCER")
        assert producer["version"] == moespresso.__version__, str(
            path.relative_to(REPOSITORY_ROOT)
        )
