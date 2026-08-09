"""Artifact-declared files stay rooted on every supported path syntax."""

import pytest

from moespresso.core.paths import UnsafeArtifactPathError, resolve_artifact_file


@pytest.mark.parametrize(
    "value",
    (
        "../shard.safetensors",
        "nested/shard.safetensors",
        r"nested\shard.safetensors",
        "/tmp/shard.safetensors",
        r"C:\shard.safetensors",
        r"\\server\share\shard.safetensors",
        ".",
        "..",
        "",
        None,
    ),
)
def test_resolve_artifact_file_rejects_non_flat_names(tmp_path, value):
    with pytest.raises(UnsafeArtifactPathError):
        resolve_artifact_file(tmp_path, value)


def test_resolve_artifact_file_accepts_a_rooted_file(tmp_path):
    expected = tmp_path / "shard.safetensors"
    assert resolve_artifact_file(tmp_path, expected.name) == expected


def test_resolve_artifact_file_rejects_a_symlink_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    (root / "shard.safetensors").symlink_to(outside)

    with pytest.raises(UnsafeArtifactPathError, match="escapes root"):
        resolve_artifact_file(root, "shard.safetensors")
