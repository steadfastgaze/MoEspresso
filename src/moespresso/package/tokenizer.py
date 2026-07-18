"""Tokenizer files as a package contract.

Tokenizer/rendering identity is a runtime contract: KV
reuse, official-vector validation, and agent tool-calls all depend on exact
rendered bytes. Package builders copy the source tokenizer files into the package
and record their identity (names + sha256 + chat-template hash) in the manifest.
Serve loads the tokenizer from the package, never from the source checkpoint.

The copy is pure file IO; loading uses mlx_lm and so is lazy.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from moespresso.package.templates import chat_template_for

_CHAT_TEMPLATE_FILE = "chat_template.jinja"
_TOKENIZER_CONFIG = "tokenizer_config.json"

# Files convert generates itself; never copy these from the source (copying would
# clobber the generated outputs or carry the wrong config). Everything else
# non-safetensors is a tokenizer/processor/aux file the runtime may need.
_GENERATED = {
    "config.json", "jang_config.json", "package_manifest.json",
    "source_inventory.json", "probe_evidence.json", "optimizer_decision.json",
    "package_plan.json",
    "conversion_report.json", "model.safetensors.index.json",
}
# Source aux files that are irrelevant / large and not needed to serve.
_SKIP = {"README.md", "LICENSE", ".gitattributes", ".DS_Store"}

# A tokenizer is unusable without this: must end up in the package.
_REQUIRED = "tokenizer.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _install_chat_template(package_dir: Path, template: str) -> str | None:
    """Overwrite the package's chat template with `template`; return the family-template.

    Writes both sources so the package has one coherent answer: the standalone
    chat_template.jinja (which HF/transformers gives priority over the config;
    tokenization_utils_base resolves the independent file first), and the embedded
    tokenizer_config.json["chat_template"] (otherwise it lingers as a stale shadow copy of
    the old template, a silent two-sources-disagree footgun). Idempotent; returns the
    template so the caller can record its identity.
    """
    (package_dir / _CHAT_TEMPLATE_FILE).write_text(template, encoding="utf-8")
    cfg_path = package_dir / _TOKENIZER_CONFIG
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["chat_template"] = template
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return template


def copy_tokenizer_into_package(source_dir: Path, package_dir: Path,
                                *, family: str | None = None) -> dict:
    """Copy the source tokenizer/processor files into the package; return identity.

    Copies the complete source aux set (every non-safetensors file except those
    convert generates or explicitly skips), matching the proven convert-moe behavior:
    a hardcoded allowlist dropped files (e.g. preprocessor_config.json /
    configuration.json) that transformers needs to pick the right tokenizer, which
    triggered a Mistral-regex fallback + incorrect tokenization. Resolves symlinks
    (HF snapshots link into blobs/).

    If `family` has a MoEspresso-owned chat template (see package/templates), it overwrites
    the copied template (both standalone + embedded) before hashing, so the package serves
    that template and `rendering_id` reflects it. Returns the manifest `tokenizer` block:
    file identities + a rendering hash (spec's rendering_id, for byte-prefix drift) +
    `chat_template_source` ("family:<id>" when overridden, else "source").
    """
    source_dir, package_dir = Path(source_dir), Path(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for src in sorted(source_dir.iterdir()):
        if not src.is_file():
            continue
        name = src.name
        if name.endswith(".safetensors") or name in _GENERATED or name in _SKIP:
            continue
        shutil.copyfile(src, package_dir / name)  # follows symlinks -> real bytes
        copied.append(name)

    # Install the family's chat template (if any) before hashing, so rendering_id covers it.
    template = chat_template_for(family)
    chat_template_source = "source"
    if template is not None:
        _install_chat_template(package_dir, template)
        chat_template_source = f"family:{family}"
        if _CHAT_TEMPLATE_FILE not in copied:  # source had only an embedded template
            copied.append(_CHAT_TEMPLATE_FILE)

    files = [{"path": name, "size_bytes": (package_dir / name).stat().st_size,
              "sha256": _sha256(package_dir / name)} for name in sorted(copied)]
    rendering = hashlib.sha256(
        "".join(sorted(f"{f['path']}:{f['sha256']}" for f in files)).encode()
    ).hexdigest()
    return {"files": files, "rendering_id": rendering,
            "chat_template_source": chat_template_source,
            "has_tokenizer": any(f["path"] == _REQUIRED for f in files)}

# Serve does not load the tokenizer separately: it uses the one jang's
# load_jangtq_model returns (mlx_lm load_tokenizer + eos/chat handling), matching the
# proven decode path (see runtime/build.build_model). This module's job is the
# package-side copy above; there is no serve-side tokenizer loader here.
