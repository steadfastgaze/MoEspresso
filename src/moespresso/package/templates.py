"""MoEspresso-owned chat templates, vendored per model family.

Some families need a chat template MoEspresso controls rather than whatever shipped in the
source tokenizer. For qwen3_5_moe we vendor the community-validated froggeric Qwen 3.5/3.6
template (Apache-2.0): it renders history append-only (preserves past <think> blocks by
default) so a KV prefix cache stays valid across turns, fixes the official template's
double-render/agentic-stall bugs, and is minijinja/C++-safe. `chat_template_for(family)`
returns the vendored template text, or None when MoEspresso does not override that family
(its source template is then kept as-is). Resolved via importlib.resources so it works from
a wheel or an editable install.
"""

from __future__ import annotations

from importlib.resources import files

# family id (see inventory/architecture_profile.PROFILES) -> vendored template filename.
_TEMPLATE_FILES = {
    "qwen3_5_dense": "qwen3_5_moe.chat_template.jinja",
    "qwen3_5_moe": "qwen3_5_moe.chat_template.jinja",
}


def chat_template_for(family: str | None) -> str | None:
    """The vendored chat template text for a family, or None if not overridden."""
    name = _TEMPLATE_FILES.get(family or "")
    if name is None:
        return None
    return files(__package__).joinpath("templates", name).read_text(encoding="utf-8")
