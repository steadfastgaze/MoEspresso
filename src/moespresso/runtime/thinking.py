"""Per-family thinking toggle resolution.

There is no cross-model standard for disabling chain-of-thought: Qwen-family
templates accept an `enable_thinking` template kwarg (the template itself emits
the empty think scaffold the model was trained on); Gemma 4 wants a token
removed from the prompt; DeepSeek R1 has no off switch at all. So `--thinking
on|off` resolves to the family's own mechanism, in order:

1. Template sniff: if the package's chat template declares `enable_thinking`,
   pass the bool through: zero per-model code, the model authors own the
   semantics. This convention is the closest thing to a standard (vLLM/SGLang
   expose generic chat_template_kwargs for the same reason).
2. Family adapter table keyed on the manifest architecture family: one tiny
   entry per ported family that needs something else. Grown only when a port
   lands and is measured; entries return template kwargs today (a family
   needing prompt surgery instead (the Gemma 4 shape) extends the seam to a
   message-mutating adapter at that point, with its own tests).
3. Refuse loudly. A user who asked for --thinking off must never silently get
   thinking-on.

Pure module: no mlx, no heavy imports, testable anywhere.
"""

from __future__ import annotations


class ThinkingToggleUnsupported(RuntimeError):
    """The loaded model has no known thinking on/off mechanism."""


# family -> adapter(thinking: bool) -> template kwargs dict.
#
# DeepSeek-V4 is intentionally absent here. MoEspresso owns its renderer, and the
# serve layer maps `--thinking` onto the DS4 contract shapes directly
# (`http.deepseek_v4_contract_template_kwargs`), never through this generic seam.
_FAMILY_ADAPTERS: dict = {}


def template_supports_enable_thinking(tokenizer) -> bool:
    """True if the package's chat template declares the `enable_thinking` kwarg."""
    template = getattr(tokenizer, "chat_template", None) or ""
    return "enable_thinking" in template


def resolve_thinking_kwargs(
    tokenizer, *, thinking: bool, family: str | None = None,
) -> dict:
    """Map a requested thinking state to this model's template kwargs.

    Raises ThinkingToggleUnsupported when neither the template nor the adapter
    table knows a mechanism: callers surface this at startup, before serving.
    """
    if template_supports_enable_thinking(tokenizer):
        return {"enable_thinking": bool(thinking)}
    adapter = _FAMILY_ADAPTERS.get(family or "")
    if adapter is not None:
        return adapter(bool(thinking))
    raise ThinkingToggleUnsupported(
        f"--thinking was requested but the chat template does not take "
        f"enable_thinking and no adapter is registered for family="
        f"{family or 'unknown'}; refusing to serve with a silent default")
