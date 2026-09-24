"""Load-time capacity policy for the bundled DeepSeek-V4 drafter.

A package that declares a bundled drafter component enables it by default when
the complete drafter-on serving state fits the reported wired-memory budget:

    usable wired budget >= resident weights + drafter files
                           + KV-and-pool state at the full served context
                           + the reserved per-request working-set margin

Every term is read from a declared fact: weights and drafter bytes from the
package manifest's identity lists, the cache term from the architecture's
layer contract at the served context limit, and the reserved per-request
working set. The budget is ``iogpu.wired_limit_mb`` when available, otherwise
Metal's reported recommended working set. An unreadable term resolves the
decision to off without raising.

The decision payload is attached to the served model and exported through the
engagement census surfaces with the values used for the comparison.
"""

from __future__ import annotations

from typing import Any

from moespresso.runtime.streaming_capacity import usable_wired_budget_bytes

# The served context limit the capacity check provisions for. Mirrors
# ``moespresso.runtime.prefix_cache.DEFAULT_CONTEXT_LIMIT`` (pinned equal by
# test; not imported to keep this module free of the serve-edge import
# chain).
DEFAULT_CONTEXT_TOKENS = 128 * 1024

# Reserved per-request working set for the default sorted-prefill split. The
# capacity calculation holds it in addition to weights and cache state.
WORKING_SET_MARGIN_BYTES = int(14.1702 * (1 << 30))

# Reserved per-request working set for the parts-32 fallback. The policy may
# select this split when the default split does not fit. Other explicit split
# values use the unsplit reservation.
WORKING_SET_MARGIN_BYTES_PARTS32 = int(11.6448 * (1 << 30))
WORKING_SET_MARGIN_BYTES_UNSPLIT = int(22.20 * (1 << 30))

# The default sorted-route split and the policy fallback.
SORT_NSPLIT_DEFAULT = 16
SORT_NSPLIT_FLOOR = 32


def working_set_margin_bytes(parts: int) -> int:
    """The reserved per-request working set for a sorted-route split."""
    if parts >= SORT_NSPLIT_FLOOR:
        return WORKING_SET_MARGIN_BYTES_PARTS32
    if parts >= SORT_NSPLIT_DEFAULT:
        return WORKING_SET_MARGIN_BYTES
    return WORKING_SET_MARGIN_BYTES_UNSPLIT

# Fixed prefill-window state of the composite cache, independent of context
# length: 41 compressed layers hold a rotating window bounded by
# sliding_window + prefill_chunk - 1 = 2175 rows at head_dim 512 in K and V
# (8 bytes per dim-pair), and the 21 ratio-4 layers add 40,960 bytes of
# compressor and indexer overlap buffers each.
PREFILL_WINDOW_STATE_BYTES = 366_120_960

# Growth of a ratio-0 (sliding-window) layer's plain K/V cache in bytes per
# token per head dimension. Each layer uses 6 x head_dim bytes per token.
SWA_BYTES_PER_TOKEN_PER_DIM = 6

# Pooled cache rows are float32.
POOLED_BYTES_PER_ELEMENT = 4

# The DSpark draft stages keep one fixed 128-row ring window each at the
# target head dimension in float32; no draft-side state grows with context.
DSPARK_WINDOW_ROWS = 128


def deepseek_v4_cache_state_bytes(architecture: dict, context_tokens: int) -> int:
    """KV-and-pool bytes of the composite cache at ``context_tokens``.

    Derived from the manifest's own layer contract. Per layer and token:
    a ratio-r pooled layer grows one float32 row of 2 x head_dim (compressor
    pooled + fp8 twin) per r tokens, plus 2 x index_head_dim on ratio-4
    layers (the indexer pool and its qat twin); a ratio-0 layer grows a
    plain KV cache at 6 x head_dim bytes per token. The fixed
    prefill-window term is context-independent. Raises KeyError/TypeError on
    a manifest missing the architecture facts; the caller fails closed.
    """
    attention = architecture["attention"]
    head_dim = int(attention["head_dim"])
    index_head_dim = int(attention["index_head_dim"])
    num_layers = int(architecture["num_hidden_layers"])
    ratios = [int(r) for r in architecture["compress_ratios"][:num_layers]]
    if len(ratios) != num_layers:
        raise ValueError(
            f"architecture declares {len(ratios)} compress ratios for "
            f"{num_layers} layers")
    per_token = 0.0
    for ratio in ratios:
        if ratio == 0:
            per_token += SWA_BYTES_PER_TOKEN_PER_DIM * head_dim
            continue
        dims = 2 * head_dim + (2 * index_head_dim if ratio == 4 else 0)
        per_token += dims * POOLED_BYTES_PER_ELEMENT / ratio
    return int(per_token * context_tokens) + PREFILL_WINDOW_STATE_BYTES


def dspark_state_bytes(architecture: dict, n_stages: int) -> int:
    """Fixed draft-side state: one float32 ring window per DSpark stage."""
    head_dim = int(architecture["attention"]["head_dim"])
    return int(n_stages) * DSPARK_WINDOW_ROWS * head_dim * POOLED_BYTES_PER_ELEMENT


def _identity_bytes(entries: Any) -> int | None:
    if not isinstance(entries, list) or not entries:
        return None
    total = 0
    for entry in entries:
        size = entry.get("size_bytes") if isinstance(entry, dict) else None
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            return None
        total += size
    return total


def evaluate_bundled_drafter_budget(
    manifest: dict,
    *,
    n_stages: int,
    context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    budget_fn: Any = None,
    sort_nsplit_env: str | None = None,
) -> dict:
    """Decide the bundled drafter default for the current runtime.

    Returns a JSON-serializable decision payload: ``decision`` (``"on"`` or
    ``"off"``), ``reason``, the selected sorted-route split, and every
    number that drove the comparison. The working-set term is
    capacity-aware: the default parts-16 margin is preferred, and when only
    the parts-32 reservation fits the policy selects parts 32
    (``sort_nsplit_source: "policy"``) before declaring off. An explicit
    ``MOESPRESSO_DSV4_IQK_SORT_NSPLIT`` pins the split: the policy then
    reserves that setting's working set and does not override the explicit
    setting. Any unreadable term resolves to off. ``budget_fn`` is
    the test seam for synthetic budgets.
    """
    if budget_fn is None:
        budget_fn = usable_wired_budget_bytes
    payload: dict[str, Any] = {
        "decision": "off",
        "context_tokens": int(context_tokens),
    }
    weights_bytes = _identity_bytes(manifest.get("files"))
    if weights_bytes is None:
        payload["reason"] = "weights-bytes-unreadable"
        return payload
    payload["weights_bytes"] = weights_bytes
    drafter = manifest.get("drafter") or {}
    drafter_bytes = _identity_bytes(drafter.get("files"))
    if drafter_bytes is None:
        payload["reason"] = "drafter-bytes-unreadable"
        return payload
    payload["drafter_bytes"] = drafter_bytes
    try:
        architecture = manifest.get("architecture") or {}
        cache_bytes = deepseek_v4_cache_state_bytes(architecture, context_tokens)
        draft_state = dspark_state_bytes(architecture, n_stages)
    except (KeyError, TypeError, ValueError) as exc:
        payload["reason"] = f"cache-contract-unreadable: {exc}"
        return payload
    payload["kv_state_bytes"] = cache_bytes
    payload["dspark_state_bytes"] = draft_state

    if sort_nsplit_env not in (None, ""):
        try:
            pinned = int(str(sort_nsplit_env).strip())
        except ValueError:
            payload["reason"] = "sort-nsplit-env-unreadable"
            return payload
        if pinned < 1 or pinned & (pinned - 1):
            payload["reason"] = "sort-nsplit-env-unreadable"
            return payload
        candidates = [(pinned, "env")]
    else:
        candidates = [(SORT_NSPLIT_DEFAULT, "default"),
                      (SORT_NSPLIT_FLOOR, "policy")]

    budget, source = budget_fn()
    payload["budget_source"] = source
    if budget is None or budget <= 0:
        payload["reason"] = "budget-unreadable"
        return payload
    payload["budget_bytes"] = int(budget)

    base = weights_bytes + drafter_bytes + cache_bytes + draft_state
    payload["required_bytes_by_nsplit"] = {
        str(parts): base + working_set_margin_bytes(parts)
        for parts, _source in candidates
    }
    for parts, parts_source in candidates:
        margin = working_set_margin_bytes(parts)
        required = base + margin
        if required <= budget:
            payload.update({
                "decision": "on",
                "reason": ("budget-fit" if parts_source != "policy"
                           else "budget-fit-nsplit32"),
                "sort_nsplit": parts,
                "sort_nsplit_source": parts_source,
                "working_set_margin_bytes": margin,
                "required_bytes": required,
            })
            return payload
    parts, parts_source = candidates[-1]
    margin = working_set_margin_bytes(parts)
    payload.update({
        "reason": "budget-exceeded",
        "sort_nsplit": parts,
        "sort_nsplit_source": parts_source,
        "working_set_margin_bytes": margin,
        "required_bytes": base + margin,
    })
    return payload
