"""Served-path glue for DeepSeek-V4 speculative decoding.

Serving selects a drafter with the ``MOESPRESSO_DS4_DRAFTER`` environment
variable: ``off``, ``dspark:<sidecar-dir>``, or ``dflash:<sidecar-dir>``. The
sidecar loads once at model-load time for
DeepSeek-V4 packages; other families ignore the variable with a notice. An
explicitly configured sidecar that fails to load refuses startup, the same
convention the disk KV store applies to an explicitly requested root.

An absent or empty variable selects automatically. Automatic selection
serves a package-declared drafter first: a manifest that declares a bundled
``drafter`` component (family ``dspark``) enables it only when every routed
expert of every layer is resident (a package serving under bounded
residency always stays plain), every declared component file is present
(a distribution shipped without the optional component serves plain,
counted, never an error), and the machine passes the wired-budget capacity
check (``drafter_policy``): resident weights plus drafter files plus
KV-and-pool state at the served 128k context plus the certified working-set
margin must fit the usable wired budget. The decision and its numbers are
attached to the model and exported through the engagement census surfaces;
an explicit ``MOESPRESSO_DS4_DRAFTER`` value in either direction is
recorded as an override. A package that declares no drafter component
serves plain: the declared component is the whole of automatic selection,
so a sidecar cannot pair with a package that does not name it. Every other
family is explicit-selection only. Any automatic miss prints one line
naming the reason and serving stays plain; the automatic path never
refuses startup.

The served speculative schedule is family-keyed: the DSpark chain serves
the measured fixed:3 submit schedule (the schedule sweep and the drafter-on
certification's configuration); other families serve the adaptive
schedule. ``MOESPRESSO_DS4_SPEC_SCHEDULE`` (``fixed:<K>`` or ``adaptive``)
overrides, failing closed on any other shape.

A request takes the speculative path only when the effective sampler is one
the acceptance rules reproduce exactly: greedy (temperature 0) or
pure-temperature sampling with no top-p, top-k, or min-p shaping, no logits
processors, and no per-token logprob capture. A drafter that advertises
``greedy_only`` (DFlash: argmax proposals with no draft distribution over
the target vocabulary) additionally restricts engagement to temperature 0;
its temperature > 0 requests take the plain path. Every other request shape
takes the plain path unchanged. Resumable DSpark requests may continue from a
target cache paired with a provenance-bound drafter capsule. The memory store
keeps that pair on a separate producer rail, and the disk tier binds the capsule
to an aligned target checkpoint. Drafters without the portable state contract
still use a fresh per-request cache because verify and rollback move cache
frontiers during each round.

Top-level imports stay light; mlx and the drafter loaders are imported at
call time so the pure serve core remains importable without them.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from moespresso.runtime.generation import GenerationResult

if TYPE_CHECKING:
    from moespresso.runtime.deepseek_v4.spec_decode import SpecPrefillProgress

DRAFTER_ENV = "MOESPRESSO_DS4_DRAFTER"
SPEC_SCHEDULE_ENV = "MOESPRESSO_DS4_SPEC_SCHEDULE"
DRAFTER_FAMILIES = ("dspark", "dflash")
# Served submit schedule per drafter family. DSpark serves the measured
# fixed:3 pick (schedule sweep at the long-prompt anchor: fixed:3 28.666
# tok/s, fixed:2 28.263, adaptive 27.621, fixed:4 26.343; the drafter-on
# certification ran fixed:3). Families without an entry serve adaptive.
DEFAULT_SPEC_SCHEDULES = {"dspark": "fixed:3"}
_ADAPTIVE_SCHEDULE = "adaptive"
_DEEPSEEK_V4_FAMILY = "deepseek_v4_flash"
_DEEPSEEK_V4_RENDERER = "deepseek_v4_dsv4"
SPEC_CACHE_SCHEMA = "deepseek_v4_spec_cache_v1"
SPEC_PRODUCER_LATTICE = "deepseek_v4_speculative"
# Restates SIDECAR_MANIFEST_NAME from the moespresso.package.deepseek_v4
# sidecar builders. Those builder modules import mlx at module scope; this
# serve seam stays importable without it.
_SIDECAR_MANIFEST_NAMES = {
    "dspark": "dspark_sidecar.json",
    "dflash": "dflash_sidecar.json",
}

_UNSET = object()


class DrafterConfigError(RuntimeError):
    """A configured drafter that cannot be parsed or loaded."""


class SpecContinuationError(RuntimeError):
    """A speculative cache continuation that fails preflight validation."""


@dataclass(frozen=True)
class DrafterConfig:
    """Parsed ``MOESPRESSO_DS4_DRAFTER`` selection."""

    family: str
    sidecar_dir: Path


@dataclass
class ServedDrafter:
    """A loaded drafter carried on the served model object.

    The serve seam reads it back through the ``_moespresso_ds4_drafter``
    model attribute, the same carrier convention as the other runtime facts
    attached at load time (prefill step size, streaming capacity). ``auto``
    records whether automatic selection installed the drafter (an absent
    ``MOESPRESSO_DS4_DRAFTER``) or an explicit selection did.
    """

    family: str
    sidecar_dir: Path
    drafter: Any
    tap: Any
    auto: bool = False
    artifact_id: str | None = None


@dataclass(frozen=True)
class SpecCacheCompanion:
    """DSpark state and provenance paired with one target-cache frontier."""

    family: str
    artifact_id: str
    schedule: str
    frontier: int
    capsule: Any
    schema: str = SPEC_CACHE_SCHEMA
    producer_lattice: str = SPEC_PRODUCER_LATTICE

    @property
    def nbytes(self) -> int:
        """Resident tensor bytes charged by the generic prefix store."""
        return int(self.capsule.nbytes)

    @property
    def producer_rail(self) -> tuple[str, ...]:
        """Hashable identity for caches produced on this numeric lattice."""
        rail = spec_cache_producer_rail(self)
        if rail is None:  # pragma: no cover - companion inputs always form a rail
            raise ValueError("speculative cache companion has no producer rail")
        return rail


@dataclass(frozen=True)
class SpecContinuation:
    """Internal input for continuing a paired speculative cache timeline."""

    target_cache: Any
    companion: SpecCacheCompanion
    prefix_offset: int


def parse_drafter_env(value: str | None) -> DrafterConfig | None:
    """Parse the drafter selection; None means no drafter is named.

    Accepts ``off`` (also an absent or empty value), ``dspark:<sidecar-dir>``,
    or ``dflash:<sidecar-dir>``. Any other shape is a
    configuration error. The serve path distinguishes an explicit ``off``
    from an absent or empty value (automatic selection) before calling; both
    map to None here.
    """
    if value is None:
        return None
    text = value.strip()
    if text in ("", "off"):
        return None
    family, sep, sidecar = text.partition(":")
    sidecar = sidecar.strip()
    if not sep or family not in DRAFTER_FAMILIES or not sidecar:
        raise DrafterConfigError(
            f"invalid {DRAFTER_ENV} value {value!r}; expected off, "
            f"dspark:<sidecar-dir>, or dflash:<sidecar-dir>")
    return DrafterConfig(family=family, sidecar_dir=Path(sidecar).expanduser())


def parse_spec_schedule(text: str) -> tuple[str, int | None]:
    """Parse a served speculative schedule: ``fixed:<K>`` or ``adaptive``."""
    value = (text or "").strip()
    if value == _ADAPTIVE_SCHEDULE:
        return (_ADAPTIVE_SCHEDULE, None)
    kind, sep, tail = value.partition(":")
    if kind == "fixed" and sep:
        try:
            submit = int(tail)
        except ValueError:
            submit = 0
        if submit >= 1:
            return ("fixed", submit)
    raise DrafterConfigError(
        f"invalid speculative schedule {text!r}; expected fixed:<K> or "
        f"{_ADAPTIVE_SCHEDULE}")


def resolve_spec_schedule(
    family: str, *, env_value: Any = _UNSET,
) -> tuple[str, int | None, str]:
    """The served schedule for ``family``: ``(kind, submit_length, label)``.

    ``MOESPRESSO_DS4_SPEC_SCHEDULE`` overrides the family default and fails
    closed on any unparseable value.
    """
    raw = os.environ.get(SPEC_SCHEDULE_ENV) if env_value is _UNSET else env_value
    text = "" if raw is None else str(raw).strip()
    label = text or DEFAULT_SPEC_SCHEDULES.get(family, _ADAPTIVE_SCHEDULE)
    kind, submit = parse_spec_schedule(label)
    return kind, submit, label


def _attach_drafter_policy(model, payload: dict) -> dict:
    """Attach the drafter decision payload to the served model.

    The census surfaces (``iqk_engagement``, ``ssd_streaming_stats``, the
    speed-stats count keys) export it so served arms attest which way the
    decision went and the numbers that drove it.
    """
    model._moespresso_ds4_drafter_policy = payload
    return payload


def _override_policy(decision: str, reason: str, env_text: str) -> dict:
    return {
        "mode": "override",
        "decision": decision,
        "reason": reason,
        "env": env_text,
    }


def _manifest_is_deepseek_v4(manifest: dict) -> bool:
    architecture = manifest.get("architecture") or {}
    return (
        architecture.get("family") == _DEEPSEEK_V4_FAMILY
        or architecture.get("prompt_renderer") == _DEEPSEEK_V4_RENDERER
    )


def _load_drafter(family: str, sidecar_dir: Path, model):
    """Load the sidecar for ``family``, sharing the target's embedding and
    language-model head. Fails closed on an unknown family."""
    if family == "dspark":
        from moespresso.runtime.deepseek_v4.dspark_load import load_dspark_sidecar

        drafter, _ = load_dspark_sidecar(
            sidecar_dir, embed=model.model.embed, lm_head=model.lm_head)
        return drafter
    if family == "dflash":
        # The DFlash sidecar carries its own pruned language-model head;
        # only the target embedding is shared.
        from moespresso.runtime.deepseek_v4.dflash_load import load_dflash_sidecar

        drafter, _ = load_dflash_sidecar(sidecar_dir, embed=model.model.embed)
        return drafter
    raise DrafterConfigError(f"unknown drafter family {family!r}")


def _full_expert_residency(model) -> bool:
    """True when every routed expert of every layer is resident.

    The signal is the runtime state the build attached, not the package
    family. A resident builder sets no
    ``_moespresso_ssd_streaming_capacity`` attribute, so an absent attribute
    means full residency. A pooled build
    sets the attribute even when the capacity planner granted the full
    expert set, so the live projection pools are checked directly: every
    pool must actually hold its whole expert set, per-layer capacity
    overrides included. Capacity alone is insufficient when an explicit
    no-prewarm setting leaves a full-size pool cold.
    """
    if getattr(model, "_moespresso_ssd_streaming_capacity", None) is None:
        return True
    from moespresso.runtime.ssd_streaming_build import _all_pools_fully_resident

    return _all_pools_fully_resident(model)


def _sidecar_artifact_id(family: str, sidecar_dir: Path) -> str | None:
    """Artifact id from the sidecar manifest, for load-time provenance.

    Printed so the serve log records which sidecar build produced the
    drafter. A sidecar paired with mismatched target weights costs
    acceptance only, never correctness (verification compares every draft
    token against the target's own logits), so this read is provenance and
    stays tolerant: an unreadable manifest yields None and the loader
    remains the authority on validity.
    """
    name = _SIDECAR_MANIFEST_NAMES.get(family)
    if name is None:
        return None
    try:
        payload = json.loads((Path(sidecar_dir) / name).read_text())
    except (OSError, ValueError):
        return None
    artifact_id = payload.get("artifact_id")
    return str(artifact_id) if artifact_id else None


def _install_drafter(
    model,
    family: str,
    sidecar_dir: Path,
    *,
    load_drafter_fn: Callable | None = None,
    install_tap_fn: Callable | None = None,
    auto: bool = False,
) -> ServedDrafter:
    """Load ``family`` from ``sidecar_dir``, install the hidden-state tap,
    and attach the served drafter to the model.

    Raises on any load or tap failure; the caller decides whether that
    refuses startup (explicit selection) or falls back to plain serving
    (automatic selection).
    """
    if load_drafter_fn is None:
        load_drafter_fn = _load_drafter
    drafter = load_drafter_fn(family, sidecar_dir, model)
    if install_tap_fn is None:
        from moespresso.runtime.deepseek_v4.spec_decode import (
            install_hidden_tap as install_tap_fn,
        )
    tap = install_tap_fn(model, drafter.tap_layer_ids, drafter.tap_transform)
    artifact_id = _sidecar_artifact_id(family, sidecar_dir)
    served = ServedDrafter(
        family=family,
        sidecar_dir=sidecar_dir,
        drafter=drafter,
        tap=tap,
        auto=auto,
        artifact_id=artifact_id,
    )
    model._moespresso_ds4_drafter = served
    label = f"{family}(auto)" if auto else family
    provenance = f" artifact={artifact_id}" if artifact_id else ""
    print(
        f"[serve] drafter={label} sidecar={sidecar_dir}{provenance} "
        f"block_size={drafter.block_size}",
        flush=True,
    )
    return served


def _bundled_drafter_component(manifest: dict) -> dict | None:
    """The manifest's declared drafter component when it is a DSpark one."""
    component = manifest.get("drafter")
    if not isinstance(component, dict):
        return None
    if component.get("family") != "dspark":
        return None
    return component


def _bundled_stage_count(package_dir: Path) -> int:
    """DSpark stage count from the bundled sidecar manifest (tolerant read
    of an already-presence-checked file; the loader re-validates)."""
    payload = json.loads(
        (Path(package_dir) / _SIDECAR_MANIFEST_NAMES["dspark"]).read_text())
    return int((payload.get("dspark") or {})["n_mtp_layers"])


def _resolve_bundled_drafter(
    model,
    manifest: dict,
    component: dict,
    *,
    package_dir: Path,
    load_drafter_fn: Callable | None,
    install_tap_fn: Callable | None,
    budget_fn: Callable | None,
) -> tuple[ServedDrafter | None, str]:
    """Capacity-gated automatic selection of the package's bundled drafter.

    The component contract is all-or-nothing: a distribution shipped
    without the optional component's files serves plain (counted, never an
    error), a partial component serves plain and names the corruption, and
    a present component enables the drafter only when the machine passes
    the wired-budget capacity check. Never raises.
    """
    policy: dict[str, Any] = {"mode": "auto", "decision": "off"}
    names = [
        str(entry.get("path"))
        for entry in component.get("files", [])
        if isinstance(entry, dict) and entry.get("path")
    ]
    if not names:
        policy["reason"] = "component-undeclared-files"
        _attach_drafter_policy(model, policy)
        print("[serve] spec: auto off (drafter component declares no files)",
              flush=True)
        return None, "off(auto:drafter-component-invalid)"
    present = [(Path(package_dir) / name).is_file() for name in names]
    if not any(present):
        policy["reason"] = "component-absent"
        _attach_drafter_policy(model, policy)
        print(
            f"[serve] spec: auto off (optional drafter component absent, "
            f"{len(names)} file(s) not present)",
            flush=True,
        )
        return None, "off(auto:drafter-absent)"
    if not all(present):
        missing = sorted(
            name for name, here in zip(names, present) if not here)
        policy["reason"] = "component-partial"
        policy["missing_files"] = missing
        _attach_drafter_policy(model, policy)
        print(
            "[serve] spec: auto off (drafter component partially present; "
            "missing: " + ", ".join(missing) + ")",
            flush=True,
        )
        return None, "off(auto:drafter-partial)"
    try:
        n_stages = _bundled_stage_count(package_dir)
    except Exception as e:
        policy["reason"] = f"sidecar-manifest-unreadable: {e}"
        _attach_drafter_policy(model, policy)
        print(f"[serve] spec: auto off (sidecar manifest unreadable: {e})",
              flush=True)
        return None, "off(auto:sidecar-manifest-unreadable)"
    from moespresso.runtime.deepseek_v4.drafter_policy import (
        evaluate_bundled_drafter_budget,
    )

    try:
        budget_kwargs = {} if budget_fn is None else {"budget_fn": budget_fn}
        decision = evaluate_bundled_drafter_budget(
            manifest,
            n_stages=n_stages,
            sort_nsplit_env=os.environ.get(
                "MOESPRESSO_DSV4_IQK_SORT_NSPLIT"),
            **budget_kwargs)
    except Exception as e:
        decision = {"decision": "off", "reason": f"policy-error: {e}"}
    policy = {"mode": "auto", **decision}
    if policy.get("decision") != "on":
        _attach_drafter_policy(model, policy)
        print(
            "[serve] spec: auto off (drafter budget: "
            f"{policy.get('reason')})", flush=True)
        return None, "off(auto:budget)"
    if policy.get("sort_nsplit_source") == "policy":
        # The drafter fits only at the parts-32 working-set floor; raise
        # the process default before anything prefills. The environment
        # variable still wins on every read, and a default that cannot be
        # applied fails the decision closed.
        try:
            from moespresso.runtime.deepseek_v4.iqk_experts import (
                set_sorted_prefill_default,
            )

            set_sorted_prefill_default(int(policy["sort_nsplit"]))
        except Exception as e:
            policy["decision"] = "off"
            policy["reason"] = f"nsplit-apply-failed: {e}"
            _attach_drafter_policy(model, policy)
            print(
                f"[serve] spec: auto off (nsplit apply failed: {e})",
                flush=True,
            )
            return None, "off(auto:nsplit-apply-failed)"
        print(
            "[serve] spec: sorted-route split raised to parts "
            f"{policy['sort_nsplit']} (drafter fits at the parts-32 "
            "working-set floor)",
            flush=True,
        )
    try:
        served = _install_drafter(
            model,
            "dspark",
            Path(package_dir),
            load_drafter_fn=load_drafter_fn,
            install_tap_fn=install_tap_fn,
            auto=True,
        )
    except Exception as e:
        policy["decision"] = "off"
        policy["reason"] = f"sidecar-failed: {e}"
        _attach_drafter_policy(model, policy)
        print(f"[serve] spec: auto off (sidecar failed: {e})", flush=True)
        return None, "off(auto:sidecar-failed)"
    _attach_drafter_policy(model, policy)
    return served, "dspark(auto)"


def _resolve_auto_drafter(
    model,
    manifest: dict,
    *,
    package_dir: Path | None,
    load_drafter_fn: Callable | None,
    install_tap_fn: Callable | None,
    residency_fn: Callable | None,
    budget_fn: Callable | None = None,
) -> tuple[ServedDrafter | None, str]:
    """Automatic drafter selection for a DeepSeek-V4 target.

    A manifest-declared bundled drafter takes the capacity-gated path and
    requires full expert residency. A package that declares no component
    serves plain: the declaration is the whole of automatic selection, and
    every other family is explicit-selection only. Any miss prints one line
    naming the reason and resolves to plain serving; this path never raises.
    """
    if residency_fn is None:
        residency_fn = _full_expert_residency
    try:
        resident = bool(residency_fn(model))
    except Exception as e:
        # A residency signal that cannot be read is not proof of full
        # residency; fail closed to plain serving instead of refusing
        # startup for an automatic feature.
        _attach_drafter_policy(model, {
            "mode": "auto", "decision": "off",
            "reason": f"residency-unknown: {e}"})
        print(
            f"[serve] spec: auto off (residency signal unavailable: {e})",
            flush=True,
        )
        return None, "off(auto:residency-unknown)"
    if not resident:
        _attach_drafter_policy(model, {
            "mode": "auto", "decision": "off",
            "reason": "bounded-residency"})
        print("[serve] spec: auto off (bounded residency)", flush=True)
        return None, "off(auto:bounded-residency)"
    component = _bundled_drafter_component(manifest)
    if component is not None and package_dir is not None:
        return _resolve_bundled_drafter(
            model,
            manifest,
            component,
            package_dir=Path(package_dir),
            load_drafter_fn=load_drafter_fn,
            install_tap_fn=install_tap_fn,
            budget_fn=budget_fn,
        )
    _attach_drafter_policy(model, {
        "mode": "auto", "decision": "off",
        "reason": "no-declared-drafter"})
    print(
        "[serve] spec: auto off (package declares no drafter component)",
        flush=True,
    )
    return None, "off(auto:no-declared-drafter)"


def resolve_env_drafter(
    model,
    manifest: dict,
    *,
    package_dir: Path | None = None,
    env_value: Any = _UNSET,
    load_drafter_fn: Callable | None = None,
    install_tap_fn: Callable | None = None,
    residency_fn: Callable | None = None,
    budget_fn: Callable | None = None,
) -> tuple[ServedDrafter | None, str | None]:
    """Resolve and install the drafter for a freshly loaded model.

    Returns ``(served, state)``: the installed drafter (or None) and the
    one-token drafter state for the runtime truth line (``dspark(auto)``,
    ``dspark``, ``dflash``, ``off``, or an ``off(auto:...)`` reason). The
    state is None for non-DeepSeek-V4 packages, which have no
    drafter contract.

    An explicit ``MOESPRESSO_DS4_DRAFTER`` selection always wins and is
    recorded as an override on the policy attestation: ``off`` disables
    speculation, and a named drafter that cannot load raises
    ``DrafterConfigError`` so startup refuses loudly instead of silently
    serving without the requested drafter. An absent or empty variable
    resolves automatically (see ``_resolve_auto_drafter``); the automatic
    path never refuses startup.
    """
    raw = os.environ.get(DRAFTER_ENV) if env_value is _UNSET else env_value
    text = "" if raw is None else str(raw).strip()
    is_deepseek_v4 = _manifest_is_deepseek_v4(manifest)
    if not text:
        if not is_deepseek_v4:
            return None, None
        return _resolve_auto_drafter(
            model,
            manifest,
            package_dir=package_dir,
            load_drafter_fn=load_drafter_fn,
            install_tap_fn=install_tap_fn,
            residency_fn=residency_fn,
            budget_fn=budget_fn,
        )
    if not is_deepseek_v4:
        family = (manifest.get("architecture") or {}).get("family")
        print(
            f"[serve] drafter=off ({DRAFTER_ENV} applies to DeepSeek-V4 "
            f"packages only; this package family is {family!r})",
            flush=True,
        )
        return None, None
    config = parse_drafter_env(text)
    if config is None:
        # An explicit off is an override of the automatic policy and is
        # recorded as one.
        _attach_drafter_policy(
            model, _override_policy("off", "env-off", text))
        return None, "off"
    try:
        served = _install_drafter(
            model,
            config.family,
            config.sidecar_dir,
            load_drafter_fn=load_drafter_fn,
            install_tap_fn=install_tap_fn,
        )
    except DrafterConfigError:
        raise
    except Exception as e:
        raise DrafterConfigError(
            f"{DRAFTER_ENV}={config.family}:{config.sidecar_dir} failed to "
            f"load: {e}") from e
    _attach_drafter_policy(
        model, _override_policy("on", "env-selected", text))
    return served, config.family


def install_env_drafter(
    model,
    manifest: dict,
    *,
    package_dir: Path | None = None,
    env_value: Any = _UNSET,
    load_drafter_fn: Callable | None = None,
    install_tap_fn: Callable | None = None,
    residency_fn: Callable | None = None,
) -> ServedDrafter | None:
    """Install the environment-selected drafter on a freshly loaded model.

    Thin wrapper over ``resolve_env_drafter`` for callers that do not need
    the truth-line state token; see it for the selection semantics.
    """
    served, _ = resolve_env_drafter(
        model,
        manifest,
        package_dir=package_dir,
        env_value=env_value,
        load_drafter_fn=load_drafter_fn,
        install_tap_fn=install_tap_fn,
        residency_fn=residency_fn,
    )
    return served


def spec_sampler_eligible(
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float | None,
    top_logprobs: int | None = None,
    greedy_only: bool = False,
) -> bool:
    """True when the acceptance rules reproduce the requested sampler exactly.

    Greedy acceptance matches argmax decoding, and the speculative sampling
    rule recovers the pure-temperature target distribution. Distribution
    shaping (top-p, top-k, min-p), logits processors (presence penalty), and
    per-token logprob capture have no speculative equivalent, so those
    requests take the plain path. ``greedy_only`` is the installed drafter's
    advertisement: sampled acceptance needs the draft distribution that
    produced the proposal, and a greedy-only drafter has none, so
    temperature > 0 is then ineligible as well.
    """
    if greedy_only and float(temperature) > 0.0:
        return False
    return (
        float(temperature) >= 0.0
        and float(top_p) == 1.0
        and int(top_k) == 0
        and float(min_p) == 0.0
        and presence_penalty in (None, 0.0)
        and top_logprobs is None
    )


def _tokenizer_eos_ids(tokenizer) -> list[int]:
    ids = getattr(tokenizer, "eos_token_ids", None)
    if ids:
        return [int(t) for t in ids]
    eos = getattr(tokenizer, "eos_token_id", None)
    return [] if eos is None else [int(eos)]


class _DecodeDiffDetokenizer:
    """Streaming detokenization over ``tokenizer.decode`` for tokenizers
    without an mlx-lm streaming detokenizer (test doubles). ``last_segment``
    consumes, matching the mlx-lm detokenizer contract."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._tokens: list[int] = []
        self._text = ""
        self._offset = 0

    def add_token(self, token: int) -> None:
        self._tokens.append(int(token))
        self._text = self._tokenizer.decode(self._tokens)

    def finalize(self) -> None:
        pass

    @property
    def last_segment(self) -> str:
        segment = self._text[self._offset:]
        self._offset = len(self._text)
        return segment


def _fresh_detokenizer(tokenizer):
    detokenizer = getattr(tokenizer, "detokenizer", None)
    if detokenizer is not None:
        reset = getattr(detokenizer, "reset", None)
        if reset is not None:
            reset()
        return detokenizer
    return _DecodeDiffDetokenizer(tokenizer)


@dataclass
class _SpecResponse:
    """Per-token response mirroring the stream-generate response surface."""

    text: str
    token: int
    logprobs: Any
    prompt_tokens: int
    generation_tokens: int
    finish_reason: str | None


class _SpecEmitter:
    """Response accounting over the spec loop's committed-token stream.

    Mirrors the plain stream contract: one incremental text segment per
    non-terminal token, first-token latency at the first commit, and a final
    response carrying the finish reason plus the finalized detokenizer tail.
    A stop token ends the stream without surfacing its own text, and the
    token that reaches ``max_tokens`` surfaces through the final response.
    """

    def __init__(
        self,
        tokenizer,
        *,
        eos_ids: Sequence[int],
        max_tokens: int,
        prompt_tokens: int,
        response_callback: Callable[[int, object], None] | None = None,
        first_token_callback: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self._detokenizer = _fresh_detokenizer(tokenizer)
        self._eos = {int(t) for t in eos_ids}
        self._max_tokens = int(max_tokens)
        self._prompt_tokens = int(prompt_tokens)
        self._response_callback = response_callback
        self._first_token_callback = first_token_callback
        self._clock = clock
        self._start = clock()
        self._parts: list[str] = []
        self._count = 0
        self._last_token: int | None = None
        self.first_token_seconds: float | None = None
        self.finish_reason: str | None = None

    def feed(self, tokens: Sequence[int]) -> None:
        for token in tokens:
            if self.finish_reason is not None:
                return
            token = int(token)
            if self.first_token_seconds is None:
                self.first_token_seconds = self._clock() - self._start
                if self._first_token_callback is not None:
                    self._first_token_callback()
            self._last_token = token
            self._count += 1
            if token in self._eos:
                self.finish_reason = "stop"
                return
            self._detokenizer.add_token(token)
            if self._count >= self._max_tokens:
                self.finish_reason = "length"
                return
            segment = self._detokenizer.last_segment
            self._parts.append(segment)
            self._emit(segment, token, None)

    def _emit(self, text: str, token: int | None, finish_reason: str | None) -> None:
        if self._response_callback is None or token is None:
            return
        self._response_callback(
            self._count,
            _SpecResponse(
                text=text,
                token=token,
                logprobs=None,
                prompt_tokens=self._prompt_tokens,
                generation_tokens=self._count,
                finish_reason=finish_reason,
            ),
        )

    def finish(self) -> tuple[str, str]:
        """Flush the detokenizer tail; returns (text, finish_reason)."""
        self._detokenizer.finalize()
        segment = self._detokenizer.last_segment
        self._parts.append(segment)
        reason = self.finish_reason or "length"
        self._emit(segment, self._last_token, reason)
        return "".join(self._parts), reason


def _run_spec_generate(model, **kwargs):
    """Run the spec loop under the same wired-limit and generation-stream
    scope the plain stream-generate path uses."""
    import mlx.core as mx
    from mlx_lm.generate import generation_stream, wired_limit

    from moespresso.runtime.deepseek_v4.spec_decode import spec_generate

    with wired_limit(model, [generation_stream]):
        with mx.stream(generation_stream):
            return spec_generate(model, **kwargs)


def _resumable_dspark(served: ServedDrafter) -> bool:
    """Whether the raw served DSpark owns the complete resume protocol."""
    if served.family != "dspark":
        return False
    return all(
        callable(getattr(served.drafter, name, None))
        for name in (
            "state_frontier",
            "state_nbytes",
            "export_state",
            "import_state",
        )
    )


def spec_cache_producer_rail(
    source: ServedDrafter | SpecCacheCompanion,
    *,
    schedule: str | None = None,
) -> tuple[str, ...] | None:
    """Return the expected producer rail for resumable DSpark state.

    A live ``ServedDrafter`` uses the supplied resolved schedule label or
    resolves the current one, and returns ``None`` unless it exposes the full
    DSpark resume protocol and a stable sidecar artifact id.
    ``SpecCacheCompanion`` uses the same constructor for its stored identity,
    preventing the lookup and publication tuples from drifting apart.
    """
    if isinstance(source, ServedDrafter):
        if not _resumable_dspark(source):
            return None
        if not isinstance(source.artifact_id, str) or not source.artifact_id:
            return None
        if schedule is None:
            _, _, schedule = resolve_spec_schedule(source.family)
        elif not isinstance(schedule, str) or not schedule:
            raise ValueError("a resolved speculative schedule is required")
        schema = SPEC_CACHE_SCHEMA
        producer_lattice = SPEC_PRODUCER_LATTICE
        family = source.family
        artifact_id = source.artifact_id
        schedule_label = schedule
    elif isinstance(source, SpecCacheCompanion):
        schema = source.schema
        producer_lattice = source.producer_lattice
        family = source.family
        artifact_id = source.artifact_id
        schedule_label = source.schedule
    else:
        return None

    return (
        "spec",
        schema,
        producer_lattice,
        family,
        artifact_id,
        schedule_label,
    )


def _prepare_spec_continuation(
    served: ServedDrafter,
    continuation: SpecContinuation,
    *,
    schedule: str,
) -> object:
    """Validate provenance and import DSpark state before response streaming."""
    if not isinstance(continuation, SpecContinuation):
        raise SpecContinuationError("invalid speculative continuation input")
    if not _resumable_dspark(served):
        raise SpecContinuationError("the served drafter cannot resume DSpark state")
    if not isinstance(continuation.companion, SpecCacheCompanion):
        raise SpecContinuationError("invalid speculative cache companion")
    companion = continuation.companion
    if companion.schema != SPEC_CACHE_SCHEMA:
        raise SpecContinuationError("speculative cache companion schema mismatch")
    if companion.producer_lattice != SPEC_PRODUCER_LATTICE:
        raise SpecContinuationError("speculative cache producer lattice mismatch")
    if companion.family != served.family:
        raise SpecContinuationError("speculative cache drafter family mismatch")
    if not served.artifact_id or companion.artifact_id != served.artifact_id:
        raise SpecContinuationError("speculative cache sidecar artifact mismatch")
    if companion.schedule != schedule:
        raise SpecContinuationError("speculative cache schedule mismatch")
    offset = continuation.prefix_offset
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise SpecContinuationError(
            "speculative continuation prefix_offset must be a non-negative int"
        )
    if (
        isinstance(companion.frontier, bool)
        or not isinstance(companion.frontier, int)
        or companion.frontier < 0
    ):
        raise SpecContinuationError("invalid speculative cache companion frontier")
    if companion.frontier != offset:
        raise SpecContinuationError("speculative cache companion frontier mismatch")
    if getattr(companion.capsule, "frontier", None) != offset:
        raise SpecContinuationError("DSpark capsule frontier mismatch")

    from moespresso.runtime.disk_kv import caches_all_at_offset

    try:
        target_aligned = caches_all_at_offset(continuation.target_cache, offset)
    except Exception:
        target_aligned = False
    if not target_aligned:
        raise SpecContinuationError("target cache frontier mismatch")
    try:
        state = served.drafter.import_state(companion.capsule)
        state_frontier = int(served.drafter.state_frontier(state))
    except Exception as exc:
        raise SpecContinuationError("DSpark capsule import failed") from exc
    if state_frontier != offset:
        raise SpecContinuationError("imported DSpark state frontier mismatch")
    return state


def _cache_candidate_skip(reason: str) -> dict[str, str]:
    return {"status": "skipped", "reason": reason}


def _prepare_spec_cache_candidate(
    served: ServedDrafter,
    generation,
    *,
    schedule: str,
    prompt_token_count: int,
) -> tuple[Any, int | None, SpecCacheCompanion | None, dict[str, str]]:
    """Validate and detach a successful generation's cache candidate.

    This does not insert into a store. Target and token-frontier failures return
    no cache because no safe key can be assigned. Once those fields validate,
    DSpark state or capsule failures retain the target as a plain-continuation
    candidate and omit only its companion.
    """
    expected_rail = spec_cache_producer_rail(served, schedule=schedule)
    if expected_rail is None:
        return None, None, None, _cache_candidate_skip("missing_sidecar_artifact")
    target_cache = getattr(generation, "target_cache", None)
    frontier = getattr(generation, "frontier", None)
    if target_cache is None or frontier is None:
        return None, None, None, _cache_candidate_skip("generation_state_missing")
    if isinstance(frontier, bool) or not isinstance(frontier, int) or frontier < 0:
        return None, None, None, _cache_candidate_skip("invalid_generation_frontier")
    if frontier < prompt_token_count:
        return None, None, None, _cache_candidate_skip("frontier_before_prompt")
    if frontier > prompt_token_count + len(generation.tokens):
        return None, None, None, _cache_candidate_skip("frontier_beyond_public_tokens")

    from moespresso.runtime.disk_kv import caches_all_at_offset

    try:
        target_aligned = caches_all_at_offset(target_cache, frontier)
    except Exception:
        target_aligned = False
    if not target_aligned:
        return None, None, None, _cache_candidate_skip("target_frontier_mismatch")

    state = getattr(generation, "drafter_state", None)
    if state is None:
        return (
            target_cache,
            frontier,
            None,
            _cache_candidate_skip("generation_state_missing"),
        )
    try:
        state_frontier = int(served.drafter.state_frontier(state))
    except Exception:
        return (
            target_cache,
            frontier,
            None,
            _cache_candidate_skip("drafter_frontier_invalid"),
        )
    if state_frontier != frontier:
        return (
            target_cache,
            frontier,
            None,
            _cache_candidate_skip("drafter_frontier_mismatch"),
        )
    try:
        capsule = served.drafter.export_state(state)
    except Exception:
        return (
            target_cache,
            frontier,
            None,
            _cache_candidate_skip("drafter_export_failed"),
        )
    if getattr(capsule, "frontier", None) != frontier:
        return (
            target_cache,
            frontier,
            None,
            _cache_candidate_skip("capsule_frontier_mismatch"),
        )
    companion = SpecCacheCompanion(
        family=served.family,
        artifact_id=served.artifact_id,
        schedule=schedule,
        frontier=frontier,
        capsule=capsule,
    )
    if companion.producer_rail != expected_rail:
        return (
            target_cache,
            frontier,
            None,
            _cache_candidate_skip("companion_producer_rail_mismatch"),
        )
    return (
        target_cache,
        frontier,
        companion,
        {"status": "ready"},
    )


def spec_generation_result(
    model,
    tokenizer,
    served: ServedDrafter,
    prompt,
    *,
    max_tokens: int,
    temperature: float,
    cached_tokens: int | None = None,
    prefill_step_size: int | None = None,
    prefill_plan: Sequence[int] | None = None,
    prefill_progress_callback: Callable[[SpecPrefillProgress], None] | None = None,
    prefill_progress_frontiers: Sequence[int] | None = None,
    response_callback: Callable[[int, object], None] | None = None,
    first_token_callback: Callable[[], None] | None = None,
    continuation_ready_callback: Callable[[], None] | None = None,
    spec_generate_fn: Callable | None = None,
    continuation: SpecContinuation | None = None,
) -> GenerationResult:
    """Serve one request through the speculative loop.

    Consumes the same pre-rendered prompt the plain path would (a string is
    encoded with the stream-generate rule, token ids pass through) and
    returns the plain path's ``GenerationResult`` surface plus the
    ``speculative`` stats block. A validated ``continuation`` supplies a target
    cache and DSpark capsule at a nonzero absolute prefix. Successful DSpark
    runs return their target cache and a detached, provenance-bound companion.
    A companion mismatch preserves any independently valid target candidate and
    omits only the companion. A paired prefill callback receives only materialized,
    equal target and raw-drafter frontiers from the speculative loop; it does
    not run during the final anchor forward or generation. ``spec_generate_fn``
    is the test seam.
    """
    if isinstance(prompt, str):
        from moespresso.runtime.prefix_cache import encode_rendered_prompt

        prompt_ids = encode_rendered_prompt(tokenizer, prompt)
    else:
        prompt_ids = [int(t) for t in prompt]

    schedule_kind, submit_length, schedule_label = resolve_spec_schedule(served.family)
    resumable = _resumable_dspark(served)
    prompt_offset = 0
    imported_state = None
    effective_cached_tokens = cached_tokens
    if continuation is not None:
        if not prompt_ids:
            raise SpecContinuationError(
                "an exact speculative cache hit requires continuation logits"
            )
        prompt_offset = continuation.prefix_offset
        if cached_tokens is not None:
            if isinstance(cached_tokens, bool) or not isinstance(cached_tokens, int):
                raise SpecContinuationError("cached_tokens must be an int")
            if cached_tokens != prompt_offset:
                raise SpecContinuationError(
                    "cached_tokens does not match speculative continuation offset"
                )
        imported_state = _prepare_spec_continuation(
            served,
            continuation,
            schedule=schedule_label,
        )
        if continuation_ready_callback is not None:
            continuation_ready_callback()
        effective_cached_tokens = prompt_offset

    prompt_token_count = prompt_offset + len(prompt_ids)
    eos_ids = _tokenizer_eos_ids(tokenizer)
    emitter = _SpecEmitter(
        tokenizer,
        eos_ids=eos_ids,
        max_tokens=int(max_tokens),
        prompt_tokens=prompt_token_count,
        response_callback=response_callback,
        first_token_callback=first_token_callback,
    )
    run = spec_generate_fn
    if run is None:
        run = _run_spec_generate
    drafter = served.drafter
    schedule_kwargs: dict[str, Any] = {}
    if schedule_kind == "fixed":
        from moespresso.runtime.deepseek_v4.spec_decode import FixedSubmitDrafter

        drafter = FixedSubmitDrafter(drafter, submit_length)
        schedule_kwargs = {"adaptive_cap": False, "confidence_threshold": 0.0}
    if resumable:
        # The schedule wrapper drives proposals and ingest. The raw drafter
        # remains the only owner of state validation, import, and export.
        schedule_kwargs["state_owner"] = served.drafter
    elif prefill_progress_callback is not None:
        raise ValueError(
            "a speculative prefill progress callback requires resumable DSpark"
        )
    if prefill_plan is not None:
        schedule_kwargs["prefill_plan"] = prefill_plan
    if prefill_progress_callback is not None:
        schedule_kwargs["prefill_progress_callback"] = prefill_progress_callback
    if prefill_progress_frontiers is not None:
        schedule_kwargs["prefill_progress_frontiers"] = prefill_progress_frontiers
    if continuation is not None:
        schedule_kwargs.update(
            target_cache=continuation.target_cache,
            drafter_state=imported_state,
            prompt_offset=prompt_offset,
        )
    t_start = time.perf_counter()
    generation = run(
        model,
        drafter=drafter,
        tap=served.tap,
        prompt_ids=prompt_ids,
        max_new_tokens=int(max_tokens),
        temperature=float(temperature),
        eos_ids=eos_ids,
        prefill_step_size=int(prefill_step_size or 2048),
        on_commit=emitter.feed,
        **schedule_kwargs,
    )
    generation_seconds = time.perf_counter() - t_start
    text, finish_reason = emitter.finish()
    stats = generation.stats
    speculative_prompt_cache = None
    cache_frontier = None
    cache_companion = None
    cache_readiness = None
    if resumable:
        try:
            (
                speculative_prompt_cache,
                cache_frontier,
                cache_companion,
                cache_readiness,
            ) = _prepare_spec_cache_candidate(
                served,
                generation,
                schedule=schedule_label,
                prompt_token_count=prompt_token_count,
            )
        except Exception:
            # Generation has already succeeded and may have streamed tokens.
            # Cache candidate preparation is optional and must never invalidate
            # completed output.
            speculative_prompt_cache = None
            cache_frontier = None
            cache_companion = None
            cache_readiness = _cache_candidate_skip("candidate_validation_failed")
    speculative = {
        "drafter": served.family,
        "schedule": schedule_label,
        "rounds": stats.rounds,
        "proposed": stats.proposed,
        "accepted": stats.accepted,
        "mean_accepted_length": stats.mean_accepted_length,
        "plain_fallbacks": stats.plain_fallbacks,
        "submit_length_counts": {str(k): v for k, v in sorted(stats.submit_length_counts.items())},
    }
    if cache_readiness is not None:
        speculative["cache_publication"] = cache_readiness
    return GenerationResult(
        text=text,
        finish_reason=finish_reason,
        prompt_tokens=prompt_token_count,
        completion_tokens=len(generation.tokens),
        cached_tokens=effective_cached_tokens,
        generated_token_ids=tuple(int(t) for t in generation.tokens),
        prompt_cache=None,
        speculative_prompt_cache=speculative_prompt_cache,
        cache_frontier=cache_frontier,
        cache_companion=cache_companion,
        first_token_seconds=emitter.first_token_seconds,
        generation_seconds=generation_seconds,
        speculative=speculative,
    )
