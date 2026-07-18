"""Opt-in calibration harness for affine role-elasticity experiments.

This is not normal CI and not a benchmark. It evaluates already-built packages
against one prompt set, records raw generated text, and compares every candidate
against a required Q8/all-8 package.

Example:

    MOESPRESSO_RUN_AFFINE_ROLE_ELASTICITY_SMOKE=1 \
    MOESPRESSO_AFFINE_ROLE_ELASTICITY_Q8_PACKAGE=/path/to/q8-package \
    MOESPRESSO_AFFINE_ROLE_ELASTICITY_PACKAGES=auto_050=/path/a,manual_050=/path/b \
    MOESPRESSO_AFFINE_ROLE_ELASTICITY_OUT=/tmp/affine_elasticity_evidence.json \
    MOESPRESSO_AFFINE_ROLE_ELASTICITY_MAX_TOKENS=4096 \
    uv run --locked python -m pytest tests/test_affine_role_elasticity_smoke.py -s
"""

from __future__ import annotations

import gc
import json
import os
import platform
from pathlib import Path

import pytest

from moespresso.core.artifact import read_artifact
from moespresso.optimize.affine_elasticity import (
    PROMPTS,
    PROMPT_SET_VERSION,
    q8_relative_summary,
    role_bit_summary,
    score_generation,
    summarize_generation,
)


RUN_ENV = "MOESPRESSO_RUN_AFFINE_ROLE_ELASTICITY_SMOKE"
Q8_ENV = "MOESPRESSO_AFFINE_ROLE_ELASTICITY_Q8_PACKAGE"
PACKAGES_ENV = "MOESPRESSO_AFFINE_ROLE_ELASTICITY_PACKAGES"
OUT_ENV = "MOESPRESSO_AFFINE_ROLE_ELASTICITY_OUT"
MAX_TOKENS_ENV = "MOESPRESSO_AFFINE_ROLE_ELASTICITY_MAX_TOKENS"
ENABLE_THINKING_ENV = "MOESPRESSO_AFFINE_ROLE_ELASTICITY_ENABLE_THINKING"


def _parse_named_packages(raw: str) -> dict[str, Path]:
    """Parse `name=/path,name2=/path2` into a mapping."""
    out: dict[str, Path] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"{PACKAGES_ENV} entry must be name=/path, got {part!r}")
        name, value = part.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"{PACKAGES_ENV} entry has an empty name: {part!r}")
        out[name] = Path(value).expanduser()
    return out


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _optimizer_decision(package: Path) -> dict | None:
    path = package / "optimizer_decision.json"
    if not path.exists():
        return None
    return read_artifact(path)


def _evaluate_package(name: str, package: Path, *, max_tokens: int,
                      enable_thinking: bool) -> dict:
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    pytest.importorskip("jang_tools.loader")

    from moespresso.package.constants import MANIFEST_NAME
    from moespresso.runtime.http import render_prompt, rendering_identity
    from moespresso.runtime.kv_policy import parse_kv_policy
    from moespresso.runtime.serve import generate_with_metadata, load_served_model

    manifest = read_artifact(package / MANIFEST_NAME)
    decision = _optimizer_decision(package)
    model, tokenizer, manifest = load_served_model(package, manifest=manifest)
    kv_policy = parse_kv_policy({"live_kv_format": "mlx_affine_q8"})
    rendering_id = rendering_identity(manifest.get("tokenizer", {}).get("rendering_id"), None)

    rows = []
    try:
        for prompt in PROMPTS:
            print(f"[affine-elasticity] {name}: {prompt.id}", flush=True)
            rendered = render_prompt(
                [{"role": "user", "content": prompt.prompt}],
                tokenizer,
                template_kwargs={"enable_thinking": enable_thinking},
            )
            result = generate_with_metadata(
                model,
                tokenizer,
                rendered,
                kv_policy=kv_policy,
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
            )
            rows.append(score_generation(
                prompt,
                result.text,
                finish_reason=result.finish_reason,
                completion_tokens=result.completion_tokens,
            ))
    finally:
        del model
        del tokenizer
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass

    return {
        "package": str(package),
        "manifest_id": manifest.get("artifact_id"),
        "optimizer_decision_id": None if decision is None else decision.get("artifact_id"),
        "role_bit_summary": {} if decision is None else role_bit_summary(decision),
        "achieved": {} if decision is None else decision.get("achieved", {}),
        "rendering_id": rendering_id,
        "generation": {
            "summary": summarize_generation(rows),
            "prompts": rows,
        },
    }


def _write_evidence(path: Path, evidence: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2))


@pytest.mark.real_model
def test_affine_role_elasticity_calibration_smoke(tmp_path):
    if os.environ.get(RUN_ENV) != "1":
        pytest.skip(f"set {RUN_ENV}=1 to run the affine role-elasticity smoke")

    q8_raw = os.environ.get(Q8_ENV)
    packages_raw = os.environ.get(PACKAGES_ENV)
    if not q8_raw:
        pytest.skip(f"set {Q8_ENV}=<q8 package path>")
    if not packages_raw:
        pytest.skip(f"set {PACKAGES_ENV}=name=/path[,name=/path...]")

    q8_package = Path(q8_raw).expanduser()
    candidates = _parse_named_packages(packages_raw)
    max_tokens = int(os.environ.get(MAX_TOKENS_ENV, "4096"))
    enable_thinking = _env_bool(ENABLE_THINKING_ENV, True)
    out_path = Path(os.environ.get(
        OUT_ENV,
        str(tmp_path / "affine_role_elasticity_evidence.json"),
    )).expanduser()

    missing = [str(path) for path in [q8_package, *candidates.values()] if not path.exists()]
    if missing:
        pytest.skip(f"package path(s) not found: {missing}")

    evidence = {
        "kind": "affine_role_elasticity_calibration_smoke",
        "prompt_set_version": PROMPT_SET_VERSION,
        "max_tokens": max_tokens,
        "enable_thinking": enable_thinking,
        "live_kv_format": "mlx_affine_q8",
        "platform": platform.platform(),
        "variants": {},
    }

    variants = evidence["variants"]
    print("[affine-elasticity] q8_baseline", flush=True)
    variants["q8_baseline"] = _evaluate_package(
        "q8_baseline",
        q8_package,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
    )
    _write_evidence(out_path, evidence)

    q8_rows = variants["q8_baseline"]["generation"]["prompts"]
    for name, package in sorted(candidates.items()):
        print(f"[affine-elasticity] {name}", flush=True)
        entry = _evaluate_package(
            name,
            package,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
        entry["q8_relative"] = q8_relative_summary(entry["generation"]["prompts"], q8_rows)
        variants[name] = entry
        _write_evidence(out_path, evidence)

    print(out_path)

    assert variants["q8_baseline"]["generation"]["prompts"]
    for name in candidates:
        assert variants[name]["generation"]["prompts"]
        assert variants[name]["q8_relative"]["per_prompt"]
