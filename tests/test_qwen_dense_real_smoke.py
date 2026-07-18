"""Opt-in real-model smoke for local dense Qwen3.5 artifacts.

Default pytest must stay fast and model-free. This file only runs when explicitly
enabled:

    MOESPRESSO_RUN_DENSE_QWEN_08B_SMOKE=1 uv run --locked python -m pytest \
        tests/test_qwen_dense_real_smoke.py -s

It requires explicit local source/imatrix paths and can be pointed at a reusable
package path with environment variables. The test writes a small evidence JSON
file next to the package so the run is inspectable after pytest exits.
"""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class DenseQwenSmokeCase:
    label: str
    run_env: str
    source_env: str
    imatrix_env: str
    out_env: str
    reuse_env: str
    target_env: str
    kv_env: str
    max_tokens_env: str
    default_target_gb: str
    default_out_name: str
    evidence_kind: str
    evidence_file: str


QWEN_08B = DenseQwenSmokeCase(
    label="0.8B",
    run_env="MOESPRESSO_RUN_DENSE_QWEN_08B_SMOKE",
    source_env="MOESPRESSO_DENSE_QWEN_08B_SOURCE",
    imatrix_env="MOESPRESSO_DENSE_QWEN_08B_IMATRIX",
    out_env="MOESPRESSO_DENSE_QWEN_08B_OUT",
    reuse_env="MOESPRESSO_DENSE_QWEN_08B_REUSE_PACKAGE",
    target_env="MOESPRESSO_DENSE_QWEN_08B_TARGET_GB",
    kv_env="MOESPRESSO_DENSE_QWEN_08B_LIVE_KV",
    max_tokens_env="MOESPRESSO_DENSE_QWEN_08B_MAX_TOKENS",
    default_target_gb="1.0",
    default_out_name="dense-qwen-08b-pg",
    evidence_kind="dense_qwen_08b_real_smoke",
    evidence_file="dense_qwen_08b_smoke_evidence.json",
)

QWEN_4B = DenseQwenSmokeCase(
    label="4B",
    run_env="MOESPRESSO_RUN_DENSE_QWEN_4B_SMOKE",
    source_env="MOESPRESSO_DENSE_QWEN_4B_SOURCE",
    imatrix_env="MOESPRESSO_DENSE_QWEN_4B_IMATRIX",
    out_env="MOESPRESSO_DENSE_QWEN_4B_OUT",
    reuse_env="MOESPRESSO_DENSE_QWEN_4B_REUSE_PACKAGE",
    target_env="MOESPRESSO_DENSE_QWEN_4B_TARGET_GB",
    kv_env="MOESPRESSO_DENSE_QWEN_4B_LIVE_KV",
    max_tokens_env="MOESPRESSO_DENSE_QWEN_4B_MAX_TOKENS",
    default_target_gb="4.0",
    default_out_name="dense-qwen-4b-pg",
    evidence_kind="dense_qwen_4b_real_smoke",
    evidence_file="dense_qwen_4b_smoke_evidence.json",
)


def _optional_path_from_env(name: str) -> Path | None:
    if name not in os.environ:
        return None
    return Path(os.environ[name]).expanduser()


def _assistant_history_content(generated_text: str) -> str:
    """Recreate the assistant turn content after a thinking-prefilled prompt.

    The Qwen template's generation prompt opens `<think>\n` before generation.
    A later history render needs that prefix in the assistant content for the
    rendered turn-1 tokens to remain a prefix of the rendered full-history turn.
    """
    stripped = generated_text.lstrip()
    if stripped.startswith("<think>") or stripped.startswith("<|im_start|>"):
        return generated_text
    return "<think>\n" + generated_text


def _common_prefix_len(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def _run_dense_qwen_calibrated_real_model_smoke(case: DenseQwenSmokeCase, tmp_path):
    if os.environ.get(case.run_env) != "1":
        pytest.skip(f"set {case.run_env}=1 to run the local {case.label} real-model smoke")

    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    pytest.importorskip("jang_tools.loader")

    source = _optional_path_from_env(case.source_env)
    imatrix = _optional_path_from_env(case.imatrix_env)
    out = Path(os.environ.get(case.out_env, str(tmp_path / case.default_out_name))).expanduser()
    target_size_gb = float(os.environ.get(case.target_env, case.default_target_gb))
    live_kv = os.environ.get(case.kv_env, "mlx_affine_q8")
    max_tokens = int(os.environ.get(case.max_tokens_env, "512"))

    if source is None:
        pytest.skip(f"set {case.source_env}=<snapshot path> to run the real-model smoke")
    if imatrix is None:
        pytest.skip(f"set {case.imatrix_env}=<imatrix path> to run the real-model smoke")
    if not source.exists():
        pytest.skip(f"source model not found: {source}")
    if not imatrix.exists():
        pytest.skip(f"imatrix not found: {imatrix}")

    from moespresso.core.artifact import read_artifact
    from moespresso.package.convert import convert
    from moespresso.package.constants import MANIFEST_NAME
    from moespresso.runtime.http import render_prompt, rendering_identity
    from moespresso.runtime.kv_policy import parse_kv_policy
    from moespresso.runtime.prefix_cache import (
        PrefixCacheGenerator,
        encode_rendered_prompt,
        make_prompt_cache_store,
    )
    from moespresso.runtime.serve import load_served_model
    from moespresso.runtime.verify import verify_package

    manifest_path = out / MANIFEST_NAME
    if manifest_path.exists() and os.environ.get(case.reuse_env) == "1":
        manifest = read_artifact(manifest_path)
    else:
        manifest = convert(
            source,
            out,
            imatrix_path=imatrix,
            target_size_gb=target_size_gb,
            shard_size_gb=2.0,
            verbose=True,
        )

    blocking = [v for v in verify_package(manifest, out) if v.blocking]
    assert not blocking

    model, tokenizer, manifest = load_served_model(out, manifest=manifest)
    kv_policy = parse_kv_policy({"live_kv_format": live_kv})
    effective_rendering_id = rendering_identity(
        manifest.get("tokenizer", {}).get("rendering_id"),
        None,
    )
    generator = PrefixCacheGenerator(
        model,
        tokenizer,
        manifest,
        make_prompt_cache_store(max_size=4),
    )

    prompt1 = render_prompt(
        [{"role": "user", "content": "What is 2+2? Answer with only the number."}],
        tokenizer,
    )
    first = generator(
        prompt1,
        kv_policy=kv_policy,
        effective_rendering_id=effective_rendering_id,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    full_history = [
        {"role": "user", "content": "What is 2+2? Answer with only the number."},
        {"role": "assistant", "content": _assistant_history_content(first.text)},
        {"role": "user", "content": "Now add 3 to that result. Answer with only the number."},
    ]
    prompt2 = render_prompt(full_history, tokenizer)
    prompt1_tokens = encode_rendered_prompt(tokenizer, prompt1)
    prompt2_tokens = encode_rendered_prompt(tokenizer, prompt2)
    assert prompt2_tokens[:len(prompt1_tokens)] == prompt1_tokens, (
        "full-history render stopped being append-only; prefix cache would not be "
        "testing a real chat-turn prefix"
    )
    first_cache_key = prompt1_tokens + list(first.generated_token_ids)
    first_cache_key_roundtrip_tokens = _common_prefix_len(prompt2_tokens, first_cache_key)
    second = generator(
        prompt2,
        kv_policy=kv_policy,
        effective_rendering_id=effective_rendering_id,
        max_tokens=max_tokens,
        temperature=0.0,
    )

    assert first.text.strip()
    assert second.text.strip()
    assert first.finish_reason == "stop"
    assert second.finish_reason == "stop"

    evidence = {
        "kind": case.evidence_kind,
        "label": case.label,
        "source": str(source),
        "imatrix": str(imatrix),
        "package": str(out),
        "manifest_id": manifest.get("artifact_id"),
        "family": manifest.get("architecture", {}).get("family"),
        "target_size_gb": target_size_gb,
        "live_kv_format": kv_policy.live_kv_format,
        "kv_group_size": kv_policy.kv_group_size,
        "quantized_kv_start": kv_policy.quantized_kv_start,
        "max_tokens": max_tokens,
        "prompt1_tokens": len(prompt1_tokens),
        "prompt2_tokens": len(prompt2_tokens),
        "first_generated_tokens": len(first.generated_token_ids),
        "first_cache_key_tokens": len(first_cache_key),
        "first_cache_key_roundtrip_tokens": first_cache_key_roundtrip_tokens,
        "verification_blocking": len(blocking),
        "first_cache_event": getattr(first, "cache_event", None),
        "second_cache_event": getattr(second, "cache_event", None),
        "second_cached_tokens": second.cached_tokens,
        "first_text": first.text,
        "second_text": second.text,
        "platform": platform.platform(),
    }
    evidence_path = out / case.evidence_file
    evidence_path.write_text(json.dumps(evidence, indent=2))


@pytest.mark.real_model
def test_dense_qwen_08b_calibrated_real_model_smoke(tmp_path):
    _run_dense_qwen_calibrated_real_model_smoke(QWEN_08B, tmp_path)


@pytest.mark.real_model
def test_dense_qwen_4b_calibrated_real_model_smoke(tmp_path):
    _run_dense_qwen_calibrated_real_model_smoke(QWEN_4B, tmp_path)
