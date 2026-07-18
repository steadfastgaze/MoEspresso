"""L2 micro-golden evidence for MoEspresso-owned correctness primitives.

Tiny, deterministic, no model, no package serve. These checks pin the local format and
transform conventions that L1 relies on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from moespresso.core.artifact import Validation, make_artifact
from moespresso.correctness.tq_reference import (
    generate_random_signs,
    hadamard_inverse,
    hadamard_rotate,
    unpack_tq_indices,
)
from moespresso.probe.weight_io import split_fused_gate_up
from moespresso.runtime.deepseek_v4.renderer import (
    ASSISTANT_SP_TOKEN,
    BOS_TOKEN,
    DSML_TOKEN,
    EOS_TOKEN,
    THINKING_END_TOKEN,
    THINKING_START_TOKEN,
    USER_SP_TOKEN,
    render_deepseek_v4_prompt,
)

PRODUCER = {"tool": "moespresso.correctness", "version": "1.0.0"}
DS4_Q0_FIXTURE_ROOT = (
    Path(__file__).resolve().parent / "fixtures" / "deepseek_v4" / "test_vectors"
)

_DS4_SPECIAL_TOKEN_IDS = {
    BOS_TOKEN: 0,
    EOS_TOKEN: 1,
    USER_SP_TOKEN: 128803,
    ASSISTANT_SP_TOKEN: 128804,
    THINKING_START_TOKEN: 128821,
    THINKING_END_TOKEN: 128822,
    DSML_TOKEN: 128825,
}


def _case(name: str, passed: bool, details: dict | None = None) -> dict:
    return {"case": name, "passed": bool(passed), **(details or {})}


def _packed_word(values: list[int], bits: int) -> int:
    word = 0
    for i, value in enumerate(values):
        word |= int(value) << (i * bits)
    return word


def _validation(name: str, message: str) -> Validation:
    return Validation("error", "correctness.golden_failed", message,
                      path=f"/{name}", phase="L2", blocking=True)


def _check_tq_unpack() -> tuple[list[dict], list[Validation]]:
    metrics, out = [], []
    cases = {
        1: [0, 1, 1, 0],
        2: [0, 1, 2, 3],
        4: [0, 5, 10, 15],
    }
    for bits, values in cases.items():
        packed = np.array([[_packed_word(values, bits)]], dtype=np.uint32)
        got = unpack_tq_indices(packed, bits=bits, in_features=len(values))[0].tolist()
        ok = got == values
        metrics.append(_case(f"tq_unpack_{bits}bit", ok, {"actual": got, "expected": values}))
        if not ok:
            out.append(_validation(f"tq_unpack_{bits}bit",
                                   f"TQ {bits}-bit unpack returned {got}, expected {values}"))
    return metrics, out


def _check_hadamard_inverse() -> tuple[list[dict], list[Validation]]:
    x = np.array([[1.0, -2.0, 3.0, -4.0]], dtype=np.float32)
    signs = generate_random_signs(4, seed=17)
    back = hadamard_inverse(hadamard_rotate(x, signs), signs)
    err = float(np.max(np.abs(back - x)))
    ok = err < 1e-6
    metrics = [_case("tq_hadamard_inverse", ok, {"max_abs": err})]
    out = [] if ok else [_validation("tq_hadamard_inverse",
                                     f"Hadamard inverse error {err} >= 1e-6")]
    return metrics, out


def _check_gate_up_split() -> tuple[list[dict], list[Validation]]:
    e0 = np.array([[1, 1], [2, 2], [101, 101], [102, 102]], dtype=np.float32)
    e1 = np.array([[3, 3], [4, 4], [103, 103], [104, 104]], dtype=np.float32)
    sample = np.concatenate([e0, e1], axis=0)
    gate, up = split_fused_gate_up(sample, n_sampled=2)
    expected_gate = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float32)
    expected_up = np.array([[101, 101], [102, 102], [103, 103], [104, 104]], dtype=np.float32)
    ok = np.array_equal(gate, expected_gate) and np.array_equal(up, expected_up)
    metrics = [_case("fused_gate_up_split", ok)]
    out = [] if ok else [_validation("fused_gate_up_split",
                                     "fused gate_up did not split into expected fixed halves")]
    return metrics, out


def _check_conv1d_trigger() -> tuple[list[dict], list[Validation]]:
    good = (8, 1, 4)
    bad = (8, 4, 1)
    good_triggers = good[-1] != 1
    bad_triggers = bad[-1] != 1
    ok = good_triggers and not bad_triggers
    metrics = [_case("conv1d_norm_shift_trigger", ok,
                     {"good_shape": list(good), "bad_shape": list(bad)})]
    out = [] if ok else [_validation("conv1d_norm_shift_trigger",
                                     "conv1d trigger golden no longer distinguishes shapes")]
    return metrics, out


def _check_affine_sidecar_shapes() -> tuple[list[dict], list[Validation]]:
    rows, cols, bits, group_size = 5, 32, 4, 32
    packed_cols = (cols * bits + 31) // 32
    groups = cols // group_size
    expected = {
        "weight": [rows, packed_cols],
        "scales": [rows, groups],
        "biases": [rows, groups],
    }
    ok = expected == {"weight": [5, 4], "scales": [5, 1], "biases": [5, 1]}
    metrics = [_case("affine_sidecar_shapes", ok, {"expected": expected})]
    out = [] if ok else [_validation("affine_sidecar_shapes",
                                     f"affine sidecar shape golden changed: {expected}")]
    return metrics, out


def l2_micro_goldens(subject: dict | None = None) -> dict:
    """Run the tiny L2 micro-golden suite and return correctness_evidence."""
    metrics: list[dict] = []
    validation: list[Validation] = []
    for fn in (_check_tq_unpack, _check_hadamard_inverse,
               _check_gate_up_split, _check_conv1d_trigger, _check_affine_sidecar_shapes):
        m, v = fn()
        metrics.extend(m)
        validation.extend(v)
    blocking = any(v.blocking for v in validation)
    return make_artifact(
        "correctness_evidence",
        subject or {"source_root": "micro_goldens", "source_format": "synthetic"},
        PRODUCER,
        status="invalid" if blocking else "valid",
        validation=validation,
        rung="L2",
        reference_provenance=[
            {"component": "tq_reference", "kind": "independent",
             "identity": "moespresso.correctness.tq_reference", "shared_with": []},
            {"component": "fused_gate_up", "kind": "independent",
             "identity": "fixed tiny source arrays", "shared_with": []},
            {"component": "conv1d_norm", "kind": "independent",
             "identity": "declared shape predicate", "shared_with": []},
            {"component": "affine_sidecars", "kind": "independent",
             "identity": "declared MLX affine sidecar shape rule", "shared_with": []},
        ],
        metrics=metrics,
        summary={"cases": len(metrics), "failed": sum(1 for m in metrics if not m["passed"])},
    )


def _ds4_q0_fixture_root(fixture_root: str | Path | None = None) -> Path:
    return Path(fixture_root) if fixture_root is not None else DS4_Q0_FIXTURE_ROOT


def _parse_ds4_official_vec(path: Path) -> list[dict]:
    cases: list[dict] = []
    current: dict | None = None
    current_step: dict | None = None
    for lineno, raw in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "case" and len(parts) == 5:
            if current is not None:
                raise ValueError(f"{path}:{lineno}: nested case before end")
            current = {
                "id": parts[1],
                "ctx": int(parts[2]),
                "steps_expected": int(parts[3]),
                "prompt_file": parts[4],
                "steps": [],
            }
            current_step = None
            continue
        if parts[0] == "step" and len(parts) == 4 and current is not None:
            current_step = {
                "index": int(parts[1]),
                "selected_hex": parts[2],
                "top_expected": int(parts[3]),
                "top": [],
            }
            current["steps"].append(current_step)
            continue
        if parts[0] == "top" and len(parts) == 3 and current_step is not None:
            current_step["top"].append({"token_hex": parts[1], "logprob": float(parts[2])})
            continue
        if parts[0] == "end" and len(parts) == 1 and current is not None:
            if len(current["steps"]) != current["steps_expected"]:
                raise ValueError(
                    f"{path}:{lineno}: case {current['id']} has {len(current['steps'])} "
                    f"steps, expected {current['steps_expected']}"
                )
            for step in current["steps"]:
                if len(step["top"]) != step["top_expected"]:
                    raise ValueError(
                        f"{path}:{lineno}: case {current['id']} step {step['index']} "
                        f"has {len(step['top'])} top tokens, expected {step['top_expected']}"
                    )
            cases.append(current)
            current = None
            current_step = None
            continue
        raise ValueError(f"{path}:{lineno}: unparseable line {line!r}")
    if current is not None:
        raise ValueError(f"{path}: unterminated case {current['id']}")
    return cases


def _official_vec_prompt_path(fixture_root: Path, prompt_file: str) -> Path:
    prefix = "tests/test-vectors/"
    if prompt_file.startswith(prefix):
        prompt_file = prompt_file[len(prefix):]
    return fixture_root / prompt_file


def _check_ds4_renderer_q0() -> tuple[list[dict], list[Validation]]:
    metrics: list[dict] = []
    out: list[Validation] = []

    expected_tokens = {
        "bos": "<｜begin▁of▁sentence｜>",
        "eos": "<｜end▁of▁sentence｜>",
        "user": "<｜User｜>",
        "assistant": "<｜Assistant｜>",
        "think_start": "<think>",
        "think_end": "</think>",
        "dsml": "｜DSML｜",
    }
    actual_tokens = {
        "bos": BOS_TOKEN,
        "eos": EOS_TOKEN,
        "user": USER_SP_TOKEN,
        "assistant": ASSISTANT_SP_TOKEN,
        "think_start": THINKING_START_TOKEN,
        "think_end": THINKING_END_TOKEN,
        "dsml": DSML_TOKEN,
    }
    ok = actual_tokens == expected_tokens
    metrics.append(_case("ds4_control_token_strings", ok, {"actual": actual_tokens}))
    if not ok:
        out.append(_validation("ds4_control_token_strings", "DS4 control-token strings changed"))

    fullwidth_tokens = {
        "bos": BOS_TOKEN,
        "eos": EOS_TOKEN,
        "user": USER_SP_TOKEN,
        "assistant": ASSISTANT_SP_TOKEN,
        "dsml": DSML_TOKEN,
    }
    bad_ascii = {name: value for name, value in fullwidth_tokens.items() if "|" in value}
    ok = not bad_ascii and all("｜" in value for value in fullwidth_tokens.values())
    metrics.append(_case("ds4_fullwidth_bar_tokens", ok, {"bad_ascii": bad_ascii}))
    if not ok:
        out.append(
            _validation(
                "ds4_fullwidth_bar_tokens",
                f"DS4 tokens must use U+FF5C fullwidth bars, not ASCII pipe: {bad_ascii}",
            )
        )

    messages = [{"role": "user", "content": "Hello"}]
    rendered_thinking = render_deepseek_v4_prompt(
        messages, template_kwargs={"enable_thinking": True})
    rendered_chat = render_deepseek_v4_prompt(
        messages, template_kwargs={"enable_thinking": False})
    expected_thinking = f"{BOS_TOKEN}{USER_SP_TOKEN}Hello{ASSISTANT_SP_TOKEN}{THINKING_START_TOKEN}"
    expected_chat = f"{BOS_TOKEN}{USER_SP_TOKEN}Hello{ASSISTANT_SP_TOKEN}{THINKING_END_TOKEN}"
    ok = rendered_thinking == expected_thinking and rendered_chat == expected_chat
    metrics.append(
        _case(
            "ds4_assistant_prefix_rule",
            ok,
            {"thinking": rendered_thinking, "chat": rendered_chat},
        )
    )
    if not ok:
        out.append(
            _validation(
                "ds4_assistant_prefix_rule",
                "assistant prefix must emit <think> in thinking mode and </think> otherwise",
            )
        )

    rendered_tool = render_deepseek_v4_prompt(
        [{"role": "user", "content": "Need"}],
        template_kwargs={"enable_thinking": False},
        tools=[{
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "look",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            },
        }],
    )
    ok = f"<{DSML_TOKEN}tool_calls>" in rendered_tool and "<|DSML|tool_calls>" not in rendered_tool
    metrics.append(_case("ds4_dsml_fullwidth_render", ok))
    if not ok:
        out.append(
            _validation(
                "ds4_dsml_fullwidth_render",
                "DSML rendering must use fullwidth-bar ｜DSML｜ tags",
            )
        )
    return metrics, out


def _check_ds4_tokenizer_q0(
    package_dir: Path,
    fixture_root: Path,
) -> tuple[list[dict], list[Validation]]:
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(package_dir)
    metrics: list[dict] = []
    out: list[Validation] = []

    for token, expected_id in _DS4_SPECIAL_TOKEN_IDS.items():
        ids = list(tokenizer.encode(token, add_special_tokens=False))
        ok = ids == [expected_id] and tokenizer.decode(ids) == token
        metrics.append(
            _case(
                f"ds4_special_token_id_{expected_id}",
                ok,
                {"token": token, "actual_ids": ids, "expected_ids": [expected_id]},
            )
        )
        if not ok:
            out.append(
                _validation(
                    f"ds4_special_token_id_{expected_id}",
                    f"tokenizer maps {token!r} to {ids}, expected {[expected_id]}",
                )
            )

    vector_path = fixture_root / "official.vec"
    cases = _parse_ds4_official_vec(vector_path)
    token_checks = 0
    prompt_checks = 0
    for case in cases:
        prompt_path = _official_vec_prompt_path(fixture_root, case["prompt_file"])
        prompt = prompt_path.read_text(encoding="utf-8")
        rendered = render_deepseek_v4_prompt(
            [{"role": "user", "content": prompt}],
            template_kwargs={"enable_thinking": False},
        )
        ids = list(tokenizer.encode(rendered, add_special_tokens=False))
        ok = (
            ids[:1] == [_DS4_SPECIAL_TOKEN_IDS[BOS_TOKEN]]
            and _DS4_SPECIAL_TOKEN_IDS[USER_SP_TOKEN] in ids
            and _DS4_SPECIAL_TOKEN_IDS[ASSISTANT_SP_TOKEN] in ids
            and ids[-1:] == [_DS4_SPECIAL_TOKEN_IDS[THINKING_END_TOKEN]]
        )
        prompt_checks += 1
        metrics.append(
            _case(
                f"ds4_official_vec_prompt_tokens_{case['id']}",
                ok,
                {"prompt_tokens": len(ids), "ctx": case["ctx"]},
            )
        )
        if not ok:
            out.append(
                _validation(
                    f"ds4_official_vec_prompt_tokens_{case['id']}",
                    f"rendered prompt for {case['id']} does not tokenize with DS4 chat sentinels",
                )
            )

        seen_hex: set[str] = set()
        for step in case["steps"]:
            hex_values = [step["selected_hex"]] + [top["token_hex"] for top in step["top"]]
            for token_hex in hex_values:
                if token_hex in seen_hex:
                    continue
                seen_hex.add(token_hex)
                token_bytes = bytes.fromhex(token_hex)
                token_text = token_bytes.decode("utf-8")
                token_ids = list(tokenizer.encode(token_text, add_special_tokens=False))
                decoded = tokenizer.decode(token_ids)
                ok = len(token_ids) == 1 and decoded.encode("utf-8") == token_bytes
                token_checks += 1
                metrics.append(
                    _case(
                        f"ds4_official_vec_token_{case['id']}_{step['index']}_{token_hex}",
                        ok,
                        {"text": token_text, "ids": token_ids},
                    )
                )
                if not ok:
                    out.append(
                        _validation(
                            f"ds4_official_vec_token_{case['id']}_{step['index']}_{token_hex}",
                            f"official.vec token bytes {token_hex} did not round-trip as one token",
                        )
                    )
    metrics.append(
        _case(
            "ds4_official_vec_fixture_coverage",
            True,
            {"cases": len(cases), "prompts": prompt_checks, "tokens": token_checks},
        )
    )
    return metrics, out


def q0_deepseek_v4_renderer_tokenizer_goldens(
    package_dir: str | Path,
    *,
    fixture_root: str | Path | None = None,
    subject: dict | None = None,
) -> dict:
    """Q0 DS4 renderer/tokenizer gate.

    Manual-only: no model forward, but it loads the real package tokenizer. The
    official-vector fixture is committed under ``correctness/fixtures`` so the
    gate is reproducible without the sibling ds4 checkout.
    """
    package_dir = Path(package_dir)
    fixture_root = _ds4_q0_fixture_root(fixture_root)
    metrics: list[dict] = []
    validation: list[Validation] = []
    for fn in (_check_ds4_renderer_q0,):
        m, v = fn()
        metrics.extend(m)
        validation.extend(v)
    m, v = _check_ds4_tokenizer_q0(package_dir, fixture_root)
    metrics.extend(m)
    validation.extend(v)
    blocking = any(v.blocking for v in validation)
    return make_artifact(
        "correctness_evidence",
        subject or {
            "source_root": str(package_dir),
            "source_format": "moespresso_package",
        },
        PRODUCER,
        status="invalid" if blocking else "valid",
        validation=validation,
        rung="Q0",
        reference_provenance=[
            {
                "component": "deepseek_v4_renderer",
                "kind": "external_reference",
                "identity": "ds4 chat encoder control-token contract",
                "shared_with": ["runtime.deepseek_v4_renderer"],
            },
            {
                "component": "official_vec",
                "kind": "captured_fixture",
                "identity": str(fixture_root / "official.vec"),
                "shared_with": [],
            },
            {
                "component": "tokenizer",
                "kind": "package_contract",
                "identity": str(package_dir / "tokenizer.json"),
                "shared_with": ["runtime package tokenizer"],
            },
        ],
        metrics=metrics,
        summary={
            "cases": len(metrics),
            "failed": sum(1 for metric in metrics if not metric["passed"]),
            "fixture_root": str(fixture_root),
        },
    )
