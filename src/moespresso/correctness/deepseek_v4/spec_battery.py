"""Speculative-decoding measurement battery for DeepSeek-V4.

Generalizes the single-prompt A/B replay (`spec_replay`) into a fixed
battery: one process loads the target package once, loads the available
drafter sidecars (DSpark, MTP, and DFlash), and runs every evaluation
prompt through a plain greedy arm plus one speculative arm per drafter. The
report is machine-readable JSON with per-prompt and per-arm throughput,
acceptance statistics, divergence against the plain stream, per-repeat
aggregates, and an environment snapshot. `--repeats N` runs the whole
battery N times in the same process and reports the per-repeat aggregate
series side by side, which is the A/B/A reading.

Tap lifecycle. `install_hidden_tap` stores a tap in a single per-layer
attribute, and a tap's `take_rows` requires a recorded row from every
one of its layers. Two drafters' taps therefore do not compose: the
DSpark tap covers three trunk layers with the hyper-connection mean
transform while the MTP tap covers the last trunk layer with the
identity transform, and installing the second tap replaces the first
one's registration on the shared layer. The battery installs a fresh tap
immediately before each speculative arm and removes it afterwards
(`uninstall_hidden_tap`), so exactly one tap is registered at any time
and every recorded row uses the active drafter's transform.

Throughput convention. Both arms share the same prefill schedule.
`tok_per_s` divides generated tokens by the whole arm wall time
including prefill, and `ratio_vs_plain` divides the two arms'
`tok_per_s`, so the shared prefill cost appears on both sides. The
plain arm additionally reports `decode_tok_per_s` over the decode phase
only, the figure used by the speed logs.

Divergence between a speculative stream and the plain stream is expected
at knife-edge positions (the multi-token verify forward and the
single-token decode forward are different numeric lattices) and is
reported, not treated as a failure; the exit code does not depend on it.

Launch protocol: a run loads the full target package plus sidecars and
must not overlap another model-loading process on the host. The operator
serializes launches (check the GPU_IN_USE.txt marker in the session
scratchpad and `pgrep` for competing loaders) before starting the
battery; the tool itself performs no cross-process coordination.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import mlx.core as mx

from moespresso.correctness.deepseek_v4.spec_replay import (
    _prefill_step_size,
    build_drafter,
    plain_greedy_generate,
)
from moespresso.runtime.deepseek_v4.spec_decode import (
    SpecStats,
    install_hidden_tap,
    spec_generate,
)
from moespresso.runtime.serve import load_served_model

PLAIN_ARM = "plain"

EVALUATION_CATEGORIES = (
    "agentic_tool_use",
    "code_generation",
    "code_explanation",
    "open_chat",
    "math_word_problem",
    "structured_extraction",
    "long_form_writing",
    "translation",
    "terse_qa",
    "planning",
)

# Evaluation prompt set for the measurement battery. These prompts are
# measurement instruments only and must never be used for tuning: no
# scheduler constant, confidence threshold, calibration prior, or kernel
# routing decision may be fitted against results measured on them. A
# battery run on prompts that shaped the tuning stops being evidence.
# Extend the set to cover a missing content class; do not edit existing
# prompts to move the numbers.
EVALUATION_PROMPTS: tuple = (
    {
        "id": "agent_next_tool_call",
        "category": "agentic_tool_use",
        "text": (
            "You are an agent with these tools: read_file(path), "
            "search(pattern, dir), run_tests(target), write_file(path, "
            "content). The user reports: \"make test fails in "
            "test_config.py with KeyError: 'cache_dir'\". Decide the next "
            "tool call. Reply with one short reasoning line, then the tool "
            "call as JSON on its own line, then stop."
        ),
    },
    {
        "id": "codegen_merge_intervals",
        "category": "code_generation",
        "text": (
            "Write a Python function merge_intervals(intervals) that takes "
            "a list of (start, end) tuples and returns the merged, sorted "
            "list of non-overlapping intervals. Include type hints, a "
            "docstring with one example, and handle the empty-list case. "
            "Then add three pytest test cases covering overlapping, "
            "touching, and disjoint intervals."
        ),
    },
    {
        "id": "explain_kadane",
        "category": "code_explanation",
        "text": (
            "Explain what the following Python function does, step by "
            "step, and name the algorithm it implements:\n"
            "\n"
            "def f(xs):\n"
            "    best = cur = xs[0]\n"
            "    for x in xs[1:]:\n"
            "        cur = max(x, cur + x)\n"
            "        best = max(best, cur)\n"
            "    return best\n"
            "\n"
            "Then state its time and space complexity and give one input "
            "where the answer is negative."
        ),
    },
    {
        "id": "chat_piano_adult",
        "category": "open_chat",
        "text": (
            "I've been thinking about learning to play the piano as an "
            "adult. I have about 30 minutes a day and no prior music "
            "experience. Is it realistic, and what would a sensible first "
            "month look like? Be honest about the frustrating parts."
        ),
    },
    {
        "id": "math_bakery_boxes",
        "category": "math_word_problem",
        "text": (
            "A bakery sells croissants in boxes of 4 and muffins in boxes "
            "of 6. On Monday it sold 23 boxes in total, containing 132 "
            "pastries. How many boxes of each did it sell? Show your "
            "working step by step, check the answer, and state it in one "
            "final sentence."
        ),
    },
    {
        "id": "extract_loan_entities",
        "category": "structured_extraction",
        "text": (
            "Extract every person, organization, date, and monetary amount "
            "from the text below into a JSON object with keys \"people\", "
            "\"organizations\", \"dates\", \"amounts\", each a list of "
            "strings. Output only the JSON.\n"
            "\n"
            "Text: \"On 14 March 2024, Elena Fischer of Nordbank AG "
            "approved a 2.4 million euro bridge loan to Kastellan "
            "Logistics, countersigned by CFO Marco Ruiz two days later. "
            "The board, chaired by Dr. Ingrid Holm, will review the terms "
            "on 1 July 2024.\""
        ),
    },
    {
        "id": "essay_urban_highways",
        "category": "long_form_writing",
        "text": (
            "Write a 400-word essay on why cities that removed urban "
            "highways often saw traffic decrease rather than increase. Use "
            "a clear introduction, two body paragraphs with concrete "
            "examples, and a conclusion. Avoid bullet points; write "
            "flowing prose."
        ),
    },
    {
        "id": "translate_archive_letter",
        "category": "translation",
        "text": (
            "Translate the following paragraph into French, keeping the "
            "register formal:\n"
            "\n"
            "\"Thank you for your inquiry regarding our archival services. "
            "Our reading room is open Tuesday through Saturday, and "
            "reproductions of fragile documents can be requested at the "
            "front desk. Please note that some collections require written "
            "permission from the donor before consultation.\"\n"
            "\n"
            "After the translation, list two translation choices you made "
            "and why."
        ),
    },
    {
        "id": "qa_five_facts",
        "category": "terse_qa",
        "text": (
            "Answer each question in one short sentence, numbered:\n"
            "1. What is the capital of Australia?\n"
            "2. Who wrote \"The Master and Margarita\"?\n"
            "3. What does DNS stand for?\n"
            "4. In what year did the Berlin Wall fall?\n"
            "5. What is the boiling point of water at sea level in "
            "Fahrenheit?"
        ),
    },
    {
        "id": "plan_postgres_migration",
        "category": "planning",
        "text": (
            "Plan the migration of a small team's on-premise PostgreSQL 12 "
            "database (about 200 GB, one primary, nightly dumps) to a "
            "managed cloud service with under 30 minutes of downtime. "
            "Produce a phased plan with numbered steps, the risks at each "
            "phase, and a rollback point per phase."
        ),
    },
)

_WARMUP_PROMPT = "Reply with a short greeting."
_WARMUP_TOKENS = 8


def validate_prompt_set(prompts: Sequence[dict]) -> None:
    """Fail closed on a malformed prompt set.

    Every entry needs a non-empty string `id`, `category`, and `text`,
    ids are unique, categories come from `EVALUATION_CATEGORIES`, and
    every category is covered so a deleted prompt cannot silently narrow
    the battery.
    """
    if not prompts:
        raise ValueError("empty prompt set")
    seen: set = set()
    for entry in prompts:
        if not isinstance(entry, dict):
            raise ValueError(f"prompt entry is not a dict: {entry!r}")
        for key in ("id", "category", "text"):
            value = entry.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"prompt entry {entry.get('id')!r}: missing or empty {key!r}"
                )
        if entry["category"] not in EVALUATION_CATEGORIES:
            raise ValueError(
                f"prompt {entry['id']!r}: unknown category {entry['category']!r}"
            )
        if entry["id"] in seen:
            raise ValueError(f"duplicate prompt id {entry['id']!r}")
        seen.add(entry["id"])
    covered = {entry["category"] for entry in prompts}
    missing = [c for c in EVALUATION_CATEGORIES if c not in covered]
    if missing:
        raise ValueError(f"prompt set does not cover categories: {missing}")


def uninstall_hidden_tap(model, tap) -> None:
    """Remove `tap`'s per-layer registrations and deactivate it.

    `install_hidden_tap` stores the tap in a single per-layer attribute,
    so a second drafter's tap replaces the first one's registration on a
    shared layer and the first tap's `take_rows` then fails on the
    missing layer. The battery therefore brackets each speculative arm
    with a fresh install and this removal. Only this tap's own
    registrations are cleared; a layer already claimed by another tap is
    left alone. The idempotent class-level call wrapper stays in place
    and costs one attribute check per layer call.
    """
    tap.active = False
    tap.rows = {}
    layers = model.model.layers
    for i in tap.layer_ids:
        if getattr(layers[i], "_moespresso_hidden_tap", None) is tap:
            layers[i]._moespresso_hidden_tap = None


def parse_mactop_snapshot(text: str) -> dict:
    """Parse one `mactop --headless --count 1` sample.

    The output is a JSON list with one entry; `gpu_temp` lives under
    `soc_metrics` and `thermal_state` at the top level. Raises
    `ValueError` on any other shape.
    """
    entries = json.loads(text)
    if not isinstance(entries, list) or not entries:
        raise ValueError("expected a non-empty JSON list")
    entry = entries[0]
    if not isinstance(entry, dict):
        raise ValueError("expected a JSON object entry")
    soc = entry.get("soc_metrics") or {}
    gpu_temp = soc.get("gpu_temp", entry.get("gpu_temp"))
    thermal_state = entry.get("thermal_state")
    if gpu_temp is None or thermal_state is None:
        raise ValueError("sample lacks gpu_temp or thermal_state")
    return {
        "available": True,
        "gpu_temp_c": float(gpu_temp),
        "thermal_state": str(thermal_state),
    }


def thermal_snapshot(timeout_seconds: float = 20.0) -> dict:
    """One-sample thermal snapshot via mactop, degrading gracefully.

    Returns `{"available": False, "reason": ...}` when mactop is not
    installed, times out, exits nonzero, or emits unparseable output.
    """
    exe = shutil.which("mactop")
    if exe is None:
        return {"available": False, "reason": "mactop not on PATH"}
    try:
        proc = subprocess.run(
            [exe, "--headless", "--count", "1"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": f"mactop failed: {exc}"}
    if proc.returncode != 0:
        return {"available": False, "reason": f"mactop exit {proc.returncode}"}
    try:
        return parse_mactop_snapshot(proc.stdout)
    except (ValueError, TypeError) as exc:
        return {"available": False, "reason": f"unparseable mactop output: {exc}"}


def divergence_stats(
    spec_tokens: Sequence[int], plain_tokens: Sequence[int]
) -> tuple[Optional[int], int]:
    """First index where the two token streams differ, and the length of
    the identical prefix.

    When one stream is a strict prefix of the other, the divergence index
    is the first position that lacks a counterpart. Equal streams return
    `(None, length)`.
    """
    n = min(len(spec_tokens), len(plain_tokens))
    for i in range(n):
        if spec_tokens[i] != plain_tokens[i]:
            return i, i
    if len(spec_tokens) != len(plain_tokens):
        return n, n
    return None, n


def plain_arm_record(
    tokens: Sequence[int],
    prefill_seconds: float,
    decode_seconds: float,
    text: str,
) -> dict:
    """Assemble the plain greedy arm's report entry."""
    n = len(tokens)
    total = prefill_seconds + decode_seconds
    return {
        "tokens_generated": n,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "total_seconds": total,
        "decode_tok_per_s": n / max(decode_seconds, 1e-9),
        "tok_per_s": n / max(total, 1e-9),
        "text": text,
    }


def spec_arm_record(
    tokens: Sequence[int],
    stats: SpecStats,
    total_seconds: float,
    text: str,
    plain_record: Optional[dict] = None,
    plain_tokens: Optional[Sequence[int]] = None,
) -> dict:
    """Assemble one speculative arm's report entry.

    `plain_record` and `plain_tokens` come from the same prompt's plain
    arm; without them the ratio and divergence fields are None.
    """
    n = len(tokens)
    tok_per_s = n / max(total_seconds, 1e-9)
    ratio = None
    if plain_record is not None and plain_record.get("tok_per_s"):
        ratio = tok_per_s / plain_record["tok_per_s"]
    if plain_tokens is not None:
        first_divergence, prefix_len = divergence_stats(tokens, plain_tokens)
        tokens_equal = first_divergence is None
    else:
        first_divergence, prefix_len, tokens_equal = None, None, None
    return {
        "tokens_generated": n,
        "total_seconds": total_seconds,
        "tok_per_s": tok_per_s,
        "ratio_vs_plain": ratio,
        "rounds": stats.rounds,
        "proposed": stats.proposed,
        "accepted": stats.accepted,
        "tau": stats.mean_accepted_length,
        "plain_fallbacks": stats.plain_fallbacks,
        "per_position_offered": list(stats.per_position_offered),
        "per_position_accepted": list(stats.per_position_accept),
        # Prefix-independent per-position agreement with the target
        # argmax: the curve comparable to a checkpoint's per-position
        # validation accuracy. `per_position_accepted` is the cumulative
        # prefix-survival count and decays multiplicatively at depth.
        "per_position_matched": list(stats.per_position_matched),
        "submit_length_counts": {
            str(k): v for k, v in sorted(stats.submit_length_counts.items())
        },
        "first_divergence_vs_plain": first_divergence,
        "identical_prefix_length": prefix_len,
        "tokens_equal_plain": tokens_equal,
        "text": text,
    }


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(statistics.fmean(values)) if values else None


def _median(values: Sequence[float]) -> Optional[float]:
    return float(statistics.median(values)) if values else None


def repeat_aggregates(prompt_results: Dict[str, dict]) -> dict:
    """Per-drafter aggregates over one repeat's prompt results.

    `prompt_results` maps prompt id to `{"category": ..., "arms": ...}`.
    Every arm other than the plain one counts as a drafter arm. Ratios
    aggregate as mean and median across prompts, with a per-category
    breakdown.
    """
    per_drafter: Dict[str, dict] = {}
    for result in prompt_results.values():
        category = result["category"]
        for arm_name, arm in result["arms"].items():
            if arm_name == PLAIN_ARM:
                continue
            slot = per_drafter.setdefault(
                arm_name, {"count": 0, "ratios": [], "taus": [], "by_cat": {}}
            )
            slot["count"] += 1
            cat = slot["by_cat"].setdefault(
                category, {"count": 0, "ratios": [], "taus": []}
            )
            cat["count"] += 1
            ratio = arm.get("ratio_vs_plain")
            if ratio is not None:
                slot["ratios"].append(ratio)
                cat["ratios"].append(ratio)
            tau = arm.get("tau")
            if tau is not None:
                slot["taus"].append(tau)
                cat["taus"].append(tau)
    out: Dict[str, dict] = {}
    for name, slot in sorted(per_drafter.items()):
        out[name] = {
            "prompts": slot["count"],
            "mean_ratio": _mean(slot["ratios"]),
            "median_ratio": _median(slot["ratios"]),
            "mean_tau": _mean(slot["taus"]),
            "per_category": {
                cat: {
                    "prompts": c["count"],
                    "mean_ratio": _mean(c["ratios"]),
                    "mean_tau": _mean(c["taus"]),
                }
                for cat, c in sorted(slot["by_cat"].items())
            },
        }
    return out


def aggregates_by_repeat(repeat_entries: Sequence[dict]) -> dict:
    """Side-by-side per-repeat aggregate series per drafter.

    Each series is aligned by repeat index (missing values become None),
    so an A/B/A run reads across one line per metric.
    """
    names = sorted(
        {name for entry in repeat_entries for name in (entry.get("aggregates") or {})}
    )
    series: Dict[str, dict] = {}
    for name in names:
        slot: Dict[str, list] = {"mean_ratio": [], "median_ratio": [], "mean_tau": []}
        for entry in repeat_entries:
            agg = (entry.get("aggregates") or {}).get(name) or {}
            slot["mean_ratio"].append(agg.get("mean_ratio"))
            slot["median_ratio"].append(agg.get("median_ratio"))
            slot["mean_tau"].append(agg.get("mean_tau"))
        series[name] = slot
    return series


def assemble_report(
    *,
    package: str,
    package_artifact_id: Optional[str],
    sidecars: dict,
    max_new_tokens: int,
    adaptive_cap: bool,
    environment: dict,
    repeat_entries: Sequence[dict],
    prompt_set: Sequence[dict],
) -> dict:
    """Assemble the battery report from already-collected repeat entries."""
    return {
        "tool": "moespresso-ds4-spec-battery",
        "package": package,
        "package_artifact_id": package_artifact_id,
        "sidecars": sidecars,
        "max_new_tokens": max_new_tokens,
        "adaptive_cap": adaptive_cap,
        "temperature": 0.0,
        "repeats": len(repeat_entries),
        "prompt_set": [
            {"id": p["id"], "category": p["category"]} for p in prompt_set
        ],
        "environment": environment,
        "runs": list(repeat_entries),
        "aggregates_by_repeat": aggregates_by_repeat(repeat_entries),
    }


def _sidecar_manifest_name(family: str) -> str:
    if family == "dspark":
        from moespresso.package.deepseek_v4.dspark_sidecar import SIDECAR_MANIFEST_NAME

        return SIDECAR_MANIFEST_NAME
    if family == "mtp":
        from moespresso.package.deepseek_v4.mtp_sidecar import SIDECAR_MANIFEST_NAME

        return SIDECAR_MANIFEST_NAME
    if family == "dflash":
        from moespresso.package.deepseek_v4.dflash_sidecar import (
            SIDECAR_MANIFEST_NAME,
        )

        return SIDECAR_MANIFEST_NAME
    raise ValueError(f"unknown drafter family: {family!r}")


def _sidecar_artifact_id(family: str, sidecar_dir: Path) -> Optional[str]:
    path = Path(sidecar_dir) / _sidecar_manifest_name(family)
    try:
        return json.loads(path.read_text()).get("artifact_id")
    except (OSError, ValueError):
        return None


def _run_spec_arm(
    model, drafter, prompt_ids: Sequence[int], max_new_tokens: int,
    eos_ids: Sequence[int], step: int, adaptive_cap: bool,
):
    """One speculative generation with the tap bracketed around it."""
    tap = install_hidden_tap(model, drafter.tap_layer_ids, drafter.tap_transform)
    try:
        t0 = time.perf_counter()
        spec = spec_generate(
            model,
            drafter,
            tap,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            eos_ids=eos_ids,
            prefill_step_size=step,
            adaptive_cap=adaptive_cap,
        )
        seconds = time.perf_counter() - t0
    finally:
        uninstall_hidden_tap(model, tap)
    return spec, seconds


def _warmup(model, tokenizer, drafters: dict, step: int, eos_ids: Sequence[int]) -> None:
    """Short untimed pass per arm so kernel compilation and first-shape
    costs do not land on the first measured prompt."""
    ids = tokenizer.encode(_WARMUP_PROMPT)
    plain_greedy_generate(model, ids, _WARMUP_TOKENS, eos_ids, step)
    for drafter in drafters.values():
        _run_spec_arm(model, drafter, ids, _WARMUP_TOKENS, eos_ids, step, True)


def run_battery(
    package_dir: Path,
    sidecar_dirs: Dict[str, Path],
    max_new_tokens: int = 300,
    adaptive_cap: bool = True,
    repeats: int = 1,
    prompts: Sequence[dict] = EVALUATION_PROMPTS,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Run the full battery and return the report dict.

    `sidecar_dirs` maps drafter family to sidecar directory; iteration
    order is arm order. The target package and every drafter stay
    resident for the whole run.
    """
    validate_prompt_set(prompts)
    if not sidecar_dirs:
        raise ValueError("at least one drafter sidecar is required")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    def emit(line: str) -> None:
        if progress is not None:
            progress(line)

    t_start = time.perf_counter()
    model, tokenizer, manifest = load_served_model(Path(package_dir))
    drafters = {
        family: build_drafter(family, Path(d), model)
        for family, d in sidecar_dirs.items()
    }
    step = _prefill_step_size(model)
    eos_ids = (
        [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    )
    environment = {
        "mlx_version": getattr(mx, "__version__", None),
        "thermal_at_start": thermal_snapshot(),
    }
    sidecar_meta = {
        family: {
            "path": str(d),
            "artifact_id": _sidecar_artifact_id(family, Path(d)),
        }
        for family, d in sidecar_dirs.items()
    }

    emit(f"[battery] warmup ({_WARMUP_TOKENS} tokens per arm)")
    _warmup(model, tokenizer, drafters, step, eos_ids)

    repeat_entries: List[dict] = []
    for r in range(repeats):
        entry: dict = {"repeat": r, "thermal": thermal_snapshot(), "prompts": {}}
        for prompt in prompts:
            prompt_ids = tokenizer.encode(prompt["text"])
            plain_tokens, prefill_s, decode_s = plain_greedy_generate(
                model, prompt_ids, max_new_tokens, eos_ids, step
            )
            arms = {
                PLAIN_ARM: plain_arm_record(
                    plain_tokens, prefill_s, decode_s, tokenizer.decode(plain_tokens)
                )
            }
            emit(
                f"[battery] repeat {r} prompt {prompt['id']} arm plain: "
                f"{arms[PLAIN_ARM]['decode_tok_per_s']:.2f} decode tok/s"
            )
            for family, drafter in drafters.items():
                spec, seconds = _run_spec_arm(
                    model, drafter, prompt_ids, max_new_tokens,
                    eos_ids, step, adaptive_cap,
                )
                arms[family] = spec_arm_record(
                    spec.tokens,
                    spec.stats,
                    seconds,
                    tokenizer.decode(spec.tokens),
                    plain_record=arms[PLAIN_ARM],
                    plain_tokens=plain_tokens,
                )
                ratio = arms[family]["ratio_vs_plain"]
                line = (
                    f"[battery] repeat {r} prompt {prompt['id']} arm {family}: "
                    f"{arms[family]['tok_per_s']:.2f} tok/s "
                    f"tau={arms[family]['tau']:.3f}"
                )
                if ratio is not None:
                    line += f" ratio={ratio:.3f}"
                emit(line)
            entry["prompts"][prompt["id"]] = {
                "category": prompt["category"],
                "prompt_tokens": len(prompt_ids),
                "arms": arms,
            }
        entry["aggregates"] = repeat_aggregates(entry["prompts"])
        repeat_entries.append(entry)

    environment["thermal_at_end"] = thermal_snapshot()
    report = assemble_report(
        package=str(package_dir),
        package_artifact_id=manifest.get("artifact_id"),
        sidecars=sidecar_meta,
        max_new_tokens=max_new_tokens,
        adaptive_cap=adaptive_cap,
        environment=environment,
        repeat_entries=repeat_entries,
        prompt_set=prompts,
    )
    report["total_seconds"] = time.perf_counter() - t_start
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--package", type=Path, required=True,
                        help="target package directory")
    parser.add_argument("--dspark-sidecar", type=Path, default=None,
                        help="DSpark drafter sidecar directory")
    parser.add_argument("--mtp-sidecar", type=Path, default=None,
                        help="MTP drafter sidecar directory")
    parser.add_argument("--dflash-sidecar", type=Path, default=None,
                        help="DFlash drafter sidecar directory")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--no-adaptive", action="store_true",
                        help="fixed full-block schedule instead of the "
                             "adaptive verify-length cap")
    parser.add_argument("--repeats", type=int, default=1,
                        help="run the whole battery this many times in one "
                             "process and report per-repeat aggregates "
                             "side by side")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    sidecar_dirs: Dict[str, Path] = {}
    if args.dspark_sidecar is not None:
        sidecar_dirs["dspark"] = args.dspark_sidecar
    if args.mtp_sidecar is not None:
        sidecar_dirs["mtp"] = args.mtp_sidecar
    if args.dflash_sidecar is not None:
        sidecar_dirs["dflash"] = args.dflash_sidecar
    if not sidecar_dirs:
        parser.error(
            "provide at least one of --dspark-sidecar, --mtp-sidecar, or "
            "--dflash-sidecar")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")

    report = run_battery(
        args.package,
        sidecar_dirs,
        max_new_tokens=args.max_new_tokens,
        adaptive_cap=not args.no_adaptive,
        repeats=args.repeats,
        progress=lambda line: print(line, file=sys.stderr, flush=True),
    )
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.json_out is not None:
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
