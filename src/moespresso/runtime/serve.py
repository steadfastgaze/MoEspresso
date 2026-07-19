"""Serving a mjtq package: build the model FROM the manifest, then generate.

The model is built by the strict manifest-driven backend (runtime/build): the
graph is instantiated from the manifest's architecture and packed weights are
installed into the declared quantized modules. Generated config sidecars are
compatibility views. The manifest remains the package source of truth and the
runtime never performs source-model archaeology.

The engine does not repeat whole-package verification on load. Run
`moespresso-verify` after a build, download, copy, or move; the expensive
integrity gate remains separate from the serve path. `build_fn(manifest,
package_dir)` is the swappable backend seam.

Needs the standard runtime dependencies (mlx, mlx-lm). The pure verifier is importable without
that extra; heavy imports are lazy so `import moespresso.runtime.serve` works
without them.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from importlib import metadata as importlib_metadata
from pathlib import Path

from moespresso.core.artifact import ArtifactError, read_artifact
from moespresso.package.constants import MANIFEST_NAME
from moespresso.runtime.generation import GenerationResult
from moespresso.runtime.kv_policy import KVPolicy, stream_generate_kv_kwargs, validate_runtime_policy
from moespresso.runtime.verify import verify_generated_sidecars, verify_package

class PackageNotFoundError(FileNotFoundError):
    """The given path is not a MoEspresso package (no manifest there)."""


def _preflight_manifest_for_cli(package_dir: Path) -> dict | None:
    """Best-effort manifest read for CLI option gating before heavy model load."""
    manifest_path = Path(package_dir) / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        return read_artifact(manifest_path)
    except (OSError, json.JSONDecodeError, ArtifactError):
        return None


def _uses_ssd_streaming_runtime(manifest: dict) -> bool:
    from moespresso.runtime.build import _runtime_adapter_kind

    return _runtime_adapter_kind(manifest) in {"jangtq_moe", "qwen_kquant_moe"}


def build_manifest_runtime(
    manifest: dict,
    package_dir: Path,
    *,
    resident_builder: Callable[[dict, Path], tuple] | None = None,
    streaming_builder: Callable[[Path], tuple] | None = None,
):
    """Build the runtime declared by the manifest.

    SSD streaming is the primary MoE runtime; fully dense packages remain
    the explicit non-streaming exception because they have no routed experts to
    stream.
    """
    package_dir = Path(package_dir)
    if _uses_ssd_streaming_runtime(manifest):
        if streaming_builder is None:
            from moespresso.runtime.ssd_streaming_build import build_ssd_streaming_model

            streaming_builder = build_ssd_streaming_model
        built = streaming_builder(package_dir)
        model, tokenizer = built[:2]
        return model, tokenizer

    if resident_builder is None:
        from moespresso.runtime.build import build_model
        resident_builder = build_model
    return resident_builder(manifest, package_dir)


def _manifest_driven_backend(
    manifest: dict,
    package_dir: Path,
):
    """The mjtq backend: build via the proven jang loader (runtime.build).

    The package carries jang-compatible sidecars (config.json/jang_config.json)
    generated from the manifest; build_model feeds them to jang's load_jangtq_model
    (TQ experts -> metal-kernel modules, affine -> mlx QuantizedLinear, per-tensor
    bits via tensor_map): no dequant at load. Returns (model, tokenizer); the
    tokenizer is loaded from the package by mlx_lm."""
    return build_manifest_runtime(manifest, package_dir)


def _apple_silicon_generation() -> int | None:
    """Apple silicon generation number (M1 -> 1), or None off Apple hardware."""
    if sys.platform != "darwin":
        return None
    try:
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"Apple M(\d+)", brand)
    return int(match.group(1)) if match else None


def default_kq_seg_tile_for_hardware(
    generation: int | None = None,
) -> str | None:
    """Default KQ_SEG_TILE to the device-A half-weight tile on Apple M5+.

    t48x128x16ah stages only the routed GEMM's weight tile in half; rows,
    activation fragments, accumulators, and output stay float32. On M3 it
    measures quality-neutral (weight rounding rel ~2e-4 against an f64
    reference, long-prompt teacher-forced NLL flat) and ~5% faster at the
    kernel, but served TTFT is within run noise, so M3 and M4 keep the
    float default and the committed token-identity anchors stay exact.
    Newer GPU generations are expected to gain more from the halved
    threadgroup footprint; the default is unmeasured there, so serving
    prints a notice and an explicit KQ_SEG_TILE always wins. Returns the
    tile it set, or None when it left the environment alone.
    """
    if "KQ_SEG_TILE" in os.environ:
        return None
    if generation is None:
        generation = _apple_silicon_generation()
    if generation is None or generation < 5:
        return None
    tile = "t48x128x16ah"
    os.environ["KQ_SEG_TILE"] = tile
    print(
        f"[serve] Apple M{generation} detected: enabling a faster GPU math "
        "path. It is expected to be safe but has not been benchmarked on "
        "this chip generation yet. To turn it off, set the environment "
        "variable KQ_SEG_TILE=t48x128x16a",
        flush=True,
    )
    return tile


def _mlx_core_already_imported() -> bool:
    return "mlx.core" in sys.modules


def _installed_mlx_version() -> str | None:
    try:
        return importlib_metadata.version("mlx")
    except importlib_metadata.PackageNotFoundError:
        return None


_ORNITH_Q4KM_COMMAND_BUFFER_MANIFEST_ID = (
    "pkg:aff416b9eeecfe9d18dd31798bb3e3ee91a0ff634297a5236f74e17a6c9c0ce0"
)
_ORNITH_Q4KM_COMMAND_BUFFER_MLX_VERSION = "0.31.2"


def default_ornith_mlx_command_buffer_limit(
    manifest: dict,
    *,
    generation: int | None = None,
    total_memory_bytes: int | None = None,
) -> str | None:
    """Set the measured Ornith MLX command-buffer element limit before load.

    MLX 0.31.2 counts array elements for ``MLX_MAX_MB_PER_BUFFER`` even though
    the variable name refers to bytes. The measured Ornith Q4_K_M routed
    gate/up pool is exactly 288 Mi-elements. That limit keeps gate/up and down
    in one command buffer while retaining a bounded commit after the pair. The
    setting adds roughly 5 GiB to the 37K decode peak, so hosts below 64 GiB
    keep MLX's smaller default. An explicit MLX setting always wins. Other
    package manifests and MLX versions remain unchanged until measured.
    """
    if "MLX_MAX_MB_PER_BUFFER" in os.environ:
        return None

    architecture = manifest.get("architecture", {})
    if (
        manifest.get("artifact_id") != _ORNITH_Q4KM_COMMAND_BUFFER_MANIFEST_ID
        or architecture.get("family") != "qwen3_5_moe"
        or architecture.get("smoke_max_experts") is not None
    ):
        return None

    if generation is None:
        generation = _apple_silicon_generation()
    if generation != 3:
        return None

    if total_memory_bytes is None:
        import psutil

        total_memory_bytes = int(psutil.virtual_memory().total)
    if total_memory_bytes < 64 * (1 << 30):
        return None

    mlx_version = _installed_mlx_version()
    if mlx_version != _ORNITH_Q4KM_COMMAND_BUFFER_MLX_VERSION:
        detected_version = mlx_version or "unavailable"
        print(
            f"[serve] Ornith decode: command-buffer tuning was not applied "
            f"because the installed MLX version ({detected_version}) is not the measured "
            f"version {_ORNITH_Q4KM_COMMAND_BUFFER_MLX_VERSION}. Set "
            "MLX_MAX_MB_PER_BUFFER=288 at process launch to opt in.",
            flush=True,
        )
        return None

    if _mlx_core_already_imported():
        print(
            "[serve] Ornith decode: MLX was imported before package load, so "
            "command-buffer tuning was not applied. Set "
            "MLX_MAX_MB_PER_BUFFER=288 at process launch to enable it.",
            flush=True,
        )
        return None

    limit = "288"
    os.environ["MLX_MAX_MB_PER_BUFFER"] = limit
    print(
        "[serve] Ornith decode: using MLX_MAX_MB_PER_BUFFER=288. "
        "Set MLX_MAX_MB_PER_BUFFER=50 to retain the smaller-memory MLX default.",
        flush=True,
    )
    return limit


def load_served_model(
    package_dir: Path,
    *,
    manifest: dict | None = None,
    build_fn: Callable[[dict, Path], tuple] = _manifest_driven_backend,
):
    """Build (model, tokenizer, manifest) from a mjtq package.

    Reads the manifest (content-hash-verified by read_artifact) and builds the
    model straight from it; the engine trusts the declared contract and does not
    repeat whole-package verification on load. Integrity is checked on demand by
    `moespresso-verify` after a build, download, copy, or move, never on the serve
    path.
    `build_fn(manifest, package_dir)` is the swappable backend (default: the strict
    manifest-driven one).
    """
    package_dir = Path(package_dir)
    if manifest is None:
        manifest_path = package_dir / MANIFEST_NAME
        if not manifest_path.is_file():
            if not package_dir.is_dir():
                raise PackageNotFoundError(
                    f"package directory not found: {package_dir}"
                    + ("" if package_dir.is_absolute() else
                       " (relative to the current directory, packages "
                       "usually live under ~/.cache/huggingface/hub/)"))
            raise PackageNotFoundError(
                f"not a MoEspresso package: {package_dir} has no "
                f"{MANIFEST_NAME}")
        manifest = read_artifact(manifest_path)

    default_ornith_mlx_command_buffer_limit(manifest)
    default_kq_seg_tile_for_hardware()
    model, tokenizer = build_fn(manifest, package_dir)
    print(_runtime_truth_line(model, manifest), flush=True)
    return model, tokenizer, manifest


def _runtime_truth_line(model, manifest: dict) -> str:
    """One honest line stating which runtime the user actually got:
    the user must never guess whether gate/hotlist/capacity are live."""
    capacity = getattr(model, "_moespresso_ssd_streaming_capacity", None)
    if capacity is None:
        return (f"[serve] runtime=resident package="
                f"{manifest.get('artifact_id', '?')[:16]}")
    hot = getattr(model, "_moespresso_ssd_hotlist", None) or {}
    try:
        from moespresso.runtime.pooled_switchglu import (
            _RING_DECODE,
            _gate_module,
            _ring_visibility_ok,
        )
        if _RING_DECODE and _ring_visibility_ok() and _gate_module() is not None:
            decode_path = "native-gate"
        elif _RING_DECODE and _ring_visibility_ok():
            decode_path = "ring-v3(native-gate-not-built)"
        else:
            decode_path = "legacy"
    except Exception:
        decode_path = "unknown"
    lookahead = getattr(model, "_moespresso_ssd_lookahead", None)
    import os as _os_cap
    cap_note = _os_cap.environ.get("MOESPRESSO_SSD_MAX_MEMORY_GB")
    cap_note = f" max_memory={cap_note}GB" if cap_note else ""
    return (f"[serve] runtime=ssd-streaming package="
            f"{manifest.get('artifact_id', '?')[:16]} capacity={capacity}"
            f"{cap_note}"
            f" hotlist={hot.get('source')} seeded={hot.get('seeded', 0)}"
            f" decode={decode_path}"
            f" lookahead={'off' if not lookahead else lookahead}")


_HOTLIST_SAVE_WARNED = [False]


def _persist_expert_demand(model) -> None:
    """Persist per-request expert demand (default-on, kill switch
    MOESPRESSO_SSD_HOTLIST=0) so the next session warm-starts from the
    saved-demand tier (~0.60 demand-mass capture vs 0.40 package hotlist vs
    0.27 arbitrary). A failed save degrades the next cold
    start, never this request: warn once, don't raise."""
    import os

    info = getattr(model, "_moespresso_ssd_hotlist", None)
    if info is None or os.environ.get("MOESPRESSO_SSD_HOTLIST", "1") == "0":
        return
    try:
        from moespresso.runtime.ssd_streaming_build import save_expert_hotlist

        save_expert_hotlist(model, Path(info["save_path"]))
    except Exception as e:  # noqa: BLE001 - availability over loudness here
        if not _HOTLIST_SAVE_WARNED[0]:
            _HOTLIST_SAVE_WARNED[0] = True
            print(f"[serve] WARNING: expert-demand hotlist save failed "
                  f"({type(e).__name__}: {e}); next cold start loses the "
                  f"saved-demand tier", flush=True)


def _token_int(token) -> int:
    return int(token.item() if hasattr(token, "item") else token)


def _decode_token_bytes(tokenizer, token_id: int) -> tuple[str, list[int]]:
    text = tokenizer.decode([int(token_id)]) if tokenizer is not None else ""
    return text, list(text.encode("utf-8", "replace"))


def _top_logprob_entries(logprobs, tokenizer, *, k: int) -> tuple[dict, ...]:
    values = logprobs.tolist() if hasattr(logprobs, "tolist") else list(logprobs)
    if values and isinstance(values[0], list):
        values = [item for row in values for item in row]
    n = min(int(k), len(values))
    if n <= 0:
        return ()
    idx = sorted(range(len(values)), key=lambda i: float(values[i]), reverse=True)[:n]
    out = []
    for token_id in idx:
        token_int = int(token_id)
        text, raw = _decode_token_bytes(tokenizer, token_int)
        out.append({
            "token_id": token_int,
            "token": {"text": text, "bytes": raw},
            "logprob": float(values[token_int]),
        })
    return tuple(out)


def _logprob_at(logprobs, token_id: int) -> float:
    value = logprobs[int(token_id)]
    return float(value.item() if hasattr(value, "item") else value)


def _prompt_token_count(tokenizer, prompt: str | list[int]) -> int | None:
    if not isinstance(prompt, str):
        return len(prompt)
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        return None
    try:
        return len(encode(prompt))
    except TypeError:
        return len(encode(prompt, add_special_tokens=False))


def _model_prefill_step_size(model, tokenizer, prompt: str | list[int]) -> int | None:
    value = getattr(model, "_moespresso_prefill_step_size", None)
    if value is None:
        return None
    max_prompt_tokens = getattr(
        model,
        "_moespresso_prefill_step_size_max_prompt_tokens",
        None,
    )
    min_prompt_tokens = getattr(
        model,
        "_moespresso_prefill_step_size_min_prompt_tokens",
        None,
    )
    if max_prompt_tokens is None and min_prompt_tokens is None:
        return int(value)
    prompt_tokens = _prompt_token_count(tokenizer, prompt)
    if prompt_tokens is None:
        return None
    if max_prompt_tokens is not None and prompt_tokens > int(max_prompt_tokens):
        return None
    if min_prompt_tokens is not None and prompt_tokens < int(min_prompt_tokens):
        return None
    return int(value)


def _ornith_raw_greedy_eligible(
    model,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float | None,
    top_logprobs: int | None,
) -> bool:
    """Gate raw argmax to the measured deterministic Ornith request shape."""
    return (
        getattr(model, "model_type", None) == "qwen3_5_moe"
        and temperature == 0.0
        and top_p == 1.0
        and top_k == 0
        and min_p == 0.0
        and presence_penalty in (None, 0.0)
        and top_logprobs is None
    )


def prefill_prompt_cache_chunks(
    model,
    prompt_cache,
    tokens: list[int],
    plan: list[int],
    *,
    progress_callback: Callable[[int, int], None] | None = None,
    total_tokens: int | None = None,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
) -> int:
    """Advance ``prompt_cache`` over the leading ``plan`` chunks of ``tokens``.

    Mirrors the mlx-lm prefill loop step for step (model call on the
    generation stream under the wired limit, optional cache quantization,
    cache-state eval, progress callback, buffer release) with per-chunk sizes
    instead of one uniform step, so a chunk end can land exactly on a
    checkpoint frontier. ``progress_callback`` receives the running processed
    count against ``total_tokens`` (the full suffix length), the mlx-lm
    callback contract. Returns the number of tokens consumed.
    """
    import mlx.core as mx
    from mlx_lm.generate import (
        generation_stream,
        maybe_quantize_kv_cache,
        wired_limit,
    )

    total = int(total_tokens if total_tokens is not None else len(tokens))
    consumed = 0
    with wired_limit(model, [generation_stream]):
        with mx.stream(generation_stream):
            for size in plan:
                chunk = mx.array(tokens[consumed:consumed + size])
                model(chunk[None], cache=prompt_cache)
                maybe_quantize_kv_cache(
                    prompt_cache,
                    quantized_kv_start=quantized_kv_start,
                    kv_group_size=kv_group_size,
                    kv_bits=kv_bits,
                )
                mx.eval([c.state for c in prompt_cache])
                consumed += size
                if progress_callback is not None:
                    progress_callback(consumed, total)
                mx.clear_cache()
    return consumed


def generate_with_metadata(
    model,
    tokenizer,
    prompt: str | list[int],
    *,
    prompt_cache=None,
    cached_tokens: int | None = None,
    kv_policy: KVPolicy | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    presence_penalty: float | None = None,
    presence_context_size: int = 20,
    top_logprobs: int | None = None,
    prefill_step_size: int | None = None,
    prefill_plan: list[int] | None = None,
    prompt_progress_callback: Callable[[int, int], None] | None = None,
    response_callback: Callable[[int, object], None] | None = None,
    persist_expert_demand: bool = True,
    stream_generate_fn: Callable | None = None,
    raw_greedy_stream_fn: Callable | None = None,
    sampler_factory: Callable | None = None,
    logits_processors_factory: Callable | None = None,
    prefill_chunks_fn: Callable | None = None,
) -> GenerationResult:
    """Generate from an already-rendered string or token ids and return metadata.

    This is the prompt-cache seam: callers may provide a prompt cache, and MLX mutates that cache
    in place. The same cache object is returned so the future cache manager can insert it
    under the full token sequence. Deterministic Ornith requests use the MoEspresso-owned
    raw-greedy stream. Other request shapes use the corresponding MLX LM generation features.
    Tests inject stream functions so the contract stays testable without a GPU/model.

    ``prefill_plan`` gives the leading prompt tokens variable-size prefill
    chunks (the disk-KV frontier writer sizes them so a chunk end lands on
    every checkpoint frontier); the remaining tail then streams at
    ``prefill_step_size``. The plan consumes at most ``len(prompt) - 1``
    tokens so generation always keeps a prompt token to feed. Reported
    ``prompt_tokens`` and the progress callback both count the whole prompt,
    so the accounting is identical with and without a plan.
    """
    kv_kwargs = {}
    if kv_policy is not None:
        validate_runtime_policy(kv_policy)
        kv_kwargs = stream_generate_kv_kwargs(kv_policy)
    if prefill_step_size is None:
        prefill_step_size = _model_prefill_step_size(model, tokenizer, prompt)
    if prefill_step_size is not None:
        kv_kwargs["prefill_step_size"] = int(prefill_step_size)

    plan_consumed = 0
    if prefill_plan:
        plan = [int(size) for size in prefill_plan]
        if isinstance(prompt, str):
            raise ValueError("a prefill plan requires a token-id prompt")
        if prompt_cache is None:
            raise ValueError("a prefill plan requires an explicit prompt cache")
        if any(size < 1 for size in plan):
            raise ValueError("prefill plan chunk sizes must be positive")
        if sum(plan) > len(prompt) - 1:
            raise ValueError(
                f"prefill plan covers {sum(plan)} tokens; at most "
                f"{len(prompt) - 1} of the {len(prompt)}-token prompt may be "
                f"pre-consumed")
    else:
        plan = None

    if prompt_progress_callback is not None and plan is None:
        kv_kwargs["prompt_progress_callback"] = prompt_progress_callback

    uses_default_stream_generate = stream_generate_fn is None
    uses_default_sampler_factory = sampler_factory is None
    if logits_processors_factory is None:
        from mlx_lm.sample_utils import (
            make_logits_processors as logits_processors_factory,
        )

    # Presence penalty is a logits processor. mlx_lm subtracts the penalty
    # from tokens seen within the configured context window.
    logits_processors = None
    if presence_penalty:
        logits_processors = logits_processors_factory(
            presence_penalty=float(presence_penalty),
            presence_context_size=int(presence_context_size),
        )
    if logits_processors:
        kv_kwargs["logits_processors"] = logits_processors

    owned_raw_greedy = (
        uses_default_stream_generate
        and uses_default_sampler_factory
        and not logits_processors
        and _ornith_raw_greedy_eligible(
            model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            top_logprobs=top_logprobs,
        )
    )

    if uses_default_stream_generate:
        if owned_raw_greedy:
            if raw_greedy_stream_fn is None:
                from moespresso.runtime.raw_greedy import (
                    stream_raw_greedy as raw_greedy_stream_fn,
                )

            stream_generate_fn = raw_greedy_stream_fn
        else:
            from mlx_lm import stream_generate as stream_generate_fn

    text_parts: list[str] = []
    generated: list[int] = []
    token_logprobs: list[float] = []
    captured_top_logprobs: list[tuple[dict, ...]] = []
    last = None
    t_start = time.perf_counter()
    first_token_seconds = None

    # Planned variable-step prefill runs first, inside the timed span so its
    # cost lands in first_token_seconds like any other prefill work. The tail
    # handed to stream_generate keeps prompt_progress_callback semantics: the
    # shifted callback reports processed counts against the whole prompt, so
    # the frontier writer's absolute accounting is unchanged.
    if plan:
        if prefill_chunks_fn is None:
            prefill_chunks_fn = prefill_prompt_cache_chunks
        total_prompt_tokens = len(prompt)
        plan_consumed = prefill_chunks_fn(
            model,
            prompt_cache,
            list(prompt),
            plan,
            progress_callback=prompt_progress_callback,
            total_tokens=total_prompt_tokens,
            kv_bits=kv_kwargs.get("kv_bits"),
            kv_group_size=kv_kwargs.get("kv_group_size", 64),
            quantized_kv_start=kv_kwargs.get("quantized_kv_start", 0),
        )
        prompt = list(prompt[plan_consumed:])
        if prompt_progress_callback is not None:
            inner_callback = prompt_progress_callback
            consumed_offset = plan_consumed

            def _shifted_progress(processed: int, _total: int) -> None:
                inner_callback(consumed_offset + int(processed),
                               total_prompt_tokens)

            kv_kwargs["prompt_progress_callback"] = _shifted_progress

    generation_kwargs = dict(kv_kwargs)
    if not owned_raw_greedy:
        if sampler_factory is None:
            from mlx_lm.sample_utils import make_sampler as sampler_factory

        generation_kwargs["sampler"] = sampler_factory(
            temp=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )

    for response in stream_generate_fn(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        prompt_cache=prompt_cache,
        **generation_kwargs,
    ):
        if first_token_seconds is None:
            first_token_seconds = time.perf_counter() - t_start
        last = response
        text_parts.append(response.text)
        token_id = _token_int(response.token)
        generated.append(token_id)
        response_logprobs = getattr(response, "logprobs", None)
        if top_logprobs is not None and response_logprobs is not None:
            captured_top_logprobs.append(
                _top_logprob_entries(response_logprobs, tokenizer, k=top_logprobs)
            )
            token_logprobs.append(_logprob_at(response_logprobs, token_id))
        if response_callback is not None:
            response_callback(len(generated), response)
    generation_seconds = time.perf_counter() - t_start

    if persist_expert_demand:
        _persist_expert_demand(model)

    if last is None:
        prompt_tokens = (
            len(prompt) + plan_consumed if not isinstance(prompt, str) else None)
        return GenerationResult(
            text="",
            finish_reason="length",
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            cached_tokens=cached_tokens,
        generated_token_ids=(),
        token_logprobs=(),
        top_logprobs=(),
        prompt_cache=prompt_cache,
    )

    return GenerationResult(
        text="".join(text_parts),
        finish_reason=last.finish_reason or "length",
        prompt_tokens=int(last.prompt_tokens) + plan_consumed,
        completion_tokens=int(last.generation_tokens),
        cached_tokens=cached_tokens,
        generated_token_ids=tuple(generated),
        token_logprobs=tuple(token_logprobs),
        top_logprobs=tuple(captured_top_logprobs),
        prompt_cache=prompt_cache,
        first_token_seconds=first_token_seconds,
        generation_seconds=generation_seconds,
    )


def generate_once(
    model, tokenizer, prompt: str, *, max_tokens: int = 2048,
    temperature: float = 0.7, top_p: float = 1.0,
    kv_policy: KVPolicy | None = None,
    stream_generate_fn: Callable | None = None,
    sampler_factory: Callable | None = None,
) -> str:
    """Compatibility wrapper: generate from an already-rendered prompt and return text.

    The chat template is applied exactly once, by the render layer (runtime.http.render_prompt
    for the server; main() below for the CLI). This function must never call
    apply_chat_template, or the prompt is templated twice (template(template(messages))),
    which corrupts the token stream and any KV prefix keyed on it. mlx_lm.generate only
    .encode()s a string prompt; it does not re-template.
    """
    return generate_with_metadata(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        kv_policy=kv_policy,
        stream_generate_fn=stream_generate_fn,
        sampler_factory=sampler_factory,
    ).text


def _generation_json_payload(
    *,
    package_dir: Path,
    manifest: dict,
    user_prompt: str,
    rendered_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    thinking: str | None,
    result: GenerationResult,
) -> dict:
    return {
        "artifact_kind": "generation_smoke",
        "package_dir": str(package_dir),
        "package_manifest_id": manifest.get("artifact_id"),
        "package_family": manifest.get("architecture", {}).get("family"),
        "user_prompt": user_prompt,
        "rendered_prompt_chars": len(rendered_prompt),
        "params": {
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "thinking": thinking,
        },
        "text": result.text,
        "finish_reason": result.finish_reason,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "cached_tokens": result.cached_tokens,
        "generated_token_ids": list(result.generated_token_ids),
        "token_logprobs": list(result.token_logprobs),
        "top_logprobs": [list(step) for step in result.top_logprobs],
        "cache_event": result.cache_event,
        "cache_entries": result.cache_entries,
        "cache_bytes": result.cache_bytes,
        "first_token_seconds": result.first_token_seconds,
        "generation_seconds": result.generation_seconds,
    }


def add_runtime_limit_arguments(parser) -> None:
    """Add the shared context and expert-residency CLI controls."""
    from moespresso.runtime.prefix_cache import DEFAULT_CONTEXT_LIMIT

    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="Maximum prompt-plus-output context tokens. Default: "
             f"{DEFAULT_CONTEXT_LIMIT:,} or the package limit, whichever is smaller.",
    )
    parser.add_argument(
        "--min-resident-experts",
        type=int,
        default=None,
        help="Require at least this many resident routed experts per layer; "
             "startup fails when the loaded capacity is smaller.",
    )


def validate_runtime_limit_arguments(parser, args) -> None:
    """Reject non-positive shared runtime limits at the CLI boundary."""
    if args.max_context_tokens is not None and args.max_context_tokens < 1:
        parser.error("--max-context-tokens must be >= 1")
    if args.min_resident_experts is not None and args.min_resident_experts < 1:
        parser.error("--min-resident-experts must be >= 1")


def main(argv: list[str] | None = None) -> int:
    """`uv run moespresso-generate <package_dir> [--prompt ...]`: load + generate.

    Generate-only entrypoint (prompt in, text out). Builds straight from the
    manifest; does not verify (run `moespresso-verify` for the integrity gate). The
    OpenAI-compatible HTTP server (`moespresso-serve`, runtime/http.py) is a thin
    layer over the same load_served_model + generate_once seam; this CLI stays the
    minimal one-shot.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="moespresso-generate",
        description="Load a MoEspresso package from its manifest, then generate.")
    parser.add_argument("package_dir", help="Path to the packaged model directory")
    parser.add_argument("--max-memory-gb", type=float, default=None,
                        help="Set the streamed runtime's startup capacity-planner "
                             "ceiling (GB). This selects expert-pool geometry and "
                             "can simulate a smaller pool; it is not an RSS cap.")
    add_runtime_limit_arguments(parser)
    parser.add_argument("--prompt", default="Hello", help="Prompt to generate from")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--thinking", choices=("off", "on", "high", "max"),
                        default=None,
                        help="Select the model family's own thinking mode: "
                             "off, on (high is the same), or max (DeepSeek-V4 "
                             "reasoning-effort preamble; DeepSeek-V4 only). "
                             "Refuses if the family has no mechanism. "
                             "Default: the template's own default "
                             "(DeepSeek-V4: off).")
    parser.add_argument("--json-out", type=Path,
                        help="Write structured generation metadata to this JSON file.")
    args = parser.parse_args(argv)
    validate_runtime_limit_arguments(parser, args)
    if args.max_memory_gb is not None:
        import os as _os_cap
        _os_cap.environ["MOESPRESSO_SSD_MAX_MEMORY_GB"] = str(args.max_memory_gb)

    pkg = Path(args.package_dir)
    if args.thinking is not None:
        from moespresso.runtime.http import (
            is_deepseek_v4_manifest,
            thinking_effort_option_error,
        )

        preflight_manifest = _preflight_manifest_for_cli(pkg)
        if preflight_manifest is not None:
            option_error = thinking_effort_option_error(
                args.thinking,
                is_deepseek_v4=is_deepseek_v4_manifest(preflight_manifest))
            if option_error is not None:
                print(option_error)
                return 2

    print(f"Loading package from its manifest: {pkg}")
    from moespresso.runtime.streaming_capacity import (
        StreamingCapacityError,
        validate_min_resident_experts,
    )

    try:
        model, tokenizer, manifest = load_served_model(pkg)
        validate_min_resident_experts(
            model,
            requested=args.min_resident_experts,
        )
    except (PackageNotFoundError, StreamingCapacityError) as e:
        print(f"FAILED: {e}")
        return 2
    from moespresso.runtime.prefix_cache import (
        declared_context_limit,
        effective_context_limit,
    )

    try:
        context_limit = effective_context_limit(
            manifest,
            requested=args.max_context_tokens,
        )
    except ValueError as e:
        print(f"FAILED: {e}")
        return 2
    print(
        f"[generate] context_limit={context_limit} "
        f"package_limit={declared_context_limit(manifest) or 'unknown'}"
    )
    print(f"  loaded ({len(manifest['tensors'])} tensors, "
          f"{len(manifest['files'])} shard(s)).")

    # Render once here (the same seam the server uses), then generate from the rendered
    # prompt: generate_once no longer templates. Family-owned render defaults are applied
    # before generation; DS4 maps --thinking onto its official encoder modes.
    from moespresso.runtime.http import (
        deepseek_v4_contract_template_kwargs,
        is_deepseek_v4_manifest,
        normalize_thinking_selection,
        render_prompt,
        thinking_effort_option_error,
    )
    selection = normalize_thinking_selection(args.thinking)
    ds4_contract = is_deepseek_v4_manifest(manifest)
    option_error = thinking_effort_option_error(
        selection, is_deepseek_v4=ds4_contract)
    if option_error is not None:
        print(option_error)
        return 2
    template_kwargs = None
    effective_thinking = selection
    if ds4_contract:
        template_kwargs = deepseek_v4_contract_template_kwargs(selection)
        effective_thinking = "max" if selection == "max" else (
            "on" if template_kwargs["enable_thinking"] else "off")
        print(f"[generate] thinking={effective_thinking} "
              f"via=deepseek_v4_contract")
    elif selection is not None:
        from moespresso.runtime.thinking import resolve_thinking_kwargs
        template_kwargs = resolve_thinking_kwargs(
            tokenizer, thinking=selection == "on",
            family=manifest.get("architecture", {}).get("family"))
        print(f"[generate] thinking={selection} via={template_kwargs}")
    prompt = render_prompt([{"role": "user", "content": args.prompt}], tokenizer,
                           template_kwargs=template_kwargs,
                           prompt_renderer=manifest.get("architecture", {}).get("prompt_renderer"))
    from moespresso.runtime.prefix_cache import (
        encode_rendered_prompt,
        validate_context_span,
    )

    prompt_tokens = encode_rendered_prompt(tokenizer, prompt)
    try:
        validate_context_span(
            limit=context_limit,
            prompt_tokens=len(prompt_tokens),
            max_tokens=args.max_tokens,
        )
    except ValueError as e:
        print(f"FAILED: {e}")
        return 2
    result = generate_with_metadata(
        model, tokenizer, prompt_tokens, max_tokens=args.max_tokens,
        temperature=args.temperature, top_p=args.top_p)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                _generation_json_payload(
                    package_dir=pkg,
                    manifest=manifest,
                    user_prompt=args.prompt,
                    rendered_prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    thinking=effective_thinking,
                    result=result,
                ),
                indent=2,
            )
            + "\n"
        )
    print(result.text)
    return 0


def verify_main(argv: list[str] | None = None) -> int:
    """`uv run moespresso-verify <package_dir>`: the on-demand integrity gate.

    Runs the manifest-contract, declared-file, tensor-key, and generated-sidecar
    checks the serve hot path deliberately skips. Exit 0 = clean, 2 = failed.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="moespresso-verify",
        description="Check a MoEspresso package matches its manifest contract.")
    parser.add_argument("package_dir", help="Path to the packaged model directory")
    args = parser.parse_args(argv)

    pkg = Path(args.package_dir)
    manifest_path = pkg / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"FAILED: cannot read {manifest_path}: {exc}")
        return 2
    if not isinstance(manifest, dict):
        print(f"FAILED: {manifest_path} must contain a JSON object")
        return 2
    issues = verify_package(manifest, pkg)
    issues.extend(verify_generated_sidecars(manifest, pkg))
    for v in issues:
        print(f"  [{v.severity}] {v.code}: {v.message}")
    blocking = [v for v in issues if v.blocking]
    if blocking:
        print(f"FAILED: {len(blocking)} blocking issue(s) in {pkg}")
        return 2
    print(f"OK: {pkg} matches its manifest "
          f"({len(manifest['tensors'])} tensors, {len(manifest['files'])} shard(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
