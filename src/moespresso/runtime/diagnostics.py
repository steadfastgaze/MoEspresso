"""A stdout speed check and resource telemetry for MoEspresso servers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import sys
import tempfile
import time
from urllib.parse import urlsplit
import uuid

from moespresso.agentlib.client import CompletionsClient
from moespresso.runtime.process_resources import gpu_inventory, process_resources


CASES = (
    ("history", "on which year was america discovered?"),
    ("coding", "how to solve fibonacci in python?"),
    ("biography", "who is berlusconi?"),
)
SPEED_TOPICS = (
    (
        "history",
        "Explain the causes of the French Revolution.",
        "Explain how the French Revolution changed European politics.",
    ),
    (
        "coding",
        "Explain how to implement Fibonacci numbers in Python.",
        "Explain how memoization improves a recursive Fibonacci function.",
    ),
    (
        "biography",
        "Who was Ada Lovelace?",
        "Explain Ada Lovelace's contribution to computing.",
    ),
)
SAMPLING = dict(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, presence_penalty=0.0)
CAPACITY_WARNING_MIN_SLOTS = 4
CAPACITY_WARNING_FRACTION = 0.03


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def make_server_diagnostics(manifest, *, context_limit, template_kwargs, generation_defaults):
    """Create one server identity; resource reads never evaluate or clear MLX."""
    versions = {}
    for name in ("moespresso", "mlx", "mlx-lm", "mlx-iqk", "mlx-kquant", "psutil"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    identity = {
        "instance_id": uuid.uuid4().hex,
        "pid": os.getpid(),
        "package_artifact_id": manifest.get("artifact_id"),
        "family": manifest.get("architecture", {}).get("family"),
        "rendering_id": manifest.get("tokenizer", {}).get("rendering_id"),
        "context_limit": context_limit,
        "template_kwargs": template_kwargs,
        "generation_defaults": generation_defaults,
        "versions": versions,
    }
    inventory = gpu_inventory()
    if inventory is not None:
        identity["gpu_inventory"] = inventory

    def snapshot():
        result = {
            "schema": "moespresso-server-diagnostics-v3",
            "identity": identity,
            "sampled_at_unix": time.time(),
            "errors": {},
        }
        try:
            import psutil

            vm, swap = psutil.virtual_memory(), psutil.swap_memory()
            result["system_memory"] = {
                "total_bytes": vm.total,
                "available_bytes": vm.available,
                "used_bytes": vm.used,
                "swap_used_bytes": swap.used,
                "paging_in_bytes": swap.sin,
                "paging_out_bytes": swap.sout,
                "paging_counter_kind": (
                    "vm_pageins_pageouts" if sys.platform == "darwin" else "psutil_sin_sout"
                ),
            }
            result["process_memory"] = dict(psutil.Process().memory_info()._asdict())
        except Exception as exc:
            result["errors"]["system_memory"] = type(exc).__name__
        try:
            result["process_io"] = process_resources()
        except Exception as exc:
            result["process_io"] = None
            result["errors"]["process_io"] = type(exc).__name__
        try:
            import psutil

            battery = psutil.sensors_battery()
            result["battery"] = (
                {"percent": battery.percent, "power_plugged": battery.power_plugged}
                if battery is not None
                else None
            )
        except Exception as exc:
            result["errors"]["battery"] = type(exc).__name__
        try:
            import mlx.core as mx

            result["device"] = mx.device_info()
            result["mlx_memory"] = {
                "active_bytes": mx.get_active_memory(),
                "cached_bytes": mx.get_cache_memory(),
                "peak_active_bytes": mx.get_peak_memory(),
            }
        except Exception as exc:
            result["errors"]["mlx_memory"] = type(exc).__name__
        return result

    return snapshot


def _numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _metrics(usage):
    timing = usage.get("moespresso", {})
    count = usage.get("completion_tokens")
    total, first = timing.get("generation_seconds"), timing.get("first_token_seconds")
    rate = None
    if all(_numeric(v) for v in (count, total, first)) and count > 1 and total > first >= 0:
        rate = (count - 1) / (total - first)
    first_time = first if all(_numeric(v) for v in (total, first)) and 0 <= first <= total else None
    return {"after_first_token_tps": rate, "first_token_seconds": first_time}


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _positive_int(value):
    return value if type(value) is int and value > 0 else None


def _gib(value):
    return value / (1 << 30) if _positive_int(value) is not None else None


def _family_label(value):
    labels = {
        "deepseek_v4_flash": "DeepSeek-V4-Flash",
        "qwen3_5_moe": "Ornith",
        "qwen4_exp": "Qwen3.8-Flash-Next",
        "qwen4_exp_text": "Qwen3.8-Flash-Next",
    }
    if not isinstance(value, str) or not value:
        return None
    return labels.get(value, value.replace("_", "-"))


def _capacity_range(stats):
    base = _positive_int(stats.get("capacity_per_layer"))
    values = [base] if base is not None else []
    for value in _mapping(stats.get("capacity_overrides")).values():
        parsed = _positive_int(value)
        if parsed is not None:
            values.append(parsed)
    if not values:
        return None, None, None
    return min(values), max(values), str(values[0]) if len(set(values)) == 1 else f"{min(values)}-{max(values)}"


def _normal_planner_capacity(stats, family):
    budget = _mapping(stats.get("capacity_budget"))
    resolution = _mapping(budget.get("planner_resolution"))
    ceilings = [
        _positive_int(resolution.get("physical_reserve_ceiling_bytes")),
        _positive_int(resolution.get("automatic_ceiling_bytes")),
        _positive_int(resolution.get("explicit_ceiling_bytes")),
    ]
    ceilings = [value for value in ceilings if value is not None]
    bytes_per_slot = _positive_int(budget.get("bytes_per_capacity_unit"))
    if not ceilings or bytes_per_slot is None:
        return None
    available = min(ceilings)
    deductions = sum(
        _positive_int(budget.get(name)) or 0
        for name in (
            "resident_base_bytes",
            "runtime_resident_bytes",
            "kv_activation_allowance_bytes",
            "safety_margin_bytes",
        )
    )
    usable = max(0, available - deductions)
    maximum = _positive_int(budget.get("max_capacity"))
    if maximum is None and family in {"qwen4_exp", "qwen4_exp_text"}:
        maximum = 512
    full_resident = _positive_int(budget.get("full_resident_expert_bytes"))
    if maximum is not None and full_resident is not None and usable >= full_resident:
        return maximum
    capacity = usable // bytes_per_slot
    return min(capacity, maximum) if maximum is not None else capacity


def _speed_environment(health):
    """Select shareable run conditions from a server health response."""
    health = _mapping(health)
    diagnostics = _mapping(health.get("diagnostics"))
    identity = _mapping(diagnostics.get("identity"))
    device = _mapping(diagnostics.get("device"))
    memory = _mapping(diagnostics.get("system_memory"))
    stats = _mapping(health.get("ssd_streaming"))
    family = identity.get("family")
    lines = []

    device_name = device.get("device_name")
    total_gib = _gib(memory.get("total_bytes"))
    inventory = identity.get("gpu_inventory")
    cores = None
    if isinstance(inventory, list) and len(inventory) == 1:
        cores = _positive_int(_mapping(inventory[0]).get("core_count"))
    hardware = []
    if isinstance(device_name, str) and device_name:
        hardware.append(device_name)
    if cores is not None:
        hardware.append(f"{cores} GPU cores")
    if total_gib is not None:
        hardware.append(f"{total_gib:.0f} GiB")
    if hardware:
        lines.append(" · ".join(hardware))

    model = _family_label(family)
    context = _positive_int(identity.get("context_limit"))
    template = _mapping(identity.get("template_kwargs"))
    if template.get("enable_thinking") is False:
        thinking = "off"
    elif isinstance(template.get("reasoning_effort"), str):
        thinking = template["reasoning_effort"]
    elif template.get("enable_thinking") is True:
        thinking = "on"
    else:
        thinking = None
    model_parts = [part for part in (
        model,
        f"context {context}" if context is not None else None,
        f"thinking {thinking}" if thinking is not None else None,
    ) if part]
    if model_parts:
        lines.append(" · ".join(model_parts))

    capacity_min, _capacity_max, capacity_label = _capacity_range(stats)
    budget = _mapping(stats.get("capacity_budget"))
    resolution = _mapping(budget.get("planner_resolution"))
    maximum = _positive_int(budget.get("max_capacity"))
    if maximum is None and family in {"qwen4_exp", "qwen4_exp_text"}:
        maximum = 512
    expert_parts = []
    if capacity_label is not None:
        suffix = f"/{maximum}" if maximum is not None else ""
        expert_parts.append(f"SSD experts {capacity_label}{suffix} per layer")
    resolved_gib = _gib(resolution.get("resolved_bytes"))
    if resolved_gib is not None:
        label = (
            "memory limit"
            if _positive_int(resolution.get("explicit_ceiling_bytes")) is not None
            else "auto memory"
        )
        expert_parts.append(f"{label} {resolved_gib:.2f} GiB")
    routing = _mapping(stats.get("cache_routing"))
    if routing.get("policy") == "prefer-resident":
        factor = routing.get("cache_factor")
        protected = routing.get("protected_routes")
        if _numeric(factor) and type(protected) is int:
            expert_parts.append(f"Cache-Prior {factor:g}/{protected}")
        else:
            expert_parts.append("Cache-Prior on")
    elif routing.get("policy") == "off":
        expert_parts.append("Cache-Prior off")
    if expert_parts:
        lines.append(" · ".join(expert_parts))

    warning = None
    condition = "normal" if diagnostics else "unavailable"
    if resolution.get("limiting_source") == "live-available" and capacity_min is not None:
        normal = _normal_planner_capacity(stats, family)
        material_loss = (
            max(CAPACITY_WARNING_MIN_SLOTS, math.ceil(normal * CAPACITY_WARNING_FRACTION))
            if normal is not None
            else None
        )
        if material_loss is not None and normal - capacity_min >= material_loss:
            condition = "memory constrained"
            warning = (
                f"WARNING: Live available memory limited the expert pool to "
                f"{capacity_label} resident experts per layer "
                f"(normal planner ceiling: about {normal}). Close memory-heavy "
                "applications such as browsers, Docker or VMs, restart MoEspresso, and "
                "rerun before sharing this result."
            )
    return {"lines": lines, "condition": condition, "warning": warning}


def _write_report(path, report):
    payload = {**report, "content_sha256": _digest(report)}
    fd, name = tempfile.mkstemp(prefix=".moespresso-diagnostic-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
    finally:
        os.unlink(name)


def _median_text(values):
    return f"{statistics.median(values):.2f}" if values else "unavailable"


def run_speed_check(client, *, max_tokens):
    """Print server decode speed and selected run conditions without writing files."""
    rows, first_times = [], []
    expected = len(SPEED_TOPICS) * 2
    code = 0
    try:
        health = client.health()
    except KeyboardInterrupt:
        print("Speed check interrupted.", file=sys.stderr)
        return 130
    except Exception:
        health = None
        print("Run conditions unavailable; continuing with timing only.", file=sys.stderr)
    environment = _speed_environment(health)
    if environment["warning"]:
        print(environment["warning"], file=sys.stderr, flush=True)

    schedule = [("warmup", "warmup", "how much is 3456+60?", min(64, max_tokens))]
    for name, topic_switch, same_topic in SPEED_TOPICS:
        schedule.extend((
            (name, "topic switch", topic_switch, max_tokens),
            (name, "same topic", same_topic, max_tokens),
        ))
    try:
        for name, locality, prompt, cap in schedule:
            print(
                "Warming up..." if name == "warmup" else f"{name}: {locality}",
                file=sys.stderr, flush=True,
            )
            response = client.complete(
                [{"role": "user", "content": prompt}], max_tokens=cap, **SAMPLING,
            )
            if name == "warmup":
                continue
            metrics = _metrics(response.usage)
            if metrics["after_first_token_tps"] is not None:
                rows.append((locality, metrics["after_first_token_tps"]))
            if metrics["first_token_seconds"] is not None:
                first_times.append(metrics["first_token_seconds"])
    except KeyboardInterrupt:
        code = 130
        print("Speed check interrupted.", file=sys.stderr)
    except Exception as exc:
        code = 1
        print(f"Speed check failed: {exc}", file=sys.stderr)
    switch_rates = [rate for locality, rate in rows if locality == "topic switch"]
    same_rates = [rate for locality, rate in rows if locality == "same topic"]
    all_rates = [rate for _locality, rate in rows]

    output = ["MoEspresso speed", *environment["lines"]]
    output.append(
        "Decode: "
        f"topic switch {_median_text(switch_rates)} · "
        f"same topic {_median_text(same_rates)} · "
        f"overall {_median_text(all_rates)} tok/s"
    )
    ttft = f"{statistics.median(first_times):.2f} s median" if first_times else "unavailable"
    output.append(
        f"TTFT: {ttft} · greedy · {len(all_rates)}/{expected} requests · "
        f"{max_tokens}-token cap"
    )
    condition = environment["condition"]
    if condition == "memory constrained":
        output.append(
            "Conditions: memory constrained; close memory-heavy applications, restart "
            "MoEspresso, and rerun"
        )
    else:
        output.append(f"Conditions: {condition}")
    print("\n".join(output), flush=True)
    return code or (0 if all_rates else 1)


def main(argv=None, *, prog="moespresso speed"):
    parser = argparse.ArgumentParser(
        prog=prog, description="Print decode speed from an existing MoEspresso server.",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="Existing server base URL.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Output cap for each measured request (default: 256).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout for each HTTP request in seconds (default: 300).",
    )
    parser.add_argument(
        "--stream", action="store_true", help="Test SSE instead of buffered replies."
    )
    args = parser.parse_args(argv)
    url = urlsplit(args.url)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path not in {"", "/"}
    ):
        parser.error("url must be an HTTP(S) server origin without credentials, query or path")
    if not 2 <= args.max_tokens <= 2048:
        parser.error("max-tokens must be 2..2048")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive and finite")
    try:
        return run_speed_check(
            CompletionsClient(args.url, timeout=args.timeout, stream=args.stream),
            max_tokens=args.max_tokens,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
