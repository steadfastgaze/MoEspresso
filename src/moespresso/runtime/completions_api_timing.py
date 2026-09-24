"""Client-clock timing and independent token counts for chat-completions APIs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics
import time
from urllib.parse import urlsplit

from moespresso.agentlib.client import ClientError, CompletionsClient
from moespresso.runtime.diagnostics import CASES, _digest, _write_report
from moespresso.runtime.timing_tokens import LocalTokenCounter


SAMPLING = {"temperature": 0.0, "top_p": 1.0}


class StreamClock:
    """Collect delivery timestamps without tokenization, output or extra requests."""

    def __init__(self, started, clock=time.perf_counter):
        self.started = started
        self.clock = clock
        self.headers_seconds = None
        self.finished_choice_seconds = None
        self.events = []
        self.server_timings = []
        self.chunks = 0

    def headers(self):
        self.headers_seconds = self.clock() - self.started

    def chunk(self, event):
        now = self.clock() - self.started
        self.chunks += 1
        if "timings" in event:
            self.server_timings.append(event["timings"])
        choices = event.get("choices") or []
        if not choices:
            return
        choice = choices[0]
        if choice.get("finish_reason") is not None:
            self.finished_choice_seconds = now
        delta = choice.get("delta") or {}
        content = delta.get("content") or ""
        reasoning = delta.get("reasoning_content", delta.get("reasoning")) or ""
        if content or reasoning:
            if not isinstance(content, str) or not isinstance(reasoning, str):
                raise ValueError("completion stream contains non-text output")
            self.events.append({"seconds": now, "content": content, "reasoning": reasoning})


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _rate(tokens, seconds):
    return tokens / seconds if tokens is not None and tokens > 0 and seconds is not None and seconds > 0 else None


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def calculate_metrics(prompt, output, observer, elapsed, usage):
    boundaries = output["delivery_boundaries"]
    first = boundaries[0]["seconds"] if boundaries else None
    last = boundaries[-1]["seconds"] if boundaries else None
    span = last - first if first is not None else None
    first_tokens = boundaries[0]["tokens"] if boundaries else 0
    gaps = [b["seconds"] - a["seconds"] for a, b in zip(boundaries, boundaries[1:])]
    api_prompt = _count(usage.get("prompt_tokens"))
    api_output = _count(usage.get("completion_tokens"))
    return {
        "request_seconds": elapsed,
        "response_headers_seconds": observer.headers_seconds,
        "first_output_seconds": first,
        "last_output_seconds": last,
        "finish_event_seconds": observer.finished_choice_seconds,
        "completion_tail_seconds": elapsed - last if last is not None else None,
        "stream_chunks": observer.chunks,
        "text_deliveries": len(boundaries),
        "inter_delivery_p50_seconds": _percentile(gaps, 0.5),
        "inter_delivery_p95_seconds": _percentile(gaps, 0.95),
        "first_delivery_tokens": first_tokens,
        "local_output_tps": _rate(output["tokens"], elapsed),
        "local_after_first_delivery_tps_estimate": _rate(output["tokens"] - first_tokens, span),
        "local_prompt_to_first_output_tps_estimate": _rate(prompt["tokens"], first),
        "api_count_output_tps": _rate(api_output, elapsed),
        "api_minus_local_prompt_tokens": api_prompt - prompt["tokens"] if api_prompt is not None else None,
        "api_minus_local_output_tokens": api_output - output["tokens"] if api_output is not None else None,
    }


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_nonfinite_number": str(value)}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _summary(rows):
    result = {}
    for name in dict.fromkeys(row["case"] for row in rows if not row["warmup"]):
        measured = [row for row in rows if row["case"] == name and not row["warmup"]
                    and row["status"] == "completed"]
        medians = {}
        for key in ("request_seconds", "first_output_seconds", "local_output_tps",
                    "local_after_first_delivery_tps_estimate"):
            values = [row["metrics"][key] for row in measured if row["metrics"][key] is not None]
            medians[key] = {"median": statistics.median(values) if values else None,
                            "samples": len(values)}
        result[name] = medians
    return result


def run_timing(client, counter, *, repeats, max_tokens, output, cases=CASES,
               label="", template_kwargs=None, clock=time.perf_counter):
    """Measure serial requests; all tokenization occurs outside each request."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not client.stream:
        raise ValueError("API timing requires a streamed response")
    if not output.parent.is_dir():
        raise ValueError(f"output directory does not exist: {output.parent}")
    template_kwargs = template_kwargs or {}
    report = {
        "schema": "completions-api-timing-v1", "status": "running",
        "server_url": client.base_url, "model": client.model, "label": label,
        "tokenizer": counter.identity, "sampling": SAMPLING,
        "chat_template_kwargs": template_kwargs, "cases": list(cases),
        "suite_sha256": _digest(cases), "max_tokens": max_tokens, "repeats": repeats,
        "requests": [], "compatibility": [],
        "measurement_contract": {
            "clock": "client monotonic perf_counter", "health_requests": 0,
            "tokenization": "always local, before prompts and after completed responses",
            "server_counts": "cross-check only; never replace independent counts",
            "first_output": "first nonempty reasoning/content delivery, not first internal token",
            "decode_estimate": "local tokens after first delivery divided by first-to-last delivery time",
            "prompt_estimate": "includes queueing, transport, caching, prefill and first delivery",
        },
        "warnings": [
            "Retokenization counts visible returned text, not hidden or stripped generated tokens.",
            "Local prompt counts depend on matching the server's tokenizer and chat template.",
            "Streaming may buffer multiple tokens; delivery gaps are not individual token latencies.",
            "Server configuration, thinking mode, drafting and caches are not changed.",
            "Use the same client measurements on every engine; this is not a quality benchmark.",
        ],
    }
    code = 0
    observer = row = None
    started = None
    try:
        schedule = [("warmup", "how much is 3456+60?", 0, min(64, max_tokens))]
        schedule.extend((name, text, repeat, max_tokens)
                        for repeat in range(1, repeats + 1) for name, text in cases)
        # Validate and count every distinct prompt before sending any request.
        prompts = {text: counter.prompt([{"role": "user", "content": text}], template_kwargs)
                   for _name, text, _repeat, _cap in schedule}
        for name, text, repeat, cap in schedule:
            row = {"case": name, "repeat": repeat, "warmup": repeat == 0,
                   "messages": [{"role": "user", "content": text}], "max_tokens": cap,
                   "local_prompt": prompts[text], "status": "running"}
            report["requests"].append(row)
            started = clock()
            observer = StreamClock(started, clock)

            def request():
                return client.complete(
                    row["messages"], max_tokens=cap, **SAMPLING,
                    chat_template_kwargs=template_kwargs or None,
                    on_start=observer.headers, on_chunk=observer.chunk,
                )

            try:
                response = request()
            except ClientError as error:
                if not (repeat == 0 and error.status in (400, 422)
                        and client.include_stream_usage and not observer.events
                        and any(word in error.message.lower() for word in ("stream_options", "include_usage"))):
                    raise
                report["compatibility"].append({
                    "action": "omit unsupported stream_options on subsequent requests",
                    "rejected_warmup_seconds": clock() - started, "error": str(error),
                })
                client.include_stream_usage = False
                started = clock()
                observer = StreamClock(started, clock)
                response = request()
            elapsed = clock() - started
            row.update(response=response.message, usage=response.usage, finish_reason=response.finish_reason,
                       server_timings=observer.server_timings)
            counted = counter.output(observer.events)
            usage = response.usage if isinstance(response.usage, dict) else {}
            row.update(status="completed", local_output=counted,
                       metrics=calculate_metrics(prompts[text], counted, observer, elapsed, usage))
            row["warnings"] = []
            for field in ("prompt_tokens", "completion_tokens"):
                if _count(usage.get(field)) is None:
                    row["warnings"].append(f"API {field} missing or invalid; independent count retained.")
            if any(row["metrics"][field] not in (None, 0) for field in (
                "api_minus_local_prompt_tokens", "api_minus_local_output_tokens",
            )):
                row["warnings"].append("API and local counts differ; inspect tokenizer, template and hidden-token conventions.")
            rate = row["metrics"]["local_after_first_delivery_tps_estimate"]
            rate_text = f"{rate:.2f} post-first-delivery tok/s estimate" if rate is not None else "decode estimate unavailable (insufficient deliveries)"
            overall = row["metrics"]["local_output_tps"]
            overall_text = f"{overall:.2f} request tok/s" if overall is not None else "no visible output tokens"
            print(f"{name} {repeat}: {counted['tokens']} local output tokens; {overall_text}; "
                  f"{rate_text}; {elapsed:.3f}s request", flush=True)
        report["status"] = "completed"
    except KeyboardInterrupt:
        code, report["status"] = 130, "interrupted"
    except Exception as error:  # noqa: BLE001 - preserve partial measurement evidence
        code, report["status"] = 1, "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if row is not None and row["status"] == "running":
            row["status"] = report["status"]
            if observer is not None:
                row["partial_text_events"] = observer.events
                row["client_elapsed_seconds"] = clock() - started
                try:
                    row["partial_local_output"] = counter.output(observer.events)
                except Exception as error:  # noqa: BLE001 - retain original failure
                    row["count_error"] = str(error)
        report["summary"] = _summary(report["requests"])
        _write_report(output, _json_safe(report))
        print(f"API timing {report['status']}: {output}", flush=True)
    return code


def main(argv=None, *, prog="moespresso completions-api-timing"):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", help="Model name expected by the server.")
    parser.add_argument("--tokenizer", required=True,
                        help="Matching local tokenizer directory or already-cached Hugging Face identifier.")
    parser.add_argument("--chat-template", type=Path, help="Local template override matching the server.")
    parser.add_argument("--chat-template-kwargs", default="{}", help="JSON template options applied locally and sent to the server.")
    parser.add_argument("--output", type=Path, default=Path(time.strftime("completions-api-timing-%Y%m%d-%H%M%S.json")))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--label", default="")
    parser.add_argument("--prompt", action="append", help="Use custom prompts instead of the three default cases.")
    args = parser.parse_args(argv)
    import json

    try:
        url = urlsplit(args.url)
        if (url.scheme not in ("http", "https") or not url.hostname or url.username or url.password
                or url.query or url.fragment or url.path not in ("", "/", "/v1", "/v1/")):
            raise ValueError("url must be a server origin or its /v1 URL, without credentials or query")
        if not 1 <= args.repeats <= 5 or not 2 <= args.max_tokens <= 2048:
            raise ValueError("repeats must be 1..5 and max-tokens must be 2..2048")
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if args.output.exists() or not args.output.parent.is_dir():
            raise ValueError("output must be a new file in an existing directory")
        options = json.loads(args.chat_template_kwargs)
        if not isinstance(options, dict) or any(key in options for key in (
            "tokenize", "add_generation_prompt", "return_dict", "return_tensors", "tools",
            "return_assistant_tokens_mask", "continue_final_message", "padding", "truncation",
            "max_length", "tokenizer_kwargs", "chat_template", "conversation",
        )):
            raise ValueError("template kwargs must be a JSON object without tokenizer control fields")
        counter = LocalTokenCounter.load(args.tokenizer, chat_template=args.chat_template)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    cases = tuple((f"prompt-{i}", text) for i, text in enumerate(args.prompt, 1)) if args.prompt else CASES
    return run_timing(
        CompletionsClient(f"{url.scheme}://{url.netloc}", model=args.model, timeout=args.timeout),
        counter, repeats=args.repeats, max_tokens=args.max_tokens, output=args.output,
        cases=cases, label=args.label, template_kwargs=options,
    )


if __name__ == "__main__":
    raise SystemExit(main())
