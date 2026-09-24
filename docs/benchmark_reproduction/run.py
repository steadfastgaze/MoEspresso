"""Generate one answer per frozen question against one chat-completions endpoint."""

import argparse
import fcntl
import json
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from common import (
    PACKAGE_ID,
    SELECTION_SHA256,
    answer_fields,
    checked_item,
    digest,
    load_questions,
    read_json,
    save_json,
)


def server_settings(client, base_url, required=False):
    try:
        response = client.get(base_url.removesuffix("/v1") + "/health", timeout=10)
        response.raise_for_status()
        health = response.json()
        identity = health["diagnostics"]["identity"]
        stats = health["ssd_streaming"]
        routing = stats.get("cache_routing", {})
        return {"package_id": identity.get("package_artifact_id"),
                "context_limit": identity.get("context_limit"),
                "thinking": identity.get("template_kwargs"),
                "versions": identity.get("versions"),
                "capacity": stats.get("capacity_per_layer"),
                "capacity_overrides": stats.get("capacity_overrides", {}),
                "cache_prior": {key: routing.get(key) for key in
                                ("policy", "cache_factor", "protected_routes")}}
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        if required:
            raise ValueError("required MoEspresso /health settings are unavailable") from None
        return None


def check_settings(settings, args):
    if args.require_capacity is not None:
        capacities = {settings["capacity"], *settings["capacity_overrides"].values()}
        if capacities != {args.require_capacity}:
            raise ValueError("server capacity differs from the required value")
    if args.require_cache_prior and settings["cache_prior"] != {
        "policy": "prefer-resident", "cache_factor": 2, "protected_routes": 2
    }:
        raise ValueError("server must use Cache-Prior factor 2 and protected routes 2")
    if args.require_package and settings["package_id"] != args.require_package:
        raise ValueError("server package ID differs")
    if settings and (settings["thinking"] or {}).get("enable_thinking") is False:
        raise ValueError("server thinking is disabled")


def read_stream(response):
    message = {"content": "", "reasoning_content": ""}
    result = {"choices": [{"message": message, "finish_reason": None}]}
    done = False
    for line in response.iter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        chunk = json.loads(data)
        if chunk.get("error"):
            raise ValueError("provider reported an error during streaming")
        result.update({key: value for key, value in chunk.items()
                       if key != "choices" and value is not None})
        for choice in chunk.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ValueError("stream contains more than one choice")
            delta = choice.get("delta", {})
            if delta.get("tool_calls") or delta.get("function_call"):
                raise ValueError("provider invoked tools")
            message["content"] += delta.get("content") or ""
            reasoning = delta.get("reasoning_content", delta.get("reasoning")) or ""
            message["reasoning_content"] += reasoning
            if choice.get("finish_reason"):
                result["choices"][0]["finish_reason"] = choice["finish_reason"]
    if not done:
        raise ValueError("stream ended before [DONE]; request outcome is uncertain")
    return result


def run_question(client, root, question, protocol, timeout):
    question_id = question["question_id"]
    item_path = root / f"{question_id}.json"
    if item_path.exists():
        checked_item(item_path, question, protocol)
        return
    marker = root / f"{question_id}.pending.json"
    response_path = root / f"{question_id}.response.json"
    request = {**protocol["request"],
               "messages": [{"role": "user", "content": question["turns"][0]}]}
    identity = {"protocol_sha256": digest(protocol), "question_sha256": digest(question),
                "request": request}
    elapsed = None
    if response_path.exists():
        if not marker.exists() or read_json(marker) != identity:
            raise ValueError("saved response lacks a compatible request marker")
        result = read_json(response_path)
    else:
        if marker.exists():
            raise ValueError(f"uncertain call: inspect provider activity before moving {marker}")
        save_json(marker, identity)
        started = time.monotonic()
        with client.stream("POST", protocol["base_url"] + "/chat/completions",
                           json=request, timeout=timeout) as response:
            if response.status_code in (400, 401, 402, 403, 404, 422, 429):
                marker.unlink()
            response.raise_for_status()
            result = read_stream(response) if request["stream"] else json.loads(response.read())
        elapsed = time.monotonic() - started
        save_json(response_path, result)
    answer = answer_fields(result, request[protocol["token_field"]])
    save_json(item_path, {"protocol_sha256": digest(protocol), "question_id": question_id,
                         "question_sha256": digest(question), "response": result,
                         "answer": answer, "seconds": elapsed})
    marker.unlink()
    print(f"{question['category']}/{question['task']}: {answer['finish_reason']}", flush=True)


def run(args):
    base_url = args.base_url.rstrip("/")
    parsed = urlsplit(base_url)
    local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment or
            (parsed.scheme == "http" and not local)):
        raise ValueError("use an HTTP(S) base URL without credentials; remote APIs require HTTPS")
    if args.max_output_tokens <= 0 or args.timeout <= 0:
        raise ValueError("output ceiling and timeout must be positive")
    key = os.environ.get(args.api_key_env)
    if not local and not key:
        raise ValueError(f"set {args.api_key_env} before calling the remote API")
    request = {"model": args.model, "reasoning_effort": args.reasoning_effort,
               args.token_field: args.max_output_tokens,
               "tool_choice": "none", "stream": args.stream}
    if local or args.require_package:
        request["chat_template_kwargs"] = {"enable_thinking": True}
    if args.sampling == "fixed":
        request.update(temperature=0, top_p=1, top_k=0, min_p=0, presence_penalty=0)
    if args.openrouter or parsed.hostname == "openrouter.ai":
        request.update(plugins=[{"id": "web", "enabled": False}],
                       provider={"require_parameters": True}, usage={"include": True})
    if args.stream:
        request["stream_options"] = {"include_usage": True}
    required = bool(args.require_capacity or args.require_cache_prior or args.require_package)
    questions = load_questions(args.offline)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".lock").open("w") as lock, httpx.Client(
        headers={"Authorization": f"Bearer {key}"} if key else {}, follow_redirects=False
    ) as client:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        settings = server_settings(client, base_url, required) if local or required else None
        check_settings(settings, args)
        protocol = {"version": 1, "selection_sha256": SELECTION_SHA256,
                    "base_url": base_url, "request": request, "token_field": args.token_field,
                    "thinking_enabled": True, "sampling": args.sampling, "server": settings}
        path = args.out / "protocol.json"
        if path.exists():
            if read_json(path) != protocol:
                raise ValueError("run settings changed; choose a new output directory")
        else:
            if any(p.name != ".lock" for p in args.out.iterdir()):
                raise ValueError("output directory has files but no protocol.json")
            save_json(path, protocol)
        save_json(args.out / "questions.json", questions)
        for question in questions:
            if settings is not None and server_settings(client, base_url, True) != settings:
                raise ValueError("server settings changed during the run")
            run_question(client, args.out, question, protocol, args.timeout)
    print("Completed 48 questions.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="API base including /v1 when required")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--reasoning-effort", default="medium",
                        choices=["low", "medium", "high", "xhigh"])
    parser.add_argument("--sampling", choices=["fixed", "provider"], default="fixed")
    parser.add_argument("--max-output-tokens", type=int, default=131072)
    parser.add_argument("--token-field", default="max_tokens",
                        choices=["max_tokens", "max_completion_tokens"])
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--openrouter", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--require-capacity", type=int)
    parser.add_argument("--require-cache-prior", action="store_true", help="require 2/2 policy")
    parser.add_argument("--require-package", nargs="?", const=PACKAGE_ID)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
