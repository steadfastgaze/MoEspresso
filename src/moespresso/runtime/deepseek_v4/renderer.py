"""DeepSeek-V4 chat prompt renderer.

DeepSeek-V4 ships a Python encoder rather than a Jinja chat template. This module
ports the prompt-construction surface MoEspresso needs at serve time: chat mode,
thinking mode, reasoning effort, tool schemas, tool calls, and merged tool
results. It intentionally does not parse completions.
"""

from __future__ import annotations

import copy
import json
from typing import Any

DEEPSEEK_V4_PROMPT_RENDERER = "deepseek_v4_dsv4"
DEEPSEEK_V4_RENDERER_VERSION = "deepseek_v4_dsv4:1"

BOS_TOKEN = "<｜begin▁of▁sentence｜>"
EOS_TOKEN = "<｜end▁of▁sentence｜>"
THINKING_START_TOKEN = "<think>"
THINKING_END_TOKEN = "</think>"
DSML_TOKEN = "｜DSML｜"
USER_SP_TOKEN = "<｜User｜>"
ASSISTANT_SP_TOKEN = "<｜Assistant｜>"
LATEST_REMINDER_SP_TOKEN = "<｜latest_reminder｜>"

DS_TASK_SP_TOKENS = {
    "action": "<｜action｜>",
    "query": "<｜query｜>",
    "authority": "<｜authority｜>",
    "domain": "<｜domain｜>",
    "title": "<｜title｜>",
    "read_url": "<｜read_url｜>",
}

REASONING_EFFORT_MAX = (
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem "
    "to resolve the root cause, rigorously stress-testing your logic against all potential "
    "paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate "
    "step, considered alternative, and rejected hypothesis to ensure absolutely no "
    "assumption is left unchecked.\n\n"
)

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}tool_calls>" block like the following:

<{dsml_token}tool_calls>
<{dsml_token}invoke name="$TOOL_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$TOOL_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


def _to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _tools_from_openai_format(tools: list[dict]) -> list[dict]:
    return [tool["function"] for tool in tools]


def _tool_calls_from_openai_format(tool_calls: list[dict]) -> list[dict]:
    return [
        {
            "name": tool_call["function"]["name"],
            "arguments": tool_call["function"]["arguments"],
        }
        for tool_call in tool_calls
    ]


def encode_arguments_to_dsml(tool_call: dict[str, str]) -> str:
    """Encode OpenAI function arguments into DSML parameter tags."""
    try:
        arguments = json.loads(tool_call["arguments"])
    except Exception:
        arguments = {"arguments": tool_call["arguments"]}

    parts = []
    for key, value in arguments.items():
        is_str = isinstance(value, str)
        rendered = value if is_str else _to_json(value)
        parts.append(
            f'<{DSML_TOKEN}parameter name="{key}" string="{str(is_str).lower()}">'
            f"{rendered}</{DSML_TOKEN}parameter>"
        )
    return "\n".join(parts)


def render_tools(tools: list[dict]) -> str:
    return TOOLS_TEMPLATE.format(
        tool_schemas="\n".join(_to_json(t) for t in tools),
        dsml_token=DSML_TOKEN,
        thinking_start_token=THINKING_START_TOKEN,
        thinking_end_token=THINKING_END_TOKEN,
    )


def find_last_user_index(messages: list[dict]) -> int:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") in {"user", "developer"}:
            return idx
    return -1


def _attach_request_tools(
    messages: list[dict],
    tools: list[dict] | None,
    response_format: dict | None,
) -> list[dict]:
    if not tools and not response_format:
        return [copy.deepcopy(m) for m in messages]

    out = [copy.deepcopy(m) for m in messages]
    target = next((m for m in out if m.get("role") == "system"), None)
    if target is None:
        target = {"role": "system", "content": ""}
        out.insert(0, target)
    if tools:
        target["tools"] = tools
    if response_format:
        target["response_format"] = response_format
    return out


def merge_tool_messages(messages: list[dict]) -> list[dict]:
    """Merge OpenAI tool messages into DS4 user content blocks."""
    merged: list[dict] = []
    for original in messages:
        msg = copy.deepcopy(original)
        role = msg.get("role")
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            if merged and merged[-1].get("role") == "user" and "content_blocks" in merged[-1]:
                merged[-1]["content_blocks"].append(block)
            else:
                merged.append({"role": "user", "content_blocks": [block]})
        elif role == "user":
            text_block = {"type": "text", "text": msg.get("content", "")}
            can_merge = (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
                and merged[-1].get("task") is None
            )
            if can_merge:
                merged[-1]["content_blocks"].append(text_block)
            else:
                new_msg = {
                    "role": "user",
                    "content": msg.get("content", ""),
                    "content_blocks": [text_block],
                }
                for key in ("task", "wo_eos", "mask"):
                    if key in msg:
                        new_msg[key] = msg[key]
                merged.append(new_msg)
        else:
            merged.append(msg)
    return merged


def sort_tool_results_by_call_order(messages: list[dict]) -> list[dict]:
    last_tool_call_order: dict[str, int] = {}
    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            last_tool_call_order = {}
            for idx, tool_call in enumerate(msg["tool_calls"]):
                tool_call_id = tool_call.get("id") or tool_call.get("function", {}).get("id", "")
                if tool_call_id:
                    last_tool_call_order[tool_call_id] = idx
        elif role == "user" and msg.get("content_blocks"):
            tool_blocks = [b for b in msg["content_blocks"] if b.get("type") == "tool_result"]
            if len(tool_blocks) > 1 and last_tool_call_order:
                sorted_blocks = sorted(
                    tool_blocks,
                    key=lambda b: last_tool_call_order.get(b.get("tool_use_id", ""), 0),
                )
                sorted_idx = 0
                new_blocks = []
                for block in msg["content_blocks"]:
                    if block.get("type") == "tool_result":
                        new_blocks.append(sorted_blocks[sorted_idx])
                        sorted_idx += 1
                    else:
                        new_blocks.append(block)
                msg["content_blocks"] = new_blocks
    return messages


def _drop_thinking_messages(messages: list[dict]) -> list[dict]:
    last_user_idx = find_last_user_index(messages)
    result = []
    keep_roles = {"user", "system", "tool", "latest_reminder", "direct_search_results"}
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role in keep_roles or idx >= last_user_idx:
            result.append(msg)
        elif role == "assistant":
            next_msg = copy.copy(msg)
            next_msg.pop("reasoning_content", None)
            result.append(next_msg)
    return result


def _render_content_blocks(msg: dict) -> str:
    blocks = msg.get("content_blocks") or []
    parts = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            parts.append(block.get("text", ""))
        elif block_type == "tool_result":
            content = block.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    else:
                        text_parts.append(f"[Unsupported {item.get('type')}]")
                content = "\n\n".join(text_parts)
            parts.append(f"<tool_result>{content}</tool_result>")
        else:
            parts.append(f"[Unsupported {block_type}]")
    return "\n\n".join(parts)


def _render_tool_calls(tool_calls: list[dict]) -> str:
    converted = _tool_calls_from_openai_format(tool_calls)
    calls = [
        f'<{DSML_TOKEN}invoke name="{tc.get("name")}">\n'
        f"{encode_arguments_to_dsml(tc)}\n"
        f"</{DSML_TOKEN}invoke>"
        for tc in converted
    ]
    rendered_calls = "\n".join(calls)
    return (
        f"\n\n<{DSML_TOKEN}tool_calls>\n"
        f"{rendered_calls}\n"
        f"</{DSML_TOKEN}tool_calls>"
    )


def render_message(
    index: int,
    messages: list[dict],
    *,
    thinking_mode: str,
    drop_thinking: bool = True,
    reasoning_effort: str | None = None,
) -> str:
    if thinking_mode not in {"chat", "thinking"}:
        raise ValueError(f"invalid thinking_mode {thinking_mode!r}")
    if reasoning_effort not in {"max", "high", None}:
        raise ValueError(f"invalid reasoning_effort {reasoning_effort!r}")

    msg = messages[index]
    role = msg.get("role")
    prompt = ""
    last_user_idx = find_last_user_index(messages)

    if index == 0 and thinking_mode == "thinking" and reasoning_effort == "max":
        prompt += REASONING_EFFORT_MAX

    if role == "system":
        prompt += msg.get("content") or ""
        tools = msg.get("tools")
        if tools:
            prompt += "\n\n" + render_tools(_tools_from_openai_format(tools))
        if msg.get("response_format"):
            prompt += "\n\n## Response Format:\n\n"
            prompt += "You MUST strictly adhere to the following schema to reply:\n"
            prompt += _to_json(msg["response_format"])

    elif role == "developer":
        content = msg.get("content")
        if not content:
            raise ValueError("developer messages require content")
        prompt += USER_SP_TOKEN + content
        tools = msg.get("tools")
        if tools:
            prompt += "\n\n" + render_tools(_tools_from_openai_format(tools))
        if msg.get("response_format"):
            prompt += "\n\n## Response Format:\n\n"
            prompt += "You MUST strictly adhere to the following schema to reply:\n"
            prompt += _to_json(msg["response_format"])

    elif role == "user":
        prompt += USER_SP_TOKEN
        prompt += _render_content_blocks(msg) if msg.get("content_blocks") else (msg.get("content") or "")

    elif role == "latest_reminder":
        prompt += LATEST_REMINDER_SP_TOKEN + (msg.get("content") or "")

    elif role == "tool":
        raise NotImplementedError("DS4 merges tool messages into user content blocks")

    elif role == "assistant":
        tool_calls = _render_tool_calls(msg["tool_calls"]) if msg.get("tool_calls") else ""
        reasoning = ""
        previous_has_task = index - 1 >= 0 and messages[index - 1].get("task") is not None
        if thinking_mode == "thinking" and not previous_has_task:
            if not drop_thinking or index > last_user_idx:
                reasoning = (msg.get("reasoning_content") or "") + THINKING_END_TOKEN
        rendered = f"{reasoning}{msg.get('content') or ''}{tool_calls}"
        prompt += rendered if msg.get("wo_eos", False) else rendered + EOS_TOKEN

    else:
        raise NotImplementedError(f"unknown DS4 role {role!r}")

    if index + 1 < len(messages) and messages[index + 1].get("role") not in {
        "assistant",
        "latest_reminder",
    }:
        return prompt

    task = msg.get("task")
    if task is not None:
        if task not in DS_TASK_SP_TOKENS:
            raise ValueError(f"invalid task {task!r}")
        if task != "action":
            prompt += DS_TASK_SP_TOKENS[task]
        else:
            prompt += ASSISTANT_SP_TOKEN
            prompt += THINKING_START_TOKEN if thinking_mode == "thinking" else THINKING_END_TOKEN
            prompt += DS_TASK_SP_TOKENS[task]
    elif role in {"user", "developer"}:
        prompt += ASSISTANT_SP_TOKEN
        if thinking_mode == "thinking" and (not drop_thinking or index >= last_user_idx):
            prompt += THINKING_START_TOKEN
        else:
            prompt += THINKING_END_TOKEN
    return prompt


def encode_messages(
    messages: list[dict],
    *,
    thinking_mode: str,
    context: list[dict] | None = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: str | None = None,
) -> str:
    context = context or []
    messages = merge_tool_messages(messages)
    messages = sort_tool_results_by_call_order(context + messages)[len(context):]
    if context:
        context = sort_tool_results_by_call_order(merge_tool_messages(context))

    full_messages = context + messages
    effective_drop_thinking = drop_thinking
    if any(m.get("tools") for m in full_messages):
        effective_drop_thinking = False

    if thinking_mode == "thinking" and effective_drop_thinking:
        full_messages = _drop_thinking_messages(full_messages)
        context_len = len(full_messages) - (len(messages))
        num_to_render = len(full_messages) - len(_drop_thinking_messages(context))
    else:
        context_len = len(context)
        num_to_render = len(messages)

    prompt = BOS_TOKEN if add_default_bos_token and len(context) == 0 else ""
    for idx in range(num_to_render):
        prompt += render_message(
            idx + context_len,
            full_messages,
            thinking_mode=thinking_mode,
            drop_thinking=effective_drop_thinking,
            reasoning_effort=reasoning_effort,
        )
    return prompt


def render_deepseek_v4_prompt(
    messages: list[dict],
    *,
    template_kwargs: dict | None = None,
    tools: list[dict] | None = None,
    response_format: dict | None = None,
) -> str:
    """Render OpenAI-style chat messages with DS4's Python encoder semantics."""
    kwargs = template_kwargs or {}
    thinking_mode = "thinking" if kwargs.get("enable_thinking", True) else "chat"
    preserve_thinking = bool(kwargs.get("preserve_thinking", False))
    drop_thinking = bool(kwargs.get("drop_thinking", not preserve_thinking))
    prepared = _attach_request_tools(messages, tools, response_format)
    return encode_messages(
        prepared,
        thinking_mode=thinking_mode,
        drop_thinking=drop_thinking,
        reasoning_effort=kwargs.get("reasoning_effort"),
    )
