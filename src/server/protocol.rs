//! Checked Chat Completions request fields before model state is touched.

use serde_json::{Value, json};

use crate::sampling::SamplingOptions;
use crate::tokenizer::{ChatOptions, ReasoningEffort};

#[derive(Debug)]
pub(super) struct ApiError {
    pub status: u16,
    pub message: String,
    pub code: Option<&'static str>,
}

impl ApiError {
    pub fn bad(message: impl Into<String>) -> Self {
        Self {
            status: 400,
            message: message.into(),
            code: None,
        }
    }

    pub fn body(&self) -> Value {
        json!({"error": {"message": self.message, "type": "invalid_request_error", "code": self.code}})
    }
}

type Checked<T> = Result<T, ApiError>;

pub(super) struct ChatRequest {
    pub model: String,
    pub messages: Vec<Value>,
    pub tools: Option<Vec<Value>>,
    pub chat: ChatOptions,
    pub sampling: SamplingOptions,
    pub max_tokens: usize,
    pub seed: Option<u64>,
    pub stream: bool,
    pub include_usage: bool,
}

fn object<'a>(value: &'a Value, name: &str) -> Checked<&'a serde_json::Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| ApiError::bad(format!("{name} must be a JSON object")))
}

fn number(value: &Value, name: &str) -> Checked<f64> {
    let result = value.as_f64().filter(|v| v.is_finite());
    result.ok_or_else(|| ApiError::bad(format!("{name} must be a finite number")))
}

fn unsigned(value: &Value, name: &str) -> Checked<u64> {
    value
        .as_u64()
        .ok_or_else(|| ApiError::bad(format!("{name} must be a nonnegative integer")))
}

fn text_content(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::Null => true,
        Value::Array(parts) => parts.iter().all(|part| {
            part.get("type").and_then(Value::as_str) == Some("text")
                && part.get("text").is_some_and(Value::is_string)
        }),
        _ => false,
    }
}

fn normalize_messages(value: &Value) -> Checked<Vec<Value>> {
    let messages = value
        .as_array()
        .filter(|m| !m.is_empty())
        .ok_or_else(|| ApiError::bad("request must include a non-empty 'messages' list"))?;
    let mut normalized = Vec::with_capacity(messages.len());
    for (index, entry) in messages.iter().enumerate() {
        let mut message = object(entry, "message")?.clone();
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| ApiError::bad(format!("messages[{index}].role must be a string")))?;
        if !matches!(role, "system" | "developer" | "user" | "assistant" | "tool") {
            return Err(ApiError::bad(format!(
                "messages[{index}] has unsupported role"
            )));
        }
        let calls = message.get("tool_calls").filter(|v| !v.is_null()).cloned();
        if calls.is_some() && role != "assistant" {
            return Err(ApiError::bad(
                "tool_calls are valid only on assistant messages",
            ));
        }
        let content = message.get("content");
        if content.is_none() && calls.is_none() {
            return Err(ApiError::bad(format!("messages[{index}] needs content")));
        }
        if content.is_some_and(|v| !text_content(v)) {
            return Err(ApiError::bad(format!(
                "messages[{index}] supports only text content"
            )));
        }
        if content.is_some_and(Value::is_null) && !(role == "assistant" && calls.is_some()) {
            return Err(ApiError::bad(format!(
                "messages[{index}] needs text content"
            )));
        }
        if let Some(calls) = calls {
            let calls = calls
                .as_array()
                .ok_or_else(|| ApiError::bad("assistant tool_calls must be a list"))?;
            let mut converted = Vec::with_capacity(calls.len());
            for (call_index, call) in calls.iter().enumerate() {
                let mut call = object(call, "tool call")?.clone();
                let function = call
                    .get_mut("function")
                    .and_then(Value::as_object_mut)
                    .ok_or_else(|| {
                        ApiError::bad(format!("tool_calls[{call_index}] needs function"))
                    })?;
                if function
                    .get("name")
                    .and_then(Value::as_str)
                    .is_none_or(str::is_empty)
                {
                    return Err(ApiError::bad(format!(
                        "tool_calls[{call_index}] needs a function name"
                    )));
                }
                match function.get("arguments") {
                    Some(Value::String(text)) => {
                        let decoded: Value = serde_json::from_str(text).map_err(|_| {
                            ApiError::bad(format!(
                                "tool_calls[{call_index}] arguments must be JSON"
                            ))
                        })?;
                        if !decoded.is_object() {
                            return Err(ApiError::bad(format!(
                                "tool_calls[{call_index}] arguments must decode to an object"
                            )));
                        }
                        function.insert("arguments".into(), decoded);
                    }
                    Some(Value::Object(_)) | None => {}
                    _ => {
                        return Err(ApiError::bad(format!(
                            "tool_calls[{call_index}] arguments must be an object or JSON string"
                        )));
                    }
                }
                converted.push(Value::Object(call));
            }
            message.insert("tool_calls".into(), Value::Array(converted));
        }
        normalized.push(Value::Object(message));
    }
    Ok(normalized)
}

fn validate_tools(value: Option<&Value>, choice: Option<&Value>) -> Checked<Option<Vec<Value>>> {
    let choice = match choice {
        None | Some(Value::Null) => "auto",
        Some(Value::String(choice)) => choice.as_str(),
        _ => return Err(ApiError::bad("tool_choice supports only 'auto' and 'none'")),
    };
    if !matches!(choice, "auto" | "none") {
        return Err(ApiError::bad("tool_choice supports only 'auto' and 'none'"));
    }
    let Some(value) = value.filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let tools = value
        .as_array()
        .ok_or_else(|| ApiError::bad("tools must be a list"))?;
    let mut validated = Vec::with_capacity(tools.len());
    for (index, tool) in tools.iter().enumerate() {
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            return Err(ApiError::bad(format!("tools[{index}] must be a function")));
        }
        let function = tool
            .get("function")
            .ok_or_else(|| ApiError::bad(format!("tools[{index}] needs function")))?;
        let name = function.get("name").and_then(Value::as_str).unwrap_or("");
        if name.is_empty() {
            return Err(ApiError::bad(format!(
                "tools[{index}] needs a function name"
            )));
        }
        if let Some(parameters) = function.get("parameters") {
            let schema = object(parameters, "function.parameters")?;
            if let Some(properties) = schema.get("properties") {
                let properties = object(properties, "function.parameters.properties")?;
                for (property, value) in properties {
                    let declaration = object(value, "tool property")?.get("type");
                    if let Some(declaration) = declaration {
                        let types: Vec<&str> = match declaration {
                            Value::String(s) => vec![s],
                            Value::Array(values) if !values.is_empty() => {
                                values.iter().map(|v| v.as_str().unwrap_or("")).collect()
                            }
                            _ => {
                                return Err(ApiError::bad(format!(
                                    "tools[{index}] property {property} has invalid type"
                                )));
                            }
                        };
                        if types.iter().any(|kind| {
                            !matches!(
                                *kind,
                                "array"
                                    | "boolean"
                                    | "integer"
                                    | "null"
                                    | "number"
                                    | "object"
                                    | "string"
                            )
                        }) {
                            return Err(ApiError::bad(format!(
                                "tools[{index}] property {property} has unknown JSON schema type"
                            )));
                        }
                    }
                }
            }
        }
        validated.push(tool.clone());
    }
    if choice == "none" {
        Ok(None)
    } else {
        Ok(Some(validated))
    }
}

fn chat_options(
    request: &serde_json::Map<String, Value>,
    default_thinking: bool,
) -> Checked<ChatOptions> {
    let mut options = ChatOptions {
        thinking: default_thinking,
        ..ChatOptions::default()
    };
    let kwargs = match request.get("chat_template_kwargs").filter(|v| !v.is_null()) {
        Some(value) => Some(object(value, "chat_template_kwargs")?),
        None => None,
    };
    if let Some(kwargs) = kwargs {
        for key in kwargs.keys() {
            if !matches!(
                key.as_str(),
                "enable_thinking" | "preserve_thinking" | "reasoning_effort"
            ) {
                return Err(ApiError::bad(format!(
                    "unsupported chat_template_kwargs field: {key}"
                )));
            }
        }
        if let Some(value) = kwargs.get("enable_thinking") {
            options.thinking = value
                .as_bool()
                .ok_or_else(|| ApiError::bad("enable_thinking must be boolean"))?;
        }
        if let Some(value) = kwargs.get("preserve_thinking") {
            options.preserve_thinking = value
                .as_bool()
                .ok_or_else(|| ApiError::bad("preserve_thinking must be boolean"))?;
        }
    }
    let effort = request
        .get("reasoning_effort")
        .or_else(|| kwargs.and_then(|v| v.get("reasoning_effort")));
    if let Some(effort) = effort {
        options.effort = match effort.as_str() {
            Some("low") => ReasoningEffort::Low,
            Some("medium") => ReasoningEffort::Medium,
            Some("xhigh") => ReasoningEffort::Xhigh,
            _ => {
                return Err(ApiError::bad(
                    "reasoning_effort must be low, medium, or xhigh",
                ));
            }
        };
    }
    Ok(options)
}

pub(super) fn parse_chat(
    value: &Value,
    model: &str,
    defaults: SamplingOptions,
    default_thinking: bool,
) -> Checked<ChatRequest> {
    let request = object(value, "request")?;
    const SUPPORTED_FIELDS: &[&str] = &[
        "model",
        "messages",
        "tools",
        "tool_choice",
        "stream",
        "stream_options",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "presence_context_size",
        "seed",
        "reasoning_effort",
        "chat_template_kwargs",
        "n",
        "frequency_penalty",
        "repetition_penalty",
        "stop",
        "logit_bias",
        "response_format",
        "parallel_tool_calls",
        "logprobs",
        "top_logprobs",
        "user",
        "metadata",
        "store",
    ];
    for key in request.keys() {
        if !SUPPORTED_FIELDS.contains(&key.as_str()) {
            return Err(ApiError::bad(format!("unsupported request field: {key}")));
        }
    }
    let response_model = match request.get("model") {
        None | Some(Value::Null) => model.to_owned(),
        Some(Value::String(value)) if !value.is_empty() => value.clone(),
        _ => return Err(ApiError::bad("model must be a nonempty string")),
    };
    let messages = normalize_messages(request.get("messages").unwrap_or(&Value::Null))?;
    let tools = validate_tools(request.get("tools"), request.get("tool_choice"))?;
    let chat = chat_options(request, default_thinking)?;
    let stream = match request.get("stream") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(value)) => *value,
        _ => return Err(ApiError::bad("stream must be a boolean")),
    };
    let mut include_usage = false;
    if let Some(value) = request.get("stream_options").filter(|v| !v.is_null()) {
        if !stream {
            return Err(ApiError::bad("stream_options requires stream=true"));
        }
        let options = object(value, "stream_options")?;
        if options.keys().any(|key| key != "include_usage") {
            return Err(ApiError::bad("unknown stream_options field"));
        }
        if let Some(value) = options.get("include_usage") {
            include_usage = value
                .as_bool()
                .ok_or_else(|| ApiError::bad("stream_options.include_usage must be boolean"))?;
        }
    }
    let max_value = request
        .get("max_tokens")
        .or_else(|| request.get("max_completion_tokens"));
    if request.contains_key("max_tokens") && request.contains_key("max_completion_tokens") {
        return Err(ApiError::bad(
            "use only one of max_tokens and max_completion_tokens",
        ));
    }
    let max_tokens = match max_value {
        Some(value) => usize::try_from(unsigned(value, "max_tokens")?)
            .map_err(|_| ApiError::bad("max_tokens is too large"))?,
        None => 2048,
    };
    if max_tokens == 0 {
        return Err(ApiError::bad("max_tokens must be positive"));
    }
    if let Some(value) = request.get("n")
        && unsigned(value, "n")? != 1
    {
        return Err(ApiError::bad("only n=1 is supported"));
    }
    for (field, neutral) in [("frequency_penalty", 0.0), ("repetition_penalty", 1.0)] {
        if let Some(value) = request.get(field).filter(|v| !v.is_null())
            && number(value, field)? != neutral
        {
            return Err(ApiError::bad(format!("non-neutral {field} is unsupported")));
        }
    }
    if request
        .get("stop")
        .is_some_and(|v| !v.is_null() && v.as_array().is_none_or(|a| !a.is_empty()))
    {
        return Err(ApiError::bad("custom stop sequences are unsupported"));
    }
    if request
        .get("logit_bias")
        .is_some_and(|v| !v.is_null() && v.as_object().is_none_or(|o| !o.is_empty()))
    {
        return Err(ApiError::bad("logit_bias is unsupported"));
    }
    if request
        .get("response_format")
        .is_some_and(|v| !v.is_null() && v.get("type").and_then(Value::as_str) != Some("text"))
    {
        return Err(ApiError::bad("structured response_format is unsupported"));
    }
    for (field, neutral) in [
        ("parallel_tool_calls", true),
        ("logprobs", false),
        ("store", false),
    ] {
        if let Some(value) = request.get(field).filter(|v| !v.is_null())
            && value.as_bool() != Some(neutral)
        {
            return Err(ApiError::bad(format!(
                "{field} is unsupported unless {neutral}"
            )));
        }
    }
    if request
        .get("top_logprobs")
        .is_some_and(|v| !v.is_null() && v.as_u64() != Some(0))
    {
        return Err(ApiError::bad("top_logprobs is unsupported"));
    }
    if request
        .get("user")
        .is_some_and(|v| !v.is_null() && !v.is_string())
    {
        return Err(ApiError::bad("user must be a string"));
    }
    if request
        .get("metadata")
        .is_some_and(|v| !v.is_null() && v.as_object().is_none_or(|o| !o.is_empty()))
    {
        return Err(ApiError::bad("request metadata is unsupported"));
    }
    let mut sampling = defaults;
    for field in ["temperature", "top_p", "min_p", "presence_penalty"] {
        if let Some(value) = request.get(field).filter(|v| !v.is_null()) {
            let parsed = number(value, field)?;
            match field {
                "temperature" => sampling.temperature = parsed,
                "top_p" => sampling.top_p = parsed,
                "min_p" => sampling.min_p = parsed,
                _ => sampling.presence_penalty = parsed,
            }
        }
    }
    if let Some(value) = request.get("top_k").filter(|v| !v.is_null()) {
        sampling.top_k = usize::try_from(unsigned(value, "top_k")?)
            .map_err(|_| ApiError::bad("top_k is too large"))?;
    }
    if let Some(value) = request
        .get("presence_context_size")
        .filter(|v| !v.is_null())
    {
        sampling.presence_context_size = usize::try_from(unsigned(value, "presence_context_size")?)
            .map_err(|_| ApiError::bad("presence_context_size is too large"))?;
    }
    let seed = request
        .get("seed")
        .filter(|v| !v.is_null())
        .map(|v| unsigned(v, "seed"))
        .transpose()?;
    Ok(ChatRequest {
        model: response_model,
        messages,
        tools,
        chat,
        sampling,
        max_tokens,
        seed,
        stream,
        include_usage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_openai_tool_history_for_qwen_template() {
        let value = json!({"model":"pkg","messages":[
            {"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"add","arguments":"{\"a\":2}"}}]},
            {"role":"tool","tool_call_id":"c1","content":"2"}
        ],"tools":[{"type":"function","function":{"name":"add","parameters":{"type":"object","properties":{"a":{"type":"integer"}}}}}]});
        let parsed = parse_chat(&value, "pkg", SamplingOptions::default(), true).unwrap();
        assert_eq!(
            parsed.messages[0]["tool_calls"][0]["function"]["arguments"],
            json!({"a":2})
        );
        assert_eq!(parsed.tools.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn friendly_model_alias_is_echoed_without_selecting_another_package() {
        let body = json!({"model":"qwen-local","messages":[{"role":"user","content":"hi"}]});
        let parsed = parse_chat(&body, "artifact-sha", SamplingOptions::default(), true).unwrap();
        assert_eq!(parsed.model, "qwen-local");
        let missing = json!({"messages":[{"role":"user","content":"hi"}]});
        assert_eq!(
            parse_chat(&missing, "artifact-sha", SamplingOptions::default(), true)
                .unwrap()
                .model,
            "artifact-sha"
        );
    }

    #[test]
    fn request_thinking_overrides_server_default_without_changing_other_options() {
        let base = json!({"model":"pkg","messages":[{"role":"user","content":"hi"}]});
        for (server_default, request_override, expected) in [
            (true, None, true),
            (false, None, false),
            (true, Some(true), true),
            (false, Some(true), true),
            (true, Some(false), false),
            (false, Some(false), false),
        ] {
            for stream in [false, true] {
                let mut body = base.clone();
                body["stream"] = json!(stream);
                if let Some(enabled) = request_override {
                    body["chat_template_kwargs"] = json!({"enable_thinking": enabled});
                }
                let parsed =
                    parse_chat(&body, "pkg", SamplingOptions::default(), server_default).unwrap();
                assert_eq!(parsed.chat.thinking, expected);
                assert!(parsed.chat.preserve_thinking);
                assert!(matches!(parsed.chat.effort, ReasoningEffort::Medium));
                assert_eq!(parsed.stream, stream);
            }
        }
        let mut explicit = base.clone();
        explicit["chat_template_kwargs"] = json!({
            "enable_thinking": false,
            "preserve_thinking": false,
            "reasoning_effort": "low"
        });
        let parsed = parse_chat(&explicit, "pkg", SamplingOptions::default(), true).unwrap();
        assert!(!parsed.chat.thinking);
        assert!(!parsed.chat.preserve_thinking);
        assert!(matches!(parsed.chat.effort, ReasoningEffort::Low));
        explicit["chat_template_kwargs"]["enable_thinking"] = json!("off");
        assert!(parse_chat(&explicit, "pkg", SamplingOptions::default(), true).is_err());
    }

    #[test]
    fn rejects_unsupported_controls_and_invalid_modalities() {
        let base = json!({"model":"pkg","messages":[{"role":"user","content":"hi"}]});
        for patch in [
            json!({"tool_choice":{"type":"function","function":{"name":"x"}}}),
            json!({"stop":["END"]}),
            json!({"frequency_penalty":1}),
            json!({"response_format":{"type":"json_object"}}),
            json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}),
            json!({"stream_options":{"include_usage":true}}),
            json!({"parallel_tool_calls":false}),
            json!({"logprobs":true}),
            json!({"unknown_sampler":1}),
        ] {
            let mut candidate = base.clone();
            candidate
                .as_object_mut()
                .unwrap()
                .extend(patch.as_object().unwrap().clone());
            assert!(
                parse_chat(&candidate, "pkg", SamplingOptions::default(), true).is_err(),
                "{candidate}"
            );
        }
    }
}
