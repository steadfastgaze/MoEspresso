//! Incremental reasoning and native Qwen XML tool-call response shaping.

use serde_json::{Map, Value, json};

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";
const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const MAX_TOOL_BLOCK: usize = 256 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Channel {
    Reasoning,
    Content,
}

#[derive(Debug, PartialEq)]
pub(super) enum Delta {
    Reasoning(String),
    Content(String),
    Tool { index: usize, entry: Value },
}

fn suffix_prefix_len(text: &str, marker: &str) -> usize {
    let limit = text.len().min(marker.len() - 1);
    (1..=limit)
        .rev()
        .find(|&length| text.ends_with(&marker[..length]))
        .unwrap_or(0)
}

struct ReasoningSplitter {
    mode: Channel,
    at_start: bool,
    buffer: String,
}

impl ReasoningSplitter {
    fn new(thinking: bool) -> Self {
        Self {
            mode: if thinking {
                Channel::Reasoning
            } else {
                Channel::Content
            },
            at_start: true,
            buffer: String::new(),
        }
    }

    fn publish(&self, text: String, out: &mut Vec<(Channel, String)>) {
        if !text.is_empty() {
            out.push((self.mode, text));
        }
    }

    fn push(&mut self, text: &str) -> Vec<(Channel, String)> {
        let mut out = Vec::new();
        self.buffer.push_str(text);
        while !self.buffer.is_empty() {
            if self.mode == Channel::Reasoning {
                if self.at_start {
                    if THINK_OPEN.starts_with(&self.buffer) && self.buffer.len() < THINK_OPEN.len()
                    {
                        break;
                    }
                    if self.buffer.starts_with(THINK_OPEN) {
                        self.buffer.drain(..THINK_OPEN.len());
                        self.at_start = false;
                        continue;
                    }
                    self.at_start = false;
                }
                if let Some(index) = self.buffer.find(THINK_CLOSE) {
                    let text = self.buffer[..index].to_owned();
                    self.buffer.drain(..index + THINK_CLOSE.len());
                    self.publish(text, &mut out);
                    self.mode = Channel::Content;
                    continue;
                }
                let end = self.buffer.len() - suffix_prefix_len(&self.buffer, THINK_CLOSE);
                let text: String = self.buffer.drain(..end).collect();
                self.publish(text, &mut out);
                break;
            }
            if let Some(index) = self.buffer.find(THINK_OPEN) {
                let text = self.buffer[..index].to_owned();
                self.buffer.drain(..index + THINK_OPEN.len());
                self.publish(text, &mut out);
                self.mode = Channel::Reasoning;
                self.at_start = false;
                continue;
            }
            let end = self.buffer.len() - suffix_prefix_len(&self.buffer, THINK_OPEN);
            let text: String = self.buffer.drain(..end).collect();
            self.publish(text, &mut out);
            break;
        }
        out
    }

    fn finish(&mut self) -> Vec<(Channel, String)> {
        let mut out = Vec::new();
        let tail = std::mem::take(&mut self.buffer);
        self.publish(tail, &mut out);
        out
    }
}

struct ToolStream {
    schemas: Map<String, Value>,
    id_prefix: String,
    buffer: String,
    held_ws: String,
    line_start: bool,
    active: Option<&'static str>,
    content: String,
    calls: Vec<Value>,
}

impl ToolStream {
    fn new(tools: &[Value], id_prefix: String) -> Self {
        let schemas = tools
            .iter()
            .filter_map(|tool| {
                let function = tool.get("function")?;
                Some((
                    function.get("name")?.as_str()?.to_owned(),
                    function.get("parameters").cloned().unwrap_or(Value::Null),
                ))
            })
            .collect();
        Self {
            schemas,
            id_prefix,
            buffer: String::new(),
            held_ws: String::new(),
            line_start: true,
            active: None,
            content: String::new(),
            calls: Vec::new(),
        }
    }

    fn consume(&mut self, end: usize) -> String {
        let consumed: String = self.buffer.drain(..end).collect();
        if !consumed.is_empty() {
            self.line_start = consumed.ends_with('\n');
        }
        consumed
    }

    fn emit_content(&mut self, text: &str, out: &mut Vec<Delta>) {
        if text.is_empty() {
            return;
        }
        let mut combined = std::mem::take(&mut self.held_ws);
        combined.push_str(text);
        let trimmed = combined.trim_end_matches(char::is_whitespace);
        let split = trimmed.len();
        if split == 0 {
            self.held_ws = combined;
            return;
        }
        let visible = combined[..split].to_owned();
        self.held_ws = combined[split..].to_owned();
        self.content.push_str(&visible);
        out.push(Delta::Content(visible));
    }

    fn at_line_start(&self, index: usize) -> bool {
        if index == 0 {
            self.line_start
        } else {
            self.buffer.as_bytes()[index - 1] == b'\n'
        }
    }

    fn find_open(&self) -> Option<(usize, &'static str, &'static str)> {
        let mut best = None;
        for (open, close) in [(TOOL_OPEN, TOOL_CLOSE), (FUNCTION_OPEN, FUNCTION_CLOSE)] {
            for (index, _) in self.buffer.match_indices(open) {
                if self.at_line_start(index) && best.is_none_or(|(earlier, _, _)| index < earlier) {
                    best = Some((index, open, close));
                    break;
                }
            }
        }
        best
    }

    fn hold_len(&self) -> usize {
        [TOOL_OPEN, FUNCTION_OPEN]
            .into_iter()
            .map(|marker| {
                let len = suffix_prefix_len(&self.buffer, marker);
                if len > 0 && self.at_line_start(self.buffer.len() - len) {
                    len
                } else {
                    0
                }
            })
            .max()
            .unwrap_or(0)
    }

    fn handle_block(&mut self, block: &str, open: &str, terminal: bool, out: &mut Vec<Delta>) {
        if let Some((name, arguments)) =
            repair_tool_block(block, open == FUNCTION_OPEN, terminal, &self.schemas)
        {
            let index = self.calls.len();
            let entry = json!({
                "id": format!("call_{}_{}", self.id_prefix, index),
                "type": "function",
                "function": {"name": name, "arguments": serde_json::to_string(&arguments).unwrap()}
            });
            self.calls.push(entry.clone());
            out.push(Delta::Tool { index, entry });
        } else {
            self.emit_content(block, out);
        }
    }

    fn push(&mut self, text: &str) -> Vec<Delta> {
        let mut out = Vec::new();
        self.buffer.push_str(text);
        loop {
            if let Some(close) = self.active {
                if let Some(index) = self.buffer.find(close) {
                    let block = self.consume(index + close.len());
                    self.active = None;
                    let open = if close == TOOL_CLOSE {
                        TOOL_OPEN
                    } else {
                        FUNCTION_OPEN
                    };
                    self.handle_block(&block, open, false, &mut out);
                    self.line_start = true;
                    continue;
                }
                if self.buffer.len() > MAX_TOOL_BLOCK {
                    let raw = self.consume(self.buffer.len());
                    self.active = None;
                    self.emit_content(&raw, &mut out);
                }
                break;
            }
            if let Some((index, open, close)) = self.find_open() {
                let before = self.consume(index);
                self.emit_content(&before, &mut out);
                self.active = Some(close);
                debug_assert!(self.buffer.starts_with(open));
                continue;
            }
            let end = self.buffer.len() - self.hold_len();
            let before = self.consume(end);
            self.emit_content(&before, &mut out);
            break;
        }
        out
    }

    fn finish(&mut self, truncated: bool) -> Vec<Delta> {
        let mut out = Vec::new();
        if let Some(close) = self.active
            && !truncated
        {
            let raw = self.consume(self.buffer.len());
            self.active = None;
            let open = if close == TOOL_CLOSE {
                TOOL_OPEN
            } else {
                FUNCTION_OPEN
            };
            self.handle_block(&raw, open, true, &mut out);
        }
        let raw = self.consume(self.buffer.len());
        self.active = None;
        self.emit_content(&raw, &mut out);
        if self.calls.is_empty() && !self.held_ws.is_empty() {
            let text = std::mem::take(&mut self.held_ws);
            self.content.push_str(&text);
            out.push(Delta::Content(text));
        }
        out
    }
}

fn json_type_matches(value: &Value, kind: &str) -> bool {
    match kind {
        "integer" => value.is_i64() || value.is_u64(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "null" => value.is_null(),
        "string" => value.is_string(),
        _ => false,
    }
}

fn typed_argument(raw: &str, declaration: Option<&Value>, lenient: bool) -> Option<Value> {
    let Some(declaration) = declaration else {
        return Some(Value::String(raw.into()));
    };
    let (types, nullable): (Vec<&str>, bool) = match declaration {
        Value::String(kind) if kind == "null" => (Vec::new(), true),
        Value::String(kind) => (vec![kind], false),
        Value::Array(types) => (
            types
                .iter()
                .filter_map(Value::as_str)
                .filter(|kind| *kind != "null")
                .collect(),
            types.iter().any(|v| v.as_str() == Some("null")),
        ),
        _ => return Some(Value::String(raw.into())),
    };
    if nullable && raw == "null" {
        return Some(Value::Null);
    }
    if types.is_empty() {
        return (!nullable).then(|| Value::String(raw.into()));
    }
    let accepts_raw = types
        .iter()
        .any(|kind| *kind == "string" || !is_known_json_type(kind));
    if lenient && !accepts_raw {
        // The repair pass rewrites all declared non-string values in a
        // malformed call, in declared union order, before strict reparse.
        for kind in &types {
            if let Some(value) = lenient_value(raw.trim(), kind) {
                return Some(value);
            }
        }
    }
    if let Ok(decoded) = serde_json::from_str::<Value>(raw)
        && types
            .iter()
            .any(|kind| *kind != "string" && json_type_matches(&decoded, kind))
    {
        return Some(decoded);
    }
    if accepts_raw {
        return Some(Value::String(raw.into()));
    }
    None
}

fn is_known_json_type(kind: &str) -> bool {
    matches!(kind, "integer" | "number" | "boolean" | "array" | "object")
}

fn integral_decimal(text: &str) -> Option<&str> {
    let (integer, fractional) = text.split_once('.')?;
    let digits = integer.strip_prefix('-').unwrap_or(integer);
    (!digits.is_empty()
        && digits.bytes().all(|byte| byte.is_ascii_digit())
        && fractional.bytes().all(|byte| byte == b'0'))
    .then_some(integer)
}

fn lenient_value(raw: &str, kind: &str) -> Option<Value> {
    if !is_known_json_type(kind) {
        return None;
    }
    let mut candidates = vec![raw.to_owned()];
    let mut unquoted = raw;
    for quote in ['\'', '"'] {
        if unquoted.len() >= 2 && unquoted.starts_with(quote) && unquoted.ends_with(quote) {
            unquoted = &unquoted[1..unquoted.len() - 1];
        }
    }
    if unquoted != raw {
        candidates.push(unquoted.to_owned());
    }
    for candidate in candidates.clone() {
        let lower = candidate.trim().to_ascii_lowercase();
        if kind == "boolean" {
            match lower.as_str() {
                "true" | "yes" | "on" => candidates.push("true".into()),
                "false" | "no" | "off" => candidates.push("false".into()),
                _ => {}
            }
        }
        if kind == "integer" || kind == "number" {
            let digits = candidate.replace(',', "");
            if digits != candidate {
                candidates.push(digits.clone());
            }
            if kind == "integer"
                && let Some(integer) = integral_decimal(&digits)
            {
                candidates.push(integer.to_owned());
            }
        }
    }
    candidates.into_iter().find_map(|candidate| {
        let decoded = serde_json::from_str::<Value>(&candidate).ok()?;
        json_type_matches(&decoded, kind).then_some(decoded)
    })
}

fn opens_nested_parameter(raw: &str) -> bool {
    raw.match_indices("<parameter=")
        .any(|(index, _)| index == 0 || raw.as_bytes()[index - 1] == b'\n')
}

fn parse_tool_block(
    block: &str,
    naked: bool,
    schemas: &Map<String, Value>,
    lenient: bool,
) -> Option<(String, Value)> {
    let inner = if naked {
        block
    } else {
        block
            .strip_prefix(TOOL_OPEN)?
            .strip_suffix(TOOL_CLOSE)?
            .trim()
    };
    let after_open = inner.strip_prefix(FUNCTION_OPEN)?;
    let (name, body) = after_open.split_once('>')?;
    if name.is_empty() || name.contains('\n') {
        return None;
    }
    let body = body.strip_suffix(FUNCTION_CLOSE)?;
    if body.contains(FUNCTION_OPEN) {
        return None;
    }
    let properties = schemas.get(name).and_then(|v| v.get("properties"));
    let mut arguments = Map::new();
    let mut rest = body.trim();
    while !rest.is_empty() {
        let after = rest.strip_prefix("<parameter=")?;
        let (key, after) = after.split_once('>')?;
        if key.is_empty() || key.contains('\n') || arguments.contains_key(key) {
            return None;
        }
        let (raw, tail) = after.split_once("</parameter>")?;
        if opens_nested_parameter(raw) {
            return None;
        }
        let raw = raw.strip_prefix('\n').unwrap_or(raw);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        let declaration = properties
            .and_then(|v| v.get(key))
            .and_then(|v| v.get("type"));
        arguments.insert(key.to_owned(), typed_argument(raw, declaration, lenient)?);
        rest = tail.trim();
    }
    Some((name.to_owned(), Value::Object(arguments)))
}

fn normalize_named_closers(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some((index, kind, canonical)) = [
        ("</function=", "function", FUNCTION_CLOSE),
        ("</parameter=", "parameter", "</parameter>"),
    ]
    .into_iter()
    .filter_map(|(marker, kind, canonical)| rest.find(marker).map(|index| (index, kind, canonical)))
    .min_by_key(|(index, _, _)| *index)
    {
        let marker_len = kind.len() + 3;
        out.push_str(&rest[..index]);
        let after = &rest[index + marker_len..];
        if let Some(end) = after.find('>')
            && !after[..end].contains('\n')
        {
            out.push_str(canonical);
            rest = &after[end + 1..];
        } else {
            out.push_str(&rest[index..index + marker_len]);
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

fn close_dangling(mut text: String) -> Option<String> {
    for (open, close) in [
        ("<parameter=", "</parameter>"),
        (FUNCTION_OPEN, FUNCTION_CLOSE),
        (TOOL_OPEN, TOOL_CLOSE),
    ] {
        let deficit = text
            .matches(open)
            .count()
            .saturating_sub(text.matches(close).count());
        let appended = deficit.checked_mul(close.len() + 1)?;
        if text.len().checked_add(appended)? > MAX_TOOL_BLOCK {
            return None;
        }
        for _ in 0..deficit {
            text.push('\n');
            text.push_str(close);
        }
    }
    Some(text)
}

fn drop_fence_lines(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        let body = line.strip_suffix('\n').unwrap_or(line);
        let fence = body.strip_prefix("```").is_some_and(|suffix| {
            suffix
                .trim_end_matches([' ', '\t'])
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
        });
        if !fence {
            out.push_str(line);
        }
    }
    out
}

fn repair_tool_block(
    block: &str,
    naked: bool,
    terminal: bool,
    schemas: &Map<String, Value>,
) -> Option<(String, Value)> {
    if let Some(parsed) = parse_tool_block(block, naked, schemas, false) {
        return Some(parsed);
    }
    // Each repair stage must still satisfy the entire strict structure.
    let mut candidate = drop_fence_lines(block);
    if let Some(parsed) = parse_tool_block(&candidate, naked, schemas, false) {
        return Some(parsed);
    }
    candidate = normalize_named_closers(&candidate);
    if let Some(parsed) = parse_tool_block(&candidate, naked, schemas, false) {
        return Some(parsed);
    }
    if naked {
        // Wrap a naked function only after a complete function closer exists.
        // Appending a closer to an unfinished naked value must not turn that
        // value into an executable call.
        if !candidate.contains(FUNCTION_CLOSE) {
            return None;
        }
        candidate = format!("{TOOL_OPEN}\n{candidate}\n{TOOL_CLOSE}");
    }
    if terminal {
        candidate = close_dangling(candidate)?;
    }
    parse_tool_block(&candidate, false, schemas, false)
        .or_else(|| parse_tool_block(&candidate, false, schemas, true))
}

pub(super) struct ResponseShaper {
    reasoning_splitter: ReasoningSplitter,
    tools: Option<ToolStream>,
    pub reasoning: String,
    pub content: String,
    pub calls: Vec<Value>,
}

impl ResponseShaper {
    pub fn new(thinking: bool, tools: Option<&[Value]>, id_prefix: String) -> Self {
        Self {
            reasoning_splitter: ReasoningSplitter::new(thinking),
            tools: tools.map(|v| ToolStream::new(v, id_prefix)),
            reasoning: String::new(),
            content: String::new(),
            calls: Vec::new(),
        }
    }

    fn route(&mut self, pieces: Vec<(Channel, String)>) -> Vec<Delta> {
        let mut out = Vec::new();
        for (channel, text) in pieces {
            match channel {
                Channel::Reasoning => {
                    self.reasoning.push_str(&text);
                    out.push(Delta::Reasoning(text));
                }
                Channel::Content => {
                    if let Some(tools) = &mut self.tools {
                        out.extend(tools.push(&text));
                    } else {
                        self.content.push_str(&text);
                        out.push(Delta::Content(text));
                    }
                }
            }
        }
        out
    }

    pub fn push(&mut self, text: &str) -> Vec<Delta> {
        let pieces = self.reasoning_splitter.push(text);
        self.route(pieces)
    }

    pub fn finish(&mut self, truncated: bool) -> Vec<Delta> {
        let pieces = self.reasoning_splitter.finish();
        let mut out = self.route(pieces);
        if let Some(tools) = &mut self.tools {
            out.extend(tools.finish(truncated));
            self.content = tools.content.clone();
            self.calls = tools.calls.clone();
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_tool(raw: &str, declaration: Value) -> ResponseShaper {
        let tools = [json!({"function": {
            "name": "read",
            "parameters": {"properties": {"value": {"type": declaration}}}
        }})];
        let mut stream = ResponseShaper::new(false, Some(&tools), "x".into());
        stream.push(&format!(
            "<tool_call><function=read><parameter=value>\n{raw}\n</parameter></function></tool_call>"
        ));
        stream.finish(false);
        stream
    }

    fn call_arguments(stream: &ResponseShaper) -> Value {
        serde_json::from_str(stream.calls[0]["function"]["arguments"].as_str().unwrap()).unwrap()
    }

    #[test]
    fn reasoning_markers_survive_arbitrary_chunks() {
        let mut stream = ResponseShaper::new(true, None, "x".into());
        let mut deltas = Vec::new();
        for part in [
            "<thi",
            "nk>scratch</th",
            "ink>answer",
            "<think>more</think>!",
        ] {
            deltas.extend(stream.push(part));
        }
        deltas.extend(stream.finish(false));
        assert_eq!(stream.reasoning, "scratchmore");
        assert_eq!(stream.content, "answer!");
        assert!(
            deltas
                .iter()
                .any(|v| matches!(v, Delta::Reasoning(s) if s == "scratch"))
        );
    }

    #[test]
    fn xml_tool_call_is_typed_and_streamed_once() {
        let tools = [
            json!({"function":{"name":"add","parameters":{"properties":{"a":{"type":"integer"},"b":{"type":"integer"}}}}}),
        ];
        let mut stream = ResponseShaper::new(false, Some(&tools), "x".into());
        let mut deltas = Vec::new();
        for part in [
            "Done\n<tool_",
            "call><function=add><parameter=a>\n2\n</parameter>",
            "<parameter=b>\n3\n</parameter></function></tool_call>",
        ] {
            deltas.extend(stream.push(part));
        }
        deltas.extend(stream.finish(false));
        assert_eq!(stream.calls.len(), 1);
        assert_eq!(
            stream.calls[0]["function"]["arguments"],
            "{\"a\":2,\"b\":3}"
        );
        assert_eq!(stream.content, "Done");
        assert_eq!(
            deltas
                .iter()
                .filter(|v| matches!(v, Delta::Tool { .. }))
                .count(),
            1
        );
    }

    #[test]
    fn malformed_and_truncated_tool_markup_remains_content() {
        let tools = [json!({"function":{"name":"add"}})];
        let raw = "<tool_call><function=add><parameter=a>2</function></tool_call>";
        let mut stream = ResponseShaper::new(false, Some(&tools), "x".into());
        stream.push(raw);
        stream.finish(false);
        assert_eq!(stream.content, raw);
        assert!(stream.calls.is_empty());

        let mut stream = ResponseShaper::new(false, Some(&tools), "x".into());
        stream.push("<tool_call><function=add><parameter=a>2");
        stream.finish(true);
        assert!(stream.calls.is_empty());
        assert_eq!(stream.content, "<tool_call><function=add><parameter=a>2");
    }

    #[test]
    fn schema_scalar_repairs_follow_strict_first_union_rules() {
        for (raw, kind, expected) in [
            ("'7'", json!("integer"), json!(7)),
            ("\"7\"", json!("integer"), json!(7)),
            ("7.0", json!("integer"), json!(7)),
            ("1,234.00", json!("integer"), json!(1234)),
            ("1,234.5", json!("number"), json!(1234.5)),
            ("True", json!("boolean"), json!(true)),
            ("yes", json!("boolean"), json!(true)),
            ("on", json!("boolean"), json!(true)),
            ("No", json!("boolean"), json!(false)),
            ("'[1,2]'", json!("array"), json!([1, 2])),
        ] {
            let stream = one_tool(raw, kind);
            assert_eq!(stream.calls.len(), 1, "{raw}");
            assert_eq!(call_arguments(&stream)["value"], expected, "{raw}");
        }
        for (raw, kind) in [
            ("True", json!("integer")),
            ("7e0", json!("integer")),
            (" null ", json!("null")),
            (" null ", json!(["integer", "null"])),
        ] {
            let stream = one_tool(raw, kind);
            assert!(stream.calls.is_empty(), "{raw}");
        }
        for (raw, kind, expected) in [
            ("5", json!(["integer", "string", "null"]), json!(5)),
            (
                "\"5\"",
                json!(["integer", "string", "null"]),
                json!("\"5\""),
            ),
            ("'7'", json!(["integer", "string", "null"]), json!("'7'")),
            ("null", json!(["integer", "string", "null"]), Value::Null),
            ("'7'", json!(["integer", "null"]), json!(7)),
            (" null ", json!(["string", "null"]), json!(" null ")),
            (" null ", json!([]), json!(" null ")),
            ("random", json!("custom"), json!("random")),
            ("a, b  c", json!("string"), json!("a, b  c")),
        ] {
            let stream = one_tool(raw, kind);
            assert_eq!(stream.calls.len(), 1, "{raw}");
            assert_eq!(call_arguments(&stream)["value"], expected, "{raw}");
        }
    }

    #[test]
    fn named_closers_and_terminal_repair_keep_completion_boundary() {
        let tools = [json!({"function": {"name": "read"}})];
        let wrapped = "<tool_call>\n<function=read>\n<parameter=path>\nREADME.md\n</parameter=path>\n</function=read>\n</tool_call>";
        let mut stream = ResponseShaper::new(false, Some(&tools), "x".into());
        stream.push(wrapped);
        assert_eq!(stream.calls.len(), 0);
        stream.finish(true);
        assert_eq!(stream.calls.len(), 1);
        assert_eq!(call_arguments(&stream)["path"], "README.md");

        let naked =
            "<function=read>\n<parameter=path>\nREADME.md\n</parameter=path>\n</function=read>";
        let mut normal = ResponseShaper::new(false, Some(&tools), "x".into());
        normal.push(naked);
        normal.finish(false);
        assert_eq!(normal.calls.len(), 1);
        let mut truncated = ResponseShaper::new(false, Some(&tools), "x".into());
        truncated.push(naked);
        truncated.finish(true);
        assert!(truncated.calls.is_empty());
        assert_eq!(truncated.content, naked);

        let dangling = "<tool_call>\n<function=read>\n<parameter=path>\nREA";
        let mut normal = ResponseShaper::new(false, Some(&tools), "x".into());
        normal.push(dangling);
        normal.finish(false);
        assert_eq!(call_arguments(&normal)["path"], "REA");
        let mut truncated = ResponseShaper::new(false, Some(&tools), "x".into());
        truncated.push(dangling);
        truncated.finish(true);
        assert!(truncated.calls.is_empty());
        assert_eq!(truncated.content, dangling);

        let naked_dangling = "<function=read>\n<parameter=path>\nREA";
        let mut normal = ResponseShaper::new(false, Some(&tools), "x".into());
        normal.push(naked_dangling);
        normal.finish(false);
        assert!(normal.calls.is_empty());
        assert_eq!(normal.content, naked_dangling);
    }

    #[test]
    fn malformed_interior_and_nested_markers_do_not_gain_closers() {
        let tools = [json!({"function": {"name": "read"}})];
        for raw in [
            "<tool_call><function=read><parameter=path>README.md</function></tool_call>",
            "<tool_call><function=read><parameter=path>README.md</parameter></tool_call>",
            "<tool_call><function=read><parameter=path>one\n<parameter=other>two</parameter></function></tool_call>",
            "<tool_call><function=read><parameter=path>x</parameter><parameter=path>y</parameter></function></tool_call>",
        ] {
            let mut stream = ResponseShaper::new(false, Some(&tools), "x".into());
            stream.push(raw);
            stream.finish(false);
            assert!(stream.calls.is_empty(), "{raw}");
            assert_eq!(stream.content, raw);
        }
        let inline = "<tool_call><function=read><parameter=path>Open with <parameter=k>, then close</parameter></function></tool_call>";
        let mut stream = ResponseShaper::new(false, Some(&tools), "x".into());
        stream.push(inline);
        stream.finish(false);
        assert_eq!(
            call_arguments(&stream)["path"],
            "Open with <parameter=k>, then close"
        );
    }

    #[test]
    fn repaired_blocks_are_split_invariant_and_reasoning_excluded() {
        let tools = [
            json!({"function": {"name": "read", "parameters": {"properties": {"limit": {"type": "integer"}}}}}),
        ];
        let text = "lead\n<tool_call><function=read><parameter=limit>'7'</parameter></function=read></tool_call><tool_call><function=read><parameter=limit>8</parameter></function></tool_call>";
        let run = |parts: &[&str]| {
            let mut stream = ResponseShaper::new(false, Some(&tools), "x".into());
            for part in parts {
                stream.push(part);
            }
            stream.finish(false);
            (stream.content, stream.calls)
        };
        let expected = run(&[text]);
        assert_eq!(expected.1.len(), 2);
        assert_eq!(expected.0, "lead");
        for cut in 0..=text.len() {
            let actual = run(&[&text[..cut], &text[cut..]]);
            assert_eq!(actual, expected, "cut={cut}");
        }

        let mut reasoning = ResponseShaper::new(true, Some(&tools), "x".into());
        reasoning.push(&format!("<think>{text}</think>answer"));
        reasoning.finish(false);
        assert!(reasoning.calls.is_empty());
        assert_eq!(reasoning.content, "answer");
    }

    #[test]
    fn repair_rewrites_all_typed_values_in_union_order() {
        let tools = [
            json!({"function": {"name": "read", "parameters": {"properties": {
                "a": {"type": ["integer", "number"]},
                "b": {"type": "integer"}
            }}}}),
        ];
        let raw = "<tool_call><function=read><parameter=a>7.0</parameter><parameter=b>'2'</parameter></function></tool_call>";
        let mut stream = ResponseShaper::new(false, Some(&tools), "x".into());
        stream.push(raw);
        stream.finish(false);
        assert_eq!(call_arguments(&stream), json!({"a": 7, "b": 2}));
    }

    #[test]
    fn fence_lines_inside_a_malformed_block_are_removed_only_during_repair() {
        let tools = [json!({"function": {"name": "read"}})];
        let raw = "<tool_call>\n```xml\n<function=read><parameter=path>x</parameter></function>\n```\n</tool_call>";
        let mut stream = ResponseShaper::new(false, Some(&tools), "x".into());
        stream.push(raw);
        stream.finish(false);
        assert_eq!(call_arguments(&stream)["path"], "x");

        let literal = "<tool_call><function=read><parameter=path>\n```xml\nkeep\n```\n</parameter></function></tool_call>";
        let mut stream = ResponseShaper::new(false, Some(&tools), "x".into());
        stream.push(literal);
        stream.finish(false);
        assert_eq!(call_arguments(&stream)["path"], "```xml\nkeep\n```");
    }
}
