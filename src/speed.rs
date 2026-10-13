//! Small served-generation speed check against an existing HTTP(S) server.

use std::io::{self, BufRead, IsTerminal, Write};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use rand_core::{OsRng, TryRngCore};
use serde_json::{Value, json};
use ureq::{Agent, http::Uri};

mod health;
use health::{ServerHealth, format_conditions, optional_text, paired_text};

const DEFAULT_URL: &str = "http://127.0.0.1:8080";
const MAX_BODY_BYTES: u64 = 1 << 20;
const PREFILL_OUTPUT_TOKENS: usize = 8;
const LONG_PREFILL_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const PREFILL_NOTES: [&str; 6] = [
    "The service accepts one request at a time. It validates input before allocation, keeps completed conversation state, and releases temporary buffers after the operation finishes. Cancellation stops new work but does not discard an earlier completed response.",
    "A storage worker reads the requested records in order. Every record has a key, a length and a payload. Readers may prepare later records concurrently, but the owner publishes a record only after the entire read completes. A short read returns an error.",
    "The cache has a byte limit and tracks recent use. When a new entry does not fit, the oldest unused entry is removed. Active readers keep their entries alive. A cache miss follows the normal storage path and must produce the same valid application result.",
    "Tests cover startup, a successful request, cancellation, an invalid request and recovery. An integration test writes a document, reads it back, changes one field and verifies the saved value. Independent requests should not overwrite another conversation's state.",
    "Each measurement reports completed work rather than queued work. Record the input size, elapsed time, output count and cache hits. Compare the same workload under both implementations. Keep small request latency separate from sustained throughput on larger inputs.",
    "The deployment has a memory ceiling and a bounded queue. Reject work that exceeds the supported context before expensive processing begins. Reduce optional cached state under pressure, preserve committed data, and resume ordinary operation when resources recover.",
];

fn prefill_prompt(approx_tokens: usize, nonce: u64) -> String {
    let mut prompt = format!(
        "{nonce:016x}\nRead these engineering review notes. After the final note, reply only with READY.\n"
    );
    // An approximate Qwen token count for these notes, not a tokenizer. The
    // reported throughput always uses the server's actual new-token count.
    let target_bytes = approx_tokens * 103 / 20;
    let mut note = 0;
    while prompt.len() < target_bytes {
        prompt.push_str(&format!(
            "\nNote {}: {}\n",
            note + 1,
            PREFILL_NOTES[note % PREFILL_NOTES.len()]
        ));
        note += 1;
    }
    prompt.push_str("\nEnd of notes. Reply only with READY.");
    prompt
}

const TOPICS: [(&str, &str, &str); 3] = [
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
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SpeedOptions {
    origin: String,
    max_tokens: usize,
    timeout: Duration,
    timeout_explicit: bool,
    stream: bool,
    compare_id: Option<u64>,
}

impl SpeedOptions {
    pub(crate) fn parse(mut args: impl Iterator<Item = String>) -> Result<Self> {
        let mut url = DEFAULT_URL.to_owned();
        let mut max_tokens = 256;
        let mut timeout = Duration::from_secs(300);
        let mut stream = false;
        let mut seen_url = false;
        let mut seen_tokens = false;
        let mut seen_timeout = false;
        let mut compare_id = None;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--compare-id" => {
                    ensure!(compare_id.is_none(), "--compare-id specified twice");
                    compare_id = Some(
                        args.next()
                            .context("--compare-id requires an integer")?
                            .parse()
                            .context("--compare-id requires an unsigned 64-bit integer")?,
                    );
                }
                "--url" => {
                    ensure!(!seen_url, "--url specified twice");
                    url = args.next().context("--url requires an HTTP(S) origin")?;
                    seen_url = true;
                }
                "--max-tokens" => {
                    ensure!(!seen_tokens, "--max-tokens specified twice");
                    max_tokens = args
                        .next()
                        .context("--max-tokens requires a value")?
                        .parse()
                        .context("--max-tokens requires an integer 2..=2048")?;
                    seen_tokens = true;
                }
                "--timeout" => {
                    ensure!(!seen_timeout, "--timeout specified twice");
                    let seconds: f64 = args
                        .next()
                        .context("--timeout requires a value")?
                        .parse()
                        .context("--timeout requires positive finite seconds")?;
                    ensure!(
                        seconds.is_finite() && seconds > 0.0,
                        "--timeout requires positive finite seconds"
                    );
                    timeout = Duration::try_from_secs_f64(seconds)
                        .context("--timeout exceeds the supported duration")?;
                    ensure!(
                        !timeout.is_zero()
                            && timeout <= Duration::from_secs(24 * 60 * 60)
                            && std::time::Instant::now().checked_add(timeout).is_some(),
                        "--timeout must resolve to a nonzero duration no longer than 24 hours"
                    );
                    seen_timeout = true;
                }
                "--stream" => {
                    ensure!(!stream, "--stream specified twice");
                    stream = true;
                }
                _ => bail!("unknown speed option: {arg}"),
            }
        }
        ensure!(
            (2..=2048).contains(&max_tokens),
            "--max-tokens must be 2..=2048"
        );
        Ok(Self {
            origin: validate_origin(&url)?,
            max_tokens,
            timeout,
            timeout_explicit: seen_timeout,
            stream,
            compare_id,
        })
    }

    fn long_prefill_timeout(&self) -> Duration {
        if self.timeout_explicit {
            self.timeout
        } else {
            LONG_PREFILL_TIMEOUT
        }
    }
}

fn validate_origin(raw: &str) -> Result<String> {
    ensure!(
        raw.trim() == raw && !raw.contains('#'),
        "--url must be an HTTP(S) origin without fragment"
    );
    let uri: Uri = raw.parse().context("--url must be an HTTP(S) origin")?;
    let scheme = uri
        .scheme_str()
        .context("--url requires an HTTP(S) scheme")?;
    ensure!(
        scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https"),
        "--url requires an HTTP(S) scheme"
    );
    let authority = uri.authority().context("--url requires a host")?;
    ensure!(
        uri.host().is_some_and(|host| !host.is_empty()) && !authority.as_str().contains('@'),
        "--url must not contain credentials"
    );
    let text = authority.as_str();
    let port = if let Some(bracketed) = text.strip_prefix('[') {
        let close = bracketed
            .find(']')
            .context("--url has an invalid IPv6 host")?;
        let suffix = &bracketed[close + 1..];
        if suffix.is_empty() {
            None
        } else {
            Some(
                suffix
                    .strip_prefix(':')
                    .context("--url has an invalid port")?,
            )
        }
    } else {
        ensure!(
            text.matches(':').count() <= 1,
            "--url requires bracketed IPv6"
        );
        text.rsplit_once(':').map(|(_, port)| port)
    };
    if let Some(port) = port {
        ensure!(
            !port.is_empty()
                && port.bytes().all(|byte| byte.is_ascii_digit())
                && port.parse::<u16>().is_ok_and(|value| value > 0),
            "--url has an invalid port"
        );
    }
    ensure!(
        uri.path().is_empty() || uri.path() == "/",
        "--url must not contain a path"
    );
    ensure!(uri.query().is_none(), "--url must not contain a query");
    Ok(format!("{}://{}", scheme.to_ascii_lowercase(), authority))
}

#[derive(Clone, Copy)]
struct Case<'a> {
    label: &'static str,
    locality: Option<&'static str>,
    prompt: &'a str,
    cap: usize,
}

fn schedule(max_tokens: usize) -> Vec<Case<'static>> {
    let mut cases = vec![Case {
        label: "warmup",
        locality: None,
        prompt: "how much is 3456+60?",
        cap: max_tokens.min(64),
    }];
    for &(label, switch, same) in &TOPICS {
        cases.push(Case {
            label,
            locality: Some("topic switch"),
            prompt: switch,
            cap: max_tokens,
        });
        cases.push(Case {
            label,
            locality: Some("same topic"),
            prompt: same,
            cap: max_tokens,
        });
    }
    cases
}

fn request_body(case: Case<'_>, stream: bool) -> Value {
    let mut body = json!({
        "messages": [{"role": "user", "content": case.prompt}],
        "max_tokens": case.cap,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "stream": stream,
    });
    if stream {
        body["stream_options"] = json!({"include_usage": true});
    }
    body
}

fn agent(timeout: Duration) -> Agent {
    let config = Agent::config_builder()
        .timeout_global(Some(timeout))
        .max_redirects(0)
        .proxy(None)
        .build();
    Agent::new_with_config(config)
}

fn bounded_response(mut response: ureq::http::Response<ureq::Body>, limit: u64) -> Result<Vec<u8>> {
    Ok(response
        .body_mut()
        .with_config()
        .limit(limit)
        .read_to_vec()?)
}

fn send_completion(agent: &Agent, options: &SpeedOptions, case: Case<'_>) -> Result<Value> {
    let mut body = request_body(case, options.stream);
    if let Some(id) = options.compare_id {
        body["messages"][0]["content"] =
            json!(format!("Comparison {id}: {}\n{}", case.label, case.prompt));
        body["chat_template_kwargs"] = json!({"enable_thinking": true, "preserve_thinking": true});
        body["reasoning_effort"] = json!("medium");
    }
    let body = serde_json::to_vec(&body)?;
    let endpoint = format!("{}/v1/chat/completions", options.origin);
    let response = agent
        .post(&endpoint)
        .header("content-type", "application/json")
        .send(body.as_slice())
        .with_context(|| format!("{} request failed", case.label))?;
    ensure!(
        response.status().as_u16() == 200,
        "{} returned HTTP {}",
        case.label,
        response.status()
    );
    let bytes = bounded_response(response, MAX_BODY_BYTES)?;
    if options.stream {
        parse_sse_usage(&bytes)
    } else {
        let value: Value =
            serde_json::from_slice(&bytes).context("completion response is not JSON")?;
        ensure!(value.is_object(), "completion response is not an object");
        ensure!(
            value.get("error").is_none_or(Value::is_null),
            "server returned a completion error"
        );
        ensure!(
            has_terminal_choice(&value),
            "completion response lacked a terminal choice"
        );
        Ok(value)
    }
}

fn parse_sse_usage(bytes: &[u8]) -> Result<Value> {
    let text = std::str::from_utf8(bytes).context("SSE response is not UTF-8")?;
    let mut usage = Value::Null;
    let mut done = false;
    let mut finish_reason = None;
    for line in text.lines() {
        let line = line.trim_end_matches('\r');
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        ensure!(!done, "SSE data followed [DONE]");
        if data == "[DONE]" {
            done = true;
            continue;
        }
        let event: Value = serde_json::from_str(data).context("invalid SSE data event")?;
        ensure!(event.is_object(), "SSE data event is not an object");
        ensure!(
            event.get("error").is_none_or(Value::is_null),
            "server returned an SSE error event"
        );
        if let Some(reason) = terminal_reason(&event) {
            finish_reason = Some(reason.to_owned());
        }
        if event.get("usage").is_some_and(|value| !value.is_null()) {
            ensure!(usage.is_null(), "SSE response repeated final usage");
            usage = event["usage"].clone();
        }
    }
    ensure!(done, "SSE response ended without [DONE]");
    ensure!(
        finish_reason.is_some(),
        "SSE response lacked a terminal choice"
    );
    Ok(json!({"usage": usage, "choices": [{"finish_reason": finish_reason}]}))
}

fn has_terminal_choice(value: &Value) -> bool {
    terminal_reason(value).is_some()
}

fn terminal_reason(value: &Value) -> Option<&str> {
    value["choices"].as_array()?.iter().find_map(|choice| {
        choice["finish_reason"]
            .as_str()
            .filter(|reason| !reason.is_empty())
    })
}

fn server_health(agent: &Agent, origin: &str) -> Option<ServerHealth> {
    let response = agent.get(&format!("{origin}/health")).call().ok()?;
    if response.status().as_u16() != 200 {
        return None;
    }
    let value: Value = serde_json::from_slice(&bounded_response(response, 64 * 1024).ok()?).ok()?;
    ServerHealth::parse(&value)
}

#[derive(Default)]
struct Metrics {
    rate: Option<f64>,
    first: Option<f64>,
    prefill_rate: Option<f64>,
    prefill_seconds: Option<f64>,
    prefill_tokens: Option<u64>,
    cached_tokens: Option<u64>,
    ready_first: Option<f64>,
    effective_prefill_rate: Option<f64>,
    completion_tokens: Option<u64>,
    finish_reason: Option<String>,
}

fn finite_number(value: &Value) -> Option<f64> {
    value.as_f64().filter(|number| number.is_finite())
}

fn metrics(response: &Value) -> Metrics {
    let usage = response.get("usage").unwrap_or(response);
    let timing = &usage["moespresso"];
    let total = finite_number(&timing["generation_seconds"]);
    let first = finite_number(&timing["first_token_seconds"]);
    let count = usage["completion_tokens"].as_u64();
    let mut result = match (total, first) {
        (Some(total), Some(first)) if total >= first && first >= 0.0 => Metrics {
            rate: count.and_then(|count| {
                if count > 1 && total > first {
                    let rate = (count - 1) as f64 / (total - first);
                    rate.is_finite().then_some(rate)
                } else {
                    None
                }
            }),
            first: Some(first),
            ..Metrics::default()
        },
        _ => Metrics::default(),
    };
    result.prefill_seconds = finite_number(&timing["prefill_seconds"]).filter(|v| *v >= 0.0);
    result.prefill_tokens = timing["prefill_tokens"].as_u64();
    result.cached_tokens = usage["prompt_tokens_details"]["cached_tokens"].as_u64();
    // Prompt-token counts can include a reused prefix. Infer newly evaluated
    // tokens only when the response explicitly reports no cache reuse.
    if result.prefill_tokens.is_none() && result.cached_tokens == Some(0) {
        result.prefill_tokens = usage["prompt_tokens"].as_u64();
    }
    result.completion_tokens = count;
    result.finish_reason = terminal_reason(response).map(str::to_owned);
    if result.cached_tokens == Some(0) {
        result.effective_prefill_rate =
            usage["prompt_tokens"]
                .as_u64()
                .zip(result.first)
                .and_then(|(tokens, seconds)| {
                    (tokens > 0 && seconds > 0.0)
                        .then_some(tokens as f64 / seconds)
                        .filter(|rate| rate.is_finite())
                });
    }
    result.prefill_rate =
        result
            .prefill_tokens
            .zip(result.prefill_seconds)
            .and_then(|(tokens, seconds)| {
                if tokens > 0 && seconds > 0.0 {
                    let rate = tokens as f64 / seconds;
                    rate.is_finite().then_some(rate)
                } else {
                    None
                }
            });
    result.ready_first =
        finite_number(&timing["ready_to_first_token_seconds"]).filter(|v| *v >= 0.0);
    result
}

fn comparison_issue(result: &Metrics, cap: usize) -> Option<&'static str> {
    if result.cached_tokens != Some(0) {
        Some("cached or unverified prefix")
    } else if result.completion_tokens != Some(cap as u64)
        || result.finish_reason.as_deref() != Some("length")
    {
        Some("early stop or unmatched output count")
    } else {
        None
    }
}

fn print_effective_prefill(label: &str, result: &Metrics, out: &mut impl Write) -> Result<()> {
    writeln!(
        out,
        "{label} effective prefill: {} tok/s (prompt / first sampled token, including first-token preparation; {} s){}",
        number_text(result.effective_prefill_rate),
        number_text(result.first),
        if result.cached_tokens == Some(0) {
            ""
        } else {
            "; cached or unverified prefix, not a cold comparison"
        }
    )?;
    out.flush()?;
    Ok(())
}

fn number_text(value: Option<f64>) -> String {
    value.map_or_else(|| "unavailable".to_owned(), |value| format!("{value:.2}"))
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Some(if sorted.len().is_multiple_of(2) {
        sorted[middle - 1] / 2.0 + sorted[middle] / 2.0
    } else {
        sorted[middle]
    })
}

fn median_text(values: &[f64]) -> String {
    median(values).map_or_else(|| "unavailable".to_owned(), |value| format!("{value:.2}"))
}

fn run_schedule(
    options: &SpeedOptions,
    start_health: Option<ServerHealth>,
    mut complete: impl FnMut(Case<'_>) -> Result<Value>,
    mut sample_end_health: impl FnMut() -> Option<ServerHealth>,
    out: &mut impl Write,
    err: &mut impl Write,
) -> Result<()> {
    let mut switch = Vec::new();
    let mut same = Vec::new();
    let mut all = Vec::new();
    let mut first_times = Vec::new();
    let mut prefill_rates = Vec::new();
    let mut comparable_decode = Vec::new();
    let mut failure = None;
    for case in schedule(options.max_tokens) {
        writeln!(
            err,
            "{}",
            if case.locality.is_none() {
                "Warming up...".to_owned()
            } else {
                format!("{}: {}", case.label, case.locality.unwrap_or_default())
            }
        )?;
        match complete(case) {
            Ok(_) if case.locality.is_none() => {}
            Ok(usage) => {
                let result = metrics(&usage);
                let prefill =
                    if result.prefill_tokens == Some(0) && result.prefill_seconds == Some(0.0) {
                        "prefill skipped (fully cached)".to_owned()
                    } else {
                        format!(
                            "prefill {} tok/s ({} s)",
                            number_text(result.prefill_rate),
                            number_text(result.prefill_seconds)
                        )
                    };
                let count_text = |count: Option<u64>| {
                    count.map_or_else(|| "unavailable".to_owned(), |count| count.to_string())
                };
                writeln!(
                    out,
                    "{}: {} · {prefill} · {} new / {} cached tokens · decode {} tok/s · first token {} s (including cache restore) · {} output tokens · finish {} · effective prefill {} tok/s",
                    case.label,
                    case.locality.unwrap_or_default(),
                    count_text(result.prefill_tokens),
                    count_text(result.cached_tokens),
                    number_text(result.rate),
                    number_text(result.ready_first),
                    count_text(result.completion_tokens),
                    result.finish_reason.as_deref().unwrap_or("unavailable"),
                    number_text(result.effective_prefill_rate)
                )?;
                out.flush()?;
                if options.compare_id.is_some() {
                    if let Some(reason) = comparison_issue(&result, case.cap) {
                        writeln!(out, "Comparison decode excluded: {reason}.")?;
                        out.flush()?;
                    } else if let Some(rate) = result.rate {
                        comparable_decode.push(rate);
                    }
                }
                if let Some(rate) = result.prefill_rate {
                    prefill_rates.push(rate);
                }
                if let Some(rate) = result.rate {
                    all.push(rate);
                    if case.locality == Some("topic switch") {
                        switch.push(rate);
                    } else {
                        same.push(rate);
                    }
                }
                if let Some(first) = result.first {
                    first_times.push(first);
                }
            }
            Err(error) => {
                writeln!(err, "Speed check failed: {error:#}")?;
                failure = Some(error);
                break;
            }
        }
    }
    let end_health = sample_end_health();
    let model = start_health
        .as_ref()
        .and_then(|health| health.model.as_deref())
        .or_else(|| {
            end_health
                .as_ref()
                .and_then(|health| health.model.as_deref())
        });
    let health_status = paired_text(
        start_health
            .as_ref()
            .and_then(|health| health.status.as_deref()),
        end_health
            .as_ref()
            .and_then(|health| health.status.as_deref()),
    );
    writeln!(out, "MoEspresso speed")?;
    writeln!(out, "Model: {}", model.unwrap_or("unavailable"))?;
    writeln!(out, "Server health (start → end): {health_status}")?;
    writeln!(
        out,
        "Prefill: {} tok/s median (new prompt tokens; includes checkpoint writes, excludes cache restore)",
        median_text(&prefill_rates)
    )?;
    writeln!(
        out,
        "Decode: topic switch {} · same topic {} · overall {} tok/s",
        median_text(&switch),
        median_text(&same),
        median_text(&all)
    )?;
    let first = median(&first_times)
        .map_or_else(|| "unavailable".to_owned(), |v| format!("{v:.2} s median"));
    writeln!(
        out,
        "First sampled token (post-lookup): {first} · greedy · {}/6 requests · {}-token cap",
        all.len(),
        options.max_tokens
    )?;
    writeln!(
        out,
        "Conditions: {}",
        format_conditions(start_health.as_ref(), end_health.as_ref())
    )?;
    if options.compare_id.is_some() {
        writeln!(
            out,
            "Comparison decode: {} tok/s median · {}/6 uncached, cap-reaching requests (requires all six; compare individual matched cases otherwise)",
            if comparable_decode.len() == 6 {
                median_text(&comparable_decode)
            } else {
                "unavailable".to_owned()
            },
            comparable_decode.len()
        )?;
        out.flush()?;
    }
    if let Some(error) = failure {
        return Err(error);
    }
    ensure!(!all.is_empty(), "server supplied no usable decode timing");
    Ok(())
}

fn prefill_context_fits(
    max_context: Option<u64>,
    approx_tokens: usize,
    label: &str,
    out: &mut impl Write,
) -> Result<bool> {
    // The client has no tokenizer. Leave room for the chat template, output and
    // the last paragraph beyond the approximate target.
    let required = (approx_tokens + PREFILL_OUTPUT_TOKENS + 256) as u64;
    if let Some(limit) = max_context
        && limit < required
    {
        writeln!(
            out,
            "{label} prefill skipped: server context limit is {limit} tokens; needs about {required}."
        )?;
        out.flush()?;
        return Ok(false);
    }
    Ok(true)
}

fn run_prefill_case(
    label: &'static str,
    approx_tokens: usize,
    timeout: Duration,
    compare_id: Option<u64>,
    complete: &mut impl FnMut(Case<'_>, Duration) -> Result<Value>,
    out: &mut impl Write,
) -> Result<()> {
    let nonce = match compare_id {
        Some(id) => id.wrapping_add(approx_tokens as u64),
        None => OsRng
            .try_next_u64()
            .context("could not create a fresh prefill prompt")?,
    };
    let prompt = prefill_prompt(approx_tokens, nonce);
    writeln!(
        out,
        "{label} prefill: running with a benchmark prompt, {PREFILL_OUTPUT_TOKENS}-token output cap and {} s timeout...",
        timeout.as_secs_f64()
    )?;
    out.flush()?;
    let usage = complete(
        Case {
            label,
            locality: None,
            prompt: &prompt,
            cap: PREFILL_OUTPUT_TOKENS,
        },
        timeout,
    )?;
    let result = metrics(&usage);
    writeln!(
        out,
        "{label} prefill: {} new / {} cached tokens · {} s · {} tok/s",
        optional_text(result.prefill_tokens),
        optional_text(result.cached_tokens),
        number_text(result.prefill_seconds),
        number_text(result.prefill_rate),
    )?;
    out.flush()?;
    print_effective_prefill(label, &result, out)?;
    Ok(())
}

fn confirm_long_prefill(input: &mut impl BufRead, out: &mut impl Write) -> Result<bool> {
    loop {
        write!(out, "Run the longer ~32K prefill test? [y/N] ")?;
        out.flush()?;
        let mut answer = String::new();
        let read = input.read_line(&mut answer)?;
        if read == 0 {
            writeln!(out)?;
            return Ok(false);
        }
        match answer.trim().to_ascii_lowercase().as_str() {
            "y" | "yes" => return Ok(true),
            "" | "n" | "no" => return Ok(false),
            _ => writeln!(out, "Please enter y or n (Enter skips).")?,
        }
    }
}

fn run_prefill_checks(
    options: &SpeedOptions,
    max_context: Option<u64>,
    interactive: bool,
    input: &mut impl BufRead,
    mut complete: impl FnMut(Case<'_>, Duration) -> Result<Value>,
    out: &mut impl Write,
) -> Result<()> {
    out.flush()?;
    writeln!(out, "\nLonger prefill checks (separate from topic medians)")?;
    if prefill_context_fits(max_context, 4096, "~4K", out)? {
        run_prefill_case(
            "~4K",
            4096,
            options.timeout,
            options.compare_id,
            &mut complete,
            out,
        )?;
    }
    if !prefill_context_fits(max_context, 32768, "~32K", out)? {
        return Ok(());
    }
    if !interactive {
        writeln!(out, "~32K prefill skipped: stdin is not a terminal.")?;
        out.flush()?;
        return Ok(());
    }
    if confirm_long_prefill(input, out)? {
        run_prefill_case(
            "~32K",
            32768,
            options.long_prefill_timeout(),
            options.compare_id,
            &mut complete,
            out,
        )?;
    } else {
        writeln!(out, "~32K prefill skipped.")?;
        out.flush()?;
    }
    Ok(())
}

pub(crate) fn run(options: SpeedOptions) -> Result<()> {
    let client = agent(options.timeout);
    let start_health = server_health(&client, &options.origin);
    if start_health.is_none() {
        eprintln!("Run conditions unavailable; continuing with timing only.");
    }
    let stdout = io::stdout();
    let stderr = io::stderr();
    let stdin = io::stdin();
    let mut out = stdout.lock();
    if let Some(id) = options.compare_id {
        writeln!(
            out,
            "Comparison ID: {id} · greedy · thinking on · reasoning medium"
        )?;
        writeln!(
            out,
            "Use the same ID, token cap and client build for both servers. Cached requests are not cold comparisons."
        )?;
        out.flush()?;
    }
    let max_context = start_health.as_ref().and_then(|health| health.max_context);
    run_schedule(
        &options,
        start_health,
        |case| send_completion(&client, &options, case),
        || server_health(&client, &options.origin),
        &mut out,
        &mut stderr.lock(),
    )?;
    run_prefill_checks(
        &options,
        max_context,
        stdin.is_terminal(),
        &mut stdin.lock(),
        |case, timeout| send_completion(&agent(timeout), &options, case),
        &mut out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::io::{Cursor, Read};
    use std::rc::Rc;
    use std::thread;
    use tiny_http::{Header, Method, Response, Server, StatusCode};

    fn mock_completion(
        body: impl Into<String>,
        content_type: &'static str,
        status: u16,
    ) -> (String, thread::JoinHandle<()>) {
        let body = body.into();
        let server = Server::http("127.0.0.1:0").unwrap();
        let port = server.server_addr().to_ip().unwrap().port();
        let worker = thread::spawn(move || {
            let mut request = server
                .recv_timeout(Duration::from_secs(3))
                .unwrap()
                .unwrap();
            assert_eq!(request.method(), &Method::Post);
            assert_eq!(request.url(), "/v1/chat/completions");
            let mut posted = String::new();
            request.as_reader().read_to_string(&mut posted).unwrap();
            let value: Value = serde_json::from_str(&posted).unwrap();
            assert_eq!(value["temperature"], 0.0);
            assert_eq!(value["top_p"], 1.0);
            assert_eq!(value["top_k"], 0);
            assert_eq!(value["min_p"], 0.0);
            assert_eq!(value["presence_penalty"], 0.0);
            assert_eq!(value["messages"][0]["role"], "user");
            let header = Header::from_bytes("Content-Type", content_type).unwrap();
            request
                .respond(
                    Response::from_string(body)
                        .with_status_code(StatusCode(status))
                        .with_header(header),
                )
                .unwrap();
        });
        (format!("http://127.0.0.1:{port}"), worker)
    }

    fn opts() -> SpeedOptions {
        SpeedOptions::parse(std::iter::empty()).unwrap()
    }

    #[derive(Clone, Default)]
    struct ObservedOutput(Rc<RefCell<(Vec<u8>, usize)>>);

    impl Write for ObservedOutput {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.borrow_mut().0.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            let mut state = self.0.borrow_mut();
            state.1 = state.0.len();
            Ok(())
        }
    }

    impl ObservedOutput {
        fn flushed(&self) -> String {
            let state = self.0.borrow();
            String::from_utf8(state.0[..state.1].to_vec()).unwrap()
        }
    }

    struct ConfirmInput {
        inner: Cursor<Vec<u8>>,
        output: ObservedOutput,
    }

    impl Read for ConfirmInput {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
            self.inner.read(bytes)
        }
    }

    impl BufRead for ConfirmInput {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            let printed = self.output.flushed();
            assert!(
                printed
                    .contains("~4K prefill: 4081 new / 16 cached tokens · 50.00 s · 81.62 tok/s")
            );
            assert!(printed.ends_with("Run the longer ~32K prefill test? [y/N] "));
            self.inner.fill_buf()
        }

        fn consume(&mut self, amount: usize) {
            self.inner.consume(amount);
        }
    }

    fn prefill_usage(tokens: u64, seconds: f64) -> Value {
        json!({"prompt_tokens": tokens + 16,
            "prompt_tokens_details": {"cached_tokens": 16},
            "moespresso": {"prefill_tokens": tokens, "prefill_seconds": seconds}})
    }

    #[test]
    fn fresh_prefill_runs_after_topics_then_obeys_terminal_confirmation() {
        for (answer, expected_calls) in [
            ("y\n", 2),
            ("yes\n", 2),
            (" YES \n", 2),
            ("n\n", 1),
            ("no\n", 1),
            ("\n", 1),
            ("", 1),
            ("maybe\nyes\n", 2),
        ] {
            let options = opts();
            let mut out = ObservedOutput::default();
            let mut topic_calls = 0;
            run_schedule(
                &options,
                None,
                |_| {
                    topic_calls += 1;
                    Ok(json!({"completion_tokens": 3, "moespresso": {
                        "generation_seconds": 2.0, "first_token_seconds": 1.0,
                        "prefill_tokens": 100, "prefill_seconds": 2.0}}))
                },
                || None,
                &mut out,
                &mut Vec::new(),
            )
            .unwrap();
            assert_eq!(topic_calls, 7);
            let mut input = ConfirmInput {
                inner: Cursor::new(answer.as_bytes().to_vec()),
                output: out.clone(),
            };
            let observed = out.clone();
            let mut prompts = Vec::new();
            run_prefill_checks(
                &options,
                Some(131072),
                true,
                &mut input,
                |case, timeout| {
                    let printed = observed.flushed();
                    assert!(printed.contains("MoEspresso speed"));
                    assert!(printed.contains("6/6 requests"));
                    assert!(printed.contains(&format!("{} prefill: running", case.label)));
                    assert_eq!(case.cap, PREFILL_OUTPUT_TOKENS);
                    prompts.push(case.prompt.to_owned());
                    match case.label {
                        "~4K" => {
                            assert_eq!(timeout, options.timeout);
                            Ok(prefill_usage(4081, 50.0))
                        }
                        "~32K" => {
                            assert!(printed.contains("~4K prefill: 4081 new"));
                            assert_eq!(timeout, LONG_PREFILL_TIMEOUT);
                            Ok(prefill_usage(32701, 650.0))
                        }
                        _ => panic!("unexpected prefill case"),
                    }
                },
                &mut out,
            )
            .unwrap();
            assert_eq!(prompts.len(), expected_calls);
            let printed = out.flushed();
            assert_eq!(printed.matches("Prefill: 50.00 tok/s median").count(), 1);
            assert_eq!(printed.matches("MoEspresso speed").count(), 1);
            if expected_calls == 2 {
                assert!(printed.contains(
                    "~32K prefill: 32701 new / 16 cached tokens · 650.00 s · 50.31 tok/s"
                ));
                assert_ne!(prompts[0].lines().next(), prompts[1].lines().next());
            } else {
                assert!(printed.ends_with("~32K prefill skipped.\n"));
            }
            if answer.starts_with("maybe") {
                assert_eq!(printed.matches("[y/N]").count(), 2);
                assert!(printed.contains("Please enter y or n"));
            }
        }
    }

    #[test]
    fn long_prefill_skips_noninteractive_and_insufficient_context_without_reading_stdin() {
        for (interactive, max_context, expected_calls, skip) in [
            (false, Some(131072), 1, "stdin is not a terminal"),
            (true, Some(32768), 1, "server context limit is 32768 tokens"),
            (true, Some(2048), 0, "~4K prefill skipped"),
        ] {
            let mut calls = 0;
            let mut input = Cursor::new(b"yes\n");
            let mut out = Vec::new();
            run_prefill_checks(
                &opts(),
                max_context,
                interactive,
                &mut input,
                |case, _| {
                    assert_eq!(case.label, "~4K");
                    calls += 1;
                    Ok(prefill_usage(4081, 50.0))
                },
                &mut out,
            )
            .unwrap();
            assert_eq!(calls, expected_calls);
            assert_eq!(input.position(), 0);
            let printed = String::from_utf8(out).unwrap();
            assert!(printed.contains(skip));
            assert!(!printed.contains("[y/N]"));
        }
    }

    #[test]
    fn long_prefill_honors_explicit_timeout_and_reports_missing_timing_honestly() {
        let options =
            SpeedOptions::parse(["--timeout", "42.5"].map(str::to_owned).into_iter()).unwrap();
        let mut calls = 0;
        let mut out = Vec::new();
        run_prefill_checks(
            &options,
            None,
            true,
            &mut Cursor::new(b"y\n"),
            |_, timeout| {
                calls += 1;
                assert_eq!(timeout, Duration::from_secs_f64(42.5));
                Ok(Value::Null)
            },
            &mut out,
        )
        .unwrap();
        assert_eq!(calls, 2);
        let printed = String::from_utf8(out).unwrap();
        assert!(printed.contains("42.5 s timeout"));
        assert!(printed.contains("~32K prefill: unavailable new / unavailable cached tokens · unavailable s · unavailable tok/s"));
    }

    #[test]
    fn prefill_prompts_start_fresh_and_have_bounded_approximate_lengths() {
        for tokens in [4096, 32768] {
            let first = prefill_prompt(tokens, 1);
            let second = prefill_prompt(tokens, 2);
            assert!(first.starts_with("0000000000000001\n"));
            assert!(second.starts_with("0000000000000002\n"));
            assert_eq!(&first[16..], &second[16..]);
            assert!(first.len() >= tokens * 103 / 20);
            assert!(first.len() < tokens * 103 / 20 + 512);
            assert!(first.ends_with("Reply only with READY."));
        }
    }

    #[test]
    fn comparison_ids_repeat_both_long_prompts_and_remain_separate() {
        let collect = |id: u64| {
            let mut options = opts();
            options.compare_id = Some(id);
            let mut prompts = Vec::new();
            run_prefill_checks(
                &options,
                Some(131072),
                true,
                &mut Cursor::new(b"y\n"),
                |case, _| {
                    prompts.push(case.prompt.to_owned());
                    Ok(Value::Null)
                },
                &mut Vec::new(),
            )
            .unwrap();
            prompts
        };
        let pair = collect(42);
        assert_eq!(pair, collect(42));
        assert_ne!(pair[0], pair[1]);
        assert_ne!(pair, collect(43));
        for args in [
            vec!["--compare-id"],
            vec!["--compare-id", "-1"],
            vec!["--compare-id", "18446744073709551616"],
            vec!["--compare-id", "1", "--compare-id", "2"],
        ] {
            assert!(SpeedOptions::parse(args.into_iter().map(str::to_owned)).is_err());
        }
    }

    #[test]
    fn exact_v3_schedule_and_sampling() {
        let cases = schedule(128);
        assert_eq!(cases.len(), 7);
        assert_eq!(cases[0].prompt, "how much is 3456+60?");
        assert_eq!(cases[0].cap, 64);
        assert_eq!(
            cases[1].prompt,
            "Explain the causes of the French Revolution."
        );
        assert_eq!(
            cases[2].prompt,
            "Explain how the French Revolution changed European politics."
        );
        assert_eq!(
            cases[3].prompt,
            "Explain how to implement Fibonacci numbers in Python."
        );
        assert_eq!(
            cases[4].prompt,
            "Explain how memoization improves a recursive Fibonacci function."
        );
        assert_eq!(cases[5].prompt, "Who was Ada Lovelace?");
        assert_eq!(
            cases[6].prompt,
            "Explain Ada Lovelace's contribution to computing."
        );
        for case in cases {
            let body = request_body(case, false);
            assert_eq!(body["temperature"], 0.0);
            assert_eq!(body["top_p"], 1.0);
            assert_eq!(body["top_k"], 0);
            assert_eq!(body["min_p"], 0.0);
            assert_eq!(body["presence_penalty"], 0.0);
            assert_eq!(body["stream"], false);
        }
    }

    #[test]
    fn origin_and_option_validation() {
        assert_eq!(opts().origin, DEFAULT_URL);
        assert_eq!(opts().max_tokens, 256);
        assert_eq!(opts().timeout, Duration::from_secs(300));
        assert_eq!(
            validate_origin("https://example.org/").unwrap(),
            "https://example.org"
        );
        assert_eq!(
            validate_origin("http://[::1]:8080/").unwrap(),
            "http://[::1]:8080"
        );
        for url in [
            "http://",
            "ftp://example.org",
            "http://user:pass@example.org",
            "http://example.org/path",
            "http://example.org/?q=1",
            "http://example.org/#fragment",
            "http://example.org:",
            "http://example.org:bogus",
            "http://example.org:0",
            "http://example.org:65536",
            "http://:8080",
            "http://::1:8080",
            " http://example.org",
            "http://example.org ",
        ] {
            assert!(validate_origin(url).is_err(), "accepted {url}");
        }
        for args in [
            vec!["--max-tokens", "1"],
            vec!["--max-tokens", "2049"],
            vec!["--timeout", "0"],
            vec!["--timeout", "NaN"],
            vec!["--timeout", "inf"],
            vec!["--timeout", "0.00000000001"],
            vec!["--timeout", "1e18"],
            vec!["--url", "http://example.org/path"],
            vec!["--stream", "--stream"],
        ] {
            assert!(SpeedOptions::parse(args.into_iter().map(str::to_owned)).is_err());
        }
    }

    #[test]
    fn invalid_or_missing_timing_does_not_become_client_wall_time() {
        assert!(metrics(&Value::Null).rate.is_none());
        assert!(
            metrics(&json!({"completion_tokens": 4, "moespresso": {
            "generation_seconds": 1.0, "first_token_seconds": 1.0}}))
            .rate
            .is_none()
        );
        assert!(
            metrics(&json!({"completion_tokens": 4, "moespresso": {
            "generation_seconds": 1.0, "first_token_seconds": 2.0}}))
            .first
            .is_none()
        );
        let valid = metrics(&json!({"completion_tokens": 4, "moespresso": {
            "generation_seconds": 2.0, "first_token_seconds": 1.0}}));
        assert_eq!(valid.rate, Some(3.0));
        assert_eq!(valid.first, Some(1.0));
    }

    #[test]
    fn prefill_counts_new_work_and_handles_complete_hits_and_old_servers() {
        let fresh = metrics(&json!({"prompt_tokens": 2148,
            "prompt_tokens_details": {"cached_tokens": 2048}, "moespresso": {
            "prefill_tokens": 100, "prefill_seconds": 2.0}}));
        assert_eq!(fresh.prefill_rate, Some(50.0));
        assert_eq!(fresh.cached_tokens, Some(2048));
        for timing in [
            json!({"prefill_tokens": 0, "prefill_seconds": 0.0}),
            json!({"prefill_tokens": 100, "prefill_seconds": 0.0}),
            json!({"prefill_tokens": 100, "prefill_seconds": -1.0}),
            json!({"first_token_seconds": 2.0}),
        ] {
            assert!(
                metrics(&json!({"prompt_tokens": 2148, "moespresso": timing}))
                    .prefill_rate
                    .is_none()
            );
        }
    }

    #[test]
    fn health_conditions_report_capacity_changes_and_missing_fields() {
        let start = ServerHealth::parse(&json!({
            "status": "ok",
            "model": "qwen-test",
            "expert_capacity": 64,
            "max_context": 131072,
            "prefill_capacity": 512,
            "adaptive_memory": true,
            "cache_routing": "prefer-resident",
            "prompt_cache": {
                "memory_enabled": true,
                "disk_enabled": true,
                "retained_bytes": 1073741824
            }
        }))
        .unwrap();
        let end = ServerHealth::parse(&json!({
            "status": "ok",
            "model": "qwen-test",
            "expert_capacity": 80,
            "max_context": 131072,
            "prefill_capacity": 512,
            "adaptive_memory": true,
            "cache_routing": "prefer-resident",
            "prompt_cache": {
                "memory_enabled": true,
                "disk_enabled": true,
                "retained_bytes": 1073741824
            }
        }))
        .unwrap();
        let formatted = format_conditions(Some(&start), Some(&end));
        assert!(formatted.contains("expert capacity 64 → 80 slots"));
        assert!(formatted.contains("max context 131072 → 131072 tokens"));
        assert!(formatted.contains("adaptive on → on"));
        assert!(formatted.contains("decode routing prefer-resident → prefer-resident"));
        assert!(formatted.contains("memory cache on → on"));
        assert!(formatted.contains("disk cache on → on"));
        assert!(formatted.contains("memory retained 1073741824 → 1073741824 bytes"));

        let partial = ServerHealth::parse(&json!({"status": "ok", "model": "qwen-test"})).unwrap();
        let missing = format_conditions(Some(&partial), None);
        assert!(missing.contains("expert capacity unavailable → unavailable slots"));
        assert!(missing.contains("memory cache unavailable → unavailable"));
        assert_eq!(format_conditions(None, None), "unavailable");
        assert!(ServerHealth::parse(&Value::Null).is_none());
    }

    #[test]
    fn sse_final_usage_and_terminal_marker_required() {
        let body = b"data: {\"choices\":[],\"usage\":null}\n\ndata: {\"choices\":[{\"finish_reason\":\"stop\"}],\"usage\":null}\n\ndata: {\"choices\":[],\"usage\":{\"completion_tokens\":2}}\n\ndata: [DONE]\n\n";
        assert_eq!(
            parse_sse_usage(body).unwrap()["usage"]["completion_tokens"],
            2
        );
        assert!(parse_sse_usage(b"data: {\"usage\":{}}\n\n").is_err());
        assert!(parse_sse_usage(b"data: {\"error\":{}}\n\ndata: [DONE]\n\n").is_err());
        assert!(parse_sse_usage(b"data: {\"usage\":{}}\n\ndata: [DONE]\n\n").is_err());
        assert!(parse_sse_usage(b"data: {\"choices\":[{\"finish_reason\":\"stop\"}],\"usage\":{}}\n\ndata: {\"usage\":{}}\n\ndata: [DONE]\n\n").is_err());
        assert!(parse_sse_usage(b"data: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\ndata: {\"usage\":{}}\n\n").is_err());
    }

    #[test]
    fn each_topic_result_is_flushed_before_the_next_request() {
        use std::cell::RefCell;
        use std::rc::Rc;

        #[derive(Clone, Default)]
        struct Output(Rc<RefCell<(Vec<u8>, usize)>>);
        impl Write for Output {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.borrow_mut().0.extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                self.0.borrow_mut().1 += 1;
                Ok(())
            }
        }
        let mut out = Output::default();
        let observed = out.clone();
        let cases = schedule(opts().max_tokens);
        let mut calls = 0;
        run_schedule(
            &opts(),
            None,
            |_| {
                if calls >= 2 {
                    let (bytes, flushes) = &*observed.0.borrow();
                    let printed = std::str::from_utf8(bytes).unwrap();
                    let previous = cases[calls - 1];
                    assert!(printed.contains(&format!(
                        "{}: {}",
                        previous.label,
                        previous.locality.unwrap()
                    )));
                    assert!(printed.contains("prefill 50.00 tok/s"));
                    assert!(printed.contains("decode 2.00 tok/s"));
                    assert!(
                        !printed.contains("MoEspresso speed"),
                        "summary printed before schedule finished"
                    );
                    assert!(*flushes >= calls - 1, "completed result was not flushed");
                }
                calls += 1;
                Ok(json!({"completion_tokens": 3, "prompt_tokens": 612,
                "prompt_tokens_details": {"cached_tokens": 512}, "moespresso": {
                "generation_seconds": 3.0, "first_token_seconds": 2.0,
                "prefill_tokens": 100, "prefill_seconds": 2.0,
                "ready_to_first_token_seconds": 2.2}}))
            },
            || None,
            &mut out,
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(calls, 7);
        let (bytes, flushes) = &*out.0.borrow();
        let text = std::str::from_utf8(bytes).unwrap();
        assert_eq!(text.matches("prefill 50.00 tok/s").count(), 6);
        assert!(text.contains("Prefill:"));
        assert!(*flushes >= 6);
    }

    #[test]
    fn schedule_preserves_partial_results_and_reports_unavailable() {
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut calls = 0;
        let failure = run_schedule(
            &opts(),
            None,
            |_| {
                calls += 1;
                if calls == 3 {
                    bail!("mock failure")
                }
                Ok(json!({"completion_tokens": 3, "moespresso": {
                "generation_seconds": 2.0, "first_token_seconds": 1.0}}))
            },
            || None,
            &mut out,
            &mut err,
        );
        assert!(failure.is_err());
        let out = String::from_utf8(out).unwrap();
        assert!(out.contains("topic switch 2.00 · same topic unavailable · overall 2.00 tok/s"));
        assert!(out.contains("1/6 requests"));
        assert!(String::from_utf8(err).unwrap().contains("mock failure"));

        let mut out = Vec::new();
        let mut err = Vec::new();
        assert!(
            run_schedule(
                &opts(),
                None,
                |_| Ok(Value::Null),
                || None,
                &mut out,
                &mut err,
            )
            .is_err()
        );
        assert!(
            String::from_utf8(out)
                .unwrap()
                .contains("overall unavailable tok/s")
        );
    }

    #[test]
    fn bounded_mock_http_reads_buffered_and_sse_usage() {
        let buffered = r#"{"choices":[{"finish_reason":"stop"}],"usage":{"completion_tokens":3,"moespresso":{"first_token_seconds":1.0,"generation_seconds":2.0}}}"#;
        let (origin, worker) = mock_completion(buffered, "application/json", 200);
        let options = SpeedOptions {
            origin,
            timeout: Duration::from_secs(3),
            ..opts()
        };
        let usage = send_completion(&agent(options.timeout), &options, schedule(8)[1]).unwrap();
        assert_eq!(metrics(&usage).rate, Some(2.0));
        worker.join().unwrap();

        let streamed = "data: {\"choices\":[{\"finish_reason\":\"stop\"}],\"usage\":null}\n\ndata: {\"choices\":[],\"usage\":{\"completion_tokens\":3,\"moespresso\":{\"first_token_seconds\":1.0,\"generation_seconds\":2.0}}}\n\ndata: [DONE]\n\n";
        let (origin, worker) = mock_completion(streamed, "text/event-stream", 200);
        let options = SpeedOptions {
            origin,
            stream: true,
            timeout: Duration::from_secs(3),
            ..opts()
        };
        let usage = send_completion(&agent(options.timeout), &options, schedule(8)[1]).unwrap();
        assert_eq!(metrics(&usage).rate, Some(2.0));
        worker.join().unwrap();
    }

    #[test]
    fn mock_http_rejects_success_status_without_terminal_choice() {
        let (origin, worker) = mock_completion(
            r#"{"usage":{"completion_tokens":3}}"#,
            "application/json",
            200,
        );
        let options = SpeedOptions {
            origin,
            timeout: Duration::from_secs(3),
            ..opts()
        };
        assert!(send_completion(&agent(options.timeout), &options, schedule(8)[1]).is_err());
        worker.join().unwrap();
    }

    #[test]
    fn mock_http_body_limit_and_global_timeout_fail_closed() {
        let (origin, worker) = mock_completion("x".repeat(1024), "application/json", 200);
        let client = agent(Duration::from_secs(3));
        let posted = serde_json::to_vec(&request_body(schedule(8)[1], false)).unwrap();
        let response = client
            .post(&format!("{origin}/v1/chat/completions"))
            .header("content-type", "application/json")
            .send(posted.as_slice())
            .unwrap();
        assert!(bounded_response(response, 16).is_err());
        worker.join().unwrap();

        let server = Server::http("127.0.0.1:0").unwrap();
        let port = server.server_addr().to_ip().unwrap().port();
        let worker = thread::spawn(move || {
            if let Some(request) = server.recv_timeout(Duration::from_secs(2)).unwrap() {
                thread::sleep(Duration::from_millis(250));
                let _ =
                    request.respond(Response::from_string("{}").with_status_code(StatusCode(200)));
            }
        });
        let options = SpeedOptions {
            origin: format!("http://127.0.0.1:{port}"),
            timeout: Duration::from_millis(50),
            ..opts()
        };
        assert!(send_completion(&agent(options.timeout), &options, schedule(8)[1]).is_err());
        worker.join().unwrap();
    }
}
