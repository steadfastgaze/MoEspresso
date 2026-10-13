use std::io::{Read, Write};
use std::net::Shutdown;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use moespresso::engine::DiskCacheMode;
use moespresso::package::Package;
use moespresso::prefill_capacity::first_checkpoint_frontier;
use moespresso::tokenizer::{ChatOptions, ReasoningEffort, TextTokenizer};
use serde_json::{Value, json};

use super::{
    Deadline, RunningServer, Scratch, agent, binary_path, connect_loopback, package_path, post_json,
};

const TOTAL_LIMIT: Duration = Duration::from_secs(20 * 60);
const REQUEST_LIMIT: Duration = Duration::from_secs(120);

pub fn run() -> Result<()> {
    ensure!(
        cfg!(target_os = "macos"),
        "cache roadtest requires macOS and Apple Silicon Metal"
    );
    let deadline = Deadline::after(TOTAL_LIMIT)?;
    for mode in [DiskCacheMode::Prefix, DiskCacheMode::Full] {
        run_mode(mode, deadline)?;
    }
    Ok(())
}

fn run_mode(mode: DiskCacheMode, deadline: Deadline) -> Result<()> {
    let binary = binary_path()?;
    let package_path = package_path()?;
    let package = Package::open(&package_path)?;
    let tokenizer = TextTokenizer::load(&package)?;
    let mut scratch = Scratch::new("cache-roadtest")?;
    let cache_root = scratch.subdir("cache")?;
    let obsolete_payload = inactive_legacy_cache(&cache_root)?;
    let first = RunningServer::start_configured(
        &binary,
        &package_path,
        &scratch,
        &cache_root,
        4096,
        "off",
        false,
        deadline,
        "20",
        |command| {
            if mode == DiskCacheMode::Full {
                command.args(["--disk-cache-mode", "full"]);
            }
        },
    )?;
    let health = first.health(deadline.bounded(Duration::from_secs(5))?)?;
    ensure!(
        health["prompt_cache"]["disk_mode"] == mode.as_str(),
        "wrong disk mode: {health}"
    );
    ensure!(
        !obsolete_payload.exists(),
        "startup did not reclaim the inactive legacy cache over the shared allowance"
    );
    let first_capacities = verify_health(&health, 4096, &package.artifact_id)?;
    println!(
        "cache roadtest: binary={} package={} context=4096 memory_budget_gib=20 expert_capacity={} prefill_capacity={} adaptive_memory=false cache_routing=off memory_cache=true disk_cache=true disk_mode={}",
        binary.display(),
        package.artifact_id,
        first_capacities.0,
        first_capacities.1,
        mode.as_str()
    );
    let prefill_width = first_capacities.1;
    let stride = first_checkpoint_frontier(1024, prefill_width)
        .context("prefill capacity has no 1024-token checkpoint stride")?;
    let (prompt, expected_prompt_tokens) = make_prompt(&tokenizer, stride)?;
    ensure!(
        expected_prompt_tokens <= 4096,
        "rendered test prompt exceeds the fixed context"
    );
    let endpoint = format!("{}/v1/chat/completions", first.origin());
    let request = completion_request(&prompt, 3, false);

    let cold = post_json(
        &agent(deadline.bounded(REQUEST_LIMIT)?),
        &endpoint,
        &request,
    )?;
    check_usage(&cold, "cold_start", 0, expected_prompt_tokens)?;
    let cold_cache = cache_usage(&cold)?;
    if mode == DiskCacheMode::Prefix {
        ensure!(
            cold_cache["disk_checkpoints_written"] == 1
                && cold_cache["last_disk_checkpoint"] == stride as u64,
            "default must save only the completed prefill anchor: {cold_cache}"
        );
    } else {
        ensure!(
            cold_cache["disk_checkpoints_written"]
                .as_u64()
                .is_some_and(|count| count >= 2)
                && cold_cache["last_disk_checkpoint"]
                    .as_u64()
                    .is_some_and(|frontier| frontier >= expected_prompt_tokens as u64
                        && frontier < expected_prompt_tokens as u64 + 3),
            "cold request did not persist its completed prompt and committed generation: {cold_cache}"
        );
    }
    ensure!(
        cold_cache["disk_checkpoint_write_seconds"]
            .as_f64()
            .is_some_and(f64::is_finite),
        "cold request did not report finite checkpoint write timing"
    );

    let memory = post_json(
        &agent(deadline.bounded(REQUEST_LIMIT)?),
        &endpoint,
        &request,
    )?;
    check_usage(
        &memory,
        "memory_hit",
        expected_prompt_tokens,
        expected_prompt_tokens,
    )?;
    ensure!(
        cache_usage(&memory)?["disk_checkpoints_written"].as_u64() == Some(0),
        "memory hit unexpectedly wrote a disk checkpoint"
    );

    // Client title/summary requests must not discard the main conversation.
    for task in [
        "Give this chat a short title.",
        "Summarize the chat in one sentence.",
    ] {
        let helper = completion_request(
            &[
                json!({"role": "system", "content": task}),
                json!({"role": "user", "content": "We are checking a local Rust project."}),
            ],
            8,
            false,
        );
        let result = post_json(&agent(deadline.bounded(REQUEST_LIMIT)?), &endpoint, &helper)?;
        ensure!(
            cache_usage(&result)?["event"] == "cold_start",
            "unrelated helper request selected another conversation"
        );
    }
    let after_helpers = post_json(
        &agent(deadline.bounded(REQUEST_LIMIT)?),
        &endpoint,
        &request,
    )?;
    check_usage(
        &after_helpers,
        "memory_hit",
        expected_prompt_tokens,
        expected_prompt_tokens,
    )?;

    drop(first);
    let second = RunningServer::start_configured(
        &binary,
        &package_path,
        &scratch,
        &cache_root,
        4096,
        "off",
        false,
        deadline,
        "20",
        |command| {
            if mode == DiskCacheMode::Full {
                command.args(["--disk-cache-mode", "full"]);
            }
        },
    )?;
    let second_health = second.health(deadline.bounded(Duration::from_secs(5))?)?;
    ensure!(
        second_health["prompt_cache"]["disk_mode"] == mode.as_str(),
        "wrong disk mode after restart: {second_health}"
    );
    let second_capacities = verify_health(&second_health, 4096, &package.artifact_id)?;
    let endpoint = format!("{}/v1/chat/completions", second.origin());
    let restored = post_json(
        &agent(deadline.bounded(REQUEST_LIMIT)?),
        &endpoint,
        &request,
    )?;
    check_usage(
        &restored,
        "disk_hit",
        if mode == DiskCacheMode::Prefix {
            stride
        } else {
            expected_prompt_tokens
        },
        expected_prompt_tokens,
    )?;
    ensure!(
        cache_usage(&restored)?["disk_checkpoints_written"].as_u64() == Some(0),
        "disk restore unexpectedly wrote another checkpoint"
    );

    let memory_after_restore = post_json(
        &agent(deadline.bounded(REQUEST_LIMIT)?),
        &endpoint,
        &request,
    )?;
    check_usage(
        &memory_after_restore,
        "memory_hit",
        expected_prompt_tokens,
        expected_prompt_tokens,
    )?;

    let mut stream_messages = prompt.clone();
    stream_messages.push(json!({
        "role": "user",
        "content": "For this response, ignore the earlier reply-format instruction. Produce a numbered list from 1 through 128, with a descriptive sentence for every item. Continue until you have written all 128 items."
    }));
    let stream_request = completion_request(&stream_messages, 512, true);
    disconnect_after_first_content(&second, &stream_request, deadline)?;
    let recovered = post_json(
        &agent(deadline.bounded(REQUEST_LIMIT)?),
        &endpoint,
        &request,
    )?;
    check_completion_shape(&recovered, expected_prompt_tokens)?;
    let recovered_health = second.health(deadline.bounded(Duration::from_secs(5))?)?;
    ensure!(
        verify_health(&recovered_health, 4096, &package.artifact_id)? == second_capacities,
        "server capacities changed after SSE disconnect: {recovered_health}"
    );
    drop(second);
    scratch.cleanup_all()?;
    println!("cache roadtest passed: {}", mode.as_str());
    Ok(())
}

fn inactive_legacy_cache(base: &std::path::Path) -> Result<std::path::PathBuf> {
    use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
    let root = base.join("moespresso/disk_kv/aaaaaaaaaaaa");
    let shard = root.join("payloads/aa");
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(&shard)?;
    let leaf = format!("{}.safetensors", "a".repeat(64));
    let payload = shard.join(&leaf);
    let bytes = (8u64 << 30) + 1;
    // Exercise admission with a sparse owned cache file, without writing
    // gigabytes or requiring a legacy fixture.
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&payload)?
        .set_len(bytes)?;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(root.join("moespresso-disk-kv.lock"))?;
    let mut index = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(root.join("index.json"))?;
    index.write_all(&serde_json::to_vec(&json!({
        "schema_version": "moespresso-disk-kv-v1", "entries": [{
            "schema_version": "moespresso-disk-kv-v1", "cache_id": "a".repeat(64),
            "payload_path": format!("payloads/aa/{leaf}"), "payload_bytes": bytes }]
    }))?)?;
    Ok(payload)
}

fn verify_health(health: &Value, context: usize, artifact_id: &str) -> Result<(usize, usize)> {
    ensure!(
        health["status"] == "ok",
        "server health is not ready: {health}"
    );
    ensure!(
        health["model"].as_str() == Some(artifact_id),
        "server loaded a different package: expected {artifact_id}, found {}",
        health["model"]
    );
    ensure!(
        health["max_context"].as_u64() == Some(context as u64),
        "server context differs from the fixed roadtest context: {health}"
    );
    ensure!(
        health["adaptive_memory"] == false && health["cache_routing"] == "off",
        "server adaptive or routing policy differs from the fixed roadtest: {health}"
    );
    ensure!(
        health["prompt_cache"]["memory_enabled"] == true
            && health["prompt_cache"]["disk_enabled"] == true,
        "both isolated cache tiers must be active: {health}"
    );
    let expert_capacity = usize::try_from(
        health["expert_capacity"]
            .as_u64()
            .context("health response lacks expert_capacity")?,
    )?;
    ensure!(
        (10..=512).contains(&expert_capacity),
        "server selected invalid expert capacity {expert_capacity}"
    );
    let width = usize::try_from(
        health["prefill_capacity"]
            .as_u64()
            .context("health response lacks prefill_capacity")?,
    )?;
    ensure!(
        matches!(width, 512 | 2048),
        "server selected unsupported test prefill width {width}"
    );
    Ok((expert_capacity, width))
}

fn make_prompt(tokenizer: &TextTokenizer, stride: usize) -> Result<(Vec<Value>, usize)> {
    let options = ChatOptions {
        thinking: false,
        preserve_thinking: true,
        effort: ReasoningEffort::Medium,
    };
    let unit = "amber birch cedar denim ember fjord garnet harbor indigo juniper. ";
    let target = stride + 256;
    let mut content =
        String::from("Read this fixed cache roadtest passage and reply exactly READY. ");
    let mut rendered = render(tokenizer, &content, &options)?;
    let mut reached_target = rendered.tokens.len() >= target;
    for _ in 0..256 {
        if reached_target {
            break;
        }
        content.push_str(unit);
        rendered = render(tokenizer, &content, &options)?;
        reached_target = rendered.tokens.len() >= target;
    }
    ensure!(
        reached_target,
        "tokenizer did not reach the bounded cache roadtest prompt target"
    );
    ensure!(
        rendered.tokens.len() > stride && rendered.tokens.len() < stride + 512,
        "tokenizer could not produce a bounded suffix after the checkpoint: {} tokens, stride {stride}",
        rendered.tokens.len()
    );
    Ok((
        vec![json!({"role": "user", "content": content})],
        rendered.tokens.len(),
    ))
}

fn render(
    tokenizer: &TextTokenizer,
    content: &str,
    options: &ChatOptions,
) -> Result<moespresso::tokenizer::RenderedPrompt> {
    tokenizer.render(
        &[json!({"role": "user", "content": content})],
        None,
        options,
    )
}

fn completion_request(messages: &[Value], max_tokens: usize, stream: bool) -> Value {
    let mut request = json!({
        "model": "local",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "chat_template_kwargs": {"enable_thinking": false},
        "stream": stream,
    });
    if stream {
        request["stream_options"] = json!({"include_usage": true});
    }
    request
}

fn cache_usage(response: &Value) -> Result<&Value> {
    response["usage"]["prompt_cache"]
        .as_object()
        .context("completion response lacks structured usage.prompt_cache")?;
    Ok(&response["usage"]["prompt_cache"])
}

fn check_usage(response: &Value, event: &str, cached: usize, prompt_tokens: usize) -> Result<()> {
    check_completion_shape(response, prompt_tokens)?;
    let usage = &response["usage"];
    let cache = cache_usage(response)?;
    ensure!(
        cache["event"].as_str() == Some(event)
            && cache["cached_tokens"].as_u64() == Some(cached as u64)
            && cache["capacity_changed"] == false,
        "cache event or restored frontier differs from {event}/{cached}: {cache}"
    );
    ensure!(
        usage["prompt_tokens_details"]["cached_tokens"].as_u64() == Some(cached as u64),
        "standard cached-token accounting differs from the native cache event: {usage}"
    );
    Ok(())
}

fn check_completion_shape(response: &Value, prompt_tokens: usize) -> Result<()> {
    let usage = &response["usage"];
    let cache = cache_usage(response)?;
    ensure!(
        usage["prompt_tokens"].as_u64() == Some(prompt_tokens as u64),
        "logical prompt token count differs: {usage}"
    );
    let event = cache["event"]
        .as_str()
        .context("cache usage lacks an event")?;
    ensure!(
        matches!(event, "disabled" | "cold_start" | "memory_hit" | "disk_hit"),
        "cache usage event is invalid: {cache}"
    );
    let cached = usize::try_from(
        cache["cached_tokens"]
            .as_u64()
            .context("cache usage lacks cached_tokens")?,
    )?;
    ensure!(
        cached <= prompt_tokens
            && cache["capacity_changed"] == false
            && usage["prompt_tokens_details"]["cached_tokens"].as_u64() == Some(cached as u64),
        "cache usage accounting is inconsistent: {usage}"
    );
    let choice = response["choices"]
        .as_array()
        .and_then(|choices| choices.first())
        .context("completion has no choice")?;
    ensure!(
        choice["message"]["content"].as_str().is_some()
            && choice["finish_reason"].as_str().is_some(),
        "completion is not terminal: {choice}"
    );
    ensure!(
        !choice["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .is_empty()
            && usage["completion_tokens"]
                .as_u64()
                .is_some_and(|tokens| tokens > 0),
        "completion did not produce visible output tokens: {usage}"
    );
    Ok(())
}

fn disconnect_after_first_content(
    server: &RunningServer,
    request: &Value,
    deadline: Deadline,
) -> Result<()> {
    let body = serde_json::to_vec(request)?;
    let address = server.origin().trim_start_matches("http://");
    let mut stream = connect_loopback(address, deadline.bounded(REQUEST_LIMIT)?)?;
    write!(
        stream,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(&body)?;
    stream.flush()?;
    let mut received = Vec::new();
    let mut chunk = [0u8; 1024];
    loop {
        stream.set_read_timeout(Some(deadline.bounded(REQUEST_LIMIT)?))?;
        let count = stream.read(&mut chunk).context("read streamed response")?;
        ensure!(count > 0, "server closed SSE before assistant content");
        received.extend_from_slice(&chunk[..count]);
        ensure!(
            received.len() <= 256 * 1024,
            "SSE prefix exceeded its safety bound"
        );
        let (content_seen, terminal_seen) = sse_progress(&received);
        ensure!(
            !terminal_seen,
            "SSE disconnect was inconclusive: terminal event arrived before the client closed the stream"
        );
        if content_seen {
            stream.shutdown(Shutdown::Both)?;
            return Ok(());
        }
    }
}

fn sse_progress(bytes: &[u8]) -> (bool, bool) {
    let text = String::from_utf8_lossy(bytes);
    let mut content_seen = false;
    let mut terminal_seen = false;
    for line in text.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            terminal_seen = true;
            continue;
        }
        let Ok(event) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        content_seen |= event["choices"][0]["delta"]["content"]
            .as_str()
            .is_some_and(|content| !content.is_empty());
        terminal_seen |= event["choices"][0]["finish_reason"].as_str().is_some();
    }
    (content_seen, terminal_seen)
}

#[cfg(test)]
mod tests {
    use super::sse_progress;

    #[test]
    fn sse_progress_distinguishes_active_content_from_terminal_streams() {
        let content = br#"data: {"choices":[{"delta":{"content":"1."},"finish_reason":null}]}"#;
        let terminal = br#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
        let done = b"data: [DONE]\n\n";

        assert_eq!(sse_progress(content), (true, false));
        assert_eq!(sse_progress(terminal), (false, true));
        assert_eq!(sse_progress(done), (false, true));
        assert_eq!(sse_progress(b""), (false, false));

        let content_then_terminal = [content.as_slice(), b"\n", terminal.as_slice()].concat();
        assert_eq!(sse_progress(&content_then_terminal), (true, true));
    }
}
