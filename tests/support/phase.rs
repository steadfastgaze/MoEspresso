use std::time::Duration;

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};

use super::{Deadline, RunningServer, Scratch, agent, binary_path, package_path, post_json};

const CONTEXT: usize = 131072;
const REQUEST_LIMIT: Duration = Duration::from_secs(180);

pub fn run() -> Result<()> {
    ensure!(
        cfg!(target_os = "macos"),
        "phased serving requires macOS Metal"
    );
    let deadline = Deadline::after(Duration::from_secs(12 * 60))?;
    let binary = binary_path()?;
    let package = package_path()?;
    let mut scratch = Scratch::new("phase-default")?;
    let cache = scratch.subdir("cache")?;
    let server = RunningServer::start_configured(
        &binary,
        &package,
        &scratch,
        &cache,
        CONTEXT,
        "off",
        true,
        deadline,
        "24",
        |command| {
            command.arg("--no-disk-cache");
        },
    )?;
    let endpoint = format!("{}/v1/chat/completions", server.origin());
    let mut turns = vec![json!({"role":"user","content":
            "The river crosses the plain. The bridge joins the banks. The town maintains the bridge. ".repeat(230)})];
    let cold = post_json(
        &agent(deadline.bounded(REQUEST_LIMIT)?),
        &endpoint,
        &request(&turns),
    )?;
    let prompt_tokens = cold["usage"]["prompt_tokens"]
        .as_u64()
        .context("cold prompt count")?;
    ensure!(
        prompt_tokens > 4096,
        "the prompt must exercise a full wide chunk"
    );
    ensure!(
        cold["usage"]["prompt_cache"]["event"] == "cold_start"
            && cold["usage"]["prompt_cache"]["cached_tokens"] == 0
            && cold["usage"]["moespresso"]["prefill_tokens"] == prompt_tokens,
        "cold request did not consume the uncached prompt"
    );
    let reply = cold["choices"][0]["message"]["content"]
        .as_str()
        .context("cold assistant reply")?;
    ensure!(
        !reply.is_empty(),
        "cold request produced no assistant reply"
    );
    turns.push(json!({"role":"assistant","content":reply}));
    turns.push(json!({"role":"user","content":"Why?"}));
    let first_followup = post_json(
        &agent(deadline.bounded(REQUEST_LIMIT)?),
        &endpoint,
        &request(&turns),
    )?;
    let generated_cached = first_followup["usage"]["prompt_cache"]["cached_tokens"]
        .as_u64()
        .context("generated prefix count")?;
    ensure!(
        generated_cached > prompt_tokens,
        "first follow-up did not reuse consumed generated tokens"
    );
    let followup_prompt = first_followup["usage"]["prompt_tokens"]
        .as_u64()
        .context("follow-up prompt count")?;
    let next_reply = first_followup["choices"][0]["message"]["content"]
        .as_str()
        .context("follow-up assistant reply")?;
    ensure!(!next_reply.is_empty(), "first follow-up produced no reply");
    turns.push(json!({"role":"assistant","content":next_reply}));
    turns.push(json!({"role":"user","content":"Explain."}));
    // The bounded cache prioritizes one prompt anchor per conversation.
    // Advance the generated reply into that anchor before interleaving.
    let helper = vec![json!({"role":"user","content":"Give a short title for a Rust project."})];
    let interleaved = post_json(
        &agent(deadline.bounded(REQUEST_LIMIT)?),
        &endpoint,
        &request(&helper),
    )?;
    ensure!(
        interleaved["usage"]["prompt_cache"]["event"] == "cold_start",
        "unrelated request selected the generated conversation"
    );
    let followup = post_json(
        &agent(deadline.bounded(REQUEST_LIMIT)?),
        &endpoint,
        &request(&turns),
    )?;
    let cached = followup["usage"]["prompt_cache"]["cached_tokens"]
        .as_u64()
        .context("follow-up cached count")?;
    let new_tokens = followup["usage"]["moespresso"]["prefill_tokens"]
        .as_u64()
        .context("follow-up new count")?;
    ensure!(
        followup["usage"]["prompt_cache"]["event"] == "memory_hit"
            && cached >= followup_prompt
            && (1..=64).contains(&new_tokens),
        "follow-up must reuse the actual committed reply after interleaving: cached={cached}, prompt={prompt_tokens}, new={new_tokens}"
    );
    ensure!(
        followup["choices"][0]["message"]["content"]
            .as_str()
            .is_some_and(|s| !s.is_empty()),
        "cached follow-up produced no reply"
    );
    let health = server.health(deadline.bounded(Duration::from_secs(5))?)?;
    ensure!(
        health["adaptive_memory"] == true
            && health["phased_prefill"] == true
            && health["cache_routing"] == "prefer-resident"
            && health["max_context"] == CONTEXT
            && health["prefill_capacity"] == 2048
            && health["prefill_workspace_ready"] == true
            && health["prompt_cache"]["memory_enabled"] == true
            && health["prompt_cache"]["disk_enabled"] == false,
        "serving did not return ordinary resources and retain its cache policy: {health}"
    );
    println!(
        "phased serving: prompt={prompt_tokens} cached={cached} suffix={new_tokens} ordinary_rows=2048 cache=memory_hit"
    );
    drop(server);
    scratch.cleanup_all()?;
    Ok(())
}

fn request(turns: &[Value]) -> Value {
    json!({"model":"local", "messages":turns, "max_tokens":8, "temperature":0.0,
        "chat_template_kwargs":{"enable_thinking":false}, "stream":false})
}
