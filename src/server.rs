//! Loopback Chat Completions serving with one synchronous inference owner.

#[cfg(test)]
mod conversation_tests;
mod protocol;
#[cfg(test)]
mod quality;
mod stream;

use std::cell::{Cell, RefCell};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::mpsc::{SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow, ensure};
use rand_core::{OsRng, TryRngCore};
use serde_json::{Value, json};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use crate::engine::{
    DiskCacheMode, DiskCheckpointStore, DiskLimits, Engine, PromptPrefixCache, StoreLimits,
};
use crate::generation::{
    FinishReason, GenerationCaches, GenerationDiskCache, GenerationOptions, GenerationStats,
};
use crate::generation::{RequestPhase, generate_with_cache_controlled, generate_with_cache_owned};
use crate::package::Package;
use crate::routing::RoutingMode;
use crate::sampling::{Sampler, SamplingOptions};
use crate::tokenizer::TextTokenizer;

use protocol::{ApiError, ChatRequest, parse_chat};
use stream::{Delta, ResponseShaper};

const MAX_BODY_BYTES: usize = 1024 * 1024;
const MAX_QUEUED_REQUESTS: usize = 8;

struct UnblockOnDrop<'a>(&'a Server);

impl Drop for UnblockOnDrop<'_> {
    fn drop(&mut self) {
        self.0.unblock();
    }
}

pub struct ServeOptions {
    pub port: u16,
    pub expert_capacity: usize,
    pub max_context: usize,
    pub thinking: bool,
    pub decode_mode: RoutingMode,
    pub cache: Option<ServeCacheOptions>,
    /// Validated budget for automatic bounded expert adjustment.
    pub expert_capacity_control: Option<crate::memory::ExpertCapacityControlBudget>,
    /// Pressure response within the fixed operating budget.
    pub adaptive_memory: bool,
    /// ANE payload reserved by admission, separately from process overhead.
    /// Zero retains GPU execution; otherwise reserve the complete allowance
    /// for the physical shape selected from the initial prefill workspace.
    pub ane_reserved_bytes: u64,
}

/// Explicit host-cache allowances. These do not include the live Session,
/// restored Metal state, allocator rounding or driver-owned memory.
pub struct ServeCacheOptions {
    pub memory_retained_bytes: u64,
    /// Covers retained memory entries plus bounded disk I/O staging.
    pub host_overlap_bytes: u64,
    pub disk: Option<ServeDiskOptions>,
}

pub struct ServeDiskOptions {
    pub mode: DiskCacheMode,
    pub root: PathBuf,
    /// Default managed namespace; explicit roots retain their own allowance.
    pub managed_cache_base: Option<PathBuf>,
    pub max_file_bytes: u64,
    pub max_io_buffer_bytes: u64,
    pub max_payload_bytes: u64,
    pub max_entries: usize,
    pub stride: usize,
    pub write_depth: usize,
    /// Explicitly requested disk storage refuses startup if its root is unavailable.
    pub required: bool,
}

impl ServeDiskOptions {
    fn open_store(
        &self,
        engine: &Engine,
        tokenizer: &TextTokenizer,
    ) -> Result<DiskCheckpointStore> {
        match &self.managed_cache_base {
            Some(base) => DiskCheckpointStore::open_managed(
                &self.root,
                engine,
                tokenizer,
                self.limits(),
                base,
            ),
            None => DiskCheckpointStore::open(&self.root, engine, tokenizer, self.limits()),
        }
    }

    fn limits(&self) -> StoreLimits {
        StoreLimits {
            mode: self.mode,
            max_entries: self.max_entries,
            max_payload_bytes: self.max_payload_bytes,
            checkpoint: DiskLimits {
                max_file_bytes: self.max_file_bytes,
                max_io_buffer_bytes: self.max_io_buffer_bytes,
            },
        }
    }
}

struct State {
    package: Package,
    tokenizer: TextTokenizer,
    engine: Engine,
    memory_cache: Option<PromptPrefixCache>,
    disk_cache: Option<GenerationDiskCache>,
    sampling_defaults: SamplingOptions,
    options: ServeOptions,
    next_id: u64,
    health: Arc<RwLock<Value>>,
    adaptive: Option<crate::adaptive_memory::Controller>,
    prefill_phase: Option<crate::generation::request_resources::PhasedPrefill>,
}

fn json_response(request: Request, status: u16, value: &Value) -> Result<()> {
    let data = serde_json::to_vec(value)?;
    let header = Header::from_bytes("Content-Type", "application/json; charset=utf-8")
        .map_err(|_| anyhow!("invalid JSON content type"))?;
    request.respond(
        Response::from_data(data)
            .with_status_code(StatusCode(status))
            .with_header(header),
    )?;
    Ok(())
}

fn api_error(request: Request, error: ApiError) -> Result<()> {
    json_response(request, error.status, &error.body())
}

fn server_error(request: Request, message: &str) -> Result<()> {
    eprintln!("server error: {message}");
    json_response(
        request,
        500,
        &json!({"error": {"message": message, "type": "server_error"}}),
    )
}

fn read_body(request: &mut Request) -> std::result::Result<Vec<u8>, ApiError> {
    if request.body_length().is_some_and(|n| n > MAX_BODY_BYTES) {
        return Err(ApiError {
            status: 413,
            message: "request body exceeds 1 MiB".into(),
            code: None,
        });
    }
    let mut body = Vec::new();
    request
        .as_reader()
        .take((MAX_BODY_BYTES + 1) as u64)
        .read_to_end(&mut body)
        .map_err(|_| ApiError::bad("could not read request body"))?;
    if body.len() > MAX_BODY_BYTES {
        return Err(ApiError {
            status: 413,
            message: "request body exceeds 1 MiB".into(),
            code: None,
        });
    }
    Ok(body)
}

fn read_json(request: &mut Request) -> std::result::Result<Value, ApiError> {
    serde_json::from_slice(&read_body(request)?)
        .map_err(|_| ApiError::bad("request body must be valid JSON"))
}

fn created_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn finish_name(finish: FinishReason, has_calls: bool) -> &'static str {
    match finish {
        FinishReason::StopToken(_) if has_calls => "tool_calls",
        FinishReason::StopToken(_) => "stop",
        FinishReason::MaxTokens | FinishReason::ContextLimit => "length",
        FinishReason::Cancelled => "length",
    }
}

fn usage(stats: GenerationStats) -> Value {
    let completion =
        stats.generated_tokens + usize::from(matches!(stats.finish, FinishReason::StopToken(_)));
    json!({
        "prompt_tokens": stats.prompt_consumed,
        "completion_tokens": completion,
        "total_tokens": stats.prompt_consumed + completion,
        "prompt_tokens_details": { "cached_tokens": stats.prompt_cache.cached_tokens },
        "prompt_cache": {
            "event": stats.prompt_cache.event,
            "cached_tokens": stats.prompt_cache.cached_tokens,
            "disk_checkpoints_written": stats.prompt_cache.disk_checkpoints_written,
            "last_disk_checkpoint": stats.prompt_cache.last_disk_checkpoint,
            "disk_checkpoint_write_seconds": stats.prompt_cache.disk_checkpoint_write_seconds,
            "capacity_changed": stats.prompt_cache.capacity_changed,
        },
        "moespresso": {
            "prefill_tokens": stats.prefill_tokens,
            "prefill_seconds": stats.prefill_seconds,
            "first_token_seconds": stats.first_token_seconds,
            "generation_seconds": stats.generation_seconds,
            "ready_to_first_token_seconds": stats.ready_to_first_token_seconds,
        },
    })
}

fn completion(
    id: &str,
    created: u64,
    model: &str,
    stats: GenerationStats,
    shaper: &ResponseShaper,
) -> Value {
    let mut message = json!({"role": "assistant", "content": shaper.content});
    if !shaper.calls.is_empty() {
        message["tool_calls"] = Value::Array(shaper.calls.clone());
        if shaper.content.is_empty() {
            message["content"] = Value::Null;
        }
    }
    if !shaper.reasoning.is_empty() {
        message["reasoning_content"] = Value::String(shaper.reasoning.clone());
    }
    json!({
        "id": id, "object": "chat.completion", "created": created, "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_name(stats.finish, !shaper.calls.is_empty())}],
        "usage": usage(stats),
    })
}

struct SseWriter {
    writer: Box<dyn Write + Send>,
    id: String,
    model: String,
    created: u64,
    include_usage: bool,
}

impl SseWriter {
    fn new(request: Request, id: &str, model: &str, created: u64, include_usage: bool) -> Self {
        Self {
            writer: request.into_writer(),
            id: id.into(),
            model: model.into(),
            created,
            include_usage,
        }
    }

    fn write_raw(&mut self, data: &[u8]) -> std::io::Result<()> {
        self.writer.write_all(data)?;
        self.writer.flush()
    }

    fn write(&mut self, data: &[u8]) -> std::io::Result<()> {
        write!(self.writer, "{:x}\r\n", data.len())?;
        self.writer.write_all(data)?;
        self.writer.write_all(b"\r\n")?;
        self.writer.flush()
    }

    fn close(&mut self) -> std::io::Result<()> {
        self.write_raw(b"0\r\n\r\n")
    }

    fn start(&mut self) -> Result<()> {
        self.write_raw(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n")?;
        self.chunk(
            vec![json!({"index": 0, "delta": {"role": "assistant"}, "finish_reason": null})],
            None,
        )
    }

    fn event(&mut self, value: &Value) -> Result<()> {
        let mut data = b"data: ".to_vec();
        serde_json::to_writer(&mut data, value)?;
        data.extend_from_slice(b"\n\n");
        self.write(&data)?;
        Ok(())
    }

    fn chunk(&mut self, choices: Vec<Value>, usage: Option<Value>) -> Result<()> {
        let mut value = json!({
            "id": self.id, "object": "chat.completion.chunk", "created": self.created,
            "model": self.model, "choices": choices,
        });
        if self.include_usage {
            value["usage"] = usage.unwrap_or(Value::Null);
        }
        self.event(&value)
    }

    fn delta(&mut self, delta: Delta) -> Result<()> {
        let fields = match delta {
            Delta::Reasoning(text) => json!({"reasoning_content": text}),
            Delta::Content(text) => json!({"content": text}),
            Delta::Tool { index, entry } => {
                let mut entry = entry;
                entry["index"] = json!(index);
                json!({"tool_calls": [entry]})
            }
        };
        self.chunk(
            vec![json!({"index": 0, "delta": fields, "finish_reason": null})],
            None,
        )
    }

    fn finish(&mut self, finish: &str, usage: Value) -> Result<()> {
        self.chunk(
            vec![json!({"index": 0, "delta": {}, "finish_reason": finish})],
            None,
        )?;
        if self.include_usage {
            self.chunk(vec![], Some(usage))?;
        }
        self.write(b"data: [DONE]\n\n")?;
        self.close()?;
        Ok(())
    }

    fn error(&mut self, message: &str) {
        eprintln!("stream error: {message}");
        let _ = self.event(
            &json!({"type": "error", "error": {"message": message, "type": "server_error"}}),
        );
        let _ = self.close();
    }
}

impl State {
    fn publish_health(&self) {
        let snapshot = json!({
            "status": "ok",
            "model": self.package.artifact_id,
            "expert_capacity": self.engine.expert_capacity(),
            "max_context": self.options.max_context,
            "prefill_capacity": self.engine.prefill_chunk_size(),
            "prefill_workspace_ready": self.engine.prefill_workspace_ready(),
            "prefill_backend": if self.engine.ane_active() { "gpu+ane" } else { "gpu" },
            "phased_prefill": self.prefill_phase.is_some(),
            "adaptive_memory": self.adaptive.is_some(),
            "cache_routing": match self.options.decode_mode {
                RoutingMode::Unbiased => "off",
                RoutingMode::CachePrior => "prefer-resident",
            },
            "prompt_cache": {
                "memory_enabled": self.memory_cache.is_some(),
                "disk_enabled": self.disk_cache.is_some(),
                "disk_mode": self.disk_cache.as_ref().map(|cache| cache.mode().as_str()),
                "retained_bytes": self.memory_cache.as_ref().and_then(|cache| cache.retained_bytes().ok()),
            },
        });
        *self
            .health
            .write()
            .unwrap_or_else(|error| error.into_inner()) = snapshot;
    }

    fn handle_request(&mut self, request: Request) -> Result<()> {
        // Pressure release cannot wait for request parsing.
        if let Err(error) = self.adapt_at_request_boundary(false) {
            return api_error(
                request,
                ApiError {
                    status: 503,
                    message: format!("memory pressure prevents admission: {error:#}"),
                    code: None,
                },
            );
        }
        self.handle_chat(request)
    }

    fn adapt_at_request_boundary(&mut self, starting_request: bool) -> Result<()> {
        let result = self.adaptive.as_mut().map_or(Ok(false), |controller| {
            let boundary = if starting_request {
                crate::adaptive_memory::Boundary::RequestStart
            } else {
                crate::adaptive_memory::Boundary::Observe
            };
            controller.check_boundary(&mut self.engine, boundary)
        });
        self.publish_health();
        result.map(|_| ())
    }

    #[cfg(test)]
    fn initialize_test_caches(&mut self) {
        // Initialize cache ownership independently of expert residency.
        self.memory_cache = None;
        self.disk_cache = None;
        if let Some(cache) = &self.options.cache {
            self.memory_cache = (cache.memory_retained_bytes > 0)
                .then(|| PromptPrefixCache::new(&self.engine, cache.memory_retained_bytes));
            if let Some(disk) = &cache.disk {
                match disk
                    .open_store(&self.engine, &self.tokenizer)
                    .and_then(|store| {
                        GenerationDiskCache::new(store, disk.stride, disk.write_depth)
                    }) {
                    Ok(cache) => self.disk_cache = Some(cache),
                    Err(error) => {
                        eprintln!("disk cache unavailable: {error:#}");
                    }
                }
            }
        }
    }

    fn next_response_id(&mut self) -> String {
        self.next_id = self.next_id.wrapping_add(1);
        format!("chatcmpl-moespresso-{}", self.next_id)
    }

    fn handle_chat(&mut self, mut request: Request) -> Result<()> {
        let body = match read_json(&mut request) {
            Ok(value) => value,
            Err(error) => return api_error(request, error),
        };
        let input = match parse_chat(
            &body,
            &self.package.artifact_id,
            self.sampling_defaults,
            self.options.thinking,
        ) {
            Ok(value) => value,
            Err(error) => return api_error(request, error),
        };
        let rendered =
            match self
                .tokenizer
                .render(&input.messages, input.tools.as_deref(), &input.chat)
            {
                Ok(value) => value,
                Err(error) => {
                    return api_error(
                        request,
                        ApiError::bad(format!("render chat template: {error:#}")),
                    );
                }
            };
        if rendered.tokens.is_empty() {
            return api_error(request, ApiError::bad("rendered prompt is empty"));
        }
        if rendered
            .tokens
            .len()
            .checked_add(input.max_tokens)
            .is_none_or(|n| n > self.options.max_context)
        {
            return api_error(
                request,
                ApiError {
                    status: 400,
                    message: format!(
                        "prompt tokens ({}) plus max_tokens ({}) exceed context limit ({})",
                        rendered.tokens.len(),
                        input.max_tokens,
                        self.options.max_context
                    ),
                    code: Some("context_length_exceeded"),
                },
            );
        }
        if let Err(error) = self.adapt_at_request_boundary(true) {
            return api_error(
                request,
                ApiError {
                    status: 503,
                    message: format!("memory pressure prevents admission: {error:#}"),
                    code: None,
                },
            );
        }
        let seed = match input.seed {
            Some(seed) => seed,
            None if input.sampling.temperature == 0.0 => 0,
            None => match OsRng.try_next_u64() {
                Ok(seed) => seed,
                Err(error) => {
                    return server_error(
                        request,
                        &format!("operating-system seed failed: {error}"),
                    );
                }
            },
        };
        if let Err(error) = Sampler::new(input.sampling, seed) {
            return api_error(request, ApiError::bad(error.to_string()));
        }
        let id = self.next_response_id();
        eprintln!(
            "request id={id} prompt_tokens={} max_tokens={} stream={} capacity={}",
            rendered.tokens.len(),
            input.max_tokens,
            input.stream,
            self.engine.expert_capacity()
        );
        let created = created_now();
        if input.stream {
            self.stream_chat(request, input, &rendered.tokens, seed, &id, created)
        } else {
            self.json_chat(request, input, &rendered.tokens, seed, &id, created)
        }
    }

    fn generation_options(&self, input: &ChatRequest, seed: u64) -> GenerationOptions {
        GenerationOptions {
            max_tokens: input.max_tokens,
            max_context: self.options.max_context,
            decode_mode: self.options.decode_mode,
            sampling: input.sampling,
            seed,
        }
    }

    fn generate_request(
        &mut self,
        prompt: &[u32],
        options: GenerationOptions,
        on_text: impl FnMut(&str) -> Result<()>,
        cancelled: impl FnMut() -> bool,
    ) -> Result<GenerationStats> {
        if let Some(controller) = &mut self.adaptive {
            controller.begin_request();
        }
        let health = Arc::clone(&self.health);
        let phased = self.prefill_phase.is_some();
        let mut owner = |engine: &mut Engine, event| {
            if let Some(controller) = &mut self.adaptive {
                let previous = engine.expert_capacity();
                if let Some(phase) = &mut self.prefill_phase {
                    phase.transition(engine, event, controller)?;
                }
                let changed = if matches!(event, RequestPhase::CompletedOperation) {
                    controller.check_boundary(
                        engine,
                        crate::adaptive_memory::Boundary::CompletedOperation,
                    )?
                } else {
                    previous != engine.expert_capacity()
                };
                if changed {
                    health.write().unwrap_or_else(|error| error.into_inner())["expert_capacity"] =
                        json!(engine.expert_capacity());
                }
                Ok(changed)
            } else {
                Ok(false)
            }
        };
        let caches = GenerationCaches {
            memory: self.memory_cache.as_mut(),
            disk: self.disk_cache.as_mut(),
        };
        let result = if phased {
            generate_with_cache_owned(
                &mut self.engine,
                &self.tokenizer,
                prompt,
                options,
                caches,
                on_text,
                cancelled,
                &mut owner,
            )
        } else {
            generate_with_cache_controlled(
                &mut self.engine,
                &self.tokenizer,
                prompt,
                options,
                caches,
                on_text,
                cancelled,
                |engine| owner(engine, RequestPhase::CompletedOperation),
            )
        };
        if let Some(controller) = &mut self.adaptive {
            controller.end_request(
                result
                    .as_ref()
                    .is_ok_and(|stats| stats.finish != FinishReason::Cancelled),
            );
        }
        self.publish_health();
        if let Ok(stats) = &result {
            let seconds = |value: Option<f64>| {
                value.map_or_else(|| "unavailable".to_owned(), |value| format!("{value:.3}"))
            };
            eprintln!(
                "request complete id=chatcmpl-moespresso-{} cache={} cached_tokens={} prefill_tokens={} prefill_seconds={} ready_to_first_seconds={} generated_tokens={} generation_seconds={} disk_writes={} disk_write_seconds={:.3} capacity={} finish={:?}",
                self.next_id,
                stats.prompt_cache.event,
                stats.prompt_cache.cached_tokens,
                stats.prefill_tokens,
                seconds(stats.prefill_seconds),
                seconds(stats.ready_to_first_token_seconds),
                stats.generated_tokens,
                seconds(stats.generation_seconds),
                stats.prompt_cache.disk_checkpoints_written,
                stats.prompt_cache.disk_checkpoint_write_seconds,
                self.engine.expert_capacity(),
                stats.finish
            );
        }
        result
    }

    fn json_chat(
        &mut self,
        request: Request,
        input: ChatRequest,
        prompt: &[u32],
        seed: u64,
        id: &str,
        created: u64,
    ) -> Result<()> {
        let mut shaper =
            ResponseShaper::new(input.chat.thinking, input.tools.as_deref(), id.into());
        let options = self.generation_options(&input, seed);
        let result = self.generate_request(
            prompt,
            options,
            |text| {
                shaper.push(text);
                Ok(())
            },
            || false,
        );
        let stats = match result {
            Ok(stats) => stats,
            Err(error) => return server_error(request, &error.to_string()),
        };
        shaper.finish(matches!(
            stats.finish,
            FinishReason::MaxTokens | FinishReason::ContextLimit
        ));
        let response = completion(id, created, &input.model, stats, &shaper);
        json_response(request, 200, &response)
    }

    fn stream_chat(
        &mut self,
        request: Request,
        input: ChatRequest,
        prompt: &[u32],
        seed: u64,
        id: &str,
        created: u64,
    ) -> Result<()> {
        let writer = RefCell::new(SseWriter::new(
            request,
            id,
            &input.model,
            created,
            input.include_usage,
        ));
        if writer.borrow_mut().start().is_err() {
            return Ok(());
        }
        let cancelled = Cell::new(false);
        let last_heartbeat = Cell::new(Instant::now());
        let mut shaper =
            ResponseShaper::new(input.chat.thinking, input.tools.as_deref(), id.into());
        let options = self.generation_options(&input, seed);
        let result = self.generate_request(
            prompt,
            options,
            |text| {
                for delta in shaper.push(text) {
                    if let Err(error) = writer.borrow_mut().delta(delta) {
                        cancelled.set(true);
                        return Err(error);
                    }
                }
                Ok(())
            },
            || {
                if cancelled.get() {
                    return true;
                }
                if last_heartbeat.get().elapsed() >= Duration::from_secs(1) {
                    last_heartbeat.set(Instant::now());
                    if writer.borrow_mut().write(b": prefill\n\n").is_err() {
                        cancelled.set(true);
                    }
                }
                cancelled.get()
            },
        );
        let stats = match result {
            Ok(stats) if stats.finish != FinishReason::Cancelled => stats,
            Ok(_) => {
                eprintln!("stream request cancelled id={id}");
                return Ok(());
            }
            Err(_) if cancelled.get() => {
                eprintln!("stream request cancelled id={id}");
                return Ok(());
            }
            Err(error) => {
                writer.borrow_mut().error(&error.to_string());
                return Ok(());
            }
        };
        for delta in shaper.finish(matches!(
            stats.finish,
            FinishReason::MaxTokens | FinishReason::ContextLimit
        )) {
            if writer.borrow_mut().delta(delta).is_err() {
                return Ok(());
            }
        }
        let finish = finish_name(stats.finish, !shaper.calls.is_empty());
        let _ = writer.borrow_mut().finish(finish, usage(stats));
        Ok(())
    }
}

// Transport remains independent of the inference owner's current request.
// Chat completions enter one bounded owner queue.
#[cfg(test)]
fn dispatch_requests(server: &Server, model: &str, sender: SyncSender<Request>) {
    let health = RwLock::new(json!({"status": "ok", "model": model}));
    dispatch_requests_with_health(server, &health, sender);
}

fn dispatch_requests_with_health(
    server: &Server,
    health: &RwLock<Value>,
    sender: SyncSender<Request>,
) {
    for request in server.incoming_requests() {
        let result = match (request.method(), request.url()) {
            (&Method::Get, "/health") => {
                let snapshot = health
                    .read()
                    .unwrap_or_else(|error| error.into_inner())
                    .clone();
                json_response(request, 200, &snapshot)
            }
            (&Method::Post, "/v1/chat/completions") => match sender.try_send(request) {
                Ok(()) => Ok(()),
                Err(TrySendError::Full(request)) => json_response(
                    request,
                    503,
                    &json!({"error": {"message": "inference queue is full", "type": "server_error"}}),
                ),
                Err(TrySendError::Disconnected(request)) => {
                    let _ = server_error(request, "inference worker is unavailable");
                    return;
                }
            },
            (_, "/health" | "/v1/chat/completions") => json_response(
                request,
                405,
                &json!({"error": {"message": "method not allowed", "type": "invalid_request_error"}}),
            ),
            _ => json_response(
                request,
                404,
                &json!({"error": {"message": "route not found", "type": "invalid_request_error"}}),
            ),
        };
        if let Err(error) = result {
            eprintln!("HTTP response failed: {error:#}");
        }
    }
}

fn validate_cache_options(cache: &ServeCacheOptions) -> Result<()> {
    ensure!(
        cache.memory_retained_bytes > 0 || cache.disk.is_some(),
        "cache requires a retained memory budget or disk storage"
    );
    let disk_io = cache
        .disk
        .as_ref()
        .map_or(0, |disk| disk.max_io_buffer_bytes);
    if let Some(disk) = &cache.disk {
        disk.limits().validate()?;
        GenerationDiskCache::validate_policy(disk.stride, disk.write_depth)?;
        ensure!(disk.root.is_absolute(), "disk cache root must be absolute");
    }
    let combined = cache
        .memory_retained_bytes
        .checked_add(disk_io)
        .ok_or_else(|| anyhow!("cache host overlap budget overflow"))?;
    ensure!(
        cache.host_overlap_bytes >= combined,
        "cache host overlap budget must cover retained entries plus disk I/O staging"
    );
    Ok(())
}

fn phased_prefill_enabled(adaptive: bool, admitted: bool, rows: usize, context: usize) -> bool {
    adaptive && admitted && rows == 2048 && context >= 4096
}

fn prefill_phase_after_quote(
    quote: Result<crate::generation::request_resources::PhasedPrefill>,
) -> Option<crate::generation::request_resources::PhasedPrefill> {
    match quote {
        Ok(phase) => Some(phase),
        Err(error) => {
            eprintln!(
                "prefill phase: enabled=false reason=wide-plan-unavailable; retaining ordinary execution: {error:#}"
            );
            None
        }
    }
}

fn validate_ane_reservation(
    reserved: u64,
    process_allowance: Option<u64>,
    workspace_rows: usize,
) -> Result<()> {
    let physical = crate::prefill_capacity::ane_physical_rows(workspace_rows)?;
    let expected = crate::memory::ane_prefill_allowance_bytes(physical)?;
    ensure!(
        reserved == 0 || reserved == expected,
        "ANE admission must reserve its complete retained payload"
    );
    if let Some(allowance) = process_allowance {
        ensure!(
            reserved == 0
                || allowance >= crate::memory::PROCESS_OVERHEAD_ALLOWANCE_BYTES + reserved,
            "capacity control budget omits the ANE reservation"
        );
    }
    Ok(())
}

/// Serve a single package on loopback. Inference is serialized on the Metal
/// owner thread; health and transport errors do not wait for generation.
pub fn serve(package_path: &str, options: ServeOptions) -> Result<()> {
    serve_with_prefill_capacity(package_path, options, crate::prefill_capacity::DEFAULT_ROWS)
}

pub fn serve_with_prefill_capacity(
    package_path: &str,
    options: ServeOptions,
    prefill_capacity: usize,
) -> Result<()> {
    crate::prefill_capacity::validate_capacity(prefill_capacity)?;
    ensure!(
        (10..=512).contains(&options.expert_capacity),
        "expert capacity must be 10..=512"
    );
    ensure!(options.max_context > 0, "max context must be positive");
    validate_ane_reservation(
        options.ane_reserved_bytes,
        options
            .expert_capacity_control
            .map(|budget| budget.process_overhead_allowance_bytes()),
        prefill_capacity,
    )?;
    if let Some(budget) = options.expert_capacity_control {
        ensure!(
            budget.initial_capacity() == options.expert_capacity,
            "capacity control budget differs from initial engine capacity"
        );
        budget.preparation_budget_bytes(options.expert_capacity, options.expert_capacity)?;
    }
    if let Some(cache) = &options.cache {
        validate_cache_options(cache)?;
    }
    let package = Package::open(package_path)?;
    let adaptive = if options.adaptive_memory {
        Some(crate::adaptive_memory::Controller::new(
            options
                .expert_capacity_control
                .ok_or_else(|| anyhow!("adaptive memory requires budgeted bounded admission"))?,
        )?)
    } else {
        None
    };
    if let Some(budget) = options.expert_capacity_control {
        let footprint = crate::memory::PackageFootprint::from_package(&package)?;
        let retained = options
            .cache
            .as_ref()
            .map_or(0, |cache| cache.memory_retained_bytes);
        let disk_io = options
            .cache
            .as_ref()
            .and_then(|cache| cache.disk.as_ref())
            .map_or(0, |disk| disk.max_io_buffer_bytes);
        budget.validate_serving(
            &footprint,
            options.max_context,
            prefill_capacity,
            retained,
            disk_io,
        )?;
    }
    let prefill_phase = if phased_prefill_enabled(
        options.adaptive_memory,
        options.expert_capacity_control.is_some(),
        prefill_capacity,
        options.max_context,
    ) {
        let retained = options
            .cache
            .as_ref()
            .map_or(0, |cache| cache.memory_retained_bytes);
        let disk_io = options
            .cache
            .as_ref()
            .and_then(|cache| cache.disk.as_ref())
            .map_or(0, |disk| disk.max_io_buffer_bytes);
        let quote = crate::generation::request_resources::PhasedPrefill::new(
            options
                .expert_capacity_control
                .ok_or_else(|| anyhow!("phased prefill requires admitted capacity control"))?,
            &crate::memory::PackageFootprint::from_package(&package)?,
            options.max_context,
            retained,
            disk_io,
        );
        let phase = prefill_phase_after_quote(quote);
        if phase.is_some() {
            eprintln!(
                "prefill phase: enabled=true ordinary_rows=2048 wide_rows=4096 minimum_new_tokens=2049"
            );
        }
        phase
    } else {
        None
    };
    ensure!(
        options.ane_reserved_bytes == 0 || prefill_capacity >= 4096 || prefill_phase.is_some(),
        "ANE admission has no eligible ordinary or phased prefill workspace"
    );
    let sampling_defaults = SamplingOptions::from_package(&package)?;
    let tokenizer = TextTokenizer::load(&package)?;
    let mut engine = Engine::load_with_prefill_capacity(
        &package,
        options.expert_capacity,
        options.max_context,
        prefill_capacity,
    )?;
    if options.ane_reserved_bytes > 0 {
        engine.prefer_ane(&package)?;
    }
    eprintln!("prefill workspace capacity={prefill_capacity}");
    let initialized: Result<_> = (|| {
        if let Some(disk) = options.cache.as_ref().and_then(|cache| cache.disk.as_ref()) {
            let first =
                crate::prefill_capacity::first_checkpoint_frontier(disk.stride, prefill_capacity)
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "none".to_owned());
            eprintln!(
                "disk checkpoint cadence: mode={} configured_stride={} first_cold_anchor={first} prefill_capacity={prefill_capacity} anchor_write_depth={} final_frontiers={}",
                disk.mode.as_str(),
                disk.stride,
                disk.write_depth,
                if disk.mode == DiskCacheMode::Full {
                    "enabled"
                } else {
                    "disabled"
                }
            );
        }
        let (memory_cache, disk_cache) = if let Some(cache) = &options.cache {
            let disk_io = cache
                .disk
                .as_ref()
                .map_or(0, |disk| disk.max_io_buffer_bytes);
            let memory = (cache.memory_retained_bytes > 0)
                .then(|| PromptPrefixCache::new(&engine, cache.memory_retained_bytes));
            let disk = if let Some(disk) = &cache.disk {
                match disk.open_store(&engine, &tokenizer) {
                    Ok(store) => Some(GenerationDiskCache::new(
                        store,
                        disk.stride,
                        disk.write_depth,
                    )?),
                    Err(error) if disk.required => return Err(error),
                    Err(error) => {
                        eprintln!("prompt cache disk unavailable; serving without it: {error:#}");
                        None
                    }
                }
            } else {
                None
            };
            if memory.is_some() || disk.is_some() {
                eprintln!(
                    "prompt cache enabled: retained={} host_overlap={} disk_io_allowance={} disk_active={}",
                    cache.memory_retained_bytes,
                    cache.host_overlap_bytes,
                    disk_io,
                    disk.is_some()
                );
            }
            (memory, disk)
        } else {
            (None, None)
        };
        let address = format!("127.0.0.1:{}", options.port);
        let server = Server::http(&address).map_err(|error| anyhow!("bind {address}: {error}"))?;
        Ok((memory_cache, disk_cache, server))
    })();
    let (memory_cache, disk_cache, server) = match initialized {
        Ok(initialized) => initialized,
        Err(error) => {
            return match engine.close_ane() {
                Ok(()) => Err(error),
                Err(cleanup) => Err(cleanup.context(format!("server startup failed: {error:#}"))),
            };
        }
    };
    let address = format!("127.0.0.1:{}", options.port);
    eprintln!(
        "☕ serving {} on http://{address}/v1/chat/completions (serial, expert_capacity={}, max_context={})",
        package.artifact_id, options.expert_capacity, options.max_context
    );
    let mut state = State {
        package,
        tokenizer,
        engine,
        memory_cache,
        disk_cache,
        sampling_defaults,
        options,
        next_id: 0,
        health: Arc::new(RwLock::new(Value::Null)),
        adaptive,
        prefill_phase,
    };
    state.publish_health();
    let health = Arc::clone(&state.health);
    let (sender, receiver) = sync_channel(MAX_QUEUED_REQUESTS);
    let served = std::thread::scope(|scope| -> Result<()> {
        let dispatcher = scope.spawn(|| dispatch_requests_with_health(&server, &health, sender));
        // Release a dispatcher waiting in recv if the inference owner unwinds.
        let _unblock = UnblockOnDrop(&server);
        loop {
            let request = if state.adaptive.is_some() {
                match receiver.recv_timeout(Duration::from_secs(1)) {
                    Ok(request) => request,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        // Idle observations permit shrink and establish quiet
                        // history, but do not allocate unused recovery slots.
                        let _ = state.adapt_at_request_boundary(false);
                        continue;
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                }
            } else {
                match receiver.recv() {
                    Ok(request) => request,
                    Err(_) => break,
                }
            };
            if let Err(error) = state.handle_request(request) {
                eprintln!("request handling failed: {error:#}");
            }
        }
        dispatcher
            .join()
            .map_err(|_| anyhow!("HTTP dispatcher failed"))?;
        Ok(())
    });
    drop(state.disk_cache.take());
    drop(state.memory_cache.take());
    let cleanup = state.engine.close_ane();
    match (served, cleanup) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(error), Err(cleanup)) => {
            Err(error.context(format!("ANE cleanup also failed: {cleanup:#}")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{Shutdown, TcpStream};
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::mpsc::{self, TryRecvError};
    use std::thread;

    #[test]
    fn ane_reservation_is_additional_to_the_complete_process_allowance() {
        let base = crate::memory::PROCESS_OVERHEAD_ALLOWANCE_BYTES;
        for workspace in [2048, 4096, 8192] {
            let physical = crate::prefill_capacity::ane_physical_rows(workspace).unwrap();
            let ane = crate::memory::ane_prefill_allowance_bytes(physical).unwrap();
            assert!(base > ane);
            assert!(validate_ane_reservation(0, None, workspace).is_ok());
            assert!(validate_ane_reservation(0, Some(base), workspace).is_ok());
            assert!(validate_ane_reservation(ane, None, workspace).is_ok());
            assert!(validate_ane_reservation(ane, Some(base), workspace).is_err());
            assert!(validate_ane_reservation(ane, Some(base + ane - 1), workspace).is_err());
            assert!(validate_ane_reservation(ane, Some(base + ane), workspace).is_ok());
            assert!(validate_ane_reservation(ane, Some(base + ane + 1), workspace).is_ok());
            for invalid in [1, ane - 1, ane + 1, u64::MAX] {
                assert!(validate_ane_reservation(invalid, None, workspace).is_err());
                assert!(validate_ane_reservation(invalid, Some(u64::MAX), workspace).is_err());
            }
        }
        let small = crate::memory::ane_prefill_allowance_bytes(4096).unwrap();
        let large = crate::memory::ane_prefill_allowance_bytes(8192).unwrap();
        assert!(validate_ane_reservation(small, Some(base + small), 8192).is_err());
        assert!(validate_ane_reservation(large, Some(base + large), 2048).is_err());
    }

    #[test]
    fn phased_prefill_defaults_only_for_eligible_admitted_serving() {
        assert!(phased_prefill_enabled(true, true, 2048, 131072));
        assert!(phased_prefill_enabled(true, true, 2048, 4096));
        for (adaptive, admitted, rows, context) in [
            (false, true, 2048, 131072),
            (true, false, 2048, 131072),
            (true, true, 512, 131072),
            (true, true, 4096, 131072),
            (true, true, 2048, 2048),
        ] {
            assert!(!phased_prefill_enabled(adaptive, admitted, rows, context));
        }
    }

    #[test]
    fn unavailable_wide_quote_keeps_ordinary_execution() {
        let error = || anyhow!("phase workspace cannot retain the adaptive expert floor");
        assert!(prefill_phase_after_quote(Err(error())).is_none());
    }

    fn send_request(address: &str, method: &str, path: &str, index: Option<usize>) -> TcpStream {
        let mut client = TcpStream::connect(address).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let marker = index.map_or(String::new(), |id| format!("X-Queue-Index: {id}\r\n"));
        write!(
            client,
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Length: 0\r\n{marker}\r\n"
        )
        .unwrap();
        client
    }

    fn read_json_response(mut client: TcpStream) -> (u16, Value) {
        let mut response = Vec::new();
        client.read_to_end(&mut response).unwrap();
        let response = String::from_utf8(response).unwrap();
        let (head, body) = response.split_once("\r\n\r\n").unwrap();
        let status = head
            .lines()
            .next()
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        (status, serde_json::from_str(body).unwrap())
    }

    fn read_until(client: &mut TcpStream, needle: &str) -> String {
        let mut received = String::new();
        let mut chunk = [0u8; 2048];
        while !received.contains(needle) {
            let n = client.read(&mut chunk).unwrap();
            assert!(n > 0, "connection ended before {needle}");
            received.push_str(&String::from_utf8_lossy(&chunk[..n]));
        }
        received
    }

    fn read_chunk(reader: &mut impl BufRead) -> Option<String> {
        let mut size = String::new();
        reader.read_line(&mut size).unwrap();
        let length = usize::from_str_radix(size.trim(), 16).unwrap();
        if length == 0 {
            let mut trailer = String::new();
            reader.read_line(&mut trailer).unwrap();
            assert_eq!(trailer, "\r\n");
            return None;
        }
        let mut bytes = vec![0u8; length];
        reader.read_exact(&mut bytes).unwrap();
        let mut end = [0u8; 2];
        reader.read_exact(&mut end).unwrap();
        assert_eq!(&end, b"\r\n");
        Some(String::from_utf8(bytes).unwrap())
    }

    #[test]
    fn stop_token_counts_for_usage_and_tool_finish() {
        let stats = GenerationStats {
            prompt_consumed: 4,
            prefill_tokens: 4,
            prefill_seconds: Some(0.2),
            generated_tokens: 2,
            frontier: 5,
            changed_routes: 0,
            finish: FinishReason::StopToken(7),
            first_token_seconds: Some(0.25),
            generation_seconds: Some(0.75),
            ready_to_first_token_seconds: Some(0.5),
            prompt_cache: crate::generation::PromptCacheStats::default(),
        };
        assert_eq!(usage(stats)["completion_tokens"], 3);
        assert_eq!(usage(stats)["moespresso"]["first_token_seconds"], 0.25);
        assert_eq!(usage(stats)["moespresso"]["generation_seconds"], 0.75);
        assert_eq!(
            usage(stats)["moespresso"]["ready_to_first_token_seconds"],
            0.5
        );
        assert_eq!(finish_name(stats.finish, true), "tool_calls");
        assert_eq!(finish_name(FinishReason::MaxTokens, true), "length");
        let first_eos = GenerationStats {
            generated_tokens: 0,
            ..stats
        };
        assert_eq!(usage(first_eos)["completion_tokens"], 1);
        let cancelled = GenerationStats {
            generated_tokens: 0,
            finish: FinishReason::Cancelled,
            first_token_seconds: None,
            generation_seconds: None,
            ready_to_first_token_seconds: None,
            ..stats
        };
        assert_eq!(usage(cancelled)["completion_tokens"], 0);
        assert!(usage(cancelled)["moespresso"]["first_token_seconds"].is_null());
        assert!(usage(cancelled)["moespresso"]["generation_seconds"].is_null());
        assert!(usage(cancelled)["moespresso"]["ready_to_first_token_seconds"].is_null());
    }

    #[test]
    fn cache_usage_reports_restored_tokens_and_committed_writes() {
        let stats = GenerationStats {
            prompt_consumed: 1537,
            prefill_tokens: 513,
            prefill_seconds: Some(1.5),
            generated_tokens: 2,
            frontier: 1538,
            changed_routes: 0,
            finish: FinishReason::MaxTokens,
            first_token_seconds: None,
            generation_seconds: None,
            ready_to_first_token_seconds: None,
            prompt_cache: crate::generation::PromptCacheStats {
                event: "disk_hit",
                cached_tokens: 1024,
                disk_checkpoints_written: 1,
                last_disk_checkpoint: 1536,
                disk_checkpoint_write_seconds: 0.2,
                capacity_changed: true,
            },
        };
        let value = usage(stats);
        assert_eq!(value["prompt_tokens"], 1537);
        assert_eq!(value["moespresso"]["prefill_tokens"], 513);
        assert_eq!(value["moespresso"]["prefill_seconds"], 1.5);
        assert_eq!(value["prompt_tokens_details"]["cached_tokens"], 1024);
        assert_eq!(value["prompt_cache"]["event"], "disk_hit");
        assert_eq!(value["prompt_cache"]["disk_checkpoints_written"], 1);
        assert_eq!(value["prompt_cache"]["last_disk_checkpoint"], 1536);
        assert_eq!(value["prompt_cache"]["capacity_changed"], true);
    }

    #[test]
    fn resolved_request_thinking_selects_response_channels() {
        for (server_default, request_override, expected) in [
            (true, None, true),
            (false, None, false),
            (true, Some(false), false),
            (false, Some(true), true),
        ] {
            let mut body = json!({"messages": [{"role": "user", "content": "hi"}]});
            if let Some(enabled) = request_override {
                body["chat_template_kwargs"] = json!({"enable_thinking": enabled});
            }
            let input =
                parse_chat(&body, "pkg", SamplingOptions::default(), server_default).unwrap();
            let mut shaper = ResponseShaper::new(input.chat.thinking, None, "id".into());
            let mut deltas = Vec::new();
            if expected {
                deltas.extend(shaper.push("scratch</thi"));
                deltas.extend(shaper.push("nk>"));
            }
            deltas.extend(shaper.push("answer"));
            deltas.extend(shaper.finish(false));
            assert_eq!(shaper.content, "answer");
            assert_eq!(shaper.reasoning, if expected { "scratch" } else { "" });
            assert_eq!(
                deltas
                    .iter()
                    .any(|delta| matches!(delta, Delta::Reasoning(_))),
                expected
            );
            assert!(
                deltas
                    .iter()
                    .any(|delta| matches!(delta, Delta::Content(text) if text == "answer"))
            );
        }
    }

    #[test]
    fn loopback_sse_flushes_first_chunk_before_completion() {
        let server = Server::http("127.0.0.1:0").unwrap();
        let address = server.server_addr().to_ip().unwrap().to_string();
        let (release, stay_open) = mpsc::channel();
        let worker = thread::spawn(move || {
            let request = server.recv().unwrap();
            let mut writer = SseWriter::new(request, "id", "model", 1, true);
            writer.start().unwrap();
            thread::sleep(Duration::from_millis(100));
            writer
                .finish(
                    "stop",
                    json!({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
                )
                .unwrap();
            stay_open.recv_timeout(Duration::from_secs(3)).unwrap();
        });
        let mut client = TcpStream::connect(address).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        client
            .write_all(b"GET /sse HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut reader = BufReader::new(&mut client);
        let mut headers = String::new();
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            headers.push_str(&line);
            if line == "\r\n" {
                break;
            }
        }
        assert!(headers.contains("text/event-stream"));
        assert!(headers.contains("Transfer-Encoding: chunked"));
        let first = read_chunk(&mut reader).unwrap();
        assert!(first.contains("\"role\":\"assistant\""));
        assert!(!first.contains("[DONE]"));
        let mut body = first;
        while let Some(chunk) = read_chunk(&mut reader) {
            body.push_str(&chunk);
        }
        assert!(body.contains("[DONE]"));
        assert!(body.contains("\"completion_tokens\":1"));
        release.send(()).unwrap();
        worker.join().unwrap();
    }

    #[test]
    fn loopback_disconnect_returns_write_error() {
        let server = Server::http("127.0.0.1:0").unwrap();
        let address = server.server_addr().to_ip().unwrap().to_string();
        let worker = thread::spawn(move || {
            let request = server.recv().unwrap();
            let mut writer = SseWriter::new(request, "id", "model", 1, false);
            writer.start().unwrap();
            thread::sleep(Duration::from_millis(100));
            let block = [b'x'; 16384];
            (0..256).any(|_| writer.write(&block).is_err())
        });
        let mut client = TcpStream::connect(address).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        client
            .write_all(b"GET /sse HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .unwrap();
        let first = read_until(&mut client, "\"role\":\"assistant\"");
        assert!(first.contains("\"role\":\"assistant\""));
        client.shutdown(Shutdown::Both).unwrap();
        drop(client);
        assert!(worker.join().unwrap());
    }

    #[test]
    fn dispatcher_returns_the_latest_published_health_without_owner_work() {
        let server = Server::http("127.0.0.1:0").unwrap();
        let address = server.server_addr().to_ip().unwrap().to_string();
        let health = RwLock::new(json!({
            "status": "ok", "model": "artifact", "expert_capacity": 160,
            "prompt_cache": {"disk_enabled": true},
        }));
        let (sender, _receiver) = sync_channel(MAX_QUEUED_REQUESTS);
        thread::scope(|scope| {
            let dispatcher =
                scope.spawn(|| dispatch_requests_with_health(&server, &health, sender));
            let _unblock = UnblockOnDrop(&server);
            let (status, initial) =
                read_json_response(send_request(&address, "GET", "/health", None));
            assert_eq!(status, 200);
            assert_eq!(initial["expert_capacity"], 160);
            {
                let mut snapshot = health.write().unwrap();
                snapshot["expert_capacity"] = json!(144);
                snapshot["prompt_cache"]["disk_enabled"] = json!(false);
            }
            let (status, updated) =
                read_json_response(send_request(&address, "GET", "/health", None));
            assert_eq!(status, 200);
            assert_eq!(updated["expert_capacity"], 144);
            assert_eq!(updated["prompt_cache"]["disk_enabled"], false);
            server.unblock();
            dispatcher.join().unwrap();
        });
    }

    #[test]
    fn dispatcher_answers_health_while_inference_request_is_held() {
        let server = Server::http("127.0.0.1:0").unwrap();
        let address = server.server_addr().to_ip().unwrap().to_string();
        let (sender, receiver) = sync_channel(MAX_QUEUED_REQUESTS);
        thread::scope(|scope| {
            let dispatcher = scope.spawn(|| dispatch_requests(&server, "artifact", sender));
            let _unblock = UnblockOnDrop(&server);
            let chat = send_request(&address, "POST", "/v1/chat/completions", None);
            let held = receiver.recv_timeout(Duration::from_secs(2)).unwrap();
            assert_eq!(held.method(), &Method::Post);
            assert_eq!(held.url(), "/v1/chat/completions");

            let (status, health) =
                read_json_response(send_request(&address, "GET", "/health", None));
            assert_eq!(status, 200);
            assert_eq!(health, json!({"status": "ok", "model": "artifact"}));

            json_response(held, 200, &json!({"held": true})).unwrap();
            let (status, body) = read_json_response(chat);
            assert_eq!(status, 200);
            assert_eq!(body, json!({"held": true}));
            server.unblock();
            dispatcher.join().unwrap();
        });
    }

    #[test]
    fn dispatcher_rejects_full_queue_without_losing_queued_order() {
        let server = Server::http("127.0.0.1:0").unwrap();
        let address = server.server_addr().to_ip().unwrap().to_string();
        let (sender, receiver) = sync_channel(MAX_QUEUED_REQUESTS);
        let mut queued_clients = Vec::new();
        for index in 0..MAX_QUEUED_REQUESTS {
            let client = send_request(&address, "POST", "/v1/chat/completions", Some(index));
            let request = server
                .recv_timeout(Duration::from_secs(2))
                .unwrap()
                .unwrap();
            sender.try_send(request).unwrap();
            queued_clients.push(client);
        }
        thread::scope(|scope| {
            let dispatcher = scope.spawn(|| dispatch_requests(&server, "artifact", sender));
            let _unblock = UnblockOnDrop(&server);
            let overflow = send_request(&address, "POST", "/v1/chat/completions", None);
            let (status, body) = read_json_response(overflow);
            assert_eq!(status, 503);
            assert_eq!(body["error"]["message"], "inference queue is full");

            for (index, client) in queued_clients.into_iter().enumerate() {
                let request = receiver.recv_timeout(Duration::from_secs(2)).unwrap();
                let marker = request
                    .headers()
                    .iter()
                    .find(|header| header.field.equiv("X-Queue-Index"))
                    .unwrap();
                assert_eq!(marker.value.as_str(), index.to_string());
                json_response(request, 200, &json!({"index": index})).unwrap();
                let (status, body) = read_json_response(client);
                assert_eq!(status, 200);
                assert_eq!(body["index"], index);
            }
            assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));
            server.unblock();
            dispatcher.join().unwrap();
        });
    }

    #[test]
    fn dispatcher_methods_routes_and_disconnected_owner() {
        let server = Server::http("127.0.0.1:0").unwrap();
        let address = server.server_addr().to_ip().unwrap().to_string();
        let (sender, receiver) = sync_channel(MAX_QUEUED_REQUESTS);
        thread::scope(|scope| {
            let dispatcher = scope.spawn(|| dispatch_requests(&server, "artifact", sender));
            let _unblock = UnblockOnDrop(&server);
            for (method, path, expected) in [
                ("POST", "/health", 405),
                ("GET", "/v1/chat/completions", 405),
                ("GET", "/v1/models", 404),
                ("POST", "/other", 404),
                ("POST", "/v1/moespresso/expert-capacity", 404),
                ("GET", "/v1/moespresso/expert-capacity", 404),
            ] {
                let (status, body) = read_json_response(send_request(&address, method, path, None));
                assert_eq!(status, expected, "{method} {path}: {body}");
                assert!(body["error"]["message"].is_string());
            }
            assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));
            drop(receiver);
            let (status, body) =
                read_json_response(send_request(&address, "POST", "/v1/chat/completions", None));
            assert_eq!(status, 500);
            assert_eq!(body["error"]["message"], "inference worker is unavailable");
            dispatcher.join().unwrap();
        });
    }

    #[test]
    fn owner_unwind_unblocks_dispatcher_and_preserves_admitted_work() {
        let server = Server::http("127.0.0.1:0").unwrap();
        let address = server.server_addr().to_ip().unwrap().to_string();
        let (sender, receiver) = sync_channel(MAX_QUEUED_REQUESTS);
        let client = send_request(&address, "POST", "/v1/chat/completions", None);
        let request = server
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap();
        sender.try_send(request).unwrap();
        thread::scope(|scope| {
            let (finished, finished_rx) = mpsc::channel();
            let server_ref = &server;
            let dispatcher = scope.spawn(move || {
                dispatch_requests(server_ref, "artifact", sender);
                finished.send(()).unwrap();
            });
            let owner_result = catch_unwind(AssertUnwindSafe(|| {
                let _unblock = UnblockOnDrop(&server);
                panic!("synthetic inference-owner unwind");
            }));
            assert!(owner_result.is_err());
            finished_rx.recv_timeout(Duration::from_secs(2)).unwrap();

            let admitted = receiver.recv_timeout(Duration::from_secs(2)).unwrap();
            json_response(admitted, 200, &json!({"admitted": true})).unwrap();
            assert!(receiver.recv_timeout(Duration::from_secs(2)).is_err());
            let (status, body) = read_json_response(client);
            assert_eq!(status, 200);
            assert_eq!(body, json!({"admitted": true}));
            dispatcher.join().unwrap();
        });
    }

    #[test]
    fn explicit_cache_budget_covers_retained_and_disk_io_overlap() {
        let io = crate::engine::IO_BUFFER_BYTES;
        let options = ServeCacheOptions {
            memory_retained_bytes: 400,
            host_overlap_bytes: 400 + io,
            disk: Some(ServeDiskOptions {
                root: PathBuf::from("/unused-by-budget-test"),
                managed_cache_base: None,
                max_file_bytes: 400,
                max_io_buffer_bytes: io,
                max_payload_bytes: 800,
                max_entries: 2,
                stride: 1024,
                write_depth: 16_384,
                mode: DiskCacheMode::Prefix,
                required: true,
            }),
        };
        assert!(validate_cache_options(&options).is_ok());
        let too_small = ServeCacheOptions {
            host_overlap_bytes: 399 + io,
            ..options
        };
        assert!(validate_cache_options(&too_small).is_err());
        let overflow = ServeCacheOptions {
            memory_retained_bytes: u64::MAX,
            host_overlap_bytes: u64::MAX,
            disk: Some(ServeDiskOptions {
                root: PathBuf::from("/unused-by-budget-test"),
                managed_cache_base: None,
                max_file_bytes: 400,
                max_io_buffer_bytes: io,
                max_payload_bytes: 800,
                max_entries: 2,
                stride: 1024,
                write_depth: 16_384,
                mode: DiskCacheMode::Prefix,
                required: true,
            }),
        };
        assert!(validate_cache_options(&overflow).is_err());

        let disk_only = ServeCacheOptions {
            memory_retained_bytes: 0,
            host_overlap_bytes: io,
            disk: overflow.disk,
        };
        assert!(validate_cache_options(&disk_only).is_ok());
        let invalid_policy = ServeCacheOptions {
            memory_retained_bytes: 0,
            host_overlap_bytes: io,
            disk: Some(ServeDiskOptions {
                stride: 513,
                ..disk_only.disk.unwrap()
            }),
        };
        assert!(validate_cache_options(&invalid_policy).is_err());
        let invalid_store = ServeCacheOptions {
            memory_retained_bytes: 0,
            host_overlap_bytes: io,
            disk: Some(ServeDiskOptions {
                max_entries: 65,
                stride: 1024,
                required: false,
                ..invalid_policy.disk.unwrap()
            }),
        };
        assert!(validate_cache_options(&invalid_store).is_err());
        let relative_root = ServeCacheOptions {
            memory_retained_bytes: 0,
            host_overlap_bytes: io,
            disk: Some(ServeDiskOptions {
                root: PathBuf::from("relative"),
                max_entries: 2,
                stride: 1024,
                ..invalid_store.disk.unwrap()
            }),
        };
        assert!(validate_cache_options(&relative_root).is_err());
        let insufficient_staging = ServeCacheOptions {
            memory_retained_bytes: 0,
            host_overlap_bytes: io,
            disk: Some(ServeDiskOptions {
                root: PathBuf::from("/unused-by-budget-test"),
                max_io_buffer_bytes: 1,
                ..relative_root.disk.unwrap()
            }),
        };
        assert!(validate_cache_options(&insufficient_staging).is_err());
        assert!(
            validate_cache_options(&ServeCacheOptions {
                memory_retained_bytes: 0,
                host_overlap_bytes: 0,
                disk: None,
            })
            .is_err()
        );
    }
}
