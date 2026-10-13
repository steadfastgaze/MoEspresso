//! Synchronous generation over one single-owner engine session.
//!
//! Prefill always uses original routing. A generated token is emitted from the
//! previous step's logits before it is consumed to obtain the next logits.
//! This keeps the committed frontier distinct from the number of visible tokens.

use std::cell::RefCell;
use std::time::Instant;

use anyhow::{Result, ensure};

#[cfg(test)]
use crate::engine::TokenOutput;
use crate::engine::{DiskCheckpointStore, Engine, PromptPrefixCache};
use crate::routing::RoutingMode;
use crate::sampling::{Sampler, SamplingOptions};
use crate::tokenizer::TextTokenizer;

pub(crate) mod request_resources;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    MaxTokens,
    StopToken(u32),
    ContextLimit,
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PromptCacheStats {
    pub event: &'static str,
    pub cached_tokens: usize,
    pub disk_checkpoints_written: usize,
    pub last_disk_checkpoint: usize,
    pub disk_checkpoint_write_seconds: f64,
    pub capacity_changed: bool,
}

impl Default for PromptCacheStats {
    fn default() -> Self {
        Self {
            event: "disabled",
            cached_tokens: 0,
            disk_checkpoints_written: 0,
            last_disk_checkpoint: 0,
            disk_checkpoint_write_seconds: 0.0,
            capacity_changed: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerationStats {
    /// Prompt tokens represented by committed state, including a restored prefix.
    pub prompt_consumed: usize,
    /// Prompt tokens actually evaluated, excluding restored prefixes.
    pub prefill_tokens: usize,
    /// Completed prompt-consumption wall time, including checkpoint writes,
    /// pressure checks and fused first sampling, excluding cache restoration.
    pub prefill_seconds: Option<f64>,
    pub generated_tokens: usize,
    /// Tokens committed to the engine. The last emitted token may be uncommitted.
    pub frontier: usize,
    pub changed_routes: u64,
    pub finish: FinishReason,
    /// Post-lookup start to first selected-ID delivery, including EOS and
    /// prompt checkpoint offers. A cancelled selection is not delivered.
    pub first_token_seconds: Option<f64>,
    /// Same origin, through the completed model loop. Per-token streaming
    /// callbacks are included; terminal text flush, final checkpoint publication,
    /// session cleanup and final HTTP response delivery are not.
    pub generation_seconds: Option<f64>,
    /// Pre-lookup readiness to first selected-ID delivery, including restore.
    pub ready_to_first_token_seconds: Option<f64>,
    /// Completed cache work for this request, never inferred from response text.
    pub prompt_cache: PromptCacheStats,
}

struct GenerationTimer {
    ready_at: Instant,
    started_at: Instant,
    first_at: Option<Instant>,
}

impl GenerationTimer {
    fn new(ready_at: Instant, started_at: Instant) -> Self {
        Self {
            ready_at,
            started_at,
            first_at: None,
        }
    }

    fn sampled(&mut self, now: impl FnOnce() -> Instant) {
        if self.first_at.is_none() {
            self.first_at = Some(now());
        }
    }

    fn finish(self, stats: &mut GenerationStats, finished_at: Instant) {
        stats.first_token_seconds = self
            .first_at
            .map(|at| at.duration_since(self.started_at).as_secs_f64());
        stats.ready_to_first_token_seconds = self
            .first_at
            .map(|at| at.duration_since(self.ready_at).as_secs_f64());
        stats.generation_seconds = Some(finished_at.duration_since(self.started_at).as_secs_f64());
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GenerationOptions {
    pub max_tokens: usize,
    pub max_context: usize,
    pub decode_mode: RoutingMode,
    pub sampling: SamplingOptions,
    pub seed: u64,
}

#[cfg(test)]
struct PromptStart {
    frontier: usize,
    logits: Vec<u16>,
}

enum GenerationAction<'a> {
    Consume {
        tokens: &'a [u32],
        mode: RoutingMode,
        sample: bool,
        history: &'a [u32],
    },
    SampleCurrent {
        history: &'a [u32],
    },
}

enum GenerationActionResult {
    Consumed {
        frontier: usize,
        changed_routes: u32,
        selected_id: Option<u32>,
    },
    Sampled(u32),
    Cancelled,
}

/// Request-scoped access to the Engine owner's optional prompt tiers.
pub struct GenerationCaches<'a> {
    pub memory: Option<&'a mut PromptPrefixCache>,
    pub disk: Option<&'a mut GenerationDiskCache>,
}

/// Indexed disk checkpoints with bounded prefill anchors. Full mode also saves
/// completed prompts and committed generated frontiers, independently of depth.
pub struct GenerationDiskCache {
    store: DiskCheckpointStore,
    stride: usize,
    write_depth: usize,
}

impl GenerationDiskCache {
    pub fn mode(&self) -> crate::engine::DiskCacheMode {
        self.store.mode()
    }

    fn saves_final_frontiers(&self) -> bool {
        self.mode() == crate::engine::DiskCacheMode::Full
    }

    pub fn new(store: DiskCheckpointStore, stride: usize, write_depth: usize) -> Result<Self> {
        Self::validate_policy(stride, write_depth)?;
        Ok(Self {
            store,
            stride,
            write_depth,
        })
    }

    fn should_write(
        &self,
        frontier: usize,
        tokens: usize,
        mode: RoutingMode,
        prefill_width: usize,
    ) -> bool {
        disk_write_eligible(
            frontier,
            tokens,
            mode,
            self.stride,
            self.write_depth,
            prefill_width,
        )
    }

    fn load_match(
        &mut self,
        engine: &Engine,
        prompt: &[u32],
    ) -> Result<Option<crate::engine::Session>> {
        self.store.load_match(engine, prompt)
    }

    fn has_match_beyond(&self, engine: &Engine, prompt: &[u32], frontier: usize) -> Result<bool> {
        self.store.has_match_beyond(engine, prompt, frontier)
    }

    fn save(&mut self, engine: &Engine, session: &crate::engine::Session) -> Result<bool> {
        self.store.save(engine, session)
    }

    fn checkpoint(
        &mut self,
        engine: &Engine,
        session: &crate::engine::Session,
        stats: &mut PromptCacheStats,
    ) -> Result<()> {
        let started = Instant::now();
        if self.save(engine, session)? {
            let elapsed = started.elapsed().as_secs_f64();
            stats.disk_checkpoints_written += 1;
            stats.last_disk_checkpoint = session.frontier()?;
            stats.disk_checkpoint_write_seconds += elapsed;
            eprintln!(
                "prompt cache event=disk_write frontier={} elapsed_ms={:.3}",
                stats.last_disk_checkpoint,
                elapsed * 1000.0
            );
        }
        Ok(())
    }
    pub(crate) fn validate_policy(stride: usize, write_depth: usize) -> Result<()> {
        ensure!(
            stride >= 512 && stride.is_multiple_of(512),
            "disk checkpoint stride must be a positive multiple of 512"
        );
        ensure!(
            write_depth > 0,
            "disk checkpoint write depth must be positive"
        );
        Ok(())
    }
}

fn disk_write_eligible(
    frontier: usize,
    tokens: usize,
    mode: RoutingMode,
    stride: usize,
    write_depth: usize,
    prefill_width: usize,
) -> bool {
    mode == RoutingMode::Unbiased
        && matches!(prefill_width, 512 | 2048 | 4096)
        && tokens == prefill_width
        && frontier >= tokens
        && stride > 0
        && frontier <= write_depth
        && frontier / stride > (frontier - tokens) / stride
}

impl GenerationCaches<'_> {
    fn enabled(&self) -> bool {
        self.memory.is_some() || self.disk.is_some()
    }
}

fn validate_request(
    engine: &Engine,
    tokenizer: &TextTokenizer,
    prompt: &[u32],
    options: GenerationOptions,
) -> Result<()> {
    ensure!(
        engine.prefill_workspace_ready(),
        "prefill workspace needs repair before generation"
    );
    validate_request_inputs(
        prompt,
        tokenizer.stop_ids(),
        options,
        engine.context_limit(),
        engine.prefill_chunk_size(),
    )
}

fn validate_request_inputs(
    prompt: &[u32],
    stop_ids: &[u32],
    options: GenerationOptions,
    engine_context: usize,
    prefill_width: usize,
) -> Result<()> {
    ensure!(!prompt.is_empty(), "generation requires a nonempty prompt");
    ensure!(!stop_ids.is_empty(), "generation requires stop token IDs");
    ensure!(options.max_tokens > 0, "max tokens must be positive");
    ensure!(
        options.max_context > 0 && options.max_context <= engine_context,
        "generation context exceeds loaded Engine capacity"
    );
    ensure!(
        prompt.len() <= options.max_context,
        "rendered prompt exceeds context limit"
    );
    ensure!(
        prompt.iter().all(|&token| token < 248_320),
        "rendered prompt contains an invalid token ID"
    );
    ensure!(
        (1..=crate::prefill_capacity::MAX_ROWS).contains(&prefill_width),
        "invalid Engine prefill width"
    );
    Ok(())
}

fn offer_memory_at(frontier: usize, prompt_len: usize, prefill_width: usize) -> bool {
    let aligned = prompt_len / prefill_width * prefill_width;
    frontier == prompt_len || (aligned > 0 && frontier == aligned)
}

#[allow(clippy::too_many_arguments)]
fn run_loop(
    prompt: &[u32],
    stop_ids: &[u32],
    options: GenerationOptions,
    prefill_width: usize,
    initial_frontier: usize,
    mut action: impl for<'a> FnMut(GenerationAction<'a>) -> Result<GenerationActionResult>,
    #[cfg(not(test))] mut on_sample: impl FnMut(),
    #[cfg(test)] mut on_sample: impl FnMut() -> Result<()>,
    mut emit: impl FnMut(u32) -> Result<()>,
    mut cancelled: impl FnMut() -> bool,
) -> Result<GenerationStats> {
    ensure!(!prompt.is_empty(), "generation requires a nonempty prompt");
    ensure!(!stop_ids.is_empty(), "generation requires stop token IDs");
    ensure!(options.max_tokens > 0, "max tokens must be positive");
    ensure!(
        (1..=crate::prefill_capacity::MAX_ROWS).contains(&prefill_width),
        "invalid prefill width"
    );
    ensure!(
        prompt.len() <= options.max_context,
        "rendered prompt exceeds context limit"
    );
    ensure!(
        initial_frontier <= prompt.len(),
        "restored frontier exceeds the rendered prompt"
    );
    let mut stats = GenerationStats {
        prompt_consumed: initial_frontier,
        prefill_tokens: 0,
        prefill_seconds: Some(0.0),
        generated_tokens: 0,
        frontier: initial_frontier,
        changed_routes: 0,
        finish: FinishReason::Cancelled,
        first_token_seconds: None,
        generation_seconds: None,
        ready_to_first_token_seconds: None,
        prompt_cache: PromptCacheStats::default(),
    };
    let mut history = prompt.to_vec();
    let mut selected = None;
    for tokens in prompt[initial_frontier..].chunks(prefill_width) {
        if cancelled() {
            return Ok(stats);
        }
        let sample = stats.frontier + tokens.len() == prompt.len();
        let prefill_started = Instant::now();
        let result = action(GenerationAction::Consume {
            tokens,
            mode: RoutingMode::Unbiased,
            sample,
            history: &history,
        })?;
        if matches!(result, GenerationActionResult::Cancelled) {
            *stats.prefill_seconds.as_mut().unwrap() += prefill_started.elapsed().as_secs_f64();
            return Ok(stats);
        }
        let GenerationActionResult::Consumed {
            frontier,
            changed_routes,
            selected_id,
        } = result
        else {
            anyhow::bail!("generation consume action returned a sample-only result");
        };
        stats.frontier += tokens.len();
        ensure!(
            frontier == stats.frontier && selected_id.is_some() == sample,
            "engine frontier differs from generation frontier"
        );
        stats.prompt_consumed += tokens.len();
        stats.prefill_tokens += tokens.len();
        *stats.prefill_seconds.as_mut().unwrap() += prefill_started.elapsed().as_secs_f64();
        stats.changed_routes += u64::from(changed_routes);
        selected = selected_id;
    }
    loop {
        if cancelled() {
            return Ok(stats);
        }
        let token = if let Some(token) = selected.take() {
            token
        } else {
            match action(GenerationAction::SampleCurrent { history: &history })? {
                GenerationActionResult::Sampled(token) => token,
                GenerationActionResult::Cancelled => return Ok(stats),
                GenerationActionResult::Consumed { .. } => {
                    anyhow::bail!("generation sample action consumed a token")
                }
            }
        };
        #[cfg(not(test))]
        on_sample();
        #[cfg(test)]
        on_sample()?;
        if stop_ids.contains(&token) {
            stats.finish = FinishReason::StopToken(token);
            return Ok(stats);
        }
        {
            emit(token)?;
        }
        stats.generated_tokens += 1;
        history.push(token);
        if stats.generated_tokens == options.max_tokens {
            stats.finish = FinishReason::MaxTokens;
            return Ok(stats);
        }
        if stats.frontier == options.max_context {
            stats.finish = FinishReason::ContextLimit;
            return Ok(stats);
        }
        if cancelled() {
            return Ok(stats);
        }
        let result = {
            action(GenerationAction::Consume {
                tokens: &[token],
                mode: options.decode_mode,
                sample: true,
                history: &history,
            })?
        };
        if matches!(result, GenerationActionResult::Cancelled) {
            return Ok(stats);
        }
        let GenerationActionResult::Consumed {
            frontier,
            changed_routes,
            selected_id,
        } = result
        else {
            anyhow::bail!("generation decode action returned a sample-only result");
        };
        stats.frontier += 1;
        ensure!(
            frontier == stats.frontier && selected_id.is_some(),
            "engine frontier differs from generation frontier"
        );
        stats.changed_routes += u64::from(changed_routes);
        selected = selected_id;
    }
}

// Diagnostic tests adapt their logit-producing closures to the same control
// flow used by production generation. The adapter has no scheduling policy.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn token_loop(
    prompt: &[u32],
    stop_ids: &[u32],
    options: GenerationOptions,
    prefill_width: usize,
    initial: Option<PromptStart>,
    mut step: impl FnMut(&[u32], RoutingMode) -> Result<TokenOutput>,
    mut sample: impl FnMut(&[u16], &[u32]) -> Result<u32>,
    emit: impl FnMut(u32) -> Result<()>,
    cancelled: impl FnMut() -> bool,
) -> Result<GenerationStats> {
    let initial = initial.unwrap_or(PromptStart {
        frontier: 0,
        logits: Vec::new(),
    });
    ensure!(
        (initial.frontier == 0) == initial.logits.is_empty(),
        "restored prompt frontier differs from the cold chunk schedule"
    );
    let mut logits = initial.logits;
    run_loop(
        prompt,
        stop_ids,
        options,
        prefill_width,
        initial.frontier,
        |action| match action {
            GenerationAction::Consume {
                tokens,
                mode,
                sample: draw,
                history,
            } => {
                let output = step(tokens, mode)?;
                logits = output.logits_bf16;
                let selected_id = if draw {
                    Some(sample(&logits, history)?)
                } else {
                    None
                };
                Ok(GenerationActionResult::Consumed {
                    frontier: output.frontier,
                    changed_routes: output.changed_routes,
                    selected_id,
                })
            }
            GenerationAction::SampleCurrent { history } => {
                Ok(GenerationActionResult::Sampled(sample(&logits, history)?))
            }
        },
        || Ok(()),
        emit,
        cancelled,
    )
}

/// Generate from a rendered prompt using a fresh session and streamed UTF-8.
/// Cancellation is observed between prefill layers, committed engine steps and emission.
/// A callback error stops without consuming the selected, uncommitted token.
pub fn generate(
    engine: &mut Engine,
    tokenizer: &TextTokenizer,
    prompt: &[u32],
    options: GenerationOptions,
    on_text: impl FnMut(&str) -> Result<()>,
    cancelled: impl FnMut() -> bool,
) -> Result<GenerationStats> {
    generate_inner(
        engine,
        tokenizer,
        prompt,
        options,
        GenerationCaches {
            memory: None,
            disk: None,
        },
        on_text,
        cancelled,
        |_| Ok(false),
        #[cfg(test)]
        None,
    )
}

/// Inspect memory first, then use a deeper valid disk prefix when available.
/// Only committed frontiers are published, including consumed generated tokens.
pub fn generate_with_cache(
    engine: &mut Engine,
    tokenizer: &TextTokenizer,
    prompt: &[u32],
    options: GenerationOptions,
    caches: GenerationCaches<'_>,
    on_text: impl FnMut(&str) -> Result<()>,
    cancelled: impl FnMut() -> bool,
) -> Result<GenerationStats> {
    generate_inner(
        engine,
        tokenizer,
        prompt,
        options,
        caches,
        on_text,
        cancelled,
        |_| Ok(false),
        #[cfg(test)]
        None,
    )
}

/// Run owner-controlled work after completed GPU readers, including layer
/// boundaries inside wide bounded prefill. The callback reports capacity
/// changes. Prompt checkpoints are published only after the whole chunk commits.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_with_cache_controlled(
    engine: &mut Engine,
    tokenizer: &TextTokenizer,
    prompt: &[u32],
    options: GenerationOptions,
    caches: GenerationCaches<'_>,
    on_text: impl FnMut(&str) -> Result<()>,
    cancelled: impl FnMut() -> bool,
    boundary: impl FnMut(&mut Engine) -> Result<bool>,
) -> Result<GenerationStats> {
    generate_inner(
        engine,
        tokenizer,
        prompt,
        options,
        caches,
        on_text,
        cancelled,
        boundary,
        #[cfg(test)]
        None,
    )
}

/// Optional request resource control, serialized through one owner callback.
/// Ordinary callers retain the completed-operation-only generation entry.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_with_cache_owned(
    engine: &mut Engine,
    tokenizer: &TextTokenizer,
    prompt: &[u32],
    options: GenerationOptions,
    caches: GenerationCaches<'_>,
    on_text: impl FnMut(&str) -> Result<()>,
    cancelled: impl FnMut() -> bool,
    mut owner: impl FnMut(&mut Engine, RequestPhase) -> Result<bool>,
) -> Result<GenerationStats> {
    let before_prepare = engine.expert_capacity();
    let prepared_changed = owner(engine, RequestPhase::Prepare)?;
    ensure!(
        engine.prefill_workspace_ready(),
        "prefill workspace needs repair before generation"
    );
    let phase_changed =
        std::cell::Cell::new(prepared_changed || before_prepare != engine.expert_capacity());
    let owner = RefCell::new(owner);
    let mut stats = generate_inner_with_phases(
        engine,
        tokenizer,
        prompt,
        options,
        caches,
        on_text,
        cancelled,
        |engine| owner.borrow_mut()(engine, RequestPhase::CompletedOperation),
        #[cfg(test)]
        None,
        Some(&mut |engine, phase| {
            let previous = engine.expert_capacity();
            let changed = owner.borrow_mut()(engine, phase)?;
            phase_changed
                .set(phase_changed.get() || changed || previous != engine.expert_capacity());
            Ok(())
        }),
    )?;
    stats.prompt_cache.capacity_changed |= phase_changed.get();
    Ok(stats)
}

fn restore_disk_prefix(
    disk: &mut GenerationDiskCache,
    engine: &Engine,
    prompt: &[u32],
    minimum_frontier: usize,
    stats: &mut PromptCacheStats,
) -> Result<Option<(crate::engine::Session, usize)>> {
    let Some(session) = disk.load_match(engine, prompt)? else {
        return Ok(None);
    };
    let frontier = session.frontier()?;
    if frontier < minimum_frontier {
        // Release the shorter device candidate before restoring healthy RAM.
        return Ok(None);
    }
    eprintln!("prompt cache event=disk_hit frontier={frontier}");
    stats.event = "disk_hit";
    stats.cached_tokens = frontier;
    Ok(Some((session, frontier)))
}

#[allow(clippy::too_many_arguments)]
fn generate_inner(
    engine: &mut Engine,
    tokenizer: &TextTokenizer,
    prompt: &[u32],
    options: GenerationOptions,
    caches: GenerationCaches<'_>,
    on_text: impl FnMut(&str) -> Result<()>,
    cancelled: impl FnMut() -> bool,
    boundary: impl FnMut(&mut Engine) -> Result<bool>,
    #[cfg(test)] sample_observer: Option<&mut dyn FnMut() -> Result<()>>,
) -> Result<GenerationStats> {
    generate_inner_with_phases(
        engine,
        tokenizer,
        prompt,
        options,
        caches,
        on_text,
        cancelled,
        boundary,
        #[cfg(test)]
        sample_observer,
        None,
    )
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum RequestPhase {
    Prepare,
    Prefill { new_tokens: usize },
    CompletedOperation,
    Decode,
    Finished,
}

type RequestPhaseHook<'a> = &'a mut dyn FnMut(&mut Engine, RequestPhase) -> Result<()>;

#[allow(clippy::too_many_arguments)]
fn generate_inner_with_phases(
    engine: &mut Engine,
    tokenizer: &TextTokenizer,
    prompt: &[u32],
    options: GenerationOptions,
    mut caches: GenerationCaches<'_>,
    on_text: impl FnMut(&str) -> Result<()>,
    cancelled: impl FnMut() -> bool,
    mut boundary: impl FnMut(&mut Engine) -> Result<bool>,
    #[cfg(test)] mut sample_observer: Option<&mut dyn FnMut() -> Result<()>>,
    mut phase: Option<RequestPhaseHook<'_>>,
) -> Result<GenerationStats> {
    validate_request(engine, tokenizer, prompt, options)?;
    let mut sampler = Sampler::new(options.sampling, options.seed)?;
    let cancelled = RefCell::new(cancelled);
    if (cancelled.borrow_mut())() {
        return Ok(GenerationStats {
            prompt_consumed: 0,
            prefill_tokens: 0,
            prefill_seconds: None,
            generated_tokens: 0,
            frontier: 0,
            changed_routes: 0,
            finish: FinishReason::Cancelled,
            first_token_seconds: None,
            generation_seconds: None,
            ready_to_first_token_seconds: None,
            prompt_cache: PromptCacheStats::default(),
        });
    }
    let ready_at = Instant::now();
    let caches_enabled = caches.enabled();
    let mut cache_stats = PromptCacheStats {
        event: if caches_enabled {
            "cold_start"
        } else {
            "disabled"
        },
        ..PromptCacheStats::default()
    };
    let mut memory_active = caches.memory.is_some();
    let mut disk_active = caches.disk.is_some();
    let mut restored = None;
    let memory_frontier = caches
        .memory
        .as_deref_mut()
        .and_then(|cache| cache.best_match(engine, prompt))
        .map_or(0, |hit| hit.matched_tokens());
    if memory_frontier > 0
        && memory_frontier < prompt.len()
        && !(cancelled.borrow_mut())()
        && let Some(disk) = caches.disk.as_deref_mut()
    {
        match disk.has_match_beyond(engine, prompt, memory_frontier) {
            Ok(true) => match restore_disk_prefix(
                disk,
                engine,
                prompt,
                memory_frontier + 1,
                &mut cache_stats,
            ) {
                Ok(candidate) => restored = candidate,
                Err(error) => {
                    eprintln!("prompt cache event=cache_error tier=disk action=read: {error:#}");
                    disk_active = false;
                }
            },
            Ok(false) => {}
            Err(error) => {
                eprintln!("prompt cache event=cache_error tier=disk action=lookup: {error:#}");
                disk_active = false;
            }
        }
    }
    if restored.is_none()
        && let Some(cache) = caches.memory.as_deref_mut()
        && let Some(hit) = cache.best_match(engine, prompt)
    {
        match engine.restore(hit.snapshot()) {
            Ok(session) => {
                eprintln!(
                    "prompt cache event=memory_hit frontier={}",
                    hit.matched_tokens()
                );
                restored = Some((session, hit.matched_tokens()));
                cache_stats.event = "memory_hit";
                cache_stats.cached_tokens = hit.matched_tokens();
            }
            Err(error) => {
                eprintln!("prompt cache event=cache_error tier=memory action=restore: {error:#}");
                memory_active = false;
            }
        }
    }
    if restored.is_none()
        && disk_active
        && !(cancelled.borrow_mut())()
        && let Some(disk) = caches.disk.as_deref_mut()
    {
        match restore_disk_prefix(disk, engine, prompt, 0, &mut cache_stats) {
            Ok(candidate) => restored = candidate,
            Err(error) => {
                eprintln!("prompt cache event=cache_error tier=disk action=read: {error:#}");
                disk_active = false;
            }
        }
    }
    if (cancelled.borrow_mut())() {
        return Ok(GenerationStats {
            prompt_consumed: cache_stats.cached_tokens,
            prefill_tokens: 0,
            prefill_seconds: None,
            generated_tokens: 0,
            frontier: cache_stats.cached_tokens,
            changed_routes: 0,
            finish: FinishReason::Cancelled,
            first_token_seconds: None,
            generation_seconds: None,
            ready_to_first_token_seconds: None,
            prompt_cache: cache_stats,
        });
    }
    if cache_stats.event == "disk_hit"
        && cache_stats.cached_tokens == prompt.len()
        && memory_active
        && let Some((session, _)) = restored.as_ref()
        && let Some(cache) = caches.memory.as_deref_mut()
    {
        match cache.offer_prompt_checkpoint(engine, session, prompt) {
            Ok(true) => eprintln!("prompt cache event=memory_fill frontier={}", prompt.len()),
            Ok(false) => {}
            Err(error) => {
                eprintln!("prompt cache event=cache_error tier=memory action=offer: {error:#}");
                memory_active = false;
            }
        }
    }
    // Lookup includes restoration and complete-hit cache insertion. Keep its
    // cost in ready-to-first, outside the generation clock shared by first/total.
    let mut timing = GenerationTimer::new(ready_at, Instant::now());
    let (mut session, initial_frontier) = match restored {
        Some((session, frontier)) => (session, frontier),
        None => {
            if caches_enabled {
                eprintln!("prompt cache event=cold_start");
            }
            (engine.start_session()?, 0)
        }
    };
    let mut decoder = tokenizer.stream_decoder();
    let mut on_text = on_text;
    let mut emitted_ids = Vec::new();
    let mut emitted_text = String::new();
    if let Some(phase) = phase.as_mut() {
        let entered = phase(
            engine,
            RequestPhase::Prefill {
                new_tokens: prompt.len() - initial_frontier,
            },
        );
        if let Err(primary) = entered {
            return match phase(engine, RequestPhase::Finished) {
                Ok(()) => Err(primary),
                Err(cleanup) => {
                    Err(primary
                        .context(format!("request resource cleanup also failed: {cleanup:#}")))
                }
            };
        }
    }
    let prefill_width = engine.prefill_chunk_size();
    let result = run_loop(
        prompt,
        tokenizer.stop_ids(),
        options,
        prefill_width,
        initial_frontier,
        |action| match action {
            GenerationAction::Consume {
                tokens,
                mode,
                sample,
                history,
            } => {
                let mut poll_cancelled = || (cancelled.borrow_mut())();
                let mut control = crate::engine::GenerationControl {
                    cancelled: &mut poll_cancelled,
                    boundary: &mut boundary,
                    capacity_changed: false,
                };
                let output = engine.consume_for_generation(
                    tokens,
                    mode,
                    &mut session,
                    history,
                    if sample { Some(&mut sampler) } else { None },
                    Some(&mut control),
                )?;
                cache_stats.capacity_changed |= control.capacity_changed;
                let Some(output) = output else {
                    return Ok(GenerationActionResult::Cancelled);
                };
                if boundary(engine)? {
                    cache_stats.capacity_changed = true;
                }
                if output.frontier == prompt.len()
                    && mode == RoutingMode::Unbiased
                    && let Some(phase) = phase.as_mut()
                {
                    phase(engine, RequestPhase::Decode)?;
                }
                if output.frontier <= prompt.len() && !(cancelled.borrow_mut())() {
                    if memory_active
                        && offer_memory_at(output.frontier, prompt.len(), prefill_width)
                        && let Some(cache) = caches.memory.as_deref_mut()
                        && let Err(error) = cache.offer_prompt_checkpoint(engine, &session, prompt)
                    {
                        eprintln!(
                            "prompt cache event=cache_error tier=memory action=offer: {error:#}"
                        );
                        memory_active = false;
                    }
                    if disk_active
                        && let Some(disk) = caches.disk.as_deref_mut()
                        && ((output.frontier == prompt.len() && disk.saves_final_frontiers())
                            || disk.should_write(
                                output.frontier,
                                tokens.len(),
                                mode,
                                prefill_width,
                            ))
                        && !(cancelled.borrow_mut())()
                    {
                        match disk.checkpoint(engine, &session, &mut cache_stats) {
                            Ok(()) => {}
                            Err(error) => {
                                eprintln!(
                                    "prompt cache event=cache_error tier=disk action=write: {error:#}"
                                );
                                disk_active = false;
                            }
                        }
                    }
                }
                Ok(GenerationActionResult::Consumed {
                    frontier: output.frontier,
                    changed_routes: output.changed_routes,
                    selected_id: output.selected_id,
                })
            }
            GenerationAction::SampleCurrent { history } => {
                let selected = engine.sample_committed(&session, history, &mut sampler)?;
                if let Some(phase) = phase.as_mut() {
                    phase(engine, RequestPhase::Decode)?;
                }
                if boundary(engine)? {
                    cache_stats.capacity_changed = true;
                }
                Ok(GenerationActionResult::Sampled(selected))
            }
        },
        || {
            timing.sampled(Instant::now);
            #[cfg(test)]
            {
                if let Some(observer) = sample_observer.as_mut() {
                    observer()?;
                }
                Ok(())
            }
        },
        |token| {
            emitted_ids.push(token);
            if let Some(text) = decoder(token)? {
                on_text(&text)?;
                emitted_text.push_str(&text);
            }
            Ok(())
        },
        || (cancelled.borrow_mut())(),
    );
    let cleanup = phase
        .as_mut()
        .map(|phase| phase(engine, RequestPhase::Finished))
        .transpose();
    let mut stats = match (result, cleanup) {
        (Err(primary), Err(cleanup)) => {
            return Err(
                primary.context(format!("request resource cleanup also failed: {cleanup:#}"))
            );
        }
        (Err(primary), Ok(_)) => return Err(primary),
        (Ok(_), Err(cleanup)) => return Err(cleanup),
        (Ok(stats), Ok(_)) => stats,
    };
    let generation_finished = Instant::now();
    if stats.finish != FinishReason::Cancelled {
        // The incremental decoder withholds an incomplete UTF-8 suffix. At a
        // terminal frontier, use the tokenizer's ordinary replacement policy.
        let complete = tokenizer.decode(&emitted_ids, false)?;
        let remainder = terminal_text(&complete, &emitted_text)?;
        if !remainder.is_empty() {
            on_text(remainder)?;
        }
        if memory_active
            && let Some(cache) = caches.memory.as_deref_mut()
            && let Err(error) = cache.offer_generated_checkpoint(engine, &session, prompt)
        {
            eprintln!("prompt cache event=cache_error tier=memory action=offer: {error:#}");
        }
        if disk_active
            && session.frontier()? > 0
            && let Some(disk) = caches.disk.as_deref_mut()
            && disk.saves_final_frontiers()
            && let Err(error) = disk.checkpoint(engine, &session, &mut cache_stats)
        {
            eprintln!("prompt cache event=cache_error tier=disk action=write: {error:#}");
        }
    }
    // Completed step/sample calls have already waited for their GPU work.
    // Cache publication after the last model operation does not belong to
    // decode throughput. Disk maintenance has separate reported timing.
    drop(decoder);
    drop(sampler);
    drop(session);
    drop(emitted_ids);
    drop(emitted_text);
    drop(on_text);
    timing.finish(&mut stats, generation_finished);
    stats.prompt_cache = cache_stats;
    Ok(stats)
}

fn terminal_text<'a>(complete: &'a str, emitted: &str) -> Result<&'a str> {
    complete
        .strip_prefix(emitted)
        .ok_or_else(|| anyhow::anyhow!("streamed text differs from complete token decoding"))
}

#[cfg(all(test, target_os = "macos"))]
mod phase_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Context, anyhow};
    use half::bf16;
    use std::cell::Cell;

    fn qualify_controlled_growth(
        controller: &mut crate::adaptive_memory::Controller,
        engine: &mut Engine,
        first_second: u64,
    ) -> Result<()> {
        use std::time::Duration;

        // Replay clean completed-operation samples inside a successful request.
        // Synthetic time checks policy mechanics, not real host pressure.
        controller.begin_request_at(Duration::from_secs(first_second));
        for second in first_second..=first_second + 121 {
            assert!(!controller.check_with_observation(
                engine,
                false,
                Duration::from_secs(second),
                Some(1),
                Some((0, 0)),
            )?);
        }
        controller.end_request_at(Duration::from_secs(first_second + 121), true);
        Ok(())
    }

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
    fn controlled_growth_exceeds_startup_and_preserves_prefill_and_decode_logits() -> Result<()> {
        use crate::adaptive_memory::Controller;
        use crate::memory::{ExpertCapacityControlBudget, PackageFootprint};
        use std::time::Duration;

        let package = crate::package::Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
        let footprint = PackageFootprint::from_package(&package)?;
        let per_slot = footprint.packed_expert_bytes_per_uniform_slot;
        // Exercise bounded growth without reducing context or cache allowances.
        let (selected, plan) = footprint.plan_ordinary_capacity_with_cache_budgets(
            131072,
            u64::MAX,
            21_230_125_056,
            2 << 30,
            1 << 30,
            1 << 29,
        )?;
        assert_eq!(plan.slots, 145);
        let startup = ExpertCapacityControlBudget::from_admission(&footprint, &selected, &plan)?;
        assert_eq!(
            startup.initial_serving_peak_bytes(),
            plan.accounted_peak_bytes
        );
        assert!(plan.accounted_peak_bytes <= plan.effective_budget_bytes);
        // One uniform slot adds its package-declared bytes while the admitted
        // context, workspace and cache commitments remain reserved.
        let next_slot_peak = plan
            .accounted_peak_bytes
            .checked_add(per_slot)
            .context("next startup slot peak overflow")?;
        assert!(next_slot_peak > plan.effective_budget_bytes);
        let (_, next_slot_plan) = footprint.plan_ordinary_capacity_with_cache_budgets(
            131072,
            u64::MAX,
            next_slot_peak,
            2 << 30,
            1 << 30,
            1 << 29,
        )?;
        assert_eq!(next_slot_plan.slots, plan.slots + 1);
        assert_eq!(next_slot_plan.accounted_peak_bytes, next_slot_peak);
        let budget =
            startup.with_operating_budget(&footprint, &selected, 131072, 512, 25_726_861_312)?;
        assert_eq!(budget.maximum_capacity(), 204);
        budget.validate_serving(&footprint, 131072, 512, 1 << 30, 1 << 29)?;
        let mut controller = Controller::new(budget)?;
        let mut engine = Engine::load(&package, 145, 131072)?;
        let prompt: Vec<_> = (0..512).map(|index| 7 + index % 32).collect();
        let mut original = engine.start_session()?;
        let baseline_prefill = engine.prefill_chunk(&prompt, &mut original)?.logits_bf16;
        let mut cache = PromptPrefixCache::new(&engine, 1 << 30);
        assert!(cache.offer_prompt_checkpoint(&engine, &original, &prompt)?);
        drop(original);
        let residents = engine.resident_experts()?;
        let before = engine.memory_stats()?.live_requested_buffer_bytes;
        qualify_controlled_growth(&mut controller, &mut engine, 0)?;
        assert_eq!(engine.expert_capacity(), 145);
        assert!(controller.check_with_observation(
            &mut engine,
            true,
            Duration::from_millis(121_500),
            Some(1),
            Some((0, 0)),
        )?);
        assert_eq!(engine.expert_capacity(), 160);
        assert_eq!(engine.resident_experts()?, residents);
        assert_eq!(
            engine.memory_stats()?.live_requested_buffer_bytes - before,
            15 * per_slot
        );
        assert!(cache.best_match(&engine, &prompt).is_some());
        drop(cache);

        let mut session = engine.start_session()?;
        assert_eq!(
            engine.prefill_chunk(&prompt, &mut session)?.logits_bf16,
            baseline_prefill
        );
        let mut actual = Vec::new();
        for token in [11, 12, 13, 14] {
            actual.push(
                engine
                    .step(token, RoutingMode::Unbiased, &mut session)?
                    .logits_bf16,
            );
        }
        drop(session);
        let residents = engine.resident_experts()?;
        for (index, target) in [176, 192, 204].into_iter().enumerate() {
            let first = 122 * (index as u64 + 1);
            qualify_controlled_growth(&mut controller, &mut engine, first)?;
            assert!(controller.check_with_observation(
                &mut engine,
                true,
                Duration::from_millis((first + 121) * 1000 + 500),
                Some(1),
                Some((0, 0)),
            )?);
            assert_eq!(engine.expert_capacity(), target);
            assert_eq!(engine.resident_experts()?, residents);
        }
        assert_eq!(engine.expert_capacity(), 204);
        let mut wider = engine.start_session()?;
        assert_eq!(
            engine.prefill_chunk(&prompt, &mut wider)?.logits_bf16,
            baseline_prefill
        );
        for (index, token) in [11, 12, 13, 14].into_iter().enumerate() {
            assert_eq!(
                engine
                    .step(token, RoutingMode::Unbiased, &mut wider)?
                    .logits_bf16,
                actual[index]
            );
        }
        // An unaccepted increment can fail active VM validation even while
        // system pressure remains normal. Roll it back without replacing
        // expert storage or discarding live recurrent/attention state.
        let before_rollback = engine.memory_stats()?;
        controller.begin_request_at(Duration::from_secs(500));
        let mut rolled_back = false;
        for second in 500..=520 {
            rolled_back |= controller.check_with_observation(
                &mut engine,
                false,
                Duration::from_secs(second),
                Some(1),
                Some(((second - 500) * 65536, 0)),
            )?;
        }
        assert!(rolled_back);
        assert_eq!(engine.expert_capacity(), 192);
        let after_rollback_memory = engine.memory_stats()?;
        assert_eq!(
            before_rollback.live_requested_buffer_bytes
                - after_rollback_memory.live_requested_buffer_bytes,
            12 * per_slot
        );
        assert_eq!(
            before_rollback.lifetime_peak_requested_buffer_bytes,
            after_rollback_memory.lifetime_peak_requested_buffer_bytes
        );
        let after_rollback = engine
            .step(15, RoutingMode::Unbiased, &mut wider)?
            .logits_bf16;
        controller.end_request_at(Duration::from_secs(520), true);

        // Retry requires successful work at the base and elapsed backoff.
        qualify_controlled_growth(&mut controller, &mut engine, 521)?;
        assert!(controller.check_with_observation(
            &mut engine,
            true,
            Duration::from_millis(642_500),
            Some(1),
            Some((0, 0)),
        )?);
        assert_eq!(engine.expert_capacity(), 204);
        qualify_controlled_growth(&mut controller, &mut engine, 643)?;
        controller.begin_request_at(Duration::from_secs(765));
        assert!(controller.check_with_observation(
            &mut engine,
            false,
            Duration::from_secs(765),
            Some(2),
            Some((0, 0)),
        )?);
        assert_eq!(engine.expert_capacity(), 192);
        let mut restored = false;
        for second in 766..=773 {
            restored |= controller.check_with_observation(
                &mut engine,
                false,
                Duration::from_secs(second),
                Some(1),
                Some((0, 0)),
            )?;
        }
        assert!(restored);
        assert_eq!(engine.expert_capacity(), 204);
        let after_restore = engine
            .step(16, RoutingMode::Unbiased, &mut wider)?
            .logits_bf16;
        controller.end_request_at(Duration::from_secs(773), true);
        drop(wider);
        drop(engine);

        let mut reference = Engine::load(&package, 145, 131072)?;
        let mut state = reference.start_session()?;
        assert_eq!(
            reference.prefill_chunk(&prompt, &mut state)?.logits_bf16,
            baseline_prefill
        );
        for (index, token) in [11, 12, 13, 14].into_iter().enumerate() {
            assert_eq!(
                reference
                    .step(token, RoutingMode::Unbiased, &mut state)?
                    .logits_bf16,
                actual[index]
            );
        }
        assert_eq!(
            reference
                .step(15, RoutingMode::Unbiased, &mut state)?
                .logits_bf16,
            after_rollback
        );
        assert_eq!(
            reference
                .step(16, RoutingMode::Unbiased, &mut state)?
                .logits_bf16,
            after_restore
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
    fn controlled_shrink_preserves_snapshots_and_unbiased_generation() -> Result<()> {
        use crate::adaptive_memory::Controller;
        use crate::memory::{ExpertCapacityControlBudget, PackageFootprint};
        use std::time::Duration;

        let package = crate::package::Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
        let tokenizer = TextTokenizer::load(&package)?;
        let footprint = PackageFootprint::from_package(&package)?;
        let (_, full) = footprint.plan_ordinary_capacity_with_cache_budgets(
            2048,
            u64::MAX,
            u64::MAX,
            2 << 30,
            1 << 30,
            0,
        )?;
        let (bounded, _) = footprint.plan_ordinary_capacity_with_cache_budgets(
            2048,
            u64::MAX,
            full.accounted_peak_bytes - 1,
            2 << 30,
            1 << 30,
            0,
        )?;
        let per_slot = footprint.packed_expert_bytes_per_uniform_slot;
        let nonexpert =
            bounded.accounted_component_sum_bytes - bounded.expert_stream_payload_at_ceiling_bytes;
        let (selected, plan) = footprint.plan_ordinary_capacity_with_cache_budgets(
            2048,
            u64::MAX,
            nonexpert + 105 * per_slot + (4 << 20),
            2 << 30,
            1 << 30,
            0,
        )?;
        assert_eq!(plan.slots, 105);
        let budget = ExpertCapacityControlBudget::from_admission(&footprint, &selected, &plan)?
            .with_operating_budget(
                &footprint,
                &selected,
                2048,
                512,
                plan.effective_budget_bytes,
            )?;
        assert_eq!(budget.initial_capacity(), 105);
        budget.validate_serving(&footprint, 2048, 512, 1 << 30, 0)?;
        let mut controller = Controller::new(budget)?;
        let mut engine = Engine::load(&package, 105, 2048)?;
        let mut cache = PromptPrefixCache::new(&engine, 1024 * 1024 * 1024);
        let cached_tokens = vec![6; 513];
        let mut cached_session = engine.start_session()?;
        engine.prefill_chunk(&cached_tokens[..512], &mut cached_session)?;
        assert!(cache.offer_prompt_checkpoint(&engine, &cached_session, &cached_tokens)?);
        engine.prefill_chunk(&cached_tokens[512..], &mut cached_session)?;
        assert!(cache.offer_prompt_checkpoint(&engine, &cached_session, &cached_tokens)?);
        assert!(cache.retained_bytes()? > 0);
        drop(cached_session);

        let prompt: Vec<_> = (0..1024).map(|index| 7 + index % 32).collect();
        let options = GenerationOptions {
            decode_mode: RoutingMode::Unbiased,
            ..options(2, 2048)
        };
        let mut actual = String::new();
        let mut boundaries = 0;
        let stats = generate_with_cache_controlled(
            &mut engine,
            &tokenizer,
            &prompt,
            options,
            GenerationCaches {
                memory: Some(&mut cache),
                disk: None,
            },
            |text| {
                actual.push_str(text);
                Ok(())
            },
            || false,
            |engine| {
                boundaries += 1;
                let changed = controller.check_with_observation(
                    engine,
                    false,
                    Duration::from_secs(boundaries - 1),
                    Some(if boundaries == 1 { 4 } else { 1 }),
                    Some((0, 0)),
                )?;
                assert_eq!(engine.expert_capacity(), 64);
                assert_eq!(changed, boundaries == 1);
                Ok(changed)
            },
        )?;
        assert!(boundaries >= 2);
        assert_eq!(stats.prompt_consumed, prompt.len());
        assert_eq!(engine.expert_capacity(), 64);
        assert!(cache.retained_bytes()? > 0);
        assert!(cache.best_match(&engine, &prompt).is_some());
        assert!(cache.best_match(&engine, &cached_tokens).is_some());

        assert!(!controller.check_with_observation(
            &mut engine,
            false,
            Duration::from_secs(boundaries),
            Some(4),
            Some((0, 0)),
        )?);
        let pressure_error = controller
            .check_with_observation(
                &mut engine,
                false,
                Duration::from_secs(boundaries + 1),
                Some(4),
                Some((0, 0)),
            )
            .unwrap_err();
        assert!(
            pressure_error
                .to_string()
                .contains("critical pressure persists")
        );
        assert_eq!(engine.expert_capacity(), 64);

        let first_quiet = boundaries + 2;
        qualify_controlled_growth(&mut controller, &mut engine, first_quiet)?;
        assert_eq!(engine.expert_capacity(), 64);
        let before_recovery = engine.memory_stats()?.live_requested_buffer_bytes;
        assert!(controller.check_with_observation(
            &mut engine,
            true,
            Duration::from_secs(first_quiet + 121),
            Some(1),
            Some((0, 0)),
        )?);
        assert_eq!(engine.expert_capacity(), 80);
        assert_eq!(
            engine.memory_stats()?.live_requested_buffer_bytes - before_recovery,
            16 * per_slot
        );
        assert!(!controller.check_with_observation(
            &mut engine,
            true,
            Duration::from_secs(first_quiet + 122),
            Some(1),
            Some((0, 0)),
        )?);
        assert_eq!(engine.expert_capacity(), 80);
        drop(cache);
        drop(engine);

        // Load the fixed-capacity reference after releasing the adjusted engine
        // so the check does not require two resident model instances.
        let mut reference = Engine::load(&package, 64, 2048)?;
        let mut expected = String::new();
        let reference_stats = generate(
            &mut reference,
            &tokenizer,
            &prompt,
            options,
            |text| {
                expected.push_str(text);
                Ok(())
            },
            || false,
        )?;
        assert_eq!(actual, expected);
        assert_eq!(stats.frontier, reference_stats.frontier);
        assert_eq!(stats.generated_tokens, reference_stats.generated_tokens);
        assert_eq!(stats.finish, reference_stats.finish);
        Ok(())
    }

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
    fn controlled_boundary_failure_releases_session_before_next_generation() -> Result<()> {
        let package = crate::package::Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
        let tokenizer = TextTokenizer::load(&package)?;
        let mut engine = Engine::load(&package, 105, 2048)?;
        let mut cache = PromptPrefixCache::new(&engine, 1024 * 1024 * 1024);
        let prompt: Vec<_> = (0..1024).map(|index| 7 + index % 32).collect();
        let options = GenerationOptions {
            decode_mode: RoutingMode::Unbiased,
            ..options(2, 2048)
        };
        let mut boundaries = 0;
        let error = generate_with_cache_controlled(
            &mut engine,
            &tokenizer,
            &prompt,
            options,
            GenerationCaches {
                memory: Some(&mut cache),
                disk: None,
            },
            |_| panic!("boundary failure precedes selected-token publication"),
            || false,
            |engine| {
                boundaries += 1;
                engine.shrink_expert_capacity(64)?;
                anyhow::bail!("injected boundary failure")
            },
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "injected boundary failure");
        assert_eq!(boundaries, 1);
        assert_eq!(engine.expert_capacity(), 64);
        assert_eq!(cache.retained_bytes()?, 0);
        let stats = generate(
            &mut engine,
            &tokenizer,
            &prompt,
            options,
            |_| Ok(()),
            || false,
        )?;
        assert_eq!(stats.prompt_consumed, prompt.len());
        Ok(())
    }

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
    fn complete_cache_hit_checks_boundary_before_single_token_publication() -> Result<()> {
        let package = crate::package::Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
        let tokenizer = TextTokenizer::load(&package)?;
        let mut engine = Engine::load(&package, 105, 2048)?;
        let mut cache = PromptPrefixCache::new(&engine, 1024 * 1024 * 1024);
        let prompt: Vec<_> = (0..512).map(|index| 7 + index % 32).collect();
        let mut session = engine.start_session()?;
        engine.prefill_chunk(&prompt, &mut session)?;
        assert!(cache.offer_prompt_checkpoint(&engine, &session, &prompt)?);
        drop(session);
        assert!(cache.best_match(&engine, &prompt).unwrap().is_complete());
        let options = GenerationOptions {
            decode_mode: RoutingMode::Unbiased,
            ..options(1, 2048)
        };
        let mut expected = String::new();
        let reference_stats = generate_with_cache(
            &mut engine,
            &tokenizer,
            &prompt,
            options,
            GenerationCaches {
                memory: Some(&mut cache),
                disk: None,
            },
            |text| {
                expected.push_str(text);
                Ok(())
            },
            || false,
        )?;
        assert!(cache.best_match(&engine, &prompt).unwrap().is_complete());
        let mut actual = String::new();
        let mut boundaries = 0;
        let stats = generate_with_cache_controlled(
            &mut engine,
            &tokenizer,
            &prompt,
            options,
            GenerationCaches {
                memory: Some(&mut cache),
                disk: None,
            },
            |text| {
                actual.push_str(text);
                Ok(())
            },
            || false,
            |engine| {
                boundaries += 1;
                engine.shrink_expert_capacity(64)?;
                Ok(true)
            },
        )?;
        assert_eq!(boundaries, 1);
        assert_eq!(engine.expert_capacity(), 64);
        assert_eq!(actual, expected);
        assert!(cache.best_match(&engine, &prompt).is_some());
        assert_eq!(stats.prompt_consumed, prompt.len());
        assert_eq!(stats.frontier, reference_stats.frontier);
        assert_eq!(stats.generated_tokens, reference_stats.generated_tokens);
        assert_eq!(stats.finish, reference_stats.finish);
        Ok(())
    }

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
    fn wide_prefill_cancellation_preserves_prior_conversation() -> Result<()> {
        for rows in [2048, 4096] {
            wide_prefill_cancellation(rows)?;
        }
        Ok(())
    }

    fn wide_prefill_cancellation(rows: usize) -> Result<()> {
        let package = crate::package::Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
        let tokenizer = TextTokenizer::load(&package)?;
        let context = rows + 2048;
        let mut engine = Engine::load_with_prefill_capacity(&package, 128, context, rows)?;
        let mut cache = PromptPrefixCache::new(&engine, 1 << 30);
        let prior: Vec<_> = (0..32).map(|i| 7 + i % 16).collect();
        let mut session = engine.start_session()?;
        engine.prefill_chunk(&prior, &mut session)?;
        assert!(cache.offer_prompt_checkpoint(&engine, &session, &prior)?);
        drop(session);
        let retained = cache.retained_bytes()?;
        let prompt: Vec<_> = (0..rows).map(|i| 61 + (i % 32) as u32).collect();
        let cancelled = Cell::new(false);
        let mut boundaries = 0;
        let stats = generate_with_cache_controlled(
            &mut engine,
            &tokenizer,
            &prompt,
            options(1, context),
            GenerationCaches {
                memory: Some(&mut cache),
                disk: None,
            },
            |_| panic!("cancelled prefill must not emit"),
            || cancelled.get(),
            |_| {
                boundaries += 1;
                cancelled.set(true);
                Ok(false)
            },
        )?;
        assert_eq!(stats.finish, FinishReason::Cancelled);
        assert_eq!(boundaries, 1);
        assert_eq!(
            stats.prompt_consumed, 0,
            "cancellation must interrupt the wide chunk before it commits"
        );
        assert_eq!(stats.prefill_tokens, 0);
        assert_eq!(stats.generated_tokens, 0);
        assert_eq!(cache.retained_bytes()?, retained);
        assert!(cache.best_match(&engine, &prior).unwrap().is_complete());
        assert!(cache.best_match(&engine, &prompt).is_none());
        let mut followup = prior.clone();
        followup.push(29);
        let resumed = generate_with_cache(
            &mut engine,
            &tokenizer,
            &followup,
            options(1, context),
            GenerationCaches {
                memory: Some(&mut cache),
                disk: None,
            },
            |_| Ok(()),
            || false,
        )?;
        assert_eq!(resumed.prompt_cache.event, "memory_hit");
        assert_eq!(resumed.prompt_cache.cached_tokens, prior.len());
        assert_eq!(resumed.prefill_tokens, 1);
        assert_eq!(resumed.prompt_consumed, followup.len());
        Ok(())
    }

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
    fn wide_prefill_pressure_release_completes_and_reuses_conversation() -> Result<()> {
        for rows in [2048, 4096] {
            wide_prefill_pressure_release(rows)?;
        }
        Ok(())
    }

    fn wide_prefill_pressure_release(rows: usize) -> Result<()> {
        let package = crate::package::Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
        let expected_release = 16
            * crate::memory::PackageFootprint::from_package(&package)?
                .packed_expert_bytes_per_uniform_slot;
        let tokenizer = TextTokenizer::load(&package)?;
        let context = rows + 2048;
        let mut engine = Engine::load_with_prefill_capacity(&package, 128, context, rows)?;
        let mut cache = PromptPrefixCache::new(&engine, 1 << 30);
        let prompt: Vec<_> = (0..rows).map(|i| 7 + (i % 32) as u32).collect();
        let mut boundaries = 0;
        let stats = generate_with_cache_controlled(
            &mut engine,
            &tokenizer,
            &prompt,
            options(1, context),
            GenerationCaches {
                memory: Some(&mut cache),
                disk: None,
            },
            |_| Ok(()),
            || false,
            |engine| {
                boundaries += 1;
                if boundaries == 1 {
                    let release = engine.shrink_expert_capacity(112)?;
                    assert_eq!(release.released_requested_bytes, expected_release);
                    return Ok(true);
                }
                Ok(false)
            },
        )?;
        assert!(
            boundaries >= 2,
            "a wide chunk must offer control before its final commit"
        );
        assert_eq!(engine.expert_capacity(), 112);
        assert_eq!(stats.prompt_consumed, prompt.len());
        assert!(stats.prompt_cache.capacity_changed);
        assert!(cache.best_match(&engine, &prompt).unwrap().is_complete());
        let mut followup = prompt.clone();
        followup.extend([39, 40, 41]);
        let resumed = generate_with_cache(
            &mut engine,
            &tokenizer,
            &followup,
            options(1, context),
            GenerationCaches {
                memory: Some(&mut cache),
                disk: None,
            },
            |_| Ok(()),
            || false,
        )?;
        assert_eq!(resumed.prompt_cache.event, "memory_hit");
        assert_eq!(resumed.prompt_cache.cached_tokens, prompt.len());
        assert_eq!(resumed.prefill_tokens, 3);
        assert_eq!(resumed.prompt_consumed, followup.len());
        Ok(())
    }

    #[test]
    #[cfg(target_os = "macos")]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
    fn default_gdn_inputs_preserve_resized_generated_conversation() -> Result<()> {
        use crate::tokenizer::ChatOptions;
        use serde_json::json;

        let package = crate::package::Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
        let tokenizer = TextTokenizer::load(&package)?;
        let mut engine = Engine::load_with_prefill_capacity(&package, 96, 4096, 512)?;
        let mut cache = PromptPrefixCache::new(&engine, 1 << 30);
        let chat = ChatOptions {
            thinking: false,
            ..ChatOptions::default()
        };
        let mut turns = vec![json!({"role":"user","content":
            "Explain transactional database publication and crash recovery in detail."})];
        let mut previous_prompt_len = 0;
        for turn in 0..2 {
            let prompt = tokenizer.render(&turns, None, &chat)?.tokens;
            let mut reply = String::new();
            let stats = generate_with_cache(
                &mut engine,
                &tokenizer,
                &prompt,
                options(32, 4096),
                GenerationCaches {
                    memory: Some(&mut cache),
                    disk: None,
                },
                |text| {
                    reply.push_str(text);
                    Ok(())
                },
                || false,
            )?;
            ensure!(stats.generated_tokens > 1);
            if turn == 0 {
                ensure!(stats.prompt_cache.cached_tokens == 0);
                turns.push(json!({"role":"assistant","content":reply}));
                turns.push(json!({"role":"user","content":"Explain why in detail."}));
                previous_prompt_len = prompt.len();
                ensure!(
                    engine
                        .recover_expert_capacity(112, 2 << 30)?
                        .added_requested_bytes
                        > 0
                );
                ensure!(engine.shrink_expert_capacity(80)?.released_requested_bytes > 0);
            } else {
                ensure!(
                    stats.prompt_cache.cached_tokens > previous_prompt_len,
                    "resized GDN conversation lost its generated prefix"
                );
                ensure!(stats.prefill_tokens < 64);
            }
        }
        Ok(())
    }

    #[test]
    fn first_eos_is_timed_before_text_and_total_uses_the_same_origin() {
        use std::time::Duration;
        let ready = Instant::now();
        let mut timing = GenerationTimer::new(ready, ready + Duration::from_secs(2));
        let mut stats = token_loop(
            &[1],
            &[7],
            options(2, 8),
            512,
            None,
            |_, _| {
                Ok(TokenOutput {
                    logits_bf16: logits(7),
                    changed_routes: 0,
                    frontier: 1,
                })
            },
            |_, _| {
                timing.sampled(|| ready + Duration::from_secs(5));
                Ok(7)
            },
            |_| panic!("EOS must be timed without text emission"),
            || false,
        )
        .unwrap();
        assert_eq!(stats.generated_tokens, 0);
        assert_eq!(stats.finish, FinishReason::StopToken(7));
        // A later sample or cleanup cannot replace the first materialization.
        timing.sampled(|| panic!("clock read after first token"));
        timing.finish(&mut stats, ready + Duration::from_secs(9));
        assert_eq!(stats.first_token_seconds, Some(3.0));
        assert_eq!(stats.ready_to_first_token_seconds, Some(5.0));
        assert_eq!(stats.generation_seconds, Some(7.0));
    }

    #[test]
    fn cancellation_after_selected_graph_does_not_deliver_or_emit() -> Result<()> {
        let prefill_completed = Cell::new(false);
        let delivered = Cell::new(0);
        let emitted = Cell::new(0);
        let ready = Instant::now();
        let mut timing = GenerationTimer::new(ready, ready);
        let prefill = run_loop(
            &[1, 2],
            &[7],
            options(3, 8),
            512,
            0,
            |action| match action {
                GenerationAction::Consume { sample, .. } => {
                    assert!(sample);
                    prefill_completed.set(true);
                    Ok(GenerationActionResult::Consumed {
                        frontier: 2,
                        changed_routes: 0,
                        selected_id: Some(4),
                    })
                }
                GenerationAction::SampleCurrent { .. } => {
                    panic!("completed prefill selected an ID")
                }
            },
            || {
                delivered.set(delivered.get() + 1);
                timing.sampled(|| ready);
                Ok(())
            },
            |_| {
                emitted.set(emitted.get() + 1);
                Ok(())
            },
            || prefill_completed.get(),
        )?;
        assert_eq!(prefill.frontier, 2);
        assert_eq!(prefill.prompt_consumed, 2);
        assert_eq!(prefill.generated_tokens, 0);
        assert_eq!(prefill.finish, FinishReason::Cancelled);
        assert_eq!(delivered.get(), 0);
        assert_eq!(emitted.get(), 0);
        assert_eq!(timing.first_at, None);

        let consumes = Cell::new(0);
        let delivered = Cell::new(0);
        let emitted = Cell::new(0);
        let mut timing = GenerationTimer::new(ready, ready);
        let decode = run_loop(
            &[1],
            &[7],
            options(3, 8),
            512,
            0,
            |action| match action {
                GenerationAction::Consume { sample, .. } => {
                    assert!(sample);
                    let count = consumes.get() + 1;
                    consumes.set(count);
                    Ok(GenerationActionResult::Consumed {
                        frontier: count,
                        changed_routes: 0,
                        selected_id: Some(if count == 1 { 4 } else { 5 }),
                    })
                }
                GenerationAction::SampleCurrent { .. } => panic!("decode already selected an ID"),
            },
            || {
                delivered.set(delivered.get() + 1);
                timing.sampled(|| ready);
                Ok(())
            },
            |_| {
                emitted.set(emitted.get() + 1);
                Ok(())
            },
            || consumes.get() == 2,
        )?;
        assert_eq!(decode.frontier, 2);
        assert_eq!(decode.generated_tokens, 1);
        assert_eq!(decode.finish, FinishReason::Cancelled);
        assert_eq!(delivered.get(), 1);
        assert_eq!(emitted.get(), 1);
        assert_eq!(timing.first_at, Some(ready));
        Ok(())
    }

    #[test]
    fn cancellation_without_a_sample_has_no_first_token_time() {
        use std::time::Duration;
        let ready = Instant::now();
        let timing = GenerationTimer::new(ready, ready);
        let mut stats = token_loop(
            &[1],
            &[7],
            options(1, 8),
            512,
            None,
            |_, _| panic!("cancelled generation must not step"),
            |_, _| panic!("cancelled generation must not sample"),
            |_| panic!("cancelled generation must not emit"),
            || true,
        )
        .unwrap();
        assert_eq!(stats.first_token_seconds, None);
        assert_eq!(stats.ready_to_first_token_seconds, None);
        assert_eq!(stats.generation_seconds, None);
        timing.finish(&mut stats, ready + Duration::from_secs(1));
        assert_eq!(stats.first_token_seconds, None);
        assert_eq!(stats.ready_to_first_token_seconds, None);
        assert_eq!(stats.generation_seconds, Some(1.0));
    }

    #[test]
    fn terminal_text_preserves_complete_prefix_and_flushes_incomplete_unicode() {
        assert_eq!(terminal_text("hello é", "hello é").unwrap(), "");
        assert_eq!(terminal_text("hello �", "hello ").unwrap(), "�");
        assert_eq!(terminal_text("�", "").unwrap(), "�");
        assert!(terminal_text("changed", "already sent").is_err());
    }

    #[test]
    fn identical_partial_prompt_hit_samples_saved_logits_without_refeeding() {
        let prompt = vec![1; 513];
        let saved = logits(4);
        let mut emitted = Vec::new();
        let stats = token_loop(
            &prompt,
            &[7],
            options(1, 1024),
            512,
            Some(PromptStart {
                frontier: 513,
                logits: saved.clone(),
            }),
            |_, _| panic!("complete cache hit must not consume prompt or last output"),
            |actual, history| {
                assert_eq!(actual, saved);
                assert_eq!(history, prompt);
                Ok(4)
            },
            |token| {
                emitted.push(token);
                Ok(())
            },
            || false,
        )
        .unwrap();
        assert_eq!(emitted, [4]);
        assert_eq!(stats.prompt_consumed, 513);
        assert_eq!(stats.prefill_tokens, 0);
        assert_eq!(stats.prefill_seconds, Some(0.0));
        assert_eq!(stats.frontier, 513);
        assert_eq!(stats.generated_tokens, 1);
        assert_eq!(stats.changed_routes, 0);
        assert_eq!(stats.finish, FinishReason::MaxTokens);
    }

    #[test]
    fn aligned_prefix_hit_consumes_only_canonical_suffix_then_decodes() {
        let prompt = vec![1; 515];
        let mut calls = Vec::new();
        let mut frontier = 512;
        let stats = token_loop(
            &prompt,
            &[7],
            options(3, 1024),
            512,
            Some(PromptStart {
                frontier,
                logits: logits(2),
            }),
            |tokens, mode| {
                calls.push((tokens.to_vec(), mode));
                frontier += tokens.len();
                Ok(TokenOutput {
                    logits_bf16: logits(if frontier == 515 { 4 } else { 7 }),
                    changed_routes: 0,
                    frontier,
                })
            },
            mock_winner,
            |token| {
                assert_eq!(token, 4);
                Ok(())
            },
            || false,
        )
        .unwrap();
        assert_eq!(
            calls,
            [
                (vec![1; 3], RoutingMode::Unbiased),
                (vec![4], RoutingMode::CachePrior),
            ]
        );
        assert_eq!(stats.prompt_consumed, 515);
        assert_eq!(stats.prefill_tokens, 3);
        assert!(stats.prefill_seconds.is_some_and(|seconds| seconds >= 0.0));
        assert_eq!(stats.frontier, 516);
        assert_eq!(stats.generated_tokens, 1);
        assert_eq!(stats.finish, FinishReason::StopToken(7));
    }

    #[test]
    fn restored_prompt_rejects_incompatible_frontiers_before_execution() {
        for (frontier, saved) in [(515, logits(4)), (512, Vec::new()), (0, logits(4))] {
            assert!(
                token_loop(
                    &vec![1; 514],
                    &[7],
                    options(1, 1024),
                    512,
                    Some(PromptStart {
                        frontier,
                        logits: saved
                    }),
                    |_, _| panic!("invalid prefix cannot execute"),
                    |_, _| panic!("invalid prefix cannot sample"),
                    |_| panic!("invalid prefix cannot emit"),
                    || false,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn cancelled_complete_hit_keeps_restored_frontier_without_sampling() {
        let stats = token_loop(
            &[1, 2],
            &[7],
            options(2, 8),
            512,
            Some(PromptStart {
                frontier: 2,
                logits: logits(4),
            }),
            |_, _| panic!("cancelled hit cannot execute"),
            |_, _| panic!("cancelled hit cannot sample"),
            |_| panic!("cancelled hit cannot emit"),
            || true,
        )
        .unwrap();
        assert_eq!(stats.prompt_consumed, 2);
        assert_eq!(stats.frontier, 2);
        assert_eq!(stats.generated_tokens, 0);
        assert_eq!(stats.finish, FinishReason::Cancelled);
    }

    #[test]
    fn chunked_prefill_advances_by_rows_and_keeps_decode_single_token() {
        let prompt = vec![1; 515];
        let mut calls = Vec::new();
        let mut frontier = 0;
        let mut samples = 0;
        let stats = token_loop(
            &prompt,
            &[7],
            options(3, 1024),
            512,
            None,
            |tokens, mode| {
                calls.push((tokens.len(), mode));
                frontier += tokens.len();
                Ok(TokenOutput {
                    logits_bf16: logits(4),
                    changed_routes: 0,
                    frontier,
                })
            },
            |_, _| {
                samples += 1;
                Ok(if samples == 1 { 4 } else { 7 })
            },
            |_| Ok(()),
            || false,
        )
        .unwrap();
        assert_eq!(
            calls,
            [
                (512, RoutingMode::Unbiased),
                (3, RoutingMode::Unbiased),
                (1, RoutingMode::CachePrior),
            ]
        );
        assert_eq!(stats.prompt_consumed, 515);
        assert_eq!(stats.frontier, 516);
        assert_eq!(stats.generated_tokens, 1);
        assert_eq!(stats.finish, FinishReason::StopToken(7));
    }

    #[test]
    fn cancellation_between_chunks_keeps_only_completed_rows() {
        let polls = Cell::new(0);
        let stats = token_loop(
            &vec![1; 515],
            &[7],
            options(1, 1024),
            512,
            None,
            |tokens, mode| {
                assert_eq!(tokens.len(), 512);
                assert_eq!(mode, RoutingMode::Unbiased);
                Ok(TokenOutput {
                    logits_bf16: logits(4),
                    changed_routes: 0,
                    frontier: 512,
                })
            },
            |_, _| panic!("cancelled prefill cannot sample"),
            |_| panic!("cancelled prefill cannot emit"),
            || {
                polls.set(polls.get() + 1);
                polls.get() == 2
            },
        )
        .unwrap();
        assert_eq!(stats.prompt_consumed, 512);
        assert_eq!(stats.frontier, 512);
        assert_eq!(stats.generated_tokens, 0);
        assert_eq!(stats.finish, FinishReason::Cancelled);
    }

    fn logits(winner: usize) -> Vec<u16> {
        let mut values = vec![bf16::from_f32(-1.0).to_bits(); 8];
        values[winner] = bf16::from_f32(2.0).to_bits();
        values
    }

    fn options(max_tokens: usize, max_context: usize) -> GenerationOptions {
        GenerationOptions {
            max_tokens,
            max_context,
            decode_mode: RoutingMode::CachePrior,
            sampling: SamplingOptions {
                temperature: 0.0,
                ..SamplingOptions::default()
            },
            seed: 0,
        }
    }

    fn mock_winner(logits: &[u16], _: &[u32]) -> Result<u32> {
        ensure!(!logits.is_empty(), "empty mock logits");
        let mut selected = 0;
        for (id, &value) in logits.iter().enumerate() {
            if bf16::from_bits(value).to_f32() > bf16::from_bits(logits[selected]).to_f32() {
                selected = id;
            }
        }
        Ok(selected as u32)
    }

    #[test]
    fn prefill_including_one_token_is_unbiased_and_decode_consumes_emitted_token() {
        let mut calls = Vec::new();
        let mut emitted = Vec::new();
        let stats = token_loop(
            &[5],
            &[7],
            options(3, 8),
            1,
            None,
            |tokens, mode| {
                let token = tokens[0];
                calls.push((token, mode));
                Ok(TokenOutput {
                    logits_bf16: logits(if token == 5 { 2 } else { 7 }),
                    changed_routes: 1,
                    frontier: calls.len(),
                })
            },
            mock_winner,
            |token| {
                emitted.push(token);
                Ok(())
            },
            || false,
        )
        .unwrap();
        assert_eq!(
            calls,
            [(5, RoutingMode::Unbiased), (2, RoutingMode::CachePrior)]
        );
        assert_eq!(emitted, [2]);
        assert_eq!(stats.prompt_consumed, 1);
        assert_eq!(stats.frontier, 2);
        assert_eq!(stats.generated_tokens, 1);
        assert_eq!(stats.changed_routes, 2);
        assert_eq!(stats.finish, FinishReason::StopToken(7));
    }

    #[test]
    fn last_prefill_logit_wins_and_context_limit_does_not_consume_last_output() {
        let mut calls = Vec::new();
        let mut emitted = Vec::new();
        let stats = token_loop(
            &[1, 2],
            &[7],
            options(4, 2),
            1,
            None,
            |tokens, mode| {
                let token = tokens[0];
                calls.push((token, mode));
                Ok(TokenOutput {
                    logits_bf16: logits(if token == 1 { 3 } else { 4 }),
                    changed_routes: 0,
                    frontier: calls.len(),
                })
            },
            mock_winner,
            |token| {
                emitted.push(token);
                Ok(())
            },
            || false,
        )
        .unwrap();
        assert_eq!(
            calls,
            [(1, RoutingMode::Unbiased), (2, RoutingMode::Unbiased)]
        );
        assert_eq!(emitted, [4]);
        assert_eq!(stats.frontier, 2);
        assert_eq!(stats.finish, FinishReason::ContextLimit);
    }

    #[test]
    fn cancellation_and_callback_failure_do_not_advance_uncommitted_token() {
        let polls = Cell::new(0);
        let mut calls = 0;
        let stats = token_loop(
            &[1],
            &[7],
            options(3, 4),
            1,
            None,
            |_, _| {
                calls += 1;
                Ok(TokenOutput {
                    logits_bf16: logits(2),
                    changed_routes: 0,
                    frontier: calls,
                })
            },
            mock_winner,
            |_| Ok(()),
            || {
                polls.set(polls.get() + 1);
                polls.get() == 3
            },
        )
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(stats.generated_tokens, 1);
        assert_eq!(stats.frontier, 1);
        assert_eq!(stats.finish, FinishReason::Cancelled);

        let mut calls = 0;
        assert!(
            token_loop(
                &[1],
                &[7],
                options(3, 4),
                1,
                None,
                |_, _| {
                    calls += 1;
                    Ok(TokenOutput {
                        logits_bf16: logits(2),
                        changed_routes: 0,
                        frontier: calls,
                    })
                },
                mock_winner,
                |_| Err(anyhow!("output closed")),
                || false,
            )
            .is_err()
        );
        assert_eq!(calls, 1);
    }

    #[test]
    fn cancellation_during_prefill_keeps_only_completed_prompt_frontier() {
        let polls = Cell::new(0);
        let mut calls = 0;
        let stats = token_loop(
            &[1, 2, 3],
            &[7],
            options(2, 8),
            1,
            None,
            |_, mode| {
                assert_eq!(mode, RoutingMode::Unbiased);
                calls += 1;
                Ok(TokenOutput {
                    logits_bf16: logits(4),
                    changed_routes: 0,
                    frontier: calls,
                })
            },
            mock_winner,
            |_| panic!("no token may be emitted before prefill completes"),
            || {
                polls.set(polls.get() + 1);
                polls.get() == 2
            },
        )
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(stats.prompt_consumed, 1);
        assert_eq!(stats.generated_tokens, 0);
        assert_eq!(stats.frontier, 1);
        assert_eq!(stats.finish, FinishReason::Cancelled);
    }

    #[test]
    fn max_tokens_stops_without_consuming_the_last_emitted_token() {
        let mut calls = 0;
        let mut emitted = Vec::new();
        let stats = token_loop(
            &[1],
            &[7],
            options(1, 8),
            1,
            None,
            |_, _| {
                calls += 1;
                Ok(TokenOutput {
                    logits_bf16: logits(4),
                    changed_routes: 0,
                    frontier: calls,
                })
            },
            mock_winner,
            |token| {
                emitted.push(token);
                Ok(())
            },
            || false,
        )
        .unwrap();
        assert_eq!(calls, 1);
        assert_eq!(emitted, [4]);
        assert_eq!(stats.generated_tokens, 1);
        assert_eq!(stats.frontier, 1);
        assert_eq!(stats.finish, FinishReason::MaxTokens);
    }

    #[test]
    fn sampling_history_includes_prompt_and_only_emitted_tokens() {
        let mut seen = Vec::new();
        let mut steps = 0;
        let stats = token_loop(
            &[10, 11],
            &[7],
            options(3, 8),
            1,
            None,
            |_, mode| {
                assert_eq!(
                    mode,
                    if steps < 2 {
                        RoutingMode::Unbiased
                    } else {
                        RoutingMode::CachePrior
                    }
                );
                steps += 1;
                Ok(TokenOutput {
                    logits_bf16: logits(steps.min(7)),
                    changed_routes: 0,
                    frontier: steps,
                })
            },
            |_, history| {
                seen.push(history.to_vec());
                Ok(if seen.len() == 1 { 4 } else { 7 })
            },
            |_| Ok(()),
            || false,
        )
        .unwrap();
        assert_eq!(seen, [vec![10, 11], vec![10, 11, 4]]);
        assert_eq!(steps, 3);
        assert_eq!(stats.generated_tokens, 1);
        assert_eq!(stats.frontier, 3);
        assert_eq!(stats.finish, FinishReason::StopToken(7));
    }

    #[test]
    fn request_validation_precedes_cache_and_session_work() {
        let valid = options(1, 1024);
        assert!(validate_request_inputs(&[7; 512], &[8], valid, 1024, 512).is_ok());
        assert!(validate_request_inputs(&[], &[8], valid, 1024, 512).is_err());
        assert!(validate_request_inputs(&[7], &[], valid, 1024, 512).is_err());
        assert!(validate_request_inputs(&[248_320], &[8], valid, 1024, 512).is_err());
        assert!(validate_request_inputs(&[7; 1025], &[8], valid, 1024, 512).is_err());
        assert!(validate_request_inputs(&[7], &[8], valid, 512, 512).is_err());
        assert!(validate_request_inputs(&[7], &[8], valid, 1024, 0).is_err());
        assert!(validate_request_inputs(&[7], &[8], options(0, 1024), 1024, 512).is_err());
    }

    #[test]
    fn memory_capture_targets_only_deepest_aligned_and_terminal() {
        assert!(!offer_memory_at(512, 1537, 512));
        assert!(!offer_memory_at(1024, 1537, 512));
        assert!(offer_memory_at(1536, 1537, 512));
        assert!(offer_memory_at(1537, 1537, 512));
        assert!(offer_memory_at(512, 512, 512));
        assert!(offer_memory_at(1, 1, 512));
    }

    #[test]
    fn disk_write_policy_saves_completed_chunks_crossing_stride_boundaries() {
        assert!(GenerationDiskCache::validate_policy(1024, 16_384).is_ok());
        assert!(GenerationDiskCache::validate_policy(1024, 1).is_ok());
        for (stride, depth) in [(0, 1024), (511, 1024), (513, 1024), (1024, 0)] {
            assert!(GenerationDiskCache::validate_policy(stride, depth).is_err());
        }
        let eligible =
            |frontier, tokens, mode| disk_write_eligible(frontier, tokens, mode, 1024, 16_384, 512);
        assert!(!eligible(512, 512, RoutingMode::Unbiased));
        assert!(eligible(1024, 512, RoutingMode::Unbiased));
        assert!(!eligible(1536, 512, RoutingMode::Unbiased));
        // A restored 1024-token prefix may produce another eligible frontier.
        assert!(eligible(2048, 512, RoutingMode::Unbiased));
        assert!(eligible(16_384, 512, RoutingMode::Unbiased));
        assert!(!eligible(17_408, 512, RoutingMode::Unbiased));
        assert!(!eligible(1024, 511, RoutingMode::Unbiased));
        assert!(!eligible(1024, 1, RoutingMode::Unbiased));
        assert!(!eligible(1024, 512, RoutingMode::CachePrior));
        assert!(!disk_write_eligible(
            1024,
            512,
            RoutingMode::Unbiased,
            1024,
            1,
            512
        ));
    }

    #[test]
    fn resumed_prefill_keeps_periodic_checkpoints_from_a_partial_frontier() {
        // A short restored conversation resumes at 29, not a multiple of the
        // workspace width. Completed chunks still cross the configured stride.
        assert!(!disk_write_eligible(
            541,
            512,
            RoutingMode::Unbiased,
            1024,
            16384,
            512
        ));
        assert!(disk_write_eligible(
            1053,
            512,
            RoutingMode::Unbiased,
            1024,
            16384,
            512
        ));
        assert!(!disk_write_eligible(
            1565,
            512,
            RoutingMode::Unbiased,
            1024,
            16384,
            512
        ));
        assert!(disk_write_eligible(
            2077,
            512,
            RoutingMode::Unbiased,
            1024,
            16384,
            512
        ));
    }

    #[test]
    fn wide_prefill_checkpoint_frontiers_do_not_split_for_stride() {
        assert!(disk_write_eligible(
            2048,
            2048,
            RoutingMode::Unbiased,
            1024,
            16384,
            2048
        ));
        assert!(!disk_write_eligible(
            1024,
            1024,
            RoutingMode::Unbiased,
            1024,
            16384,
            2048
        ));
        assert!(disk_write_eligible(
            4096,
            2048,
            RoutingMode::Unbiased,
            1536,
            16384,
            2048
        ));
        assert!(disk_write_eligible(
            6144,
            2048,
            RoutingMode::Unbiased,
            1536,
            16384,
            2048
        ));
        assert!(!disk_write_eligible(
            2048,
            2048,
            RoutingMode::Unbiased,
            1024,
            1024,
            2048
        ));
        assert!(offer_memory_at(4096, 4097, 2048));
        assert!(offer_memory_at(4097, 4097, 2048));
        assert!(!offer_memory_at(2048, 4097, 2048));
    }

    #[test]
    fn restored_prefix_can_offer_a_later_disk_frontier_without_refeeding() {
        let prompt = vec![1; 2051];
        let mut frontier = 1024;
        let mut writes = Vec::new();
        let stats = token_loop(
            &prompt,
            &[7],
            options(1, 4096),
            512,
            Some(PromptStart {
                frontier,
                logits: logits(4),
            }),
            |tokens, mode| {
                frontier += tokens.len();
                if disk_write_eligible(frontier, tokens.len(), mode, 1024, 16_384, 512) {
                    writes.push(frontier);
                }
                Ok(TokenOutput {
                    logits_bf16: logits(7),
                    changed_routes: 0,
                    frontier,
                })
            },
            mock_winner,
            |_| panic!("stop token must not emit"),
            || false,
        )
        .unwrap();
        assert_eq!(writes, [2048]);
        assert_eq!(stats.prompt_consumed, 2051);
        assert_eq!(stats.frontier, 2051);
        assert_eq!(stats.finish, FinishReason::StopToken(7));
    }
}
