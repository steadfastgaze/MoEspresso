//! Real HTTP conversations with deterministic pressure and time, real cache
//! snapshots and the production request-admission path.

use super::*;
use crate::adaptive_memory::{Boundary, Controller};
use crate::memory::{ExpertCapacityControlBudget, PackageFootprint};
use std::os::unix::fs::DirBuilderExt;
use std::thread;

struct CacheRoot(PathBuf);

impl Drop for CacheRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn request(server: &Server, state: &mut State, messages: Value, stream: bool) -> Result<Value> {
    let url = format!("http://{}/v1/chat/completions", server.server_addr());
    let mut body = json!({
        "model": state.package.artifact_id, "messages": messages,
        "max_tokens": 8, "temperature": 0, "stream": stream,
    });
    if stream {
        body["stream_options"] = json!({"include_usage": true});
    }
    let client = thread::spawn(move || -> Result<Value> {
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(180)))
            .max_redirects(0)
            .proxy(None)
            .build()
            .new_agent();
        let mut response = agent
            .post(&url)
            .header("Content-Type", "application/json")
            .send(serde_json::to_vec(&body)?)?;
        let text = response.body_mut().read_to_string()?;
        if !stream {
            return Ok(serde_json::from_str(&text)?);
        }
        let mut usage = Value::Null;
        let mut content = String::new();
        let mut terminal = false;
        for line in text.lines().filter_map(|line| line.strip_prefix("data: ")) {
            if line == "[DONE]" {
                terminal = true;
                continue;
            }
            let event: Value = serde_json::from_str(line)?;
            ensure!(event["error"].is_null(), "SSE error: {event}");
            if event["usage"].is_object() {
                usage = event["usage"].clone();
            }
            if let Some(delta) = event["choices"][0]["delta"]["content"].as_str() {
                content.push_str(delta);
            }
        }
        ensure!(terminal && usage.is_object(), "stream lacks terminal usage");
        Ok(
            json!({"usage": usage, "choices": [{"message": {"role": "assistant", "content": content}}]}),
        )
    });
    let incoming = server
        .recv_timeout(Duration::from_secs(5))?
        .ok_or_else(|| anyhow!("no HTTP request"))?;
    state.handle_request(incoming)?;
    if let Some(cache) = &state.memory_cache {
        assert!(
            cache.retained_bytes()? <= state.options.cache.as_ref().unwrap().memory_retained_bytes
        );
    }
    client.join().map_err(|_| anyhow!("HTTP client panicked"))?
}

fn qualify_growth(state: &mut State, first: u64) -> Result<()> {
    let controller = state.adaptive.as_mut().unwrap();
    controller.begin_request_at(Duration::from_secs(first));
    assert!(!controller.check_with_observation(
        &mut state.engine,
        false,
        Duration::from_secs(first),
        Some(1),
        Some((0, 0)),
    )?);
    // Defer the growth transaction until the real HTTP follow-up below.
    assert!(controller.try_begin_workspace_phase(&state.engine, state.engine.expert_capacity())?);
    for second in first + 1..=first + 20 {
        assert!(!controller.check_with_observation(
            &mut state.engine,
            false,
            Duration::from_secs(second),
            Some(1),
            Some((0, 0)),
        )?);
    }
    assert!(!controller.finish_workspace_phase(&mut state.engine)?);
    controller.end_request_at(Duration::from_secs(first + 20), true);
    controller.fix_observation(Duration::from_millis((first + 20) * 1000 + 100), 1);
    Ok(())
}

fn fixture(write_depth: usize, mode: DiskCacheMode) -> Result<(CacheRoot, State, Server)> {
    let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
    let footprint = PackageFootprint::from_package(&package)?;
    let (selected, plan) = footprint.plan_ordinary_capacity_with_cache_budgets(
        131072,
        u64::MAX,
        21_230_125_056,
        2 << 30,
        1 << 30,
        1 << 29,
    )?;
    let budget = ExpertCapacityControlBudget::from_admission(&footprint, &selected, &plan)?
        .with_operating_budget(&footprint, &selected, 131072, 512, 25_726_861_312)?;
    ensure!(
        plan.slots > 64 && plan.slots < budget.maximum_capacity(),
        "test requires room to shrink and grow"
    );
    let root = CacheRoot(std::env::temp_dir().join(format!(
        "moespresso-conversation-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos(),
    )));
    let options = ServeOptions {
        port: 0,
        expert_capacity: plan.slots,
        max_context: 131072,
        thinking: false,
        decode_mode: RoutingMode::CachePrior,
        cache: Some(ServeCacheOptions {
            memory_retained_bytes: 1 << 30,
            host_overlap_bytes: 3 << 29,
            disk: Some(ServeDiskOptions {
                mode,
                root: root.0.clone(),
                managed_cache_base: None,
                max_file_bytes: 1 << 30,
                max_io_buffer_bytes: crate::engine::IO_BUFFER_BYTES,
                max_payload_bytes: 2 << 30,
                max_entries: 8,
                stride: 1024,
                write_depth,
                required: true,
            }),
        }),
        expert_capacity_control: Some(budget),
        adaptive_memory: true,
        ane_reserved_bytes: 0,
    };
    std::fs::DirBuilder::new().mode(0o700).create(&root.0)?;
    let tokenizer = TextTokenizer::load(&package)?;
    let engine = Engine::load(&package, plan.slots, options.max_context)?;
    let sampling_defaults = SamplingOptions::from_package(&package)?;
    let mut state = State {
        package,
        tokenizer,
        engine,
        memory_cache: None,
        disk_cache: None,
        sampling_defaults,
        options,
        next_id: 0,
        health: Arc::new(RwLock::new(Value::Null)),
        adaptive: Some(Controller::new(budget)?),
        prefill_phase: None,
    };
    state.initialize_test_caches();
    state
        .adaptive
        .as_mut()
        .unwrap()
        .fix_observation(Duration::ZERO, 1);
    let server = Server::http("127.0.0.1:0").map_err(|error| anyhow!("{error}"))?;
    Ok((root, state, server))
}

#[test]
#[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
fn conversation_state_survives_resizing_and_interleaving() -> Result<()> {
    let (_root, mut state, server) = fixture(16384, DiskCacheMode::Full)?;
    let mut conversation = json!([
        {"role": "system", "content": format!("Answer briefly. Context notes:\n{}",
            "The sample project uses Rust. Its tests check parsing and request handling.\n".repeat(80))},
        {"role": "user", "content": "Hello. What language does this project use?"},
    ]);
    let first = request(&server, &mut state, conversation.clone(), false)?;
    assert_eq!(first["usage"]["prompt_cache"]["event"], "cold_start");
    assert!(first["usage"]["prompt_tokens"].as_u64().unwrap() > 1024);
    assert!(
        first["usage"]["prompt_cache"]["disk_checkpoints_written"]
            .as_u64()
            .unwrap()
            > 0
    );
    let expected_prefix = first["usage"]["prompt_tokens"].as_u64().unwrap();
    let assistant = first["choices"][0]["message"].clone();
    assert!(
        assistant["content"]
            .as_str()
            .is_some_and(|text| !text.is_empty())
    );
    conversation.as_array_mut().unwrap().extend([
        assistant,
        json!({"role": "user", "content": "What do its tests check?"}),
    ]);

    // A useful follow-up must coexist with actual growth, not suppress it.
    qualify_growth(&mut state, 1)?;
    let before = state.engine.expert_capacity();
    let warm = request(&server, &mut state, conversation.clone(), true)?;
    assert_eq!(
        warm["usage"]["prompt_cache"]["event"], "memory_hit",
        "{warm}"
    );
    assert!(
        warm["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap()
            > expected_prefix,
        "follow-up must reuse consumed generated tokens too: {warm}"
    );
    assert!(
        state.engine.expert_capacity() > before,
        "warm request must allow actual growth"
    );
    let warm_prompt = warm["usage"]["prompt_tokens"].as_u64().unwrap();
    let warm_cached = warm["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap();
    assert_eq!(
        warm["usage"]["moespresso"]["prefill_tokens"],
        warm_prompt - warm_cached
    );
    assert!(
        warm["usage"]["moespresso"]["prefill_seconds"]
            .as_f64()
            .is_some_and(|seconds| seconds > 0.0)
    );

    // Losing the RAM tier must not make admission throw away the disk prefix.
    state.memory_cache.as_mut().unwrap().clear();
    let disk = request(&server, &mut state, conversation.clone(), false)?;
    assert_eq!(disk["usage"]["prompt_cache"]["event"], "disk_hit", "{disk}");
    assert!(
        disk["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap()
            >= 1024
    );
    // Active compression rolls back the actual trial without discarding state.
    let trial = state.engine.expert_capacity();
    let controller = state.adaptive.as_mut().unwrap();
    controller.begin_request_at(Duration::from_secs(22));
    let mut rollbacks = 0;
    for second in 22..=40 {
        rollbacks += usize::from(controller.check_with_observation(
            &mut state.engine,
            false,
            Duration::from_secs(second),
            Some(1),
            Some(((second - 21) * 4096, 0)),
        )?);
    }
    controller.end_request_at(Duration::from_secs(40), false);
    controller.fix_observation(Duration::from_secs(41), 1);
    assert_eq!(rollbacks, 1);
    assert!(state.engine.expert_capacity() < trial);
    let rollback = request(&server, &mut state, conversation.clone(), false)?;
    assert_eq!(
        rollback["usage"]["prompt_cache"]["event"], "memory_hit",
        "trial rollback discarded conversation: {rollback}"
    );

    let cold = request(
        &server,
        &mut state,
        json!([
            {"role": "system", "content": "Answer briefly about arithmetic."},
            {"role": "user", "content": "What is two plus two?"},
        ]),
        false,
    )?;
    assert_eq!(cold["usage"]["prompt_cache"]["event"], "cold_start");
    let interleaved = request(&server, &mut state, conversation.clone(), false)?;
    assert_eq!(
        interleaved["usage"]["prompt_cache"]["event"], "memory_hit",
        "unrelated request displaced conversation: {interleaved}"
    );

    // Pressure release is mandatory even with an exact memory-cache hit.
    let grown = state.engine.expert_capacity();
    state
        .adaptive
        .as_mut()
        .unwrap()
        .fix_observation(Duration::from_secs(42), 2);
    let shrunk = request(&server, &mut state, conversation.clone(), false)?;
    assert!(state.engine.expert_capacity() < grown);
    assert_eq!(
        shrunk["usage"]["prompt_cache"]["event"], "memory_hit",
        "pressure release discarded conversation: {shrunk}"
    );
    // Recovery cannot skip settling after pressure.
    state
        .adaptive
        .as_mut()
        .unwrap()
        .fix_observation(Duration::from_secs(43), 1);
    assert!(
        !state
            .adaptive
            .as_mut()
            .unwrap()
            .check_boundary(&mut state.engine, Boundary::Observe)?
    );

    let mut short = json!([
        {"role": "system", "content": "Answer briefly about fruit."},
        {"role": "user", "content": "Name a yellow fruit."},
    ]);
    let short_first = request(&server, &mut state, short.clone(), false)?;
    let short_prompt = short_first["usage"]["prompt_tokens"].as_u64().unwrap();
    assert!(short_prompt < 512);
    short.as_array_mut().unwrap().extend([
        short_first["choices"][0]["message"].clone(),
        json!({"role": "user", "content": "What color was it?"}),
    ]);
    let _ = request(&server, &mut state, conversation.clone(), false)?;
    let short_next = request(&server, &mut state, short, true)?;
    assert!(
        matches!(
            short_next["usage"]["prompt_cache"]["event"].as_str(),
            Some("memory_hit" | "disk_hit")
        ),
        "short follow-up lost its state: {short_next}"
    );
    assert!(
        short_next["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap()
            > short_prompt,
        "short follow-up must reuse generated state: {short_next}"
    );
    Ok(())
}

#[test]
#[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
fn short_generated_conversation_survives_restart_at_another_capacity() -> Result<()> {
    // An anchor depth smaller than the prompt must not prevent end-of-turn persistence.
    let (root, mut state, server) = fixture(1, DiskCacheMode::Full)?;
    let mut conversation = json!([
        {"role": "system", "content": "Answer briefly about fruit."},
        {"role": "user", "content": "Name a yellow fruit."},
    ]);
    let first = request(&server, &mut state, conversation.clone(), false)?;
    let prompt_tokens = first["usage"]["prompt_tokens"].as_u64().unwrap();
    assert!(prompt_tokens < 512);
    assert!(first["usage"]["completion_tokens"].as_u64().unwrap() > 0);
    conversation.as_array_mut().unwrap().extend([
        first["choices"][0]["message"].clone(),
        json!({"role": "user", "content": "What color was it?"}),
    ]);

    // Drop all engine and RAM-cache ownership, reopen the same disk root with
    // a different startup capacity, and issue an ordinary rendered follow-up.
    let State {
        package,
        tokenizer,
        engine,
        memory_cache,
        disk_cache,
        sampling_defaults,
        mut options,
        ..
    } = state;
    drop(memory_cache);
    drop(disk_cache);
    drop(engine);
    options.expert_capacity = 128;
    let engine = Engine::load(&package, options.expert_capacity, options.max_context)?;
    let mut restarted = State {
        package,
        tokenizer,
        engine,
        memory_cache: None,
        disk_cache: None,
        sampling_defaults,
        options,
        next_id: 0,
        health: Arc::new(RwLock::new(Value::Null)),
        adaptive: None,
        prefill_phase: None,
    };
    restarted.initialize_test_caches();
    let followup = request(&server, &mut restarted, conversation.clone(), true)?;
    assert_eq!(
        followup["usage"]["prompt_cache"]["event"], "disk_hit",
        "restart replayed a short conversation: {followup}"
    );
    assert!(
        followup["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap()
            > prompt_tokens,
        "restart must reuse committed generated state, not only a prompt checkpoint: {followup}"
    );

    // A damaged newest checkpoint must not prevent using an older valid prefix.
    let frontier = followup["usage"]["prompt_tokens"].as_u64().unwrap();
    let index: Value = serde_json::from_slice(&std::fs::read(root.0.join("index.json"))?)?;
    let entry = index["entries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["frontier"].as_u64() == Some(frontier))
        .expect("completed follow-up prompt must be persisted");
    let slot = entry["slot"].as_u64().unwrap();
    let path = root.0.join(format!("payload-{slot:02}.bin"));
    let mut bytes = std::fs::read(&path)?;
    *bytes.last_mut().unwrap() ^= 1;
    std::fs::write(path, bytes)?;
    restarted.memory_cache.as_mut().unwrap().clear();
    let recovered = request(&server, &mut restarted, conversation, false)?;
    assert_eq!(recovered["usage"]["prompt_cache"]["event"], "disk_hit");
    let cached = recovered["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap();
    assert!(cached > prompt_tokens && cached < frontier, "{recovered}");
    Ok(())
}

#[test]
#[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
fn disk_retention_keeps_interleaved_conversations_while_one_chain_advances() -> Result<()> {
    let (root, mut state, server) = fixture(1, DiskCacheMode::Full)?;
    let title = json!([
        {"role": "system", "content": "Produce a short conversation title."},
        {"role": "user", "content": "We are discussing a Rust project."},
    ]);
    let summary = json!([
        {"role": "system", "content": "Summarize briefly."},
        {"role": "user", "content": "A Rust project has parsing tests."},
    ]);
    let mut main = json!([
        {"role": "system", "content": "Answer briefly about fruit."},
        {"role": "user", "content": "Name a yellow fruit."},
    ]);
    let mut answer = request(&server, &mut state, main.clone(), false)?;
    let title_tokens =
        request(&server, &mut state, title.clone(), false)?["usage"]["prompt_tokens"]
            .as_u64()
            .unwrap();
    let summary_tokens =
        request(&server, &mut state, summary.clone(), false)?["usage"]["prompt_tokens"]
            .as_u64()
            .unwrap();
    for _ in 0..4 {
        main.as_array_mut().unwrap().extend([
            answer["choices"][0]["message"].clone(),
            json!({"role": "user", "content": "What color was it?"}),
        ]);
        answer = request(&server, &mut state, main.clone(), false)?;
    }
    let index: Value = serde_json::from_slice(&std::fs::read(root.0.join("index.json"))?)?;
    assert!(index["entries"].as_array().unwrap().len() <= 8);
    let retained: u64 = index["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["file_bytes"].as_u64().unwrap())
        .sum();
    assert!(retained <= 2 << 30);
    // Discard RAM so the assertions exercise retained disk state, not a
    // coincidental memory hit while disk history has evicted the other chats.
    for (messages, expected) in [(title, title_tokens), (summary, summary_tokens)] {
        state.memory_cache.as_mut().unwrap().clear();
        let resumed = request(&server, &mut state, messages, false)?;
        assert_eq!(
            resumed["usage"]["prompt_cache"]["event"], "disk_hit",
            "{resumed}"
        );
        assert_eq!(
            resumed["usage"]["prompt_tokens_details"]["cached_tokens"],
            expected
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
fn followup_uses_generated_disk_state_when_ram_can_retain_only_the_prompt() -> Result<()> {
    let (root, mut state, server) = fixture(1, DiskCacheMode::Full)?;
    // One full host snapshot fits; a prompt plus generated snapshot does not.
    let allowance = 400 << 20;
    state.options.cache.as_mut().unwrap().memory_retained_bytes = allowance;
    state.memory_cache = Some(PromptPrefixCache::new(&state.engine, allowance));
    let mut conversation = json!([
        {"role": "system", "content": "Answer briefly about fruit."},
        {"role": "user", "content": "Name a yellow fruit."},
    ]);
    let first = request(&server, &mut state, conversation.clone(), false)?;
    let prompt = first["usage"]["prompt_tokens"].as_u64().unwrap();
    conversation.as_array_mut().unwrap().extend([
        first["choices"][0]["message"].clone(),
        json!({"role": "user", "content": "What color was it?"}),
    ]);
    let followup = request(&server, &mut state, conversation.clone(), true)?;
    assert_eq!(
        followup["usage"]["prompt_cache"]["event"], "disk_hit",
        "{followup}"
    );
    assert!(
        followup["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap()
            > prompt,
        "generated state was replayed: {followup}"
    );

    // A corrupt deeper candidate must not displace the healthy RAM prompt.
    let retained_prompt = followup["usage"]["prompt_tokens"].as_u64().unwrap();
    let index: Value = serde_json::from_slice(&std::fs::read(root.0.join("index.json"))?)?;
    let entry = index["entries"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|entry| entry["frontier"].as_u64().unwrap() > retained_prompt)
        .max_by_key(|entry| entry["frontier"].as_u64().unwrap())
        .expect("generated continuation must be persisted");
    let damaged_frontier = entry["frontier"].as_u64().unwrap();
    let path = root.0.join(format!(
        "payload-{:02}.bin",
        entry["slot"].as_u64().unwrap()
    ));
    let mut bytes = std::fs::read(&path)?;
    *bytes.last_mut().unwrap() ^= 1;
    std::fs::write(path, bytes)?;
    conversation.as_array_mut().unwrap().extend([
        followup["choices"][0]["message"].clone(),
        json!({"role": "user", "content": "Which fruit was it?"}),
    ]);
    let recovered = request(&server, &mut state, conversation, false)?;
    assert_eq!(
        recovered["usage"]["prompt_cache"]["event"], "memory_hit",
        "{recovered}"
    );
    assert_eq!(
        recovered["usage"]["prompt_tokens_details"]["cached_tokens"],
        retained_prompt
    );
    let index: Value = serde_json::from_slice(&std::fs::read(root.0.join("index.json"))?)?;
    assert!(
        index["entries"]
            .as_array()
            .unwrap()
            .iter()
            .all(|entry| { entry["frontier"].as_u64() != Some(damaged_frontier) }),
        "damaged deeper checkpoint was not inspected"
    );
    Ok(())
}

#[test]
#[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
fn prefix_disk_cache_keeps_shared_anchors_without_turn_writes() -> Result<()> {
    let (_root, mut state, server) = fixture(4096, DiskCacheMode::default())?;
    let cache = state.options.cache.as_mut().unwrap();
    cache.memory_retained_bytes = 2 << 30;
    cache.disk.as_mut().unwrap().max_payload_bytes = 8 << 30;
    state.initialize_test_caches();
    let shared = format!("Answer briefly. Notes:\n{}", "alpha ".repeat(1536));
    let mut conversation = json!([
        {"role": "system", "content": format!("{shared}{}", "beta ".repeat(3072))},
        {"role": "user", "content": "Name a yellow fruit."},
    ]);
    let first = request(&server, &mut state, conversation.clone(), false)?;
    let prompt = first["usage"]["prompt_tokens"].as_u64().unwrap();
    assert!(prompt > 4096, "fixture must exceed the disk write depth");
    assert_eq!(
        first["usage"]["prompt_cache"]["disk_checkpoints_written"],
        4
    );
    assert_eq!(first["usage"]["prompt_cache"]["last_disk_checkpoint"], 4096);
    let root = &state
        .options
        .cache
        .as_ref()
        .unwrap()
        .disk
        .as_ref()
        .unwrap()
        .root;
    let index: Value = serde_json::from_slice(&std::fs::read(root.join("index.json"))?)?;
    let mut frontiers: Vec<_> = index["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["frontier"].as_u64().unwrap())
        .collect();
    frontiers.sort_unstable();
    assert_eq!(frontiers, [1024, 2048, 3072, 4096]);

    conversation.as_array_mut().unwrap().extend([
        first["choices"][0]["message"].clone(),
        json!({"role": "user", "content": "What color was it?"}),
    ]);
    let warm = request(&server, &mut state, conversation.clone(), true)?;
    assert_eq!(warm["usage"]["prompt_cache"]["event"], "memory_hit");
    assert!(
        warm["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap()
            > prompt
    );
    assert_eq!(warm["usage"]["prompt_cache"]["disk_checkpoints_written"], 0);
    let short = request(
        &server,
        &mut state,
        json!([{"role": "user", "content": "Say hello."}]),
        false,
    )?;
    assert!(short["usage"]["prompt_tokens"].as_u64().unwrap() < 512);
    assert_eq!(
        short["usage"]["prompt_cache"]["disk_checkpoints_written"],
        0
    );

    // Reopen both engine and store, then branch before the deeper saved anchors.
    let State {
        package,
        tokenizer,
        engine,
        memory_cache,
        disk_cache,
        sampling_defaults,
        mut options,
        ..
    } = state;
    drop(memory_cache);
    drop(disk_cache);
    drop(engine);
    options.expert_capacity = 128;
    let engine = Engine::load(&package, options.expert_capacity, options.max_context)?;
    let mut state = State {
        package,
        tokenizer,
        engine,
        sampling_defaults,
        options,
        memory_cache: None,
        disk_cache: None,
        next_id: 0,
        health: Arc::new(RwLock::new(Value::Null)),
        adaptive: None,
        prefill_phase: None,
    };
    state.initialize_test_caches();
    let branched = request(
        &server,
        &mut state,
        json!([
            {"role": "system", "content": format!("{shared}Different notes.")},
            {"role": "user", "content": "Name a green fruit."},
        ]),
        false,
    )?;
    assert_eq!(branched["usage"]["prompt_cache"]["event"], "disk_hit");
    assert_eq!(
        branched["usage"]["prompt_tokens_details"]["cached_tokens"],
        1024
    );
    assert_eq!(
        branched["usage"]["prompt_cache"]["disk_checkpoints_written"],
        0
    );

    // Below RAM admission, reuse stops at the disk anchor and evaluates the tail.
    state.options.cache.as_mut().unwrap().memory_retained_bytes = 1;
    state.memory_cache = Some(PromptPrefixCache::new(&state.engine, 1));
    for _ in 0..2 {
        let resumed = request(&server, &mut state, conversation.clone(), false)?;
        assert_eq!(resumed["usage"]["prompt_cache"]["event"], "disk_hit");
        assert_eq!(
            resumed["usage"]["prompt_tokens_details"]["cached_tokens"],
            4096
        );
        assert!(
            resumed["usage"]["moespresso"]["prefill_tokens"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert_eq!(
            resumed["usage"]["prompt_cache"]["disk_checkpoints_written"],
            0
        );
        assert_eq!(state.memory_cache.as_ref().unwrap().retained_bytes()?, 0);
    }
    Ok(())
}
