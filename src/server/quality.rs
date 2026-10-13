//! Small project-owned whole-model regression cases. No external oracle is implied.

use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use half::bf16;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::stream::ResponseShaper;
use crate::engine::Engine;
use crate::package::Package;
use crate::routing::RoutingMode;
use crate::tokenizer::{ChatOptions, TextTokenizer};

const VOCABULARY: usize = 248_320;
const DEFAULT_PREFILL: usize = 512;
const LONG_FRONTIERS: [usize; 7] = [2047, 2048, 2049, 8447, 8448, 8449, 10240];
const GENERATION_LIMIT: usize = 96;

struct QualityPrefill {
    candidate: usize,
    reference: Option<usize>,
}

impl QualityPrefill {
    fn parse(candidate: Option<&str>, reference: Option<&str>, recording: bool) -> Result<Self> {
        let candidate = candidate
            .map(str::parse::<usize>)
            .transpose()
            .context("MOESPRESSO_QUALITY_PREFILL_ROWS must be an integer")?
            .unwrap_or(DEFAULT_PREFILL);
        let reference = reference
            .map(str::parse::<usize>)
            .transpose()
            .context("MOESPRESSO_QUALITY_REFERENCE_PREFILL_ROWS must be an integer")?;
        crate::prefill_capacity::validate_capacity(candidate)?;
        if let Some(rows) = reference {
            crate::prefill_capacity::validate_capacity(rows)?;
        }
        ensure!(
            !recording || reference.is_none(),
            "recording cannot declare a reference prefill width"
        );
        Ok(Self {
            candidate,
            reference,
        })
    }

    fn from_env(recording: bool) -> Result<Self> {
        fn setting(name: &str) -> Result<Option<String>> {
            match std::env::var(name) {
                Ok(value) => Ok(Some(value)),
                Err(std::env::VarError::NotPresent) => Ok(None),
                Err(_) => anyhow::bail!("{name} must be an integer"),
            }
        }
        let candidate = setting("MOESPRESSO_QUALITY_PREFILL_ROWS")?;
        let reference = setting("MOESPRESSO_QUALITY_REFERENCE_PREFILL_ROWS")?;
        Self::parse(candidate.as_deref(), reference.as_deref(), recording)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Recipe {
    version: u32,
    package: String,
    capacity: usize,
    routing: String,
    prefill: usize,
    context: usize,
    cases: Vec<Case>,
}

fn check_recipe(
    candidate: &Recipe,
    reference: &Recipe,
    declared_reference_prefill: Option<usize>,
) -> Result<()> {
    ensure!(
        reference.prefill == declared_reference_prefill.unwrap_or(candidate.prefill),
        "baseline prefill width differs; a cross-width comparison requires its explicitly declared reference width"
    );
    ensure!(
        candidate.version == reference.version
            && candidate.package == reference.package
            && candidate.capacity == reference.capacity
            && candidate.routing == reference.routing
            && candidate.context == reference.context
            && candidate.cases == reference.cases,
        "baseline recipe differs (package, tokens, frontiers, capacity, routing, context or version); only the declared reference prefill width may differ"
    );
    Ok(())
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Case {
    name: String,
    tokens: Vec<u32>,
    frontiers: Vec<usize>,
    generation: Option<Generation>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Generation {
    prompt: Vec<u32>,
    tools: Vec<Value>,
    expected: Expected,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
enum Expected {
    Answer(String),
    Tool { name: String, arguments: Value },
}

#[derive(Clone, Copy)]
struct Tolerance {
    logit_rms: f64,
    mean_nll_increase: f64,
}

impl Tolerance {
    fn from_env() -> Result<Self> {
        fn value(name: &str) -> Result<f64> {
            let value = std::env::var(name)
                .unwrap_or_else(|_| "0".into())
                .parse::<f64>()
                .with_context(|| format!("{name} must be a nonnegative finite number"))?;
            ensure!(value.is_finite() && value >= 0.0, "invalid {name}");
            Ok(value)
        }
        Ok(Self {
            logit_rms: value("MOESPRESSO_QUALITY_LOGIT_RMS")?,
            mean_nll_increase: value("MOESPRESSO_QUALITY_NLL_INCREASE")?,
        })
    }
}

fn values(row: &[u16]) -> Result<Vec<f64>> {
    ensure!(!row.is_empty(), "empty vocabulary row");
    let result: Vec<f64> = row
        .iter()
        .map(|&bits| f64::from(bf16::from_bits(bits).to_f32()))
        .collect();
    ensure!(result.iter().all(|v| v.is_finite()), "nonfinite logits");
    Ok(result)
}

fn nll(row: &[u16], target: u32) -> Result<f64> {
    let row = values(row)?;
    let target = *row
        .get(target as usize)
        .context("target exceeds vocabulary")?;
    let maximum = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(maximum + row.iter().map(|v| (v - maximum).exp()).sum::<f64>().ln() - target)
}

fn greedy(row: &[u16]) -> Result<u32> {
    let row = values(row)?;
    let mut selected = 0;
    for index in 1..row.len() {
        if row[index] > row[selected] {
            selected = index;
        }
    }
    Ok(selected as u32)
}

#[derive(Debug)]
struct Difference {
    maximum_row_rms: f64,
    mean_nll_increase: f64,
    largest_token_nll_increase: f64,
    changed_greedy: usize,
}

fn difference(actual: &[Vec<u16>], baseline: &[Vec<u16>], targets: &[u32]) -> Result<Difference> {
    ensure!(
        !targets.is_empty() && actual.len() == targets.len() && baseline.len() == targets.len(),
        "logit row count differs"
    );
    let mut result = Difference {
        maximum_row_rms: 0.0,
        mean_nll_increase: 0.0,
        largest_token_nll_increase: f64::NEG_INFINITY,
        changed_greedy: 0,
    };
    for ((row, reference), &target) in actual.iter().zip(baseline).zip(targets) {
        ensure!(row.len() == reference.len(), "vocabulary width differs");
        let a = values(row)?;
        let b = values(reference)?;
        let rms =
            (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64).sqrt();
        let increase = nll(row, target)? - nll(reference, target)?;
        result.maximum_row_rms = result.maximum_row_rms.max(rms);
        result.mean_nll_increase += increase / targets.len() as f64;
        result.largest_token_nll_increase = result.largest_token_nll_increase.max(increase);
        result.changed_greedy += usize::from(greedy(row)? != greedy(reference)?);
    }
    Ok(result)
}

fn check_difference(value: &Difference, tolerance: Tolerance) -> Result<()> {
    ensure!(
        value.maximum_row_rms <= tolerance.logit_rms
            && value.mean_nll_increase <= tolerance.mean_nll_increase,
        "numerical baseline changed; review the per-case differences before changing tolerances or recording a new baseline"
    );
    Ok(())
}

fn grade(generation: &Generation, output: &str, stopped: bool) -> Result<()> {
    ensure!(
        stopped,
        "generation reached its token limit without stopping"
    );
    let tools = (!generation.tools.is_empty()).then_some(generation.tools.as_slice());
    let mut shaped = ResponseShaper::new(false, tools, "quality".into());
    shaped.push(output);
    shaped.finish(false);
    ensure!(shaped.reasoning.is_empty(), "unexpected reasoning output");
    match &generation.expected {
        Expected::Answer(answer) => ensure!(
            shaped.calls.is_empty() && shaped.content.trim() == answer,
            "answer check failed: expected {answer:?}, found {:?}",
            shaped.content
        ),
        Expected::Tool { name, arguments } => {
            ensure!(
                shaped.content.trim().is_empty(),
                "unexpected text beside tool call"
            );
            ensure!(shaped.calls.len() == 1, "expected exactly one tool call");
            let function = &shaped.calls[0]["function"];
            let actual: Value = serde_json::from_str(
                function["arguments"]
                    .as_str()
                    .context("tool arguments are absent")?,
            )?;
            ensure!(
                function["name"] == *name && actual == *arguments,
                "tool call differs: {function}"
            );
        }
    }
    Ok(())
}

fn chat_case(
    tokenizer: &TextTokenizer,
    name: &str,
    instruction: &str,
    continuation: &str,
    tools: Vec<Value>,
    expected: Expected,
) -> Result<Case> {
    let prompt = tokenizer.render(
        &[json!({"role": "user", "content": instruction})],
        (!tools.is_empty()).then_some(tools.as_slice()),
        &ChatOptions {
            thinking: false,
            ..Default::default()
        },
    )?;
    // Tokenize the complete text, allowing a BPE merge at the answer boundary.
    let tokens = tokenizer.encode(&format!("{}{continuation}", prompt.text))?;
    let common = prompt
        .tokens
        .iter()
        .zip(&tokens)
        .take_while(|(a, b)| a == b)
        .count();
    ensure!(
        common > 0 && common < tokens.len(),
        "invalid teacher-forced suffix"
    );
    Ok(Case {
        name: name.into(),
        frontiers: (common..tokens.len()).collect(),
        tokens,
        generation: Some(Generation {
            prompt: prompt.tokens,
            tools,
            expected,
        }),
    })
}

fn short_cases(tokenizer: &TextTokenizer) -> Result<Vec<Case>> {
    Ok(vec![
        chat_case(
            tokenizer,
            "record-retrieval",
            "The relay station access code is amber-otter-52. Reply with only the access code.",
            "amber-otter-52",
            vec![],
            Expected::Answer("amber-otter-52".into()),
        )?,
        chat_case(
            tokenizer,
            "tile-count",
            "A box has 17 green tiles and 8 blue tiles. Four green tiles are removed. How many tiles remain? Reply with only the number.",
            "21",
            vec![],
            Expected::Answer("21".into()),
        )?,
        chat_case(
            tokenizer,
            "inventory-tool",
            "Call lookup_inventory with sku dawn-17. Do not answer from memory. Make exactly one tool call and no other text.",
            "<tool_call>\n<function=lookup_inventory>\n<parameter=sku>\ndawn-17\n</parameter>\n</function>\n</tool_call>",
            vec![json!({"type": "function", "function": {
                "name": "lookup_inventory", "description": "Look up a stock item by its SKU.",
                "parameters": {"type": "object", "properties": {"sku": {"type": "string"}},
                    "required": ["sku"], "additionalProperties": false}
            }})],
            Expected::Tool {
                name: "lookup_inventory".into(),
                arguments: json!({"sku": "dawn-17"}),
            },
        )?,
    ])
}

fn long_cases(tokenizer: &TextTokenizer) -> Result<Vec<Case>> {
    let mut text =
        String::from("The following is a maintenance ledger for a fictional observatory.\n");
    for day in 0..1800 {
        use std::fmt::Write;
        writeln!(
            text,
            "Entry {day}: the amber team checked {} lenses, then stored {} blue filters in cabinet {}.",
            day % 17 + 3,
            day % 11 + 2,
            day % 23
        )?;
    }
    let mut tokens = tokenizer.encode(&text)?;
    ensure!(tokens.len() > 10240, "long fixture is too short");
    tokens.truncate(10241);
    Ok(vec![Case {
        name: "attention-and-packed-kv-boundaries".into(),
        tokens,
        // QSA switches above 2048; the first KVarN tile seals at 8448.
        frontiers: LONG_FRONTIERS.to_vec(),
        generation: None,
    }])
}

fn prefill_chunk_end(consumed: usize, frontier: usize, width: usize) -> usize {
    consumed.saturating_add(width).min(frontier)
}

fn forced_rows(engine: &mut Engine, case: &Case, mode: RoutingMode) -> Result<Vec<Vec<u16>>> {
    let mut session = engine.start_session()?;
    let mut rows = Vec::with_capacity(case.frontiers.len());
    for &frontier in &case.frontiers {
        ensure!(
            frontier > 0 && frontier < case.tokens.len(),
            "invalid scored frontier"
        );
        let mut consumed = session.frontier()?;
        ensure!(frontier >= consumed, "frontiers are not ordered");
        while consumed < frontier {
            // Continuation rows use decode routing; long corpus spans use unbiased prefill.
            if !rows.is_empty() && case.generation.is_some() {
                engine.step(case.tokens[consumed], mode, &mut session)?;
                consumed += 1;
            } else {
                let end = prefill_chunk_end(consumed, frontier, engine.prefill_chunk_size());
                engine.prefill_chunk(&case.tokens[consumed..end], &mut session)?;
                consumed = end;
            }
        }
        let row = session.last_logits_bf16()?;
        ensure!(row.len() == VOCABULARY, "model vocabulary width differs");
        rows.push(row.to_vec());
    }
    Ok(rows)
}

fn generate_and_grade(
    engine: &mut Engine,
    tokenizer: &TextTokenizer,
    generation: &Generation,
    mode: RoutingMode,
) -> Result<()> {
    let mut session = engine.start_session()?;
    for chunk in generation.prompt.chunks(engine.prefill_chunk_size()) {
        engine.prefill_chunk(chunk, &mut session)?;
    }
    let mut output = Vec::new();
    let mut stopped = false;
    for _ in 0..GENERATION_LIMIT {
        let token = greedy(session.last_logits_bf16()?)?;
        if tokenizer.stop_ids().contains(&token) {
            stopped = true;
            break;
        }
        output.push(token);
        engine.step(token, mode, &mut session)?;
    }
    grade(generation, &tokenizer.decode(&output, false)?, stopped)
}

fn decode_rows(bytes: &[u8], count: usize, width: usize) -> Result<Vec<Vec<u16>>> {
    let expected = count
        .checked_mul(width)
        .and_then(|n| n.checked_mul(2))
        .context("row extent overflow")?;
    ensure!(
        count > 0 && width > 0 && bytes.len() == expected,
        "baseline logit extent differs"
    );
    let rows: Vec<Vec<u16>> = bytes
        .chunks_exact(width * 2)
        .map(|row| {
            row.as_chunks::<2>()
                .0
                .iter()
                .map(|v| u16::from_le_bytes([v[0], v[1]]))
                .collect()
        })
        .collect();
    for row in &rows {
        values(row)?;
    }
    Ok(rows)
}

fn run(long: bool) -> Result<()> {
    let root = PathBuf::from(std::env::var_os("MOESPRESSO_QUALITY_BASELINE").context(
        "MOESPRESSO_QUALITY_BASELINE is required; recording is a separate explicit command",
    )?);
    let directory = root.join(if long { "long" } else { "short" });
    let recording = match std::env::var("MOESPRESSO_QUALITY_RECORD").as_deref() {
        Err(_) | Ok("0") => false,
        Ok("1") => true,
        _ => anyhow::bail!("MOESPRESSO_QUALITY_RECORD must be 0 or 1"),
    };
    let prefill = QualityPrefill::from_env(recording)?;
    // Missing references fail before package loading or GPU allocation.
    let reference: Option<Recipe> = if recording {
        ensure!(
            !directory.exists(),
            "baseline already exists; recording never overwrites it"
        );
        None
    } else {
        Some(serde_json::from_slice(&fs::read(directory.join("recipe.json"))
            .context("quality baseline is absent or unreadable; use an explicitly selected known-good build to record it")?)?)
    };
    let tolerance = Tolerance::from_env()?;
    let capacity: usize = std::env::var("MOESPRESSO_QUALITY_CAPACITY")
        .unwrap_or_else(|_| "512".into())
        .parse()
        .context("invalid quality capacity")?;
    ensure!(
        (64..=512).contains(&capacity),
        "quality capacity must be between 64 and 512"
    );
    let package = Package::open(
        std::env::var_os("MOESPRESSO_TEST_PACKAGE")
            .context("MOESPRESSO_TEST_PACKAGE is required")?,
    )?;
    let tokenizer = TextTokenizer::load(&package)?;
    let mode = if capacity == 512 {
        RoutingMode::Unbiased
    } else {
        RoutingMode::CachePrior
    };
    let recipe = Recipe {
        version: 1,
        package: package.artifact_id.clone(),
        capacity,
        routing: if capacity == 512 {
            "unbiased"
        } else {
            "cache-prior-2-2"
        }
        .into(),
        prefill: prefill.candidate,
        context: if long { 10496 } else { 2048 },
        cases: if long {
            long_cases(&tokenizer)?
        } else {
            short_cases(&tokenizer)?
        },
    };
    if let Some(reference) = &reference {
        check_recipe(&recipe, reference, prefill.reference)?;
    }
    let row_count = recipe.cases.iter().map(|case| case.frontiers.len()).sum();
    let reference_rows = if recording {
        None
    } else {
        Some(decode_rows(
            &fs::read(directory.join("logits.bf16le"))?,
            row_count,
            VOCABULARY,
        )?)
    };
    println!(
        "quality: suite={} mode={} capacity={} candidate_prefill={} reference_prefill={} rows={} logit_rms_limit={} mean_nll_increase_limit={} recording={recording}",
        if long { "long" } else { "short" },
        recipe.routing,
        capacity,
        prefill.candidate,
        reference
            .as_ref()
            .map_or_else(|| "none".into(), |recipe| recipe.prefill.to_string()),
        row_count,
        tolerance.logit_rms,
        tolerance.mean_nll_increase
    );
    let started = Instant::now();
    let mut engine =
        Engine::load_with_prefill_capacity(&package, capacity, recipe.context, prefill.candidate)?;
    ensure!(
        engine.prefill_chunk_size() == prefill.candidate,
        "actual quality prefill capacity differs from selected width"
    );
    let mut all_rows = Vec::with_capacity(row_count);
    let mut failed = false;
    for case in &recipe.cases {
        let rows = forced_rows(&mut engine, case, mode)?;
        let targets: Vec<u32> = case
            .frontiers
            .iter()
            .map(|&frontier| case.tokens[frontier])
            .collect();
        let mean_nll = rows
            .iter()
            .zip(&targets)
            .map(|(row, &target)| nll(row, target))
            .collect::<Result<Vec<_>>>()?
            .iter()
            .sum::<f64>()
            / rows.len() as f64;
        println!(
            "quality: case={} frontiers={:?} mean_nll={mean_nll:.8}",
            case.name, case.frontiers
        );
        if let Some(reference) = &reference_rows {
            let compared = difference(
                &rows,
                &reference[all_rows.len()..all_rows.len() + rows.len()],
                &targets,
            )?;
            println!(
                "quality: case={} max_logit_rms={:.8} mean_nll_increase={:.8} largest_token_nll_increase={:.8} changed_greedy={}/{}",
                case.name,
                compared.maximum_row_rms,
                compared.mean_nll_increase,
                compared.largest_token_nll_increase,
                compared.changed_greedy,
                rows.len()
            );
            if let Err(error) = check_difference(&compared, tolerance) {
                eprintln!("quality: case={} {error}", case.name);
                failed = true;
            }
        }
        if let Some(generation) = &case.generation {
            generate_and_grade(&mut engine, &tokenizer, generation, mode)
                .with_context(|| format!("{} generated-output check", case.name))?;
            println!("quality: case={} generated-output check passed", case.name);
        }
        all_rows.extend(rows);
    }
    ensure!(
        !failed,
        "quality baseline differences require review; this is not by itself a finding of worse model quality"
    );
    if recording {
        fs::create_dir_all(&root)?;
        fs::create_dir(&directory)?;
        let mut logits = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(directory.join("logits.bf16le"))?,
        );
        for row in &all_rows {
            for value in row {
                logits.write_all(&value.to_le_bytes())?;
            }
        }
        logits.flush()?;
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join("recipe.json"))?
            .write_all(&serde_json::to_vec_pretty(&recipe)?)?;
        println!(
            "quality: baseline recorded in {}; no independent numerical comparison was performed",
            directory.display()
        );
    } else {
        println!("quality: baseline comparison passed");
    }
    println!(
        "quality: elapsed_seconds={:.1}",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

#[test]
#[ignore = "requires an explicitly selected real package and quality baseline"]
fn short() -> Result<()> {
    run(false)
}

#[test]
#[ignore = "optional 10K-token real-package quality baseline comparison"]
fn long() -> Result<()> {
    run(true)
}

fn row(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .map(|&value| bf16::from_f32(value).to_bits())
        .collect()
}

#[test]
fn scorer_uses_normalized_nll_and_detects_changed_distributions() -> Result<()> {
    let uniform = row(&[0.0, 0.0]);
    assert!((nll(&uniform, 0)? - 2_f64.ln()).abs() < 1e-12);
    assert!((nll(&row(&[1000.0, 1000.0]), 0)? - nll(&uniform, 0)?).abs() < 1e-12);
    let original = vec![row(&[2.0, 1.0])];
    let changed = vec![row(&[2.0, 1.5])];
    let difference = difference(&changed, &original, &[0])?;
    assert_eq!(difference.changed_greedy, 0);
    assert!(difference.mean_nll_increase > 0.0);
    assert!(
        check_difference(
            &difference,
            Tolerance {
                logit_rms: 0.0,
                mean_nll_increase: 0.0
            }
        )
        .is_err()
    );
    assert!(
        check_difference(
            &difference,
            Tolerance {
                logit_rms: 1.0,
                mean_nll_increase: 1.0
            }
        )
        .is_ok()
    );
    Ok(())
}

#[test]
fn malformed_or_nonfinite_rows_cannot_pass() {
    assert!(nll(&row(&[f32::NAN]), 0).is_err());
    assert!(nll(&row(&[f32::INFINITY]), 0).is_err());
    assert!(nll(&row(&[0.0]), 1).is_err());
    assert!(difference(&[], &[], &[]).is_err());
    assert!(difference(&[row(&[0.0])], &[row(&[0.0, 1.0])], &[0]).is_err());
    assert!(decode_rows(&[0, 0, 0], 1, 1).is_err());
    assert!(decode_rows(&[0], 1, 1).is_err());
    assert!(decode_rows(&bf16::NAN.to_bits().to_le_bytes(), 1, 1).is_err());
}

#[test]
fn output_grading_requires_finished_exact_answer_and_structured_call() -> Result<()> {
    let mut generation = Generation {
        prompt: vec![],
        tools: vec![],
        expected: Expected::Answer("21".into()),
    };
    grade(&generation, "21", true)?;
    assert!(grade(&generation, "The answer is 21", true).is_err());
    assert!(grade(&generation, "21", false).is_err());
    generation.tools = vec![
        json!({"type":"function","function":{"name":"lookup_inventory","parameters":{
        "type":"object","properties":{"sku":{"type":"string"}},"required":["sku"]}}}),
    ];
    generation.expected = Expected::Tool {
        name: "lookup_inventory".into(),
        arguments: json!({"sku":"dawn-17"}),
    };
    let valid = "<tool_call><function=lookup_inventory><parameter=sku>dawn-17</parameter></function></tool_call>";
    grade(&generation, valid, true)?;
    assert!(grade(&generation, "lookup_inventory dawn-17", true).is_err());
    assert!(grade(&generation, &valid.replace("dawn-17", "dawn-18"), true).is_err());
    assert!(grade(&generation, &format!("{valid}{valid}"), true).is_err());
    assert!(grade(&generation, valid, false).is_err());
    Ok(())
}

fn synthetic_recipe(prefill: usize) -> Recipe {
    Recipe {
        version: 1,
        package: "synthetic-package".into(),
        capacity: 128,
        routing: "cache-prior-2-2".into(),
        prefill,
        context: 10496,
        cases: vec![Case {
            name: "synthetic".into(),
            tokens: vec![10, 20, 30],
            frontiers: vec![1, 2],
            generation: None,
        }],
    }
}

#[test]
fn wider_quality_prefill_requires_a_declared_reference_width() -> Result<()> {
    let reference = synthetic_recipe(512);
    let candidate = synthetic_recipe(4096);
    assert!(check_recipe(&candidate, &reference, None).is_err());
    check_recipe(&candidate, &reference, Some(512))?;
    assert!(check_recipe(&candidate, &reference, Some(2048)).is_err());
    check_recipe(&reference, &reference, None)?;
    Ok(())
}

#[test]
fn declared_reference_width_does_not_relax_other_recipe_fields() {
    let reference = synthetic_recipe(512);
    for change in 0..7 {
        let mut candidate = synthetic_recipe(4096);
        match change {
            0 => candidate.version = 2,
            1 => candidate.package = "different-package".into(),
            2 => candidate.capacity = 192,
            3 => candidate.routing = "unbiased".into(),
            4 => candidate.context += 1,
            5 => candidate.cases[0].tokens[0] += 1,
            _ => candidate.cases[0].frontiers.reverse(),
        }
        assert!(check_recipe(&candidate, &reference, Some(512)).is_err());
    }
}

#[test]
fn quality_prefill_selection_is_explicit_and_recording_has_no_reference_override() -> Result<()> {
    assert_eq!(QualityPrefill::parse(None, None, false)?.candidate, 512);
    let selected = QualityPrefill::parse(Some("4096"), Some("512"), false)?;
    assert_eq!(selected.candidate, 4096);
    assert_eq!(selected.reference, Some(512));
    assert!(QualityPrefill::parse(Some("4096"), Some("512"), true).is_err());
    for value in ["0", "513", "4097", "not-a-width"] {
        assert!(QualityPrefill::parse(Some(value), None, false).is_err());
    }
    assert!(QualityPrefill::parse(Some("4096"), Some("513"), false).is_err());
    Ok(())
}

#[test]
fn wider_quality_schedule_executes_4096_rows_and_preserves_scored_boundaries() {
    assert_eq!(LONG_FRONTIERS, [2047, 2048, 2049, 8447, 8448, 8449, 10240]);
    let mut consumed = 0;
    let mut maximum = 0;
    let mut checkpoints = Vec::new();
    for frontier in LONG_FRONTIERS {
        while consumed < frontier {
            let end = prefill_chunk_end(consumed, frontier, 4096);
            maximum = maximum.max(end - consumed);
            consumed = end;
        }
        checkpoints.push(consumed);
    }
    assert_eq!(maximum, 4096);
    assert_eq!(checkpoints, LONG_FRONTIERS);
}

#[test]
fn changed_greedy_selection_is_diagnostic_without_being_a_quality_requirement() -> Result<()> {
    let reference = vec![row(&[0.01, 0.0])];
    let actual = vec![row(&[0.0, 0.01])];
    let compared = difference(&actual, &reference, &[0])?;
    assert_eq!(compared.changed_greedy, 1);
    check_difference(
        &compared,
        Tolerance {
            logit_rms: 0.1,
            mean_nll_increase: 0.1,
        },
    )?;
    Ok(())
}
