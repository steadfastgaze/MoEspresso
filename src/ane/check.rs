//! Deterministic model-free qualification of one dynamic ANE projection.

use super::{DynamicLayout, DynamicProjection, ProjectionShape};
use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use std::{io, time::Instant};

const ROWS: usize = 64;
const CHANNELS: usize = 512;
const ITERATIONS: usize = 3;
const GUARD: usize = 32;
const CANARY: u16 = 0x7e55;

/// One stage boundary. Iterations are zero-based; begin events have zero timing.
#[derive(Clone, Debug, Serialize)]
pub struct CheckEvent {
    pub stage: &'static str,
    pub iteration: Option<usize>,
    pub completed: bool,
    pub checked_elements: usize,
    pub elapsed_seconds: f64,
}

/// Successful compile, changed-operand evaluations, exact readback and unload.
#[derive(Clone, Debug, Serialize)]
pub struct CheckReport {
    pub rows: usize,
    pub input_channels: usize,
    pub output_channels: usize,
    pub evaluations: usize,
    pub checked_elements: usize,
    pub input_surface_bytes: usize,
    pub output_surface_bytes: usize,
    pub compile_seconds: f64,
    pub load_seconds: f64,
    pub compile_load_wall_seconds: f64,
    pub evaluate_seconds: [f64; ITERATIONS],
    pub unload_seconds: f64,
    pub total_seconds: f64,
    /// Process memory after unload, when the host query succeeds; not a peak.
    pub footprint_bytes: Option<u64>,
    pub resident_bytes: Option<u64>,
}

// Exact FP16 representation of a small integer divided by 2^denominator.
fn dyadic(value: i32, denominator: i32) -> u16 {
    if value == 0 {
        return 0;
    }
    let sign = if value < 0 { 0x8000 } else { 0 };
    let magnitude = value.unsigned_abs();
    let exponent = magnitude.ilog2();
    let fraction = (magnitude << (10 - exponent)) & 1023;
    sign | (((exponent as i32 - denominator + 15) as u16) << 10) | fraction as u16
}

fn input_value(channel: usize, row: usize, iteration: usize) -> i32 {
    let value = ((channel * 7 + row * 3 + iteration * 11) % 31) as i32 - 15;
    if value == 0 { 16 } else { value }
}

fn fill_input(layout: DynamicLayout, iteration: usize, combined: &mut [u16]) -> Result<()> {
    ensure!(
        layout.shape == ProjectionShape::new(ROWS, CHANNELS, CHANNELS)?
            && iteration < ITERATIONS
            && combined.len() == layout.input_elements,
        "ANE check fixture extent differs"
    );
    combined.fill(0);
    for channel in 0..CHANNELS {
        for row in 0..ROWS {
            combined[channel * layout.spatial + row] =
                dyadic(input_value(channel, row, iteration), 4);
        }
    }
    for output in 0..CHANNELS {
        let source = (output * 17 + iteration * 23 + 1) % CHANNELS;
        let negative = !(output + iteration).is_multiple_of(2);
        combined[source * layout.spatial + ROWS + output] = if negative { 0xb800 } else { 0x3800 };
    }
    Ok(())
}

fn verify_output(iteration: usize, output: &[u16]) -> Result<()> {
    ensure!(
        iteration < ITERATIONS && output.len() == CHANNELS * ROWS,
        "ANE check output extent differs"
    );
    for channel in 0..CHANNELS {
        let source = (channel * 17 + iteration * 23 + 1) % CHANNELS;
        let sign = if !(channel + iteration).is_multiple_of(2) {
            -1
        } else {
            1
        };
        for row in 0..ROWS {
            let expected = dyadic(sign * input_value(source, row, iteration), 5);
            let actual = output[channel * ROWS + row];
            if actual & 0x7c00 == 0x7c00 || actual != expected {
                bail!(
                    "iteration {iteration} output[{channel},{row}] {actual:04x} != {expected:04x}"
                );
            }
        }
    }
    Ok(())
}

fn emit(
    on_event: &mut impl FnMut(CheckEvent) -> io::Result<()>,
    stage: &'static str,
    iteration: Option<usize>,
    completed: bool,
    checked_elements: usize,
    elapsed_seconds: f64,
) -> Result<()> {
    on_event(CheckEvent {
        stage,
        iteration,
        completed,
        checked_elements,
        elapsed_seconds,
    })
    .with_context(|| format!("report ANE {stage} stage"))
}

fn run_evaluations(
    projection: &mut DynamicProjection,
    layout: DynamicLayout,
    combined: &mut [u16],
    guarded_output: &mut [u16],
    on_event: &mut impl FnMut(CheckEvent) -> io::Result<()>,
) -> Result<[f64; ITERATIONS]> {
    let n = layout.shape.output_elements()?;
    let mut times = [0.0; ITERATIONS];
    for (iteration, seconds) in times.iter_mut().enumerate() {
        emit(on_event, "prepare", Some(iteration), false, 0, 0.0)?;
        let start = Instant::now();
        fill_input(layout, iteration, combined)?;
        guarded_output.fill(CANARY);
        emit(
            on_event,
            "prepare",
            Some(iteration),
            true,
            0,
            start.elapsed().as_secs_f64(),
        )?;

        emit(on_event, "write", Some(iteration), false, 0, 0.0)?;
        let start = Instant::now();
        projection
            .write_combined_input(combined)
            .with_context(|| format!("ANE check input and weight write iteration {iteration}"))?;
        emit(
            on_event,
            "write",
            Some(iteration),
            true,
            0,
            start.elapsed().as_secs_f64(),
        )?;

        emit(on_event, "evaluate", Some(iteration), false, 0, 0.0)?;
        let elapsed = projection
            .evaluate()
            .with_context(|| format!("ANE check evaluation iteration {iteration}"))?;
        *seconds = elapsed.as_secs_f64();
        emit(on_event, "evaluate", Some(iteration), true, 0, *seconds)?;

        emit(on_event, "read", Some(iteration), false, 0, 0.0)?;
        let start = Instant::now();
        projection
            .read_output_channel_major(&mut guarded_output[GUARD..GUARD + n])
            .with_context(|| format!("ANE check readback iteration {iteration}"))?;
        emit(
            on_event,
            "read",
            Some(iteration),
            true,
            0,
            start.elapsed().as_secs_f64(),
        )?;

        emit(on_event, "verify", Some(iteration), false, 0, 0.0)?;
        let start = Instant::now();
        ensure!(
            guarded_output[..GUARD]
                .iter()
                .chain(&guarded_output[GUARD + n..])
                .all(|&word| word == CANARY),
            "ANE check host output guard changed"
        );
        verify_output(iteration, &guarded_output[GUARD..GUARD + n])?;
        emit(
            on_event,
            "verify",
            Some(iteration),
            true,
            n,
            start.elapsed().as_secs_f64(),
        )?;
    }
    Ok(times)
}

fn add_cleanup_error(result: &mut Result<CheckReport>, error: anyhow::Error) {
    let primary = std::mem::replace(result, Err(anyhow::anyhow!("ANE check cleanup pending")));
    *result = match primary {
        Ok(_) => Err(error),
        Err(primary) if !super::startup_fallback_allowed(&error) => {
            Err(error.context(format!("ANE check failed before cleanup: {primary:#}")))
        }
        Err(primary) => Err(primary.context(format!(
            "additional ANE cleanup/reporting failure: {error:#}"
        ))),
    };
}

/// Exercise one fixed projection without loading a model package or using Metal.
/// The caller renders or persists progress and supplies any process deadline.
/// Success includes three changed input/weight pairs and explicit native unload.
/// This qualifies only the exercised runtime path, not model numerical quality.
pub fn check(mut on_event: impl FnMut(CheckEvent) -> io::Result<()>) -> Result<CheckReport> {
    let total = Instant::now();
    let shape = ProjectionShape::new(ROWS, CHANNELS, CHANNELS)?;
    let layout = DynamicLayout::new(shape)?;
    let mut combined = vec![0; layout.input_elements];
    let n = shape.output_elements()?;
    let mut output = vec![CANARY; n + GUARD * 2];
    emit(&mut on_event, "compile_load", None, false, 0, 0.0)?;
    let start = Instant::now();
    let mut projection =
        DynamicProjection::compile(shape).context("ANE check IOSurface creation/compile/load")?;
    let compile_load_wall_seconds = start.elapsed().as_secs_f64();
    let mut result = (|| {
        emit(
            &mut on_event,
            "compile_load",
            None,
            true,
            0,
            compile_load_wall_seconds,
        )?;
        let (compile_seconds, load_seconds) = projection.compile_stage_seconds()?;
        ensure!(
            [compile_seconds, load_seconds]
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0),
            "ANE check compile/load timing is invalid"
        );
        let evaluate_seconds = run_evaluations(
            &mut projection,
            layout,
            &mut combined,
            &mut output,
            &mut on_event,
        )?;
        Ok(CheckReport {
            rows: ROWS,
            input_channels: CHANNELS,
            output_channels: CHANNELS,
            evaluations: ITERATIONS,
            checked_elements: n * ITERATIONS,
            input_surface_bytes: layout.input_elements * 2,
            output_surface_bytes: n * 2,
            compile_seconds,
            load_seconds,
            compile_load_wall_seconds,
            evaluate_seconds,
            unload_seconds: 0.0,
            total_seconds: 0.0,
            footprint_bytes: None,
            resident_bytes: None,
        })
    })();

    // A progress callback error must not bypass explicit unload of a loaded owner.
    if let Err(error) = emit(&mut on_event, "unload", None, false, 0, 0.0) {
        add_cleanup_error(&mut result, error);
    }
    let start = Instant::now();
    let close = projection.close().context("ANE check explicit unload");
    let unload_seconds = start.elapsed().as_secs_f64();
    match close {
        Err(error) => add_cleanup_error(&mut result, error),
        Ok(()) => {
            if let Err(error) = emit(&mut on_event, "unload", None, true, 0, unload_seconds) {
                add_cleanup_error(&mut result, error);
            }
        }
    }
    if let Ok(report) = &mut result {
        report.unload_seconds = unload_seconds;
        report.total_seconds = total.elapsed().as_secs_f64();
        if let Ok((footprint, resident)) = super::memory_stats() {
            report.footprint_bytes = Some(footprint);
            report.resident_bytes = Some(resident);
        }
    }
    result
}

#[cfg(test)]
mod tests;
