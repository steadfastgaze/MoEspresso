use super::*;
use half::f16;

fn sparse_reference(layout: DynamicLayout, combined: &[u16]) -> Vec<u16> {
    let mut output = vec![0; CHANNELS * ROWS];
    for out in 0..CHANNELS {
        let nonzero: Vec<_> = (0..CHANNELS)
            .filter_map(|channel| {
                let word = combined[channel * layout.spatial + ROWS + out];
                (word != 0).then_some((channel, word))
            })
            .collect();
        assert_eq!(nonzero.len(), 1);
        let (source, weight) = nonzero[0];
        let weight = f64::from(f16::from_bits(weight).to_f32());
        assert!(weight == 0.5 || weight == -0.5);
        for row in 0..ROWS {
            let input = f64::from(f16::from_bits(combined[source * layout.spatial + row]).to_f32());
            output[out * ROWS + row] = f16::from_f64(input * weight).to_bits();
        }
    }
    output
}

#[test]
fn diagnostic_dyadics_match_independent_half_conversion() {
    for value in -16..=16 {
        for denominator in [4, 5] {
            let expected = f16::from_f32(value as f32 / (1 << denominator) as f32);
            assert_eq!(dyadic(value, denominator), expected.to_bits());
        }
    }
}

#[test]
fn all_three_changed_operands_match_independent_sparse_matrix_reference() {
    let shape = ProjectionShape::new(ROWS, CHANNELS, CHANNELS).unwrap();
    let layout = DynamicLayout::new(shape).unwrap();
    let mut combined = vec![0; layout.input_elements];
    let mut previous: Option<Vec<u16>> = None;
    for iteration in 0..ITERATIONS {
        fill_input(layout, iteration, &mut combined).unwrap();
        if let Some(previous) = &previous {
            for channel in 0..CHANNELS {
                let start = channel * layout.spatial;
                assert!(
                    combined[start..start + ROWS]
                        .iter()
                        .zip(&previous[start..start + ROWS])
                        .all(|(a, b)| a != b)
                );
                assert_ne!(
                    &combined[start + ROWS..start + layout.spatial],
                    &previous[start + ROWS..start + layout.spatial]
                );
            }
        }
        let output = sparse_reference(layout, &combined);
        verify_output(iteration, &output).unwrap();
        let mut wrong = output.clone();
        wrong[0] ^= 1;
        assert!(verify_output(iteration, &wrong).is_err());
        wrong[0] = 0x7e00;
        assert!(verify_output(iteration, &wrong).is_err());
        assert!(verify_output(iteration, &output[..output.len() - 1]).is_err());
        previous = Some(combined.clone());
    }
    assert_eq!(layout.input_elements * 2, 589_824);
    assert_eq!(shape.output_elements().unwrap() * 2, 65_536);
    assert_eq!(shape.output_elements().unwrap() * ITERATIONS, 98_304);
    assert!(fill_input(layout, ITERATIONS, &mut combined).is_err());
}

#[test]
fn exact_oracle_rejects_transposed_weights_output_sign_and_stale_weights() {
    let shape = ProjectionShape::new(ROWS, CHANNELS, CHANNELS).unwrap();
    let layout = DynamicLayout::new(shape).unwrap();
    let mut original = vec![0; layout.input_elements];
    fill_input(layout, 0, &mut original).unwrap();
    let output = sparse_reference(layout, &original);
    let mut transposed_output = vec![0; output.len()];
    for row in 0..ROWS {
        for channel in 0..CHANNELS {
            transposed_output[row * CHANNELS + channel] = output[channel * ROWS + row];
        }
    }
    assert!(verify_output(0, &transposed_output).is_err());

    let mut transposed_weights = original.clone();
    for channel in 0..CHANNELS {
        for out in 0..CHANNELS {
            transposed_weights[channel * layout.spatial + ROWS + out] =
                original[out * layout.spatial + ROWS + channel];
        }
    }
    assert!(verify_output(0, &sparse_reference(layout, &transposed_weights)).is_err());

    let mut wrong_sign = original.clone();
    let source = (0..CHANNELS)
        .find(|&channel| wrong_sign[channel * layout.spatial + ROWS] != 0)
        .unwrap();
    wrong_sign[source * layout.spatial + ROWS] ^= 0x8000;
    assert!(verify_output(0, &sparse_reference(layout, &wrong_sign)).is_err());

    let mut stale_weights = original.clone();
    fill_input(layout, 1, &mut stale_weights).unwrap();
    for channel in 0..CHANNELS {
        let start = channel * layout.spatial + ROWS;
        stale_weights[start..start + CHANNELS].copy_from_slice(&original[start..start + CHANNELS]);
    }
    assert!(verify_output(1, &sparse_reference(layout, &stale_weights)).is_err());
}

#[test]
fn progress_errors_keep_the_stage_and_cleanup_errors_keep_the_primary_failure() {
    let mut callback = |event: CheckEvent| {
        assert_eq!(event.stage, "read");
        assert_eq!(event.iteration, Some(1));
        assert!(!event.completed);
        Err(io::Error::new(
            io::ErrorKind::BrokenPipe,
            "diagnostic writer closed",
        ))
    };
    let error = emit(&mut callback, "read", Some(1), false, 0, 0.0).unwrap_err();
    let description = format!("{error:#}");
    assert!(description.contains("report ANE read stage"));
    assert!(description.contains("diagnostic writer closed"));
    let mut result = Err(anyhow::anyhow!("primary evaluation failure"));
    add_cleanup_error(&mut result, anyhow::anyhow!("unload failure"));
    add_cleanup_error(&mut result, anyhow::anyhow!("progress failure"));
    let description = format!("{:#}", result.unwrap_err());
    for expected in [
        "primary evaluation failure",
        "unload failure",
        "progress failure",
    ] {
        assert!(description.contains(expected));
    }
}

#[test]
fn failed_cleanup_remains_fatal_when_the_check_already_failed() {
    let mut result = Err(anyhow::anyhow!("fixture failed"));
    let cleanup = super::super::Error::new("unload failed".into()).without_completed_cleanup();
    add_cleanup_error(&mut result, anyhow::Error::new(cleanup));
    let error = result.unwrap_err();
    assert!(!super::super::startup_fallback_allowed(&error));
    let message = format!("{error:#}");
    assert!(message.contains("fixture failed"));
    assert!(message.contains("unload failed"));
}

#[test]
#[ignore = "requires ANE support and exclusive ANE ownership"]
fn identical_projection_owners_coexist_and_unload_independently() -> Result<()> {
    unsafe extern "C" {
        fn dyn_ane_dense_test_distinct_live_directories(
            first: *mut std::ffi::c_void,
            second: *mut std::ffi::c_void,
        ) -> std::ffi::c_int;
    }
    let shape = ProjectionShape::new(ROWS, CHANNELS, CHANNELS)?;
    let layout = DynamicLayout::new(shape)?;
    let mut combined = vec![0; layout.input_elements];
    let mut output = vec![CANARY; shape.output_elements()? + GUARD * 2];
    let mut first = None;
    let mut second = None;
    let mut result = (|| -> Result<()> {
        first = Some(DynamicProjection::compile(shape).context("compile first ANE owner")?);
        second =
            Some(DynamicProjection::compile(shape).context("compile second identical ANE owner")?);
        let first_handle = first
            .as_ref()
            .context("first ANE owner absent")?
            .handle
            .context("first native handle absent")?
            .as_ptr();
        let second_handle = second
            .as_ref()
            .context("second ANE owner absent")?
            .handle
            .context("second native handle absent")?
            .as_ptr();
        ensure!(
            unsafe { dyn_ane_dense_test_distinct_live_directories(first_handle, second_handle) }
                == 0,
            "simultaneous ANE owners must retain distinct exclusively claimed directories"
        );
        let mut on_event = |_: CheckEvent| Ok(());
        run_evaluations(
            first.as_mut().context("first ANE owner absent")?,
            layout,
            &mut combined,
            &mut output,
            &mut on_event,
        )?;
        run_evaluations(
            second.as_mut().context("second ANE owner absent")?,
            layout,
            &mut combined,
            &mut output,
            &mut on_event,
        )?;
        first
            .take()
            .context("first ANE owner absent")?
            .close()
            .context("explicitly close first ANE owner")?;
        run_evaluations(
            second.as_mut().context("second ANE owner absent")?,
            layout,
            &mut combined,
            &mut output,
            &mut on_event,
        )?;
        Ok(())
    })();
    for (label, owner) in [("first", first.take()), ("second", second.take())] {
        if let Some(owner) = owner
            && let Err(cleanup) = owner
                .close()
                .with_context(|| format!("explicitly close {label} ANE owner"))
        {
            result = match result {
                Ok(()) => Err(cleanup),
                Err(primary) if !super::super::startup_fallback_allowed(&cleanup) => {
                    Err(cleanup.context(format!("ANE coexistence check failed: {primary:#}")))
                }
                Err(primary) => {
                    Err(primary.context(format!("additional ANE cleanup failure: {cleanup:#}")))
                }
            };
        }
    }
    result
}
