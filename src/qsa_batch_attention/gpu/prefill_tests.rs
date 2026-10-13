use super::*;
use crate::package::DirectWeight;
use half::bf16;

fn upload_words(metal: &Metal, words: &[u16]) -> Result<Buffer> {
    let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let buffer = metal.shared_buffer(bytes.len())?;
    buffer.write_bytes(0, &bytes)?;
    Ok(buffer)
}

fn number(word: u16) -> f64 {
    f64::from(bf16::from_bits(word).to_f32())
}

#[test]
fn matrix_tiles_and_short_tails_match_fp64_at_nonzero_offsets() -> Result<()> {
    const ACTIVE: usize = 138;
    const CAPACITY: usize = ACTIVE + 2;
    const KEYS: usize = 65;
    const WIDTH: usize = 256;
    let metal = Metal::new()?;
    // The projection is unused in this attention-consumer test. Construct a
    // small synthetic matrix without requiring package files or model weights.
    let projection = DenseBatch::new(
        &metal,
        &DirectWeight {
            role: "synthetic unused projection".into(),
            format: WeightFormat::Bf16,
            shape: vec![32, 32],
            bytes: vec![0; 32 * 32 * 2],
        },
    )?;
    let attention = QsaBatchAttention::from_projection(&metal, projection)?;
    let output = QsaBatchAttentionOutput::new_device(&metal, CAPACITY)?;
    assert_eq!(output.attention.byte_len(), 2);
    assert_eq!(output.sigmoid.byte_len(), 2);
    let diagnostic_sentinel = 0x55aau16.to_le_bytes();
    output.attention.write_bytes(0, &diagnostic_sentinel)?;
    output.sigmoid.write_bytes(0, &diagnostic_sentinel)?;
    let output_sentinel = 0x7fc1u16;
    output.gated.write_bytes(
        0,
        &(0..CAPACITY * EXPANDED)
            .flat_map(|_| output_sentinel.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;

    let queries: Vec<_> = (0..ACTIVE * EXPANDED)
        .map(|i| {
            let row = i / EXPANDED;
            let head = i / WIDTH % 24;
            let dim = i % WIDTH;
            bf16::from_f32(((row * 11 + head * 19 + dim * 7) % 61) as f32 / 32.0 - 0.9375).to_bits()
        })
        .collect();
    let gates: Vec<_> = (0..ACTIVE * EXPANDED)
        .map(|i| {
            let row = i / EXPANDED;
            let head = i / WIDTH % 24;
            bf16::from_f32(((row * 3 + head * 5 + i % WIDTH) % 33) as f32 / 8.0 - 2.0).to_bits()
        })
        .collect();
    let q = upload_words(&metal, &queries)?;
    let gate = upload_words(&metal, &gates)?;
    let selected = PendingKvarnRows::new(&metal, 64)?;
    let counts: Vec<usize> = (0..64)
        .map(|row| [1, 15, 16, 17, 31, 32, 33, KEYS][row % 8])
        .collect();
    let mut validity = vec![0u8; 64 * SELECTED];
    let mut keys = vec![0u16; 64 * 2 * KEYS * WIDTH];
    let mut values = vec![0u16; keys.len()];
    for row in 0..64 {
        for slot in 0..counts[row] {
            validity[row * SELECTED + slot] = u8::from(slot == 0 || (slot + row) % 11 != 0);
        }
        for head in 0..2 {
            let base = (row * 2 + head) * KEYS * WIDTH;
            for slot in 0..KEYS {
                for dim in 0..WIDTH {
                    let index = base + slot * WIDTH + dim;
                    keys[index] = bf16::from_f32(
                        ((row * 5 + head * 23 + slot * 17 + dim * 3) % 67) as f32 / 64.0 - 0.5,
                    )
                    .to_bits();
                    values[index] = bf16::from_f32(
                        ((row * 17 + head * 31 + slot * 13 + dim * 7) % 97) as f32 / 64.0 - 0.75,
                    )
                    .to_bits();
                }
            }
            // Only the active prefix is initialized. The consumer must not
            // read spare selected-row capacity or masked-out tail slots.
            let offset = (row * 2 + head) * SELECTED * WIDTH * 2;
            for (buffer, source) in [(&selected.keys, &keys), (&selected.values, &values)] {
                buffer.write_bytes(
                    offset,
                    &source[base..base + KEYS * WIDTH]
                        .iter()
                        .flat_map(|word| word.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            }
        }
    }
    let valid = metal.shared_buffer(validity.len())?;
    valid.write_bytes(0, &validity)?;
    let count = metal.shared_buffer(64 * 4)?;
    count.write_bytes(
        0,
        &counts
            .iter()
            .flat_map(|&v| (v as u32).to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let status = metal.shared_buffer(4)?;
    status.write_bytes(0, &0u32.to_le_bytes())?;
    let partials = metal.shared_buffer(24 * 64 * 258 * 4)?;
    let tiles = [(0, 64), (64, 64), (128, 8), (136, 2)];
    let mut stream = metal.stream()?;
    for (tile_index, &(offset, rows)) in tiles.iter().enumerate() {
        // SAFETY: Every active selected prefix and query row is initialized.
        // The same gathered tile and partials are reused on the ordered stream.
        unsafe {
            attention.encode_tile_online(
                stream.batch()?,
                &q,
                &gate,
                &selected,
                &valid,
                &count,
                offset,
                rows,
                &output,
                &status,
                &partials,
            )?;
        }
        if tile_index == 0 {
            stream.flush()?;
        }
    }
    assert!(output.gated.read_bytes(0, &mut []).is_err());
    stream.wait()?;
    let mut flag = [0; 4];
    status.read_bytes(0, &mut flag)?;
    assert_eq!(flag, [0; 4]);
    for buffer in [&output.attention, &output.sigmoid] {
        let mut bytes = [0; 2];
        buffer.read_bytes(0, &mut bytes)?;
        assert_eq!(
            bytes, diagnostic_sentinel,
            "disabled diagnostic output was written"
        );
    }
    let mut bytes = vec![0; output.gated.byte_len()];
    output.gated.read_bytes(0, &mut bytes)?;
    let actual: Vec<_> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| u16::from_le_bytes(*v))
        .collect();
    assert!(
        actual[ACTIVE * EXPANDED..]
            .iter()
            .all(|&v| v == output_sentinel),
        "attention tile overwrote inactive output rows"
    );

    let max_value = values
        .iter()
        .map(|&v| number(v).abs())
        .fold(0.0f64, f64::max);
    let mut maximum = [0.0f64; 2];
    let mut squared = [0.0f64; 2];
    let mut maximum_ratio = [0.0f64; 2];
    let mut samples = [0usize; 2];
    let mut mismatched = [0usize; 2];
    for &(offset, rows) in &tiles {
        let path = usize::from(rows <= 8);
        for local_row in 0..rows {
            let row = offset + local_row;
            for head in 0..24 {
                let kv_head = head / 12;
                let input_base = (local_row * 2 + kv_head) * KEYS * WIDTH;
                let query_base = row * EXPANDED + head * WIDTH;
                let mut scores = Vec::new();
                for slot in 0..counts[local_row] {
                    if validity[local_row * SELECTED + slot] == 0 {
                        continue;
                    }
                    let score: f64 = (0..WIDTH)
                        .map(|dim| {
                            number(queries[query_base + dim])
                                * number(keys[input_base + slot * WIDTH + dim])
                                / 16.0
                        })
                        .sum();
                    scores.push((slot, score));
                }
                let largest = scores
                    .iter()
                    .map(|(_, score)| *score)
                    .fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<_> = scores
                    .iter()
                    .map(|&(slot, score)| (slot, (score - largest).exp()))
                    .collect();
                let denominator: f64 = weights.iter().map(|(_, weight)| weight).sum();
                assert!(
                    weights.iter().all(|(_, weight)| *weight >= 2f64.powi(-14)),
                    "fixture softmax weights must stay in the normal half range for its roundoff bound"
                );
                for dim in 0..WIDTH {
                    let attended: f64 = weights
                        .iter()
                        .map(|&(slot, weight)| {
                            weight * number(values[input_base + slot * WIDTH + dim])
                        })
                        .sum::<f64>()
                        / denominator;
                    let expected = attended / (1.0 + (-number(gates[query_base + dim])).exp());
                    let word = actual[query_base + dim];
                    let error = (number(word) - expected).abs();
                    let rounded = bf16::from_f32(expected.abs() as f32).to_bits();
                    let ulp = number(rounded + 1) - number(rounded);
                    // BF16 inputs in this fixture are exactly representable in
                    // half after query scaling. The MM path rounds positive
                    // softmax weights to half before its value product: one
                    // half unit roundoff bounds their weighted-average error.
                    let half_rounding = if path == 0 { max_value / 2048.0 } else { 0.0 };
                    let fp32_accumulation =
                        f64::from(f32::EPSILON) * counts[local_row] as f64 * max_value;
                    let allowance = ulp + half_rounding + fp32_accumulation;
                    maximum[path] = maximum[path].max(error);
                    squared[path] += error * error;
                    maximum_ratio[path] = maximum_ratio[path].max(error / allowance);
                    samples[path] += 1;
                    mismatched[path] +=
                        usize::from(word != bf16::from_f32(expected as f32).to_bits());
                    assert!(
                        number(word).is_finite() && error <= allowance,
                        "QSA path={path} row={row} head={head} dim={dim}: actual={} reference={expected} error={error} allowance={allowance}",
                        number(word)
                    );
                }
            }
        }
    }
    for path in 0..2 {
        eprintln!(
            "prefill QSA path={} max_abs={:.8} rms={:.8} max_allowance_ratio={:.4} mismatched_bf16_words={}",
            if path == 0 { "MM" } else { "short-online" },
            maximum[path],
            (squared[path] / samples[path] as f64).sqrt(),
            maximum_ratio[path],
            mismatched[path]
        );
    }
    Ok(())
}
