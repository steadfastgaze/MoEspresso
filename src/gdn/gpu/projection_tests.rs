use super::input_front::{StackedFront, WIDTHS};
use super::*;
use half::f16;

const GUARD: [u8; 16] = [0x5a; 16];

fn row_scale(projection: usize, row: usize, rows: usize) -> i8 {
    let magnitude = if row + 1 == rows {
        17 + projection as i8
    } else {
        (row % 11 + 1) as i8 + projection as i8
    };
    if projection.is_multiple_of(2) {
        magnitude
    } else {
        -magnitude
    }
}

fn constant_q6(projection: usize, rows: usize) -> DirectWeight {
    let mut bytes = Vec::with_capacity(rows * HIDDEN / 256 * 210);
    for row in 0..rows {
        let mut block = [0u8; 210];
        block[..128].fill(0x11);
        block[128..192].fill(0xaa);
        block[192..208].fill(row_scale(projection, row, rows) as u8);
        block[208..210].copy_from_slice(&f16::from_f32(1.0 / 256.0).to_bits().to_le_bytes());
        for _ in 0..HIDDEN / 256 {
            bytes.extend_from_slice(&block);
        }
    }
    DirectWeight {
        role: "synthetic GDN input projection".into(),
        format: WeightFormat::Q6K,
        shape: vec![rows, HIDDEN],
        bytes,
    }
}

fn signed_input(case: usize) -> Vec<u16> {
    (0..HIDDEN)
        .map(|index| {
            let magnitude = ((index + case * 5) % 11 + 1) as f32 / 32.0;
            let value = if (index + case).is_multiple_of(3) {
                -magnitude
            } else {
                magnitude
            };
            bf16::from_f32(value).to_bits()
        })
        .collect()
}

fn input_buffer(metal: &Metal, words: &[u16]) -> Result<Buffer> {
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    finite_bf16(&bytes, HIDDEN, "GDN projection input")?;
    uploaded(metal, &bytes)
}

fn guarded_output(metal: &Metal, rows: usize) -> Result<Buffer> {
    uploaded(metal, &vec![GUARD[0]; rows * 2 + GUARD.len()])
}

fn guarded_outputs(metal: &Metal) -> Result<[Buffer; 4]> {
    Ok([
        guarded_output(metal, CONV)?,
        guarded_output(metal, VALUE)?,
        guarded_output(metal, HEADS)?,
        guarded_output(metal, HEADS)?,
    ])
}

fn projection_values(outputs: &[Buffer; 4]) -> Result<Vec<u16>> {
    let mut values = Vec::with_capacity(WIDTHS.iter().sum());
    for (output, rows) in outputs.iter().zip(WIDTHS) {
        let mut bytes = vec![0; rows * 2];
        output.read_bytes(0, &mut bytes)?;
        finite_bf16(&bytes, rows, "completed GDN projection")?;
        values.extend(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|word| u16::from_le_bytes(*word)),
        );
        let mut guard = [0; GUARD.len()];
        output.read_bytes(rows * 2, &mut guard)?;
        ensure!(guard == GUARD, "GDN projection changed its output guard");
    }
    Ok(values)
}

fn check_synthetic(words: &[u16], values: &[u16]) -> Result<()> {
    ensure!(
        values.len() == WIDTHS.iter().sum::<usize>(),
        "GDN projection output widths differ"
    );
    let sum: f64 = words
        .iter()
        .map(|&word| f64::from(bf16::from_bits(word).to_f32()))
        .sum();
    ensure!(sum != 0.0, "signed GDN projection input sums to zero");
    let mut offset = 0;
    for (projection, rows) in WIDTHS.into_iter().enumerate() {
        for row in 0..rows {
            // Each packed value is q=1. The row scale and FP16 block scale
            // give a dot product independent of the SIMD projection fold.
            let expected =
                bf16::from_f32((sum * f64::from(row_scale(projection, row, rows)) / 256.0) as f32)
                    .to_bits();
            ensure!(
                values[offset + row] == expected && bf16::from_bits(expected).to_f32() != 0.0,
                "GDN projection {projection} row {row} differs from its independent dot product"
            );
        }
        offset += rows;
    }
    Ok(())
}

#[test]
fn input_projection_front_preserves_signed_rows_and_output_guards() -> Result<()> {
    let metal = Metal::new()?;
    let projections = [
        DenseMatvec::new(&metal, &constant_q6(0, CONV))?,
        DenseMatvec::new(&metal, &constant_q6(1, VALUE))?,
        DenseMatvec::new(&metal, &constant_q6(2, HEADS))?,
        DenseMatvec::new(&metal, &constant_q6(3, HEADS))?,
    ];
    let words = signed_input(0);
    let hidden = input_buffer(&metal, &words)?;
    let separate = guarded_outputs(&metal)?;
    let mut batch = metal.batch()?;
    for (projection, output) in projections.iter().zip(&separate) {
        // SAFETY: Valid Q6 rows, finite BF16 input and distinct guarded
        // outputs remain retained by the command through completion.
        unsafe { projection.encode(&mut batch, &hidden, output, OutputPrecision::Bf16)? };
    }
    batch.submit().wait()?;
    let reference = projection_values(&separate)?;
    check_synthetic(&words, &reference)?;
    eprintln!("established GDN input projections completed with exact signed rows and guards");

    let front = StackedFront::new(&metal, projections.each_ref())?;
    let stacked = guarded_outputs(&metal)?;
    let mut batch = metal.batch()?;
    front.encode(&mut batch, &hidden, stacked.each_ref())?;
    batch.submit().wait()?;
    let candidate = projection_values(&stacked)?;
    check_synthetic(&words, &candidate)?;
    ensure!(candidate == reference, "stacked GDN projections differ");

    let mut batch = metal.batch()?;
    let short_hidden = metal.shared_buffer(HIDDEN * 2 - 2)?;
    ensure!(
        front
            .encode(&mut batch, &short_hidden, stacked.each_ref())
            .is_err(),
        "GDN input front accepted a short hidden row"
    );
    let mut invalid = stacked.clone();
    invalid[0] = metal.shared_buffer(CONV * 2 - 2)?;
    ensure!(
        front
            .encode(&mut batch, &hidden, invalid.each_ref())
            .is_err(),
        "GDN input front accepted a short output"
    );
    invalid = stacked.clone();
    invalid[3] = invalid[2].clone();
    ensure!(
        front
            .encode(&mut batch, &hidden, invalid.each_ref())
            .is_err(),
        "GDN input front accepted overlapping outputs"
    );
    invalid = stacked.clone();
    invalid[2] = hidden.clone();
    ensure!(
        front
            .encode(&mut batch, &hidden, invalid.each_ref())
            .is_err(),
        "GDN input front accepted an output aliasing hidden"
    );
    invalid = stacked.clone();
    invalid[0] = front.matrices[0].binding().buffer.clone();
    ensure!(
        front
            .encode(&mut batch, &hidden, invalid.each_ref())
            .is_err(),
        "GDN input front accepted an output aliasing a weight"
    );
    ensure!(
        StackedFront::new(
            &metal,
            [
                &projections[1],
                &projections[0],
                &projections[2],
                &projections[3]
            ],
        )
        .is_err(),
        "GDN input front accepted swapped role geometry"
    );
    front.encode(&mut batch, &hidden, stacked.each_ref())?;
    ensure!(
        hidden.read_bytes(0, &mut []).is_err()
            && stacked[0].read_bytes(0, &mut []).is_err()
            && front.matrices[0]
                .binding()
                .buffer
                .write_bytes(0, &[])
                .is_err(),
        "GDN input front exposed a retained allocation before completion"
    );
    batch.submit().wait()?;
    ensure!(
        projection_values(&stacked)? == reference,
        "GDN projections changed after rejected invalid bindings"
    );
    Ok(())
}
