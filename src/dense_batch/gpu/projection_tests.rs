// Test-only component comparisons. Production dispatch is not selectable.
use super::*;
use half::f16;

#[test]
fn completed_quantized_projection_methods_preserve_rows_and_guards() -> Result<()> {
    let metal = Metal::new()?;
    for format in [WeightFormat::Q6K, WeightFormat::Q8_0] {
        for columns in [1, 9] {
            let mut bytes = Vec::new();
            for column in 0..columns {
                let scale = (column + 1) as i8;
                match format {
                    WeightFormat::Q6K => {
                        let mut block = [0u8; 210];
                        // Reconstructs scale / 256 at every input column.
                        block[..128].fill(0x11);
                        block[128..192].fill(0xaa);
                        block[192..208].fill(scale as u8);
                        block[208..].copy_from_slice(&f16::from_f32(1.0 / 256.0).to_le_bytes());
                        bytes.extend_from_slice(&block);
                    }
                    WeightFormat::Q8_0 => {
                        for _ in 0..8 {
                            bytes.extend_from_slice(&f16::from_f32(1.0 / 256.0).to_le_bytes());
                            bytes.extend_from_slice(&[scale as u8; 32]);
                        }
                    }
                    WeightFormat::Bf16 => unreachable!(),
                }
            }
            let projection = DenseBatch::new(
                &metal,
                &DirectWeight {
                    role: "synthetic quantized projection".into(),
                    format,
                    shape: vec![columns, 256],
                    bytes,
                },
            )?;
            let scratch = DenseBatchScratch::new(&metal, 4)?;
            for rows in [17, 41, 64] {
                let input = metal.shared_buffer(rows * 256 * 2)?;
                input.write_bytes(
                    0,
                    &(0..rows)
                        .flat_map(|row| {
                            let value = bf16::from_f32((row as f32 - 32.0) / 64.0).to_bits();
                            (0..256).flat_map(move |_| value.to_le_bytes())
                        })
                        .collect::<Vec<_>>(),
                )?;
                let output_bytes = rows * columns * 2;
                let output = metal.shared_buffer(output_bytes + 64)?;
                for method in [DenseBatchMethod::Matmul, DenseBatchMethod::Matvec] {
                    output.write_bytes(0, &vec![0xa5; output_bytes + 64])?;
                    let mut command = metal.batch()?;
                    // SAFETY: Finite BF16 rows, distinct guarded output and ordered scratch.
                    unsafe {
                        projection.test_encode_quantized(
                            &mut command,
                            &input,
                            &output,
                            rows,
                            method,
                            &scratch,
                        )?;
                    }
                    command.submit().wait()?;
                    let mut actual = vec![0; output_bytes + 64];
                    output.read_bytes(0, &mut actual)?;
                    assert_eq!(&actual[output_bytes..], &[0xa5; 64]);
                    for (index, word) in
                        actual[..output_bytes].as_chunks::<2>().0.iter().enumerate()
                    {
                        let expected = bf16::from_f32(
                            ((index / columns) as f32 - 32.0) / 64.0 * (index % columns + 1) as f32,
                        );
                        assert_eq!(
                            u16::from_le_bytes(*word),
                            expected.to_bits(),
                            "{format:?} {method:?} row/column {index}"
                        );
                    }
                }
                // Unsupported multi-vector rows and insufficient extents fail before submission.
                let mut command = metal.batch()?;
                assert!(
                    unsafe {
                        projection.test_encode_quantized(
                            &mut command,
                            &input,
                            &output,
                            rows,
                            DenseBatchMethod::MultiVector,
                            &scratch,
                        )
                    }
                    .is_err()
                );
                let short = metal.shared_buffer(2)?;
                assert!(
                    unsafe {
                        projection.test_encode_quantized(
                            &mut command,
                            &short,
                            &output,
                            rows,
                            DenseBatchMethod::Matvec,
                            &scratch,
                        )
                    }
                    .is_err()
                );
                assert!(
                    unsafe {
                        projection.test_encode_quantized(
                            &mut command,
                            &input,
                            &short,
                            rows,
                            DenseBatchMethod::Matmul,
                            &scratch,
                        )
                    }
                    .is_err()
                );
            }
        }
    }
    Ok(())
}
