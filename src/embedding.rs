//! Q8_0 token embedding lookup without expanding the embedding table.

use anyhow::{Context, Result, ensure};
use half::{bf16, f16};

use crate::package::{DirectWeight, WeightFormat};

pub struct Embedding {
    packed: Vec<u8>,
    vocabulary: usize,
    width: usize,
    row_bytes: usize,
}

impl Embedding {
    pub fn new(weight: DirectWeight) -> Result<Self> {
        ensure!(
            weight.format == WeightFormat::Q8_0 && weight.shape.len() == 2,
            "token embedding requires a Q8_0 matrix"
        );
        let vocabulary = weight.shape[0];
        let width = weight.shape[1];
        ensure!(
            vocabulary > 0 && width > 0 && width.is_multiple_of(32),
            "invalid Q8_0 embedding dimensions"
        );
        let row_bytes = (width / 32)
            .checked_mul(34)
            .context("embedding row size overflow")?;
        ensure!(
            vocabulary.checked_mul(row_bytes) == Some(weight.bytes.len()),
            "embedding byte count differs from matrix shape"
        );
        Ok(Self {
            packed: weight.bytes,
            vocabulary,
            width,
            row_bytes,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn vocabulary(&self) -> usize {
        self.vocabulary
    }

    /// Decode only the requested rows into caller-owned BF16 storage.
    pub fn lookup_into(&self, tokens: &[u32], output: &mut [u16]) -> Result<()> {
        ensure!(
            tokens.len().checked_mul(self.width) == Some(output.len()),
            "embedding output size differs from token count"
        );
        ensure!(
            tokens.iter().all(|&id| (id as usize) < self.vocabulary),
            "embedding token ID exceeds vocabulary"
        );
        for (&token, output_row) in tokens.iter().zip(output.chunks_exact_mut(self.width)) {
            let offset = token as usize * self.row_bytes;
            let row = &self.packed[offset..offset + self.row_bytes];
            for (block, out) in row
                .as_chunks::<34>()
                .0
                .iter()
                .zip(output_row.as_chunks_mut::<32>().0)
            {
                let scale = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
                for (&packed, value) in block[2..].iter().zip(out) {
                    // The package embedding path decodes to FP16 before the
                    // BF16 model boundary; retaining both rounds matters.
                    let decoded = f16::from_f32(scale * (packed as i8 as f32));
                    *value = bf16::from_f32(decoded.to_f32()).to_bits();
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example() -> Embedding {
        let mut bytes = Vec::new();
        for scale in [0.5, -0.125, 2.0] {
            bytes.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            bytes.extend((0..32).map(|v| (v as i8 - 16) as u8));
        }
        Embedding::new(DirectWeight {
            role: "embed_tokens".into(),
            format: WeightFormat::Q8_0,
            shape: vec![3, 32],
            bytes,
        })
        .unwrap()
    }

    #[test]
    fn token_order_and_repeated_rows_are_preserved() {
        let embedding = example();
        let mut output = [0; 96];
        embedding.lookup_into(&[2, 0, 2], &mut output).unwrap();
        assert_eq!(bf16::from_bits(output[0]).to_f32(), -32.0);
        assert_eq!(bf16::from_bits(output[32]).to_f32(), -8.0);
        assert_eq!(&output[..32], &output[64..]);
    }

    #[test]
    fn fp16_decode_rounding_precedes_bf16_output() {
        let mut bytes = vec![0x03, 0x10];
        bytes.extend([(-96_i8) as u8; 32]);
        let embedding = Embedding::new(DirectWeight {
            role: "embed_tokens".into(),
            format: WeightFormat::Q8_0,
            shape: vec![1, 32],
            bytes,
        })
        .unwrap();
        let mut output = [0; 32];
        embedding.lookup_into(&[0], &mut output).unwrap();
        assert_eq!(output, [0xbd40; 32]);
        let direct = bf16::from_f32(f16::from_bits(0x1003).to_f32() * -96.0).to_bits();
        assert_eq!(direct, 0xbd41);
    }

    #[test]
    fn invalid_token_leaves_destination_unchanged() {
        let embedding = example();
        let mut output = [0xabcd; 64];
        assert!(embedding.lookup_into(&[0, 3], &mut output).is_err());
        assert_eq!(output, [0xabcd; 64]);
        assert!(embedding.lookup_into(&[0], &mut output).is_err());
    }
}
