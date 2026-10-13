//! Load declared dense and raw parameters without expanding quantized weights.

use super::{Package, checked_mul, read_header};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

#[derive(Debug, Deserialize)]
pub(super) struct DirectEntry {
    pub module_path: String,
    role: String,
    format: String,
    format_params: Value,
    key_prefix: String,
    shard: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WeightFormat {
    Bf16,
    Q6K,
    Q8_0,
}

/// A parameter in its package format; quantized matrices stay packed.
pub struct DirectWeight {
    pub role: String,
    pub format: WeightFormat,
    /// Logical dimensions. Raw tensors retain their declared storage axis order.
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

/// Validated package geometry for one direct parameter, without its payload.
#[derive(Clone, Debug)]
pub struct DirectHeader {
    pub module_path: String,
    pub role: String,
    pub format: WeightFormat,
    pub shape: Vec<usize>,
    pub bytes: usize,
}

struct ResolvedHeader {
    direct: DirectHeader,
    offset: u64,
}

#[derive(Deserialize)]
struct ArrayHeader {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

impl Package {
    fn open_direct_header(&self, shard: &str) -> Result<(File, Value, u64, u64)> {
        let (path, size) = self.declared_file(shard)?;
        let mut file = File::open(path)?;
        ensure!(
            file.metadata()?.len() == size,
            "parameter shard size differs from manifest"
        );
        let (header, data_start) = read_header(&mut file)?;
        Ok((file, header, data_start, size))
    }

    /// Inspect every direct parameter through shard headers only. The result
    /// includes the packed matrix bytes actually loaded by `direct_weight`.
    pub fn direct_headers(&self) -> Result<Vec<DirectHeader>> {
        let mut shards: BTreeMap<&str, (Value, u64, u64)> = BTreeMap::new();
        let mut result = Vec::with_capacity(self.direct.len());
        for entry in self.direct.values() {
            if !shards.contains_key(entry.shard.as_str()) {
                let (_file, header, data_start, size) = self.open_direct_header(&entry.shard)?;
                shards.insert(&entry.shard, (header, data_start, size));
            }
            let (header, data_start, size) = &shards[entry.shard.as_str()];
            result.push(resolve_header(entry, header, *data_start, *size)?.direct);
        }
        Ok(result)
    }

    pub fn direct_weight(&self, module_path: &str) -> Result<DirectWeight> {
        let entry = self
            .direct
            .get(module_path)
            .with_context(|| format!("no declared parameter {module_path}"))?;
        let (mut file, header, data_start, size) = self.open_direct_header(&entry.shard)?;
        let resolved = resolve_header(entry, &header, data_start, size)?;
        file.seek(SeekFrom::Start(resolved.offset))?;
        let mut payload = vec![0; resolved.direct.bytes];
        file.read_exact(&mut payload)?;
        Ok(DirectWeight {
            role: resolved.direct.role,
            format: resolved.direct.format,
            shape: resolved.direct.shape,
            bytes: payload,
        })
    }
}

fn resolve_header(
    entry: &DirectEntry,
    header: &Value,
    data_start: u64,
    size: u64,
) -> Result<ResolvedHeader> {
    let format = weight_format(entry)?;
    let key = if format == WeightFormat::Bf16 {
        entry.key_prefix.clone()
    } else {
        format!("{}.weight", entry.key_prefix)
    };
    let array: ArrayHeader = serde_json::from_value(
        header
            .get(&key)
            .with_context(|| format!("missing declared parameter key {key}"))?
            .clone(),
    )?;
    let (shape, bytes) = array_geometry(format, &array)?;
    let length = array.data_offsets[1]
        .checked_sub(array.data_offsets[0])
        .context("reversed parameter range")?;
    ensure!(length == bytes as u64, "parameter range differs from shape");
    let end = data_start
        .checked_add(array.data_offsets[1])
        .context("parameter range overflow")?;
    ensure!(end <= size, "parameter exceeds shard length");
    // Scales are embedded in Q6_K/Q8_0 blocks. The separate one-byte scales
    // tensor is a package interface placeholder and is not loaded by runtime.
    if format != WeightFormat::Bf16 {
        let scales_key = format!("{}.scales", entry.key_prefix);
        let scales: ArrayHeader = serde_json::from_value(
            header
                .get(&scales_key)
                .context("missing quantized scales placeholder")?
                .clone(),
        )?;
        ensure!(
            scales.dtype == "U8"
                && scales.shape == [1]
                && scales.data_offsets[1].checked_sub(scales.data_offsets[0]) == Some(1)
                && data_start
                    .checked_add(scales.data_offsets[1])
                    .is_some_and(|v| v <= size),
            "unexpected quantized scales placeholder"
        );
    }
    let offset = data_start
        .checked_add(array.data_offsets[0])
        .context("parameter offset overflow")?;
    Ok(ResolvedHeader {
        direct: DirectHeader {
            module_path: entry.module_path.clone(),
            role: entry.role.clone(),
            format,
            shape,
            bytes,
        },
        offset,
    })
}

fn weight_format(entry: &DirectEntry) -> Result<WeightFormat> {
    match entry.format.as_str() {
        "raw_dtype_passthrough" => Ok(WeightFormat::Bf16),
        "kquant" => {
            let p = &entry.format_params;
            let (format, bits, group, bytes) = match p.get("kquant_codec").and_then(Value::as_str) {
                Some("q6_k") => (WeightFormat::Q6K, 6, 256, 210),
                Some("q8_0") => (WeightFormat::Q8_0, 8, 32, 34),
                _ => anyhow::bail!("unsupported dense quantization codec"),
            };
            for (field, expected) in [
                ("bits", bits),
                ("group_size", group),
                ("weights_per_block", group),
                ("bytes_per_block", bytes),
            ] {
                ensure!(
                    p.get(field).and_then(Value::as_u64) == Some(expected),
                    "invalid dense quantization field {field}"
                );
            }
            Ok(format)
        }
        _ => anyhow::bail!("unsupported direct parameter format {}", entry.format),
    }
}

fn array_geometry(format: WeightFormat, array: &ArrayHeader) -> Result<(Vec<usize>, usize)> {
    ensure!(
        !array.shape.is_empty() && array.shape.iter().all(|&n| n > 0),
        "empty parameter shape"
    );
    let elements = array.shape.iter().try_fold(1, |n, &d| checked_mul(n, d))?;
    match format {
        WeightFormat::Bf16 => {
            ensure!(array.dtype == "BF16", "raw parameter is not BF16");
            Ok((array.shape.clone(), checked_mul(elements, 2)?))
        }
        WeightFormat::Q6K | WeightFormat::Q8_0 => {
            ensure!(
                array.dtype == "U8" && array.shape.len() == 2,
                "invalid packed matrix shape/dtype"
            );
            let (block_bytes, block_weights) = if format == WeightFormat::Q6K {
                (210, 256)
            } else {
                (34, 32)
            };
            ensure!(
                array.shape[1].is_multiple_of(block_bytes),
                "partial quantized block"
            );
            let input = checked_mul(array.shape[1] / block_bytes, block_weights)?;
            Ok((vec![array.shape[0], input], elements))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn packed_shape_resolves_to_logical_matrix_without_dequantizing() {
        let array = ArrayHeader {
            dtype: "U8".into(),
            shape: vec![48, 2100],
            data_offsets: [0, 100800],
        };
        assert_eq!(
            array_geometry(WeightFormat::Q6K, &array).unwrap(),
            (vec![48, 2560], 100800)
        );
        assert!(array_geometry(WeightFormat::Q8_0, &array).is_err());
    }

    #[test]
    fn raw_dtype_and_size_are_checked() {
        let mut array = ArrayHeader {
            dtype: "BF16".into(),
            shape: vec![4, 16],
            data_offsets: [0, 128],
        };
        assert_eq!(array_geometry(WeightFormat::Bf16, &array).unwrap().1, 128);
        array.dtype = "F16".into();
        assert!(array_geometry(WeightFormat::Bf16, &array).is_err());
    }

    #[test]
    fn direct_headers_validate_packed_geometry_without_loading_payload() {
        let root = std::env::temp_dir().join(format!(
            "moespresso-direct-header-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let header = serde_json::json!({
            "layer.weight": {"dtype": "U8", "shape": [2, 34], "data_offsets": [0, 68]},
            "layer.scales": {"dtype": "U8", "shape": [1], "data_offsets": [68, 69]},
        });
        let encoded = serde_json::to_vec(&header).unwrap();
        let path = root.join("direct.safetensors");
        let mut file = File::create(&path).unwrap();
        file.write_all(&(encoded.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&encoded).unwrap();
        let size = 8 + encoded.len() as u64 + 69;
        file.set_len(size).unwrap();
        drop(file);
        let entry = DirectEntry {
            module_path: "layers.0.matrix".into(),
            role: "matrix".into(),
            format: "kquant".into(),
            format_params: serde_json::json!({
                "kquant_codec": "q8_0", "bits": 8, "group_size": 32,
                "weights_per_block": 32, "bytes_per_block": 34,
            }),
            key_prefix: "layer".into(),
            shard: "direct.safetensors".into(),
        };
        let package = Package {
            root: root.clone(),
            artifact_id: "synthetic".into(),
            manifest_digest: [0; 32],
            files: BTreeMap::from([("direct.safetensors".into(), size)]),
            experts: Vec::new(),
            direct: BTreeMap::from([(entry.module_path.clone(), entry)]),
            architecture: Value::Null,
            tokenizer: Value::Null,
            ple_provider: None,
        };
        let headers = package.direct_headers().unwrap();
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].module_path, "layers.0.matrix");
        assert_eq!(headers[0].format, WeightFormat::Q8_0);
        assert_eq!(headers[0].shape, [2, 32]);
        assert_eq!(headers[0].bytes, 68);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn direct_header_rejects_invalid_scales_placeholder() {
        let entry = DirectEntry {
            module_path: "matrix".into(),
            role: "matrix".into(),
            format: "kquant".into(),
            format_params: serde_json::json!({
                "kquant_codec": "q8_0", "bits": 8, "group_size": 32,
                "weights_per_block": 32, "bytes_per_block": 34,
            }),
            key_prefix: "layer".into(),
            shard: "unused".into(),
        };
        let header = serde_json::json!({
            "layer.weight": {"dtype": "U8", "shape": [2, 34], "data_offsets": [0, 68]},
            "layer.scales": {"dtype": "U8", "shape": [2], "data_offsets": [68, 70]},
        });
        assert!(resolve_header(&entry, &header, 0, 70).is_err());
    }
}
