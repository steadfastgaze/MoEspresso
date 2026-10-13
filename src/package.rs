//! Manifest-driven byte ranges for routed expert weights.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

mod direct;
mod hotlist;
pub use direct::{DirectHeader, DirectWeight, WeightFormat};
pub use hotlist::ExpertHotlist;

const MAX_HEADER_BYTES: u64 = 16 * 1024 * 1024;
const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;

pub(crate) const PREFILL_READ_STACK_BYTES: usize = 256 * 1024;
pub(crate) const PREFILL_READ_CONTROL_BYTES: usize = 32 * 1024;
pub(crate) const PREFILL_READ_WORKERS: usize = 4;
pub(crate) const PREFILL_READ_ROWS: usize = 4;

#[cfg(any(target_os = "macos", all(test, unix)))]
mod prefill_reader;
#[cfg(any(target_os = "macos", all(test, unix)))]
pub(crate) use prefill_reader::ExpertRowWorkspace;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Projection {
    Gate,
    Up,
    Down,
}

impl Projection {
    pub const ALL: [Self; 3] = [Self::Gate, Self::Up, Self::Down];

    pub fn name(self) -> &'static str {
        match self {
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
        }
    }

    fn bundle_name(self) -> &'static str {
        match self {
            Self::Gate => "gate_proj",
            Self::Up => "up_proj",
            Self::Down => "down_proj",
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Gate => 0,
            Self::Up => 1,
            Self::Down => 2,
        }
    }
}

#[derive(Debug, Deserialize)]
struct Manifest {
    schema_version: Version,
    artifact_kind: String,
    artifact_id: String,
    package_format: String,
    package_format_version: u32,
    required_features: Vec<String>,
    files: Vec<PackageFile>,
    tensors: Vec<Value>,
    architecture: Value,
    tokenizer: Value,
    #[serde(default)]
    ple_provider: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct Version {
    major: u32,
}

#[derive(Debug, Deserialize)]
struct PackageFile {
    path: String,
    size_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct ExpertTensor {
    layer_index: usize,
    projection: String,
    format: String,
    format_params: Format,
    key_prefix: String,
    shard: String,
}

#[derive(Debug, Deserialize)]
struct Format {
    iqk_codec: String,
    layout: String,
    logical_shape: [usize; 2],
    stored_shape: [usize; 2],
    bits: usize,
    ggml_type: usize,
    bytes_per_block: usize,
    weights_per_block: usize,
    row_meta_bytes: usize,
    zero_padding: usize,
}

/// Package inventory. Opening this inventory does not read or hash model payloads.
pub struct Package {
    root: PathBuf,
    pub artifact_id: String,
    manifest_digest: [u8; 32],
    files: BTreeMap<String, u64>,
    experts: Vec<ExpertTensor>,
    direct: BTreeMap<String, direct::DirectEntry>,
    architecture: Value,
    tokenizer: Value,
    ple_provider: Option<Value>,
}

impl Package {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        let mut manifest_bytes = Vec::new();
        File::open(root.join("package_manifest.json"))
            .context("open package manifest")?
            .take(MAX_MANIFEST_BYTES + 1)
            .read_to_end(&mut manifest_bytes)?;
        ensure!(
            u64::try_from(manifest_bytes.len())? <= MAX_MANIFEST_BYTES,
            "package manifest exceeds the read limit"
        );
        let manifest_digest: [u8; 32] = Sha256::digest(&manifest_bytes).into();
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
        ensure!(
            manifest.schema_version.major == 1,
            "unsupported manifest schema"
        );
        ensure!(
            manifest.artifact_kind == "package_manifest",
            "wrong artifact kind"
        );
        ensure!(
            manifest.package_format == "mjtq" && manifest.package_format_version == 1,
            "unsupported package format"
        );
        ensure!(
            manifest
                .required_features
                .iter()
                .all(|v| v == "calibration"),
            "unsupported required package feature"
        );
        let mut files = BTreeMap::new();
        for file in manifest.files {
            relative_path(&file.path)?;
            ensure!(
                files.insert(file.path, file.size_bytes).is_none(),
                "duplicate package file"
            );
        }
        let mut experts = Vec::new();
        let mut direct = BTreeMap::new();
        for tensor in manifest.tensors {
            if tensor.get("kind").and_then(Value::as_str) == Some("expert") {
                experts.push(serde_json::from_value(tensor)?);
            } else {
                let entry: direct::DirectEntry = serde_json::from_value(tensor)?;
                ensure!(
                    direct.insert(entry.module_path.clone(), entry).is_none(),
                    "duplicate direct parameter module"
                );
            }
        }
        ensure!(!experts.is_empty(), "package has no routed experts");
        Ok(Self {
            root: root.to_owned(),
            artifact_id: manifest.artifact_id,
            manifest_digest,
            files,
            experts,
            direct,
            architecture: manifest.architecture,
            tokenizer: manifest.tokenizer,
            ple_provider: manifest.ple_provider,
        })
    }

    /// Content identity of the manifest bytes, independent of package location.
    /// This does not independently verify the declared weight payloads.
    pub fn manifest_digest(&self) -> [u8; 32] {
        self.manifest_digest
    }

    /// Declared PLE storage contract, interpreted by the model's PLE reader.
    pub fn ple_provider(&self) -> Option<&Value> {
        self.ple_provider.as_ref()
    }

    pub fn architecture(&self) -> &Value {
        &self.architecture
    }

    /// Resolve one tokenizer asset declared separately from the weight files.
    pub fn tokenizer_file(&self, name: &str) -> Result<PathBuf> {
        ensure!(
            self.tokenizer["has_tokenizer"] == true,
            "package has no tokenizer"
        );
        let entries = self.tokenizer["files"]
            .as_array()
            .context("missing tokenizer inventory")?;
        let matches: Vec<_> = entries
            .iter()
            .filter(|entry| entry["path"].as_str() == Some(name))
            .collect();
        ensure!(
            matches.len() == 1,
            "tokenizer asset missing or duplicated: {name}"
        );
        let path = self.root.join(relative_path(name)?);
        let size = matches[0]["size_bytes"]
            .as_u64()
            .context("missing tokenizer asset size")?;
        ensure!(
            path.metadata()?.len() == size,
            "tokenizer asset size differs: {name}"
        );
        Ok(path)
    }

    /// Resolve a file explicitly listed by the package without reading it.
    pub fn declared_file(&self, path: &str) -> Result<(PathBuf, u64)> {
        let relative = relative_path(path)?;
        let size = *self
            .files
            .get(path)
            .context("file absent from package manifest")?;
        Ok((self.root.join(relative), size))
    }

    /// Resolve one layer's declared bundle, validating header geometry without loading weights.
    pub fn expert_layer(&self, layer: usize) -> Result<ExpertLayer> {
        let tensors: Vec<_> = self
            .experts
            .iter()
            .filter(|t| t.layer_index == layer)
            .collect();
        ensure!(
            tensors.len() == 3,
            "layer {layer} must declare three expert projections"
        );
        let first = tensors[0];
        ensure!(
            tensors
                .iter()
                .all(|t| t.shard == first.shard && t.key_prefix == first.key_prefix),
            "expert projections must share one declared bundle"
        );
        let path = self.root.join(relative_path(&first.shard)?);
        let expected_size = self
            .files
            .get(&first.shard)
            .context("shard absent from manifest files")?;
        let mut file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let file_metadata = file.metadata()?;
        ensure!(
            file_metadata.len() == *expected_size,
            "shard size differs from manifest"
        );
        let (header, data_start) = read_header(&mut file)?;
        let metadata_text = header
            .get("__metadata__")
            .and_then(|v| v.get("expert_bundles"))
            .and_then(Value::as_str)
            .context("missing expert bundle metadata")?;
        let mut metadata: Bundles = serde_json::from_str(metadata_text)?;
        ensure!(metadata.version == 1, "unsupported expert bundle schema");
        let geometry = metadata
            .layers
            .remove(&layer.to_string())
            .context("missing layer geometry")?;
        let key = format!("{}.tq_bundle", first.key_prefix);
        let tensor: TensorHeader = serde_json::from_value(
            header
                .get(&key)
                .with_context(|| format!("missing declared tensor {key}"))?
                .clone(),
        )?;
        ensure!(
            tensor.dtype == "U8" && tensor.shape == [geometry.num_experts, geometry.row_bytes],
            "bundle tensor shape or dtype disagrees with metadata"
        );
        let payload_bytes = checked_mul(geometry.num_experts, geometry.row_bytes)?;
        let tensor_bytes = tensor.data_offsets[1]
            .checked_sub(tensor.data_offsets[0])
            .context("reversed tensor byte range")?;
        ensure!(
            tensor_bytes == payload_bytes as u64,
            "bundle byte range disagrees with shape"
        );
        let end = data_start
            .checked_add(tensor.data_offsets[1])
            .context("tensor range overflow")?;
        ensure!(end <= *expected_size, "bundle exceeds shard length");
        let projections = validate_geometry(&geometry, &tensors)?;
        Ok(ExpertLayer {
            file: Arc::new(file),
            #[cfg(unix)]
            file_identity: file_identity(&file_metadata),
            path,
            layer,
            num_experts: geometry.num_experts,
            row_bytes: geometry.row_bytes,
            data_start: data_start + tensor.data_offsets[0],
            projections,
        })
    }
}

#[derive(Debug, Deserialize)]
struct Bundles {
    version: u32,
    layers: BTreeMap<String, Geometry>,
}

#[derive(Debug, Deserialize)]
struct Geometry {
    num_experts: usize,
    row_bytes: usize,
    projections: BTreeMap<String, ProjectionMetadata>,
}

#[derive(Debug, Deserialize)]
struct ProjectionMetadata {
    codec: String,
    iqk_codec: String,
    layout: String,
    bits: usize,
    ggml_type: usize,
    bytes_per_block: usize,
    weights_per_block: usize,
    row_meta_bytes: usize,
    bytes_per_row: usize,
    in_features: usize,
    logical_in_features: usize,
    stored_in_features: usize,
    zero_padding: usize,
    blocks: Blocks,
}

#[derive(Debug, Deserialize)]
struct Blocks {
    offset: usize,
    nbytes: usize,
    shape: [usize; 2],
    dtype: String,
}

#[derive(Debug, Deserialize)]
struct TensorHeader {
    dtype: String,
    shape: [usize; 2],
    data_offsets: [u64; 2],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectionLayout {
    pub codec: String,
    pub out_features: usize,
    pub logical_in_features: usize,
    pub stored_in_features: usize,
    pub offset: usize,
    pub nbytes: usize,
}

impl ProjectionLayout {
    pub fn bytes<'a>(&self, row: &'a [u8]) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(self.nbytes)
            .context("projection range overflow")?;
        row.get(self.offset..end)
            .context("expert row is shorter than projection range")
    }
}

#[derive(Clone)]
pub struct ExpertLayer {
    // Bind positional reads to the descriptor whose header was validated.
    // Cloned metadata shares the descriptor without duplicating it.
    file: Arc<File>,
    #[cfg(unix)]
    file_identity: (u64, u64),
    path: PathBuf,
    pub layer: usize,
    pub num_experts: usize,
    pub row_bytes: usize,
    data_start: u64,
    projections: [ProjectionLayout; 3],
}

impl ExpertLayer {
    #[cfg(unix)]
    pub fn read_row_into(&self, expert: usize, bytes: &mut [u8]) -> Result<()> {
        read_expert_row(
            &self.file,
            self.num_experts,
            self.row_bytes,
            self.data_start,
            expert,
            bytes,
        )
    }
    pub(crate) fn same_source(&self, other: &Self) -> bool {
        let same_file = Arc::ptr_eq(&self.file, &other.file);
        #[cfg(unix)]
        let same_file = same_file || self.file_identity == other.file_identity;
        same_file
            && self.path == other.path
            && self.layer == other.layer
            && self.num_experts == other.num_experts
            && self.row_bytes == other.row_bytes
            && self.data_start == other.data_start
            && self.projections == other.projections
    }

    pub fn projection(&self, projection: Projection) -> &ProjectionLayout {
        &self.projections[projection.index()]
    }

    pub fn read_row(&self, expert: usize) -> Result<Vec<u8>> {
        ensure!(expert < self.num_experts, "expert ID out of range");
        let offset = self
            .data_start
            .checked_add(checked_mul(expert, self.row_bytes)? as u64)
            .context("expert offset overflow")?;
        let mut file = File::open(&self.path)?;
        #[cfg(unix)]
        ensure!(
            file_identity(&file.metadata()?) == self.file_identity,
            "expert bundle path no longer names the validated file"
        );
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0; self.row_bytes];
        file.read_exact(&mut bytes)
            .context("read complete expert row")?;
        Ok(bytes)
    }
}

#[cfg(unix)]
fn file_identity(metadata: &std::fs::Metadata) -> (u64, u64) {
    use std::os::unix::fs::MetadataExt;
    (metadata.dev(), metadata.ino())
}

#[cfg(all(test, unix))]
mod reader_tests;

#[cfg(unix)]
fn read_expert_row(
    file: &File,
    experts: usize,
    row_bytes: usize,
    data_start: u64,
    expert: usize,
    bytes: &mut [u8],
) -> Result<()> {
    use std::os::unix::fs::FileExt;
    ensure!(expert < experts, "expert ID out of range");
    ensure!(
        bytes.len() == row_bytes,
        "expert row destination length differs"
    );
    let offset = data_start
        .checked_add(checked_mul(expert, row_bytes)? as u64)
        .context("expert offset overflow")?;
    offset
        .checked_add(row_bytes as u64)
        .context("expert row end overflow")?;
    // Positional exact reads retry interruptions and short reads. A failed row
    // stays unpublished regardless of bytes already written into staging.
    file.read_exact_at(bytes, offset)
        .context("read complete expert row")
}

fn validate_geometry(g: &Geometry, tensors: &[&ExpertTensor]) -> Result<[ProjectionLayout; 3]> {
    ensure!(g.num_experts > 0 && g.row_bytes > 0, "empty expert bundle");
    ensure!(
        g.projections.len() == 3,
        "bundle must contain three projections"
    );
    let mut offset = 0;
    let mut layouts = Vec::new();
    for projection in Projection::ALL {
        let p = g
            .projections
            .get(projection.bundle_name())
            .context("missing bundle projection")?;
        let matches: Vec<_> = tensors
            .iter()
            .filter(|t| t.projection == projection.name())
            .collect();
        ensure!(
            matches.len() == 1,
            "missing or duplicate manifest projection"
        );
        let t = matches[0];
        let f = &t.format_params;
        let (bits, ggml_type, block_bytes) = match p.iqk_codec.as_str() {
            "iq2_k" => (2, 137, 76),
            "iq3_k" => (3, 138, 110),
            _ => anyhow::bail!("unsupported expert codec {}", p.iqk_codec),
        };
        ensure!(
            p.codec == "iqk" && t.format == "iqk" && p.iqk_codec == f.iqk_codec,
            "manifest and bundle codecs disagree"
        );
        ensure!(
            p.layout == "iqk_relayout" && f.layout == p.layout,
            "unsupported expert layout"
        );
        ensure!(
            (
                p.bits,
                p.ggml_type,
                p.bytes_per_block,
                p.weights_per_block,
                p.row_meta_bytes
            ) == (bits, ggml_type, block_bytes, 256, 0),
            "invalid bundle quantization geometry"
        );
        ensure!(
            (
                f.bits,
                f.ggml_type,
                f.bytes_per_block,
                f.weights_per_block,
                f.row_meta_bytes
            ) == (bits, ggml_type, block_bytes, 256, 0),
            "invalid manifest quantization geometry"
        );
        ensure!(
            p.stored_in_features > 0 && p.stored_in_features.is_multiple_of(256),
            "expert input width must contain whole quantization blocks"
        );
        let row_bytes = checked_mul(p.stored_in_features / 256, block_bytes)?;
        ensure!(
            p.blocks.shape[0] > 0
                && p.blocks.shape[1] == row_bytes
                && p.bytes_per_row == row_bytes
                && p.in_features == p.stored_in_features,
            "projection shape disagrees with quantization geometry"
        );
        ensure!(
            p.logical_in_features > 0
                && p.logical_in_features.checked_add(p.zero_padding) == Some(p.stored_in_features),
            "invalid expert padding"
        );
        ensure!(
            f.stored_shape == [p.blocks.shape[0], p.stored_in_features]
                && f.logical_shape == [p.blocks.shape[0], p.logical_in_features]
                && f.zero_padding == p.zero_padding,
            "manifest and bundle shapes disagree"
        );
        let nbytes = checked_mul(p.blocks.shape[0], row_bytes)?;
        ensure!(
            p.blocks.dtype == "U8" && p.blocks.offset == offset && p.blocks.nbytes == nbytes,
            "projection byte ranges must exactly tile the expert row"
        );
        layouts.push(ProjectionLayout {
            codec: p.iqk_codec.clone(),
            out_features: p.blocks.shape[0],
            logical_in_features: p.logical_in_features,
            stored_in_features: p.stored_in_features,
            offset,
            nbytes,
        });
        offset = offset.checked_add(nbytes).context("expert row overflow")?;
    }
    ensure!(
        offset == g.row_bytes,
        "projection ranges do not fill expert row"
    );
    Ok(layouts
        .try_into()
        .expect("three projections were validated"))
}

fn relative_path(name: &str) -> Result<&Path> {
    let path = Path::new(name);
    ensure!(
        !name.is_empty() && path.components().all(|c| matches!(c, Component::Normal(_))),
        "package file must be a relative path without parent traversal"
    );
    Ok(path)
}

fn checked_mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).context("package size overflow")
}

fn read_header(file: &mut File) -> Result<(Value, u64)> {
    let size = file.metadata()?.len();
    let mut prefix = [0; 8];
    file.read_exact(&mut prefix)?;
    let len = u64::from_le_bytes(prefix);
    ensure!(
        (2..=MAX_HEADER_BYTES).contains(&len),
        "invalid safetensors header length"
    );
    let data_start = 8 + len;
    ensure!(data_start <= size, "truncated safetensors header");
    let mut bytes = vec![0; len as usize];
    file.read_exact(&mut bytes)?;
    Ok((serde_json::from_slice(&bytes)?, data_start))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_identity_follows_contents_across_relocation() -> Result<()> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "moespresso-manifest-{}-{stamp}",
            std::process::id()
        ));
        std::fs::create_dir(&root)?;
        struct Cleanup(PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Cleanup(root.clone());
        let original = root.join("original");
        let relocated = root.join("relocated");
        std::fs::create_dir(&original)?;
        let mut manifest = serde_json::json!({
            "schema_version": {"major": 1},
            "artifact_kind": "package_manifest", "artifact_id": "fixture",
            "package_format": "mjtq", "package_format_version": 1,
            "required_features": [], "files": [], "architecture": {}, "tokenizer": {},
            "tensors": [{
                "kind": "expert", "layer_index": 0, "projection": "gate", "format": "iqk",
                "key_prefix": "layer.experts", "shard": "weights.safetensors",
                "format_params": {"iqk_codec": "iq2_k", "layout": "iqk_relayout",
                    "logical_shape": [2,240], "stored_shape": [2,256], "bits": 2,
                    "ggml_type": 137, "bytes_per_block": 76, "weights_per_block": 256,
                    "row_meta_bytes": 0, "zero_padding": 16}
            }]
        });
        std::fs::write(
            original.join("package_manifest.json"),
            serde_json::to_vec(&manifest)?,
        )?;
        let identity = Package::open(&original)?.manifest_digest();
        std::fs::rename(&original, &relocated)?;
        assert_eq!(identity, Package::open(&relocated)?.manifest_digest());
        manifest["architecture"] = serde_json::json!({"hidden_size": 256});
        std::fs::write(
            relocated.join("package_manifest.json"),
            serde_json::to_vec(&manifest)?,
        )?;
        assert_ne!(identity, Package::open(&relocated)?.manifest_digest());
        Ok(())
    }

    #[test]
    fn package_paths_cannot_escape_root() {
        for path in ["", "/tmp/model", "../model", "a/../model", "./model"] {
            assert!(relative_path(path).is_err(), "accepted {path}");
        }
        assert!(relative_path("weights/model.safetensors").is_ok());
    }

    #[test]
    fn projection_cannot_read_outside_row() {
        let p = ProjectionLayout {
            codec: "iq2_k".into(),
            out_features: 1,
            logical_in_features: 256,
            stored_in_features: 256,
            offset: 2,
            nbytes: 3,
        };
        assert_eq!(p.bytes(&[0, 1, 2, 3, 4]).unwrap(), [2, 3, 4]);
        assert!(p.bytes(&[0, 1, 2, 3]).is_err());
    }

    #[test]
    fn multiplication_rejects_overflow() {
        assert!(checked_mul(usize::MAX, 2).is_err());
    }

    fn fixture() -> (Geometry, Vec<ExpertTensor>) {
        let mut projections = serde_json::Map::new();
        let mut tensors = Vec::new();
        for (index, projection) in Projection::ALL.into_iter().enumerate() {
            projections.insert(
                projection.bundle_name().into(),
                serde_json::json!({
                    "codec": "iqk", "iqk_codec": "iq2_k", "layout": "iqk_relayout",
                    "bits": 2, "ggml_type": 137, "bytes_per_block": 76,
                    "weights_per_block": 256, "row_meta_bytes": 0, "bytes_per_row": 76,
                    "in_features": 256, "logical_in_features": 240,
                    "stored_in_features": 256, "zero_padding": 16,
                    "blocks": {"offset": index * 152, "nbytes": 152, "shape": [2,76], "dtype": "U8"}
                }),
            );
            tensors.push(
                serde_json::from_value(serde_json::json!({
                    "layer_index": 0, "projection": projection.name(), "format": "iqk",
                    "key_prefix": "layer.experts", "shard": "weights.safetensors",
                    "format_params": {"iqk_codec": "iq2_k", "layout": "iqk_relayout",
                        "logical_shape": [2,240], "stored_shape": [2,256], "bits": 2,
                        "ggml_type": 137, "bytes_per_block": 76, "weights_per_block": 256,
                        "row_meta_bytes": 0, "zero_padding": 16}
                }))
                .unwrap(),
            );
        }
        let geometry = serde_json::from_value(serde_json::json!({
            "num_experts": 4, "row_bytes": 456, "projections": projections
        }))
        .unwrap();
        (geometry, tensors)
    }

    #[test]
    fn bundle_geometry_matches_manifest_and_tiles_row() {
        let (geometry, tensors) = fixture();
        let refs: Vec<_> = tensors.iter().collect();
        let layouts = validate_geometry(&geometry, &refs).unwrap();
        assert_eq!(layouts[2].offset, 304);
        assert_eq!(layouts[2].logical_in_features, 240);
    }

    #[test]
    fn bundle_rejects_overlap_padding_and_codec_mismatches() {
        for case in 0..5 {
            let (mut geometry, mut tensors) = fixture();
            match case {
                0 => {
                    geometry
                        .projections
                        .get_mut("up_proj")
                        .unwrap()
                        .blocks
                        .offset -= 1
                }
                1 => {
                    geometry
                        .projections
                        .get_mut("down_proj")
                        .unwrap()
                        .zero_padding = 0
                }
                2 => tensors[0].format_params.ggml_type = 138,
                3 => tensors[1].projection = "gate".into(),
                4 => geometry.row_bytes += 1,
                _ => unreachable!(),
            }
            let refs: Vec<_> = tensors.iter().collect();
            assert!(
                validate_geometry(&geometry, &refs).is_err(),
                "accepted invalid case {case}"
            );
        }
    }
}
