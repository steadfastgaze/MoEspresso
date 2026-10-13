//! Optional cold-start expert ranking shipped beside the package manifest.

use super::Package;
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
pub struct ExpertHotlist {
    version: u32,
    kind: String,
    pub layers: BTreeMap<usize, BTreeMap<u32, u64>>,
}

impl ExpertHotlist {
    fn from_reader(reader: impl std::io::Read) -> Result<Self> {
        let hotlist: Self = serde_json::from_reader(reader)?;
        ensure!(
            hotlist.version == 1 && hotlist.kind == "expert_hotlist",
            "unsupported expert hotlist"
        );
        ensure!(
            hotlist
                .layers
                .iter()
                .all(|(&layer, counts)| layer < 48 && counts.keys().all(|&id| id < 512)),
            "hotlist layer or expert exceeds model geometry"
        );
        Ok(hotlist)
    }
}

impl Package {
    pub fn expert_hotlist(&self) -> Result<Option<ExpertHotlist>> {
        let path = self.root.join("expert_hotlist.json");
        let file = match std::fs::File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error).context("open package expert hotlist"),
        };
        Ok(Some(ExpertHotlist::from_reader(file)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn parse(value: &Value) -> Result<ExpertHotlist> {
        ExpertHotlist::from_reader(serde_json::to_vec(value)?.as_slice())
    }

    fn fixture() -> Value {
        json!({"version": 1, "kind": "expert_hotlist", "layers": {
            "0": {"0": 0, "511": 123}, "47": {"3": 8}
        }})
    }

    #[test]
    fn accepts_sparse_boundary_ids_and_preserves_exact_counts() {
        let hotlist = parse(&fixture()).unwrap();
        assert_eq!(hotlist.layers.len(), 2);
        assert_eq!(hotlist.layers[&0][&0], 0);
        assert_eq!(hotlist.layers[&0][&511], 123);
        assert_eq!(hotlist.layers[&47][&3], 8);
        let mut empty = fixture();
        empty["layers"] = json!({});
        assert!(parse(&empty).unwrap().layers.is_empty());
        let mut annotated = fixture();
        annotated["source"] = json!({"kind": "synthetic-test-counts"});
        assert!(parse(&annotated).is_ok());
    }

    #[test]
    fn rejects_unsupported_schema_missing_fields_and_malformed_json() {
        for (field, invalid) in [
            ("version", json!(2)),
            ("version", json!("1")),
            ("kind", json!("other")),
            ("layers", Value::Null),
            ("layers", json!([])),
        ] {
            let mut value = fixture();
            value[field] = invalid;
            assert!(parse(&value).is_err(), "accepted invalid {field}");
        }
        for field in ["version", "kind", "layers"] {
            let mut value = fixture();
            value.as_object_mut().unwrap().remove(field);
            assert!(parse(&value).is_err(), "accepted missing {field}");
        }
        assert!(ExpertHotlist::from_reader(b"{\"version\":1".as_slice()).is_err());
        assert!(ExpertHotlist::from_reader(b"null".as_slice()).is_err());
    }

    #[test]
    fn rejects_out_of_range_ids_and_noninteger_or_negative_counts() {
        for layers in [
            json!({"48": {"0": 1}}),
            json!({"-1": {"0": 1}}),
            json!({"first": {"0": 1}}),
            json!({"0": {"512": 1}}),
            json!({"0": {"-1": 1}}),
            json!({"0": {"expert": 1}}),
            json!({"0": {"0": -1}}),
            json!({"0": {"0": 1.5}}),
            json!({"0": {"0": true}}),
            json!({"0": {"0": "1"}}),
            json!({"0": {"0": null}}),
            json!({"0": []}),
        ] {
            let mut value = fixture();
            value["layers"] = layers.clone();
            assert!(parse(&value).is_err(), "accepted malformed layers {layers}");
        }
    }

    struct TemporaryPackage(Package);

    impl TemporaryPackage {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let root = std::env::temp_dir().join(format!(
                "moespresso-hotlist-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&root).unwrap();
            Self(Package {
                root,
                artifact_id: "synthetic-hotlist-test".into(),
                manifest_digest: [0; 32],
                files: BTreeMap::new(),
                experts: Vec::new(),
                direct: BTreeMap::new(),
                architecture: Value::Null,
                tokenizer: Value::Null,
                ple_provider: None,
            })
        }
    }

    impl Drop for TemporaryPackage {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(self.0.root.join("expert_hotlist.json"));
            let _ = std::fs::remove_dir(&self.0.root);
        }
    }

    #[test]
    fn optional_sidecar_distinguishes_absence_valid_data_and_invalid_data() {
        let package = TemporaryPackage::new();
        assert!(package.0.expert_hotlist().unwrap().is_none());
        let path = package.0.root.join("expert_hotlist.json");
        std::fs::write(&path, serde_json::to_vec(&fixture()).unwrap()).unwrap();
        assert_eq!(
            package.0.expert_hotlist().unwrap().unwrap().layers[&0][&511],
            123
        );
        std::fs::write(&path, b"{}").unwrap();
        assert!(package.0.expert_hotlist().is_err());
    }
}
