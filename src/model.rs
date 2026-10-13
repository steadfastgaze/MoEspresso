//! Resolve the package's declared Qwen execution order once at load time.

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use crate::package::Package;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixerKind {
    Gdn,
    Qsa,
}

pub struct QwenConfig {
    pub layers: Vec<MixerKind>,
    pub ple_layer_index: usize,
    pub eps: f32,
    pub eos_token_id: u32,
    pub max_positions: usize,
}

impl QwenConfig {
    pub fn from_package(package: &Package) -> Result<Self> {
        Self::parse(
            package.architecture(),
            package
                .ple_provider()
                .context("Qwen package requires PLE")?,
        )
    }

    fn parse(architecture: &Value, ple: &Value) -> Result<Self> {
        ensure!(
            architecture["family"] == "qwen4_exp" && architecture["modality"] == "text",
            "unsupported model family or modality"
        );
        let config = &architecture["config"];
        for (field, expected) in [
            ("hidden_size", 2560),
            ("hc_count", 4),
            ("hc_lowrank", 320),
            ("num_hidden_layers", 48),
            ("num_experts", 512),
            ("num_experts_per_tok", 10),
            ("moe_intermediate_size", 640),
            ("shared_expert_intermediate_size", 640),
            ("linear_num_key_heads", 16),
            ("linear_num_value_heads", 48),
            ("linear_key_head_dim", 128),
            ("linear_value_head_dim", 128),
            ("linear_conv_kernel_dim", 4),
            ("num_attention_heads", 24),
            ("num_key_value_heads", 2),
            ("head_dim", 256),
            ("indexer_head_dim", 128),
            ("indexer_n_heads", 4),
            ("indexer_kv_heads", 1),
            ("indexer_compress_ratio", 4),
            ("indexer_budget", 2048),
            ("vocab_size", 248320),
            ("ngram_size", 3),
            ("heads_per_ngram", 8),
            ("ple_embed_dim", 2560),
            ("ple_conv_kernel_size", 4),
            ("split_ngram_parts", 128),
        ] {
            ensure!(
                config[field].as_u64() == Some(expected),
                "unsupported Qwen field {field}"
            );
            if let Some(summary) = architecture.get(field) {
                ensure!(
                    summary == &config[field],
                    "architecture summary differs at {field}"
                );
            }
        }
        for (field, expected) in [
            ("dtype", "bfloat16"),
            ("mamba_ssm_dtype", "float32"),
            ("hidden_act", "silu"),
            ("output_gate_type", "sigmoid"),
            ("model_type", "qwen4_exp_text"),
        ] {
            ensure!(
                config[field].as_str() == Some(expected),
                "unsupported Qwen field {field}"
            );
        }
        ensure!(
            config["attention_bias"] == false && config["tie_word_embeddings"] == false,
            "unsupported attention bias or tied vocabulary weights"
        );
        ensure!(
            config
                .get("norm_topk_prob")
                .is_none_or(|value| value == true),
            "Qwen routing requires normalized selected probabilities"
        );
        ensure!(
            config["full_attention_interval"].as_u64() == Some(4),
            "unsupported QSA layer interval"
        );
        let rope = &config["rope_parameters"];
        ensure!(
            rope["rope_type"] == "default"
                && rope["rope_theta"].as_f64() == Some(10_000_000.0)
                && rope["partial_rotary_factor"].as_f64() == Some(0.25)
                && rope["mrope_interleaved"] == true
                && rope["mrope_section"] == serde_json::json!([11, 11, 10]),
            "unsupported Qwen rotary geometry"
        );
        let declared_layers = config["layer_types"]
            .as_array()
            .context("missing Qwen layer sequence")?;
        ensure!(
            declared_layers.len() == 48 && architecture["layer_types"] == config["layer_types"],
            "Qwen layer sequence length or summary differs"
        );
        let layers = declared_layers
            .iter()
            .map(|kind| match kind.as_str() {
                Some("linear_attention") => Ok(MixerKind::Gdn),
                Some("full_attention") => Ok(MixerKind::Qsa),
                _ => anyhow::bail!("unsupported Qwen mixer kind"),
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            layers.iter().enumerate().all(|(index, &kind)| kind
                == if index % 4 == 3 {
                    MixerKind::Qsa
                } else {
                    MixerKind::Gdn
                }),
            "unsupported Qwen layer order"
        );
        let eps = config["rms_norm_eps"]
            .as_f64()
            .context("missing norm epsilon")? as f32;
        ensure!(eps == 1e-6, "unsupported norm epsilon");
        let eos_token_id = u32::try_from(
            config["eos_token_id"]
                .as_u64()
                .context("missing EOS token")?,
        )?;
        ensure!(eos_token_id == 248044, "unsupported PLE EOS token");
        let max_positions = usize::try_from(
            config["max_position_embeddings"]
                .as_u64()
                .context("missing context limit")?,
        )?;
        ensure!(
            max_positions > 0 && max_positions <= 262144,
            "unsupported context limit"
        );
        let ple_ids = config["ple_layer_ids"]
            .as_array()
            .context("missing PLE layer IDs")?;
        ensure!(ple_ids.len() == 1, "exactly one PLE insertion is supported");
        let ple_layer_index = usize::try_from(
            ple_ids[0]
                .as_u64()
                .context("invalid PLE layer ID")?
                .checked_sub(1)
                .context("PLE config layer IDs are one-based")?,
        )?;
        ensure!(
            ple_layer_index == 1 && ple["layer_index"].as_u64() == Some(ple_layer_index as u64),
            "PLE requires one-based model layer 2 and zero-based provider layer 1"
        );
        Ok(Self {
            layers,
            ple_layer_index,
            eps,
            eos_token_id,
            max_positions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn architecture() -> Value {
        let layer_types: Vec<_> = (0..48)
            .map(|index| {
                if index % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect();
        let mut config: Value = serde_json::from_str(
            r#"{
                "hidden_size": 2560, "hc_count": 4, "hc_lowrank": 320,
                "num_hidden_layers": 48, "num_experts": 512, "num_experts_per_tok": 10,
                "moe_intermediate_size": 640, "shared_expert_intermediate_size": 640,
                "linear_num_key_heads": 16, "linear_num_value_heads": 48,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4, "num_attention_heads": 24,
                "num_key_value_heads": 2, "head_dim": 256,
                "indexer_head_dim": 128, "indexer_n_heads": 4, "indexer_kv_heads": 1,
                "indexer_compress_ratio": 4, "indexer_budget": 2048,
                "vocab_size": 248320, "ngram_size": 3, "heads_per_ngram": 8,
                "ple_embed_dim": 2560, "ple_conv_kernel_size": 4, "split_ngram_parts": 128,
                "dtype": "bfloat16", "mamba_ssm_dtype": "float32", "hidden_act": "silu",
                "output_gate_type": "sigmoid", "model_type": "qwen4_exp_text",
                "attention_bias": false, "tie_word_embeddings": false,
                "full_attention_interval": 4,
                "rope_parameters": {
                    "rope_type": "default", "rope_theta": 10000000,
                    "partial_rotary_factor": 0.25, "mrope_interleaved": true,
                    "mrope_section": [11, 11, 10]
                },
                "rms_norm_eps": 0.000001, "eos_token_id": 248044,
                "max_position_embeddings": 262144, "ple_layer_ids": [2]
            }"#,
        )
        .unwrap();
        config["layer_types"] = json!(layer_types);
        json!({
            "family": "qwen4_exp", "modality": "text",
            "layer_types": layer_types, "config": config
        })
    }

    fn parse(architecture: &Value) -> Result<QwenConfig> {
        QwenConfig::parse(architecture, &json!({"layer_index": 1}))
    }

    #[test]
    fn released_configuration_resolves_execution_order_and_one_based_ple() {
        let config = parse(&architecture()).unwrap();
        assert_eq!(config.layers.len(), 48);
        assert_eq!(
            &config.layers[..4],
            &[
                MixerKind::Gdn,
                MixerKind::Gdn,
                MixerKind::Gdn,
                MixerKind::Qsa
            ]
        );
        assert_eq!(
            config
                .layers
                .iter()
                .filter(|&&kind| kind == MixerKind::Qsa)
                .count(),
            12
        );
        assert_eq!(config.ple_layer_index, 1);
        assert_eq!(config.eps, 1e-6);
        assert_eq!(config.eos_token_id, 248044);
        assert_eq!(config.max_positions, 262144);
    }

    #[test]
    fn selected_probabilities_accept_default_or_true_and_reject_other_values() {
        let mut value = architecture();
        assert!(parse(&value).is_ok());
        value["config"]["norm_topk_prob"] = json!(true);
        assert!(parse(&value).is_ok());
        for invalid in [json!(false), Value::Null, json!(1), json!("true")] {
            value["config"]["norm_topk_prob"] = invalid.clone();
            assert!(parse(&value).is_err(), "accepted norm_topk_prob={invalid}");
        }
    }

    #[test]
    fn ple_rejects_zero_alternate_out_of_range_and_mismatched_insertions() {
        for (ids, provider) in [
            (json!([0]), json!(0)),
            (json!([1]), json!(0)),
            (json!([3]), json!(2)),
            (json!([49]), json!(48)),
            (json!([2]), json!(2)),
            (json!([2]), json!("1")),
            (json!([]), json!(1)),
            (json!([2, 3]), json!(1)),
            (json!([-1]), json!(1)),
            (json!(["2"]), json!(1)),
        ] {
            let mut value = architecture();
            value["config"]["ple_layer_ids"] = ids.clone();
            assert!(
                QwenConfig::parse(&value, &json!({"layer_index": provider})).is_err(),
                "accepted PLE ids={ids}, provider={provider}"
            );
        }
        let mut missing = architecture();
        missing["config"]
            .as_object_mut()
            .unwrap()
            .remove("ple_layer_ids");
        assert!(parse(&missing).is_err());
        assert!(QwenConfig::parse(&architecture(), &Value::Null).is_err());
    }

    #[test]
    fn summaries_must_match_and_matching_summaries_cannot_override_execution_order() {
        let mut value = architecture();
        value["hidden_size"] = json!(2560);
        assert!(parse(&value).is_ok());
        value["hidden_size"] = json!(10240);
        assert!(parse(&value).is_err());

        let mut missing = architecture();
        missing.as_object_mut().unwrap().remove("layer_types");
        assert!(parse(&missing).is_err());
        let mut mismatch = architecture();
        mismatch["layer_types"][0] = json!("full_attention");
        assert!(parse(&mismatch).is_err());

        for invalid_layers in [
            json!(vec!["linear_attention"; 48]),
            json!(vec!["full_attention"; 48]),
            json!(vec!["linear_attention"; 47]),
            json!(vec!["unknown_attention"; 48]),
        ] {
            let mut value = architecture();
            value["config"]["layer_types"] = invalid_layers.clone();
            value["layer_types"] = invalid_layers;
            assert!(parse(&value).is_err());
        }
        let mut interval = architecture();
        interval["config"]["full_attention_interval"] = json!(8);
        assert!(parse(&interval).is_err());
    }

    #[test]
    fn context_and_fixed_geometry_fail_closed_on_invalid_json_values() {
        let mut value = architecture();
        value["config"]["max_position_embeddings"] = json!(1);
        assert_eq!(parse(&value).unwrap().max_positions, 1);
        for invalid in [
            json!(0),
            json!(262145),
            json!(-1),
            json!(true),
            json!("2048"),
        ] {
            value["config"]["max_position_embeddings"] = invalid.clone();
            assert!(parse(&value).is_err(), "accepted context limit {invalid}");
        }
        for (field, invalid) in [
            ("hidden_size", json!(10240)),
            ("num_experts_per_tok", json!(8)),
            ("dtype", json!("float16")),
            ("rms_norm_eps", json!(0)),
            ("eos_token_id", json!(248043)),
            ("attention_bias", json!(true)),
        ] {
            let mut value = architecture();
            value["config"][field] = invalid;
            assert!(parse(&value).is_err(), "accepted incompatible {field}");
        }
        let mut rotary = architecture();
        rotary["config"]["rope_parameters"]["mrope_section"] = json!([10, 11, 11]);
        assert!(parse(&rotary).is_err());
    }
}
