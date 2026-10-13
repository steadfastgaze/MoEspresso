use std::fmt::Display;

use serde_json::Value;

#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct ServerHealth {
    pub(super) status: Option<String>,
    pub(super) model: Option<String>,
    pub(super) expert_capacity: Option<u64>,
    pub(super) max_context: Option<u64>,
    pub(super) prefill_capacity: Option<u64>,
    pub(super) adaptive_memory: Option<bool>,
    pub(super) cache_routing: Option<String>,
    pub(super) memory_enabled: Option<bool>,
    pub(super) disk_enabled: Option<bool>,
    pub(super) disk_mode: Option<String>,
    pub(super) retained_bytes: Option<u64>,
    mlx: Option<MlxHealth>,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct MlxHealth {
    streaming: Option<bool>,
    slot_budget: Option<u64>,
    capacity_layout: Option<String>,
    routing_factor: Option<f64>,
    protected_routes: Option<u64>,
}

impl ServerHealth {
    pub(super) fn parse(value: &Value) -> Option<Self> {
        if !value.is_object() {
            return None;
        }
        let cache = &value["prompt_cache"];
        let streaming = value.get("ssd_streaming").filter(|stats| stats.is_object());
        let mlx = streaming.map(|stats| MlxHealth {
            streaming: stats["enabled"].as_bool(),
            slot_budget: stats["capacity_per_layer"].as_u64(),
            capacity_layout: capacity_layout(stats),
            routing_factor: stats["cache_routing"]["cache_factor"].as_f64(),
            protected_routes: stats["cache_routing"]["protected_routes"].as_u64(),
        });
        let is_mlx = mlx.is_some();
        Some(Self {
            status: value["status"].as_str().map(str::to_owned),
            model: value["model"].as_str().map(str::to_owned),
            // The MLX field is a uniform budget before per-layer redistribution,
            // not the native uniform runtime capacity.
            expert_capacity: value["expert_capacity"].as_u64(),
            max_context: if is_mlx {
                value["diagnostics"]["identity"]["context_limit"].as_u64()
            } else {
                value["max_context"].as_u64()
            },
            prefill_capacity: if is_mlx {
                None
            } else {
                value["prefill_capacity"].as_u64()
            },
            adaptive_memory: if is_mlx {
                streaming.and_then(|stats| stats["adaptive_growth"]["enabled"].as_bool())
            } else {
                value["adaptive_memory"].as_bool()
            },
            cache_routing: if is_mlx {
                streaming.and_then(|stats| stats["cache_routing"]["policy"].as_str())
            } else {
                value["cache_routing"].as_str()
            }
            .map(str::to_owned),
            // Empty entry and byte counts do not establish whether RAM caching
            // is disabled; the MLX health endpoint does not publish this flag.
            memory_enabled: if is_mlx {
                None
            } else {
                cache["memory_enabled"].as_bool()
            },
            disk_enabled: if is_mlx {
                cache["disk"]["enabled"].as_bool()
            } else {
                cache["disk_enabled"].as_bool()
            },
            disk_mode: cache["disk_mode"].as_str().map(str::to_owned),
            retained_bytes: if is_mlx {
                cache["bytes"].as_u64()
            } else {
                cache["retained_bytes"].as_u64()
            },
            mlx,
        })
    }
}

fn capacity_layout(stats: &Value) -> Option<String> {
    let overrides = stats["capacity_overrides"].as_object()?;
    if overrides.is_empty() {
        return Some("uniform".to_owned());
    }
    let capacities = overrides
        .values()
        .map(Value::as_u64)
        .collect::<Option<Vec<_>>>()?;
    let min = capacities.iter().min()?;
    let max = capacities.iter().max()?;
    Some(format!("per-layer overrides {min}–{max} slots"))
}

pub(super) fn optional_text<T: Display>(value: Option<T>) -> String {
    value.map_or_else(|| "unavailable".to_owned(), |value| value.to_string())
}

pub(super) fn paired_text<T: Display>(start: Option<T>, end: Option<T>) -> String {
    format!("{} → {}", optional_text(start), optional_text(end))
}

fn enabled_text(value: Option<bool>) -> String {
    value.map_or_else(
        || "unavailable".to_owned(),
        |enabled| if enabled { "on" } else { "off" }.to_owned(),
    )
}

fn paired_enabled(start: Option<bool>, end: Option<bool>) -> String {
    format!("{} → {}", enabled_text(start), enabled_text(end))
}

pub(super) fn format_conditions(
    start: Option<&ServerHealth>,
    end: Option<&ServerHealth>,
) -> String {
    if start.is_none() && end.is_none() {
        return "unavailable".to_owned();
    }
    if start.is_some_and(|health| health.mlx.is_some())
        || end.is_some_and(|health| health.mlx.is_some())
    {
        return format_mlx_conditions(start, end);
    }
    let field =
        |get: fn(&ServerHealth) -> Option<u64>| paired_text(start.and_then(get), end.and_then(get));
    let bool_field = |get: fn(&ServerHealth) -> Option<bool>| {
        paired_enabled(start.and_then(get), end.and_then(get))
    };
    let routing = paired_text(
        start.and_then(|health| health.cache_routing.as_deref()),
        end.and_then(|health| health.cache_routing.as_deref()),
    );
    format!(
        "health sampled start → end: expert capacity {} slots; max context {} tokens; prefill {} rows; adaptive {}; decode routing {}; memory cache {}; disk cache {}; disk mode {}; memory retained {} bytes",
        field(|health| health.expert_capacity),
        field(|health| health.max_context),
        field(|health| health.prefill_capacity),
        bool_field(|health| health.adaptive_memory),
        routing,
        bool_field(|health| health.memory_enabled),
        bool_field(|health| health.disk_enabled),
        paired_text(
            start.and_then(|health| health.disk_mode.as_deref()),
            end.and_then(|health| health.disk_mode.as_deref()),
        ),
        field(|health| health.retained_bytes),
    )
}

fn format_mlx_conditions(start: Option<&ServerHealth>, end: Option<&ServerHealth>) -> String {
    let first = start.and_then(|health| health.mlx.as_ref());
    let last = end.and_then(|health| health.mlx.as_ref());
    let field =
        |get: fn(&ServerHealth) -> Option<u64>| paired_text(start.and_then(get), end.and_then(get));
    let bool_field = |get: fn(&ServerHealth) -> Option<bool>| {
        paired_enabled(start.and_then(get), end.and_then(get))
    };
    format!(
        "MLX health sampled start → end: SSD streaming {}; expert slot budget {} per layer; layout {}; max context {} tokens; prefill unavailable; adaptive growth {}; decode routing {}; routing factor {}; protected routes {}; autonomous unavailable; memory cache unavailable; disk cache {}; memory retained {} bytes",
        paired_enabled(
            first.and_then(|h| h.streaming),
            last.and_then(|h| h.streaming)
        ),
        paired_text(
            first.and_then(|h| h.slot_budget),
            last.and_then(|h| h.slot_budget)
        ),
        paired_text(
            first.and_then(|h| h.capacity_layout.as_deref()),
            last.and_then(|h| h.capacity_layout.as_deref())
        ),
        field(|health| health.max_context),
        bool_field(|health| health.adaptive_memory),
        paired_text(
            start.and_then(|h| h.cache_routing.as_deref()),
            end.and_then(|h| h.cache_routing.as_deref())
        ),
        paired_text(
            first.and_then(|h| h.routing_factor),
            last.and_then(|h| h.routing_factor)
        ),
        paired_text(
            first.and_then(|h| h.protected_routes),
            last.and_then(|h| h.protected_routes)
        ),
        bool_field(|health| health.disk_enabled),
        field(|health| health.retained_bytes),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn native_health_reports_cache_conditions() {
        let native = ServerHealth::parse(&json!({
            "status": "ok", "model": "qwen-test", "expert_capacity": 128,
            "max_context": 131072, "prefill_capacity": 2048,
            "adaptive_memory": true, "cache_routing": "prefer-resident",
            "prompt_cache": {
                "memory_enabled": true, "disk_enabled": false, "retained_bytes": 1024
            }
        }))
        .unwrap();
        assert_eq!(
            format_conditions(Some(&native), Some(&native)),
            "health sampled start → end: expert capacity 128 → 128 slots; max context 131072 → 131072 tokens; prefill 2048 → 2048 rows; adaptive on → on; decode routing prefer-resident → prefer-resident; memory cache on → on; disk cache off → off; disk mode unavailable → unavailable; memory retained 1024 → 1024 bytes"
        );
        for mode in ["prefix", "full"] {
            let enabled = ServerHealth::parse(&json!({
                "status": "ok",
                "prompt_cache": {"disk_enabled": true, "disk_mode": mode}
            }))
            .unwrap();
            assert!(
                format_conditions(Some(&enabled), Some(&enabled))
                    .contains(&format!("disk cache on → on; disk mode {mode} → {mode}"))
            );
        }
    }

    #[test]
    fn mlx_health_uses_nested_fields_without_guessing_unpublished_settings() {
        let mlx = ServerHealth::parse(&json!({
            "status": "ok", "model": "qwen-test",
            "prompt_cache": {
                "default_live_kv_format": "qwen4-composite", "entries": 1,
                "bytes": 4096, "disk": {"enabled": false}
            },
            "ssd_streaming": {
                "enabled": true, "capacity_per_layer": 209,
                "capacity_overrides": {"2": 112, "3": 256},
                "adaptive_growth": null,
                "cache_routing": {"policy": "prefer-resident", "cache_factor": 2.0, "protected_routes": 2}
            },
            "diagnostics": {
                "schema": "moespresso-server-diagnostics-v3",
                "identity": {"context_limit": 131072, "instance_id": "not-reported", "pid": 42}
            }
        })).unwrap();
        assert_eq!(mlx.expert_capacity, None);
        assert_eq!(mlx.max_context, Some(131072));
        assert_eq!(mlx.prefill_capacity, None);
        assert_eq!(mlx.adaptive_memory, None);
        assert_eq!(mlx.memory_enabled, None);
        assert_eq!(mlx.disk_enabled, Some(false));
        assert_eq!(mlx.retained_bytes, Some(4096));
        let output = format_conditions(Some(&mlx), Some(&mlx));
        assert!(output.contains("expert slot budget 209 → 209 per layer"));
        assert!(output.contains("per-layer overrides 112–256 slots"));
        assert!(output.contains("max context 131072 → 131072 tokens"));
        assert!(output.contains("prefill unavailable"));
        assert!(output.contains("adaptive growth unavailable → unavailable"));
        assert!(output.contains("routing factor 2 → 2; protected routes 2 → 2"));
        assert!(output.contains("autonomous unavailable; memory cache unavailable"));
        assert!(output.contains("disk cache off → off; memory retained 4096 → 4096 bytes"));
        assert!(!output.contains("not-reported"));
    }

    #[test]
    fn mlx_uniform_and_missing_fields_remain_distinct() {
        let uniform = ServerHealth::parse(&json!({
            "ssd_streaming": {
                "enabled": true, "capacity_per_layer": 186, "capacity_overrides": {},
                "adaptive_growth": {"enabled": true}
            },
            "prompt_cache": {"entries": 0, "bytes": 0, "disk": {"enabled": true}}
        }))
        .unwrap();
        let output = format_conditions(Some(&uniform), None);
        assert!(output.contains("layout uniform → unavailable"));
        assert!(output.contains("max context unavailable → unavailable tokens"));
        assert!(output.contains("adaptive growth on → unavailable"));
        assert!(output.contains("memory cache unavailable; disk cache on → unavailable"));
        assert_eq!(format_conditions(None, None), "unavailable");
        assert!(ServerHealth::parse(&Value::Null).is_none());
        assert_eq!(
            capacity_layout(&json!({"capacity_overrides": {"2": "invalid"}})),
            None
        );
        assert_eq!(capacity_layout(&json!({})), None);
    }
}
