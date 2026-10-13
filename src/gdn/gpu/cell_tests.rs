use super::*;
use crate::package::Package;
use serde_json::Value;
use std::path::PathBuf;

#[test]
#[ignore = "requires MOESPRESSO_TEST_PACKAGE, MOESPRESSO_GDN_CELL_FIXTURE and exclusive Metal ownership"]
fn restored_cell_preserves_captured_stages_state_and_frontier() -> Result<()> {
    let root = PathBuf::from(
        std::env::var_os("MOESPRESSO_GDN_CELL_FIXTURE")
            .context("MOESPRESSO_GDN_CELL_FIXTURE must select the saved gdn_cell_v1 fixture")?,
    );
    let manifest_bytes = std::fs::read(root.join("manifest.json"))?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    ensure!(
        manifest["schema"] == "qwen4-gdn-cell-fixture-v1",
        "unsupported GDN cell fixture schema"
    );
    let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
    ensure!(
        manifest["artifact_id"].as_str() == Some(package.artifact_id.as_str()),
        "GDN fixture and package artifact differ"
    );
    let module = manifest["module_path"]
        .as_str()
        .context("GDN fixture module path")?;
    let steps = manifest["steps"].as_array().context("GDN fixture steps")?;
    ensure!(
        steps.len() >= 3,
        "GDN fixture needs consecutive captured tokens"
    );
    let step = &steps[1];
    let read = |record: &Value, name: &str| -> Result<Vec<u8>> {
        let path = record["files"][name]["path"]
            .as_str()
            .context("GDN fixture payload path")?;
        Ok(std::fs::read(root.join(path))?)
    };
    let input = read(step, "input")?;
    let initial_conv = read(&steps[0], "conv_state")?;
    let initial_recurrent = read(&steps[0], "recurrent_state")?;
    let expected_output = read(step, "output")?;
    let expected_conv = read(step, "conv_state")?;
    let expected_recurrent = read(step, "recurrent_state")?;
    let before = step["frontier_before"]
        .as_u64()
        .context("GDN fixture initial frontier")?;
    let after = step["frontier_after"]
        .as_u64()
        .context("GDN fixture final frontier")?;
    ensure!(
        before > 0
            && before.checked_add(1) == Some(after)
            && steps[0]["frontier_after"].as_u64() == Some(before),
        "GDN fixture state frontiers differ"
    );
    ensure!(
        initial_conv.iter().any(|&byte| byte != 0)
            && initial_recurrent.iter().any(|&byte| byte != 0),
        "restored GDN cell requires nonzero captured states"
    );
    finite_bf16(&input, HIDDEN, "captured GDN hidden")?;
    finite_bf16(&expected_output, HIDDEN, "captured GDN output")?;
    GdnState::validate_parts(&expected_conv, &expected_recurrent)?;
    let metal = Metal::new()?;
    let names = [
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
        "conv1d",
        "A_log",
        "dt_bias",
        "norm",
    ];
    let weights: Vec<DirectWeight> = names
        .into_iter()
        .map(|name| package.direct_weight(&format!("{module}.{name}")))
        .collect::<Result<_>>()?;
    let mut cell = GdnCell::new(
        &metal,
        GdnWeights {
            qkv: &weights[0],
            gate: &weights[1],
            beta: &weights[2],
            decay: &weights[3],
            output: &weights[4],
            conv: &weights[5],
            a_log: &weights[6],
            dt_bias: &weights[7],
            norm: &weights[8],
            eps: manifest["norm_eps"]
                .as_f64()
                .context("GDN fixture norm epsilon")? as f32,
        },
    )?;
    drop(weights);
    let restore = || -> Result<GdnState> {
        let state = GdnState::from_bytes(&metal, &initial_conv, &initial_recurrent, before)?;
        let (conv, recurrent) = state.snapshot()?;
        ensure!(
            state.frontier()? == before && conv == initial_conv && recurrent == initial_recurrent,
            "restored GDN state differs before execution"
        );
        Ok(state)
    };
    let check = |bytes: &[u8], frontier: u64, state: &GdnState| -> Result<()> {
        let (conv, recurrent) = state.snapshot()?;
        ensure!(
            bytes == expected_output.as_slice()
                && frontier == after
                && state.frontier()? == after
                && conv == expected_conv
                && recurrent == expected_recurrent,
            "completed GDN output, state or frontier differs from the captured reference"
        );
        Ok(())
    };
    let mut state = restore()?;
    let stages = cell.submit(&input, &mut state)?.wait_stages()?;
    for (name, bytes) in [
        ("qkv", stages.qkv_bf16.as_slice()),
        ("gate", stages.gate_bf16.as_slice()),
        ("beta_logits", stages.beta_logits_bf16.as_slice()),
        ("decay_logits", stages.decay_logits_bf16.as_slice()),
        ("query", stages.query_bf16.as_slice()),
        ("key", stages.key_bf16.as_slice()),
        ("value", stages.value_bf16.as_slice()),
        ("beta", stages.beta_bf16.as_slice()),
        ("decay", stages.decay_f32.as_slice()),
        ("recurrence", stages.recurrence_bf16.as_slice()),
        ("norm_gate", stages.norm_gate_bf16.as_slice()),
        ("output", stages.output_bf16.as_slice()),
        ("conv_state", stages.conv_state_bf16.as_slice()),
        ("recurrent_state", stages.recurrent_state_f32.as_slice()),
    ] {
        ensure!(
            bytes == read(step, name)?.as_slice(),
            "restored GDN {name} differs from captured stage"
        );
    }
    check(&stages.output_bf16, stages.frontier, &state)?;
    let mut state = restore()?;
    let output = cell.submit(&input, &mut state)?.wait()?;
    check(&output.output_bf16, output.frontier, &state)?;
    Ok(())
}
