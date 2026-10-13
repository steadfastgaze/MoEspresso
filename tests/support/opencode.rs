use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};

use super::{
    ChildGuard, Deadline, RunningServer, Scratch, binary_path, executable_path,
    limit_child_file_size, package_path, read_bounded_text, wait_child,
};
use moespresso::package::Package;

#[cfg(target_os = "macos")]
use std::os::unix::process::CommandExt;

const TOTAL_LIMIT: Duration = Duration::from_secs(15 * 60);

pub fn run() -> Result<()> {
    ensure!(
        cfg!(target_os = "macos"),
        "OpenCode acceptance requires macOS and Apple Silicon Metal"
    );
    let deadline = Deadline::after(TOTAL_LIMIT)?;
    let binary = binary_path()?;
    let package = package_path()?;
    let opencode = executable_path("MOESPRESSO_OPENCODE_BINARY", "opencode")?;
    let mut scratch = Scratch::new("opencode-acceptance")?;
    let version = opencode_version(&opencode, &scratch)?;

    let cache_root = scratch.subdir("native-cache")?;
    let work = scratch.subdir("work")?;
    let temporary = scratch.subdir("tmp")?;
    let package_metadata = Package::open(&package)?;
    let fixture = work.join("acceptance.txt");
    let initial = "The acceptance marker is green-otter-42.\n";
    fs::write(&fixture, initial)?;

    let server = RunningServer::start(
        &binary,
        &package,
        &scratch,
        &cache_root,
        16384,
        "on",
        true,
        deadline,
    )?;
    let health = server.health(deadline.bounded(Duration::from_secs(5))?)?;
    ensure!(
        health["model"].as_str() == Some(package_metadata.artifact_id.as_str())
            && health["max_context"].as_u64() == Some(16384)
            && health["adaptive_memory"] == true
            && health["cache_routing"] == "prefer-resident"
            && health["prompt_cache"]["memory_enabled"] == true
            && health["prompt_cache"]["disk_enabled"] == true,
        "OpenCode server startup policy differs: {health}"
    );
    let expert_capacity = health["expert_capacity"]
        .as_u64()
        .context("OpenCode server health lacks expert_capacity")?;
    let prefill_capacity = health["prefill_capacity"]
        .as_u64()
        .context("OpenCode server health lacks prefill_capacity")?;
    ensure!(
        (10..=512).contains(&expert_capacity) && matches!(prefill_capacity, 512 | 2048),
        "OpenCode server selected unsupported capacity: {health}"
    );
    println!(
        "OpenCode acceptance: binary={} opencode={} version={} package={} context=16384 memory_budget_gib=20 expert_capacity={} prefill_capacity={} adaptive_memory=true cache_routing=prefer-resident memory_cache=true disk_cache=true",
        binary.display(),
        opencode.display(),
        version,
        package_metadata.artifact_id,
        expert_capacity,
        prefill_capacity
    );

    let config = opencode_config(server.origin(), &fixture, &work)?;
    let config_text = serde_json::to_string(&config)?;
    let output_path = scratch.path().join("client.jsonl");
    let error_path = scratch.path().join("client.stderr");
    let stdout = File::create(&output_path)?;
    let stderr = File::create(&error_path)?;
    let prompt = format!(
        "Use read to inspect {}. Change only its marker from green-otter-42 to blue-fox-73 with edit, then use read to confirm the file. Reply with only the new marker. Do not use any other file or tool.",
        fixture.display()
    );
    let mut command = Command::new(&opencode);
    command
        .args([
            "run",
            "--standalone",
            "--model",
            "native/qwen-local#acceptance",
            "--format",
            "json",
        ])
        .arg(prompt)
        .current_dir(&work)
        .env_clear()
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    for key in ["PATH", "LANG", "LC_ALL", "TERM", "SHELL", "USER", "LOGNAME"] {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    command
        .env("OPENCODE_TEST_HOME", scratch.path())
        .env("HOME", scratch.path())
        .env("TMPDIR", &temporary)
        .env("XDG_CONFIG_HOME", scratch.path().join("config"))
        .env("XDG_DATA_HOME", scratch.path().join("data"))
        .env("XDG_CACHE_HOME", scratch.path().join("cache"))
        .env("XDG_STATE_HOME", scratch.path().join("state"))
        .env("OPENCODE_CONFIG_CONTENT", config_text)
        .env("OPENCODE_DISABLE_PROJECT_CONFIG", "1")
        .env("OPENCODE_DISABLE_AUTOUPDATE", "1")
        .env("OPENCODE_DISABLE_MODELS_FETCH", "1");
    limit_child_file_size(&mut command, 8 * 1024 * 1024);
    #[cfg(target_os = "macos")]
    command.process_group(0);
    let mut child =
        ChildGuard::new_process_group(command.spawn().context("start isolated OpenCode client")?);
    let status = wait_child(
        &mut child,
        deadline.bounded(Duration::from_secs(14 * 60))?,
        "OpenCode read/edit/read-back workflow",
    )?;
    ensure!(
        status.success(),
        "OpenCode exited with {status}: {}",
        read_bounded_text(&error_path, 8 * 1024 * 1024)?
    );
    let _stderr = read_bounded_text(&error_path, 8 * 1024 * 1024)?;
    verify_client_output(&output_path, &fixture, &work)?;
    ensure!(
        read_bounded_text(&fixture, 8 * 1024 * 1024)? == "The acceptance marker is blue-fox-73.\n",
        "OpenCode did not leave the exact expected fixture bytes"
    );
    drop(child);
    let logs = server.logs()?;
    let mut reused_turns = 0;
    for line in logs
        .lines()
        .filter(|line| line.starts_with("request complete "))
    {
        println!("{line}");
        let count = |name: &str| -> Option<usize> {
            line.split_whitespace()
                .find_map(|field| field.strip_prefix(name)?.parse().ok())
        };
        if line.contains("cache=memory_hit") || line.contains("cache=disk_hit") {
            let cached = count("cached_tokens=").context("cached request lacks token count")?;
            let evaluated =
                count("prefill_tokens=").context("cached request lacks prefill count")?;
            if cached >= 1024 && evaluated < cached {
                reused_turns += 1;
            }
        }
    }
    ensure!(
        reused_turns >= 2,
        "OpenCode tool continuations did not reuse substantial prefixes: {logs}"
    );
    ensure!(
        !logs.contains("adaptive memory: state="),
        "normal serving emitted per-sample trace"
    );
    drop(server);
    scratch.cleanup_all()?;
    println!(
        "OpenCode {version}: restricted read/edit/read-back passed with {reused_turns} reused-prefix turns"
    );
    Ok(())
}

fn opencode_version(binary: &Path, scratch: &Scratch) -> Result<String> {
    let stdout_path = scratch.path().join("opencode-version.stdout");
    let stderr_path = scratch.path().join("opencode-version.stderr");
    let xdg_config = scratch.subdir("version-config")?;
    let xdg_data = scratch.subdir("version-data")?;
    let xdg_cache = scratch.subdir("version-cache")?;
    let xdg_state = scratch.subdir("version-state")?;
    let mut command = Command::new(binary);
    command
        .arg("--version")
        .stdin(Stdio::null())
        .env_clear()
        .stdout(Stdio::from(File::create(&stdout_path)?))
        .stderr(Stdio::from(File::create(&stderr_path)?));
    for key in ["PATH", "LANG", "LC_ALL", "TERM"] {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    command
        .env("HOME", scratch.path())
        .env("XDG_CONFIG_HOME", xdg_config)
        .env("XDG_DATA_HOME", xdg_data)
        .env("XDG_CACHE_HOME", xdg_cache)
        .env("XDG_STATE_HOME", xdg_state);
    limit_child_file_size(&mut command, 4096);
    let mut child = ChildGuard::new(
        command
            .spawn()
            .context("start OpenCode version preflight")?,
    );
    let status = wait_child(
        &mut child,
        Duration::from_secs(10),
        "OpenCode version preflight",
    )?;
    let text = read_bounded_text(&stdout_path, 4096)?;
    ensure!(
        status.success(),
        "OpenCode --version failed: {}",
        read_bounded_text(&stderr_path, 4096)?
    );
    let version = text.trim();
    ensure!(
        !version.is_empty(),
        "OpenCode --version returned no version"
    );
    Ok(version.to_owned())
}

fn opencode_config(origin: &str, fixture: &Path, work: &Path) -> Result<Value> {
    ensure!(
        origin.starts_with("http://127.0.0.1:"),
        "OpenCode upstream is not loopback"
    );
    let fixture = fs::canonicalize(fixture)?;
    let work = fs::canonicalize(work)?;
    let fixture_text = fixture.to_string_lossy().into_owned();
    let relative_fixture = fixture
        .strip_prefix(Path::new("/"))
        .context("fixture is not an absolute path")?
        .to_string_lossy()
        .into_owned();
    let work_glob = format!("{}/*", work.display());
    let mut permissions = vec![json!({"action":"*","resource":"*","effect":"deny"})];
    for action in ["read", "edit"] {
        for resource in [
            fixture_text.as_str(),
            relative_fixture.as_str(),
            "acceptance.txt",
        ] {
            permissions.push(json!({"action":action,"resource":resource,"effect":"allow"}));
        }
    }
    for resource in [work.display().to_string(), work_glob] {
        permissions
            .push(json!({"action":"external_directory","resource":resource,"effect":"allow"}));
    }
    Ok(json!({
        "$schema": "https://opencode.ai/config.json",
        "model": "native/qwen-local#acceptance",
        "update": "disable",
        "share": "disabled",
        "snapshots": false,
        "compaction": {"auto":false},
        "permissions": permissions,
        "agents": {
            "build": {"request":{"body":{"temperature":0}}},
            "title": {"model":"native/qwen-local#acceptance"}
        },
        "experimental": {
            "policies": [
                {"action":"provider.use","resource":"*","effect":"deny"},
                {"action":"provider.use","resource":"native","effect":"allow"}
            ]
        },
        "providers": {
            "native": {
                "package": "@opencode/ai/providers/openai-compatible",
                "name": "Native acceptance",
                "settings": {
                    "baseURL": format!("{origin}/v1"),
                    "apiKey": "local-acceptance-dummy"
                },
                "models": {
                    "qwen-local": {
                        "name": "Qwen local",
                        "capabilities": {"tools":true,"input":["text"],"output":["text"]},
                        "limit": {"context": 16384, "output": 512},
                        "variants": [{
                            "id":"acceptance",
                            "settings":{"reasoningEffort":"medium"},
                            "body":{"chat_template_kwargs":{"enable_thinking":true}}
                        }]
                    }
                }
            }
        }
    }))
}

fn verify_client_output(path: &Path, fixture: &Path, work: &Path) -> Result<()> {
    let text = read_bounded_text(path, 8 * 1024 * 1024)?;
    let expected_fixture = fs::canonicalize(fixture)?;
    let work = fs::canonicalize(work)?;
    let mut tools: Vec<(String, Value)> = Vec::new();
    let mut final_text = String::new();
    let mut errors = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let event: Value = serde_json::from_str(line)
            .with_context(|| format!("OpenCode output line {} is not JSON", index + 1))?;
        match event["type"].as_str() {
            Some("tool_use") => {
                let part = &event["part"];
                let name = part["tool"]
                    .as_str()
                    .context("OpenCode tool event lacks a name")?;
                ensure!(
                    matches!(name, "read" | "edit"),
                    "OpenCode used a tool outside the read/edit allowlist: {name}"
                );
                let state = &part["state"];
                ensure!(
                    state["status"] == "completed",
                    "OpenCode tool did not complete: {part}"
                );
                let input_path = state["input"]["filePath"]
                    .as_str()
                    .or_else(|| state["input"]["path"].as_str())
                    .context("OpenCode tool input lacks a file path")?;
                let resolved = if Path::new(input_path).is_absolute() {
                    PathBuf::from(input_path)
                } else {
                    work.join(input_path)
                };
                ensure!(
                    fs::canonicalize(&resolved)? == expected_fixture,
                    "OpenCode touched a file outside the acceptance fixture: {}",
                    resolved.display()
                );
                tools.push((name.to_owned(), state.clone()));
            }
            Some("text") => {
                if let Some(value) = event["part"]["text"].as_str() {
                    final_text.push_str(value);
                }
            }
            Some("error") => errors.push(event.to_string()),
            _ => {}
        }
    }
    ensure!(errors.is_empty(), "OpenCode reported errors: {errors:?}");
    let read_before = tools.iter().position(|(name, state)| {
        name == "read"
            && state["output"]
                .as_str()
                .is_some_and(|output| output.contains("green-otter-42"))
    });
    let edit = tools.iter().enumerate().find(|(index, (name, state))| {
        Some(*index) > read_before
            && name == "edit"
            && state["input"].to_string().contains("green-otter-42")
            && state["input"].to_string().contains("blue-fox-73")
    });
    let read_back = tools.iter().enumerate().find(|(index, (name, state))| {
        Some(*index) > edit.map(|(at, _)| at)
            && name == "read"
            && state["output"]
                .as_str()
                .is_some_and(|output| output.contains("blue-fox-73"))
    });
    ensure!(
        read_before.is_some() && edit.is_some() && read_back.is_some(),
        "OpenCode did not complete read, edit, then read-back; tool sequence: {:?}",
        tools.iter().map(|(name, _)| name).collect::<Vec<_>>()
    );
    ensure!(
        final_text.trim() == "blue-fox-73",
        "OpenCode final response differs: {final_text:?}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use anyhow::Result;
    use serde_json::{Value, json};

    use super::{Scratch, verify_client_output};

    #[test]
    fn empty_output_and_final_marker_without_tools_are_rejected() -> Result<()> {
        let mut scratch = Scratch::new("opencode-output-guards")?;
        let work = scratch.subdir("work")?;
        let fixture = work.join("acceptance.txt");
        fs::write(&fixture, "The acceptance marker is green-otter-42.\n")?;
        let output = scratch.path().join("client.jsonl");

        fs::write(&output, "")?;
        assert!(verify_client_output(&output, &fixture, &work).is_err());

        fs::write(
            &output,
            "{\"type\":\"text\",\"part\":{\"text\":\"blue-fox-73\"}}\n",
        )?;
        let error = verify_client_output(&output, &fixture, &work)
            .expect_err("a final marker without tool events must not pass");
        assert!(format!("{error:#}").contains("read, edit, then read-back"));

        scratch.cleanup_all()?;
        Ok(())
    }

    #[test]
    fn accepts_an_ordered_synthetic_read_edit_readback_trace() -> Result<()> {
        let mut scratch = Scratch::new("opencode-output-valid")?;
        let work = scratch.subdir("work")?;
        let fixture = work.join("acceptance.txt");
        fs::write(&fixture, "The acceptance marker is blue-fox-73.\n")?;
        let fixture_path = fixture.to_string_lossy().into_owned();
        let events: [Value; 4] = [
            json!({
                "type": "tool_use",
                "part": {
                    "tool": "read",
                    "state": {
                        "status": "completed",
                        "input": {"filePath": fixture_path},
                        "output": "The acceptance marker is green-otter-42."
                    }
                }
            }),
            json!({
                "type": "tool_use",
                "part": {
                    "tool": "edit",
                    "state": {
                        "status": "completed",
                        "input": {
                            "filePath": fixture_path,
                            "oldString": "green-otter-42",
                            "newString": "blue-fox-73"
                        },
                        "output": "updated"
                    }
                }
            }),
            json!({
                "type": "tool_use",
                "part": {
                    "tool": "read",
                    "state": {
                        "status": "completed",
                        "input": {"filePath": fixture_path},
                        "output": "The acceptance marker is blue-fox-73."
                    }
                }
            }),
            json!({"type": "text", "part": {"text": "blue-fox-73"}}),
        ];
        let output = scratch.path().join("client.jsonl");
        let lines = events
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&output, format!("{lines}\n"))?;

        verify_client_output(&output, &fixture, &work)?;
        scratch.cleanup_all()?;
        Ok(())
    }
}
