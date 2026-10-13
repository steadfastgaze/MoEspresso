#![cfg(target_os = "macos")]

use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

use serde_json::{Value, json};
use tiny_http::{Response, Server};

fn run_fixture(
    fork: bool,
    stream: bool,
    imperfect: bool,
    compare_id: &str,
) -> (String, Vec<Value>) {
    let server = Server::http("127.0.0.1:0").unwrap();
    let origin = format!("http://{}", server.server_addr().to_ip().unwrap());
    let worker = thread::spawn(move || {
        let mut requests = Vec::new();
        for _ in 0..10 {
            let mut request = server
                .recv_timeout(Duration::from_secs(10))
                .unwrap()
                .unwrap();
            let body = if request.url() == "/health" {
                if fork {
                    json!({"status": "ok", "model": "fixture", "ssd_streaming": {
                        "enabled": true, "capacity_per_layer": 209,
                        "capacity_overrides": {"0": 192, "1": 224},
                        "cache_routing": {"policy": "prefer-resident", "cache_factor": 2.0, "protected_routes": 2}},
                        "prompt_cache": {"bytes": 0, "disk": {"enabled": false}},
                        "diagnostics": {"identity": {"context_limit": 131072}}
                    }).to_string()
                } else {
                    json!({"status": "ok", "model": "fixture", "expert_capacity": 186,
                        "prefill_capacity": 2048, "max_context": 131072})
                    .to_string()
                }
            } else {
                assert_eq!(request.url(), "/v1/chat/completions");
                let mut posted = String::new();
                request.as_reader().read_to_string(&mut posted).unwrap();
                let posted: Value = serde_json::from_str(&posted).unwrap();
                let cap = posted["max_tokens"].as_u64().unwrap();
                requests.push(posted);
                let index = requests.len();
                let cached = u64::from(imperfect && (index == 2 || index == 8));
                let early = imperfect && index == 3;
                let tokens = if index == 8 { 4098 } else { 30 };
                let mut usage = json!({"prompt_tokens": tokens, "completion_tokens": if early {3} else {cap},
                    "prompt_tokens_details": {"cached_tokens": cached},
                    "moespresso": {"generation_seconds": 5.0, "first_token_seconds": 2.0,
                        "ready_to_first_token_seconds": 2.1}});
                if !fork {
                    usage["moespresso"]["prefill_tokens"] = json!(tokens - cached);
                    usage["moespresso"]["prefill_seconds"] = json!(1.0);
                }
                let finish = if early { "stop" } else { "length" };
                if stream {
                    format!(
                        "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                        json!({"choices": [{"finish_reason": finish}]}),
                        json!({"choices": [], "usage": usage})
                    )
                } else {
                    json!({"choices": [{"finish_reason": finish}], "usage": usage}).to_string()
                }
            };
            request.respond(Response::from_string(body)).unwrap();
        }
        requests
    });
    let mut command = Command::new(env!("CARGO_BIN_EXE_moespresso"));
    command.args([
        "speed",
        "--url",
        &origin,
        "--max-tokens",
        "8",
        "--timeout",
        "5",
        "--compare-id",
        compare_id,
    ]);
    if stream {
        command.arg("--stream");
    }
    let output = command.stdin(Stdio::null()).output().unwrap();
    let requests = worker.join().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    (String::from_utf8(output.stdout).unwrap(), requests)
}

#[test]
fn native_and_fork_receive_identical_requests_and_share_only_comparable_metrics() {
    let (native, native_requests) = run_fixture(false, false, false, "42");
    let (fork, fork_requests) = run_fixture(true, false, false, "42");
    assert_eq!(native_requests, fork_requests);
    for request in &native_requests {
        assert_eq!(request["temperature"], 0.0);
        assert_eq!(request["chat_template_kwargs"]["enable_thinking"], true);
        assert_eq!(request["chat_template_kwargs"]["preserve_thinking"], true);
        assert_eq!(request["reasoning_effort"], "medium");
    }
    for output in [&native, &fork] {
        assert!(output.contains("~4K effective prefill: 2049.00 tok/s"));
        assert!(output.contains("Comparison decode: 2.33 tok/s median · 6/6"));
        assert!(output.contains("~32K prefill skipped: stdin is not a terminal."));
    }
    assert!(native.contains("~4K prefill: 4098 new / 0 cached tokens · 1.00 s · 4098.00 tok/s"));
    assert!(
        fork.contains(
            "~4K prefill: 4098 new / 0 cached tokens · unavailable s · unavailable tok/s"
        )
    );
    assert!(fork.contains("expert slot budget 209 → 209 per layer"));
    assert!(fork.contains("per-layer overrides 192–224 slots"));
}

#[test]
fn streamed_fork_cache_hits_and_early_stops_do_not_become_matched_results() {
    let (output, requests) = run_fixture(true, true, true, "43");
    assert!(output.contains("Comparison decode excluded: cached or unverified prefix."));
    assert!(output.contains("Comparison decode excluded: early stop or unmatched output count."));
    assert!(output.contains("Comparison decode: unavailable tok/s median · 4/6"));
    assert!(output.contains("~4K effective prefill: unavailable tok/s"));
    assert!(output.contains("not a cold comparison"));
    assert_eq!(requests[0]["stream_options"]["include_usage"], true);
}
