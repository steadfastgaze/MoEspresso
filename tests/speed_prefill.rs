#![cfg(target_os = "macos")]

use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

use serde_json::{Value, json};
use tiny_http::{Response, Server};

#[test]
fn speed_runs_4k_after_topics_and_skips_32k_without_a_terminal() {
    let server = Server::http("127.0.0.1:0").unwrap();
    let origin = format!("http://{}", server.server_addr().to_ip().unwrap());
    let worker = thread::spawn(move || {
        let mut completions = 0;
        for _ in 0..10 {
            let mut request = server
                .recv_timeout(Duration::from_secs(10))
                .unwrap()
                .unwrap();
            let response = if request.url() == "/health" {
                json!({"status": "ok", "model": "fixture", "max_context": 131072})
            } else {
                assert_eq!(request.url(), "/v1/chat/completions");
                completions += 1;
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).unwrap();
                let body: Value = serde_json::from_str(&body).unwrap();
                let prompt = body["messages"][0]["content"].as_str().unwrap();
                let tokens = if completions == 8 {
                    assert!(prompt.len() > 16000 && prompt.len() < 30000);
                    assert_eq!(body["max_tokens"], 8);
                    4098
                } else {
                    assert!(prompt.len() < 200);
                    20
                };
                json!({"choices": [{"finish_reason": "length"}], "usage": {
                    "prompt_tokens": tokens, "completion_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 0},
                    "moespresso": {"generation_seconds": 3.0, "first_token_seconds": 2.0,
                        "prefill_tokens": tokens, "prefill_seconds": 2.0,
                        "ready_to_first_token_seconds": 2.0}
                }})
            };
            request
                .respond(Response::from_string(response.to_string()))
                .unwrap();
        }
        assert_eq!(completions, 8);
    });
    let output = Command::new(env!("CARGO_BIN_EXE_moespresso"))
        .args([
            "speed",
            "--url",
            &origin,
            "--max-tokens",
            "8",
            "--timeout",
            "5",
        ])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    worker.join().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("~4K prefill: 4098 new / 0 cached tokens"));
    assert!(text.contains("~32K prefill skipped: stdin is not a terminal."));
    assert!(text.find("MoEspresso speed").unwrap() < text.find("~4K prefill:").unwrap());
}
