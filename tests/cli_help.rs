#![cfg(target_os = "macos")]

use std::process::Command;

#[test]
fn disk_location_and_budget_are_documented_cli_options() {
    let output = Command::new(env!("CARGO_BIN_EXE_moespresso"))
        .args([
            "serve",
            "--disk-cache-dir",
            "/unused-cache",
            "--disk-cache-gb",
            "2.5",
            "--help",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output);
    assert!(output.stderr.is_empty());
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("--disk-cache-dir PATH"));
    assert!(text.contains("--disk-cache-gb GIB"));
    assert!(text.contains("--disk-cache-mode MODE"));
    assert!(text.contains("default: prefix"));
    assert!(!text.contains("environment settings"));
}

#[test]
fn help_prints_to_stdout_without_loading_a_model_or_contacting_a_server() {
    for (args, usage, option) in [
        (vec!["--help"], "Usage: moespresso <COMMAND>", "serve"),
        (
            vec!["serve", "--help"],
            "Usage: moespresso serve PACKAGE",
            "--no-disk-cache",
        ),
        (
            vec!["generate", "--help"],
            "Usage: moespresso generate PACKAGE",
            "--prompt TEXT",
        ),
        (
            vec!["speed", "--help"],
            "Usage: moespresso speed",
            "--url HTTP_ORIGIN",
        ),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_moespresso"))
            .args(args)
            .output()
            .unwrap();
        assert!(output.status.success());
        assert!(output.stderr.is_empty());
        let text = String::from_utf8(output.stdout).unwrap();
        assert!(text.contains(usage));
        assert!(text.contains(option));
    }
}
