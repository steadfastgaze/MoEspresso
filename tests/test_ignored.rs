#![cfg(target_os = "macos")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const CARGO: &str = r#"#!/bin/sh
set -eu
stage=run
case " $* " in *' --list '*) stage=list ;; esac
printf '%s\n' "$stage" >> "$FIXTURE_ROOT/launches"
if [ "$stage" = list ] && [ "$FIXTURE_STAGE" != list ]; then
    if [ "$FIXTURE_STAGE" != wrong ]; then printf '%s: test\n' "$FIXTURE_TEST"; fi
    exit 0
fi
sh "$FIXTURE_ROOT/leaf.sh" &
fixture_child=$!
trap 'wait "$fixture_child" || :; printf reaped > "$FIXTURE_ROOT/reaped"; exit 7' HUP INT TERM
printf '%s\n' "$$" > "$FIXTURE_ROOT/leader"
wait "$fixture_child"
"#;

const LEAF: &str = r#"#!/bin/sh
set -eu
if [ "$FIXTURE_STAGE" = stubborn ]; then
    trap '' HUP INT TERM
else
    trap 'exit 0' HUP INT TERM
fi
printf '%s\n' "$$" > "$FIXTURE_ROOT/descendant"
while :; do sleep 1; done
"#;

struct Fixture {
    root: PathBuf,
    wrapper: Option<Child>,
    unrelated: Child,
}

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "moespresso-ignored-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let root = root.canonicalize().unwrap();
        fs::create_dir(root.join("bin")).unwrap();
        fs::create_dir(root.join("tmp")).unwrap();
        std::os::unix::fs::symlink(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts"),
            root.join("scripts"),
        )
        .unwrap();
        fs::write(root.join("bin/cargo"), CARGO).unwrap();
        fs::write(root.join("leaf.sh"), LEAF).unwrap();
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(root.join("bin/cargo"), fs::Permissions::from_mode(0o700)).unwrap();
        let unrelated = Command::new("sleep").arg("30").spawn().unwrap();
        Self {
            root,
            wrapper: None,
            unrelated,
        }
    }

    fn launch(&mut self, stage: &str) {
        self.launch_probe(stage, "lib-release-live", "fixture::probe");
    }

    fn launch_probe(&mut self, stage: &str, target: &str, name: &str) {
        let mut command = Command::new("sh");
        command
            .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/test-ignored.sh"))
            .args([target, name]);
        self.launch_command(stage, name, command);
    }

    fn launch_command(&mut self, stage: &str, name: &str, mut command: Command) {
        let path = std::env::join_paths(
            std::iter::once(self.root.join("bin"))
                .chain(std::env::split_paths(&std::env::var_os("PATH").unwrap())),
        )
        .unwrap();
        self.wrapper = Some(
            command
                .current_dir(&self.root)
                .env("PATH", path)
                .env("FIXTURE_ROOT", &self.root)
                .env("FIXTURE_STAGE", stage)
                .env("FIXTURE_TEST", name)
                .env("MOESPRESSO_TEST_PACKAGE", "synthetic-package")
                .env("TMPDIR", self.root.join("tmp"))
                .stdout(fs::File::create(self.root.join("stdout")).unwrap())
                .stderr(Stdio::null())
                .spawn()
                .unwrap(),
        );
    }

    fn pid(&self, name: &str) -> i32 {
        let started = Instant::now();
        loop {
            if let Ok(value) = fs::read_to_string(self.root.join(name))
                && let Ok(pid) = value.trim().parse()
            {
                return pid;
            }
            assert!(started.elapsed() < Duration::from_secs(5), "missing {name}");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn wait(&mut self) -> std::process::ExitStatus {
        let started = Instant::now();
        loop {
            if let Some(status) = self.wrapper.as_mut().unwrap().try_wait().unwrap() {
                return status;
            }
            assert!(
                started.elapsed() < Duration::from_secs(8),
                "wrapper did not exit"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn assert_clean(&mut self, leader: i32, descendant: i32) {
        let started = Instant::now();
        while running(leader) || running(descendant) || running(-leader) {
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "owned processes survived"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(self.unrelated.try_wait().unwrap().is_none());
        assert_eq!(fs::read_dir(self.root.join("tmp")).unwrap().count(), 0);
    }
}

fn running(pid: i32) -> bool {
    // SAFETY: Signal zero observes only an exact fixture PID or its owned group.
    unsafe { libc::kill(pid, 0) == 0 }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if let Some(wrapper) = self.wrapper.as_mut() {
            if wrapper.try_wait().ok().flatten().is_none() {
                // SAFETY: Interrupt only this fixture's unreaped spawned wrapper.
                unsafe { libc::kill(wrapper.id() as i32, libc::SIGTERM) };
            }
            let started = Instant::now();
            while wrapper.try_wait().ok().flatten().is_none()
                && started.elapsed() < Duration::from_secs(4)
            {
                std::thread::sleep(Duration::from_millis(20));
            }
            let _ = wrapper.kill();
            let _ = wrapper.wait();
        }
        for name in ["leader", "descendant"] {
            if let Ok(value) = fs::read_to_string(self.root.join(name))
                && let Ok(pid) = value.trim().parse::<i32>()
                && running(pid)
            {
                // SAFETY: Last-resort cleanup targets only live fixture PIDs.
                unsafe { libc::kill(pid, libc::SIGKILL) };
            }
        }
        let _ = self.unrelated.kill();
        let _ = self.unrelated.wait();
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn live_wrapper_forwards_each_signal_to_its_owned_group_and_reaps() {
    for signal in [libc::SIGINT, libc::SIGHUP, libc::SIGTERM] {
        for stage in ["list", "run"] {
            let mut fixture = Fixture::new();
            fixture.launch(stage);
            let leader = fixture.pid("leader");
            let descendant = fixture.pid("descendant");
            // SAFETY: Deliberately signal only the wrapper, not its process group.
            assert_eq!(
                unsafe { libc::kill(fixture.wrapper.as_ref().unwrap().id() as i32, signal) },
                0
            );
            assert!(!fixture.wait().success());
            fixture.assert_clean(leader, descendant);
        }
    }
}

#[test]
fn live_wrapper_escalates_only_its_stubborn_group() {
    let mut fixture = Fixture::new();
    fixture.launch("stubborn");
    let leader = fixture.pid("leader");
    let descendant = fixture.pid("descendant");
    let started = Instant::now();
    // SAFETY: Only this fixture's wrapper receives interruption.
    unsafe { libc::kill(fixture.wrapper.as_ref().unwrap().id() as i32, libc::SIGTERM) };
    assert!(!fixture.wait().success());
    assert!(started.elapsed() >= Duration::from_millis(1800));
    fixture.assert_clean(leader, descendant);
}

#[test]
fn live_wrapper_refuses_missing_exact_test_before_inference() {
    let mut fixture = Fixture::new();
    fixture.launch("wrong");
    assert!(!fixture.wait().success());
    assert_eq!(
        fs::read_to_string(fixture.root.join("launches")).unwrap(),
        "list\n"
    );
    assert!(!fixture.root.join("leader").exists());
    assert_eq!(fs::read_dir(fixture.root.join("tmp")).unwrap().count(), 0);
    assert!(fixture.unrelated.try_wait().unwrap().is_none());
}
