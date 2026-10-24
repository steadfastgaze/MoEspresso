//! Bounded command-line check of the private ANE runtime.

use anyhow::{Context, Result, anyhow};
use moespresso::ane::{CheckEvent, check};
use std::io::{self, Write};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const DEADLINE: Duration = Duration::from_secs(30);

// This watchdog is restricted to the standalone diagnostic process. Native
// calls and cleanup can block; terminating must not acquire stdio locks or run
// process-exit handlers that could wait on the blocked owner thread.
struct Deadline {
    cancel: Sender<()>,
    worker: JoinHandle<()>,
}

impl Deadline {
    fn arm(duration: Duration) -> Result<Self> {
        let (cancel, receiver) = mpsc::channel();
        let worker = thread::Builder::new()
            .name("ane-check-deadline".into())
            .spawn(move || {
                if matches!(
                    receiver.recv_timeout(duration),
                    Err(RecvTimeoutError::Timeout)
                ) {
                    // SAFETY: _exit terminates only this standalone process and
                    // deliberately skips cleanup on an unresponsive runtime.
                    unsafe { libc::_exit(124) }
                }
            })
            .context("start ANE check deadline before loading the runtime")?;
        Ok(Self { cancel, worker })
    }

    fn finish(self) -> Result<()> {
        let _ = self.cancel.send(());
        self.worker
            .join()
            .map_err(|_| anyhow!("ANE check deadline thread panicked"))
    }
}

fn progress(output: &mut impl Write, event: CheckEvent) -> io::Result<()> {
    if !event.completed && !matches!(event.stage, "prepare" | "verify") {
        if let Some(iteration) = event.iteration {
            writeln!(output, "  {} (evaluation {})", event.stage, iteration + 1)?;
        } else {
            writeln!(output, "  {}", event.stage)?;
        }
        output.flush()?;
    }
    Ok(())
}

pub(crate) fn run() -> Result<()> {
    let deadline = Deadline::arm(DEADLINE)?;
    let result = (|| {
        let mut output = io::stdout().lock();
        writeln!(
            output,
            "☕ Checking ANE (30-second deadline; timeout exits with status 124)."
        )?;
        output.flush()?;
        let report = check(|event| progress(&mut output, event))?;
        writeln!(
            output,
            "☕ ANE check passed: {} evaluations, {} exact values verified, {:.3}s.",
            report.evaluations, report.checked_elements, report.total_seconds
        )?;
        writeln!(
            output,
            "IOSurface storage: {} input bytes, {} output bytes. Explicit unload passed.",
            report.input_surface_bytes, report.output_surface_bytes
        )?;
        writeln!(
            output,
            "This checks runtime access on this Mac, not model performance or quality."
        )?;
        output.flush()?;
        Ok(())
    })();
    deadline.finish()?;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn completed_work_cancels_and_joins_the_deadline() {
        Deadline::arm(Duration::from_secs(5))
            .unwrap()
            .finish()
            .unwrap();
    }

    #[test]
    fn timeout_terminates_the_diagnostic_process() {
        let output = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "check_ane::tests::deadline_child", "--nocapture"])
            .env("MOESPRESSO_ANE_DEADLINE_TEST", "1")
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(124), "{output:?}");
    }

    #[test]
    fn deadline_child() {
        if std::env::var_os("MOESPRESSO_ANE_DEADLINE_TEST").is_none() {
            return;
        }
        let _deadline = Deadline::arm(Duration::from_millis(20)).unwrap();
        // Retain the sender while simulating a blocked native call.
        thread::sleep(Duration::from_secs(5));
        panic!("deadline did not terminate the child");
    }

    #[test]
    fn native_stage_progress_is_flushed_before_the_call() {
        let mut output = Vec::new();
        progress(
            &mut output,
            CheckEvent {
                stage: "evaluate",
                iteration: Some(1),
                completed: false,
                checked_elements: 0,
                elapsed_seconds: 0.0,
            },
        )
        .unwrap();
        assert_eq!(output, b"  evaluate (evaluation 2)\n");
    }
}
