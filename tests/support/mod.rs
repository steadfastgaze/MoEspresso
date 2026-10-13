pub mod cache;
pub mod opencode;
pub mod phase;

use std::fs::{self, File};
use std::io::Read;
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, ensure};
use serde_json::Value;
use ureq::{Agent, http::Response};

#[cfg(unix)]
use std::os::unix::fs::DirBuilderExt;

const BODY_LIMIT: u64 = 4 * 1024 * 1024;

pub struct Scratch {
    path: PathBuf,
    cleaned: bool,
}

impl Scratch {
    pub fn new(label: &str) -> Result<Self> {
        let parent = std::env::temp_dir();
        for attempt in 0..8u32 {
            let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
            let path = parent.join(format!(
                "moespresso-{label}-{}-{stamp}-{attempt}",
                std::process::id()
            ));
            let mut builder = fs::DirBuilder::new();
            #[cfg(unix)]
            builder.mode(0o700);
            match builder.create(&path) {
                Ok(()) => {
                    let path = match fs::canonicalize(&path) {
                        Ok(path) => path,
                        Err(error) => {
                            let _ = fs::remove_dir(&path);
                            return Err(error).with_context(|| {
                                format!("resolve acceptance directory {}", path.display())
                            });
                        }
                    };
                    return Ok(Self {
                        path,
                        cleaned: false,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error).with_context(|| format!("create {}", path.display()));
                }
            }
        }
        Err(anyhow!("could not create an isolated acceptance directory"))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn cleanup_all(&mut self) -> Result<()> {
        fs::remove_dir_all(&self.path)
            .with_context(|| format!("remove test acceptance directory {}", self.path.display()))?;
        self.cleaned = true;
        Ok(())
    }

    pub fn subdir(&self, name: &str) -> Result<PathBuf> {
        let path = self.path.join(name);
        fs::create_dir_all(&path).with_context(|| format!("create {}", path.display()))?;
        Ok(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if !self.cleaned {
            eprintln!("acceptance artifacts retained at {}", self.path.display());
        }
    }
}

#[derive(Clone, Copy)]
pub struct Deadline {
    end: Instant,
}

impl Deadline {
    pub fn after(duration: Duration) -> Result<Self> {
        Ok(Self {
            end: Instant::now()
                .checked_add(duration)
                .context("acceptance deadline overflow")?,
        })
    }

    pub fn remaining(self) -> Result<Duration> {
        self.end
            .checked_duration_since(Instant::now())
            .filter(|duration| !duration.is_zero())
            .context("acceptance run exceeded its total time limit")
    }

    pub fn bounded(self, maximum: Duration) -> Result<Duration> {
        Ok(self.remaining()?.min(maximum))
    }
}

pub fn package_path() -> Result<PathBuf> {
    let path = std::env::var_os("MOESPRESSO_TEST_PACKAGE")
        .context("MOESPRESSO_TEST_PACKAGE is required for this ignored acceptance test")?;
    let path = fs::canonicalize(PathBuf::from(path)).context("resolve MOESPRESSO_TEST_PACKAGE")?;
    ensure!(
        path.is_dir(),
        "MOESPRESSO_TEST_PACKAGE is not a directory: {}",
        path.display()
    );
    Ok(path)
}

pub fn binary_path() -> Result<PathBuf> {
    let path = std::env::var_os("MOESPRESSO_TEST_BINARY")
        .map(PathBuf::from)
        .or_else(|| option_env!("CARGO_BIN_EXE_moespresso").map(PathBuf::from))
        .context(
            "set MOESPRESSO_TEST_BINARY or run through Cargo with the moespresso binary target",
        )?;
    let path = fs::canonicalize(path).context("resolve native test binary")?;
    ensure!(
        path.is_file(),
        "test binary does not exist: {}",
        path.display()
    );
    Ok(path)
}

pub fn executable_path(env_name: &str, default_name: &str) -> Result<PathBuf> {
    if let Some(path) = std::env::var_os(env_name).map(PathBuf::from) {
        let path = fs::canonicalize(path).with_context(|| format!("resolve {env_name}"))?;
        ensure!(
            path.is_file(),
            "{env_name} is not a file: {}",
            path.display()
        );
        return Ok(path);
    }
    let path_var = std::env::var_os("PATH").context("PATH is required to find OpenCode")?;
    for directory in std::env::split_paths(&path_var) {
        let candidate = directory.join(default_name);
        if candidate.is_file() {
            return fs::canonicalize(candidate).context("resolve executable from PATH");
        }
    }
    Err(anyhow!(
        "{default_name} was not found on PATH; set {env_name}"
    ))
}

pub fn agent(timeout: Duration) -> Agent {
    let config = Agent::config_builder()
        .timeout_global(Some(timeout))
        .max_redirects(0)
        .proxy(None)
        .build();
    Agent::new_with_config(config)
}

#[cfg(target_os = "macos")]
pub fn limit_child_file_size(command: &mut Command, maximum: u64) {
    use std::os::unix::process::CommandExt;

    unsafe {
        command.pre_exec(move || {
            let limit = libc::rlimit {
                rlim_cur: maximum as libc::rlim_t,
                rlim_max: maximum as libc::rlim_t,
            };
            if libc::setrlimit(libc::RLIMIT_FSIZE, &limit) == 0 {
                Ok(())
            } else {
                Err(std::io::Error::last_os_error())
            }
        });
    }
}

#[cfg(not(target_os = "macos"))]
pub fn limit_child_file_size(_command: &mut Command, _maximum: u64) {}

fn response_json(mut response: Response<ureq::Body>) -> Result<Value> {
    ensure!(
        response.status().as_u16() == 200,
        "HTTP request returned {}",
        response.status()
    );
    let bytes = response
        .body_mut()
        .with_config()
        .limit(BODY_LIMIT)
        .read_to_vec()
        .context("read bounded HTTP response")?;
    serde_json::from_slice(&bytes).context("HTTP response is not JSON")
}

pub fn get_json(agent: &Agent, url: &str) -> Result<Value> {
    response_json(
        agent
            .get(url)
            .call()
            .with_context(|| format!("GET {url}"))?,
    )
}

pub fn post_json(agent: &Agent, url: &str, value: &Value) -> Result<Value> {
    let bytes = serde_json::to_vec(value)?;
    response_json(
        agent
            .post(url)
            .header("content-type", "application/json")
            .send(bytes.as_slice())
            .with_context(|| format!("POST {url}"))?,
    )
}

pub struct RunningServer {
    child: ChildGuard,
    origin: String,
    stdout: PathBuf,
    stderr: PathBuf,
}

impl RunningServer {
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        binary: &Path,
        package: &Path,
        scratch: &Scratch,
        cache_root: &Path,
        max_context: usize,
        thinking: &str,
        adaptive: bool,
        deadline: Deadline,
    ) -> Result<Self> {
        Self::start_configured(
            binary,
            package,
            scratch,
            cache_root,
            max_context,
            thinking,
            adaptive,
            deadline,
            "20",
            |_| {},
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn start_configured(
        binary: &Path,
        package: &Path,
        scratch: &Scratch,
        cache_root: &Path,
        max_context: usize,
        thinking: &str,
        adaptive: bool,
        deadline: Deadline,
        memory_gb: &str,
        configure: impl FnOnce(&mut Command),
    ) -> Result<Self> {
        let port = free_loopback_port()?;
        let origin = format!("http://127.0.0.1:{port}");
        let stdout = scratch.path().join(format!("server-{port}.stdout"));
        let stderr = scratch.path().join(format!("server-{port}.stderr"));
        let xdg = scratch.subdir("xdg")?;
        let stdout_file = File::create(&stdout)?;
        let stderr_file = File::create(&stderr)?;
        let mut command = Command::new(binary);
        command
            .arg("serve")
            .arg(package)
            .args([
                "--max-memory-gb",
                memory_gb,
                "--max-context",
                &max_context.to_string(),
                "--port",
                &port.to_string(),
                "--prompt-cache-gb",
                "1",
                "--thinking",
                thinking,
            ])
            .env("XDG_CACHE_HOME", cache_root)
            .env("XDG_CONFIG_HOME", xdg.join("config"))
            .env("XDG_DATA_HOME", xdg.join("data"))
            .env("XDG_STATE_HOME", xdg.join("state"))
            .stdout(Stdio::from(stdout_file))
            .stderr(Stdio::from(stderr_file));
        if !adaptive {
            command.args(["--no-adaptive-memory", "--cache-routing", "off"]);
        }
        configure(&mut command);
        if !command
            .get_args()
            .any(|argument| argument == "--no-disk-cache")
        {
            command.args(["--disk-cache-gb", "8"]);
        }
        let child = ChildGuard::new(command.spawn().context("start native acceptance server")?);
        let mut server = Self {
            child,
            origin,
            stdout,
            stderr,
        };
        server.wait_ready(deadline.bounded(Duration::from_secs(180))?)?;
        Ok(server)
    }

    pub fn origin(&self) -> &str {
        &self.origin
    }

    pub fn health(&self, timeout: Duration) -> Result<Value> {
        get_json(&agent(timeout), &format!("{}/health", self.origin))
    }

    fn wait_ready(&mut self, timeout: Duration) -> Result<()> {
        let end = Instant::now()
            .checked_add(timeout)
            .context("server startup deadline overflow")?;
        loop {
            if let Some(status) = self.child.try_wait()? {
                return Err(anyhow!(
                    "native server exited during startup ({status}): {}",
                    self.logs()?
                ));
            }
            let remaining = end
                .checked_duration_since(Instant::now())
                .filter(|duration| !duration.is_zero())
                .context("native server did not become healthy within 180 seconds")?;
            match get_json(
                &agent(remaining.min(Duration::from_secs(2))),
                &format!("{}/health", self.origin),
            ) {
                Ok(value) if value["status"] == "ok" => return Ok(()),
                _ => thread::sleep(Duration::from_millis(200).min(remaining)),
            }
        }
    }

    pub fn logs(&self) -> Result<String> {
        Ok(format!(
            "stdout:\n{}\nstderr:\n{}",
            read_bounded_text(&self.stdout, 1024 * 1024)?,
            read_bounded_text(&self.stderr, 1024 * 1024)?
        ))
    }
}

fn free_loopback_port() -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?.port())
}

pub struct ChildGuard {
    child: Option<Child>,
    process_group: bool,
}

impl ChildGuard {
    pub fn new(child: Child) -> Self {
        Self {
            child: Some(child),
            process_group: false,
        }
    }

    pub fn new_process_group(child: Child) -> Self {
        Self {
            child: Some(child),
            process_group: true,
        }
    }

    pub fn try_wait(&mut self) -> Result<Option<ExitStatus>> {
        self.child
            .as_mut()
            .context("child process was already consumed")?
            .try_wait()
            .context("check child process status")
    }

    pub fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            if self.process_group {
                kill_process_group(child.id());
            }
            if child.try_wait().ok().flatten().is_none() {
                let _ = child.kill();
            }
            let _ = child.wait();
        }
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        self.stop();
    }
}

pub fn wait_child(
    child: &mut ChildGuard,
    timeout: Duration,
    description: &str,
) -> Result<ExitStatus> {
    let end = Instant::now()
        .checked_add(timeout)
        .context("child deadline overflow")?;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        if Instant::now() >= end {
            child.stop();
            return Err(anyhow!("{description} exceeded its {timeout:?} timeout"));
        }
        thread::sleep(Duration::from_millis(100));
    }
}

#[cfg(target_os = "macos")]
fn kill_process_group(process_id: u32) {
    if let Ok(process_group) = i32::try_from(process_id) {
        // The group is created exclusively for this child before it is spawned.
        unsafe {
            libc::kill(-process_group, libc::SIGKILL);
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn kill_process_group(_process_id: u32) {}

pub fn read_bounded_text(path: &Path, maximum: usize) -> Result<String> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let limit = u64::try_from(maximum)?
        .checked_add(1)
        .context("output limit overflow")?;
    let mut bytes = Vec::new();
    file.take(limit)
        .read_to_end(&mut bytes)
        .with_context(|| format!("read {}", path.display()))?;
    ensure!(
        bytes.len() <= maximum,
        "{} exceeds its {maximum}-byte acceptance output limit",
        path.display()
    );
    String::from_utf8(bytes).with_context(|| format!("{} is not valid UTF-8", path.display()))
}

pub fn connect_loopback(address: &str, timeout: Duration) -> Result<TcpStream> {
    let address = address.parse().context("parse loopback server address")?;
    let stream =
        TcpStream::connect_timeout(&address, timeout).context("connect to loopback server")?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    Ok(stream)
}
