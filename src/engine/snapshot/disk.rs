//! Explicit-root native prompt checkpoints.

mod store;
mod stream;
pub use store::{DiskCheckpointStore, StoreLimits};
pub use stream::IO_BUFFER_BYTES;

use stream::{read_session, ring_payload_bytes, write_session};

use std::ffi::CString;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::Path;
use std::rc::Rc;

use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};

use super::{PRODUCER_VERSION, VOCAB};
#[cfg(test)]
use super::{SavedLayer, SessionSnapshot};
use crate::engine::{Engine, ProducerSpan, Session};
use crate::kvarn::{
    SINK_BYTES, SINK_TOKENS, TAIL_BYTES, TILE_RECORD_BYTES, TILE_TOKENS, partition,
};
use crate::model::{MixerKind, QwenConfig};
use crate::qsa_index::{COMPRESS_RATIO, INDEX_WIDTH};
use crate::tokenizer::TextTokenizer;

/// Disk write and retention policy; checkpoint payloads are shared by both modes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DiskCacheMode {
    #[default]
    Prefix,
    Full,
}

impl DiskCacheMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Prefix => "prefix",
            Self::Full => "full",
        }
    }
}

const MAGIC: &[u8; 16] = b"MOESP4NATIVECHK1";
const WIRE_VERSION: u32 = 2;
const HEADER_BYTES: usize = 120;
const SECTION_HEADER_BYTES: u64 = 12;
const TRAILER_BYTES: u64 = 32;
const CHUNK_BYTES: usize = 8192;
const PLE_WORDS: usize = 9 * 10240;
const GDN_CONV_BYTES: usize = 3 * 10240 * 2;
const GDN_RECURRENT_BYTES: usize = 48 * 128 * 128 * 4;
const SLOT: &str = "checkpoint.bin";
const TEMP: &str = "checkpoint.tmp";
const LOCK: &str = "checkpoint.lock";

const TOKENS: u16 = 1;
const LOGITS: u16 = 2;
const PLE: u16 = 3;
const GDN_CONV: u16 = 4;
const GDN_RECURRENT: u16 = 5;
const QSA_COMPRESSED: u16 = 6;
const QSA_PENDING: u16 = 7;
const KVARN_RECORDS: u16 = 8;
const KVARN_SINK_KEYS: u16 = 9;
const KVARN_SINK_VALUES: u16 = 10;
const KVARN_TAIL_KEYS: u16 = 11;
const KVARN_TAIL_VALUES: u16 = 12;

/// A persisted checkpoint failed a deterministic integrity or semantic check.
/// I/O, allocation and live restore failures intentionally use other errors.
#[derive(Debug)]
struct InvalidCheckpoint(&'static str);

impl std::fmt::Display for InvalidCheckpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for InvalidCheckpoint {}

/// Callers supply both limits. Neither is a process-memory or disk-space peak.
#[derive(Clone, Copy)]
pub struct DiskLimits {
    pub max_file_bytes: u64,
    pub max_io_buffer_bytes: u64,
}

/// One locked native slot. Its root and lock inode stay pinned for its lifetime.
pub struct DiskCheckpointSlot {
    root: File,
    _lock: File,
    owner: Rc<()>,
    scope: [u8; 32],
    limits: DiskLimits,
}

fn checked_add(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right)
        .context("checkpoint byte count overflow")
}

fn checked_bytes(count: usize, width: usize) -> Result<u64> {
    Ok(u64::try_from(
        count
            .checked_mul(width)
            .context("checkpoint section size overflow")?,
    )?)
}

fn hash_part(hash: &mut Sha256, bytes: &[u8]) -> Result<()> {
    hash.update(u64::try_from(bytes.len())?.to_le_bytes());
    hash.update(bytes);
    Ok(())
}

fn scope(engine: &Engine, tokenizer: &TextTokenizer) -> Result<[u8; 32]> {
    let (manifest, assets) = tokenizer.checkpoint_identity();
    ensure!(
        manifest == engine.package_manifest_digest,
        "tokenizer belongs to another package"
    );
    let mut hash = Sha256::new();
    hash_part(&mut hash, b"moespresso-native-qwen4-checkpoint-scope-v4")?;
    hash_part(&mut hash, &manifest)?;
    hash_part(&mut hash, &assets)?;
    hash_part(&mut hash, &WIRE_VERSION.to_le_bytes())?;
    hash_part(&mut hash, &PRODUCER_VERSION.to_le_bytes())?;
    hash_part(&mut hash, &engine.config.eos_token_id.to_le_bytes())?;
    hash_part(&mut hash, &engine.config.eps.to_bits().to_le_bytes())?;
    hash_part(
        &mut hash,
        &u64::try_from(engine.config.max_positions)?.to_le_bytes(),
    )?;
    for kind in &engine.config.layers {
        hash_part(
            &mut hash,
            &[match kind {
                MixerKind::Gdn => 1,
                MixerKind::Qsa => 2,
            }],
        )?;
    }
    Ok(hash.finalize().into())
}

fn prefix_hash(tokens: &[u32]) -> [u8; 32] {
    let mut hash = Sha256::new();
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
    hash.finalize().into()
}

fn regular_owned(file: &File) -> Result<()> {
    let metadata = file.metadata()?;
    ensure!(metadata.is_file(), "checkpoint leaf is not a regular file");
    ensure!(
        metadata.uid() == unsafe { libc::geteuid() },
        "checkpoint leaf has another owner"
    );
    ensure!(metadata.nlink() == 1, "checkpoint leaf has multiple links");
    ensure!(
        metadata.permissions().mode() & 0o077 == 0,
        "checkpoint leaf is not private"
    );
    Ok(())
}

fn open_leaf(root: &File, name: &str, flags: i32) -> Result<File> {
    let name = CString::new(name)?;
    let fd = unsafe {
        libc::openat(
            root.as_raw_fd(),
            name.as_ptr(),
            flags | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
            0o600,
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let file = unsafe { File::from_raw_fd(fd) };
    regular_owned(&file)?;
    Ok(file)
}

fn open_root(path: &Path) -> Result<File> {
    ensure!(path.is_absolute(), "checkpoint root must be absolute");
    let root_metadata = fs::symlink_metadata(path)?;
    ensure!(
        root_metadata.is_dir() && !root_metadata.file_type().is_symlink(),
        "checkpoint root is not a directory"
    );
    let canonical = path.canonicalize()?;
    let raw = CString::new(canonical.as_os_str().as_bytes())?;
    let fd = unsafe {
        libc::open(
            raw.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let file = unsafe { File::from_raw_fd(fd) };
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_dir() && metadata.uid() == unsafe { libc::geteuid() },
        "checkpoint root is not owned by this process"
    );
    ensure!(
        metadata.dev() == root_metadata.dev() && metadata.ino() == root_metadata.ino(),
        "checkpoint root changed while opening"
    );
    ensure!(
        metadata.permissions().mode() & 0o077 == 0,
        "checkpoint root is not private"
    );
    Ok(file)
}

fn remove_stale_temp(root: &File) -> Result<()> {
    match open_leaf(root, TEMP, libc::O_RDONLY) {
        Ok(file) => drop(file),
        Err(err)
            if err
                .downcast_ref::<std::io::Error>()
                .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound) =>
        {
            return Ok(());
        }
        Err(err) => return Err(err).context("invalid checkpoint temporary leaf"),
    }
    let name = CString::new(TEMP)?;
    let status = unsafe { libc::unlinkat(root.as_raw_fd(), name.as_ptr(), 0) };
    ensure!(status == 0, "cannot remove stale checkpoint temporary leaf");
    Ok(())
}

fn check_existing_slot(root: &File) -> Result<()> {
    match open_leaf(root, SLOT, libc::O_RDONLY) {
        Ok(file) => {
            drop(file);
            Ok(())
        }
        Err(err)
            if err
                .downcast_ref::<std::io::Error>()
                .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound) =>
        {
            Ok(())
        }
        Err(err) => Err(err).context("invalid checkpoint slot leaf"),
    }
}

fn publish_temp_inner(
    root: &File,
    expected: u64,
    sync_directory: fn(&File) -> std::io::Result<()>,
) -> Result<()> {
    let temp = open_leaf(root, TEMP, libc::O_RDONLY)?;
    ensure!(
        temp.metadata()?.len() == expected,
        "checkpoint temporary length differs"
    );
    check_existing_slot(root)?;
    drop(temp);
    let from = CString::new(TEMP)?;
    let to = CString::new(SLOT)?;
    let status = unsafe {
        libc::renameat(
            root.as_raw_fd(),
            from.as_ptr(),
            root.as_raw_fd(),
            to.as_ptr(),
        )
    };
    ensure!(
        status == 0,
        "checkpoint atomic rename failed: {}",
        std::io::Error::last_os_error()
    );
    sync_directory(root).context(
        "checkpoint rename completed but directory sync failed; durability is uncertain",
    )?;
    Ok(())
}

fn publish_temp(root: &File, expected: u64) -> Result<()> {
    publish_temp_inner(root, expected, File::sync_all)
}

impl DiskCheckpointSlot {
    /// Open one explicit private directory and hold its nonblocking lock.
    pub fn open(
        root: &Path,
        engine: &Engine,
        tokenizer: &TextTokenizer,
        limits: DiskLimits,
    ) -> Result<Self> {
        ensure!(
            limits.max_file_bytes > (HEADER_BYTES as u64 + TRAILER_BYTES)
                && limits.max_io_buffer_bytes >= TILE_RECORD_BYTES as u64,
            "checkpoint limits must cover a header and one packed record of I/O staging"
        );
        let scope = scope(engine, tokenizer)?;
        let root = open_root(root)?;
        let lock = open_leaf(&root, LOCK, libc::O_RDWR | libc::O_CREAT)?;
        lock.try_lock()
            .context("checkpoint root is already locked")?;
        Ok(Self {
            root,
            _lock: lock,
            owner: engine.cache_identity.clone(),
            scope,
            limits,
        })
    }

    fn check_owner(&self, engine: &Engine) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.owner, &engine.cache_identity),
            "checkpoint slot belongs to another engine"
        );
        Ok(())
    }
}

fn expected_layout(
    config: &QwenConfig,
    max_context: usize,
    frontier: usize,
) -> Result<(u64, u32, u64)> {
    ensure!(
        frontier > 0 && frontier <= max_context && frontier <= config.max_positions,
        "checkpoint frontier is empty or exceeds context"
    );
    let mut payload = checked_add(checked_bytes(frontier, 4)?, checked_bytes(VOCAB, 2)?)?;
    payload = checked_add(payload, checked_bytes(PLE_WORDS, 2)?)?;
    let mut sections = 3u32;
    let records = partition(frontier).1.saturating_sub(SINK_TOKENS) / TILE_TOKENS;
    let record_bytes = checked_bytes(records, TILE_RECORD_BYTES)?;
    let compressed_bytes = checked_bytes(frontier / COMPRESS_RATIO * INDEX_WIDTH, 2)?;
    let pending_bytes = checked_bytes(frontier % COMPRESS_RATIO * INDEX_WIDTH, 2)?;
    let (sink, tail) = ring_payload_bytes(frontier);
    for kind in &config.layers {
        match kind {
            MixerKind::Gdn => {
                payload = checked_add(
                    payload,
                    u64::try_from(GDN_CONV_BYTES + GDN_RECURRENT_BYTES)?,
                )?;
                sections = sections
                    .checked_add(2)
                    .context("checkpoint section count overflow")?;
            }
            MixerKind::Qsa => {
                for bytes in [
                    compressed_bytes,
                    pending_bytes,
                    record_bytes,
                    u64::try_from(sink)?,
                    u64::try_from(sink)?,
                    u64::try_from(tail)?,
                    u64::try_from(tail)?,
                ] {
                    payload = checked_add(payload, bytes)?;
                }
                sections = sections
                    .checked_add(7)
                    .context("checkpoint section count overflow")?;
            }
        }
    }
    let framing = u64::from(sections)
        .checked_mul(SECTION_HEADER_BYTES)
        .context("checkpoint framing overflow")?;
    let file_bytes = checked_add(
        checked_add(HEADER_BYTES as u64, framing)?,
        checked_add(payload, TRAILER_BYTES)?,
    )?;
    Ok((file_bytes, sections, payload))
}

fn header(
    scope: &[u8; 32],
    tokens: &[u32],
    engine: &Engine,
    file_bytes: u64,
    sections: u32,
) -> Result<[u8; HEADER_BYTES]> {
    let mut result = [0u8; HEADER_BYTES];
    result[..16].copy_from_slice(MAGIC);
    result[16..20].copy_from_slice(&WIRE_VERSION.to_le_bytes());
    result[20..52].copy_from_slice(scope);
    result[52..84].copy_from_slice(&prefix_hash(tokens));
    result[84..92].copy_from_slice(&u64::try_from(tokens.len())?.to_le_bytes());
    result[92..100].copy_from_slice(&file_bytes.to_le_bytes());
    result[100..104].copy_from_slice(&sections.to_le_bytes());
    result[108..112].copy_from_slice(&u32::try_from(engine.max_context)?.to_le_bytes());
    result[112..116].copy_from_slice(&PRODUCER_VERSION.to_le_bytes());
    result[116..120].copy_from_slice(&engine.config.eos_token_id.to_le_bytes());
    Ok(result)
}

fn field_u32(bytes: &[u8], start: usize) -> u32 {
    u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap())
}

fn field_u64(bytes: &[u8], start: usize) -> u64 {
    u64::from_le_bytes(bytes[start..start + 8].try_into().unwrap())
}

fn validate_header_geometry(
    head: &[u8; HEADER_BYTES],
    file_bytes: u64,
    expected: u64,
    sections: u32,
    eos: u32,
) -> Result<()> {
    ensure!(
        field_u64(head, 92) == expected
            && file_bytes == expected
            && field_u32(head, 100) == sections
            && field_u32(head, 104) == 0
            && u64::from(field_u32(head, 108)) >= field_u64(head, 84)
            && field_u32(head, 112) == PRODUCER_VERSION
            && field_u32(head, 116) == eos,
        "checkpoint header geometry or length differs"
    );
    Ok(())
}

struct HashWriter<'a> {
    file: &'a mut File,
    hash: Sha256,
    written: u64,
}

impl<'a> HashWriter<'a> {
    fn new(file: &'a mut File) -> Self {
        Self {
            file,
            hash: Sha256::new(),
            written: 0,
        }
    }
    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        self.file.write_all(bytes)?;
        self.hash.update(bytes);
        self.written = checked_add(self.written, u64::try_from(bytes.len())?)?;
        Ok(())
    }
    fn section(&mut self, kind: u16, layer: u16, bytes: u64) -> Result<()> {
        let mut header = [0u8; SECTION_HEADER_BYTES as usize];
        header[..2].copy_from_slice(&kind.to_le_bytes());
        header[2..4].copy_from_slice(&layer.to_le_bytes());
        header[4..12].copy_from_slice(&bytes.to_le_bytes());
        self.write(&header)
    }
    fn u32_section(&mut self, kind: u16, layer: u16, values: &[u32], expected: u64) -> Result<()> {
        ensure!(
            checked_bytes(values.len(), 4)? == expected,
            "checkpoint source section shape differs"
        );
        self.section(kind, layer, expected)?;
        self.words32(values)
    }
    fn words32(&mut self, words: &[u32]) -> Result<()> {
        let mut chunk = [0u8; CHUNK_BYTES];
        for block in words.chunks(CHUNK_BYTES / 4) {
            for (index, word) in block.iter().enumerate() {
                chunk[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
            self.write(&chunk[..block.len() * 4])?;
        }
        Ok(())
    }
    fn finish(self) -> Result<()> {
        self.file.write_all(&self.hash.finalize())?;
        Ok(())
    }
}

struct HashReader<'a> {
    file: &'a mut File,
    hash: Sha256,
}

impl<'a> HashReader<'a> {
    fn new(file: &'a mut File) -> Self {
        Self {
            file,
            hash: Sha256::new(),
        }
    }
    fn read(&mut self, bytes: &mut [u8]) -> Result<()> {
        self.file.read_exact(bytes)?;
        self.hash.update(bytes);
        Ok(())
    }
    fn section(&mut self, kind: u16, layer: u16, bytes: u64) -> Result<()> {
        let mut header = [0u8; SECTION_HEADER_BYTES as usize];
        self.read(&mut header)?;
        if u16::from_le_bytes(header[..2].try_into().unwrap()) != kind
            || u16::from_le_bytes(header[2..4].try_into().unwrap()) != layer
            || u64::from_le_bytes(header[4..12].try_into().unwrap()) != bytes
        {
            return Err(
                InvalidCheckpoint("checkpoint section type, order or length differs").into(),
            );
        }
        Ok(())
    }
    fn finish(self) -> Result<()> {
        let mut trailer = [0u8; 32];
        self.file.read_exact(&mut trailer)?;
        if trailer != self.hash.finalize().as_slice() {
            return Err(InvalidCheckpoint("checkpoint digest differs").into());
        }
        let mut extra = [0u8; 1];
        if self.file.read(&mut extra)? != 0 {
            return Err(InvalidCheckpoint("checkpoint has trailing bytes").into());
        }
        Ok(())
    }
}

impl DiskCheckpointSlot {
    /// Save a completed conversation frontier into the one slot.
    /// A declined size admission returns false without replacing the old slot.
    pub fn save(&mut self, engine: &Engine, session: &Session) -> Result<bool> {
        self.check_owner(engine)?;
        engine.validate_snapshot_source(session)?;
        ensure!(
            Rc::ptr_eq(&session.cache_origin, &engine.cache_identity),
            "session cache producer differs from engine"
        );
        let frontier = session.frontier;
        let (expected, sections, _) =
            expected_layout(&engine.config, engine.max_context, frontier)?;
        if expected > self.limits.max_file_bytes {
            return Ok(false);
        }
        check_existing_slot(&self.root)?;
        remove_stale_temp(&self.root)?;
        let mut temp = open_leaf(
            &self.root,
            TEMP,
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
        )?;
        write_session(
            &mut temp,
            engine,
            session,
            &self.scope,
            expected,
            sections,
            self.limits.max_io_buffer_bytes,
        )?;
        temp.sync_all()?;
        drop(temp);
        publish_temp(&self.root, expected)?;
        Ok(true)
    }

    /// Return a fully restored Session only after exact prefix and state checks.
    pub fn load_match(&self, engine: &Engine, requested_tokens: &[u32]) -> Result<Option<Session>> {
        self.check_owner(engine)?;
        let mut slot = match open_leaf(&self.root, SLOT, libc::O_RDONLY) {
            Ok(slot) => slot,
            Err(err)
                if err
                    .downcast_ref::<std::io::Error>()
                    .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound) =>
            {
                return Ok(None);
            }
            Err(err) => return Err(err).context("open checkpoint slot"),
        };
        read_session(
            &mut slot,
            engine,
            &self.scope,
            requested_tokens,
            self.limits,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom};
    use std::os::unix::fs::symlink;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TempRoot(std::path::PathBuf);

    impl TempRoot {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let root = std::env::temp_dir().join(format!(
                "moespresso-native-checkpoint-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
            Self(root)
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn committed_layout_is_compact_and_context_bounded() -> Result<()> {
        let config = QwenConfig {
            layers: (0..48)
                .map(|index| {
                    if index % 4 == 0 {
                        MixerKind::Qsa
                    } else {
                        MixerKind::Gdn
                    }
                })
                .collect(),
            ple_layer_index: 1,
            eps: 1e-6,
            eos_token_id: 248_044,
            max_positions: 262_144,
        };
        assert_eq!(
            expected_layout(&config, 1024, 512)?,
            (129_119_244, 159, 129_117_184)
        );
        for frontier in [0, 1025, usize::MAX] {
            assert!(expected_layout(&config, 1024, frontier).is_err());
        }
        for frontier in [1, 27, 511, 513, 1024] {
            assert!(expected_layout(&config, 1024, frontier).is_ok());
        }
        assert!(expected_layout(&config, 262_144, 262_144)?.0 > 324_154_380);
        Ok(())
    }

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
    fn wide_prefill_memory_disk_roundtrip_and_cross_width_reuse() -> Result<()> {
        wide_prefill_roundtrip(2048, 2048)
    }

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
    fn wide_prefill_packed_memory_disk_roundtrip_and_cross_width_reuse() -> Result<()> {
        wide_prefill_roundtrip(2048, 10240)
    }

    #[test]
    #[ignore = "requires MOESPRESSO_TEST_PACKAGE and exclusive GPU ownership"]
    fn explicit_4096_prefill_memory_disk_roundtrip_and_cross_width_reuse() -> Result<()> {
        wide_prefill_roundtrip(4096, 4096)?;
        wide_prefill_roundtrip(4096, 10240)
    }

    fn wide_prefill_roundtrip(rows: usize, prefix: usize) -> Result<()> {
        use crate::engine::PromptPrefixCache;
        use crate::package::Package;
        use crate::routing::RoutingMode;
        fn fingerprint(snapshot: &SessionSnapshot) -> [u8; 32] {
            let mut hash = Sha256::new();
            hash.update(snapshot.frontier.to_le_bytes());
            for value in &snapshot.consumed_tokens {
                hash.update(value.to_le_bytes());
            }
            for value in &snapshot.last_logits_bf16 {
                hash.update(value.to_le_bytes());
            }
            hash.update(snapshot.ple.offset.to_le_bytes());
            for value in snapshot.ple.context {
                hash.update(value.to_le_bytes());
            }
            for value in &snapshot.ple.conv {
                hash.update(value.to_le_bytes());
            }
            for state in &snapshot.layers {
                match state {
                    SavedLayer::Gdn(state) => {
                        hash.update(state.frontier.to_le_bytes());
                        hash.update(&state.conv_bf16);
                        hash.update(&state.recurrent_f32);
                    }
                    SavedLayer::Qsa(state) => {
                        hash.update(state.index_frontier.to_le_bytes());
                        for value in &state.compressed_index_bf16 {
                            hash.update(value.to_le_bytes());
                        }
                        for value in &state.pending_index_bf16 {
                            hash.update(value.to_le_bytes());
                        }
                        hash.update(state.kv.frontier.to_le_bytes());
                        for bytes in [
                            &state.kv.records,
                            &state.kv.sink_keys,
                            &state.kv.sink_values,
                            &state.kv.tail_keys,
                            &state.kv.tail_values,
                        ] {
                            hash.update(bytes);
                        }
                    }
                }
            }
            hash.finalize().into()
        }
        let package = Package::open(std::env::var("MOESPRESSO_TEST_PACKAGE")?)?;
        let tokenizer = TextTokenizer::load(&package)?;
        let slots = std::env::var("MOESPRESSO_TEST_EXPERT_CAPACITY")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()?
            .unwrap_or(178);
        let slot_root = TempRoot::new();
        let store_root = TempRoot::new();
        let limits = DiskLimits {
            max_file_bytes: 1 << 30,
            max_io_buffer_bytes: IO_BUFFER_BYTES,
        };
        let store_limits = StoreLimits {
            mode: DiskCacheMode::Full,
            max_entries: 2,
            max_payload_bytes: 1 << 30,
            checkpoint: limits,
        };
        let tokens: Vec<u32> = (7..u32::try_from(prefix + 9)?).collect();
        let context = prefix + 2048;
        let wide_scope;
        {
            let mut engine = Engine::load_with_prefill_capacity(&package, slots, context, rows)?;
            wide_scope = scope(&engine, &tokenizer)?;
            let mut session = engine.start_session()?;
            for chunk in tokens[..prefix].chunks(rows) {
                engine.prefill_chunk(chunk, &mut session)?;
            }
            let expected = {
                let snapshot = engine.snapshot(&session)?;
                let tiles = partition(prefix).1.saturating_sub(SINK_TOKENS) / TILE_TOKENS;
                if prefix == 10240 {
                    assert!(tiles > 0);
                }
                for state in &snapshot.layers {
                    if let SavedLayer::Qsa(state) = state {
                        assert_eq!(state.kv.records.len(), tiles * TILE_RECORD_BYTES);
                    }
                }
                fingerprint(&snapshot)
            };
            let mut memory = PromptPrefixCache::new(&engine, 1 << 30);
            assert!(memory.offer_prompt_checkpoint(&engine, &session, &tokens[..prefix + 1])?);
            let mut resumed = engine.restore(
                memory
                    .best_match(&engine, &tokens[..prefix + 1])
                    .unwrap()
                    .snapshot(),
            )?;
            assert_eq!(fingerprint(&engine.snapshot(&resumed)?), expected);
            {
                let mut slot = DiskCheckpointSlot::open(&slot_root.0, &engine, &tokenizer, limits)?;
                assert!(slot.save(&engine, &session)?);
                let mut store =
                    DiskCheckpointStore::open(&store_root.0, &engine, &tokenizer, store_limits)?;
                assert!(store.save(&engine, &session)?);
            }
            let slot = DiskCheckpointSlot::open(&slot_root.0, &engine, &tokenizer, limits)?;
            // Allocation failure partway through restore drops the unpublished
            // candidate without damaging the checkpoint or live engine state.
            let live = engine.memory_stats()?.live_requested_buffer_bytes;
            let fault = engine.metal.test_fail_allocation_after(4)?;
            assert!(slot.load_match(&engine, &tokens).is_err());
            drop(fault);
            assert_eq!(engine.memory_stats()?.live_requested_buffer_bytes, live);
            let disk_session = slot
                .load_match(&engine, &tokens)?
                .context("missing wide slot checkpoint")?;
            assert_eq!(fingerprint(&engine.snapshot(&disk_session)?), expected);
            let mut store =
                DiskCheckpointStore::open(&store_root.0, &engine, &tokenizer, store_limits)?;
            let mut disk_resumed = store
                .load_match(&engine, &tokens)?
                .context("missing wide indexed checkpoint")?;
            assert_eq!(fingerprint(&engine.snapshot(&disk_resumed)?), expected);
            for candidate in [&mut session, &mut resumed, &mut disk_resumed] {
                engine.step(tokens[prefix], RoutingMode::Unbiased, candidate)?;
            }
            let terminal = fingerprint(&engine.snapshot(&session)?);
            assert_eq!(fingerprint(&engine.snapshot(&resumed)?), terminal);
            assert_eq!(fingerprint(&engine.snapshot(&disk_resumed)?), terminal);
            assert!(memory.offer_prompt_checkpoint(&engine, &session, &tokens[..prefix + 1])?);
            assert!(
                memory
                    .best_match(&engine, &tokens[..prefix + 1])
                    .unwrap()
                    .is_complete()
            );
            assert_eq!(
                memory
                    .best_match(&engine, &tokens)
                    .unwrap()
                    .matched_tokens(),
                prefix + 1
            );
            assert!(store.save(&engine, &session)?);
        }
        let engine = Engine::load_with_prefill_capacity(&package, slots, context, 512)?;
        assert_eq!(wide_scope, scope(&engine, &tokenizer)?);
        let slot = DiskCheckpointSlot::open(&slot_root.0, &engine, &tokenizer, limits)?;
        assert_eq!(
            slot.load_match(&engine, &tokens)?.unwrap().frontier()?,
            prefix
        );
        let mut store =
            DiskCheckpointStore::open(&store_root.0, &engine, &tokenizer, store_limits)?;
        assert_eq!(
            store.load_match(&engine, &tokens)?.unwrap().frontier()?,
            prefix + 1
        );
        Ok(())
    }

    #[test]
    fn header_rejects_geometry_and_producer_mismatches() -> Result<()> {
        let mut head = [0u8; HEADER_BYTES];
        head[92..100].copy_from_slice(&1000u64.to_le_bytes());
        head[100..104].copy_from_slice(&159u32.to_le_bytes());
        head[84..92].copy_from_slice(&512u64.to_le_bytes());
        head[108..112].copy_from_slice(&1024u32.to_le_bytes());
        head[112..116].copy_from_slice(&PRODUCER_VERSION.to_le_bytes());
        head[116..120].copy_from_slice(&248_044u32.to_le_bytes());
        validate_header_geometry(&head, 1000, 1000, 159, 248_044)?;
        for (start, replacement) in [
            (92, 1001u64.to_le_bytes().to_vec()),
            (100, 158u32.to_le_bytes().to_vec()),
            (104, 1u32.to_le_bytes().to_vec()),
            (108, 511u32.to_le_bytes().to_vec()),
            (112, 1u32.to_le_bytes().to_vec()),
            (112, (PRODUCER_VERSION - 1).to_le_bytes().to_vec()),
            (112, (PRODUCER_VERSION + 1).to_le_bytes().to_vec()),
            (116, 248_043u32.to_le_bytes().to_vec()),
        ] {
            let mut corrupt = head;
            corrupt[start..start + replacement.len()].copy_from_slice(&replacement);
            assert!(validate_header_geometry(&corrupt, 1000, 1000, 159, 248_044).is_err());
        }
        assert!(validate_header_geometry(&head, 999, 1000, 159, 248_044).is_err());
        Ok(())
    }

    #[test]
    fn streamed_digest_rejects_corruption_truncation_and_trailing_bytes() -> Result<()> {
        let root = TempRoot::new();
        let path = root.0.join(SLOT);
        let mut file = File::create(&path)?;
        let mut writer = HashWriter::new(&mut file);
        writer.write(b"header")?;
        writer.section(TOKENS, 0, 4)?;
        writer.words32(&[7])?;
        writer.finish()?;
        drop(file);
        let validate = || -> Result<()> {
            let mut file = File::open(&path)?;
            let mut reader = HashReader::new(&mut file);
            let mut head = [0; 6];
            reader.read(&mut head)?;
            ensure!(&head == b"header", "test header differs");
            reader.section(TOKENS, 0, 4)?;
            let mut token = [0; 4];
            reader.read(&mut token)?;
            ensure!(u32::from_le_bytes(token) == 7, "test token differs");
            reader.finish()
        };
        validate()?;
        let original = fs::read(&path)?;
        let mut file = fs::OpenOptions::new().write(true).open(&path)?;
        file.seek(SeekFrom::End(-1))?;
        file.write_all(&[0xff])?;
        drop(file);
        assert!(
            validate()
                .unwrap_err()
                .downcast_ref::<InvalidCheckpoint>()
                .is_some()
        );
        fs::write(&path, &original)?;
        let file = fs::OpenOptions::new().write(true).open(&path)?;
        file.set_len(6 + SECTION_HEADER_BYTES + 4 + TRAILER_BYTES - 1)?;
        drop(file);
        assert!(validate().is_err());
        fs::write(&path, &original)?;
        let mut file = fs::OpenOptions::new().append(true).open(&path)?;
        file.write_all(&[0])?;
        drop(file);
        assert!(
            validate()
                .unwrap_err()
                .downcast_ref::<InvalidCheckpoint>()
                .is_some()
        );
        Ok(())
    }

    #[test]
    fn failed_temporary_publication_preserves_old_slot() -> Result<()> {
        let root = TempRoot::new();
        let directory = open_root(&root.0)?;
        fs::write(root.0.join(SLOT), b"old")?;
        fs::set_permissions(root.0.join(SLOT), fs::Permissions::from_mode(0o600))?;
        fs::write(root.0.join(TEMP), b"new")?;
        fs::set_permissions(root.0.join(TEMP), fs::Permissions::from_mode(0o600))?;
        assert!(publish_temp(&directory, 4).is_err());
        assert_eq!(fs::read(root.0.join(SLOT))?, b"old");
        publish_temp(&directory, 3)?;
        assert_eq!(fs::read(root.0.join(SLOT))?, b"new");
        assert!(!root.0.join(TEMP).exists());
        Ok(())
    }

    #[test]
    fn post_rename_sync_failure_keeps_new_final_slot() -> Result<()> {
        fn fail_sync(_: &File) -> std::io::Result<()> {
            Err(std::io::Error::other("injected directory sync failure"))
        }
        let root = TempRoot::new();
        let directory = open_root(&root.0)?;
        fs::write(root.0.join(SLOT), b"old")?;
        fs::set_permissions(root.0.join(SLOT), fs::Permissions::from_mode(0o600))?;
        fs::write(root.0.join(TEMP), b"new")?;
        fs::set_permissions(root.0.join(TEMP), fs::Permissions::from_mode(0o600))?;
        let error = publish_temp_inner(&directory, 3, fail_sync).unwrap_err();
        assert!(error.to_string().contains("durability is uncertain"));
        assert_eq!(fs::read(root.0.join(SLOT))?, b"new");
        assert!(!root.0.join(TEMP).exists());
        Ok(())
    }

    #[test]
    fn private_root_lock_and_fixed_leaves_reject_symlinks_and_fifo() -> Result<()> {
        let root = TempRoot::new();
        let directory = open_root(&root.0)?;
        let lock = open_leaf(&directory, LOCK, libc::O_RDWR | libc::O_CREAT)?;
        lock.try_lock()?;
        let second = open_leaf(&directory, LOCK, libc::O_RDWR)?;
        assert!(second.try_lock().is_err());
        drop(second);
        symlink(&root.0, root.0.join("root-link"))?;
        assert!(open_root(&root.0.join("root-link")).is_err());
        symlink(root.0.join(LOCK), root.0.join(SLOT))?;
        assert!(check_existing_slot(&directory).is_err());
        symlink(root.0.join(LOCK), root.0.join(TEMP))?;
        assert!(remove_stale_temp(&directory).is_err());
        fs::remove_file(root.0.join(TEMP))?;
        let fifo = CString::new(root.0.join(TEMP).as_os_str().as_bytes())?;
        ensure!(
            unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) } == 0,
            "create FIFO test leaf"
        );
        assert!(remove_stale_temp(&directory).is_err());
        Ok(())
    }
}
