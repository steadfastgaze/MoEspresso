//! Bounded, target-only index over the native prompt-checkpoint wire.

use std::collections::BTreeSet;
use std::ffi::CString;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::AsRawFd;
use std::path::Path;
use std::rc::Rc;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use super::*;

mod managed;
use managed::ManagedCache;

const INDEX: &str = "index.json";
const INDEX_TEMP: &str = "index.tmp";
const PAYLOAD_TEMP: &str = "payload.tmp";
const INDEX_VERSION: u32 = 1;
const MAX_ENTRIES: usize = 64;
const PAYLOAD_SLOTS: usize = MAX_ENTRIES + 1;
const MAX_INDEX_BYTES: u64 = 32 * 1024;
const MAX_INDEXED_FILE_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Indexed live-payload limits. Atomic replacement can temporarily retain an
/// additional payload and temporary file beyond `max_payload_bytes`.
#[derive(Clone, Copy)]
pub struct StoreLimits {
    pub mode: DiskCacheMode,
    pub max_entries: usize,
    pub max_payload_bytes: u64,
    pub checkpoint: DiskLimits,
}

impl StoreLimits {
    pub(crate) fn validate(self) -> Result<()> {
        ensure!(
            (1..=MAX_ENTRIES).contains(&self.max_entries)
                && self.max_payload_bytes > 0
                && self.checkpoint.max_file_bytes > HEADER_BYTES as u64 + TRAILER_BYTES
                && self.checkpoint.max_file_bytes <= MAX_INDEXED_FILE_BYTES
                && self.checkpoint.max_io_buffer_bytes >= TILE_RECORD_BYTES as u64,
            "checkpoint store limits are outside supported bounds"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    slot: u8,
    scope: [u8; 32],
    prefix: [u8; 32],
    frontier: u32,
    file_bytes: u64,
    recency: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct IndexFile {
    version: u32,
    next_recency: u64,
    entries: Vec<Entry>,
}

impl IndexFile {
    fn empty() -> Self {
        Self {
            version: INDEX_VERSION,
            next_recency: 1,
            entries: Vec::new(),
        }
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.version == INDEX_VERSION,
            "checkpoint index version differs"
        );
        ensure!(
            self.next_recency > 0 && self.entries.len() <= MAX_ENTRIES,
            "checkpoint index geometry differs"
        );
        let mut slots = BTreeSet::new();
        let mut keys = BTreeSet::new();
        let mut times = BTreeSet::new();
        for entry in &self.entries {
            ensure!(
                usize::from(entry.slot) < PAYLOAD_SLOTS
                    && entry.frontier > 0
                    && entry.frontier <= 262_144
                    && (HEADER_BYTES as u64 + TRAILER_BYTES) <= entry.file_bytes
                    && entry.file_bytes <= MAX_INDEXED_FILE_BYTES
                    && entry.recency > 0
                    && entry.recency < self.next_recency,
                "checkpoint index entry is outside fixed geometry"
            );
            ensure!(
                slots.insert(entry.slot),
                "checkpoint index repeats a payload slot"
            );
            ensure!(
                keys.insert((entry.scope, entry.frontier, entry.prefix)),
                "checkpoint index repeats a prefix identity"
            );
            ensure!(
                times.insert(entry.recency),
                "checkpoint index repeats recency"
            );
        }
        Ok(())
    }

    fn live_bytes(&self) -> Result<u64> {
        self.entries.iter().try_fold(0u64, |total, entry| {
            total
                .checked_add(entry.file_bytes)
                .context("checkpoint index byte sum overflow")
        })
    }

    fn next_time(&mut self) -> Result<u64> {
        let value = self.next_recency;
        self.next_recency = value
            .checked_add(1)
            .context("checkpoint index recency overflow")?;
        Ok(value)
    }

    fn matching(&self, scope: [u8; 32], tokens: &[u32], dead: &BTreeSet<u8>) -> Vec<Entry> {
        let mut matches: Vec<_> = self
            .entries
            .iter()
            .filter(|entry| {
                entry.scope == scope
                    && !dead.contains(&entry.slot)
                    && usize::try_from(entry.frontier).ok().is_some_and(|n| {
                        n <= tokens.len() && entry.prefix == prefix_hash(&tokens[..n])
                    })
            })
            .cloned()
            .collect();
        matches.sort_by(|left, right| {
            right
                .frontier
                .cmp(&left.frontier)
                .then_with(|| right.recency.cmp(&left.recency))
        });
        matches
    }

    fn contains_key(&self, scope: [u8; 32], frontier: usize, prefix: [u8; 32]) -> bool {
        self.entries.iter().any(|entry| {
            entry.scope == scope && entry.frontier as usize == frontier && entry.prefix == prefix
        })
    }

    fn superseded(&self, scope: [u8; 32], tokens: &[u32]) -> Vec<u8> {
        // Keep the nearest two ancestors beside the incoming frontier. This
        // retains the current prompt and one recovery prefix while append-only
        // history stops competing with independent conversations.
        self.matching(scope, tokens, &BTreeSet::new())
            .into_iter()
            .skip(2)
            .map(|entry| entry.slot)
            .collect()
    }

    fn victims(&self, max_entries: usize, max_bytes: u64, incoming: u64) -> Result<Vec<u8>> {
        let mut retained = self.entries.clone();
        retained.sort_by_key(|entry| (entry.recency, entry.slot));
        let mut bytes = self.live_bytes()?;
        let mut victims = Vec::new();
        while retained.len() >= max_entries
            || bytes
                .checked_add(incoming)
                .is_none_or(|sum| sum > max_bytes)
        {
            let Some(oldest) = retained.first() else {
                anyhow::bail!("checkpoint cannot fit within store byte budget");
            };
            bytes -= oldest.file_bytes;
            victims.push(oldest.slot);
            retained.remove(0);
        }
        Ok(victims)
    }
}

fn not_found(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<std::io::Error>()
        .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound)
}

fn payload_leaf(slot: u8) -> String {
    format!("payload-{slot:02}.bin")
}

fn existing_leaf(root: &File, name: &str) -> Result<Option<File>> {
    match open_leaf(root, name, libc::O_RDONLY) {
        Ok(file) => Ok(Some(file)),
        Err(error) if not_found(&error) => Ok(None),
        Err(error) => Err(error).with_context(|| format!("invalid checkpoint leaf {name}")),
    }
}

fn unlink_leaf(root: &File, name: &str) -> Result<bool> {
    let Some(file) = existing_leaf(root, name)? else {
        return Ok(false);
    };
    drop(file);
    let name = CString::new(name)?;
    let status = unsafe { libc::unlinkat(root.as_raw_fd(), name.as_ptr(), 0) };
    ensure!(
        status == 0,
        "checkpoint leaf unlink failed: {}",
        std::io::Error::last_os_error()
    );
    Ok(true)
}

fn rename_leaf(root: &File, from: &str, to: &str) -> Result<()> {
    let from = CString::new(from)?;
    let to = CString::new(to)?;
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
        "checkpoint leaf rename failed: {}",
        std::io::Error::last_os_error()
    );
    Ok(())
}

fn read_index(root: &File) -> Result<Option<IndexFile>> {
    let Some(mut file) = existing_leaf(root, INDEX)? else {
        return Ok(None);
    };
    let length = file.metadata()?.len();
    ensure!(
        length > 0 && length <= MAX_INDEX_BYTES,
        "checkpoint index size exceeds admission"
    );
    let length = usize::try_from(length)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length)?;
    bytes.resize(length, 0);
    file.read_exact(&mut bytes)?;
    let index: IndexFile =
        serde_json::from_slice(&bytes).context("checkpoint index JSON is malformed")?;
    index.validate()?;
    Ok(Some(index))
}

#[derive(Debug)]
enum PublishError {
    Before(anyhow::Error),
    After(anyhow::Error),
}

fn publish_index_with_sync(
    root: &File,
    candidate: &IndexFile,
    sync_directory: fn(&File) -> std::io::Result<()>,
) -> std::result::Result<(), PublishError> {
    let before = || -> Result<()> {
        candidate.validate()?;
        let bytes = serde_json::to_vec(candidate)?;
        ensure!(
            bytes.len() as u64 <= MAX_INDEX_BYTES,
            "checkpoint index serialization exceeds admission"
        );
        unlink_leaf(root, INDEX_TEMP)?;
        let mut temp = open_leaf(
            root,
            INDEX_TEMP,
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
        )?;
        temp.write_all(&bytes)?;
        temp.sync_all()?;
        drop(temp);
        existing_leaf(root, INDEX)?;
        rename_leaf(root, INDEX_TEMP, INDEX)?;
        Ok(())
    };
    before().map_err(PublishError::Before)?;
    sync_directory(root)
        .context(
            "checkpoint index rename completed but directory sync failed; durability is uncertain",
        )
        .map_err(PublishError::After)
}

fn publish_index(root: &File, candidate: &IndexFile) -> std::result::Result<(), PublishError> {
    publish_index_with_sync(root, candidate, File::sync_all)
}

fn publish_payload_with_sync(
    root: &File,
    slot: u8,
    expected: u64,
    sync_directory: fn(&File) -> std::io::Result<()>,
) -> std::result::Result<(), PublishError> {
    let before = || -> Result<()> {
        let temporary = existing_leaf(root, PAYLOAD_TEMP)?
            .context("checkpoint payload temporary is missing")?;
        ensure!(
            temporary.metadata()?.len() == expected,
            "checkpoint payload temporary length differs"
        );
        ensure!(
            existing_leaf(root, &payload_leaf(slot))?.is_none(),
            "checkpoint spare payload leaf is occupied"
        );
        drop(temporary);
        rename_leaf(root, PAYLOAD_TEMP, &payload_leaf(slot))?;
        Ok(())
    };
    before().map_err(PublishError::Before)?;
    sync_directory(root)
        .context("checkpoint payload rename completed but directory sync failed; durability is uncertain")
        .map_err(PublishError::After)
}

fn publish_payload(root: &File, slot: u8, expected: u64) -> std::result::Result<(), PublishError> {
    publish_payload_with_sync(root, slot, expected, File::sync_all)
}

/// One exclusive owner of a small set of native prompt payloads.
pub struct DiskCheckpointStore {
    root: File,
    _lock: File,
    owner: Rc<()>,
    scope: [u8; 32],
    limits: StoreLimits,
    index: IndexFile,
    dead_slots: BTreeSet<u8>,
    writes_disabled: bool,
    restores_disabled: bool,
    managed: Option<ManagedCache>,
}

fn cleanup_unindexed(root: &File, index: &IndexFile) -> Result<()> {
    let indexed: BTreeSet<_> = index.entries.iter().map(|entry| entry.slot).collect();
    let mut removed = unlink_leaf(root, PAYLOAD_TEMP)? | unlink_leaf(root, INDEX_TEMP)?;
    for slot in 0..PAYLOAD_SLOTS {
        let slot = u8::try_from(slot)?;
        if !indexed.contains(&slot) {
            removed |= unlink_leaf(root, &payload_leaf(slot))?;
        }
    }
    if removed {
        root.sync_all()?;
    }
    Ok(())
}

fn initialize_or_read_index(root: &File) -> Result<IndexFile> {
    if let Some(index) = read_index(root)? {
        return Ok(index);
    }
    // A missing index is new only when no generated payload leaf exists. A
    // missing or malformed index never authorizes deletion of old payloads.
    for slot in 0..PAYLOAD_SLOTS {
        ensure!(
            existing_leaf(root, &payload_leaf(u8::try_from(slot)?))?.is_none(),
            "checkpoint index is missing while payloads exist"
        );
    }
    ensure!(
        existing_leaf(root, PAYLOAD_TEMP)?.is_none(),
        "checkpoint index is missing while a payload temporary exists"
    );
    unlink_leaf(root, INDEX_TEMP)?;
    let index = IndexFile::empty();
    match publish_index(root, &index) {
        Ok(()) => Ok(index),
        Err(PublishError::Before(error) | PublishError::After(error)) => Err(error),
    }
}

fn validate_selected_header(file: &mut File, entry: &Entry, engine: &Engine) -> Result<()> {
    if file.metadata()?.len() != entry.file_bytes {
        return Err(InvalidCheckpoint("indexed checkpoint file length differs").into());
    }
    let mut head = [0u8; HEADER_BYTES];
    file.read_exact(&mut head)?;
    let expected = expected_layout(&engine.config, engine.max_context, entry.frontier as usize)
        .map_err(|_| InvalidCheckpoint("indexed checkpoint frontier differs from engine"))?;
    if &head[..16] != MAGIC
        || field_u32(&head, 16) != WIRE_VERSION
        || head[20..52] != entry.scope
        || head[52..84] != entry.prefix
        || field_u64(&head, 84) != u64::from(entry.frontier)
        || field_u64(&head, 92) != entry.file_bytes
        || entry.file_bytes != expected.0
        || field_u32(&head, 100) != expected.1
        || field_u32(&head, 104) != 0
        || u64::from(field_u32(&head, 108)) < u64::from(entry.frontier)
        || field_u32(&head, 112) != PRODUCER_VERSION
        || field_u32(&head, 116) != engine.config.eos_token_id
    {
        return Err(InvalidCheckpoint("indexed checkpoint metadata differs from payload").into());
    }
    file.seek(SeekFrom::Start(0))?;
    Ok(())
}

fn reconcile_and_prune(
    root: &File,
    mut index: IndexFile,
    limits: StoreLimits,
) -> Result<IndexFile> {
    // Reconcile referenced regular files before trusting their byte claims
    // or sweeping generated orphans. Suspicious leaf types fail closed.
    let mut invalid = Vec::new();
    for entry in &index.entries {
        match existing_leaf(root, &payload_leaf(entry.slot))? {
            Some(file) if file.metadata()?.len() == entry.file_bytes => {}
            _ => invalid.push(entry.slot),
        }
    }
    if !invalid.is_empty() {
        let mut candidate = index.clone();
        candidate
            .entries
            .retain(|entry| !invalid.contains(&entry.slot));
        match publish_index(root, &candidate) {
            Ok(()) => {
                index = candidate;
                for slot in invalid {
                    unlink_leaf(root, &payload_leaf(slot))?;
                }
                root.sync_all()?;
            }
            Err(PublishError::Before(error) | PublishError::After(error)) => return Err(error),
        }
    }
    let mut indexed_bytes = index.live_bytes()?;
    if index.entries.len() > limits.max_entries
        || indexed_bytes > limits.max_payload_bytes
        || index
            .entries
            .iter()
            .any(|entry| entry.file_bytes > limits.checkpoint.max_file_bytes)
    {
        let mut candidate = index.clone();
        let mut victims = Vec::new();
        while candidate.entries.len() > limits.max_entries
            || indexed_bytes > limits.max_payload_bytes
            || candidate
                .entries
                .iter()
                .any(|entry| entry.file_bytes > limits.checkpoint.max_file_bytes)
        {
            let victim = candidate
                .entries
                .iter()
                .filter(|entry| entry.file_bytes > limits.checkpoint.max_file_bytes)
                .min_by_key(|entry| (entry.recency, entry.slot))
                .or_else(|| {
                    candidate
                        .entries
                        .iter()
                        .min_by_key(|entry| (entry.recency, entry.slot))
                })
                .context("checkpoint index cannot be pruned")?
                .clone();
            candidate.entries.retain(|entry| entry.slot != victim.slot);
            indexed_bytes -= victim.file_bytes;
            victims.push(victim.slot);
        }
        match publish_index(root, &candidate) {
            Ok(()) => {
                index = candidate;
                for slot in victims {
                    unlink_leaf(root, &payload_leaf(slot))?;
                }
                root.sync_all()?;
            }
            Err(PublishError::Before(error) | PublishError::After(error)) => return Err(error),
        }
    }
    cleanup_unindexed(root, &index)?;
    Ok(index)
}

impl DiskCheckpointStore {
    /// Open one explicit private root. The lock is held for this store's life.
    pub fn open(
        root: &Path,
        engine: &Engine,
        tokenizer: &TextTokenizer,
        limits: StoreLimits,
    ) -> Result<Self> {
        limits.validate()?;
        let scope = scope(engine, tokenizer)?;
        let root = open_root(root)?;
        let lock = open_leaf(&root, LOCK, libc::O_RDWR | libc::O_CREAT)?;
        lock.try_lock()
            .context("checkpoint root is already locked")?;
        let index = reconcile_and_prune(&root, initialize_or_read_index(&root)?, limits)?;
        Ok(Self {
            root,
            _lock: lock,
            owner: engine.cache_identity.clone(),
            scope,
            limits,
            index,
            dead_slots: BTreeSet::new(),
            writes_disabled: false,
            restores_disabled: false,
            managed: None,
        })
    }

    /// Share retention with recognized native and legacy roots below the
    /// managed user-cache base. Explicit roots use `open`.
    pub fn open_managed(
        root: &Path,
        engine: &Engine,
        tokenizer: &TextTokenizer,
        limits: StoreLimits,
        base: &Path,
    ) -> Result<Self> {
        ensure!(
            root.parent() == Some(base.join("v4/disk_kv").as_path())
                && root
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.len() == 64
                        && name
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))),
            "managed root must be a native package directory below its cache base"
        );
        let mut store = Self::open(root, engine, tokenizer, limits)?;
        let managed = ManagedCache::open(base)?;
        let reservation = managed.reserve(
            &store.root,
            store.index.live_bytes()?,
            0,
            limits.max_payload_bytes,
        )?;
        store.index = reconcile_and_prune(
            &store.root,
            store.index,
            StoreLimits {
                max_payload_bytes: reservation.allowance,
                ..limits
            },
        )?;
        store.managed = Some(managed);
        Ok(store)
    }

    pub fn mode(&self) -> DiskCacheMode {
        self.limits.mode
    }

    fn check_owner(&self, engine: &Engine) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.owner, &engine.cache_identity),
            "checkpoint store belongs to another engine"
        );
        Ok(())
    }

    fn install_index_with_sync(
        &mut self,
        candidate: IndexFile,
        sync_directory: fn(&File) -> std::io::Result<()>,
    ) -> Result<()> {
        match publish_index_with_sync(&self.root, &candidate, sync_directory) {
            Ok(()) => {
                self.index = candidate;
                Ok(())
            }
            Err(PublishError::Before(error)) => Err(error),
            Err(PublishError::After(error)) => {
                // The rename is visible. Old payloads stay until a later open
                // establishes which index survived the uncertain directory sync.
                self.index = candidate;
                self.writes_disabled = true;
                Err(error)
            }
        }
    }

    fn install_index(&mut self, candidate: IndexFile) -> Result<()> {
        self.install_index_with_sync(candidate, File::sync_all)
    }

    fn remove_invalid(&mut self, entry: &Entry) -> Result<()> {
        self.dead_slots.insert(entry.slot);
        if self.writes_disabled {
            return Ok(());
        }
        let mut candidate = self.index.clone();
        candidate.entries.retain(|row| row.slot != entry.slot);
        if let Err(error) = self.install_index(candidate) {
            self.writes_disabled = true;
            return Err(error).context("checkpoint invalid-entry removal failed");
        }
        if let Err(error) = unlink_leaf(&self.root, &payload_leaf(entry.slot))
            .and_then(|_| self.root.sync_all().map_err(Into::into))
        {
            self.writes_disabled = true;
            return Err(error).context("checkpoint invalid payload cleanup failed");
        }
        self.dead_slots.remove(&entry.slot);
        Ok(())
    }

    fn touch_after_restore(&mut self, entry: &Entry) {
        if self.writes_disabled {
            return;
        }
        let mut candidate = self.index.clone();
        let time = match candidate.next_time() {
            Ok(time) => time,
            Err(error) => {
                self.writes_disabled = true;
                eprintln!("checkpoint LRU update unavailable: {error:#}");
                return;
            }
        };
        let Some(row) = candidate
            .entries
            .iter_mut()
            .find(|row| row.slot == entry.slot)
        else {
            return;
        };
        row.recency = time;
        if let Err(error) = self.install_index(candidate) {
            self.writes_disabled = true;
            eprintln!("checkpoint LRU update unavailable: {error:#}");
        }
    }

    /// Inspect indexed prefixes without allocating a restored session.
    pub(crate) fn has_match_beyond(
        &self,
        engine: &Engine,
        requested: &[u32],
        frontier: usize,
    ) -> Result<bool> {
        self.check_owner(engine)?;
        Ok(!self.restores_disabled
            && self.index.entries.iter().any(|entry| {
                let count = entry.frontier as usize;
                entry.scope == self.scope
                    && !self.dead_slots.contains(&entry.slot)
                    && count > frontier
                    && count <= requested.len()
                    && entry.prefix == prefix_hash(&requested[..count])
            }))
    }

    /// Return the longest valid exact-prefix state, then persist its LRU touch.
    /// A live restoration failure retains its healthy payload and disables
    /// further restores for this store instance.
    pub fn load_match(&mut self, engine: &Engine, requested: &[u32]) -> Result<Option<Session>> {
        self.check_owner(engine)?;
        if self.restores_disabled {
            return Ok(None);
        }
        for entry in self.index.matching(self.scope, requested, &self.dead_slots) {
            let opened = match existing_leaf(&self.root, &payload_leaf(entry.slot)) {
                Ok(opened) => opened,
                Err(error) => {
                    self.restores_disabled = true;
                    return Err(error).context("checkpoint payload open unavailable");
                }
            };
            let Some(mut file) = opened else {
                self.remove_invalid(&entry)?;
                continue;
            };
            let session = match validate_selected_header(&mut file, &entry, engine).and_then(|_| {
                read_session(
                    &mut file,
                    engine,
                    &self.scope,
                    requested,
                    self.limits.checkpoint,
                )
            }) {
                Ok(Some(session)) => session,
                Ok(None) => {
                    drop(file);
                    self.remove_invalid(&entry)?;
                    continue;
                }
                Err(error) if error.downcast_ref::<InvalidCheckpoint>().is_some() => {
                    drop(file);
                    self.remove_invalid(&entry)?;
                    continue;
                }
                Err(error) => {
                    self.restores_disabled = true;
                    return Err(error).context("checkpoint read unavailable");
                }
            };
            drop(file);
            self.touch_after_restore(&entry);
            return Ok(Some(session));
        }
        Ok(None)
    }

    /// Write one committed conversation frontier, deduplicating its scoped prefix.
    /// A false result means size admission declined or the key already exists.
    pub fn save(&mut self, engine: &Engine, session: &Session) -> Result<bool> {
        self.check_owner(engine)?;
        if self.writes_disabled {
            return Ok(false);
        }
        engine.validate_snapshot_source(session)?;
        ensure!(
            Rc::ptr_eq(&session.cache_origin, &engine.cache_identity),
            "session cache producer differs from engine"
        );
        let frontier = session.frontier;
        let (expected, sections, _) =
            expected_layout(&engine.config, engine.max_context, frontier)?;
        let key = prefix_hash(&session.consumed_tokens);
        if self.index.contains_key(self.scope, frontier, key) {
            return Ok(false);
        }
        if expected > self.limits.checkpoint.max_file_bytes
            || expected > self.limits.max_payload_bytes
        {
            return Ok(false);
        }
        if let Err(error) = cleanup_unindexed(&self.root, &self.index) {
            self.writes_disabled = true;
            return Err(error).context("checkpoint generated-leaf cleanup unavailable");
        }
        let mut candidate = self.index.clone();
        // Prefix mode retains earlier shared anchors until ordinary budget/LRU
        // eviction. Deeper history does not make an earlier branch obsolete.
        let mut victims = match self.limits.mode {
            DiskCacheMode::Prefix => Vec::new(),
            DiskCacheMode::Full => candidate.superseded(self.scope, &session.consumed_tokens),
        };
        candidate
            .entries
            .retain(|entry| !victims.contains(&entry.slot));
        // Hold aggregate admission through publication and victim deletion.
        let reservation = self
            .managed
            .as_ref()
            .map(|managed| {
                managed.reserve(
                    &self.root,
                    candidate.live_bytes()?,
                    expected,
                    self.limits.max_payload_bytes,
                )
            })
            .transpose()?;
        let allowance = reservation
            .as_ref()
            .map_or(self.limits.max_payload_bytes, |reservation| {
                reservation.allowance
            });
        if expected > allowance {
            self.index = reconcile_and_prune(
                &self.root,
                self.index.clone(),
                StoreLimits {
                    max_payload_bytes: allowance,
                    ..self.limits
                },
            )?;
            return Ok(false);
        }
        victims.extend(candidate.victims(self.limits.max_entries, allowance, expected)?);
        let slot = (0..PAYLOAD_SLOTS)
            .map(u8::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?
            .into_iter()
            .find(|slot| !self.index.entries.iter().any(|entry| entry.slot == *slot))
            .context("checkpoint spare payload slot is unavailable")?;
        ensure!(
            existing_leaf(&self.root, &payload_leaf(slot))?.is_none(),
            "checkpoint spare payload leaf is occupied"
        );
        candidate
            .entries
            .retain(|entry| !victims.contains(&entry.slot));
        let recency = candidate.next_time()?;
        candidate.entries.push(Entry {
            slot,
            scope: self.scope,
            prefix: key,
            frontier: u32::try_from(frontier)?,
            file_bytes: expected,
            recency,
        });
        candidate.validate()?;
        ensure!(
            serde_json::to_vec(&candidate)?.len() as u64 <= MAX_INDEX_BYTES,
            "checkpoint index candidate exceeds admission"
        );
        unlink_leaf(&self.root, PAYLOAD_TEMP)?;
        let mut temp = open_leaf(
            &self.root,
            PAYLOAD_TEMP,
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
        )?;
        write_session(
            &mut temp,
            engine,
            session,
            &self.scope,
            expected,
            sections,
            self.limits.checkpoint.max_io_buffer_bytes,
        )?;
        temp.sync_all()?;
        drop(temp);
        match publish_payload(&self.root, slot, expected) {
            Ok(()) => {}
            Err(PublishError::Before(error)) => return Err(error),
            Err(PublishError::After(error)) => {
                self.writes_disabled = true;
                return Err(error);
            }
        }
        if let Err(error) = self.install_index(candidate) {
            if !self.writes_disabled {
                // No index rename occurred. The old index still owns every old
                // payload; the new generated leaf is safe to remove.
                let _ = unlink_leaf(&self.root, &payload_leaf(slot));
                let _ = self.root.sync_all();
            }
            return Err(error);
        }
        for victim in victims {
            if let Err(error) = unlink_leaf(&self.root, &payload_leaf(victim)) {
                self.writes_disabled = true;
                return Err(error).context("checkpoint victim cleanup failed after index commit");
            }
        }
        if let Err(error) = self.root.sync_all() {
            self.writes_disabled = true;
            return Err(error).context("checkpoint victim cleanup directory sync failed");
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_ROOT: AtomicU64 = AtomicU64::new(0);

    struct PrivateRoot(PathBuf);

    impl PrivateRoot {
        fn new() -> Result<Self> {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos();
            let sequence = NEXT_ROOT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "moespresso-index-{}-{nonce}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&path)?;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o700))?;
            Ok(Self(path))
        }
        fn open(&self) -> Result<File> {
            open_root(&self.0)
        }
    }

    impl Drop for PrivateRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn row(slot: u8, frontier: u32, recency: u64, tokens: &[u32]) -> Entry {
        Entry {
            slot,
            scope: [7; 32],
            prefix: prefix_hash(&tokens[..frontier as usize]),
            frontier,
            file_bytes: 200,
            recency,
        }
    }

    fn limits() -> StoreLimits {
        StoreLimits {
            mode: DiskCacheMode::Full,
            max_entries: 2,
            max_payload_bytes: 500,
            checkpoint: DiskLimits {
                max_file_bytes: 400 * 1024 * 1024,
                max_io_buffer_bytes: 400 * 1024 * 1024,
            },
        }
    }

    #[test]
    fn index_rejects_duplicate_identity_slots_and_unbounded_metadata() -> Result<()> {
        limits().validate()?;
        assert!(
            StoreLimits {
                max_entries: 0,
                ..limits()
            }
            .validate()
            .is_err()
        );
        assert!(
            StoreLimits {
                max_entries: 65,
                ..limits()
            }
            .validate()
            .is_err()
        );
        let tokens = vec![7; 1024];
        let mut index = IndexFile {
            version: INDEX_VERSION,
            next_recency: 3,
            entries: vec![row(0, 512, 1, &tokens), row(1, 1024, 2, &tokens)],
        };
        index.validate()?;
        index.entries[1].slot = 0;
        assert!(index.validate().is_err());
        index.entries[1].slot = 1;
        index.entries[1].frontier = 512;
        index.entries[1].prefix = index.entries[0].prefix;
        assert!(index.validate().is_err());
        index.entries[1].frontier = 1024;
        index.entries[1].prefix = prefix_hash(&tokens);
        index.entries[1].recency = 1;
        assert!(index.validate().is_err());
        index.entries[1].recency = 2;
        index.entries[1].file_bytes = MAX_INDEXED_FILE_BYTES + 1;
        assert!(index.validate().is_err());
        index.entries[1].file_bytes = 200;
        index.entries[1].slot = 65;
        assert!(index.validate().is_err());
        index.next_recency = u64::MAX;
        assert!(index.next_time().is_err());
        Ok(())
    }

    #[test]
    fn longest_prefix_and_persisted_lru_choose_deterministic_victim() -> Result<()> {
        let tokens = vec![7; 1536];
        let mut index = IndexFile {
            version: INDEX_VERSION,
            next_recency: 4,
            entries: vec![row(3, 512, 1, &tokens), row(0, 1024, 2, &tokens)],
        };
        index.validate()?;
        let matches = index.matching([7; 32], &tokens, &BTreeSet::new());
        assert_eq!(
            matches.iter().map(|v| v.frontier).collect::<Vec<_>>(),
            [1024, 512]
        );
        assert!(index.contains_key([7; 32], 512, prefix_hash(&tokens[..512])));
        assert!(!index.contains_key([8; 32], 512, prefix_hash(&tokens[..512])));
        assert_eq!(index.victims(2, 500, 200)?, [3]);
        assert_eq!(index.next_time()?, 4);
        assert_eq!(index.next_recency, 5);
        let mut changed = tokens.clone();
        changed[0] = 8;
        assert!(
            index
                .matching([7; 32], &changed, &BTreeSet::new())
                .is_empty()
        );
        Ok(())
    }

    #[test]
    fn durable_empty_index_precedes_payload_and_restart_cleans_only_owned_orphans() -> Result<()> {
        let root = PrivateRoot::new()?;
        let directory = root.open()?;
        let index = initialize_or_read_index(&directory)?;
        assert!(index.entries.is_empty());
        assert!(existing_leaf(&directory, INDEX)?.is_some());
        let tokens = vec![7; 512];
        let mut candidate = index.clone();
        let time = candidate.next_time()?;
        candidate.entries.push(row(0, 512, time, &tokens));
        let payload = open_leaf(
            &directory,
            &payload_leaf(0),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
        )?;
        payload.set_len(200)?;
        payload.sync_all()?;
        publish_index(&directory, &candidate).map_err(|e| anyhow!("{e:?}"))?;
        let orphan = open_leaf(
            &directory,
            &payload_leaf(1),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
        )?;
        orphan.set_len(200)?;
        orphan.sync_all()?;
        let unrelated = root.0.join("unrelated.txt");
        fs::write(&unrelated, b"keep")?;
        drop(orphan);
        drop(payload);
        drop(directory);
        let reopened = root.open()?;
        let loaded = initialize_or_read_index(&reopened)?;
        assert_eq!(loaded.entries, candidate.entries);
        cleanup_unindexed(&reopened, &loaded)?;
        assert!(existing_leaf(&reopened, &payload_leaf(0))?.is_some());
        assert!(existing_leaf(&reopened, &payload_leaf(1))?.is_none());
        assert_eq!(fs::read(unrelated)?, b"keep");
        Ok(())
    }

    #[test]
    fn missing_or_malformed_index_never_authorizes_payload_cleanup() -> Result<()> {
        let root = PrivateRoot::new()?;
        let directory = root.open()?;
        let payload = open_leaf(
            &directory,
            &payload_leaf(0),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
        )?;
        payload.set_len(200)?;
        drop(payload);
        assert!(initialize_or_read_index(&directory).is_err());
        assert!(existing_leaf(&directory, &payload_leaf(0))?.is_some());
        let mut index = open_leaf(
            &directory,
            INDEX,
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
        )?;
        index.write_all(b"{")?;
        index.sync_all()?;
        drop(index);
        assert!(initialize_or_read_index(&directory).is_err());
        assert!(existing_leaf(&directory, &payload_leaf(0))?.is_some());
        let index = open_leaf(&directory, INDEX, libc::O_WRONLY | libc::O_TRUNC)?;
        index.set_len(MAX_INDEX_BYTES + 1)?;
        drop(index);
        assert!(read_index(&directory).is_err());
        assert!(existing_leaf(&directory, &payload_leaf(0))?.is_some());
        Ok(())
    }

    #[test]
    fn restart_reconciles_bad_referenced_lengths_before_orphan_cleanup() -> Result<()> {
        let root = PrivateRoot::new()?;
        let directory = root.open()?;
        let mut index = initialize_or_read_index(&directory)?;
        let tokens = vec![7; 1536];
        for (slot, frontier, bytes) in [(0, 512, None), (1, 1024, Some(199)), (2, 1536, Some(200))]
        {
            let time = index.next_time()?;
            index.entries.push(row(slot, frontier, time, &tokens));
            if let Some(bytes) = bytes {
                let file = open_leaf(
                    &directory,
                    &payload_leaf(slot),
                    libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
                )?;
                file.set_len(bytes)?;
            }
        }
        publish_index(&directory, &index).map_err(|e| anyhow!("{e:?}"))?;
        let orphan = open_leaf(
            &directory,
            &payload_leaf(3),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
        )?;
        orphan.set_len(200)?;
        let kept = reconcile_and_prune(
            &directory,
            initialize_or_read_index(&directory)?,
            StoreLimits {
                max_entries: 3,
                max_payload_bytes: 600,
                ..limits()
            },
        )?;
        assert_eq!(kept.entries.len(), 1);
        assert_eq!(kept.entries[0].slot, 2);
        assert_eq!(read_index(&directory)?.unwrap().entries, kept.entries);
        for slot in [0, 1, 3] {
            assert!(existing_leaf(&directory, &payload_leaf(slot))?.is_none());
        }
        assert_eq!(
            existing_leaf(&directory, &payload_leaf(2))?
                .unwrap()
                .metadata()?
                .len(),
            200
        );
        Ok(())
    }

    #[test]
    fn lower_limits_evict_persisted_lru_using_actual_file_bytes() -> Result<()> {
        let root = PrivateRoot::new()?;
        let directory = root.open()?;
        let mut index = initialize_or_read_index(&directory)?;
        let tokens = vec![7; 1536];
        for (slot, frontier) in [(0, 512), (1, 1024), (2, 1536)] {
            let time = index.next_time()?;
            index.entries.push(row(slot, frontier, time, &tokens));
            let file = open_leaf(
                &directory,
                &payload_leaf(slot),
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
            )?;
            file.set_len(200)?;
        }
        publish_index(&directory, &index).map_err(|e| anyhow!("{e:?}"))?;
        let two = reconcile_and_prune(
            &directory,
            initialize_or_read_index(&directory)?,
            StoreLimits {
                max_entries: 2,
                max_payload_bytes: 400,
                ..limits()
            },
        )?;
        assert_eq!(
            two.entries.iter().map(|e| e.slot).collect::<Vec<_>>(),
            [1, 2]
        );
        assert!(existing_leaf(&directory, &payload_leaf(0))?.is_none());
        let one = reconcile_and_prune(
            &directory,
            initialize_or_read_index(&directory)?,
            StoreLimits {
                max_entries: 1,
                max_payload_bytes: 200,
                ..limits()
            },
        )?;
        assert_eq!(one.entries.iter().map(|e| e.slot).collect::<Vec<_>>(), [2]);
        assert!(existing_leaf(&directory, &payload_leaf(1))?.is_none());
        assert!(existing_leaf(&directory, &payload_leaf(2))?.is_some());
        Ok(())
    }

    #[test]
    fn post_rename_index_sync_fault_keeps_visible_candidate_and_payloads() -> Result<()> {
        fn fail_sync(_: &File) -> std::io::Result<()> {
            Err(std::io::Error::other("injected directory sync failure"))
        }
        let root = PrivateRoot::new()?;
        let directory = root.open()?;
        let old = initialize_or_read_index(&directory)?;
        let payload = open_leaf(
            &directory,
            &payload_leaf(0),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
        )?;
        payload.set_len(200)?;
        let mut candidate = old.clone();
        let time = candidate.next_time()?;
        candidate.entries.push(row(0, 512, time, &vec![7; 512]));
        assert!(matches!(
            publish_index_with_sync(&directory, &candidate, fail_sync),
            Err(PublishError::After(_))
        ));
        assert_eq!(read_index(&directory)?.unwrap().entries, candidate.entries);
        assert!(existing_leaf(&directory, &payload_leaf(0))?.is_some());
        Ok(())
    }

    #[test]
    fn payload_publication_orders_before_index_and_preserves_old_on_faults() -> Result<()> {
        fn fail_sync(_: &File) -> std::io::Result<()> {
            Err(std::io::Error::other(
                "injected payload directory sync failure",
            ))
        }
        let root = PrivateRoot::new()?;
        let directory = root.open()?;
        let index = initialize_or_read_index(&directory)?;
        let temp = open_leaf(
            &directory,
            PAYLOAD_TEMP,
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
        )?;
        temp.set_len(200)?;
        temp.sync_all()?;
        assert!(matches!(
            publish_payload_with_sync(&directory, 0, 201, File::sync_all),
            Err(PublishError::Before(_))
        ));
        assert!(existing_leaf(&directory, PAYLOAD_TEMP)?.is_some());
        assert!(existing_leaf(&directory, &payload_leaf(0))?.is_none());
        assert!(matches!(
            publish_payload_with_sync(&directory, 0, 200, fail_sync),
            Err(PublishError::After(_))
        ));
        assert!(existing_leaf(&directory, &payload_leaf(0))?.is_some());
        assert!(read_index(&directory)?.unwrap().entries == index.entries);
        Ok(())
    }

    #[test]
    fn uncertain_index_commit_freezes_touch_and_invalid_removal() -> Result<()> {
        fn fail_sync(_: &File) -> std::io::Result<()> {
            Err(std::io::Error::other("injected directory sync failure"))
        }
        let root = PrivateRoot::new()?;
        let directory = root.open()?;
        let lock = open_leaf(&directory, LOCK, libc::O_RDWR | libc::O_CREAT)?;
        lock.try_lock()?;
        let index = initialize_or_read_index(&directory)?;
        let mut store = DiskCheckpointStore {
            root: directory,
            _lock: lock,
            owner: Rc::new(()),
            scope: [7; 32],
            limits: limits(),
            index,
            dead_slots: BTreeSet::new(),
            writes_disabled: false,
            restores_disabled: false,
            managed: None,
        };
        let payload = open_leaf(
            &store.root,
            &payload_leaf(0),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
        )?;
        payload.set_len(200)?;
        let mut candidate = store.index.clone();
        let time = candidate.next_time()?;
        let entry = row(0, 512, time, &vec![7; 512]);
        candidate.entries.push(entry.clone());
        assert!(store.install_index_with_sync(candidate, fail_sync).is_err());
        assert!(store.writes_disabled);
        let visible = fs::read(root.0.join(INDEX))?;
        store.touch_after_restore(&entry);
        store.remove_invalid(&entry)?;
        assert_eq!(fs::read(root.0.join(INDEX))?, visible);
        assert!(store.dead_slots.contains(&entry.slot));
        assert!(existing_leaf(&store.root, &payload_leaf(entry.slot))?.is_some());
        Ok(())
    }

    #[test]
    fn generated_leaf_symlinks_and_root_lock_fail_closed() -> Result<()> {
        let root = PrivateRoot::new()?;
        let directory = root.open()?;
        let lock = open_leaf(&directory, LOCK, libc::O_RDWR | libc::O_CREAT)?;
        lock.try_lock()?;
        let second = open_leaf(&directory, LOCK, libc::O_RDWR)?;
        assert!(second.try_lock().is_err());
        let index = initialize_or_read_index(&directory)?;
        symlink(root.0.join("missing"), root.0.join(payload_leaf(0)))?;
        assert!(cleanup_unindexed(&directory, &index).is_err());
        assert!(
            root.0
                .join(payload_leaf(0))
                .symlink_metadata()?
                .file_type()
                .is_symlink()
        );
        Ok(())
    }
}
