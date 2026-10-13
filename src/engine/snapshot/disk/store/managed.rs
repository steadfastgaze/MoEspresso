//! Shared retention for recognized caches below the managed user-cache base.

use super::*;
use serde_json::Value;
use std::ffi::CStr;
use std::os::fd::FromRawFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;

pub(super) struct ManagedCache {
    base: File,
}

pub(super) struct Reservation {
    _lock: File,
    pub(super) allowance: u64,
}

impl ManagedCache {
    pub(super) fn open(base: &Path) -> Result<Self> {
        ensure!(base.is_absolute(), "managed cache base must be absolute");
        let name = CString::new(base.as_os_str().as_bytes())?;
        let fd = unsafe {
            libc::open(
                name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        ensure!(
            fd >= 0,
            "managed cache base unavailable: {}",
            std::io::Error::last_os_error()
        );
        let base = unsafe { File::from_raw_fd(fd) };
        directory_owned(&base)?;
        Ok(Self { base })
    }

    pub(super) fn reserve(
        &self,
        current: &File,
        retained: u64,
        incoming: u64,
        limit: u64,
    ) -> Result<Reservation> {
        let lock = open_leaf(&self.base, "retention.lock", libc::O_RDWR | libc::O_CREAT)?;
        lock.try_lock()?;
        let current = current.metadata()?;
        let mut roots = Vec::new();
        if let Some(v4) = child_directory(&self.base, "v4")?
            && let Some(native) = child_directory(&v4, "disk_kv")?
        {
            for name in names(&native)?.into_iter().filter(|name| hex(name, 64)) {
                let root = child_directory(&native, &name)?.context("managed root disappeared")?;
                let metadata = root.metadata()?;
                if metadata.dev() != current.dev() || metadata.ino() != current.ino() {
                    roots.push(native_root(root)?);
                }
            }
        }
        if let Some(legacy) = child_directory(&self.base, "disk_kv")? {
            for name in names(&legacy)?.into_iter().filter(|name| hex(name, 12)) {
                let root =
                    child_directory(&legacy, &name)?.context("managed legacy root disappeared")?;
                roots.push(legacy_root(root)?);
            }
        }
        // Complete discovery and ownership checks before any eviction. Active
        // root locks stay untouched; inactive locks remain held through cleanup.
        let mut other_bytes = roots
            .iter()
            .try_fold(0, |sum, root| checked_add(sum, root.bytes))?;
        roots.sort_by_key(|root| root.age);
        let desired = retained.saturating_add(incoming).min(limit);
        let mut reclaimed = 0;
        for root in roots.iter().filter(|root| root.reclaimable()) {
            if other_bytes <= limit.saturating_sub(desired) {
                break;
            }
            root.clear()?;
            other_bytes -= root.bytes;
            reclaimed += root.bytes;
        }
        if reclaimed > 0 {
            eprintln!("prompt cache event=disk_cleanup reclaimed_bytes={reclaimed}");
        }
        Ok(Reservation {
            _lock: lock,
            allowance: limit.saturating_sub(other_bytes),
        })
    }
}

fn hex(name: &str, length: usize) -> bool {
    name.len() == length
        && name
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn directory_owned(directory: &File) -> Result<()> {
    let metadata = directory.metadata()?;
    ensure!(
        metadata.is_dir()
            && metadata.uid() == unsafe { libc::geteuid() }
            && metadata.mode() & 0o022 == 0,
        "managed cache directory has unsafe ownership or permissions"
    );
    Ok(())
}

fn child_directory(parent: &File, name: &str) -> Result<Option<File>> {
    let name = CString::new(name)?;
    let fd = unsafe {
        libc::openat(
            parent.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        let error = std::io::Error::last_os_error();
        if error.kind() == std::io::ErrorKind::NotFound {
            return Ok(None);
        }
        return Err(error.into());
    }
    let file = unsafe { File::from_raw_fd(fd) };
    directory_owned(&file)?;
    Ok(Some(file))
}

fn owned_file(parent: &File, name: &str, writable: bool) -> Result<Option<File>> {
    let name = CString::new(name)?;
    let fd = unsafe {
        libc::openat(
            parent.as_raw_fd(),
            name.as_ptr(),
            (if writable {
                libc::O_RDWR
            } else {
                libc::O_RDONLY
            }) | libc::O_NOFOLLOW
                | libc::O_NONBLOCK
                | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        let error = std::io::Error::last_os_error();
        if error.kind() == std::io::ErrorKind::NotFound {
            return Ok(None);
        }
        return Err(error.into());
    }
    let file = unsafe { File::from_raw_fd(fd) };
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file()
            && metadata.uid() == unsafe { libc::geteuid() }
            && metadata.nlink() == 1
            && metadata.mode() & 0o022 == 0,
        "managed cache leaf has unsafe type, ownership or permissions"
    );
    Ok(Some(file))
}

fn names(directory: &File) -> Result<Vec<String>> {
    // Open a separate directory position; dup would share an enumeration
    // offset with the pinned descriptor and skip entries on later scans.
    let fd = unsafe {
        libc::openat(
            directory.as_raw_fd(),
            c".".as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC,
        )
    };
    ensure!(fd >= 0, "managed directory descriptor unavailable");
    let stream = unsafe { libc::fdopendir(fd) };
    if stream.is_null() {
        unsafe {
            libc::close(fd);
        }
        return Err(std::io::Error::last_os_error().into());
    }
    struct Close(*mut libc::DIR);
    impl Drop for Close {
        fn drop(&mut self) {
            unsafe {
                libc::closedir(self.0);
            }
        }
    }
    let _close = Close(stream);
    let mut names = Vec::new();
    let mut count = 0;
    loop {
        unsafe {
            *libc::__error() = 0;
        }
        let entry = unsafe { libc::readdir(stream) };
        if entry.is_null() {
            let error = std::io::Error::last_os_error();
            ensure!(
                error.raw_os_error() == Some(0),
                "managed directory enumeration failed: {error}"
            );
            break;
        }
        count += 1;
        ensure!(
            count <= 8192,
            "managed cache directory exceeds enumeration allowance"
        );
        let name = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) };
        if let Ok(name) = name.to_str()
            && !matches!(name, "." | "..")
        {
            names.push(name.to_owned());
        }
    }
    names.sort();
    Ok(names)
}

struct Payload {
    parent: Rc<File>,
    name: String,
    bytes: u64,
}

enum Contents {
    Native(Option<IndexFile>),
    Legacy {
        target: Option<Value>,
        companion: Option<(Rc<File>, Value)>,
        complete: bool,
    },
}

struct Root {
    directory: Rc<File>,
    lock: Option<File>,
    contents: Contents,
    payloads: Vec<Payload>,
    bytes: u64,
    age: (i64, i64),
}

impl Root {
    fn reclaimable(&self) -> bool {
        self.bytes > 0
            && self.lock.is_some()
            && match &self.contents {
                Contents::Native(index) => index.is_some(),
                Contents::Legacy {
                    target, complete, ..
                } => target.is_some() && *complete,
            }
    }

    fn clear(&self) -> Result<()> {
        match &self.contents {
            Contents::Native(Some(index)) => {
                let mut empty = index.clone();
                empty.entries.clear();
                publish_index(&self.directory, &empty).map_err(|error| match error {
                    PublishError::Before(error) | PublishError::After(error) => error,
                })?;
                cleanup_unindexed(&self.directory, &empty)?;
            }
            Contents::Legacy {
                target: Some(target),
                companion,
                complete: true,
            } => {
                empty_legacy_index(&self.directory, target)?;
                if let Some((directory, index)) = companion {
                    empty_legacy_index(directory, index)?;
                }
                for payload in &self.payloads {
                    // Recheck each leaf while the legacy root lock is held.
                    if owned_file(&payload.parent, &payload.name, false)?.is_some() {
                        let name = CString::new(payload.name.as_str())?;
                        ensure!(
                            unsafe { libc::unlinkat(payload.parent.as_raw_fd(), name.as_ptr(), 0) }
                                == 0,
                            "managed legacy payload removal failed: {}",
                            std::io::Error::last_os_error()
                        );
                        payload.parent.sync_all()?;
                    }
                }
            }
            _ => anyhow::bail!("managed root has no recognized index for cleanup"),
        }
        Ok(())
    }
}

fn lock_root(root: &File, name: &str) -> Result<Option<File>> {
    let Some(lock) = owned_file(root, name, true)? else {
        return Ok(None);
    };
    if lock.try_lock().is_ok() {
        Ok(Some(lock))
    } else {
        Ok(None)
    }
}

fn age(root: &File) -> Result<(i64, i64)> {
    let metadata = owned_file(root, INDEX, false)?
        .map(|file| file.metadata())
        .transpose()?;
    Ok(metadata.map_or((i64::MAX, 0), |metadata| {
        (metadata.mtime(), metadata.mtime_nsec())
    }))
}

fn native_root(directory: File) -> Result<Root> {
    let directory = Rc::new(directory);
    let lock = lock_root(&directory, LOCK)?;
    let index = read_index(&directory).ok().flatten();
    let age = age(&directory)?;
    let mut payloads = Vec::new();
    for name in (0..PAYLOAD_SLOTS)
        .map(|slot| payload_leaf(slot as u8))
        .chain([PAYLOAD_TEMP.to_owned()])
    {
        if let Some(file) = existing_leaf(&directory, &name)? {
            payloads.push(Payload {
                parent: directory.clone(),
                name,
                bytes: file.metadata()?.len(),
            });
        }
    }
    let bytes = payloads
        .iter()
        .try_fold(0, |sum, payload| checked_add(sum, payload.bytes))?;
    Ok(Root {
        directory,
        lock,
        contents: Contents::Native(index),
        payloads,
        bytes,
        age,
    })
}

fn legacy_index(root: &File, schema: &str) -> Result<Option<Value>> {
    let Some(file) = owned_file(root, INDEX, false)? else {
        return Ok(None);
    };
    if file.metadata()?.len() > 4 << 20 {
        return Ok(None);
    }
    let mut bytes = Vec::new();
    file.take((4 << 20) + 1).read_to_end(&mut bytes)?;
    if bytes.len() > 4 << 20 {
        return Ok(None);
    }
    let Ok(mut index) = serde_json::from_slice::<Value>(&bytes) else {
        return Ok(None);
    };
    if index["schema_version"] != schema
        || !index["entries"].is_array()
        || index.as_object().is_none_or(|object| object.len() != 2)
    {
        return Ok(None);
    }
    // Only a recognized empty-index replacement survives discovery. Metadata
    // from old conversations is neither imported nor retained across roots.
    index["entries"] = Value::Array(Vec::new());
    Ok(Some(index))
}

fn empty_legacy_index(root: &File, index: &Value) -> Result<()> {
    let mut empty = index.clone();
    empty["entries"] = Value::Array(Vec::new());
    let name = "index.json.tmp";
    if owned_file(root, name, false)?.is_some() {
        let raw = CString::new(name)?;
        ensure!(
            unsafe { libc::unlinkat(root.as_raw_fd(), raw.as_ptr(), 0) } == 0,
            "managed legacy index temporary removal failed"
        );
    }
    let mut temporary = open_leaf(root, name, libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL)?;
    temporary.write_all(&serde_json::to_vec(&empty)?)?;
    temporary.sync_all()?;
    drop(temporary);
    rename_leaf(root, name, INDEX)?;
    root.sync_all()?;
    Ok(())
}

fn payload_name(name: &str, quarantine: bool) -> bool {
    let Some(stem) = name.strip_suffix(".safetensors") else {
        return false;
    };
    match stem.split_once('.') {
        None => hex(stem, 64),
        Some((id, suffix)) => {
            hex(id, 64)
                && (suffix == "tmp"
                    || (quarantine
                        && !suffix.is_empty()
                        && suffix.bytes().all(|byte| byte.is_ascii_digit())))
        }
    }
}

fn collect_payloads(
    root: &Rc<File>,
    kind: &str,
    sharded: bool,
    payloads: &mut Vec<Payload>,
) -> Result<()> {
    let Some(parent) = child_directory(root, kind)? else {
        return Ok(());
    };
    let parent = Rc::new(parent);
    if sharded {
        for shard in names(&parent)?.into_iter().filter(|name| hex(name, 2)) {
            let directory =
                Rc::new(child_directory(&parent, &shard)?.context("legacy shard disappeared")?);
            for name in names(&directory)?
                .into_iter()
                .filter(|name| payload_name(name, false) && name.starts_with(&shard))
            {
                if let Some(file) = owned_file(&directory, &name, false)? {
                    payloads.push(Payload {
                        parent: directory.clone(),
                        name,
                        bytes: file.metadata()?.len(),
                    });
                }
            }
        }
    } else {
        for name in names(&parent)?
            .into_iter()
            .filter(|name| payload_name(name, true))
        {
            if let Some(file) = owned_file(&parent, &name, false)? {
                payloads.push(Payload {
                    parent: parent.clone(),
                    name,
                    bytes: file.metadata()?.len(),
                });
            }
        }
    }
    Ok(())
}

fn legacy_root(directory: File) -> Result<Root> {
    let directory = Rc::new(directory);
    let lock = lock_root(&directory, "moespresso-disk-kv.lock")?;
    let age = age(&directory)?;
    let target = legacy_index(&directory, "moespresso-disk-kv-v1")?;
    let mut payloads = Vec::new();
    collect_payloads(&directory, "payloads", true, &mut payloads)?;
    collect_payloads(&directory, "quarantine", false, &mut payloads)?;
    let mut complete = true;
    let companion = if let Some(attachments) = child_directory(&directory, "attachments")? {
        let attachments = Rc::new(attachments);
        let before = payloads.len();
        collect_payloads(&attachments, "payloads", true, &mut payloads)?;
        collect_payloads(&attachments, "quarantine", false, &mut payloads)?;
        let index = legacy_index(&attachments, "moespresso-disk-kv-attachment-v1")?;
        complete = index.is_some() || payloads.len() == before;
        index.map(|index| (attachments, index))
    } else {
        None
    };
    let bytes = payloads
        .iter()
        .try_fold(0, |sum, payload| checked_add(sum, payload.bytes))?;
    Ok(Root {
        directory,
        lock,
        contents: Contents::Legacy {
            target,
            companion,
            complete,
        },
        payloads,
        bytes,
        age,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
    use std::path::PathBuf;

    struct Tree(PathBuf);
    impl Tree {
        fn new() -> Result<Self> {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos();
            let base = std::env::temp_dir()
                .join(format!("moespresso-managed-{}-{stamp}", std::process::id()));
            std::fs::DirBuilder::new().mode(0o700).create(&base)?;
            Ok(Self(base))
        }
        fn directory(&self, relative: &str) -> Result<PathBuf> {
            let path = self.0.join(relative);
            std::fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(&path)?;
            Ok(path)
        }
    }
    impl Drop for Tree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn leaf(path: &Path, bytes: &[u8]) -> Result<File> {
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)?;
        file.write_all(bytes)?;
        Ok(file)
    }

    fn native(tree: &Tree, key: char, bytes: usize) -> Result<PathBuf> {
        let root = tree.directory(&format!("v4/disk_kv/{}", key.to_string().repeat(64)))?;
        let dir = open_root(&root)?;
        leaf(&root.join(LOCK), &[])?;
        if bytes > 0 {
            leaf(&root.join("payload-00.bin"), &vec![0; bytes])?;
            let mut index = IndexFile::empty();
            index.next_recency = 2;
            index.entries.push(Entry {
                slot: 0,
                scope: [7; 32],
                prefix: [8; 32],
                frontier: 1,
                file_bytes: bytes as u64,
                recency: 1,
            });
            publish_index(&dir, &index).map_err(|error| match error {
                PublishError::Before(error) | PublishError::After(error) => error,
            })?;
        } else {
            initialize_or_read_index(&dir)?;
        }
        Ok(root)
    }

    fn legacy(tree: &Tree, key: char, bytes: usize) -> Result<(PathBuf, PathBuf)> {
        let root = tree.directory(&format!("disk_kv/{}", key.to_string().repeat(12)))?;
        let payload = root.join(format!("payloads/aa/{}.safetensors", "a".repeat(64)));
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(payload.parent().unwrap())?;
        leaf(&payload, &vec![0; bytes])?;
        leaf(&root.join("moespresso-disk-kv.lock"), &[])?;
        let index = json!({ "schema_version": "moespresso-disk-kv-v1", "entries": [{
            "schema_version": "moespresso-disk-kv-v1", "cache_id": "a".repeat(64),
            "payload_path": format!("payloads/aa/{}.safetensors", "a".repeat(64)),
            "payload_bytes": bytes }] });
        leaf(&root.join("index.json"), &serde_json::to_vec(&index)?)?;
        Ok((root, payload))
    }

    #[test]
    fn aggregate_cleanup_reclaims_inactive_native_and_legacy_without_touching_live_or_manual_files()
    -> Result<()> {
        let tree = Tree::new()?;
        let current = native(&tree, 'a', 200)?;
        let inactive = native(&tree, 'b', 200)?;
        let (old, old_payload) = legacy(&tree, 'c', 300)?;
        let attachment_directory =
            tree.directory("disk_kv/cccccccccccc/attachments/payloads/bb")?;
        let attachment = attachment_directory.join(format!("{}.safetensors", "b".repeat(64)));
        leaf(&attachment, &[4; 80])?;
        leaf(
            &old.join("attachments/index.json"),
            &serde_json::to_vec(&json!({
            "schema_version": "moespresso-disk-kv-attachment-v1", "entries": [] }))?,
        )?;
        let quarantine_directory = tree.directory("disk_kv/cccccccccccc/quarantine")?;
        let quarantined = quarantine_directory.join(format!("{}.123.safetensors", "a".repeat(64)));
        leaf(&quarantined, &[2; 40])?;
        std::fs::set_permissions(&old, std::fs::Permissions::from_mode(0o755))?;
        std::fs::set_permissions(
            old.join("moespresso-disk-kv.lock"),
            std::fs::Permissions::from_mode(0o644),
        )?;
        let (active, active_payload) = legacy(&tree, 'd', 200)?;
        let live_lock = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(active.join("moespresso-disk-kv.lock"))?;
        // Use the actual Python runtime's lock operation to check interoperability.
        assert_eq!(
            unsafe { libc::flock(live_lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        let manual = tree.directory("disk_kv/manual")?;
        leaf(&manual.join("notes.bin"), b"preserve")?;
        leaf(&old.join("notes.txt"), b"preserve")?;
        let manager = ManagedCache::open(&tree.0)?;
        let reservation = manager.reserve(&open_root(&current)?, 200, 150, 600)?;
        assert_eq!(reservation.allowance, 400);
        assert!(!inactive.join("payload-00.bin").exists());
        assert!(!old_payload.exists());
        assert!(!attachment.exists());
        assert!(!quarantined.exists());
        assert_eq!(std::fs::read(active_payload)?.len(), 200);
        assert_eq!(std::fs::read(manual.join("notes.bin"))?, b"preserve");
        assert_eq!(std::fs::read(old.join("notes.txt"))?, b"preserve");
        assert!(
            read_index(&open_root(&inactive)?)?
                .unwrap()
                .entries
                .is_empty()
        );
        let old_index: serde_json::Value =
            serde_json::from_slice(&std::fs::read(old.join("index.json"))?)?;
        assert!(old_index["entries"].as_array().unwrap().is_empty());
        drop(reservation);
        let newer = native(&tree, 'e', 200)?;
        let repeated = manager.reserve(&open_root(&current)?, 200, 150, 600)?;
        assert_eq!(repeated.allowance, 400);
        assert!(!newer.join("payload-00.bin").exists());
        Ok(())
    }

    #[test]
    fn active_or_unrecognized_stores_reduce_available_bytes_without_being_deleted() -> Result<()> {
        let tree = Tree::new()?;
        let current = native(&tree, 'a', 200)?;
        let protected = native(&tree, 'b', 200)?;
        std::fs::write(protected.join(INDEX), b"unrecognized index")?;
        let active = native(&tree, 'c', 200)?;
        let lock = open_leaf(&open_root(&active)?, LOCK, libc::O_RDWR)?;
        lock.try_lock()?;
        let manager = ManagedCache::open(&tree.0)?;
        let reservation = manager.reserve(&open_root(&current)?, 200, 100, 300)?;
        assert_eq!(reservation.allowance, 0);
        assert_eq!(std::fs::read(protected.join(INDEX))?, b"unrecognized index");
        assert!(protected.join("payload-00.bin").exists());
        assert!(active.join("payload-00.bin").exists());
        Ok(())
    }

    #[test]
    fn suspicious_managed_leaves_refuse_cleanup_before_any_root_is_changed() -> Result<()> {
        for hard_link in [false, true] {
            let tree = Tree::new()?;
            let current = native(&tree, 'a', 200)?;
            let ordinary = native(&tree, 'b', 200)?;
            let suspicious = native(&tree, 'c', 0)?;
            let outside = tree.0.join("outside.bin");
            leaf(&outside, &[3; 200])?;
            let link = suspicious.join("payload-00.bin");
            if hard_link {
                std::fs::hard_link(&outside, &link)?;
            } else {
                std::os::unix::fs::symlink(&outside, &link)?;
            }
            let manager = ManagedCache::open(&tree.0)?;
            assert!(
                manager
                    .reserve(&open_root(&current)?, 200, 100, 200)
                    .is_err()
            );
            assert_eq!(std::fs::read(&outside)?, [3; 200]);
            assert!(ordinary.join("payload-00.bin").exists());
            assert_eq!(
                read_index(&open_root(&ordinary)?)?.unwrap().entries.len(),
                1
            );
        }
        Ok(())
    }
}
