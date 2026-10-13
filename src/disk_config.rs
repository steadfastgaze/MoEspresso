//! Native serving disk policy, resolved before model allocation.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs::{self, DirBuilder};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use moespresso::engine::DiskCacheMode;
use moespresso::package::Package;
use moespresso::server::ServeDiskOptions;

const ENV_KEYS: [&str; 2] = ["XDG_CACHE_HOME", "HOME"];
type Environment = BTreeMap<&'static str, OsString>;

pub(crate) struct Resolution {
    pub disk: Option<ServeDiskOptions>,
    pub unavailable: Option<String>,
}

impl Resolution {
    fn off() -> Self {
        Self {
            disk: None,
            unavailable: None,
        }
    }
}

fn resolve_with(
    mode: Option<DiskCacheMode>,
    env: &Environment,
    fixed_capacity: bool,
    disabled: bool,
    explicit_root: Option<&Path>,
    byte_budget: Option<u64>,
    package_manifest_digest: impl FnOnce() -> [u8; 32],
) -> Result<Resolution> {
    let required = explicit_root.is_some() || byte_budget.is_some() || mode.is_some();
    ensure!(
        !required || (!disabled && !fixed_capacity),
        "disk cache options conflict with --no-disk-cache and --expert-capacity"
    );
    if disabled || fixed_capacity {
        return Ok(Resolution::off());
    }
    let max_payload_bytes = byte_budget.unwrap_or(8 << 30);
    ensure!(max_payload_bytes > 0, "disk cache budget must be positive");
    let resolve_root = || -> Result<(PathBuf, Option<PathBuf>)> {
        if let Some(root) = explicit_root {
            ensure!(
                !root.as_os_str().is_empty(),
                "disk cache directory must not be empty"
            );
            return Ok((std::path::absolute(root)?, None));
        }
        let base = match env.get("XDG_CACHE_HOME").filter(|value| !value.is_empty()) {
            Some(base) => PathBuf::from(base),
            None => PathBuf::from(
                env.get("HOME")
                    .filter(|value| !value.is_empty())
                    .context("HOME or XDG_CACHE_HOME is required for the default disk cache")?,
            )
            .join(".cache"),
        };
        ensure!(base.is_absolute(), "disk cache base must be absolute");
        let key = package_manifest_digest()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let managed_base = base.join("moespresso");
        Ok((
            managed_base.join("v4/disk_kv").join(key),
            Some(managed_base),
        ))
    };
    let (root, managed_cache_base) = match resolve_root() {
        Ok(root) => root,
        Err(error) if !required => {
            return Ok(Resolution {
                disk: None,
                unavailable: Some(format!("{error:#}")),
            });
        }
        Err(error) => return Err(error),
    };
    Ok(Resolution {
        disk: Some(ServeDiskOptions {
            mode: mode.unwrap_or_default(),
            root,
            managed_cache_base,
            max_file_bytes: max_payload_bytes.min(8 << 30),
            max_io_buffer_bytes: moespresso::engine::IO_BUFFER_BYTES,
            max_payload_bytes,
            max_entries: 64,
            stride: 1024,
            write_depth: 16384,
            required,
        }),
        unavailable: None,
    })
}

pub(crate) fn resolve(
    package: &Package,
    fixed_capacity: bool,
    disabled: bool,
    explicit_root: Option<&Path>,
    byte_budget: Option<u64>,
    mode: Option<DiskCacheMode>,
) -> Result<Resolution> {
    let env = ENV_KEYS
        .into_iter()
        .filter_map(|key| std::env::var_os(key).map(|value| (key, value)))
        .collect();
    resolve_with(
        mode,
        &env,
        fixed_capacity,
        disabled,
        explicit_root,
        byte_budget,
        || package.manifest_digest(),
    )
}

/// Create missing directories privately without changing existing permissions.
/// The store independently pins and checks the final root before using it.
pub(crate) fn prepare_root(root: &Path) -> Result<()> {
    ensure!(root.is_absolute(), "disk cache root must be absolute");
    DirBuilder::new().recursive(true).mode(0o700).create(root)?;
    let metadata = fs::symlink_metadata(root)?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "disk cache root must be a directory, not a symlink"
    );
    // SAFETY: geteuid takes no pointers and returns the process effective UID.
    ensure!(
        metadata.uid() == unsafe { libc::geteuid() },
        "disk cache root has another owner"
    );
    ensure!(
        metadata.permissions().mode() & 0o077 == 0,
        "disk cache root must already be private (0700)"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn env(pairs: &[(&'static str, &str)]) -> Environment {
        pairs
            .iter()
            .map(|&(key, value)| (key, value.into()))
            .collect()
    }
    fn package() -> [u8; 32] {
        [0xab; 32]
    }

    #[test]
    fn disabled_cache_does_not_resolve_storage() -> Result<()> {
        for (fixed, disabled) in [(false, true), (true, false), (true, true)] {
            let result = resolve_with(
                None,
                &Environment::new(),
                fixed,
                disabled,
                None,
                None,
                || panic!("disabled disk cache must not resolve package identity"),
            )?;
            assert!(result.disk.is_none());
            assert!(result.unavailable.is_none());
        }
        Ok(())
    }

    #[test]
    fn defaults_use_manifest_identity_and_bounded_limits() -> Result<()> {
        let a = resolve_with(
            None,
            &env(&[("HOME", "/fixture-home")]),
            false,
            false,
            None,
            None,
            package,
        )?
        .disk
        .unwrap();
        let b = resolve_with(
            None,
            &env(&[("XDG_CACHE_HOME", "/cache")]),
            false,
            false,
            None,
            None,
            package,
        )?
        .disk
        .unwrap();
        assert!(
            a.root
                .starts_with("/fixture-home/.cache/moespresso/v4/disk_kv")
        );
        assert!(b.root.starts_with("/cache/moespresso/v4/disk_kv"));
        assert_eq!(a.root.file_name(), b.root.file_name());
        assert_eq!(a.root.file_name().unwrap(), "ab".repeat(32).as_str());
        assert!(!a.required);
        assert_eq!(a.mode, DiskCacheMode::Prefix);
        assert_eq!(
            a.managed_cache_base.as_deref(),
            Some(Path::new("/fixture-home/.cache/moespresso"))
        );
        assert_eq!(
            b.managed_cache_base.as_deref(),
            Some(Path::new("/cache/moespresso"))
        );
        assert_eq!((a.max_entries, a.stride, a.write_depth), (64, 1024, 16384));
        assert_eq!(
            (a.max_payload_bytes, a.max_file_bytes, a.max_io_buffer_bytes),
            (8 << 30, 8 << 30, moespresso::engine::IO_BUFFER_BYTES)
        );
        let different = resolve_with(
            None,
            &env(&[("HOME", "/fixture-home")]),
            false,
            false,
            None,
            None,
            || [0xcd; 32],
        )?
        .disk
        .unwrap();
        assert_ne!(a.root, different.root);
        Ok(())
    }

    #[test]
    fn explicit_location_and_budget_are_required_and_conflicts_fail_closed() -> Result<()> {
        let configured = resolve_with(
            None,
            &Environment::new(),
            false,
            false,
            Some(Path::new("/explicit")),
            Some(2 << 30),
            || panic!("explicit root needs no package identity lookup"),
        )?
        .disk
        .unwrap();
        assert_eq!(configured.root, Path::new("/explicit"));
        assert_eq!(configured.max_payload_bytes, 2 << 30);
        assert_eq!((configured.stride, configured.write_depth), (1024, 16384));
        assert!(configured.required);
        assert!(configured.managed_cache_base.is_none());
        for (fixed, disabled) in [(true, false), (false, true)] {
            for (root, bytes) in [(Some(Path::new("/explicit")), None), (None, Some(123))] {
                assert!(
                    resolve_with(
                        None,
                        &Environment::new(),
                        fixed,
                        disabled,
                        root,
                        bytes,
                        package
                    )
                    .is_err()
                );
            }
        }
        assert!(
            resolve_with(
                None,
                &Environment::new(),
                false,
                false,
                Some(Path::new("")),
                None,
                package
            )
            .is_err()
        );
        assert!(
            resolve_with(
                None,
                &Environment::new(),
                false,
                false,
                None,
                Some(0),
                package
            )
            .is_err()
        );
        let relative = resolve_with(
            None,
            &Environment::new(),
            false,
            false,
            Some(Path::new("local-cache")),
            None,
            package,
        )?
        .disk
        .unwrap();
        assert_eq!(relative.root, std::env::current_dir()?.join("local-cache"));
        Ok(())
    }

    #[test]
    fn unavailable_default_location_is_soft_but_explicit_configuration_is_fatal() -> Result<()> {
        for input in [Environment::new(), env(&[("XDG_CACHE_HOME", "relative")])] {
            let resolved = resolve_with(None, &input, false, false, None, None, package)?;
            assert!(resolved.disk.is_none() && resolved.unavailable.is_some());
            assert!(resolve_with(None, &input, false, false, None, Some(1024), package).is_err());
        }
        Ok(())
    }

    #[test]
    fn explicit_full_mode_is_required_and_does_not_change_storage_identity() -> Result<()> {
        let env = env(&[("HOME", "/fixture-home")]);
        let prefix = resolve_with(None, &env, false, false, None, None, package)?
            .disk
            .unwrap();
        let full = resolve_with(
            Some(DiskCacheMode::Full),
            &env,
            false,
            false,
            None,
            None,
            package,
        )?
        .disk
        .unwrap();
        assert_eq!(full.mode, DiskCacheMode::Full);
        assert!(full.required);
        assert_eq!(full.root, prefix.root);
        for (fixed, disabled) in [(true, false), (false, true)] {
            assert!(
                resolve_with(
                    Some(DiskCacheMode::Prefix),
                    &env,
                    fixed,
                    disabled,
                    None,
                    None,
                    package
                )
                .is_err()
            );
        }
        assert!(
            resolve_with(
                Some(DiskCacheMode::Full),
                &Environment::new(),
                false,
                false,
                None,
                None,
                package
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn directory_creation_is_private_and_never_repairs_existing_permissions() -> Result<()> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let base = std::env::temp_dir().join(format!(
            "moespresso-disk-config-{}-{stamp}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        struct Cleanup(PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Cleanup(base.clone());
        let root = base.join("nested/root");
        prepare_root(&root)?;
        assert_eq!(fs::metadata(&root)?.permissions().mode() & 0o777, 0o700);
        assert_eq!(
            fs::metadata(root.parent().unwrap())?.permissions().mode() & 0o777,
            0o700
        );
        fs::set_permissions(&root, fs::Permissions::from_mode(0o755))?;
        assert!(prepare_root(&root).is_err());
        assert_eq!(fs::metadata(&root)?.permissions().mode() & 0o777, 0o755);
        let link = base.join("link");
        std::os::unix::fs::symlink(&root, &link)?;
        assert!(prepare_root(&link).is_err());
        Ok(())
    }
}
