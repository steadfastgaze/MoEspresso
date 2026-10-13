//! Read-only macOS memory observations for a caller-owned admission policy.
//!
//! Free plus inactive pages approximate reclaimable memory. They are a
//! snapshot, not an allocation guarantee. Metal's recommended working set is
//! observed separately through `Metal::memory_stats`; these quantities must
//! not be added as independent pools of memory.

use anyhow::{Context, Result, ensure};
use std::ffi::CStr;
use std::mem::{offset_of, size_of, zeroed};
use std::ptr;

const MIB: u64 = 1 << 20;

unsafe extern "C" {
    fn mach_port_deallocate(
        task: libc::mach_port_t,
        name: libc::mach_port_t,
    ) -> libc::kern_return_t;
}

struct HostPort(libc::host_t);

impl Drop for HostPort {
    #[allow(deprecated)]
    fn drop(&mut self) {
        // SAFETY: mach_host_self supplied this send-right reference; the task
        // self port is a process-owned special port, not a new owned right.
        let _ = unsafe { mach_port_deallocate(libc::mach_task_self(), self.0) };
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostMemoryObservation {
    /// Installed dynamic RAM reported by `hw.memsize`.
    pub physical_ram_bytes: u64,
    /// `HOST_VM_INFO64` free and inactive pages at the sampling instant.
    pub free_inactive_bytes: u64,
    /// Positive `iogpu.wired_limit_mb`, if the host exposes it.
    pub gpu_wired_limit_bytes: Option<u64>,
}

fn resolve(
    physical_ram_bytes: Option<u64>,
    vm_pages: Option<(u64, u64, u64)>,
    wired_limit_mb: Option<u64>,
) -> Result<HostMemoryObservation> {
    let physical_ram_bytes = physical_ram_bytes.context("hw.memsize is unavailable")?;
    ensure!(physical_ram_bytes > 0, "hw.memsize must be positive");
    let (free_pages, inactive_pages, page_bytes) =
        vm_pages.context("HOST_VM_INFO64 or page size is unavailable")?;
    ensure!(page_bytes > 0, "host page size must be positive");
    let free_inactive_bytes = free_pages
        .checked_add(inactive_pages)
        .and_then(|pages| pages.checked_mul(page_bytes))
        .context("free and inactive page byte count overflow")?;
    ensure!(
        free_inactive_bytes <= physical_ram_bytes,
        "free and inactive pages exceed installed RAM"
    );
    let gpu_wired_limit_bytes = wired_limit_mb
        .filter(|&megabytes| megabytes > 0)
        .and_then(|megabytes| megabytes.checked_mul(MIB));
    Ok(HostMemoryObservation {
        physical_ram_bytes,
        free_inactive_bytes,
        gpu_wired_limit_bytes,
    })
}

fn sysctl_positive(name: &CStr) -> Option<u64> {
    let mut len = 0usize;
    // SAFETY: `name` is NUL-terminated, and the call only queries byte length.
    if unsafe { libc::sysctlbyname(name.as_ptr(), ptr::null_mut(), &mut len, ptr::null_mut(), 0) }
        != 0
    {
        return None;
    }
    if len != size_of::<i32>() && len != size_of::<i64>() {
        return None;
    }
    let mut raw = [0u8; size_of::<i64>()];
    // SAFETY: The output buffer is eight bytes and `len` was checked above.
    if unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            raw.as_mut_ptr().cast(),
            &mut len,
            ptr::null_mut(),
            0,
        )
    } != 0
    {
        return None;
    }
    let value = match len {
        4 => i64::from(i32::from_ne_bytes(raw[..4].try_into().ok()?)),
        8 => i64::from_ne_bytes(raw),
        _ => return None,
    };
    u64::try_from(value).ok().filter(|&value| value > 0)
}

// libc deprecates its Mach bindings in favor of a separate crate. This narrow
// observation uses the existing C ABI without adding another wrapper layer.
#[allow(deprecated)]
fn vm_pages() -> Result<(u64, u64, u64)> {
    // SAFETY: This Mach struct contains only integer counters and reserved
    // integer fields. The kernel fills it before any counters are read.
    let mut stats: libc::vm_statistics64_data_t = unsafe { zeroed() };
    let mut count = libc::HOST_VM_INFO64_COUNT;
    let required_count = u32::try_from(
        (offset_of!(libc::vm_statistics64_data_t, inactive_count) + size_of::<libc::natural_t>())
            .div_ceil(size_of::<libc::integer_t>()),
    )?;
    // SAFETY: `stats` has the ABI layout and count requested by this flavor;
    // both output pointers remain valid for the duration of the call.
    // SAFETY: mach_host_self returns a send right or a null port.
    let host = unsafe { libc::mach_host_self() };
    ensure!(host != 0, "mach_host_self returned a null port");
    let host = HostPort(host);
    let status = unsafe {
        libc::host_statistics64(
            host.0,
            libc::HOST_VM_INFO64,
            (&raw mut stats).cast::<libc::integer_t>(),
            &raw mut count,
        )
    };
    ensure!(
        status == libc::KERN_SUCCESS && count >= required_count,
        "HOST_VM_INFO64 failed or omitted free/inactive counters: status={status}, count={count}"
    );
    // SAFETY: sysconf only reads the host page-size setting.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page_bytes = u64::try_from(page_size).context("host page size is unavailable")?;
    Ok((
        u64::from(stats.free_count),
        u64::from(stats.inactive_count),
        page_bytes,
    ))
}

/// Sample required host memory quantities without changing host settings.
/// The optional GPU wired limit is unknown when its sysctl is absent or invalid.
pub fn observe_host_memory() -> Result<HostMemoryObservation> {
    let physical = sysctl_positive(c"hw.memsize");
    let pages = vm_pages()?;
    let wired_mb = sysctl_positive(c"iogpu.wired_limit_mb");
    resolve(physical, Some(pages), wired_mb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_observations_fail_closed() {
        assert!(resolve(None, Some((1, 1, 4096)), None).is_err());
        assert!(resolve(Some(4096), None, None).is_err());
        assert!(resolve(Some(0), Some((0, 0, 4096)), None).is_err());
        assert!(resolve(Some(4096), Some((0, 0, 0)), None).is_err());
    }

    #[test]
    fn checked_pages_and_optional_wired_limit() {
        assert!(resolve(Some(u64::MAX), Some((u64::MAX, 1, 1)), None).is_err());
        assert!(resolve(Some(u64::MAX), Some((1, 1, u64::MAX)), None).is_err());
        assert!(resolve(Some(4096), Some((1, 1, 4096)), None).is_err());
        let observed = resolve(Some(16 * MIB), Some((2, 3, 4096)), Some(u64::MAX)).unwrap();
        assert_eq!(observed.free_inactive_bytes, 5 * 4096);
        assert_eq!(observed.gpu_wired_limit_bytes, None);
        assert_eq!(
            resolve(Some(16 * MIB), Some((2, 3, 4096)), Some(8))
                .unwrap()
                .gpu_wired_limit_bytes,
            Some(8 * MIB)
        );
    }

    #[test]
    fn macos_host_observation_is_plausible() {
        let observed = observe_host_memory().unwrap();
        assert!(observed.physical_ram_bytes > 0);
        assert!(observed.free_inactive_bytes <= observed.physical_ram_bytes);
        println!("host memory observation: {observed:?}");
    }
}
