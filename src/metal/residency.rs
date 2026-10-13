//! Scoped Metal residency requests without extending Rust buffer ownership.

use super::{BufferInner, Metal};
use anyhow::{Context, Result, ensure};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLAllocation, MTLCommandQueue, MTLDevice, MTLGPUFamily, MTLResidencySet,
    MTLResidencySetDescriptor,
};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::{Rc, Weak};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Support {
    Available,
    UnsupportedMacOs,
    UnsupportedMetal,
}

impl Support {
    fn for_device(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        if !objc2::available!(macos = 15.0) {
            Self::UnsupportedMacOs
        } else if !device.supportsFamily(MTLGPUFamily::Metal3) {
            Self::UnsupportedMetal
        } else {
            Self::Available
        }
    }

    fn reason(self) -> &'static str {
        match self {
            Self::Available => "supported",
            Self::UnsupportedMacOs => "unsupported-macos",
            Self::UnsupportedMetal => "unsupported-metal",
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct Policy {
    enabled: bool,
    reason: &'static str,
}

fn select_policy(default_enabled: bool, support: Support) -> Policy {
    Policy {
        enabled: default_enabled && support == Support::Available,
        reason: if default_enabled {
            support.reason()
        } else {
            "disabled"
        },
    }
}

pub(super) struct PrefillResidency {
    set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    buffers: RefCell<HashMap<*const BufferInner, Weak<BufferInner>>>,
    active: Cell<bool>,
}

impl PrefillResidency {
    pub(super) fn for_policy(
        device: &ProtocolObject<dyn MTLDevice>,
        default_enabled: bool,
        report_default: Option<&str>,
    ) -> Option<Self> {
        let policy = select_policy(default_enabled, Support::for_device(device));
        let mut fallback = None;
        let residency = if policy.enabled {
            match Self::new(device) {
                Ok(residency) => Some(residency),
                Err(error) => {
                    fallback = Some(error);
                    None
                }
            }
        } else {
            None
        };
        if let Some(default) = report_default {
            let detail = fallback
                .as_ref()
                .map(|error| format!(" detail={error:#}"))
                .unwrap_or_default();
            eprintln!(
                "prefill residency: enabled={} default={} reason={}{}",
                residency.is_some(),
                default,
                if fallback.is_some() {
                    "set-creation-failed"
                } else {
                    policy.reason
                },
                detail,
            );
        }
        residency
    }

    fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let descriptor = MTLResidencySetDescriptor::new();
        descriptor.setLabel(Some(&NSString::from_str("MoEspresso prefill residency")));
        let set = device
            .newResidencySetWithDescriptor_error(&descriptor)
            .context("create prefill Metal residency set")?;
        Ok(Self {
            set,
            buffers: RefCell::new(HashMap::new()),
            active: Cell::new(false),
        })
    }

    pub(super) fn track(&self, buffer: &Rc<BufferInner>) -> Result<()> {
        {
            let mut buffers = self.buffers.borrow_mut();
            buffers.try_reserve(1)?;
            buffers.insert(Rc::as_ptr(buffer), Rc::downgrade(buffer));
        }
        if self.active.get() {
            self.set.addAllocation(allocation(buffer));
            // Commit before any caller can submit this new allocation.
            self.set.commit();
        }
        Ok(())
    }

    pub(super) fn untrack(&self, buffer: &BufferInner) {
        self.buffers
            .borrow_mut()
            .remove(&(buffer as *const BufferInner));
        if self.active.get() && self.set.containsAllocation(allocation(buffer)) {
            self.set.removeAllocation(allocation(buffer));
            self.set.commit();
        }
    }

    pub(super) fn begin(&self, metal: &Metal) -> Result<PrefillResidencyGuard> {
        ensure!(
            !self.active.get(),
            "prefill residency guard is already active"
        );
        let mut live = Vec::new();
        {
            let buffers = self.buffers.borrow();
            live.try_reserve(buffers.len())?;
            live.extend(buffers.values().filter_map(Weak::upgrade));
        }
        // Fallible preparation and registry borrows finish before Metal owns
        // allocations. No Rust buffer ownership is retained by the guard.
        for buffer in &live {
            self.set.addAllocation(allocation(buffer));
        }
        self.set.commit();
        self.set.requestResidency();
        metal.0.queue.addResidencySet(&self.set);
        self.active.set(true);
        Ok(PrefillResidencyGuard {
            metal: metal.clone(),
        })
    }

    fn finish(&self, metal: &Metal) {
        if self.active.replace(false) {
            metal.0.queue.removeResidencySet(&self.set);
            self.set.endResidency();
            self.set.removeAllAllocations();
            self.set.commit();
        }
    }
}

fn allocation(buffer: &BufferInner) -> &ProtocolObject<dyn MTLAllocation> {
    ProtocolObject::<dyn MTLAllocation>::from_ref(&*buffer.raw)
}

/// Holds only the device identity; buffer retirement remains with the stream.
pub(crate) struct PrefillResidencyGuard {
    metal: Metal,
}

impl Drop for PrefillResidencyGuard {
    fn drop(&mut self) {
        if let Some(residency) = &self.metal.0.prefill_residency {
            residency.finish(&self.metal);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Metal, Support, allocation, select_policy};
    use anyhow::Result;
    use objc2_metal::MTLResidencySet;
    use std::rc::Rc;

    #[test]
    fn prefill_residency_defaults_to_supported_bounded_engines_only() {
        let bounded = select_policy(true, Support::Available);
        assert!(bounded.enabled);
        assert_eq!(bounded.reason, "supported");
        assert!(!select_policy(false, Support::Available).enabled);
        for support in [Support::UnsupportedMacOs, Support::UnsupportedMetal] {
            let bounded = select_policy(true, support);
            assert!(!bounded.enabled);
            assert_eq!(bounded.reason, support.reason());
            assert!(!select_policy(false, support).enabled);
        }
    }

    #[test]
    fn disabled_prefill_residency_does_not_track_buffers() -> Result<()> {
        let metal = Metal::with_prefill_residency(false)?;
        let buffer = metal.shared_buffer(4)?;
        assert!(metal.0.prefill_residency.is_none());
        assert!(metal.prefill_residency_guard()?.is_none());
        buffer.write_bytes(0, &[1, 2, 3, 4])?;
        Ok(())
    }

    #[test]
    fn prefill_residency_guard_tracks_new_and_dropped_allocations() -> Result<()> {
        let metal = Metal::with_prefill_residency(true)?;
        let first = metal.shared_buffer(4)?;
        let alias = first.clone();
        let retired = metal.shared_buffer(8)?;
        let weak = Rc::downgrade(&retired.0);
        drop(retired);
        assert!(weak.upgrade().is_none());
        let residency = metal.0.prefill_residency.as_ref().unwrap();
        assert_eq!(residency.buffers.borrow().len(), 1);
        assert_eq!(residency.set.allocationCount(), 0);
        let guard = metal.prefill_residency_guard()?.unwrap();
        assert!(residency.active.get());
        assert_eq!(residency.set.allocationCount(), 1);
        assert!(residency.set.containsAllocation(allocation(&first.0)));
        assert!(metal.prefill_residency_guard().is_err());
        // Residency does not introduce command reservations or relax them.
        first.write_bytes(0, &[1, 2, 3, 4])?;
        let requested_before = metal.memory_stats()?.live_requested_buffer_bytes;
        let rejected = {
            let fault = metal.test_fail_allocation_after(0)?;
            let result = metal.shared_buffer(32);
            assert!(fault.fired());
            result
        };
        assert!(rejected.is_err());
        assert_eq!(residency.buffers.borrow().len(), 1);
        assert_eq!(residency.set.allocationCount(), 1);
        assert_eq!(
            metal.memory_stats()?.live_requested_buffer_bytes,
            requested_before
        );
        let transient = metal.shared_buffer(16)?;
        assert_eq!(residency.set.allocationCount(), 2);
        let transient_weak = Rc::downgrade(&transient.0);
        drop(transient);
        assert!(transient_weak.upgrade().is_none());
        assert_eq!(residency.set.allocationCount(), 1);
        drop(first);
        assert_eq!(residency.set.allocationCount(), 1);
        drop(alias);
        assert_eq!(residency.set.allocationCount(), 0);
        assert!(residency.buffers.borrow().is_empty());
        drop(guard);
        assert!(!residency.active.get());
        assert_eq!(residency.set.allocationCount(), 0);
        drop(metal.prefill_residency_guard()?);
        Ok(())
    }

    #[test]
    fn prefill_residency_guard_cleans_up_after_submission_failure() -> Result<()> {
        let metal = Metal::with_prefill_residency(true)?;
        let buffer = metal.shared_buffer(4)?;
        let pipeline = metal.compile_msl_strict(
            "#include <metal_stdlib>\nusing namespace metal;\nkernel void touch(device uint* out [[buffer(0)]]) { out[0] = 17; }",
            "touch",
        )?;
        let failed: Result<()> = (|| {
            let _guard = metal.prefill_residency_guard()?;
            let mut stream = metal.stream()?;
            // SAFETY: One thread writes the checked uint extent.
            unsafe {
                stream.batch()?.dispatch_1d(
                    &pipeline,
                    &[super::super::Binding {
                        buffer: &buffer,
                        offset: 0,
                    }],
                    1,
                    1,
                )?;
            }
            let reservation = buffer.write_bytes(0, &[0; 4]);
            assert!(reservation.is_err());
            stream.flush()?;
            let _fault = metal.test_fail_completed_command_after(0)?;
            stream.wait()?;
            Ok(())
        })();
        assert!(failed.is_err());
        let residency = metal.0.prefill_residency.as_ref().unwrap();
        assert!(!residency.active.get());
        assert_eq!(residency.set.allocationCount(), 0);
        buffer.write_bytes(0, &[0; 4])?;
        drop(metal.prefill_residency_guard()?);
        Ok(())
    }
}
