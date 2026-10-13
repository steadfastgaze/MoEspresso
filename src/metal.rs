//! Small Metal resource and command layer for explicit model execution.
//!
//! A batch retains every bound or declared buffer until its command completes.
//! A stream may submit ordered batches that share those buffers. Safe host byte
//! access fails until every encoded or submitted use retires. The crate-private
//! unsafe strided copier additionally permits proven disjoint destination spans.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::ptr;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, anyhow, bail, ensure};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLEvent, MTLLibrary, MTLResource, MTLResourceOptions,
    MTLResourceUsage, MTLSharedEvent, MTLSize,
};

mod residency;
pub(crate) use residency::PrefillResidencyGuard;

static NEXT_OWNER_ID: AtomicUsize = AtomicUsize::new(1);

fn next_owner_id() -> Result<usize> {
    NEXT_OWNER_ID
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
            next.checked_add(1)
        })
        .map_err(|_| anyhow!("Metal command owner ID space exhausted"))
}

struct MetalInner {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    accounting: BufferAccounting,
    pipelines: RefCell<HashMap<String, SourcePipelines>>,
    prefill_residency: Option<residency::PrefillResidency>,
    #[cfg(test)]
    faults: Rc<TestFaults>,
}

#[cfg(test)]
#[derive(Clone, Copy, Eq, PartialEq)]
enum TestFaultKind {
    Allocation,
    CompletedCommand,
    HandoffEncoding,
}

#[cfg(test)]
#[derive(Default)]
struct TestFaults {
    armed: Cell<Option<(TestFaultKind, usize)>>,
    active: Cell<bool>,
    fired: Cell<bool>,
}

#[cfg(test)]
impl TestFaults {
    fn arm(self: &Rc<Self>, kind: TestFaultKind, skip: usize) -> Result<TestFaultGuard> {
        ensure!(!self.active.get(), "a Metal test fault is already armed");
        self.active.set(true);
        self.fired.set(false);
        self.armed.set(Some((kind, skip)));
        Ok(TestFaultGuard(self.clone()))
    }

    fn check(&self, kind: TestFaultKind) -> Result<()> {
        if let Some((armed_kind, skip)) = self.armed.get()
            && armed_kind == kind
        {
            if skip > 0 {
                self.armed.set(Some((kind, skip - 1)));
            } else {
                self.armed.set(None);
                self.fired.set(true);
                bail!(match kind {
                    TestFaultKind::Allocation => "injected Metal allocation rejection",
                    TestFaultKind::CompletedCommand => "injected completed Metal command error",
                    TestFaultKind::HandoffEncoding => "injected decode handoff encoding error",
                });
            }
        }
        Ok(())
    }
}

/// One device's test fault is cleared on ordinary return and unwind.
#[cfg(test)]
pub(crate) struct TestFaultGuard(Rc<TestFaults>);

#[cfg(test)]
impl TestFaultGuard {
    pub(crate) fn fired(&self) -> bool {
        self.0.fired.get()
    }
}

#[cfg(test)]
impl Drop for TestFaultGuard {
    fn drop(&mut self) {
        self.0.armed.set(None);
        self.0.active.set(false);
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum CompileMode {
    Default,
    Strict,
}

#[derive(Default)]
struct SourcePipelines {
    default: ModePipelines,
    strict: ModePipelines,
}

impl SourcePipelines {
    fn mode(&self, mode: CompileMode) -> &ModePipelines {
        match mode {
            CompileMode::Default => &self.default,
            CompileMode::Strict => &self.strict,
        }
    }

    fn mode_mut(&mut self, mode: CompileMode) -> &mut ModePipelines {
        match mode {
            CompileMode::Default => &mut self.default,
            CompileMode::Strict => &mut self.strict,
        }
    }
}

#[derive(Default)]
struct ModePipelines {
    library: Option<Retained<ProtocolObject<dyn MTLLibrary>>>,
    functions: HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

#[derive(Default)]
struct BufferAccounting {
    live: Cell<u64>,
    peak: Cell<u64>,
}

impl BufferAccounting {
    fn add(&self, bytes: u64) -> Result<()> {
        let next = self
            .live
            .get()
            .checked_add(bytes)
            .context("Metal requested buffer byte count overflow")?;
        self.live.set(next);
        self.peak.set(self.peak.get().max(next));
        Ok(())
    }

    fn remove(&self, bytes: u64) -> bool {
        if let Some(next) = self.live.get().checked_sub(bytes) {
            self.live.set(next);
            true
        } else {
            // Drop must remain non-panicking, including during unwinding.
            self.live.set(0);
            false
        }
    }
}

/// On-demand device observation and this Metal identity's requested shared
/// buffer counts. Device current allocation includes resources outside these
/// buffers and is not process RSS or a measured process peak.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryStats {
    pub live_requested_buffer_bytes: u64,
    pub lifetime_peak_requested_buffer_bytes: u64,
    pub device_current_allocated_bytes: u64,
    /// A performance recommendation from Metal, not a hard allocation limit.
    pub device_recommended_max_working_set_bytes: u64,
}

/// A Metal device and command queue. Clone to share the same device identity.
#[derive(Clone)]
pub struct Metal(Rc<MetalInner>);

impl Metal {
    pub(crate) fn shared_event(&self) -> Result<SharedEvent> {
        Ok(SharedEvent(
            self.0
                .device
                .newSharedEvent()
                .context("Metal shared event unavailable")?,
        ))
    }
    pub fn new() -> Result<Self> {
        Self::with_prefill_residency_policy(false, None)
    }

    /// Bounded engines request prefill residency by default when supported.
    pub(crate) fn for_engine(expert_capacity: usize) -> Result<Self> {
        let bounded = expert_capacity < 512;
        Self::with_prefill_residency_policy(
            bounded,
            Some(if bounded {
                "bounded-on"
            } else {
                "full-resident-off"
            }),
        )
    }

    #[cfg(test)]
    fn with_prefill_residency(enabled: bool) -> Result<Self> {
        Self::with_prefill_residency_policy(enabled, None)
    }

    fn with_prefill_residency_policy(
        default_enabled: bool,
        report_default: Option<&str>,
    ) -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().context("no Metal device is available")?;
        let queue = device
            .newCommandQueue()
            .context("Metal could not create a command queue")?;
        let prefill_residency =
            residency::PrefillResidency::for_policy(&device, default_enabled, report_default);
        Ok(Self(Rc::new(MetalInner {
            device,
            queue,
            accounting: BufferAccounting::default(),
            pipelines: RefCell::new(HashMap::new()),
            prefill_residency,
            #[cfg(test)]
            faults: Rc::new(TestFaults::default()),
        })))
    }

    /// Temporarily request residency for live shared allocations during prefill.
    /// The caller must drain its stream before dropping the returned guard.
    pub(crate) fn prefill_residency_guard(&self) -> Result<Option<PrefillResidencyGuard>> {
        self.0
            .prefill_residency
            .as_ref()
            .map(|residency| residency.begin(self))
            .transpose()
    }

    #[cfg(test)]
    pub(crate) fn test_fail_allocation_after(&self, skip: usize) -> Result<TestFaultGuard> {
        self.0.faults.arm(TestFaultKind::Allocation, skip)
    }

    #[cfg(test)]
    pub(crate) fn test_fail_handoff_after(&self, skip: usize) -> Result<TestFaultGuard> {
        self.0.faults.arm(TestFaultKind::HandoffEncoding, skip)
    }
    #[cfg(test)]
    pub(crate) fn test_handoff_encoding_checkpoint(&self) -> Result<()> {
        self.0.faults.check(TestFaultKind::HandoffEncoding)
    }

    #[cfg(test)]
    pub(crate) fn test_fail_completed_command_after(&self, skip: usize) -> Result<TestFaultGuard> {
        self.0.faults.arm(TestFaultKind::CompletedCommand, skip)
    }

    pub(crate) fn device_name(&self) -> String {
        self.0.device.name().to_string()
    }

    /// Query owned requested bytes and current device counters. This does not
    /// sample the device on allocation or retain a device high-water mark.
    pub fn memory_stats(&self) -> Result<MemoryStats> {
        Ok(MemoryStats {
            live_requested_buffer_bytes: self.0.accounting.live.get(),
            lifetime_peak_requested_buffer_bytes: self.0.accounting.peak.get(),
            device_current_allocated_bytes: u64::try_from(self.0.device.currentAllocatedSize())?,
            device_recommended_max_working_set_bytes: self.0.device.recommendedMaxWorkingSetSize(),
        })
    }

    /// Allocate CPU-visible storage. Keep this buffer for repeated dispatches.
    pub fn shared_buffer(&self, byte_len: usize) -> Result<Buffer> {
        if byte_len == 0 {
            bail!("Metal buffer length must be nonzero");
        }
        #[cfg(test)]
        self.0.faults.check(TestFaultKind::Allocation)?;
        let raw = self
            .0
            .device
            .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
            .with_context(|| format!("Metal could not allocate {byte_len} shared bytes"))?;
        let requested_bytes = u64::try_from(byte_len)?;
        let inner = Rc::new(BufferInner {
            raw,
            device: self.0.clone(),
            reserved_by: Cell::new(0),
            uses: Cell::new(0),
            requested_bytes,
            accounted: Cell::new(false),
        });
        self.0.accounting.add(requested_bytes)?;
        inner.accounted.set(true);
        if let Some(residency) = &self.0.prefill_residency {
            residency.track(&inner)?;
        }
        Ok(Buffer(inner))
    }

    /// Compile an MSL source string and one named compute function for reuse.
    pub fn compile_msl(&self, source: &str, function: &str) -> Result<Pipeline> {
        self.compile_with_mode(source, function, CompileMode::Default)
    }

    /// Preserve a kernel's no-fast-math compilation contract.
    pub fn compile_msl_strict(&self, source: &str, function: &str) -> Result<Pipeline> {
        self.compile_with_mode(source, function, CompileMode::Strict)
    }

    fn compile_with_mode(
        &self,
        source: &str,
        function: &str,
        mode: CompileMode,
    ) -> Result<Pipeline> {
        let mut sources = self.0.pipelines.borrow_mut();
        if let Some(raw) = sources
            .get(source)
            .and_then(|entry| entry.mode(mode).functions.get(function))
        {
            return Ok(Pipeline {
                raw: raw.clone(),
                device: self.0.clone(),
                name: function.into(),
            });
        }

        let library = if let Some(library) = sources
            .get(source)
            .and_then(|entry| entry.mode(mode).library.as_ref())
        {
            library.clone()
        } else {
            let options = match mode {
                CompileMode::Default => None,
                CompileMode::Strict => {
                    let options = MTLCompileOptions::new();
                    // Match the reference extension's -fno-fast-math contract.
                    #[allow(deprecated)]
                    options.setFastMathEnabled(false);
                    Some(options)
                }
            };
            let library = self
                .0
                .device
                .newLibraryWithSource_options_error(&NSString::from_str(source), options.as_deref())
                .map_err(|err| anyhow!("Metal library compilation failed: {err}"))?;
            sources
                .entry(source.to_owned())
                .or_default()
                .mode_mut(mode)
                .library = Some(library.clone());
            library
        };
        let name = NSString::from_str(function);
        let kernel = library
            .newFunctionWithName(&name)
            .with_context(|| format!("MSL function {function:?} was not found"))?;
        let raw = self
            .0
            .device
            .newComputePipelineStateWithFunction_error(&kernel)
            .map_err(|err| anyhow!("Metal pipeline compilation failed for {function:?}: {err}"))?;
        sources
            .get_mut(source)
            .expect("a compiled Metal library has a cache entry")
            .mode_mut(mode)
            .functions
            .insert(function.to_owned(), raw.clone());
        Ok(Pipeline {
            raw,
            device: self.0.clone(),
            name: function.into(),
        })
    }

    pub fn batch(&self) -> Result<Batch> {
        self.batch_for_owner(next_owner_id()?)
    }

    fn batch_for_owner(&self, owner: usize) -> Result<Batch> {
        let command = self
            .0
            .queue
            .commandBuffer()
            .context("Metal could not create a command buffer")?;
        Ok(Batch {
            command,
            device: self.0.clone(),
            buffers: Vec::new(),
            buffer_ids: HashSet::new(),
            pipelines: Vec::new(),
            encoder: None,
            owner,
            has_work: false,
        })
    }

    /// Start one ordered command owner. Its batches may share device buffers.
    pub fn stream(&self) -> Result<Stream> {
        Ok(Stream {
            metal: self.clone(),
            owner: next_owner_id()?,
            active: None,
            pending: Vec::new(),
            submitted: 0,
            retired: 0,
            publication_gates: Vec::new(),
        })
    }
}

struct BufferInner {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    device: Rc<MetalInner>,
    // Zero means available to the host; otherwise the command owner ID.
    reserved_by: Cell<usize>,
    // One use per distinct batch that bound or declared this allocation.
    uses: Cell<usize>,
    requested_bytes: u64,
    accounted: Cell<bool>,
}

impl Drop for BufferInner {
    fn drop(&mut self) {
        if let Some(residency) = &self.device.prefill_residency {
            residency.untrack(self);
        }
        if self.accounted.get() {
            let _ = self.device.accounting.remove(self.requested_bytes);
        }
    }
}

/// Shared Metal storage. CPU byte access is checked against encoded GPU work.
#[derive(Clone)]
pub struct Buffer(Rc<BufferInner>);

#[derive(Clone)]
pub(crate) struct SharedEvent(Retained<ProtocolObject<dyn MTLSharedEvent>>);

impl SharedEvent {
    pub(crate) fn point(&self, value: u64) -> Result<EventPoint> {
        ensure!(
            value > self.0.signaledValue(),
            "event epoch must exceed completed values"
        );
        Ok(EventPoint {
            event: self.clone(),
            value,
        })
    }
}

/// One monotonic event frontier. Signaling orders memory, but does not assert
/// that publication succeeded; consumers validate a separate publication flag.
#[derive(Clone)]
pub(crate) struct EventPoint {
    event: SharedEvent,
    value: u64,
}
impl EventPoint {
    pub(crate) fn reached(&self) -> bool {
        self.event.0.signaledValue() >= self.value
    }
    pub(crate) fn release(&self) {
        if !self.reached() {
            self.event.0.setSignaledValue(self.value);
        }
    }
}

/// Owner-thread access to an allocation whose future readers are gated.
pub(crate) struct PublicationLease {
    buffer: Buffer,
    gate: EventPoint,
}
impl PublicationLease {
    /// # Safety
    /// Earlier readers must have completed. Every future access to this exact
    /// allocation must wait for `gate`; the caller retains that exclusion and
    /// both the allocation and event until submission completion.
    pub(crate) unsafe fn new(buffer: &Buffer, gate: &EventPoint) -> Result<Self> {
        ensure!(!gate.reached(), "publication gate already released");
        Ok(Self {
            buffer: buffer.clone(),
            gate: gate.clone(),
        })
    }
    pub(crate) fn write(&self, bytes: &[u8]) -> Result<()> {
        ensure!(!self.gate.reached(), "publication lease was consumed");
        ensure!(
            bytes.len() == self.buffer.byte_len(),
            "publication span differs"
        );
        // SAFETY: The constructor establishes exclusion until the gate opens.
        unsafe {
            ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.buffer.0.raw.contents().as_ptr().cast(),
                bytes.len(),
            );
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_rows(
        &self,
        offset: usize,
        bytes: &[u8],
        rows: usize,
        source_row: usize,
        source_offset: usize,
        width: usize,
    ) -> Result<()> {
        ensure!(!self.gate.reached(), "publication lease was consumed");
        // SAFETY: The lease excludes access throughout this checked copy.
        unsafe {
            self.buffer.write_strided_rows_disjoint(
                offset,
                bytes,
                rows,
                source_row,
                source_offset,
                width,
            )
        }
    }
}

impl Buffer {
    /// Return the GPU address of a nonempty checked allocation span.
    /// Address lookup does not read or publish bytes. The allocation must be
    /// declared with `Batch::use_indirect_reads` before a shader reads it through
    /// this address, and any pending publication must retain its event gate.
    pub fn gpu_address(&self, offset: usize, byte_len: usize) -> Result<u64> {
        checked_gpu_address(self.0.raw.gpuAddress(), self.byte_len(), offset, byte_len)
    }

    /// # Safety
    /// The event follows the final GPU writer and no later command or host
    /// operation can write this allocation before the read completes.
    pub(crate) unsafe fn read_after_event(
        &self,
        ready: &EventPoint,
        bytes: &mut [u8],
    ) -> Result<()> {
        ensure!(
            ready.reached(),
            "buffer producer has not signaled readiness"
        );
        ensure!(bytes.len() == self.byte_len(), "event read extent differs");
        // SAFETY: The caller proves completion of all writes to this allocation.
        unsafe {
            ptr::copy_nonoverlapping(
                self.0.raw.contents().as_ptr().cast(),
                bytes.as_mut_ptr(),
                bytes.len(),
            );
        }
        Ok(())
    }
    pub(crate) fn shares_allocation(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
    pub fn byte_len(&self) -> usize {
        self.0.raw.length()
    }

    pub fn write_bytes(&self, offset: usize, bytes: &[u8]) -> Result<()> {
        self.check_host_access(offset, bytes.len())?;
        // SAFETY: Metal shared storage is valid for `byte_len` bytes. Bounds and
        // in-flight access are checked above; this wrapper does not expose the
        // raw pointer or allow cross-thread host access.
        unsafe {
            ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.0.raw.contents().as_ptr().cast::<u8>().add(offset),
                bytes.len(),
            );
        }
        Ok(())
    }

    /// Copy a checked row span directly into one contiguous stream slot.
    /// The whole allocation must be idle before any row is written.
    pub(crate) fn write_strided_rows(
        &self,
        destination_offset: usize,
        source: &[u8],
        rows: usize,
        source_row_bytes: usize,
        source_offset: usize,
        stream_row_bytes: usize,
    ) -> Result<()> {
        self.check_host_access(0, self.byte_len())?;
        // SAFETY: Ordinary host access excludes every encoded/submitted use.
        unsafe {
            self.write_strided_rows_disjoint(
                destination_offset,
                source,
                rows,
                source_row_bytes,
                source_offset,
                stream_row_bytes,
            )
        }
    }

    /// Copy into a checked span while other spans may have GPU readers.
    ///
    /// # Safety
    /// No encoded or submitted GPU access, or host alias, may touch the exact
    /// destination span during this call. The caller must retain that exclusion
    /// even for encoded commands that have not yet been submitted.
    pub(crate) unsafe fn write_strided_rows_disjoint(
        &self,
        destination_offset: usize,
        source: &[u8],
        rows: usize,
        source_row_bytes: usize,
        source_offset: usize,
        stream_row_bytes: usize,
    ) -> Result<()> {
        let length = rows
            .checked_mul(stream_row_bytes)
            .context("strided destination byte count overflow")?;
        ensure!(
            destination_offset
                .checked_add(length)
                .is_some_and(|end| end <= self.byte_len()),
            "strided destination exceeds Metal buffer"
        );
        // SAFETY: The caller excludes access to this exact checked range,
        // and the temporary slice cannot escape this call.
        // The pure copier validates source geometry before writing any row.
        let destination = unsafe {
            std::slice::from_raw_parts_mut(
                self.0
                    .raw
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(destination_offset),
                length,
            )
        };
        crate::host_copy::copy_strided_rows(
            destination,
            source,
            rows,
            source_row_bytes,
            source_offset,
            stream_row_bytes,
        )
    }

    pub fn read_bytes(&self, offset: usize, bytes: &mut [u8]) -> Result<()> {
        self.check_host_access(offset, bytes.len())?;
        // SAFETY: Same storage and access checks as `write_bytes`.
        unsafe {
            ptr::copy_nonoverlapping(
                self.0.raw.contents().as_ptr().cast::<u8>().add(offset),
                bytes.as_mut_ptr(),
                bytes.len(),
            );
        }
        Ok(())
    }

    fn check_host_access(&self, offset: usize, len: usize) -> Result<()> {
        if self.0.reserved_by.get() != 0 {
            bail!("Metal buffer is reserved by a command batch");
        }
        if offset > self.byte_len() || len > self.byte_len() - offset {
            bail!("Metal buffer byte range exceeds allocation");
        }
        Ok(())
    }
}

fn checked_gpu_address(
    base: u64,
    allocation_bytes: usize,
    offset: usize,
    byte_len: usize,
) -> Result<u64> {
    ensure!(byte_len != 0, "Metal GPU address span must be nonempty");
    ensure!(
        offset < allocation_bytes && byte_len <= allocation_bytes - offset,
        "Metal GPU address span exceeds allocation"
    );
    ensure!(base != 0, "Metal buffer GPU address is unavailable");
    let address = base
        .checked_add(u64::try_from(offset)?)
        .context("Metal GPU address offset overflow")?;
    address
        .checked_add(u64::try_from(byte_len - 1)?)
        .context("Metal GPU address span overflow")?;
    Ok(address)
}

/// A compiled compute function bound to its device.
#[derive(Clone)]
pub struct Pipeline {
    raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    device: Rc<MetalInner>,
    name: Rc<str>,
}

impl Pipeline {
    pub fn max_threads_per_threadgroup(&self) -> usize {
        self.raw.maxTotalThreadsPerThreadgroup()
    }
}

/// A buffer binding. Slice order maps directly to MSL buffer indices 0, 1, ...
#[derive(Clone, Copy)]
pub struct Binding<'a> {
    pub buffer: &'a Buffer,
    pub offset: usize,
}

/// A reusable encoding scope for sequential compute dispatches.
pub struct Batch {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    device: Rc<MetalInner>,
    buffers: Vec<Buffer>,
    // Only an index: `buffers` owns each distinct batch use through retirement.
    buffer_ids: HashSet<*const BufferInner>,
    pipelines: Vec<Pipeline>,
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    owner: usize,
    has_work: bool,
}

impl Batch {
    fn validate_buffer(&self, buffer: &Buffer) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.device, &buffer.0.device),
            "Metal buffer belongs to another device context"
        );
        let owner = buffer.0.reserved_by.get();
        ensure!(
            owner == 0 || owner == self.owner,
            "Metal buffer is reserved by another command batch"
        );
        Ok(())
    }

    fn retain_buffer(&mut self, buffer: &Buffer) -> Result<()> {
        let id = Rc::as_ptr(&buffer.0);
        if !self.buffer_ids.contains(&id) {
            let uses = buffer
                .0
                .uses
                .get()
                .checked_add(1)
                .context("Metal buffer use count overflow")?;
            self.buffers.try_reserve(1)?;
            self.buffer_ids.try_reserve(1)?;
            buffer.0.reserved_by.set(self.owner);
            buffer.0.uses.set(uses);
            self.buffers.push(buffer.clone());
            self.buffer_ids.insert(id);
        }
        Ok(())
    }

    /// Declare allocations read through GPU addresses on the current encoder.
    /// These allocations share direct bindings' reservations and are retained
    /// until batch cancellation or submitted command completion. All fallible
    /// retention finishes before resources are declared to Metal. A retention
    /// failure keeps prior reservations until the batch is canceled or retires.
    ///
    /// Call before each dispatch that uses the addresses. Event waits or signals
    /// can end an encoder, so declarations must be repeated on the next encoder.
    /// This declaration does not publish bytes or replace a publication wait.
    pub fn use_indirect_reads(&mut self, buffers: &[&Buffer]) -> Result<()> {
        for buffer in buffers {
            self.validate_buffer(buffer)?;
        }
        if buffers.is_empty() {
            return Ok(());
        }
        if self.encoder.is_none() {
            self.encoder = Some(
                self.command
                    .computeCommandEncoder()
                    .context("Metal could not create a compute encoder")?,
            );
        }
        for buffer in buffers {
            self.retain_buffer(buffer)?;
        }
        let encoder = self.encoder.as_ref().expect("encoder created above");
        for buffer in buffers {
            encoder.useResource_usage(
                ProtocolObject::<dyn MTLResource>::from_ref(&*buffer.0.raw),
                MTLResourceUsage::Read,
            );
        }
        Ok(())
    }

    pub(crate) fn signal_event(&mut self, point: &EventPoint) {
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }
        self.command.encodeSignalEvent_value(
            ProtocolObject::<dyn MTLEvent>::from_ref(&*point.event.0),
            point.value,
        );
        self.has_work = true;
    }
    /// Encode one one-dimensional grid. Bindings must match the MSL signature.
    ///
    /// # Safety
    ///
    /// The caller must ensure that every thread's MSL reads and writes stay
    /// within the bound buffers, that offsets and element layouts match the
    /// function's arguments, that binding indices are supported by the
    /// function and device, and that concurrent threads do not race. Every
    /// allocation read through a GPU address must have been declared to this
    /// encoder with `use_indirect_reads` and obey any publication gate.
    pub unsafe fn dispatch_1d(
        &mut self,
        pipeline: &Pipeline,
        bindings: &[Binding<'_>],
        grid_threads: usize,
        group_threads: usize,
    ) -> Result<()> {
        // SAFETY: The caller supplies the same shader/buffer contract; this
        // entry adds no inline arguments.
        unsafe {
            self.dispatch_1d_with_inline(pipeline, bindings, &[], grid_threads, group_threads)
        }
    }

    /// Encode a grid with copied constants after the regular buffer bindings.
    /// Metal captures each argument at encoding time. The caller may reuse its
    /// CPU bytes immediately without changing already queued dispatches.
    ///
    /// # Safety
    /// The caller must satisfy `dispatch_1d`'s shader, bounds and race contract.
    /// Inline argument layouts must match the shader bindings starting at
    /// `bindings.len()`, and the shader must not write those constants.
    pub unsafe fn dispatch_1d_with_inline(
        &mut self,
        pipeline: &Pipeline,
        bindings: &[Binding<'_>],
        inline_args: &[&[u8]],
        grid_threads: usize,
        group_threads: usize,
    ) -> Result<()> {
        if bindings.len().saturating_add(inline_args.len()) > 31 {
            bail!("Metal dispatch exceeds supported buffer binding count");
        }
        if inline_args
            .iter()
            .any(|bytes| bytes.is_empty() || bytes.len() >= 4096)
        {
            bail!("Metal inline arguments must contain 1 through 4095 bytes");
        }
        if !Rc::ptr_eq(&self.device, &pipeline.device) {
            bail!("Metal pipeline belongs to another device context");
        }
        if grid_threads == 0 || group_threads == 0 {
            bail!("Metal dispatch dimensions must be nonzero");
        }
        if group_threads > pipeline.max_threads_per_threadgroup() {
            bail!("Metal threadgroup exceeds pipeline limit");
        }
        for (index, binding) in bindings.iter().enumerate() {
            self.validate_buffer(binding.buffer)?;
            if binding.offset >= binding.buffer.byte_len() {
                bail!(
                    "Metal kernel {} binding {index}: offset {} exceeds allocation {}",
                    pipeline.name,
                    binding.offset,
                    binding.buffer.byte_len()
                );
            }
        }

        if self.encoder.is_none() {
            self.encoder = Some(
                self.command
                    .computeCommandEncoder()
                    .context("Metal could not create a compute encoder")?,
            );
        }
        self.encoder
            .as_ref()
            .expect("encoder created above")
            .setComputePipelineState(&pipeline.raw);
        for (index, binding) in bindings.iter().enumerate() {
            self.retain_buffer(binding.buffer)?;
            let encoder = self.encoder.as_ref().expect("encoder created above");
            // SAFETY: This batch retains each buffer until GPU completion,
            // validates offsets, and excludes host access while reserved.
            // The caller must supply the MSL-expected element type and bounds.
            unsafe {
                encoder.setBuffer_offset_atIndex(
                    Some(&binding.buffer.0.raw),
                    binding.offset,
                    index,
                );
            }
        }
        let encoder = self.encoder.as_ref().expect("encoder created above");
        for (index, bytes) in inline_args.iter().enumerate() {
            // SAFETY: Nonempty slices are readable for the call. Metal copies
            // their contents; no host pointer is retained by the GPU command.
            // Binding count was checked and shader layout belongs to caller.
            unsafe {
                encoder.setBytes_length_atIndex(
                    ptr::NonNull::new(bytes.as_ptr().cast_mut().cast())
                        .expect("nonempty inline argument pointer"),
                    bytes.len(),
                    bindings.len() + index,
                );
            }
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: grid_threads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: group_threads,
                height: 1,
                depth: 1,
            },
        );
        self.has_work = true;
        self.pipelines.push(pipeline.clone());
        Ok(())
    }

    /// Commit the encoded work. The returned object owns its buffers until wait.
    pub fn submit(mut self) -> InFlight {
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }
        self.command.commit();
        InFlight {
            command: self.command.clone(),
            buffers: std::mem::take(&mut self.buffers),
            _pipelines: std::mem::take(&mut self.pipelines),
            owner: self.owner,
            finished: false,
            #[cfg(test)]
            faults: self.device.faults.clone(),
        }
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }
        release_buffers(&self.buffers, self.owner);
    }
}

/// Submitted GPU work. Dropping it waits before releasing referenced buffers.
#[must_use = "wait for GPU completion or drop to block until completion"]
pub struct InFlight {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    buffers: Vec<Buffer>,
    _pipelines: Vec<Pipeline>,
    owner: usize,
    finished: bool,
    #[cfg(test)]
    faults: Rc<TestFaults>,
}

impl InFlight {
    pub fn wait(mut self) -> Result<()> {
        self.finish()
    }

    fn finish(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        self.command.waitUntilCompleted();
        self.finished = true;
        release_buffers(&self.buffers, self.owner);
        if self.command.status() != MTLCommandBufferStatus::Completed {
            if let Some(err) = self.command.error() {
                bail!("Metal command failed: {err}");
            }
            bail!(
                "Metal command ended with status {:?}",
                self.command.status()
            );
        }
        // Test errors follow successful real completion and reservation release.
        #[cfg(test)]
        self.faults.check(TestFaultKind::CompletedCommand)?;
        Ok(())
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

fn release_buffers(buffers: &[Buffer], owner: usize) {
    for buffer in buffers {
        if buffer.0.reserved_by.get() == owner {
            let remaining = buffer
                .0
                .uses
                .get()
                .checked_sub(1)
                .expect("Metal command buffer use retired twice");
            buffer.0.uses.set(remaining);
            if remaining == 0 {
                buffer.0.reserved_by.set(0);
            }
        }
    }
}

/// Ordered batches under one buffer owner. Submitted commands drain on drop.
pub struct Stream {
    metal: Metal,
    owner: usize,
    active: Option<Batch>,
    pending: Vec<InFlight>,
    submitted: u64,
    retired: u64,
    publication_gates: Vec<EventPoint>,
}

/// Evidence of a submission on one ordered command owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Completion {
    owner: usize,
    sequence: u64,
}

fn validate_completion(owner: usize, submitted: u64, marker: Completion) -> Result<()> {
    ensure!(
        marker.owner == owner,
        "Metal completion belongs to another stream"
    );
    ensure!(
        marker.sequence > 0 && marker.sequence <= submitted,
        "Metal completion was not submitted"
    );
    Ok(())
}

fn next_submission(submitted: u64) -> Result<u64> {
    submitted
        .checked_add(1)
        .context("Metal submission sequence overflow")
}

fn retire_results(retired: &mut u64, results: impl Iterator<Item = Result<()>>) -> Result<()> {
    let mut first_error = None;
    for result in results {
        if let Err(err) = result
            && first_error.is_none()
        {
            first_error = Some(err);
        }
        *retired += 1;
    }
    first_error.map_or(Ok(()), Err)
}

impl Stream {
    pub(crate) fn register_publication(&mut self, point: &EventPoint) -> Result<()> {
        ensure!(
            !point.reached(),
            "publication gate already released before registration"
        );
        if !self
            .publication_gates
            .iter()
            .any(|held| held.value == point.value && std::ptr::eq(&*held.event.0, &*point.event.0))
        {
            self.publication_gates.push(point.clone());
        }
        Ok(())
    }
    /// Register release ownership before encoding a GPU publication wait.
    pub(crate) fn wait_for_publication(&mut self, point: &EventPoint) -> Result<()> {
        self.register_publication(point)?;
        let batch = self.batch()?;
        if let Some(encoder) = batch.encoder.take() {
            encoder.endEncoding();
        }
        batch.command.encodeWaitForEvent_value(
            ProtocolObject::<dyn MTLEvent>::from_ref(&*point.event.0),
            point.value,
        );
        batch.has_work = true;
        Ok(())
    }
    pub fn batch(&mut self) -> Result<&mut Batch> {
        if self.active.is_none() {
            self.active = Some(self.metal.batch_for_owner(self.owner)?);
        }
        Ok(self.active.as_mut().expect("active batch created above"))
    }

    /// Commit the active batch without waiting; the queue preserves order.
    pub fn flush(&mut self) -> Result<()> {
        if self.active.as_ref().is_some_and(|batch| batch.has_work) {
            self.flush_marked()?;
        } else {
            self.active.take();
        }
        Ok(())
    }

    /// Submit actual work and capture its ordered completion frontier.
    pub(crate) fn flush_marked(&mut self) -> Result<Completion> {
        ensure!(
            self.active.as_ref().is_some_and(|batch| batch.has_work),
            "Metal marked flush requires encoded work"
        );
        let sequence = next_submission(self.submitted)?;
        self.pending
            .push(self.active.take().expect("checked active batch").submit());
        self.submitted = sequence;
        Ok(Completion {
            owner: self.owner,
            sequence,
        })
    }

    /// Retire only through this marker, without submitting active work.
    pub(crate) fn wait_through(&mut self, marker: Completion) -> Result<()> {
        if self.publication_gates.iter().any(|gate| !gate.reached()) {
            self.abort_and_wait()?;
            bail!("cannot retire a stream with unresolved expert publication");
        }
        validate_completion(self.owner, self.submitted, marker)?;
        self.drain_through(marker.sequence)
    }

    fn drain_through(&mut self, sequence: u64) -> Result<()> {
        let count = usize::try_from(sequence.saturating_sub(self.retired))?;
        ensure!(
            count <= self.pending.len(),
            "Metal completion queue is inconsistent"
        );
        retire_results(
            &mut self.retired,
            self.pending.drain(..count).map(InFlight::wait),
        )
    }

    /// Cancel unsubmitted work and complete every submitted command on error.
    pub(crate) fn abort_and_wait(&mut self) -> Result<()> {
        for gate in &self.publication_gates {
            gate.release();
        }
        self.active.take();
        let result = self.drain_through(self.submitted);
        self.publication_gates.clear();
        result
    }

    /// Complete every submitted command, including after an earlier failure.
    pub fn wait(&mut self) -> Result<()> {
        if self.publication_gates.iter().any(|gate| !gate.reached()) {
            self.abort_and_wait()?;
            bail!("cannot complete a stream with unresolved expert publication");
        }
        let flush = self.flush();
        if flush.is_err() {
            self.active.take();
        }
        let drain = self.drain_through(self.submitted);
        self.publication_gates.clear();
        flush.and(drain)
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // Unsubmitted work is canceled before submitted commands are drained.
        let _ = self.abort_and_wait();
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;
    use std::rc::Rc;

    use super::{Binding, BufferAccounting, Metal};

    const CACHE_SOURCE: &str = "#include <metal_stdlib>\nusing namespace metal;\nkernel void first(device uint* out [[buffer(0)]]) { out[0] = 1; }\nkernel void second(device uint* out [[buffer(0)]]) { out[0] = 2; }";

    #[test]
    fn gpu_address_span_validation_is_cpu_only() {
        use super::checked_gpu_address;

        assert_eq!(checked_gpu_address(100, 8, 4, 4).unwrap(), 104);
        assert_eq!(checked_gpu_address(u64::MAX, 1, 0, 1).unwrap(), u64::MAX);
        assert!(checked_gpu_address(0, 8, 0, 4).is_err());
        assert!(checked_gpu_address(100, 8, 0, 0).is_err());
        assert!(checked_gpu_address(100, 8, 8, 1).is_err());
        assert!(checked_gpu_address(100, 8, 4, 5).is_err());
        assert!(checked_gpu_address(100, 0, 0, 1).is_err());
        assert!(checked_gpu_address(100, 8, usize::MAX, 1).is_err());
        assert!(checked_gpu_address(u64::MAX, 2, 1, 1).is_err());
        assert!(checked_gpu_address(u64::MAX, 2, 0, 2).is_err());
    }

    #[test]
    fn indirect_reads_retain_unbound_sources_and_share_direct_reservations() {
        let metal = Metal::new().unwrap();
        let source = "#include <metal_stdlib>\nusing namespace metal;\nkernel void indirect_read(device const ulong* addresses [[buffer(0)]], device uint* output [[buffer(1)]], device const uint* direct [[buffer(2)]]) { device const uint* input = (device const uint*)addresses[0]; output[0] = input[0] + direct[0]; }";
        let pipeline = metal.compile_msl_strict(source, "indirect_read").unwrap();
        let input = metal.shared_buffer(8).unwrap();
        let alias = input.clone();
        let direct = metal.shared_buffer(4).unwrap();
        let addresses = metal.shared_buffer(8).unwrap();
        let output = metal.shared_buffer(4).unwrap();
        input.write_bytes(4, &17u32.to_le_bytes()).unwrap();
        direct.write_bytes(0, &3u32.to_le_bytes()).unwrap();
        let address = input.gpu_address(4, 4).unwrap();
        addresses.write_bytes(0, &address.to_le_bytes()).unwrap();
        let mut batch = metal.batch().unwrap();
        batch
            .use_indirect_reads(&[&input, &alias, &direct, &direct])
            .unwrap();
        assert_eq!(batch.buffers.len(), 2);
        assert_eq!(input.0.uses.get(), 1);
        assert_eq!(direct.0.uses.get(), 1);
        assert_eq!(input.gpu_address(4, 4).unwrap(), address);
        assert!(input.write_bytes(4, &0u32.to_le_bytes()).is_err());
        // SAFETY: One thread reads the checked uint span through a declared
        // source address and a directly bound uint, then writes one output uint.
        unsafe {
            batch
                .dispatch_1d(
                    &pipeline,
                    &[
                        Binding {
                            buffer: &addresses,
                            offset: 0,
                        },
                        Binding {
                            buffer: &output,
                            offset: 0,
                        },
                        Binding {
                            buffer: &direct,
                            offset: 0,
                        },
                    ],
                    1,
                    1,
                )
                .unwrap();
        }
        assert_eq!(batch.buffers.len(), 4);
        assert_eq!(batch.buffer_ids.len(), 4);
        assert_eq!(direct.0.uses.get(), 1);
        let flight = batch.submit();
        drop(input);
        drop(alias);
        assert_eq!(
            metal.memory_stats().unwrap().live_requested_buffer_bytes,
            24
        );
        flight.wait().unwrap();
        assert_eq!(
            metal.memory_stats().unwrap().live_requested_buffer_bytes,
            16
        );
        assert_eq!(direct.0.uses.get(), 0);
        let mut bytes = [0; 4];
        output.read_bytes(0, &mut bytes).unwrap();
        assert_eq!(u32::from_le_bytes(bytes), 20);
    }

    #[test]
    fn indirect_read_validation_and_partial_retention_preserve_batch_ownership() {
        let metal = Metal::new().unwrap();
        let foreign = Metal::new().unwrap().shared_buffer(4).unwrap();
        let first = metal.shared_buffer(4).unwrap();
        let second = metal.shared_buffer(4).unwrap();
        let mut batch = metal.batch().unwrap();
        batch.use_indirect_reads(&[]).unwrap();
        assert!(batch.encoder.is_none());
        assert!(batch.use_indirect_reads(&[&first, &foreign]).is_err());
        assert!(batch.buffers.is_empty());
        assert!(batch.encoder.is_none());

        let mut competing = metal.batch().unwrap();
        competing.use_indirect_reads(&[&second]).unwrap();
        assert!(batch.use_indirect_reads(&[&first, &second]).is_err());
        assert!(batch.buffers.is_empty());
        first.read_bytes(0, &mut [0; 4]).unwrap();
        drop(competing);

        second.0.uses.set(usize::MAX);
        let error = batch.use_indirect_reads(&[&first, &second]).unwrap_err();
        assert!(error.to_string().contains("use count overflow"));
        assert_eq!(batch.buffers.len(), 1);
        assert_eq!(batch.buffer_ids.len(), 1);
        assert_eq!(first.0.uses.get(), 1);
        assert_eq!(second.0.reserved_by.get(), 0);
        second.0.uses.set(0);
        assert!(first.write_bytes(0, &1u32.to_le_bytes()).is_err());
        batch.use_indirect_reads(&[&first, &second]).unwrap();
        assert_eq!(first.0.uses.get(), 1);
        assert_eq!(second.0.uses.get(), 1);
        drop(batch);
        assert_eq!(first.0.uses.get(), 0);
        assert_eq!(second.0.uses.get(), 0);
        first.write_bytes(0, &1u32.to_le_bytes()).unwrap();
        second.write_bytes(0, &2u32.to_le_bytes()).unwrap();
    }

    #[test]
    fn test_faults_are_one_shot_isolated_and_cleared_on_unwind() {
        use super::{TestFaultKind, TestFaults};
        use std::panic::{AssertUnwindSafe, catch_unwind};

        let faults = Rc::new(TestFaults::default());
        let other = Rc::new(TestFaults::default());
        let guard = faults.arm(TestFaultKind::Allocation, 1).unwrap();
        assert!(faults.arm(TestFaultKind::CompletedCommand, 0).is_err());
        faults.check(TestFaultKind::CompletedCommand).unwrap();
        other.check(TestFaultKind::Allocation).unwrap();
        faults.check(TestFaultKind::Allocation).unwrap();
        assert!(!guard.fired());
        assert_eq!(
            faults
                .check(TestFaultKind::Allocation)
                .unwrap_err()
                .to_string(),
            "injected Metal allocation rejection"
        );
        assert!(guard.fired());
        faults.check(TestFaultKind::Allocation).unwrap();
        drop(guard);
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let _guard = faults.arm(TestFaultKind::CompletedCommand, 0).unwrap();
                panic!("unwind clears an unconsumed test fault");
            }))
            .is_err()
        );
        faults.check(TestFaultKind::CompletedCommand).unwrap();
        let guard = faults.arm(TestFaultKind::CompletedCommand, 0).unwrap();
        assert_eq!(
            faults
                .check(TestFaultKind::CompletedCommand)
                .unwrap_err()
                .to_string(),
            "injected completed Metal command error"
        );
        assert!(guard.fired());
        faults.check(TestFaultKind::CompletedCommand).unwrap();
    }

    #[test]
    fn completion_validation_and_failed_retirement_are_cpu_only() {
        use super::{Completion, next_submission, retire_results, validate_completion};
        let marker = Completion {
            owner: 7,
            sequence: 2,
        };
        assert!(validate_completion(7, 2, marker).is_ok());
        assert!(validate_completion(8, 2, marker).is_err());
        assert!(validate_completion(7, 1, marker).is_err());
        assert!(
            validate_completion(
                7,
                2,
                Completion {
                    owner: 7,
                    sequence: 0
                }
            )
            .is_err()
        );
        assert_eq!(next_submission(0).unwrap(), 1);
        assert!(next_submission(u64::MAX).is_err());
        let mut retired = 3;
        let mut visited = Vec::new();
        let error = retire_results(
            &mut retired,
            (0..3).map(|i| {
                visited.push(i);
                anyhow::bail!("failure {i}")
            }),
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "failure 0");
        assert_eq!(visited, [0, 1, 2]);
        assert_eq!(retired, 6);
        retire_results(&mut retired, std::iter::empty()).unwrap();
        assert_eq!(retired, 6);
    }

    #[test]
    fn marked_retirement_keeps_gated_reader_and_disjoint_copy_safe() -> anyhow::Result<()> {
        use objc2::rc::Retained;
        use objc2::runtime::ProtocolObject;
        use objc2_metal::{
            MTLCommandBuffer, MTLCommandBufferStatus, MTLDevice, MTLEvent, MTLSharedEvent,
        };

        struct ReleaseEvent(Retained<ProtocolObject<dyn MTLSharedEvent>>);
        impl Drop for ReleaseEvent {
            fn drop(&mut self) {
                self.0.setSignaledValue(1);
            }
        }

        let metal = Metal::new()?;
        let pipeline = metal.compile_msl_strict(
            "#include <metal_stdlib>\nusing namespace metal;\nkernel void read_slot(device const uint* input [[buffer(0)]], device uint* out [[buffer(1)]]) { out[0] = input[0]; }",
            "read_slot",
        )?;
        let pool = metal.shared_buffer(8)?;
        pool.write_bytes(0, &17u32.to_le_bytes())?;
        pool.write_bytes(4, &23u32.to_le_bytes())?;
        let first = metal.shared_buffer(4)?;
        let second = metal.shared_buffer(4)?;
        let canceled = metal.shared_buffer(4)?;
        let mut stream = metal.stream()?;
        assert!(stream.flush_marked().is_err());
        unsafe {
            stream.batch()?.dispatch_1d(
                &pipeline,
                &[
                    Binding {
                        buffer: &pool,
                        offset: 0,
                    },
                    Binding {
                        buffer: &first,
                        offset: 0,
                    },
                ],
                1,
                1,
            )?;
        }
        let before = stream.flush_marked()?;

        // This guard is declared after the stream, so unwinding signals before
        // Stream::drop can wait for the event-blocked command.
        let release = ReleaseEvent(metal.0.device.newSharedEvent().expect("shared event"));
        stream
            .batch()?
            .command
            .encodeWaitForEvent_value(ProtocolObject::<dyn MTLEvent>::from_ref(&*release.0), 1);
        // Both submissions read only slot zero. The copied slot one is disjoint.
        unsafe {
            stream.batch()?.dispatch_1d(
                &pipeline,
                &[
                    Binding {
                        buffer: &pool,
                        offset: 0,
                    },
                    Binding {
                        buffer: &second,
                        offset: 0,
                    },
                ],
                1,
                1,
            )?;
        }
        let current = stream.flush_marked()?;
        assert_ne!(
            stream.pending.last().unwrap().command.status(),
            MTLCommandBufferStatus::Completed
        );
        assert!(
            pool.write_strided_rows(4, &31u32.to_le_bytes(), 1, 4, 0, 4)
                .is_err()
        );
        unsafe {
            pool.write_strided_rows_disjoint(4, &31u32.to_le_bytes(), 1, 4, 0, 4)?;
        }
        stream.wait_through(before)?;
        stream.wait_through(before)?;
        assert_eq!(stream.pending.len(), 1);
        assert!(pool.write_bytes(0, &[0; 4]).is_err());
        assert!(second.read_bytes(0, &mut [0; 4]).is_err());
        let mut bytes = [0; 4];
        first.read_bytes(0, &mut bytes)?;
        assert_eq!(u32::from_le_bytes(bytes), 17);
        let mut foreign = metal.stream()?;
        assert!(foreign.wait_through(current).is_err());
        // Canceling active work must not submit it while draining both readers.
        unsafe {
            stream.batch()?.dispatch_1d(
                &pipeline,
                &[
                    Binding {
                        buffer: &pool,
                        offset: 4,
                    },
                    Binding {
                        buffer: &canceled,
                        offset: 0,
                    },
                ],
                1,
                1,
            )?;
        }
        let active_command = stream.active.as_ref().unwrap().command.clone();
        drop(release);
        stream.abort_and_wait()?;
        assert_eq!(active_command.status(), MTLCommandBufferStatus::NotEnqueued);
        stream.wait_through(current)?;
        second.read_bytes(0, &mut bytes)?;
        assert_eq!(u32::from_le_bytes(bytes), 17);
        pool.read_bytes(4, &mut bytes)?;
        assert_eq!(u32::from_le_bytes(bytes), 31);
        pool.write_bytes(0, &[0; 4])?;
        canceled.write_bytes(0, &[0; 4])?;
        assert_eq!(pool.0.uses.get(), 0);
        assert_eq!(metal.memory_stats()?.live_requested_buffer_bytes, 20);
        Ok(())
    }

    #[test]
    fn inline_arguments_capture_each_dispatch_before_cpu_reuse() {
        let metal = Metal::new().unwrap();
        let pipeline = metal.compile_msl_strict(
            "#include <metal_stdlib>\nusing namespace metal;\nkernel void capture(device uint* out [[buffer(0)]], constant uint2& args [[buffer(1)]]) { out[args.y] = args.x; }",
            "capture",
        ).unwrap();
        let output = metal.shared_buffer(8).unwrap();
        let bindings = [Binding {
            buffer: &output,
            offset: 0,
        }];
        let mut batch = metal.batch().unwrap();
        let mut args = [0u8; 8];
        args[..4].copy_from_slice(&7u32.to_le_bytes());
        // SAFETY: The two captured indices name distinct output words.
        unsafe {
            batch
                .dispatch_1d_with_inline(&pipeline, &bindings, &[&args], 1, 1)
                .unwrap();
        }
        args[..4].copy_from_slice(&11u32.to_le_bytes());
        args[4..].copy_from_slice(&1u32.to_le_bytes());
        unsafe {
            batch
                .dispatch_1d_with_inline(&pipeline, &bindings, &[&args], 1, 1)
                .unwrap();
        }
        args.fill(0xff);
        assert!(output.read_bytes(0, &mut [0; 8]).is_err());
        batch.submit().wait().unwrap();
        let mut bytes = [0; 8];
        output.read_bytes(0, &mut bytes).unwrap();
        assert_eq!(u32::from_le_bytes(bytes[..4].try_into().unwrap()), 7);
        assert_eq!(u32::from_le_bytes(bytes[4..].try_into().unwrap()), 11);

        let mut invalid = metal.batch().unwrap();
        unsafe {
            assert!(
                invalid
                    .dispatch_1d_with_inline(&pipeline, &bindings, &[&[]], 1, 1)
                    .is_err()
            );
            assert!(
                invalid
                    .dispatch_1d_with_inline(&pipeline, &bindings, &[&[0; 4096]], 1, 1)
                    .is_err()
            );
        }
        assert!(!invalid.has_work);
    }

    #[test]
    fn libraries_and_pipelines_reuse_exact_source_function_and_compile_mode() {
        let metal = Metal::new().unwrap();
        assert!(metal.compile_msl("kernel void {", "first").is_err());
        assert!(metal.0.pipelines.borrow().is_empty());

        assert!(metal.compile_msl(CACHE_SOURCE, "missing").is_err());
        {
            let sources = metal.0.pipelines.borrow();
            let cached = sources.get(CACHE_SOURCE).unwrap();
            assert!(cached.default.library.is_some());
            assert!(cached.default.functions.is_empty());
        }

        let first = metal.compile_msl(CACHE_SOURCE, "first").unwrap();
        let repeated = metal.compile_msl(CACHE_SOURCE, "first").unwrap();
        assert!(ptr::eq(&*first.raw, &*repeated.raw));
        let second = metal.compile_msl(CACHE_SOURCE, "second").unwrap();
        assert!(!ptr::eq(&*first.raw, &*second.raw));

        let strict = metal.compile_msl_strict(CACHE_SOURCE, "first").unwrap();
        let strict_repeated = metal.compile_msl_strict(CACHE_SOURCE, "first").unwrap();
        assert!(ptr::eq(&*strict.raw, &*strict_repeated.raw));
        let changed_source = format!("{CACHE_SOURCE}\n");
        metal.compile_msl(&changed_source, "first").unwrap();

        let sources = metal.0.pipelines.borrow();
        assert_eq!(sources.len(), 2);
        let cached = sources.get(CACHE_SOURCE).unwrap();
        assert_eq!(cached.default.functions.len(), 2);
        assert_eq!(cached.strict.functions.len(), 1);
        assert!(cached.strict.library.is_some());
    }

    #[test]
    fn pipeline_cache_is_per_metal_identity_and_does_not_keep_owner_alive() {
        let metal = Metal::new().unwrap();
        let other = Metal::new().unwrap();
        let pipeline = metal.compile_msl(CACHE_SOURCE, "first").unwrap();
        let other_pipeline = other.compile_msl(CACHE_SOURCE, "first").unwrap();
        assert!(!Rc::ptr_eq(&pipeline.device, &other_pipeline.device));
        assert_eq!(metal.0.pipelines.borrow().len(), 1);
        assert_eq!(other.0.pipelines.borrow().len(), 1);

        let owner = Rc::downgrade(&metal.0);
        drop(metal);
        assert!(owner.upgrade().is_some());
        drop(pipeline);
        assert!(owner.upgrade().is_none());
    }

    #[test]
    fn requested_bytes_track_live_and_lifetime_peak_with_checked_arithmetic() {
        let bytes = BufferAccounting::default();
        bytes.add(8).unwrap();
        bytes.add(4).unwrap();
        assert_eq!(bytes.live.get(), 12);
        assert_eq!(bytes.peak.get(), 12);
        assert!(bytes.remove(8));
        bytes.add(1).unwrap();
        assert_eq!(bytes.live.get(), 5);
        assert_eq!(bytes.peak.get(), 12);
        assert!(bytes.add(u64::MAX).is_err());
        assert_eq!(bytes.live.get(), 5);
        assert!(bytes.remove(5));
        assert_eq!(bytes.live.get(), 0);
        assert_eq!(bytes.peak.get(), 12);
    }

    #[test]
    fn requested_bytes_follow_last_buffer_and_inflight_owner() {
        let metal = Metal::new().unwrap();
        let other_handle = metal.clone();
        assert!(metal.shared_buffer(0).is_err());
        assert_eq!(
            other_handle
                .memory_stats()
                .unwrap()
                .live_requested_buffer_bytes,
            0
        );
        let source = "#include <metal_stdlib>\nusing namespace metal;\nkernel void touch(device uint* output [[buffer(0)]], uint tid [[thread_position_in_grid]]) { if (tid == 0) output[0] = 1; }";
        let pipeline = metal.compile_msl(source, "touch").unwrap();
        let buffer = metal.shared_buffer(4).unwrap();
        let clone = buffer.clone();
        assert_eq!(
            other_handle
                .memory_stats()
                .unwrap()
                .live_requested_buffer_bytes,
            4
        );
        drop(clone);
        let mut batch = metal.batch().unwrap();
        // SAFETY: One thread writes the first uint of a four-byte buffer.
        unsafe {
            batch
                .dispatch_1d(
                    &pipeline,
                    &[Binding {
                        buffer: &buffer,
                        offset: 0,
                    }],
                    1,
                    1,
                )
                .unwrap();
        }
        let flight = batch.submit();
        drop(buffer);
        assert_eq!(
            other_handle
                .memory_stats()
                .unwrap()
                .live_requested_buffer_bytes,
            4
        );
        flight.wait().unwrap();
        let stats = other_handle.memory_stats().unwrap();
        assert_eq!(stats.live_requested_buffer_bytes, 0);
        assert_eq!(stats.lifetime_peak_requested_buffer_bytes, 4);
    }

    #[test]
    fn chained_buffers_remain_reserved_until_completion_or_unsubmitted_drop() {
        let metal = Metal::new().unwrap();
        let source = "#include <metal_stdlib>\nusing namespace metal;\nkernel void produce(device uint* out [[buffer(0)]]) { out[0] = 3; }\nkernel void consume(device const uint* input [[buffer(0)]], device uint* out [[buffer(1)]]) { out[0] = 2 * input[0] + 1; }";
        let produce = metal.compile_msl(source, "produce").unwrap();
        let consume = metal.compile_msl(source, "consume").unwrap();
        let intermediate = metal.shared_buffer(4).unwrap();
        let output = metal.shared_buffer(4).unwrap();
        let producer_bindings = [Binding {
            buffer: &intermediate,
            offset: 0,
        }];
        let consumer_bindings = [
            Binding {
                buffer: &intermediate,
                offset: 0,
            },
            Binding {
                buffer: &output,
                offset: 0,
            },
        ];
        let mut bytes = [0u8; 4];
        for drop_flight in [false, true] {
            let mut batch = metal.batch().unwrap();
            // SAFETY: Each single-thread kernel accesses one allocated uint.
            unsafe {
                batch
                    .dispatch_1d(&produce, &producer_bindings, 1, 1)
                    .unwrap();
                batch
                    .dispatch_1d(&consume, &consumer_bindings, 1, 1)
                    .unwrap();
            }
            assert!(intermediate.read_bytes(0, &mut bytes).is_err());
            assert!(intermediate.write_bytes(0, &bytes).is_err());
            assert!(output.read_bytes(0, &mut bytes).is_err());
            let flight = batch.submit();
            assert!(output.read_bytes(0, &mut bytes).is_err());
            if drop_flight {
                drop(flight);
            } else {
                flight.wait().unwrap();
            }
            output.read_bytes(0, &mut bytes).unwrap();
            assert_eq!(u32::from_le_bytes(bytes), 7);
        }
        let mut batch = metal.batch().unwrap();
        // SAFETY: The first dispatch writes one allocated uint; the second
        // rejects its zero grid before creating an encoder or binding buffers.
        unsafe {
            batch
                .dispatch_1d(&produce, &producer_bindings, 1, 1)
                .unwrap();
            assert!(
                batch
                    .dispatch_1d(&consume, &consumer_bindings, 0, 1)
                    .is_err()
            );
        }
        assert!(intermediate.read_bytes(0, &mut bytes).is_err());
        assert!(intermediate.write_bytes(0, &bytes).is_err());
        drop(batch);
        intermediate.read_bytes(0, &mut bytes).unwrap();
        intermediate.write_bytes(0, &0u32.to_le_bytes()).unwrap();
    }

    #[test]
    fn batch_identity_index_retains_aliases_once_and_distinct_buffers_separately() {
        let metal = Metal::new().unwrap();
        let source = "#include <metal_stdlib>\nusing namespace metal;\nkernel void combine(device uint* first [[buffer(0)]], device const uint* alias [[buffer(1)]], device const uint* other [[buffer(2)]]) { first[0] = alias[0] + other[0]; }";
        let pipeline = metal.compile_msl(source, "combine").unwrap();
        let first = metal.shared_buffer(4).unwrap();
        let alias = first.clone();
        let other = metal.shared_buffer(4).unwrap();
        first.write_bytes(0, &1u32.to_le_bytes()).unwrap();
        other.write_bytes(0, &2u32.to_le_bytes()).unwrap();
        let bindings = [
            Binding {
                buffer: &first,
                offset: 0,
            },
            Binding {
                buffer: &alias,
                offset: 0,
            },
            Binding {
                buffer: &other,
                offset: 0,
            },
        ];
        let mut batch = metal.batch().unwrap();
        // SAFETY: One thread reads and writes only the first valid uint.
        unsafe {
            batch.dispatch_1d(&pipeline, &bindings, 1, 1).unwrap();
            batch.dispatch_1d(&pipeline, &bindings, 1, 1).unwrap();
        }
        assert_eq!(batch.buffers.len(), 2);
        assert_eq!(batch.buffer_ids.len(), 2);
        assert_eq!(first.0.uses.get(), 1);
        assert_eq!(other.0.uses.get(), 1);
        assert!(first.read_bytes(0, &mut [0; 4]).is_err());
        assert!(other.read_bytes(0, &mut [0; 4]).is_err());
        batch.submit().wait().unwrap();
        assert_eq!(first.0.uses.get(), 0);
        assert_eq!(other.0.uses.get(), 0);
        let mut result = [0; 4];
        first.read_bytes(0, &mut result).unwrap();
        assert_eq!(u32::from_le_bytes(result), 5);
        other.read_bytes(0, &mut [0; 4]).unwrap();
    }

    #[test]
    fn batch_identity_index_retry_after_use_overflow_and_abandon() {
        let metal = Metal::new().unwrap();
        let pipeline = metal.compile_msl(CACHE_SOURCE, "first").unwrap();
        let buffer = metal.shared_buffer(4).unwrap();
        let binding = [Binding {
            buffer: &buffer,
            offset: 0,
        }];
        let mut batch = metal.batch().unwrap();
        buffer.0.uses.set(usize::MAX);
        // SAFETY: The only GPU access is one uint within the allocation.
        let error = unsafe { batch.dispatch_1d(&pipeline, &binding, 1, 1) }.unwrap_err();
        assert!(error.to_string().contains("use count overflow"));
        assert!(batch.buffers.is_empty());
        assert!(batch.buffer_ids.is_empty());
        assert_eq!(buffer.0.reserved_by.get(), 0);
        buffer.0.uses.set(0);
        unsafe { batch.dispatch_1d(&pipeline, &binding, 1, 1) }.unwrap();
        assert_eq!(batch.buffers.len(), 1);
        assert_eq!(batch.buffer_ids.len(), 1);
        assert_eq!(buffer.0.uses.get(), 1);
        assert!(buffer.read_bytes(0, &mut [0; 4]).is_err());
        drop(batch);
        assert_eq!(buffer.0.uses.get(), 0);
        assert_eq!(buffer.0.reserved_by.get(), 0);
        buffer.read_bytes(0, &mut [0; 4]).unwrap();
    }

    #[test]
    fn ordered_stream_reuses_buffers_across_flushes_and_excludes_other_owners() {
        let metal = Metal::new().unwrap();
        let first = metal.compile_msl(CACHE_SOURCE, "first").unwrap();
        let second = metal.compile_msl(CACHE_SOURCE, "second").unwrap();
        let buffer = metal.shared_buffer(4).unwrap();
        let binding = [Binding {
            buffer: &buffer,
            offset: 0,
        }];
        let mut stream = metal.stream().unwrap();
        unsafe { stream.batch().unwrap().dispatch_1d(&first, &binding, 1, 1) }.unwrap();
        assert_eq!(stream.active.as_ref().unwrap().buffer_ids.len(), 1);
        let first_completion = stream.flush_marked().unwrap();
        unsafe { stream.batch().unwrap().dispatch_1d(&second, &binding, 1, 1) }.unwrap();
        assert_eq!(stream.active.as_ref().unwrap().buffer_ids.len(), 1);
        assert_eq!(buffer.0.uses.get(), 2);
        stream.flush().unwrap();
        let mut unrelated = metal.batch().unwrap();
        assert!(unsafe { unrelated.dispatch_1d(&first, &binding, 1, 1) }.is_err());
        let mut bytes = [0; 4];
        assert!(buffer.read_bytes(0, &mut bytes).is_err());
        assert!(buffer.write_bytes(0, &bytes).is_err());
        stream.wait_through(first_completion).unwrap();
        assert_eq!(buffer.0.uses.get(), 1);
        assert!(buffer.read_bytes(0, &mut bytes).is_err());
        assert!(buffer.write_bytes(0, &bytes).is_err());
        stream.wait().unwrap();
        assert_eq!(buffer.0.uses.get(), 0);
        buffer.read_bytes(0, &mut bytes).unwrap();
        assert_eq!(u32::from_le_bytes(bytes), 2);
        unsafe { unrelated.dispatch_1d(&first, &binding, 1, 1) }.unwrap();
        unrelated.submit().wait().unwrap();
    }

    #[test]
    fn dropping_stream_cancels_active_work_and_drains_flushed_uses() {
        let metal = Metal::new().unwrap();
        let first = metal.compile_msl(CACHE_SOURCE, "first").unwrap();
        let second = metal.compile_msl(CACHE_SOURCE, "second").unwrap();
        let buffer = metal.shared_buffer(4).unwrap();
        let binding = [Binding {
            buffer: &buffer,
            offset: 0,
        }];
        {
            let mut stream = metal.stream().unwrap();
            unsafe { stream.batch().unwrap().dispatch_1d(&first, &binding, 1, 1) }.unwrap();
            stream.flush().unwrap();
            unsafe { stream.batch().unwrap().dispatch_1d(&second, &binding, 1, 1) }.unwrap();
            assert!(buffer.write_bytes(0, &0u32.to_le_bytes()).is_err());
        }
        let mut bytes = [0; 4];
        buffer.read_bytes(0, &mut bytes).unwrap();
        assert_eq!(u32::from_le_bytes(bytes), 1);
    }
}
