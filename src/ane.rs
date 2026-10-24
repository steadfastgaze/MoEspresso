//! One synchronous dynamic-weight FP16 projection on the Apple Neural Engine.
//!
//! The combined input is `[input_channels, rows + output_channels]`: each
//! channel stores activation rows followed by input-major weights. Output is
//! channel-major `[output_channels, rows]`. All entries are raw FP16 bits.
//! The owner is neither `Send` nor `Sync`; no asynchronous handle escapes it.

use std::{
    ffi::{CStr, c_char, c_int, c_void},
    fmt,
    marker::PhantomData,
    ptr::NonNull,
    rc::Rc,
    time::{Duration, Instant},
};

mod check;
mod mil;
use mil::mil;

pub use check::{CheckEvent, CheckReport, check};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProjectionShape {
    pub(crate) rows: usize,
    pub(crate) input_channels: usize,
    pub(crate) output_channels: usize,
}

#[derive(Debug)]
pub(crate) struct Error {
    message: String,
    cleanup_complete: bool,
}

impl Error {
    fn new(message: String) -> Self {
        Self {
            message,
            cleanup_complete: true,
        }
    }

    fn without_completed_cleanup(mut self) -> Self {
        self.cleanup_complete = false;
        self
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for Error {}
pub(crate) type Result<T> = std::result::Result<T, Error>;

pub(crate) fn startup_fallback_allowed(error: &anyhow::Error) -> bool {
    error.chain().all(|cause| {
        cause
            .downcast_ref::<Error>()
            .is_none_or(|error| error.cleanup_complete)
    })
}

impl ProjectionShape {
    pub(crate) fn new(rows: usize, input_channels: usize, output_channels: usize) -> Result<Self> {
        let shape = Self {
            rows,
            input_channels,
            output_channels,
        };
        shape.validate()?;
        Ok(shape)
    }

    fn validate(self) -> Result<()> {
        if self.rows < 64
            || !self.rows.is_multiple_of(16)
            || self.input_channels == 0
            || self.output_channels == 0
        {
            return Err(Error::new(
                "rows must be at least 64 and divisible by 16; channel counts must be positive"
                    .into(),
            ));
        }
        if [self.rows, self.input_channels, self.output_channels]
            .iter()
            .any(|&value| value > i32::MAX as usize)
        {
            return Err(Error::new("MIL dimension exceeds i32".into()));
        }
        self.input_elements()?;
        self.output_elements()?;
        self.weight_elements()?;
        Ok(())
    }

    pub(crate) fn input_elements(self) -> Result<usize> {
        checked_product(self.rows, self.input_channels)
    }
    pub(crate) fn output_elements(self) -> Result<usize> {
        checked_product(self.rows, self.output_channels)
    }
    pub(crate) fn weight_elements(self) -> Result<usize> {
        checked_product(self.input_channels, self.output_channels)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DynamicLayout {
    pub(crate) shape: ProjectionShape,
    pub(crate) spatial: usize,
    pub(crate) input_elements: usize,
}

impl DynamicLayout {
    pub(crate) fn new(shape: ProjectionShape) -> Result<Self> {
        shape.validate()?;
        let spatial = shape
            .rows
            .checked_add(shape.output_channels)
            .filter(|&n| n <= i32::MAX as usize)
            .ok_or_else(|| Error::new("dynamic input spatial dimension exceeds i32".into()))?;
        let input_elements = checked_product(shape.input_channels, spatial)?;
        Ok(Self {
            shape,
            spatial,
            input_elements,
        })
    }
}

fn checked_product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .filter(|n| *n <= isize::MAX as usize / 2)
        .ok_or_else(|| Error::new("tensor size overflow".into()))
}

fn check_length(actual: usize, expected: usize, name: &str) -> Result<()> {
    if actual != expected {
        return Err(Error::new(format!(
            "{name} has {actual} elements; expected {expected}"
        )));
    }
    Ok(())
}

unsafe extern "C" {
    fn dyn_ane_memory_stats(footprint: *mut u64, resident: *mut u64) -> c_int;
    fn dyn_ane_compile_stages(handle: *mut c_void, compile: *mut f64, load: *mut f64) -> c_int;
    fn dyn_ane_projection_io_create(
        input_bytes: usize,
        output_bytes: usize,
        error: *mut c_char,
        error_cap: usize,
        cleanup_complete: *mut c_int,
    ) -> *mut c_void;
    fn dyn_ane_projection_io_destroy(
        handle: *mut c_void,
        error: *mut c_char,
        error_cap: usize,
    ) -> c_int;
    fn dyn_ane_dense_create(
        mil: *const u8,
        mil_len: usize,
        io: *mut c_void,
        input_bytes: usize,
        output_bytes: usize,
        error: *mut c_char,
        error_cap: usize,
        cleanup_complete: *mut c_int,
    ) -> *mut c_void;
    fn dyn_ane_dense_write(
        handle: *mut c_void,
        data: *const u16,
        elements: usize,
        error: *mut c_char,
        error_cap: usize,
    ) -> c_int;
    fn dyn_ane_dense_evaluate(handle: *mut c_void, error: *mut c_char, error_cap: usize) -> c_int;
    fn dyn_ane_dense_read(
        handle: *mut c_void,
        data: *mut u16,
        elements: usize,
        error: *mut c_char,
        error_cap: usize,
    ) -> c_int;
    fn dyn_ane_dense_destroy(handle: *mut c_void, error: *mut c_char, error_cap: usize) -> c_int;
}

/// Process physical footprint and resident size in bytes, queried without ANE access.
pub(crate) fn memory_stats() -> Result<(u64, u64)> {
    let (mut footprint, mut resident) = (0, 0);
    if unsafe { dyn_ane_memory_stats(&mut footprint, &mut resident) } != 0 {
        return Err(Error::new("task_vm_info memory query failed".into()));
    }
    Ok((footprint, resident))
}

fn native_error(buffer: &[c_char; 2048]) -> Error {
    // Every native error path NUL-terminates this caller-owned buffer.
    Error::new(
        unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned(),
    )
}

// The native program retains a second pair reference before private compile/load.
// A failed unload keeps that reference even when this Rust owner is dropped.
struct IoOwner(Option<NonNull<c_void>>);

impl IoOwner {
    fn handle(&self) -> *mut c_void {
        self.0.expect("live ANE I/O owner").as_ptr()
    }

    fn close(&mut self) -> Result<()> {
        let Some(handle) = self.0.take() else {
            return Ok(());
        };
        let mut error = [0; 2048];
        if unsafe {
            dyn_ane_projection_io_destroy(handle.as_ptr(), error.as_mut_ptr(), error.len())
        } != 0
        {
            return Err(native_error(&error).without_completed_cleanup());
        }
        Ok(())
    }
}

impl Drop for IoOwner {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            eprintln!("ANE I/O cleanup failed: {error}");
        }
    }
}

#[derive(Default)]
struct ExecutionState {
    input_ready: bool,
    output_ready: bool,
    poisoned: bool,
}

impl ExecutionState {
    fn healthy(&self) -> Result<()> {
        if self.poisoned {
            return Err(Error::new(
                "projection is poisoned after a native operation failure".into(),
            ));
        }
        Ok(())
    }
    fn begin_write(&mut self) -> Result<()> {
        self.healthy()?;
        self.input_ready = false;
        self.output_ready = false;
        Ok(())
    }
    fn input_written(&mut self) {
        self.input_ready = true;
        self.output_ready = false;
    }
    fn begin_evaluate(&mut self) -> Result<()> {
        self.healthy()?;
        if !self.input_ready {
            return Err(Error::new("input must be written before evaluation".into()));
        }
        self.output_ready = false;
        Ok(())
    }
    fn output_completed(&mut self) {
        self.output_ready = true;
    }
    fn require_output(&self) -> Result<()> {
        self.healthy()?;
        if !self.output_ready {
            return Err(Error::new("no completed output is available".into()));
        }
        Ok(())
    }
    fn finish_native(&mut self, status: c_int, error: &[c_char; 2048]) -> Result<()> {
        if status != 0 {
            self.poisoned = true;
            self.input_ready = false;
            self.output_ready = false;
            return Err(native_error(error));
        }
        self.healthy()
    }
}

/// One loaded dynamic-weight program and its exclusively owned IOSurface pair.
///
/// Input writes, evaluation and output reads are synchronous. A native failure
/// permanently poisons execution; explicit `close` still attempts unload. A
/// failed unload quarantines the native model, request and surface references.
pub(crate) struct DynamicProjection {
    handle: Option<NonNull<c_void>>,
    _io: IoOwner,
    layout: DynamicLayout,
    state: ExecutionState,
    _exclusive_thread: PhantomData<Rc<()>>,
}

impl DynamicProjection {
    pub(crate) fn compile(shape: ProjectionShape) -> Result<Self> {
        let layout = DynamicLayout::new(shape)?;
        let source = mil(shape)?;
        let mut error = [0; 2048];
        let mut cleanup_complete = 1;
        let io = unsafe {
            dyn_ane_projection_io_create(
                layout.input_elements * 2,
                shape.output_elements()? * 2,
                error.as_mut_ptr(),
                error.len(),
                &mut cleanup_complete,
            )
        };
        let mut io = IoOwner(Some(NonNull::new(io).ok_or_else(|| {
            let error = native_error(&error);
            if cleanup_complete == 0 {
                error.without_completed_cleanup()
            } else {
                error
            }
        })?));
        let handle = unsafe {
            dyn_ane_dense_create(
                source.as_ptr(),
                source.len(),
                io.handle(),
                layout.input_elements * 2,
                shape.output_elements()? * 2,
                error.as_mut_ptr(),
                error.len(),
                &mut cleanup_complete,
            )
        };
        let Some(handle) = NonNull::new(handle) else {
            let mut failure = native_error(&error);
            if cleanup_complete == 0 {
                failure = failure.without_completed_cleanup();
            }
            if let Err(cleanup) = io.close() {
                failure.message = format!("{}; I/O cleanup failed: {cleanup}", failure.message);
                failure.cleanup_complete = false;
            }
            return Err(failure);
        };
        Ok(Self {
            handle: Some(handle),
            _io: io,
            layout,
            state: ExecutionState::default(),
            _exclusive_thread: PhantomData,
        })
    }

    fn handle(&self) -> Result<*mut c_void> {
        self.state.healthy()?;
        self.handle
            .map(NonNull::as_ptr)
            .ok_or_else(|| Error::new("projection is closed".into()))
    }

    /// Copy exactly `[input_channels, rows + output_channels]` raw FP16 cells.
    /// Each channel contains activation rows followed by its weight row.
    pub(crate) fn write_combined_input(&mut self, combined: &[u16]) -> Result<()> {
        check_length(combined.len(), self.layout.input_elements, "combined input")?;
        self.state.begin_write()?;
        let mut error = [0; 2048];
        let status = unsafe {
            dyn_ane_dense_write(
                self.handle()?,
                combined.as_ptr(),
                combined.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        self.state.finish_native(status, &error)?;
        self.state.input_written();
        Ok(())
    }

    /// Private compile and load wall times, excluding surface creation and copies.
    pub(crate) fn compile_stage_seconds(&self) -> Result<(f64, f64)> {
        let (mut compile, mut load) = (0.0, 0.0);
        if unsafe { dyn_ane_compile_stages(self.handle()?, &mut compile, &mut load) } != 0 {
            return Err(Error::new(
                "compile timing requested for an unloaded program".into(),
            ));
        }
        Ok((compile, load))
    }

    /// Completed synchronous evaluation wall time, excluding the I/O copies.
    /// The same combined input may be evaluated again until a new input write.
    pub(crate) fn evaluate(&mut self) -> Result<Duration> {
        self.state.begin_evaluate()?;
        let mut error = [0; 2048];
        let start = Instant::now();
        let status =
            unsafe { dyn_ane_dense_evaluate(self.handle()?, error.as_mut_ptr(), error.len()) };
        let elapsed = start.elapsed();
        self.state.finish_native(status, &error)?;
        self.state.output_completed();
        Ok(elapsed)
    }

    /// Copy completed channel-major output `[output_channels, rows]`.
    pub(crate) fn read_output_channel_major(&mut self, output: &mut [u16]) -> Result<()> {
        check_length(output.len(), self.layout.shape.output_elements()?, "output")?;
        self.state.require_output()?;
        let mut error = [0; 2048];
        let status = unsafe {
            dyn_ane_dense_read(
                self.handle()?,
                output.as_mut_ptr(),
                output.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        self.state.finish_native(status, &error)
    }

    /// Unload explicitly; report unload and owned-directory cleanup failures.
    pub(crate) fn close(mut self) -> Result<()> {
        self.close_inner()
    }
    fn close_inner(&mut self) -> Result<()> {
        let primary = if let Some(handle) = self.handle.take() {
            let mut error = [0; 2048];
            let status =
                unsafe { dyn_ane_dense_destroy(handle.as_ptr(), error.as_mut_ptr(), error.len()) };
            if status == 0 {
                Ok(())
            } else {
                Err(native_error(&error).without_completed_cleanup())
            }
        } else {
            Ok(())
        };
        let cleanup = self._io.close();
        match (primary, cleanup) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
            (Err(primary), Err(cleanup)) => Err(Error::new(format!(
                "{primary}; I/O cleanup failed: {cleanup}"
            ))
            .without_completed_cleanup()),
        }
    }
}

impl Drop for DynamicProjection {
    fn drop(&mut self) {
        if let Err(error) = self.close_inner() {
            eprintln!("ANE cleanup failed: {error}");
        }
    }
}

#[cfg(test)]
mod runtime_tests;
