//! Compact validation for device-owned execution intermediates.

use anyhow::{Result, ensure};

use crate::metal::{Batch, Binding, Buffer, Metal, Pipeline};

/// Sticky request status. Reset only at a quiescent request boundary and inspect
/// only after every command that writes it has completed.
pub(crate) struct DeviceStatus {
    status: Buffer,
    bf16: Pipeline,
    f32: Pipeline,
}

impl DeviceStatus {
    pub(crate) fn new(metal: &Metal) -> Result<Self> {
        let source = include_str!("../kernels/metal/device_status.metal");
        let status = Self {
            status: metal.shared_buffer(4)?,
            bf16: metal.compile_msl_strict(source, "check_finite_bf16")?,
            f32: metal.compile_msl_strict(source, "check_finite_f32")?,
        };
        status.reset()?;
        Ok(status)
    }

    pub(crate) fn reset(&self) -> Result<()> {
        self.status.write_bytes(0, &0u32.to_le_bytes())
    }

    /// Kernels may atomically set this flag; they must never clear it.
    pub(crate) fn flag_buffer(&self) -> &Buffer {
        &self.status
    }

    pub(crate) fn encode_bf16(&self, batch: &mut Batch, buffer: &Buffer) -> Result<()> {
        ensure!(
            buffer.byte_len().is_multiple_of(2),
            "finite-check buffer has a partial element"
        );
        self.encode_bf16_count(batch, buffer, buffer.byte_len() / 2)
    }

    pub(crate) fn encode_bf16_count(
        &self,
        batch: &mut Batch,
        buffer: &Buffer,
        count: usize,
    ) -> Result<()> {
        self.encode(batch, buffer, count, 2, &self.bf16)
    }

    pub(crate) fn encode_f32(&self, batch: &mut Batch, buffer: &Buffer) -> Result<()> {
        ensure!(
            buffer.byte_len().is_multiple_of(4),
            "finite-check buffer has a partial element"
        );
        self.encode(batch, buffer, buffer.byte_len() / 4, 4, &self.f32)
    }

    fn encode(
        &self,
        batch: &mut Batch,
        buffer: &Buffer,
        count: usize,
        width: usize,
        pipeline: &Pipeline,
    ) -> Result<()> {
        ensure!(
            count <= buffer.byte_len() / width,
            "finite-check extent exceeds buffer"
        );
        ensure!(count > 0, "finite-check buffer is empty");
        ensure!(
            u32::try_from(count).is_ok(),
            "finite-check grid exceeds uint indexing"
        );
        // SAFETY: The exact grid covers each element once. Exponent checks use
        // integer bits and cannot turn nonfinite values into memory addresses.
        unsafe {
            batch.dispatch_1d(
                pipeline,
                &[
                    Binding { buffer, offset: 0 },
                    Binding {
                        buffer: &self.status,
                        offset: 0,
                    },
                ],
                count,
                128,
            )?;
        }
        Ok(())
    }

    pub(crate) fn check(&self) -> Result<()> {
        let mut bytes = [0; 4];
        self.status.read_bytes(0, &mut bytes)?;
        ensure!(
            u32::from_le_bytes(bytes) == 0,
            "device execution produced nonfinite values"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_flags_are_sticky_and_require_completion() -> Result<()> {
        let metal = Metal::new()?;
        let status = DeviceStatus::new(&metal)?;
        let values = metal.shared_buffer(12)?;
        let finite = [0x0000u16, 0x8000, 0x0001, 0x8001, 0x7f7f, 0xff7f];
        let bytes: Vec<_> = finite.iter().flat_map(|v| v.to_le_bytes()).collect();
        values.write_bytes(0, &bytes)?;
        let mut stream = metal.stream()?;
        status.encode_bf16(stream.batch()?, &values)?;
        assert!(status.check().is_err());
        assert!(status.reset().is_err());
        stream.wait()?;
        status.check()?;
        for invalid in [0x7f80u16, 0xff80, 0x7fc1] {
            status.reset()?;
            values.write_bytes(0, &invalid.to_le_bytes())?;
            status.encode_bf16(stream.batch()?, &values)?;
            stream.wait()?;
            assert!(status.check().is_err());
            values.write_bytes(0, &0u16.to_le_bytes())?;
            status.encode_bf16(stream.batch()?, &values)?;
            stream.wait()?;
            assert!(status.check().is_err());
        }
        status.reset()?;
        status.check()?;
        Ok(())
    }

    #[test]
    fn fp32_status_checks_exponents_without_float_arithmetic() -> Result<()> {
        let metal = Metal::new()?;
        let status = DeviceStatus::new(&metal)?;
        let values = metal.shared_buffer(24)?;
        let finite = [0u32, 0x8000_0000, 1, 0x8000_0001, 0x7f7f_ffff, 0xff7f_ffff];
        let bytes: Vec<_> = finite.iter().flat_map(|v| v.to_le_bytes()).collect();
        values.write_bytes(0, &bytes)?;
        let mut stream = metal.stream()?;
        status.encode_f32(stream.batch()?, &values)?;
        stream.wait()?;
        status.check()?;
        for invalid in [0x7f80_0000u32, 0xff80_0000, 0x7fc0_0001] {
            status.reset()?;
            values.write_bytes(0, &invalid.to_le_bytes())?;
            status.encode_f32(stream.batch()?, &values)?;
            stream.wait()?;
            assert!(status.check().is_err());
        }
        Ok(())
    }

    #[test]
    fn active_extent_does_not_scan_unused_capacity() -> Result<()> {
        let metal = Metal::new()?;
        let status = DeviceStatus::new(&metal)?;
        let values = metal.shared_buffer(4)?;
        values.write_bytes(0, &[0, 0, 0x80, 0x7f])?;
        let mut stream = metal.stream()?;
        status.encode_bf16_count(stream.batch()?, &values, 1)?;
        stream.wait()?;
        status.check()?;
        assert!(
            status
                .encode_bf16_count(stream.batch()?, &values, 3)
                .is_err()
        );
        status.encode_bf16_count(stream.batch()?, &values, 2)?;
        stream.wait()?;
        assert!(status.check().is_err());
        Ok(())
    }
}
