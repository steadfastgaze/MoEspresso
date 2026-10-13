//! Persistent device-to-owner route publication for one bounded expert pool.

use crate::metal::{
    Batch, Binding, Buffer, EventPoint, Metal, Pipeline, PublicationLease, SharedEvent, Stream,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use std::time::{Duration, Instant};

pub(crate) const REQUESTED_BYTES: usize = 22 * 4 + 4;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SelectedRoutes {
    pub(crate) ids: [u32; 10],
    pub(crate) scores: [u16; 10],
    pub(crate) changed: u32,
}

fn parse(bytes: &[u8; 88]) -> Result<SelectedRoutes> {
    let mut words = [0u32; 22];
    for (word, bytes) in words.iter_mut().zip(bytes.as_chunks::<4>().0) {
        *word = u32::from_le_bytes(*bytes);
    }
    ensure!(
        words[21] == 0,
        "device execution produced nonfinite values before expert demand"
    );
    ensure!(words[20] <= 10, "router change count exceeds top-k");
    let ids: [u32; 10] = words[..10].try_into()?;
    let mut scores = [0; 10];
    for i in 0..10 {
        ensure!(
            ids[i] < 512 && !ids[..i].contains(&ids[i]),
            "invalid or repeated selected expert"
        );
        scores[i] = u16::try_from(words[10 + i])?;
        ensure!(
            bf16::from_bits(scores[i]).is_finite(),
            "nonfinite selected score"
        );
    }
    Ok(SelectedRoutes {
        ids,
        scores,
        changed: words[20],
    })
}

pub(crate) struct PendingDemand {
    pub(crate) ready: EventPoint,
    pub(crate) publication: EventPoint,
}

pub(crate) struct DecodeHandoff {
    ready: SharedEvent,
    publication: SharedEvent,
    epoch: u64,
    mailbox: Buffer,
    flag: Buffer,
    export: Pipeline,
    pending: Option<PendingDemand>,
}

impl DecodeHandoff {
    pub(crate) fn new(metal: &Metal) -> Result<Self> {
        let mailbox = metal.shared_buffer(88)?;
        let flag = metal.shared_buffer(4)?;
        flag.write_bytes(0, &0u32.to_le_bytes())?;
        ensure!(
            mailbox.byte_len() + flag.byte_len() == REQUESTED_BYTES,
            "decode handoff inventory differs"
        );
        Ok(Self {
            ready: metal.shared_event()?,
            publication: metal.shared_event()?,
            epoch: 0,
            mailbox,
            flag,
            export: metal.compile_msl_strict(
                include_str!("../kernels/metal/decode_handoff.metal"),
                "export_decode_routes",
            )?,
            pending: None,
        })
    }
    pub(crate) fn begin(&mut self, stream: &mut Stream) -> Result<()> {
        if let Some(pending) = &self.pending {
            ensure!(
                pending.publication.reached(),
                "previous expert demand remains unresolved"
            );
        }
        // Safe access also proves all previous flag readers have retired.
        self.flag.write_bytes(0, &0u32.to_le_bytes())?;
        let epoch = self
            .epoch
            .checked_add(1)
            .context("decode event epoch exhausted")?;
        let pending = PendingDemand {
            ready: self.ready.point(epoch)?,
            publication: self.publication.point(epoch)?,
        };
        // Release ownership precedes publication of pending host state and
        // every fallible router/validation/export encoding operation.
        stream.register_publication(&pending.publication)?;
        self.pending = Some(pending);
        self.epoch = epoch;
        Ok(())
    }
    pub(crate) fn pending(&self) -> Result<&PendingDemand> {
        self.pending.as_ref().context("no pending expert demand")
    }
    pub(crate) fn flag(&self) -> &Buffer {
        &self.flag
    }
    pub(crate) fn check_quiescent(&self) -> Result<()> {
        if let Some(pending) = &self.pending {
            ensure!(
                pending.publication.reached(),
                "expert demand publication remains unresolved"
            );
        }
        // Event release does not establish command completion. Ordinary safe
        // access rejects both encoded and submitted users of these buffers.
        self.flag.read_bytes(0, &mut [])?;
        self.mailbox.read_bytes(0, &mut [])?;
        Ok(())
    }
    pub(crate) fn encode_export(
        &self,
        batch: &mut Batch,
        ids: &Buffer,
        scores: &Buffer,
        changed: &Buffer,
        status: &Buffer,
    ) -> Result<()> {
        let buffers = [ids, scores, changed, status, &self.mailbox]
            .map(|buffer| Binding { buffer, offset: 0 });
        // SAFETY: Fixed ten-route buffers and a 22-word mailbox. The mailbox
        // is written once per epoch and remains immutable until retirement.
        unsafe {
            batch.dispatch_1d(&self.export, &buffers, 1, 1)?;
        }
        batch.signal_event(&self.pending()?.ready);
        Ok(())
    }
    pub(crate) fn read(&self) -> Result<SelectedRoutes> {
        let ready = &self.pending()?.ready;
        let deadline = Instant::now() + Duration::from_secs(10);
        while !ready.reached() {
            ensure!(
                Instant::now() < deadline,
                "expert route readiness timed out"
            );
            std::thread::yield_now();
        }
        let mut bytes = [0; 88];
        // SAFETY: The export is this mailbox's only writer in the current
        // epoch. Later status updates use a separate device allocation.
        unsafe {
            self.mailbox.read_after_event(ready, &mut bytes)?;
        }
        parse(&bytes)
    }
    pub(crate) fn publish(&mut self) -> Result<()> {
        let point = &self.pending()?.publication;
        // SAFETY: The owning stream gates every flag reader on this point.
        unsafe { PublicationLease::new(&self.flag, point)? }.write(&1u32.to_le_bytes())?;
        point.release();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires exclusive GPU ownership"]
    fn quiescence_rejects_unreleased_and_live_handoff_users() -> Result<()> {
        let metal = Metal::new()?;
        let mut handoff = DecodeHandoff::new(&metal)?;
        handoff.check_quiescent()?;
        let mut stream = metal.stream()?;
        handoff.begin(&mut stream)?;
        assert!(handoff.check_quiescent().is_err());
        drop(stream);
        handoff.check_quiescent()?;
        let kernel = metal.compile_msl(
            "#include <metal_stdlib>\nusing namespace metal; kernel void observe(device uint* p [[buffer(0)]], uint i [[thread_position_in_grid]]) { if (i == 0) p[0] = p[0]; }",
            "observe",
        )?;
        for buffer in [&handoff.flag, &handoff.mailbox] {
            for submit in [false, true] {
                let mut batch = metal.batch()?;
                // SAFETY: Each handoff allocation contains at least one word;
                // the one-thread dispatch accesses only that word.
                unsafe {
                    batch.dispatch_1d(&kernel, &[Binding { buffer, offset: 0 }], 1, 1)?;
                }
                assert!(handoff.check_quiescent().is_err());
                if submit {
                    let flight = batch.submit();
                    assert!(handoff.check_quiescent().is_err());
                    flight.wait()?;
                } else {
                    drop(batch);
                }
                handoff.check_quiescent()?;
            }
        }
        Ok(())
    }

    #[test]
    fn failed_publication_guards_both_routed_stages_and_releases_before_drain() -> Result<()> {
        let metal = Metal::new()?;
        let source = format!(
            "{}\n{}",
            include_str!("../kernels/metal/iqk_routed.metal"),
            include_str!("../kernels/metal/iqk_bounded.metal")
        );
        for codec in ["iq2", "iq3"] {
            let gate = metal.compile_msl_strict(&source, &format!("bounded_gate_{codec}"))?;
            let down = metal.compile_msl_strict(&source, &format!("bounded_down_{codec}"))?;
            for flag_value in [0u32, 2] {
                let flag = metal.shared_buffer(4)?;
                flag.write_bytes(0, &flag_value.to_le_bytes())?;
                // These intentionally cannot address a selected expert or a
                // complete activation row. The uniform failure branch must
                // run before reading any of them, including order and slots.
                let poison = metal.shared_buffer(4)?;
                poison.write_bytes(0, &u32::MAX.to_le_bytes())?;
                let output = metal.shared_buffer(2560 * 2 + 16)?;
                output.write_bytes(0, &vec![0xa5; output.byte_len()])?;
                let binding = |buffer| Binding { buffer, offset: 0 };
                let publication = metal.shared_event()?.point(1)?;
                {
                    let mut stream = metal.stream()?;
                    stream.wait_for_publication(&publication)?;
                    let mut gate_bindings = vec![binding(&poison); 20];
                    gate_bindings.push(binding(&flag));
                    let mut down_bindings = vec![binding(&poison); 12];
                    down_bindings.push(Binding {
                        buffer: &output,
                        offset: 8,
                    });
                    down_bindings.push(binding(&flag));
                    // SAFETY: Neither guarded stage accesses poisoned inputs
                    // or intermediate outputs when flag != 1. Down writes
                    // exactly 2560 words into the output's guarded interior.
                    unsafe {
                        stream
                            .batch()?
                            .dispatch_1d(&gate, &gate_bindings, 400 * 80, 80)?;
                        stream
                            .batch()?
                            .dispatch_1d(&down, &down_bindings, 160 * 320, 320)?;
                    }
                    stream.flush()?;
                    assert!(!publication.reached());
                    if flag_value == 0 {
                        // Also require successful command retirement instead
                        // of discarding completion errors in Drop.
                        stream.abort_and_wait()?;
                    }
                    // The invalid-flag arm tests Drop releasing the committed
                    // wait before draining, without explicit cleanup.
                }
                assert!(publication.reached());
                let mut input = [0; 4];
                poison.read_bytes(0, &mut input)?;
                assert_eq!(input, u32::MAX.to_le_bytes());
                let mut result = vec![0; output.byte_len()];
                output.read_bytes(0, &mut result)?;
                assert_eq!(&result[..8], &[0xa5; 8]);
                assert!(result[8..8 + 2560 * 2].iter().all(|&byte| byte == 0));
                assert_eq!(&result[8 + 2560 * 2..], &[0xa5; 8]);
                // All host reservations must have retired before reuse.
                poison.write_bytes(0, &[0; 4])?;
                output.write_bytes(0, &[0; 8])?;
            }
        }
        Ok(())
    }

    #[test]
    fn early_encoding_error_releases_epoch_before_wait_exists() -> Result<()> {
        let metal = Metal::new()?;
        let mut handoff = DecodeHandoff::new(&metal)?;
        let kernel=metal.compile_msl("#include <metal_stdlib>\nusing namespace metal; kernel void empty_kernel(uint i [[thread_position_in_grid]]) {}", "empty_kernel")?;
        {
            let mut stream = metal.stream()?;
            handoff.begin(&mut stream)?;
            assert!(!handoff.pending()?.publication.reached());
            // SAFETY: No bindings or dispatch occur: zero grid is rejected.
            let error = unsafe { stream.batch()?.dispatch_1d(&kernel, &[], 0, 1) }
                .err()
                .context("invalid encode was accepted")?;
            assert_eq!(
                error.to_string(),
                "Metal dispatch dimensions must be nonzero"
            );
        }
        assert!(handoff.pending()?.publication.reached());
        let mut retry = metal.stream()?;
        handoff.begin(&mut retry)?;
        assert_eq!(handoff.epoch, 2);
        assert!(!handoff.pending()?.publication.reached());
        drop(retry);
        assert!(handoff.pending()?.publication.reached());
        Ok(())
    }
    #[test]
    fn rejects_malformed_routes_before_any_expert_address() {
        let mut words = [0u32; 22];
        for i in 0..10 {
            words[i] = i as u32;
            words[10 + i] = 0x3f80;
        }
        let pack = |words: [u32; 22]| -> [u8; 88] {
            words
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        };
        assert!(parse(&pack(words)).is_ok());
        for (index, value) in [(0, 512), (1, 0), (10, 0x7f80), (20, 11), (21, 1)] {
            let mut bad = words;
            bad[index] = value;
            assert!(parse(&pack(bad)).is_err());
        }
    }
}
