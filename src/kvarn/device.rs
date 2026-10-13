//! Session-owned shared Metal storage; host copies exist only at explicit inspection boundaries.

use super::*;
use crate::metal::{Batch, Binding, Buffer, Metal, Pipeline};

const ENCODE_SOURCE: &str = include_str!("../../kernels/metal/kvarn_encode.metal");

pub(super) struct DeviceAppendKernels {
    pub(super) append: Pipeline,
    pub(super) copy: Pipeline,
    pub(super) prepare: Pipeline,
    pub(super) pack_key: Pipeline,
    pub(super) pack_value: Pipeline,
}

impl DeviceAppendKernels {
    pub(super) fn new(metal: &Metal) -> Result<Self> {
        Ok(Self {
            append: metal.compile_msl_strict(ENCODE_SOURCE, "kvarn_append_chunk")?,
            copy: metal.compile_msl_strict(ENCODE_SOURCE, "kvarn_copy_records")?,
            prepare: metal.compile_msl_strict(ENCODE_SOURCE, "kvarn_prepare_seal")?,
            pack_key: metal.compile_msl_strict(ENCODE_SOURCE, "kvarn_pack_key")?,
            pack_value: metal.compile_msl_strict(ENCODE_SOURCE, "kvarn_pack_value")?,
        })
    }
}

pub struct DeviceKvarnState {
    metal: Metal,
    max_context: usize,
    frontier: usize,
    valid: bool,
    records: Buffer,
    sink_keys: Buffer,
    sink_values: Buffer,
    tail_keys: Buffer,
    tail_values: Buffer,
    pending_frontier: Option<usize>,
    pending_seal_status: Option<Buffer>,
}

fn zero_buffer(metal: &Metal, bytes: usize) -> Result<Buffer> {
    let buffer = metal.shared_buffer(bytes.max(2))?;
    let zeros = [0u8; 16_384];
    for start in (0..buffer.byte_len()).step_by(zeros.len()) {
        buffer.write_bytes(start, &zeros[..zeros.len().min(buffer.byte_len() - start)])?;
    }
    Ok(buffer)
}

fn copy_buffer(source: &Buffer, target: &Buffer, bytes: usize) -> Result<()> {
    let mut scratch = [0u8; 16_384];
    for start in (0..bytes).step_by(scratch.len()) {
        let n = scratch.len().min(bytes - start);
        source.read_bytes(start, &mut scratch[..n])?;
        target.write_bytes(start, &scratch[..n])?;
    }
    Ok(())
}

impl DeviceKvarnState {
    /// Import validated, quiescent checkpoint buffers. Packed records contain
    /// only completed tiles; sink and tail buffers retain their physical strides.
    pub(crate) fn from_buffers(
        metal: &Metal,
        max_context: usize,
        frontier: usize,
        buffers: [Buffer; 5],
    ) -> Result<Self> {
        ensure!(
            (1..=MAX_CONTEXT_TOKENS).contains(&max_context) && frontier <= max_context,
            "KVarN restored frontier exceeds context"
        );
        let records_bytes =
            partition(frontier).1.saturating_sub(SINK_TOKENS) / TILE_TOKENS * TILE_RECORD_BYTES;
        for (buffer, bytes) in buffers.iter().zip([
            records_bytes.max(2),
            SINK_BYTES,
            SINK_BYTES,
            TAIL_BYTES,
            TAIL_BYTES,
        ]) {
            ensure!(
                buffer.byte_len() == bytes,
                "KVarN restored buffer geometry differs"
            );
            buffer.read_bytes(0, &mut [])?;
        }
        let [records, sink_keys, sink_values, tail_keys, tail_values] = buffers;
        Ok(Self {
            metal: metal.clone(),
            max_context,
            frontier,
            valid: true,
            records,
            sink_keys,
            sink_values,
            tail_keys,
            tail_values,
            pending_frontier: None,
            pending_seal_status: None,
        })
    }

    pub fn zero(metal: &Metal, max_context: usize) -> Result<Self> {
        ensure!(
            (1..=MAX_CONTEXT_TOKENS).contains(&max_context),
            "KVarN context capacity is outside the released range"
        );
        Ok(Self {
            metal: metal.clone(),
            max_context,
            frontier: 0,
            valid: true,
            records: zero_buffer(metal, 0)?,
            sink_keys: zero_buffer(metal, SINK_BYTES)?,
            sink_values: zero_buffer(metal, SINK_BYTES)?,
            tail_keys: zero_buffer(metal, TAIL_BYTES)?,
            tail_values: zero_buffer(metal, TAIL_BYTES)?,
            pending_frontier: None,
            pending_seal_status: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn restore(
        metal: &Metal,
        max_context: usize,
        frontier: usize,
        records: &[u8],
        sink_keys: &[u8],
        sink_values: &[u8],
        tail_keys: &[u8],
        tail_values: &[u8],
    ) -> Result<Self> {
        KvarnState::validate_parts(
            max_context,
            frontier,
            records,
            sink_keys,
            sink_values,
            tail_keys,
            tail_values,
        )?;
        let upload = |bytes: &[u8]| -> Result<Buffer> {
            let buffer = zero_buffer(metal, bytes.len())?;
            buffer.write_bytes(0, bytes)?;
            Ok(buffer)
        };
        Ok(Self {
            metal: metal.clone(),
            max_context,
            frontier,
            valid: true,
            records: upload(records)?,
            sink_keys: upload(sink_keys)?,
            sink_values: upload(sink_values)?,
            tail_keys: upload(tail_keys)?,
            tail_values: upload(tail_values)?,
            pending_frontier: None,
            pending_seal_status: None,
        })
    }

    pub fn frontier(&self) -> Result<usize> {
        ensure!(self.valid, "KVarN state contains an incomplete append");
        Ok(self.frontier)
    }
    pub fn context_limit(&self) -> usize {
        self.max_context
    }

    pub(crate) fn next_token_backing_growth(&self) -> Result<Option<(usize, usize)>> {
        let next = self
            .frontier()?
            .checked_add(1)
            .context("KVarN frontier overflow")?;
        Ok(
            record_growth_at_frontier(next, self.records.byte_len(), self.max_context)?
                .map(|target| (self.records.byte_len(), target)),
        )
    }
    pub fn record_count(&self) -> usize {
        partition(self.frontier).1.saturating_sub(SINK_TOKENS) / TILE_TOKENS
    }
    pub fn record_bytes(&self) -> usize {
        self.record_count() * TILE_RECORD_BYTES
    }

    /// Borrowed allocations are immutable until every command reader completes.
    pub(crate) fn buffers(&self) -> Result<[(&Buffer, usize); 5]> {
        self.frontier()?;
        Ok([
            (&self.records, self.record_bytes()),
            (&self.sink_keys, SINK_BYTES),
            (&self.sink_values, SINK_BYTES),
            (&self.tail_keys, TAIL_BYTES),
            (&self.tail_values, TAIL_BYTES),
        ])
    }

    pub fn safe_chunk_tokens(&self, requested: usize) -> Result<usize> {
        self.frontier()?;
        ensure!(requested > 0, "KVarN append width must be positive");
        let next_seal = SINK_TOKENS + EXACT_SUFFIX + (self.record_count() + 1) * TILE_TOKENS;
        Ok(requested.min(next_seal - self.frontier))
    }

    /// Encode one token-major BF16 K/V row after its pending attention readers.
    /// The caller must complete the command owner before publishing this state.
    pub fn encode_append_token(
        &mut self,
        batch: &mut Batch,
        encoder: &KvarnEncoder,
        keys: &Buffer,
        values: &Buffer,
    ) -> Result<()> {
        self.encode_append_chunk(batch, encoder, keys, values, 1)
    }

    /// Append projected rows in segments separated by record seals. A seal
    /// consumes the old ring before the row that makes that record eligible.
    /// The encoder owns reusable normalization storage for the ordered stream.
    pub fn encode_append_chunk(
        &mut self,
        batch: &mut Batch,
        encoder: &KvarnEncoder,
        keys: &Buffer,
        values: &Buffer,
        rows: usize,
    ) -> Result<()> {
        let frontier = self.frontier()?;
        ensure!(
            (1..=crate::prefill_capacity::MAX_ROWS).contains(&rows),
            "KVarN append chunk size differs"
        );
        ensure!(
            keys.byte_len() >= rows * TOKEN_BYTES && values.byte_len() >= rows * TOKEN_BYTES,
            "KVarN append chunk K/V extent differs"
        );
        let next = frontier
            .checked_add(rows)
            .context("KVarN frontier overflow")?;
        ensure!(
            next <= self.max_context,
            "KVarN append chunk exceeds context capacity"
        );
        let old_records = self.record_count();
        let next_records = partition(next).1.saturating_sub(SINK_TOKENS) / TILE_TOKENS;
        // Reserve once for the complete chunk. This retains at most one old
        // record backing per layer while all newly sealed records share it.
        let grown = record_growth_at_frontier(next, self.records.byte_len(), self.max_context)?
            .map(|bytes| self.metal.shared_buffer(bytes))
            .transpose()?;
        let status = (next_records > old_records)
            .then(|| zero_buffer(&self.metal, 4))
            .transpose()?;
        let kernels = encoder.append_kernels()?;
        self.valid = false;
        if let Some(new_records) = grown {
            let valid_bytes = old_records * TILE_RECORD_BYTES;
            if valid_bytes > 0 {
                let bindings =
                    [&self.records, &new_records].map(|buffer| Binding { buffer, offset: 0 });
                // SAFETY: The old immutable prefix is copied to disjoint
                // storage before any new record is packed.
                unsafe {
                    batch.dispatch_1d(&kernels.copy, &bindings, valid_bytes, 256)?;
                }
            }
            self.records = new_records;
        }
        let mut position = frontier;
        let mut records = old_records;
        while position < next {
            let seal_before = SINK_TOKENS + EXACT_SUFFIX + (records + 1) * TILE_TOKENS - 1;
            if position == seal_before {
                encoder.encode_seal(
                    batch,
                    &self.tail_keys,
                    &self.tail_values,
                    &self.records,
                    SINK_TOKENS + records * TILE_TOKENS,
                    records * TILE_RECORD_BYTES,
                    status.as_ref().context("KVarN seal status missing")?,
                )?;
                records += 1;
                continue;
            }
            ensure!(
                position < seal_before,
                "KVarN append passed a seal boundary"
            );
            let count = (next - position).min(seal_before - position);
            let offset = (position - frontier) * TOKEN_BYTES;
            let bindings = [
                (keys, offset),
                (values, offset),
                (&self.sink_keys, 0),
                (&self.sink_values, 0),
                (&self.tail_keys, 0),
                (&self.tail_values, 0),
            ]
            .map(|(buffer, offset)| Binding { buffer, offset });
            let args = [
                u32::try_from(position)?.to_le_bytes(),
                u32::try_from(count)?.to_le_bytes(),
            ]
            .concat();
            // SAFETY: Segment positions are unique and do not cross a seal.
            // Each token/head/channel writes one disjoint sink or ring element.
            unsafe {
                batch.dispatch_1d_with_inline(
                    &kernels.append,
                    &bindings,
                    &[&args],
                    count * KV_HEADS * HEAD_DIM,
                    256,
                )?;
            }
            position += count;
        }
        ensure!(records == next_records, "KVarN segment seal count differs");
        self.pending_frontier = Some(next);
        self.pending_seal_status = status;
        Ok(())
    }

    /// Publish only after the containing stream completed successfully.
    pub(crate) fn publish_after_wait(&mut self) -> Result<usize> {
        let next = self
            .pending_frontier
            .context("KVarN has no encoded append to publish")?;
        ensure!(!self.valid, "KVarN append was already published");
        // A zero-byte host access checks command ownership without copying KV.
        for buffer in [
            &self.sink_keys,
            &self.sink_values,
            &self.tail_keys,
            &self.tail_values,
            &self.records,
        ] {
            buffer.read_bytes(0, &mut [])?;
        }
        if let Some(status) = &self.pending_seal_status {
            let mut flag = [0u8; 4];
            status.read_bytes(0, &mut flag)?;
            ensure!(
                u32::from_le_bytes(flag) == 0,
                "KVarN sealed metadata is nonfinite or exceeds FP16"
            );
        }
        self.frontier = next;
        self.pending_frontier = None;
        self.pending_seal_status = None;
        self.valid = true;
        Ok(next)
    }

    /// Temporary host representation for component inspection, never retained by a session.
    pub fn readback(&self) -> Result<KvarnState> {
        let mut arrays = Vec::with_capacity(5);
        for (buffer, len) in self.buffers()? {
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(len)
                .context("KVarN readback reservation failed")?;
            bytes.resize(len, 0);
            buffer.read_bytes(0, &mut bytes)?;
            arrays.push(bytes);
        }
        let [records, sink_keys, sink_values, tail_keys, tail_values]: [Vec<u8>; 5] = arrays
            .try_into()
            .map_err(|_| anyhow::anyhow!("KVarN readback parts differ"))?;
        Ok(KvarnState {
            max_context: self.max_context,
            frontier: self.frontier,
            records,
            sink_keys,
            sink_values,
            tail_keys,
            tail_values,
        })
    }

    /// Append finite head-major rows only after all pending attention readers finish.
    pub fn append(&mut self, encoder: &KvarnEncoder, keys: &[u8], values: &[u8]) -> Result<()> {
        self.frontier()?;
        ensure!(
            !keys.is_empty() && keys.len().is_multiple_of(TOKEN_BYTES),
            "KVarN append keys must contain complete tokens"
        );
        ensure!(
            values.len() == keys.len(),
            "KVarN append values differ in shape"
        );
        finite_bf16(keys, "KVarN append keys")?;
        finite_bf16(values, "KVarN append values")?;
        let tokens = keys.len() / TOKEN_BYTES;
        let next = self
            .frontier
            .checked_add(tokens)
            .context("KVarN frontier overflow")?;
        ensure!(
            next <= self.max_context,
            "KVarN context capacity is exhausted"
        );
        ensure!(
            tokens <= self.safe_chunk_tokens(tokens)?,
            "KVarN append crosses a tile seal; split at safe_chunk_tokens"
        );
        // Check every reservation before reading or mutating shared state.
        for (buffer, _) in self.buffers()? {
            buffer.read_bytes(0, &mut [])?;
        }
        let next_records = partition(next).1.saturating_sub(SINK_TOKENS) / TILE_TOKENS;
        let grown = if let Some(bytes) =
            record_growth_at_frontier(next, self.records.byte_len(), self.max_context)?
        {
            let buffer = zero_buffer(&self.metal, bytes)?;
            copy_buffer(&self.records, &buffer, self.record_bytes())?;
            Some(buffer)
        } else {
            None
        };
        let record = if next_records > self.record_count() {
            let tile_start = SINK_TOKENS + self.record_count() * TILE_TOKENS;
            let mut tile_keys = vec![0; TILE_TOKENS * TOKEN_BYTES];
            let mut tile_values = vec![0; TILE_TOKENS * TOKEN_BYTES];
            for token in 0..TILE_TOKENS {
                let logical = tile_start + token;
                for head in 0..KV_HEADS {
                    let target = (token * KV_HEADS + head) * HEAD_DIM * 2;
                    if logical < self.frontier {
                        let source = (head * TAIL_CAPACITY
                            + (logical - SINK_TOKENS) % TAIL_CAPACITY)
                            * HEAD_DIM
                            * 2;
                        self.tail_keys
                            .read_bytes(source, &mut tile_keys[target..target + HEAD_DIM * 2])?;
                        self.tail_values
                            .read_bytes(source, &mut tile_values[target..target + HEAD_DIM * 2])?;
                    } else {
                        let source = (head * tokens + logical - self.frontier) * HEAD_DIM * 2;
                        tile_keys[target..target + HEAD_DIM * 2]
                            .copy_from_slice(&keys[source..source + HEAD_DIM * 2]);
                        tile_values[target..target + HEAD_DIM * 2]
                            .copy_from_slice(&values[source..source + HEAD_DIM * 2]);
                    }
                }
            }
            let record = encoder.encode_tile(&tile_keys, &tile_values)?;
            ensure!(
                record.len() == TILE_RECORD_BYTES,
                "KVarN encoder returned an incomplete tile record"
            );
            Some(record)
        } else {
            None
        };
        // A physical write error leaves the state poisoned, including if a later
        // row fails after earlier rows were copied. Frontier publication is last.
        self.valid = false;
        if let Some(buffer) = grown {
            self.records = buffer;
        }
        for token in 0..tokens {
            let logical = self.frontier + token;
            for head in 0..KV_HEADS {
                let source = (head * tokens + token) * HEAD_DIM * 2;
                let (k, v, index) = if logical < SINK_TOKENS {
                    (
                        &self.sink_keys,
                        &self.sink_values,
                        head * SINK_TOKENS + logical,
                    )
                } else {
                    (
                        &self.tail_keys,
                        &self.tail_values,
                        head * TAIL_CAPACITY + (logical - SINK_TOKENS) % TAIL_CAPACITY,
                    )
                };
                k.write_bytes(index * HEAD_DIM * 2, &keys[source..source + HEAD_DIM * 2])?;
                v.write_bytes(index * HEAD_DIM * 2, &values[source..source + HEAD_DIM * 2])?;
            }
        }
        if let Some(record) = record {
            self.records.write_bytes(self.record_bytes(), &record)?;
        }
        self.frontier = next;
        self.valid = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kvarn_pending::PendingKvarnView;
    use crate::metal::Binding;

    fn data(bytes: usize, salt: usize) -> Vec<u8> {
        (0..bytes / 2)
            .flat_map(|i| {
                bf16::from_f32(((i * 17 + salt) % 127) as f32 / 256. - 0.25)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect()
    }

    fn compare(device: &DeviceKvarnState, host: &KvarnState) {
        let actual = device.readback().unwrap();
        assert_eq!(actual.frontier(), host.frontier());
        assert_eq!(actual.records(), host.records());
        assert_eq!(actual.sink_keys(), host.sink_keys());
        assert_eq!(actual.sink_values(), host.sink_values());
        assert_eq!(actual.tail_keys(), host.tail_keys());
        assert_eq!(actual.tail_values(), host.tail_values());
    }

    #[test]
    fn encoded_chunk_reuses_seal_workspace_across_layers_and_boundaries() -> Result<()> {
        let metal = Metal::new()?;
        let encoder = KvarnEncoder::new(&metal)?;
        let sink = data(SINK_BYTES, 3);
        let tail = data(TAIL_BYTES, 19);
        let rows = 512;
        let keys = data(rows * TOKEN_BYTES, 23);
        let values = data(rows * TOKEN_BYTES, 53);
        let window = |source: &[u8], start: usize, count: usize| {
            let mut result = Vec::with_capacity(count * TOKEN_BYTES);
            for head in 0..KV_HEADS {
                for row in start..start + count {
                    let begin = (row * KV_HEADS + head) * HEAD_DIM * 2;
                    result.extend_from_slice(&source[begin..begin + HEAD_DIM * 2]);
                }
            }
            result
        };
        let mut cases = Vec::new();
        for frontier in [0, 127, 8446, 8575] {
            let records = vec![
                0;
                partition(frontier).1.saturating_sub(SINK_TOKENS) / TILE_TOKENS
                    * TILE_RECORD_BYTES
            ];
            let mut host =
                KvarnState::restore(9216, frontier, &records, &sink, &sink, &tail, &tail)?;
            let device = DeviceKvarnState::restore(
                &metal, 9216, frontier, &records, &sink, &sink, &tail, &tail,
            )?;
            let mut cursor = 0;
            while cursor < rows {
                let count = host.safe_chunk_tokens(rows - cursor)?;
                host.append(
                    &encoder,
                    &window(&keys, cursor, count),
                    &window(&values, cursor, count),
                )?;
                cursor += count;
            }
            cases.push((device, host));
        }
        let k = zero_buffer(&metal, keys.len())?;
        let v = zero_buffer(&metal, values.len())?;
        k.write_bytes(0, &keys)?;
        v.write_bytes(0, &values)?;
        let mut stream = metal.stream()?;
        for (index, (device, _)) in cases.iter_mut().enumerate() {
            let committed = device.frontier()?;
            device.encode_append_chunk(stream.batch()?, &encoder, &k, &v, rows)?;
            assert_eq!(
                device.frontier, committed,
                "committed frontier changed before publication"
            );
            assert!(device.frontier().is_err());
            assert!(device.publish_after_wait().is_err());
            if index == 2 {
                // The next layer reuses normalization storage while this
                // layer's four seals remain owned by the submitted command.
                stream.flush()?;
            }
        }
        stream.wait()?;
        for (mut device, host) in cases {
            assert_eq!(device.publish_after_wait()?, host.frontier());
            compare(&device, &host);
        }
        Ok(())
    }

    #[test]
    fn device_append_matches_host_at_sink_seals_wrap_and_growth() {
        let metal = Metal::new().unwrap();
        let encoder = KvarnEncoder::new(&metal).unwrap();
        let sink = data(SINK_BYTES, 3);
        let tail = data(TAIL_BYTES, 19);
        for (frontier, count) in [
            (0, 3),
            (127, 3),
            (8447, 1),
            (8448, 3),
            (8575, 1),
            (8576, 3),
            (8703, 1),
        ] {
            let records = vec![
                0;
                partition(frontier).1.saturating_sub(SINK_TOKENS) / TILE_TOKENS
                    * TILE_RECORD_BYTES
            ];
            let mut host =
                KvarnState::restore(9216, frontier, &records, &sink, &sink, &tail, &tail).unwrap();
            let mut device = DeviceKvarnState::restore(
                &metal, 9216, frontier, &records, &sink, &sink, &tail, &tail,
            )
            .unwrap();
            let keys = data(count * TOKEN_BYTES, 23);
            let values = data(count * TOKEN_BYTES, 53);
            host.append(&encoder, &keys, &values).unwrap();
            device.append(&encoder, &keys, &values).unwrap();
            compare(&device, &host);
            let before = device.frontier().unwrap();
            assert!(device.append(&encoder, &[0; 3], &[0; 3]).is_err());
            assert_eq!(device.frontier().unwrap(), before);
            let bad = [0xc0, 0x7f].repeat(TOKEN_BYTES / 2);
            assert!(device.append(&encoder, &bad, &keys[..TOKEN_BYTES]).is_err());
            assert_eq!(device.frontier().unwrap(), before);
        }
    }

    #[test]
    fn encoded_device_append_publishes_only_after_ordered_completion() {
        let metal = Metal::new().unwrap();
        let encoder = KvarnEncoder::new(&metal).unwrap();
        let mut state = DeviceKvarnState::zero(&metal, 256).unwrap();
        let row = data(TOKEN_BYTES, 11);
        let keys = zero_buffer(&metal, TOKEN_BYTES).unwrap();
        let values = zero_buffer(&metal, TOKEN_BYTES).unwrap();
        keys.write_bytes(0, &row).unwrap();
        values.write_bytes(0, &row).unwrap();
        let mut stream = metal.stream().unwrap();
        state
            .encode_append_token(stream.batch().unwrap(), &encoder, &keys, &values)
            .unwrap();
        assert!(state.frontier().is_err());
        assert!(state.publish_after_wait().is_err());
        stream.wait().unwrap();
        assert_eq!(state.publish_after_wait().unwrap(), 1);
        let mut host = KvarnState::zero(256).unwrap();
        host.append(&encoder, &row, &row).unwrap();
        compare(&state, &host);
    }

    #[test]
    fn encoded_device_append_matches_host_codec_at_seals_and_ring_wrap() {
        let metal = Metal::new().unwrap();
        let encoder = KvarnEncoder::new(&metal).unwrap();
        let sink = data(SINK_BYTES, 3);
        let tail = data(TAIL_BYTES, 19);
        for frontier in [8447, 8575, 8703] {
            let records = vec![
                0;
                partition(frontier).1.saturating_sub(SINK_TOKENS) / TILE_TOKENS
                    * TILE_RECORD_BYTES
            ];
            let mut host =
                KvarnState::restore(9216, frontier, &records, &sink, &sink, &tail, &tail).unwrap();
            let mut state = DeviceKvarnState::restore(
                &metal, 9216, frontier, &records, &sink, &sink, &tail, &tail,
            )
            .unwrap();
            let row = data(TOKEN_BYTES, 23 + frontier);
            let keys = zero_buffer(&metal, TOKEN_BYTES).unwrap();
            let values = zero_buffer(&metal, TOKEN_BYTES).unwrap();
            keys.write_bytes(0, &row).unwrap();
            values.write_bytes(0, &row).unwrap();
            let mut stream = metal.stream().unwrap();
            state
                .encode_append_token(stream.batch().unwrap(), &encoder, &keys, &values)
                .unwrap();
            assert!(state.publish_after_wait().is_err());
            stream.wait().unwrap();
            assert_eq!(state.publish_after_wait().unwrap(), frontier + 1);
            host.append(&encoder, &row, &row).unwrap();
            compare(&state, &host);
        }
    }

    #[test]
    fn encoded_device_append_rejects_unrepresentable_seal_metadata() {
        let metal = Metal::new().unwrap();
        let encoder = KvarnEncoder::new(&metal).unwrap();
        let sink = data(SINK_BYTES, 3);
        let huge = [0x7f, 0x7f].repeat(TAIL_BYTES / 2);
        let mut state =
            DeviceKvarnState::restore(&metal, 9216, 8447, &[], &sink, &sink, &huge, &huge).unwrap();
        let row = data(TOKEN_BYTES, 23);
        let keys = zero_buffer(&metal, TOKEN_BYTES).unwrap();
        let values = zero_buffer(&metal, TOKEN_BYTES).unwrap();
        keys.write_bytes(0, &row).unwrap();
        values.write_bytes(0, &row).unwrap();
        let mut stream = metal.stream().unwrap();
        state
            .encode_append_token(stream.batch().unwrap(), &encoder, &keys, &values)
            .unwrap();
        stream.wait().unwrap();
        assert!(state.publish_after_wait().is_err());
        assert!(state.frontier().is_err());
    }

    #[test]
    fn pending_reservations_reject_mutation_and_outlive_state_owner() {
        let metal = Metal::new().unwrap();
        let encoder = KvarnEncoder::new(&metal).unwrap();
        let mut state = DeviceKvarnState::zero(&metal, 256).unwrap();
        let row = data(TOKEN_BYTES, 7);
        state.append(&encoder, &row, &row).unwrap();
        let pending = zero_buffer(&metal, TOKEN_BYTES).unwrap();
        pending.write_bytes(0, &row).unwrap();
        let mut ids = vec![-1; 2051];
        ids[..2].copy_from_slice(&[0, 1]);
        // SAFETY: Completed singleton row, no state writes while a batch owns it.
        let view = unsafe {
            PendingKvarnView::from_device(&metal, &state, &pending, &pending, 1).unwrap()
        };
        let mut batch = metal.batch().unwrap();
        let rows = view.encode_rows(&mut batch, &ids, 0).unwrap();
        assert!(state.append(&encoder, &row, &row).is_err());
        assert_eq!(state.frontier().unwrap(), 1);
        let pipeline = metal.compile_msl_strict("#include <metal_stdlib>\nusing namespace metal; kernel void noop(device uint* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { if (i == 0) x[0] = 0; }", "noop").unwrap();
        // SAFETY: Zero grid is rejected before encoding any memory access.
        assert!(
            unsafe {
                batch.dispatch_1d(
                    &pipeline,
                    &[Binding {
                        buffer: &rows.keys,
                        offset: 0,
                    }],
                    0,
                    1,
                )
            }
            .is_err()
        );
        drop(batch);
        state.append(&encoder, &row, &row).unwrap();
        let mut ids = vec![-1; 2051];
        ids[..3].copy_from_slice(&[0, 1, 2]);
        // SAFETY: Same completed finite singleton; new view captures new counters.
        let view = unsafe {
            PendingKvarnView::from_device(&metal, &state, &pending, &pending, 1).unwrap()
        };
        let mut batch = metal.batch().unwrap();
        let rows = view.encode_rows(&mut batch, &ids, 0).unwrap();
        let flight = batch.submit();
        drop(view);
        drop(state);
        assert!(rows.keys.read_bytes(0, &mut [0; 2]).is_err());
        flight.wait().unwrap();
        let mut first = vec![0; HEAD_DIM * 2];
        rows.keys.read_bytes(0, &mut first).unwrap();
        assert_eq!(first, row[..HEAD_DIM * 2]);
    }
}
