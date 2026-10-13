//! Native committed-state I/O with one bounded staging allocation.

use super::*;
use crate::decoder::GdnLayerState;
use crate::engine::{LayerState, QsaState};
use crate::kvarn::{DeviceKvarnState, HEAD_DIM, KV_HEADS, TAIL_CAPACITY, TOKEN_BYTES};
use crate::metal::{Buffer, Metal};
use crate::ple::DevicePleState;
use crate::qsa_index::DeviceQsaIndexState;
use std::cell::OnceCell;

pub const IO_BUFFER_BYTES: u64 = 8 << 20;

struct Staging(Vec<u8>);

impl Staging {
    fn new(limit: u64) -> Result<Self> {
        let bytes = usize::try_from(limit.min(IO_BUFFER_BYTES))?;
        ensure!(
            bytes >= TILE_RECORD_BYTES,
            "checkpoint I/O buffer is too small"
        );
        let mut data = Vec::new();
        data.try_reserve_exact(bytes)?;
        ensure!(
            data.capacity() as u64 <= limit,
            "checkpoint I/O allocation exceeds allowance"
        );
        data.resize(bytes, 0);
        Ok(Self(data))
    }

    fn chunk_bytes(&self, kind: u16) -> usize {
        let alignment = if kind == KVARN_RECORDS {
            TILE_RECORD_BYTES
        } else {
            4
        };
        self.0.len() / alignment * alignment
    }
}

fn validate_values(kind: u16, bytes: &[u8]) -> Result<()> {
    let valid = if kind == KVARN_RECORDS {
        crate::kvarn::finite_record_metadata(bytes).is_ok()
    } else if kind == GDN_RECURRENT {
        bytes.len().is_multiple_of(4)
            && bytes
                .as_chunks::<4>()
                .0
                .iter()
                .all(|word| f32::from_le_bytes(*word).is_finite())
    } else {
        bytes.len().is_multiple_of(2)
            && bytes
                .as_chunks::<2>()
                .0
                .iter()
                .all(|word| half::bf16::from_bits(u16::from_le_bytes(*word)).is_finite())
    };
    if !valid {
        return Err(InvalidCheckpoint("checkpoint tensor contains invalid values").into());
    }
    Ok(())
}

/// Physical head-major ring prefixes. Once a ring has wrapped, every physical
/// row is saved; before that, rows never written by the model are omitted.
fn spans(frontier: usize, kind: u16, bytes: usize) -> Vec<(usize, usize)> {
    let rows = match kind {
        KVARN_SINK_KEYS | KVARN_SINK_VALUES => Some((SINK_TOKENS, frontier.min(SINK_TOKENS))),
        KVARN_TAIL_KEYS | KVARN_TAIL_VALUES => Some((
            TAIL_CAPACITY,
            frontier.saturating_sub(SINK_TOKENS).min(TAIL_CAPACITY),
        )),
        _ => None,
    };
    match rows {
        Some((stride, live)) => (0..KV_HEADS)
            .map(|head| (head * stride * HEAD_DIM * 2, live * HEAD_DIM * 2))
            .collect(),
        None => vec![(0, bytes)],
    }
}

pub(super) fn ring_payload_bytes(frontier: usize) -> (usize, usize) {
    (
        frontier.min(SINK_TOKENS) * TOKEN_BYTES,
        frontier.saturating_sub(SINK_TOKENS).min(TAIL_CAPACITY) * TOKEN_BYTES,
    )
}

fn write_buffer(
    writer: &mut HashWriter<'_>,
    staging: &mut Staging,
    kind: u16,
    layer: u16,
    buffer: &Buffer,
    bytes: usize,
    frontier: usize,
) -> Result<()> {
    let spans = spans(frontier, kind, bytes);
    let total: usize = spans.iter().map(|(_, len)| len).sum();
    writer.section(kind, layer, total as u64)?;
    for (start, len) in spans {
        ensure!(
            start <= buffer.byte_len() && len <= buffer.byte_len() - start,
            "checkpoint source span exceeds tensor"
        );
        for offset in (0..len).step_by(staging.chunk_bytes(kind)) {
            let width = (len - offset).min(staging.chunk_bytes(kind));
            let scratch = &mut staging.0[..width];
            buffer.read_bytes(start + offset, scratch)?;
            validate_values(kind, scratch)?;
            writer.write(scratch)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn read_buffer(
    reader: &mut HashReader<'_>,
    staging: &mut Staging,
    metal: &Metal,
    kind: u16,
    layer: u16,
    bytes: usize,
    allocation: usize,
    frontier: usize,
) -> Result<Buffer> {
    let spans = spans(frontier, kind, bytes);
    let total: usize = spans.iter().map(|(_, len)| len).sum();
    reader.section(kind, layer, total as u64)?;
    let buffer = metal.shared_buffer(allocation.max(2))?;
    // Restore into an unpublished candidate. Unused ring/index rows are zero,
    // matching newly created device state, and never read from the file.
    if total < buffer.byte_len() {
        let zero_bytes = buffer.byte_len().min(staging.0.len());
        staging.0[..zero_bytes].fill(0);
        for start in (0..buffer.byte_len()).step_by(zero_bytes) {
            buffer.write_bytes(
                start,
                &staging.0[..(buffer.byte_len() - start).min(zero_bytes)],
            )?;
        }
    }
    for (start, len) in spans {
        ensure!(
            start <= buffer.byte_len() && len <= buffer.byte_len() - start,
            "checkpoint target span exceeds tensor"
        );
        for offset in (0..len).step_by(staging.chunk_bytes(kind)) {
            let width = (len - offset).min(staging.chunk_bytes(kind));
            let scratch = &mut staging.0[..width];
            reader.read(scratch)?;
            validate_values(kind, scratch)?;
            buffer.write_bytes(start + offset, scratch)?;
        }
    }
    Ok(buffer)
}

pub(super) fn write_session(
    file: &mut File,
    engine: &Engine,
    session: &Session,
    scope: &[u8; 32],
    expected: u64,
    sections: u32,
    limit: u64,
) -> Result<()> {
    engine.validate_snapshot_source(session)?;
    let mut staging = Staging::new(limit)?;
    let mut writer = HashWriter::new(file);
    writer.write(&header(
        scope,
        &session.consumed_tokens,
        engine,
        expected,
        sections,
    )?)?;
    writer.u32_section(
        TOKENS,
        0,
        &session.consumed_tokens,
        checked_bytes(session.frontier, 4)?,
    )?;
    let frontier = session.frontier;
    write_buffer(
        &mut writer,
        &mut staging,
        LOGITS,
        0,
        &session.logits_device,
        VOCAB * 2,
        frontier,
    )?;
    write_buffer(
        &mut writer,
        &mut staging,
        PLE,
        0,
        session.ple.device_history(),
        PLE_WORDS * 2,
        frontier,
    )?;
    for (index, layer) in session.layers.iter().enumerate() {
        let index = u16::try_from(index)?;
        match layer {
            LayerState::Gdn(state) => {
                let (conv, recurrent) = state.recurrent()?.buffers()?;
                write_buffer(
                    &mut writer,
                    &mut staging,
                    GDN_CONV,
                    index,
                    conv,
                    GDN_CONV_BYTES,
                    frontier,
                )?;
                write_buffer(
                    &mut writer,
                    &mut staging,
                    GDN_RECURRENT,
                    index,
                    recurrent,
                    GDN_RECURRENT_BYTES,
                    frontier,
                )?;
            }
            LayerState::Qsa(state) => {
                for (kind, (buffer, bytes)) in [QSA_COMPRESSED, QSA_PENDING]
                    .into_iter()
                    .zip(state.index.checkpoint_buffers()?)
                {
                    write_buffer(
                        &mut writer,
                        &mut staging,
                        kind,
                        index,
                        buffer,
                        bytes,
                        frontier,
                    )?;
                }
                for (kind, (buffer, bytes)) in [
                    KVARN_RECORDS,
                    KVARN_SINK_KEYS,
                    KVARN_SINK_VALUES,
                    KVARN_TAIL_KEYS,
                    KVARN_TAIL_VALUES,
                ]
                .into_iter()
                .zip(state.kv.buffers()?)
                {
                    write_buffer(
                        &mut writer,
                        &mut staging,
                        kind,
                        index,
                        buffer,
                        bytes,
                        frontier,
                    )?;
                }
            }
        }
    }
    ensure!(
        writer.written == expected - TRAILER_BYTES,
        "checkpoint writer length differs"
    );
    writer.finish()
}

pub(super) fn read_session(
    file: &mut File,
    engine: &Engine,
    scope: &[u8; 32],
    requested: &[u32],
    limits: DiskLimits,
) -> Result<Option<Session>> {
    let file_bytes = file.metadata()?.len();
    if !(HEADER_BYTES as u64 + TRAILER_BYTES..=limits.max_file_bytes).contains(&file_bytes) {
        return Err(InvalidCheckpoint("checkpoint file size is outside admission").into());
    }
    let mut reader = HashReader::new(file);
    let mut head = [0u8; HEADER_BYTES];
    reader.read(&mut head)?;
    if &head[..16] != MAGIC || field_u32(&head, 16) != WIRE_VERSION {
        return Err(InvalidCheckpoint("checkpoint wire version differs").into());
    }
    if head[20..52] != scope[..] {
        return Ok(None);
    }
    let frontier = usize::try_from(field_u64(&head, 84))?;
    let (expected, sections, _) = expected_layout(&engine.config, engine.max_context, frontier)?;
    validate_header_geometry(
        &head,
        file_bytes,
        expected,
        sections,
        engine.config.eos_token_id,
    )
    .context(InvalidCheckpoint("checkpoint header geometry differs"))?;
    if requested.len() < frontier || head[52..84] != prefix_hash(&requested[..frontier]) {
        return Ok(None);
    }
    let mut staging = Staging::new(limits.max_io_buffer_bytes)?;
    reader.section(TOKENS, 0, checked_bytes(frontier, 4)?)?;
    let mut consumed_tokens = Vec::new();
    consumed_tokens.try_reserve_exact(frontier)?;
    // Tokens become the candidate's own lineage, not a second snapshot copy.
    for offset in (0..frontier).step_by(staging.0.len() / 4) {
        let count = (frontier - offset).min(staging.0.len() / 4);
        reader.read(&mut staging.0[..count * 4])?;
        for (word, expected) in staging.0[..count * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .zip(&requested[offset..])
        {
            let token = u32::from_le_bytes(*word);
            if token >= VOCAB as u32 || token != *expected {
                return Err(InvalidCheckpoint("checkpoint token lineage differs").into());
            }
            consumed_tokens.push(token);
        }
    }
    let logits_device = read_buffer(
        &mut reader,
        &mut staging,
        &engine.metal,
        LOGITS,
        0,
        VOCAB * 2,
        VOCAB * 2,
        frontier,
    )?;
    let ple_buffer = read_buffer(
        &mut reader,
        &mut staging,
        &engine.metal,
        PLE,
        0,
        PLE_WORDS * 2,
        PLE_WORDS * 2,
        frontier,
    )?;
    let ple =
        DevicePleState::from_buffer(ple_buffer, &consumed_tokens, engine.config.eos_token_id)?;
    let mut layers = Vec::with_capacity(engine.config.layers.len());
    let compressed = frontier / COMPRESS_RATIO * INDEX_WIDTH * 2;
    let pending = frontier % COMPRESS_RATIO * INDEX_WIDTH * 2;
    let records =
        partition(frontier).1.saturating_sub(SINK_TOKENS) / TILE_TOKENS * TILE_RECORD_BYTES;
    for (index, kind) in engine.config.layers.iter().enumerate() {
        let index = u16::try_from(index)?;
        let mut read = |kind, bytes, allocation| {
            read_buffer(
                &mut reader,
                &mut staging,
                &engine.metal,
                kind,
                index,
                bytes,
                allocation,
                frontier,
            )
        };
        layers.push(match kind {
            MixerKind::Gdn => LayerState::Gdn(GdnLayerState::from_buffers(
                read(GDN_CONV, GDN_CONV_BYTES, GDN_CONV_BYTES)?,
                read(GDN_RECURRENT, GDN_RECURRENT_BYTES, GDN_RECURRENT_BYTES)?,
                frontier as u64,
            )?),
            MixerKind::Qsa => LayerState::Qsa(QsaState {
                index: DeviceQsaIndexState::from_buffers(
                    frontier as u64,
                    read(QSA_COMPRESSED, compressed, compressed.max(INDEX_WIDTH * 2))?,
                    read(QSA_PENDING, pending, COMPRESS_RATIO * INDEX_WIDTH * 2)?,
                    engine.max_context,
                )?,
                kv: DeviceKvarnState::from_buffers(
                    &engine.metal,
                    engine.max_context,
                    frontier,
                    [
                        read(KVARN_RECORDS, records, records.max(2))?,
                        read(KVARN_SINK_KEYS, SINK_BYTES, SINK_BYTES)?,
                        read(KVARN_SINK_VALUES, SINK_BYTES, SINK_BYTES)?,
                        read(KVARN_TAIL_KEYS, TAIL_BYTES, TAIL_BYTES)?,
                        read(KVARN_TAIL_VALUES, TAIL_BYTES, TAIL_BYTES)?,
                    ],
                )?,
            }),
        });
    }
    reader.finish()?;
    let session = Session {
        owner: engine.identity.clone(),
        cache_origin: engine.cache_identity.clone(),
        frontier,
        valid: true,
        layers,
        ple,
        eos_token_id: engine.config.eos_token_id,
        consumed_tokens,
        logits_device,
        last_logits_bf16: OnceCell::new(),
        producer_spans: vec![ProducerSpan::Restored { count: frontier }],
    };
    engine.validate_snapshot_source(&session)?;
    Ok(Some(session))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom};
    use std::os::unix::fs::OpenOptionsExt;
    use std::path::PathBuf;

    struct ScratchFile(PathBuf);
    impl ScratchFile {
        fn new() -> Result<(Self, File)> {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "moespresso-state-io-{}-{nonce}",
                std::process::id()
            ));
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&path)?;
            Ok((Self(path), file))
        }
    }
    impl Drop for ScratchFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn bounded_io_preserves_head_strides_before_and_after_tail_wrap() -> Result<()> {
        let metal = Metal::new()?;
        let mut staging = Staging::new((TILE_RECORD_BYTES * 2) as u64)?;
        assert_eq!(staging.0.capacity(), TILE_RECORD_BYTES * 2);
        for frontier in [1, 129, 512, 8449, 10000] {
            for kind in [KVARN_SINK_KEYS, KVARN_TAIL_VALUES] {
                let allocation = if kind == KVARN_SINK_KEYS {
                    SINK_BYTES
                } else {
                    TAIL_BYTES
                };
                let source = metal.shared_buffer(allocation)?;
                let pattern = [0x00u8, 0x3f].repeat(8192);
                for offset in (0..allocation).step_by(pattern.len()) {
                    source.write_bytes(
                        offset,
                        &pattern[..(allocation - offset).min(pattern.len())],
                    )?;
                }
                let ranges = spans(frontier, kind, allocation);
                for (head, &(start, length)) in ranges.iter().enumerate() {
                    let word = if head == 0 {
                        [0x80, 0x3f]
                    } else {
                        [0x00, 0x40]
                    };
                    let rows = word.repeat(8192);
                    for offset in (0..length).step_by(rows.len()) {
                        source.write_bytes(
                            start + offset,
                            &rows[..(length - offset).min(rows.len())],
                        )?;
                    }
                }
                let (_guard, mut file) = ScratchFile::new()?;
                let mut writer = HashWriter::new(&mut file);
                write_buffer(
                    &mut writer,
                    &mut staging,
                    kind,
                    3,
                    &source,
                    allocation,
                    frontier,
                )?;
                writer.finish()?;
                let live: usize = ranges.iter().map(|(_, len)| len).sum();
                assert_eq!(
                    file.metadata()?.len(),
                    live as u64 + SECTION_HEADER_BYTES + TRAILER_BYTES
                );
                file.seek(SeekFrom::Start(0))?;
                let mut reader = HashReader::new(&mut file);
                let restored = read_buffer(
                    &mut reader,
                    &mut staging,
                    &metal,
                    kind,
                    3,
                    allocation,
                    allocation,
                    frontier,
                )?;
                reader.finish()?;
                let mut bytes = [0u8; 1024];
                for offset in (0..allocation).step_by(bytes.len()) {
                    let count = (allocation - offset).min(bytes.len());
                    restored.read_bytes(offset, &mut bytes[..count])?;
                    for (index, word) in bytes[..count].as_chunks::<2>().0.iter().enumerate() {
                        let position = offset + index * 2;
                        let head = ranges
                            .iter()
                            .position(|&(start, len)| (start..start + len).contains(&position));
                        assert_eq!(
                            *word,
                            match head {
                                Some(0) => [0x80, 0x3f],
                                Some(_) => [0x00, 0x40],
                                None => [0, 0],
                            }
                        );
                    }
                }
                assert_eq!(staging.0.capacity(), TILE_RECORD_BYTES * 2);
            }
        }
        Ok(())
    }

    #[test]
    fn streamed_restore_rejects_nonfinite_state_even_with_valid_file_checksum() -> Result<()> {
        let metal = Metal::new()?;
        for (kind, bad) in [
            (GDN_RECURRENT, f32::NAN.to_le_bytes().to_vec()),
            (LOGITS, vec![0x80, 0x7f, 0, 0]),
        ] {
            let (_guard, mut file) = ScratchFile::new()?;
            let mut writer = HashWriter::new(&mut file);
            writer.section(kind, 0, bad.len() as u64)?;
            writer.write(&bad)?;
            writer.finish()?;
            file.seek(SeekFrom::Start(0))?;
            let mut reader = HashReader::new(&mut file);
            let mut staging = Staging::new(IO_BUFFER_BYTES)?;
            let result = read_buffer(
                &mut reader,
                &mut staging,
                &metal,
                kind,
                0,
                bad.len(),
                bad.len(),
                1,
            );
            assert!(result.is_err_and(|error| error.downcast_ref::<InvalidCheckpoint>().is_some()));
        }
        let mut records = vec![0; TILE_RECORD_BYTES];
        assert!(validate_values(KVARN_RECORDS, &records).is_ok());
        records[TILE_RECORD_BYTES - 2..]
            .copy_from_slice(&half::f16::INFINITY.to_bits().to_le_bytes());
        assert!(validate_values(KVARN_RECORDS, &records).is_err());
        Ok(())
    }
}
