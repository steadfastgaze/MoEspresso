//! Read a committed KVarN snapshot alongside an exact, uncommitted prefill chunk.

#[cfg(target_os = "macos")]
mod gpu {
    use crate::kvarn::{DeviceKvarnState, KvarnState, SINK_TOKENS};
    use crate::metal::{Batch, Binding, Buffer, Metal, Pipeline};
    use crate::qsa_index::DeviceQsaChunkSelection;
    use anyhow::{Result, ensure};

    pub(crate) fn reconstruction_fits(
        frontier: usize,
        rows: usize,
        keys: usize,
        values: usize,
    ) -> bool {
        rows > 0
            && frontier
                .checked_add(rows)
                .and_then(|n| n.checked_mul(1024))
                .is_some_and(|bytes| bytes <= keys && bytes <= values)
    }

    // Bound reconstruction by the allocated selected-row shape of indexed
    // tiles, excluding a short legacy tail. This is not an active-row count
    // or a guarantee of lower decode work or elapsed time.
    pub(crate) fn reconstruction_eligible(
        frontier: usize,
        rows: usize,
        keys: usize,
        values: usize,
    ) -> bool {
        let tail = rows % 64;
        let indexed_rows = rows - if (1..=8).contains(&tail) { tail } else { 0 };
        reconstruction_fits(frontier, rows, keys, values)
            && indexed_rows
                .checked_mul(2051)
                .zip(frontier.checked_add(rows))
                .is_some_and(|(materialized_rows, total)| total <= materialized_rows)
    }

    const SELECTED: usize = 2051;
    const ROW_BYTES: usize = 2 * SELECTED * 256 * 2;

    fn upload(metal: &Metal, bytes: &[u8]) -> Result<Buffer> {
        let buffer = metal.shared_buffer(bytes.len().max(2))?;
        if !bytes.is_empty() {
            buffer.write_bytes(0, bytes)?;
        }
        Ok(buffer)
    }

    /// Query-major selected rows `[queries, 2, 2051, 256]` and validity masks.
    pub struct PendingKvarnRows {
        pub keys: Buffer,
        pub values: Buffer,
        pub valid: Buffer,
        pub queries: usize,
    }

    impl PendingKvarnRows {
        pub fn new(metal: &Metal, capacity: usize) -> Result<Self> {
            ensure!(
                (1..=64).contains(&capacity),
                "pending KVarN tile capacity differs"
            );
            Ok(Self {
                keys: metal.shared_buffer(capacity * ROW_BYTES)?,
                values: metal.shared_buffer(capacity * ROW_BYTES)?,
                valid: metal.shared_buffer(capacity * SELECTED)?,
                queries: capacity,
            })
        }

        pub fn byte_len(&self) -> usize {
            self.keys.byte_len() + self.values.byte_len() + self.valid.byte_len()
        }
    }

    /// Borrowed token-major scratch, written before its ordered GPU consumers.
    /// Ending this Rust borrow does not retire those consumers.
    pub(crate) struct ReconstructedKvarnRows<'a> {
        pub(crate) keys: &'a Buffer,
        pub(crate) values: &'a Buffer,
        pub(crate) frontier: usize,
        pub(crate) total_rows: usize,
    }

    /// The committed storage is retained and never changed by gathers. Host
    /// reference states are uploaded once; device states share their allocations.
    /// Pending rows stay exact even when appending this chunk will seal a tile.
    pub struct PendingKvarnView {
        metal: Metal,
        frontier: usize,
        tokens: usize,
        records: Buffer,
        sink_keys: Buffer,
        sink_values: Buffer,
        tail_keys: Buffer,
        tail_values: Buffer,
        pending_keys: Buffer,
        pending_values: Buffer,
        committed_counts: Buffer,
        pending_counts: Buffer,
        query_counts: Buffer,
        gather: Pipeline,
        gather_tile: Pipeline,
        reconstruct: Pipeline,
        overlay: Pipeline,
        overlay_tile: Pipeline,
        selection_guard_tile: Pipeline,
    }

    impl PendingKvarnView {
        /// Bind session-owned committed storage without copying it.
        ///
        /// # Safety
        /// Pending buffers contain token-major BF16 rows produced before their
        /// gather consumers in the same ordered stream. The caller validates
        /// finiteness before publication and orders append after every reader.
        pub unsafe fn from_device(
            metal: &Metal,
            state: &DeviceKvarnState,
            keys: &Buffer,
            values: &Buffer,
            tokens: usize,
        ) -> Result<Self> {
            let buffers = state.buffers()?.map(|(buffer, _)| buffer.clone());
            Self::from_buffers(
                metal,
                state.frontier()?,
                state.context_limit(),
                state.record_count(),
                buffers,
                keys,
                values,
                tokens,
            )
        }

        /// Create a snapshot view without publishing pending tokens.
        ///
        /// # Safety
        /// Pending buffers contain finite BF16 `[tokens,2,256]` rows. Their
        /// producer must complete before gather commands read them; the command
        /// owner excludes concurrent mutation until all gathers complete.
        pub unsafe fn new(
            metal: &Metal,
            state: &KvarnState,
            keys: &Buffer,
            values: &Buffer,
            tokens: usize,
        ) -> Result<Self> {
            Self::validate_pending(
                state.frontier(),
                state.context_limit(),
                keys,
                values,
                tokens,
            )?;
            let tail = |bytes: &[u8]| {
                if state.frontier() <= SINK_TOKENS {
                    metal.shared_buffer(2)
                } else {
                    upload(metal, bytes)
                }
            };
            let buffers = [
                upload(metal, state.records())?,
                upload(metal, state.sink_keys())?,
                upload(metal, state.sink_values())?,
                tail(state.tail_keys())?,
                tail(state.tail_values())?,
            ];
            Self::from_buffers(
                metal,
                state.frontier(),
                state.context_limit(),
                state.record_count(),
                buffers,
                keys,
                values,
                tokens,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn from_buffers(
            metal: &Metal,
            frontier: usize,
            context: usize,
            record_count: usize,
            buffers: [Buffer; 5],
            keys: &Buffer,
            values: &Buffer,
            tokens: usize,
        ) -> Result<Self> {
            Self::validate_pending(frontier, context, keys, values, tokens)?;
            let [records, sink_keys, sink_values, tail_keys, tail_values] = buffers;
            let gather = metal.compile_msl_strict(
                include_str!("../kernels/metal/kvarn_decode.metal"),
                "kvarn_gather",
            )?;
            let overlay = metal.compile_msl_strict(
                include_str!("../kernels/metal/kvarn_pending.metal"),
                "kvarn_pending_overlay",
            )?;
            let gather_tile = metal.compile_msl_strict(
                include_str!("../kernels/metal/kvarn_decode.metal"),
                "kvarn_gather_tile",
            )?;
            let reconstruct = metal.compile_msl_strict(
                include_str!("../kernels/metal/kvarn_decode.metal"),
                "kvarn_reconstruct_prefill",
            )?;
            let overlay_tile = metal.compile_msl_strict(
                include_str!("../kernels/metal/kvarn_pending.metal"),
                "kvarn_pending_overlay_tile",
            )?;
            let selection_guard_tile = metal.compile_msl_strict(
                include_str!("../kernels/metal/kvarn_pending.metal"),
                "kvarn_pending_selection_guard_tile",
            )?;
            ensure!(
                gather.max_threads_per_threadgroup() >= 256
                    && overlay.max_threads_per_threadgroup() >= 256
                    && gather_tile.max_threads_per_threadgroup() >= 256
                    && overlay_tile.max_threads_per_threadgroup() >= 256
                    && selection_guard_tile.max_threads_per_threadgroup() >= 128,
                "pending KVarN requires 256-thread groups"
            );
            let committed_counts = [frontier as u32, record_count as u32, SELECTED as u32];
            let visible = frontier + tokens;
            let expected_selection = if visible <= SELECTED {
                visible
            } else {
                2048 + visible % 4
            };
            let pending_counts = [frontier as u32, tokens as u32, expected_selection as u32];
            let mut query_counts = Vec::with_capacity(tokens * 12);
            for row in 0..tokens {
                let visible = frontier + row + 1;
                let selected = if visible <= SELECTED {
                    visible
                } else {
                    2048 + visible % 4
                };
                for count in [frontier, row + 1, selected] {
                    query_counts.extend_from_slice(&u32::try_from(count)?.to_le_bytes());
                }
            }
            Ok(Self {
                metal: metal.clone(),
                frontier,
                tokens,
                records,
                sink_keys,
                sink_values,
                tail_keys,
                tail_values,
                pending_keys: keys.clone(),
                pending_values: values.clone(),
                committed_counts: upload(
                    metal,
                    &committed_counts
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?,
                pending_counts: upload(
                    metal,
                    &pending_counts
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?,
                query_counts: upload(metal, &query_counts)?,
                gather,
                gather_tile,
                reconstruct,
                overlay,
                overlay_tile,
                selection_guard_tile,
            })
        }

        fn validate_pending(
            frontier: usize,
            context: usize,
            keys: &Buffer,
            values: &Buffer,
            tokens: usize,
        ) -> Result<()> {
            ensure!(
                (1..=crate::prefill_capacity::MAX_ROWS).contains(&tokens),
                "pending KVarN chunk exceeds supported prefill capacity"
            );
            ensure!(
                frontier.checked_add(tokens).is_some_and(|n| n <= context),
                "pending KVarN exceeds model context"
            );
            ensure!(
                keys.byte_len() >= tokens * 1024 && values.byte_len() >= tokens * 1024,
                "pending KVarN extent differs"
            );
            Ok(())
        }

        /// Gather a bounded tile of consecutive causal query selections.
        pub fn encode_rows(
            &self,
            batch: &mut Batch,
            ids: &[i32],
            query_offset: usize,
        ) -> Result<PendingKvarnRows> {
            ensure!(
                !ids.is_empty() && ids.len().is_multiple_of(SELECTED),
                "pending KVarN selection shape differs"
            );
            let queries = ids.len() / SELECTED;
            ensure!(queries <= 64, "pending KVarN query tile exceeds chunk");
            let keys = self.metal.shared_buffer(queries * ROW_BYTES)?;
            let values = self.metal.shared_buffer(queries * ROW_BYTES)?;
            let valid = self.metal.shared_buffer(ids.len())?;
            self.encode_rows_into(batch, ids, query_offset, &keys, &values, &valid)?;
            Ok(PendingKvarnRows {
                keys,
                values,
                valid,
                queries,
            })
        }

        /// Guard and gather one device-selected multiquery tile. Each shader
        /// output index addresses one ushort, so the work grid is half the
        /// validated byte extent of either selected K/V output.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn encode_selection_guard(
            &self,
            batch: &mut Batch,
            selection: &DeviceQsaChunkSelection,
            tile_offset: usize,
            rows: usize,
            selected: &PendingKvarnRows,
            effective_valid: &Buffer,
            safe_counts: &Buffer,
        ) -> Result<Buffer> {
            ensure!(
                (1..=64).contains(&rows)
                    && rows <= selected.queries
                    && tile_offset
                        .checked_add(rows)
                        .is_some_and(|end| end <= self.tokens && end <= selection.rows),
                "pending KVarN device tile exceeds chunk"
            );
            let output_bytes = rows
                .checked_mul(ROW_BYTES)
                .ok_or_else(|| anyhow::anyhow!("pending KVarN tile extent overflow"))?;
            ensure!(
                selection.ids.byte_len() >= selection.rows * SELECTED * 4
                    && selection.valid.byte_len() >= selection.rows * SELECTED
                    && selection.count.byte_len() >= selection.rows * 4
                    && selection.status.byte_len() >= 4
                    && selected.keys.byte_len() >= output_bytes
                    && selected.values.byte_len() >= output_bytes
                    && effective_valid.byte_len() >= rows * SELECTED
                    && safe_counts.byte_len() >= rows * 4,
                "pending KVarN device tile geometry differs"
            );
            let count = upload(&self.metal, &u32::try_from(rows)?.to_le_bytes())?;
            let ids_offset = tile_offset * SELECTED * 4;
            let guard = [
                Binding {
                    buffer: &selection.ids,
                    offset: ids_offset,
                },
                Binding {
                    buffer: &selection.valid,
                    offset: tile_offset * SELECTED,
                },
                Binding {
                    buffer: &selection.count,
                    offset: tile_offset * 4,
                },
                Binding {
                    buffer: &self.query_counts,
                    offset: tile_offset * 12,
                },
                Binding {
                    buffer: effective_valid,
                    offset: 0,
                },
                Binding {
                    buffer: safe_counts,
                    offset: 0,
                },
                Binding {
                    buffer: &selection.status,
                    offset: 0,
                },
                Binding {
                    buffer: &count,
                    offset: 0,
                },
            ];
            // SAFETY: The guard checks every selector before any address use.
            unsafe {
                batch.dispatch_1d(&self.selection_guard_tile, &guard, rows * SELECTED, 128)?;
            }
            Ok(count)
        }

        #[allow(clippy::too_many_arguments)]
        pub(crate) fn encode_tile_device_into(
            &self,
            batch: &mut Batch,
            selection: &DeviceQsaChunkSelection,
            tile_offset: usize,
            rows: usize,
            selected: &PendingKvarnRows,
            effective_valid: &Buffer,
            safe_counts: &Buffer,
        ) -> Result<()> {
            let count = self.encode_selection_guard(
                batch,
                selection,
                tile_offset,
                rows,
                selected,
                effective_valid,
                safe_counts,
            )?;
            let output_bytes = rows * ROW_BYTES;
            let ids_offset = tile_offset * SELECTED * 4;
            let gather = [
                (&self.records, 0),
                (&self.sink_keys, 0),
                (&self.sink_values, 0),
                (&self.tail_keys, 0),
                (&self.tail_values, 0),
                (&selection.ids, ids_offset),
                (effective_valid, 0),
                (&self.committed_counts, 0),
                (&selected.keys, 0),
                (&selected.values, 0),
                (&count, 0),
            ]
            .map(|(buffer, offset)| Binding { buffer, offset });
            let overlay = [
                (&self.pending_keys, 0),
                (&self.pending_values, 0),
                (&selection.ids, ids_offset),
                (effective_valid, 0),
                (&self.query_counts, tile_offset * 12),
                (&selected.keys, 0),
                (&selected.values, 0),
                (&count, 0),
            ]
            .map(|(buffer, offset)| Binding { buffer, offset });
            // SAFETY: The guard validates every ID before either decoder reads
            // it. The output grid contains exactly output_bytes / 2 ushorts.
            unsafe {
                batch.dispatch_1d(&self.gather_tile, &gather, output_bytes / 2, 256)?;
                batch.dispatch_1d(&self.overlay_tile, &overlay, output_bytes / 2, 256)?;
            }
            Ok(())
        }

        pub(crate) fn encode_reconstruct<'a>(
            &self,
            batch: &mut Batch,
            scratch: &'a PendingKvarnRows,
        ) -> Result<Option<ReconstructedKvarnRows<'a>>> {
            if !reconstruction_eligible(
                self.frontier,
                self.tokens,
                scratch.keys.byte_len(),
                scratch.values.byte_len(),
            ) {
                return Ok(None);
            }
            let shape = [
                u32::try_from(self.frontier)?,
                u32::try_from(
                    crate::kvarn::partition(self.frontier)
                        .1
                        .saturating_sub(SINK_TOKENS)
                        / crate::kvarn::TILE_TOKENS,
                )?,
                u32::try_from(self.tokens)?,
            ];
            let total = self
                .frontier
                .checked_add(self.tokens)
                .ok_or_else(|| anyhow::anyhow!("reconstruction frontier overflow"))?;
            u32::try_from(total)?;
            let threads = total
                .checked_mul(512)
                .ok_or_else(|| anyhow::anyhow!("reconstruction grid overflow"))?;
            let shape = upload(
                &self.metal,
                &shape
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let bindings = [
                &self.records,
                &self.sink_keys,
                &self.sink_values,
                &self.tail_keys,
                &self.tail_values,
                &self.pending_keys,
                &self.pending_values,
                &shape,
                &scratch.keys,
                &scratch.values,
            ]
            .map(|buffer| Binding { buffer, offset: 0 });
            // SAFETY: One group owns one token/head. Source extents and the
            // committed partition were validated by the view; scratch covers
            // the entire chunk, including exact pending rows.
            unsafe {
                batch.dispatch_1d(&self.reconstruct, &bindings, threads, 256)?;
            }
            Ok(Some(ReconstructedKvarnRows {
                keys: &scratch.keys,
                values: &scratch.values,
                frontier: self.frontier,
                total_rows: total,
            }))
        }

        /// Gather into reusable query-major scratch owned by the caller. The
        /// host-written validity buffer must have no outstanding command users.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn encode_rows_into(
            &self,
            batch: &mut Batch,
            ids: &[i32],
            query_offset: usize,
            keys: &Buffer,
            values: &Buffer,
            valid: &Buffer,
        ) -> Result<()> {
            ensure!(
                !ids.is_empty() && ids.len().is_multiple_of(SELECTED),
                "pending KVarN selection shape differs"
            );
            let queries = ids.len() / SELECTED;
            ensure!(
                keys.byte_len() == queries * ROW_BYTES
                    && values.byte_len() == queries * ROW_BYTES
                    && valid.byte_len() == ids.len(),
                "pending KVarN gather output extent differs"
            );
            ensure!(
                queries <= 64
                    && query_offset
                        .checked_add(queries)
                        .is_some_and(|n| n <= self.tokens),
                "pending KVarN query tile exceeds chunk"
            );
            let mut flags = Vec::with_capacity(ids.len());
            for (query, row) in ids.as_chunks::<SELECTED>().0.iter().enumerate() {
                let visible = self.frontier + query_offset + query + 1;
                let mut previous = -1;
                let mut padded = false;
                for &id in row {
                    if id == -1 {
                        padded = true;
                        flags.push(0);
                        continue;
                    }
                    ensure!(
                        !padded && id >= 0 && id > previous && (id as usize) < visible,
                        "pending KVarN selection must be an ascending causal set with -1 padding"
                    );
                    previous = id;
                    flags.push(1);
                }
                ensure!(previous >= 0, "pending KVarN query has no visible rows");
            }
            let ids = upload(
                &self.metal,
                &ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            )?;
            valid.write_bytes(0, &flags)?;
            // SAFETY: Validated IDs are bounded by the exact pending extent and
            // committed state. Each query owns disjoint output and mask slices.
            unsafe {
                for query in 0..queries {
                    let id_offset = query * SELECTED * 4;
                    let flag_offset = query * SELECTED;
                    let output_offset = query * ROW_BYTES;
                    let bindings = [
                        (&self.records, 0),
                        (&self.sink_keys, 0),
                        (&self.sink_values, 0),
                        (&self.tail_keys, 0),
                        (&self.tail_values, 0),
                        (&ids, id_offset),
                        (valid, flag_offset),
                        (&self.committed_counts, 0),
                        (keys, output_offset),
                        (values, output_offset),
                    ]
                    .map(|(buffer, offset)| Binding { buffer, offset });
                    batch.dispatch_1d(&self.gather, &bindings, SELECTED * 2 * 256, 256)?;
                    let bindings = [
                        (&self.pending_keys, 0),
                        (&self.pending_values, 0),
                        (&ids, id_offset),
                        (valid, flag_offset),
                        (&self.pending_counts, 0),
                        (keys, output_offset),
                        (values, output_offset),
                    ]
                    .map(|(buffer, offset)| Binding { buffer, offset });
                    batch.dispatch_1d(&self.overlay, &bindings, SELECTED * 2 * 256, 256)?;
                }
            }
            Ok(())
        }
    }
}

#[cfg(target_os = "macos")]
pub use gpu::{PendingKvarnRows, PendingKvarnView};
#[cfg(target_os = "macos")]
pub(crate) use gpu::{ReconstructedKvarnRows, reconstruction_fits};

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::kvarn::{
        DeviceKvarnState, HEAD_DIM, KV_HEADS, SINK_TOKENS, TAIL_CAPACITY, TOKEN_BYTES,
    };
    use crate::metal::{Buffer, Metal};
    use crate::qsa_index::DeviceQsaChunkSelection;
    use anyhow::Result;
    use half::bf16;

    #[test]
    fn reconstruction_work_and_capacity_boundaries() {
        use super::gpu::reconstruction_eligible as eligible;
        let each = 64 * 2 * 2051 * 256 * 2;
        assert!(eligible(14336, 1024, each, each));
        assert!(eligible(14336, 64, each, each));
        assert!(!eligible(128000, 16, each, each));
        assert!(eligible(16 * 2051 - 16, 16, each, each));
        assert!(!eligible(16 * 2051 - 15, 16, each, each));
        assert!(eligible(131200, 64, each, each));
        assert!(!eligible(131201, 64, each, each));
        assert!(!eligible(262080, 64, each, each));
        assert!(!eligible(131200, 64, each - 1, each));
        assert!(!eligible(usize::MAX, 64, each, each));
        assert!(eligible(131199, 65, each, each));
        assert!(!eligible(131200, 65, usize::MAX, usize::MAX));
        for rows in 0..=8 {
            assert!(!eligible(0, rows, each, each));
        }
    }

    fn word(token: usize, head: usize, channel: usize, salt: usize) -> u16 {
        let code = ((token * 17 + head * 97 + channel * 7 + salt) % 257) as i32 - 128;
        bf16::from_f32(code as f32 / 256.0).to_bits()
    }

    #[test]
    fn host_selection_rejects_invalid_queries_and_blocks_unfinished_readback() -> Result<()> {
        const SELECTED: usize = 2051;
        let metal = Metal::new()?;
        let state = crate::kvarn::KvarnState::zero(2)?;
        let pending = |salt| -> Result<Buffer> {
            let bytes: Vec<_> = (0..2)
                .flat_map(|token| {
                    (0..KV_HEADS).flat_map(move |head| {
                        (0..HEAD_DIM)
                            .flat_map(move |channel| word(token, head, channel, salt).to_le_bytes())
                    })
                })
                .collect();
            let buffer = metal.shared_buffer(bytes.len())?;
            buffer.write_bytes(0, &bytes)?;
            Ok(buffer)
        };
        let keys = pending(3)?;
        let values = pending(53)?;
        // SAFETY: Completed finite token-major rows are immutable during gather.
        let view = unsafe { PendingKvarnView::new(&metal, &state, &keys, &values, 2)? };
        let mut ids = vec![-1; SELECTED];
        ids[0] = 0;
        let mut rejected = metal.batch()?;
        assert!(view.encode_rows(&mut rejected, &ids, 2).is_err());
        ids[0] = 1;
        assert!(view.encode_rows(&mut rejected, &ids, 0).is_err());
        ids[0] = -1;
        assert!(view.encode_rows(&mut rejected, &ids, 0).is_err());
        ids[0] = 0;
        ids[1] = 0;
        assert!(view.encode_rows(&mut rejected, &ids, 1).is_err());
        ids[1] = -1;
        ids[2] = 1;
        assert!(view.encode_rows(&mut rejected, &ids, 1).is_err());
        drop(rejected);
        ids[2] = -1;
        let mut batch = metal.batch()?;
        let rows = view.encode_rows(&mut batch, &ids, 0)?;
        assert_eq!(rows.queries, 1);
        for buffer in [&rows.keys, &rows.values, &rows.valid] {
            assert!(buffer.read_bytes(0, &mut []).is_err());
        }
        batch.submit().wait()?;
        let mut valid = vec![0; SELECTED];
        rows.valid.read_bytes(0, &mut valid)?;
        assert_eq!(valid[0], 1);
        assert!(valid[1..].iter().all(|&flag| flag == 0));
        for (buffer, salt) in [(&rows.keys, 3), (&rows.values, 53)] {
            let mut bytes = vec![0; buffer.byte_len()];
            buffer.read_bytes(0, &mut bytes)?;
            for head in 0..KV_HEADS {
                for slot in 0..SELECTED {
                    for channel in 0..HEAD_DIM {
                        let at = ((head * SELECTED + slot) * HEAD_DIM + channel) * 2;
                        let expected = if slot == 0 {
                            word(0, head, channel, salt)
                        } else {
                            0
                        };
                        assert_eq!(u16::from_le_bytes(bytes[at..at + 2].try_into()?), expected);
                    }
                }
            }
        }
        assert_eq!(state.frontier(), 0);
        assert_eq!(state.record_count(), 0);
        Ok(())
    }

    #[test]
    fn wide_device_tiles_match_host_bytes_and_exclude_future_pending_rows() -> Result<()> {
        const ROWS: usize = 2048;
        const FRONTIER: usize = 8192;
        const SELECTED: usize = 2051;
        const ROW_BYTES: usize = KV_HEADS * SELECTED * HEAD_DIM * 2;
        let metal = Metal::new()?;
        let upload = |data: &[u8]| -> Result<Buffer> {
            let out = metal.shared_buffer(data.len())?;
            out.write_bytes(0, data)?;
            Ok(out)
        };
        let committed = |rows: usize, start: usize, salt| -> Vec<u8> {
            (0..KV_HEADS)
                .flat_map(|head| {
                    (0..rows).flat_map(move |row| {
                        (0..HEAD_DIM).flat_map(move |channel| {
                            word(start + row, head, channel, salt).to_le_bytes()
                        })
                    })
                })
                .collect()
        };
        let sink_k = committed(SINK_TOKENS, 0, 3);
        let sink_v = committed(SINK_TOKENS, 0, 53);
        let tail_k = committed(TAIL_CAPACITY, SINK_TOKENS, 3);
        let tail_v = committed(TAIL_CAPACITY, SINK_TOKENS, 53);
        let state = DeviceKvarnState::restore(
            &metal,
            FRONTIER + ROWS,
            FRONTIER,
            &[],
            &sink_k,
            &sink_v,
            &tail_k,
            &tail_v,
        )?;
        let pending = |salt| -> Vec<u8> {
            (0..ROWS)
                .flat_map(|row| {
                    (0..KV_HEADS).flat_map(move |head| {
                        (0..HEAD_DIM).flat_map(move |channel| {
                            word(FRONTIER + row, head, channel, salt).to_le_bytes()
                        })
                    })
                })
                .collect()
        };
        let pending_k = pending(3);
        let pending_v = pending(53);
        let key = upload(&pending_k)?;
        let value = upload(&pending_v)?;
        // SAFETY: Completed finite token-major rows; this test never appends to state.
        let view = unsafe { PendingKvarnView::from_device(&metal, &state, &key, &value, ROWS)? };
        let mut ids = vec![-1i32; ROWS * SELECTED];
        let mut valid = vec![0u8; ROWS * SELECTED];
        let mut counts = Vec::with_capacity(ROWS);
        for query in 0..ROWS {
            let visible = FRONTIER + query + 1;
            // A canonical sparse selection:511 early groups, the last complete
            // group, and the causal uncompressed remainder. It covers sinks,
            // committed tails and the current exact pending row.
            let selected: Vec<_> = (0..2044)
                .chain((visible / 4 - 1) * 4..visible / 4 * 4)
                .chain(visible / 4 * 4..visible)
                .collect();
            counts.push(selected.len() as u32);
            for (slot, id) in selected.into_iter().enumerate() {
                ids[query * SELECTED + slot] = id as i32;
                valid[query * SELECTED + slot] = 1;
            }
        }
        let selection = DeviceQsaChunkSelection {
            ids: upload(&ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?,
            valid: upload(&valid)?,
            count: upload(
                &counts
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?,
            status: upload(&0u32.to_le_bytes())?,
            rows: ROWS,
        };
        let selected = PendingKvarnRows::new(&metal, 64)?;
        let effective = metal.shared_buffer(64 * SELECTED)?;
        let safe = metal.shared_buffer(64 * 4)?;
        {
            let total = FRONTIER + ROWS;
            let sentinel = [0x35, 0x71, 0x26, 0x64];
            selected.keys.write_bytes(total * TOKEN_BYTES, &sentinel)?;
            selected
                .values
                .write_bytes(total * TOKEN_BYTES, &sentinel)?;
            let mut batch = metal.batch()?;
            let raw = view.encode_reconstruct(&mut batch, &selected)?.unwrap();
            assert_eq!((raw.frontier, raw.total_rows), (FRONTIER, total));
            assert!(raw.keys.read_bytes(0, &mut [0; 2]).is_err());
            batch.submit().wait()?;
            for (buffer, salt) in [(raw.keys, 3), (raw.values, 53)] {
                let mut actual = vec![0; total * TOKEN_BYTES];
                buffer.read_bytes(0, &mut actual)?;
                for (index, value) in actual.as_chunks::<2>().0.iter().enumerate() {
                    let channel = index % HEAD_DIM;
                    let head = index / HEAD_DIM % KV_HEADS;
                    let token = index / (HEAD_DIM * KV_HEADS);
                    assert_eq!(u16::from_le_bytes(*value), word(token, head, channel, salt));
                }
                let mut after = [0; 4];
                buffer.read_bytes(total * TOKEN_BYTES, &mut after)?;
                assert_eq!(after, sentinel);
            }
        }
        let run = |view: &PendingKvarnView, tile: usize| -> Result<u32> {
            selection.status.write_bytes(0, &0u32.to_le_bytes())?;
            let mut batch = metal.batch()?;
            view.encode_tile_device_into(
                &mut batch, &selection, tile, 64, &selected, &effective, &safe,
            )?;
            batch.submit().wait()?;
            let mut status = [0; 4];
            selection.status.read_bytes(0, &mut status)?;
            Ok(u32::from_le_bytes(status))
        };
        let read_query = |query: usize| -> Result<(Vec<u8>, Vec<u8>)> {
            let local = query % 64;
            let mut key = vec![0; ROW_BYTES];
            let mut value = vec![0; ROW_BYTES];
            selected.keys.read_bytes(local * ROW_BYTES, &mut key)?;
            selected.values.read_bytes(local * ROW_BYTES, &mut value)?;
            Ok((key, value))
        };
        for tile in [0, 64, 448, 512, 960, 1024, 1984] {
            assert_eq!(run(&view, tile)?, 0);
            for query in [tile, tile + 63] {
                let actual = read_query(query)?;
                let mut mask = vec![0; SELECTED];
                let mut count = [0; 4];
                effective.read_bytes((query % 64) * SELECTED, &mut mask)?;
                safe.read_bytes((query % 64) * 4, &mut count)?;
                assert_eq!(u32::from_le_bytes(count), counts[query]);
                assert_eq!(mask, valid[query * SELECTED..(query + 1) * SELECTED]);
                for (bytes, salt) in [(&actual.0, 3), (&actual.1, 53)] {
                    for head in 0..KV_HEADS {
                        for slot in 0..SELECTED {
                            for channel in 0..HEAD_DIM {
                                let id = ids[query * SELECTED + slot];
                                let expected = if id < 0 {
                                    0
                                } else {
                                    word(id as usize, head, channel, salt)
                                };
                                let at = ((head * SELECTED + slot) * HEAD_DIM + channel) * 2;
                                assert_eq!(
                                    u16::from_le_bytes(bytes[at..at + 2].try_into()?),
                                    expected,
                                    "query={query} head={head} slot={slot} channel={channel}"
                                );
                            }
                        }
                    }
                }
            }
            // Change every pending row after the first query of this tile.
            // The later query must observe the mutation while the first remains exact.
            let before = read_query(tile)?;
            let later_before = read_query(tile + 63)?;
            let mut changed_k = pending_k.clone();
            let mut changed_v = pending_v.clone();
            let begin = (tile + 1) * TOKEN_BYTES;
            for data in [&mut changed_k, &mut changed_v] {
                for bytes in data[begin..].as_chunks_mut::<2>().0 {
                    bytes.copy_from_slice(&0x3f00u16.to_le_bytes());
                }
            }
            let changed_k = upload(&changed_k)?;
            let changed_v = upload(&changed_v)?;
            // SAFETY: Completed finite altered rows; state and selections remain immutable.
            let changed = unsafe {
                PendingKvarnView::from_device(&metal, &state, &changed_k, &changed_v, ROWS)?
            };
            assert_eq!(run(&changed, tile)?, 0);
            assert_eq!(
                read_query(tile)?,
                before,
                "future pending rows leaked into query={tile}"
            );
            assert_ne!(
                read_query(tile + 63)?,
                later_before,
                "mutated pending rows were not consumed by query={}",
                tile + 63
            );
        }
        // A future ID inside the allocated pending buffer must still be rejected.
        let query = 512;
        let slot = counts[query] as usize - 1;
        selection.ids.write_bytes(
            (query * SELECTED + slot) * 4,
            &((FRONTIER + query + 1) as i32).to_le_bytes(),
        )?;
        assert_ne!(run(&view, query)?, 0);
        let mut flag = [1];
        effective.read_bytes(slot, &mut flag)?;
        assert_eq!(flag, [0]);
        let invalid = read_query(query)?;
        for data in [&invalid.0, &invalid.1] {
            for head in 0..KV_HEADS {
                let at = (head * SELECTED + slot) * HEAD_DIM * 2;
                assert!(data[at..at + HEAD_DIM * 2].iter().all(|&byte| byte == 0));
            }
        }
        assert_eq!(state.frontier()?, FRONTIER);
        assert_eq!(state.record_count(), 0);
        Ok(())
    }
}
