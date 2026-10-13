use crate::iqk::{Geometry, IqkGemv};
use crate::metal::{Buffer, Metal};
use crate::package::Projection;
use anyhow::{Context, Result, ensure};
use std::ops::Range;

pub(super) const CHUNK_SLOTS: usize = 16;
pub(super) const MAX_BOUNDED_SLOTS: usize = 511;
// Startup needs at most 32 slabs. Aligned growth can split a partial startup
// tail once, so release and regrowth need at most 33 descriptors. Arbitrary
// smaller growth requests still fail before exceeding the fixed segment limit.
pub(super) const MAX_CHUNKS: usize = 2 * 512usize.div_ceil(16);

fn release_prefix(capacity: usize, target: usize) -> Result<()> {
    ensure!(
        target > 0 && target.is_multiple_of(CHUNK_SLOTS) && target < capacity,
        "expert release target must be a smaller positive whole-chunk prefix"
    );
    Ok(())
}

fn growth_target(capacity: usize, target: usize) -> Result<()> {
    ensure!(
        target > capacity && target <= MAX_BOUNDED_SLOTS,
        "expert growth target must be a larger bounded capacity below 512"
    );
    Ok(())
}

fn growth_ranges(capacity: usize, target: usize) -> impl Iterator<Item = Range<usize>> {
    let mut start = capacity;
    std::iter::from_fn(move || {
        if start >= target {
            return None;
        }
        let end = ((start / CHUNK_SLOTS + 1) * CHUNK_SLOTS).min(target);
        let range = start..end;
        start = end;
        Some(range)
    })
}

fn slab_layout(geometries: [Geometry; 3], slots: usize) -> Result<([[usize; 6]; 3], usize)> {
    ensure!(
        (1..=CHUNK_SLOTS).contains(&slots),
        "invalid fixed chunk slot extent"
    );
    let mut offsets = [[0usize; 6]; 3];
    let mut bytes = 0usize;
    for (projection, geometry) in geometries.into_iter().enumerate() {
        for (stream, slot_bytes) in IqkGemv::slot_stream_bytes(geometry)?
            .into_iter()
            .enumerate()
        {
            if slot_bytes == 0 {
                continue;
            }
            bytes = bytes.checked_add(15).context("chunk alignment overflow")? & !15;
            offsets[projection][stream] = bytes;
            bytes = bytes
                .checked_add(
                    slots
                        .checked_mul(slot_bytes)
                        .context("chunk stream size overflow")?,
                )
                .context("chunk size overflow")?;
        }
    }
    Ok((offsets, bytes))
}

struct Chunk {
    range: Range<usize>,
    buffer: Buffer,
    projections: [IqkGemv; 3],
}

fn allocate_chunk(
    metal: &Metal,
    geometries: [Geometry; 3],
    range: Range<usize>,
    tables: [&Buffer; 3],
) -> Result<Chunk> {
    let slots = range.len();
    let (offsets, bytes) = slab_layout(geometries, slots)?;
    let buffer = metal.shared_buffer(bytes)?;
    let projections = [
        IqkGemv::slab_view(metal, geometries[0], slots, &buffer, offsets[0], tables[0])?,
        IqkGemv::slab_view(metal, geometries[1], slots, &buffer, offsets[1], tables[1])?,
        IqkGemv::slab_view(metal, geometries[2], slots, &buffer, offsets[2], tables[2])?,
    ];
    Ok(Chunk {
        range,
        buffer,
        projections,
    })
}

fn write_chunk_addresses(
    chunk: &Chunk,
    geometries: [Geometry; 3],
    addresses: &Buffer,
) -> Result<()> {
    for slot in chunk.range.clone() {
        let local = slot - chunk.range.start;
        let mut row = Vec::with_capacity(18 * 8);
        for (projection, geometry) in chunk.projections.iter().zip(geometries) {
            let bindings = projection.stream_bindings();
            for (binding, bytes) in bindings
                .into_iter()
                .take(6)
                .zip(IqkGemv::slot_stream_bytes(geometry)?)
            {
                let offset = binding
                    .offset
                    .checked_add(
                        local
                            .checked_mul(bytes)
                            .context("slot stream offset overflow")?,
                    )
                    .context("slot address offset overflow")?;
                row.extend_from_slice(
                    &binding
                        .buffer
                        .gpu_address(offset, bytes.max(1))?
                        .to_le_bytes(),
                );
            }
        }
        addresses.write_bytes(slot * 18 * 8, &row)?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) struct ChunkGrowthQuote {
    pub(super) added_requested_bytes: u64,
    pub(super) preparation_requested_bytes: u64,
}

pub(super) struct PreparedChunksGrowth {
    chunks: Vec<Chunk>,
    addresses: Buffer,
    target: usize,
    pub(super) quote: ChunkGrowthQuote,
}

pub(super) struct FixedChunks {
    chunks: Vec<Chunk>,
    capacity: usize,
    geometries: [Geometry; 3],
    pub(super) addresses: Buffer,
    pub(super) ready: Buffer,
}

fn projection_index(kind: Projection) -> usize {
    match kind {
        Projection::Gate => 0,
        Projection::Up => 1,
        Projection::Down => 2,
    }
}

impl FixedChunks {
    pub(super) fn new(metal: &Metal, geometries: [Geometry; 3], capacity: usize) -> Result<Self> {
        ensure!(
            (10..512).contains(&capacity),
            "fixed chunks require bounded expert capacity"
        );
        let tables = geometries
            .into_iter()
            .map(|geometry| {
                let bytes = geometry.codec().value_table_fp16();
                let table = metal.shared_buffer(bytes.len())?;
                table.write_bytes(0, &bytes)?;
                Ok(table)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut chunks = Vec::new();
        chunks.try_reserve_exact(MAX_CHUNKS)?;
        for start in (0..capacity).step_by(CHUNK_SLOTS) {
            let slots = (capacity - start).min(CHUNK_SLOTS);
            chunks.push(allocate_chunk(
                metal,
                geometries,
                start..start + slots,
                [&tables[0], &tables[1], &tables[2]],
            )?);
        }
        let addresses = metal.shared_buffer(
            MAX_BOUNDED_SLOTS
                .checked_mul(18 * 8)
                .context("chunk address size overflow")?,
        )?;
        for chunk in &chunks {
            write_chunk_addresses(chunk, geometries, &addresses)?;
        }
        let ready = metal.shared_buffer(4)?;
        ready.write_bytes(0, &1u32.to_le_bytes())?;
        Ok(Self {
            chunks,
            capacity,
            geometries,
            addresses,
            ready,
        })
    }

    pub(super) fn count(&self) -> usize {
        self.chunks.len()
    }

    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(super) fn range(&self, chunk: usize) -> Result<Range<usize>> {
        Ok(self
            .chunks
            .get(chunk)
            .context("expert chunk is out of range")?
            .range
            .clone())
    }

    pub(super) fn location(&self, slot: usize) -> Result<(usize, usize)> {
        ensure!(
            slot < self.capacity,
            "expert slot exceeds fixed chunk capacity"
        );
        let chunk = self.chunks.partition_point(|chunk| chunk.range.end <= slot);
        let range = &self
            .chunks
            .get(chunk)
            .context("expert slot is outside chunk ranges")?
            .range;
        ensure!(range.contains(&slot), "expert slot is outside chunk ranges");
        Ok((chunk, slot - range.start))
    }

    pub(super) fn projection(&self, chunk: usize, kind: Projection) -> Result<&IqkGemv> {
        Ok(&self
            .chunks
            .get(chunk)
            .context("expert chunk is out of range")?
            .projections[projection_index(kind)])
    }

    pub(super) fn check_writable(&self) -> Result<()> {
        for chunk in &self.chunks {
            chunk.buffer.read_bytes(0, &mut [])?;
        }
        Ok(())
    }

    pub(super) fn buffers(&self) -> impl Iterator<Item = &Buffer> {
        self.chunks.iter().map(|chunk| &chunk.buffer)
    }

    pub(super) fn check_quiescent(&self) -> Result<()> {
        self.check_writable()?;
        self.addresses.read_bytes(0, &mut [])?;
        self.ready.read_bytes(0, &mut [])?;
        for projection in &self.chunks[0].projections {
            projection.stream_bindings()[6]
                .buffer
                .read_bytes(0, &mut [])?;
        }
        Ok(())
    }

    pub(super) fn prepare_suffix(&self, target: usize) -> Result<(usize, u64)> {
        release_prefix(self.capacity, target)?;
        let keep = self
            .chunks
            .iter()
            .position(|chunk| chunk.range.end == target)
            .context("expert release target is not a complete chunk boundary")?
            + 1;
        let bytes = self.chunks[keep..].iter().try_fold(0u64, |total, chunk| {
            total
                .checked_add(u64::try_from(chunk.buffer.byte_len())?)
                .context("released expert byte count overflow")
        })?;
        Ok((keep, bytes))
    }

    // The exclusive prepared pool borrow proves the prefix and quiescence.
    // Metadata covers every bounded capacity; dispatch guards the active prefix.
    pub(super) fn commit_suffix(&mut self, keep: usize, target: usize) {
        self.chunks.truncate(keep);
        self.capacity = target;
    }

    pub(super) fn quote_growth(&self, target: usize) -> Result<ChunkGrowthQuote> {
        ensure!(
            self.addresses.byte_len().is_multiple_of(18 * 8),
            "expert address table extent differs"
        );
        growth_target(self.capacity, target)?;
        let new_chunks = growth_ranges(self.capacity, target).count();
        let total_chunks = self
            .chunks
            .len()
            .checked_add(new_chunks)
            .context("expert growth chunk count overflow")?;
        ensure!(
            total_chunks <= MAX_CHUNKS,
            "expert growth would exceed the {MAX_CHUNKS}-segment storage limit"
        );
        let mut end = 0;
        for chunk in &self.chunks {
            ensure!(
                chunk.range.start == end
                    && chunk.range.start < chunk.range.end
                    && chunk.range.end <= self.capacity
                    && chunk.range.end <= (chunk.range.start / CHUNK_SLOTS + 1) * CHUNK_SLOTS,
                "expert growth chunks do not cover the retained prefix"
            );
            end = chunk.range.end;
        }
        ensure!(
            self.addresses.byte_len() == MAX_BOUNDED_SLOTS * 18 * 8
                && end == self.capacity
                && self.chunks.capacity() >= total_chunks,
            "expert growth prefix or retained chunk capacity differs"
        );
        let mut prepared_weight_bytes = 0u64;
        for range in growth_ranges(self.capacity, target) {
            let (_, bytes) = slab_layout(self.geometries, range.len())?;
            prepared_weight_bytes = prepared_weight_bytes
                .checked_add(u64::try_from(bytes)?)
                .context("expert growth byte count overflow")?;
        }
        let preparation_requested_bytes = prepared_weight_bytes
            .checked_add(u64::try_from(self.addresses.byte_len())?)
            .context("expert growth preparation byte count overflow")?;
        Ok(ChunkGrowthQuote {
            added_requested_bytes: prepared_weight_bytes,
            preparation_requested_bytes,
        })
    }

    pub(super) fn prepare_growth(
        &self,
        metal: &Metal,
        target: usize,
    ) -> Result<PreparedChunksGrowth> {
        let quote = self.quote_growth(target)?;
        self.check_quiescent()?;
        let mut chunks = Vec::new();
        chunks.try_reserve_exact(growth_ranges(self.capacity, target).count())?;
        let tables = self.chunks[0]
            .projections
            .each_ref()
            .map(|projection| projection.stream_bindings()[6].buffer);
        for range in growth_ranges(self.capacity, target) {
            chunks.push(allocate_chunk(metal, self.geometries, range, tables)?);
        }
        let addresses = metal.shared_buffer(MAX_BOUNDED_SLOTS * 18 * 8)?;
        for chunk in self.chunks.iter().chain(&chunks) {
            write_chunk_addresses(chunk, self.geometries, &addresses)?;
        }
        Ok(PreparedChunksGrowth {
            chunks,
            addresses,
            target,
            quote,
        })
    }

    pub(super) fn commit_growth(&mut self, prepared: PreparedChunksGrowth) {
        let PreparedChunksGrowth {
            mut chunks,
            addresses,
            target,
            ..
        } = prepared;
        self.chunks.append(&mut chunks);
        self.addresses = addresses;
        self.capacity = target;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iqk::{Codec, SplitStreams};

    #[test]
    fn suffix_targets_preserve_complete_prefix() {
        for (capacity, valid) in [
            (65, (16..65).step_by(16).collect::<Vec<_>>()),
            (102, (16..102).step_by(16).collect()),
            (178, (16..178).step_by(16).collect()),
            (511, (16..511).step_by(16).collect()),
        ] {
            for target in valid {
                assert!(release_prefix(capacity, target).is_ok());
            }
            for target in [0, 1, 9, 10, 15, 17, capacity, capacity + 16] {
                assert!(release_prefix(capacity, target).is_err());
            }
        }
        for capacity in [10, 15, 16] {
            assert!(release_prefix(capacity, 16).is_err());
        }
    }

    #[test]
    fn growth_targets_accept_every_larger_bounded_capacity() {
        for capacity in [10, 64, 65, 95, 117, 128, 178, 192, 510, 511] {
            for target in 0..=513 {
                assert_eq!(
                    growth_target(capacity, target).is_ok(),
                    target > capacity && target < 512
                );
            }
        }
    }

    #[test]
    fn appended_ranges_end_at_logical_chunk_boundaries() {
        assert_eq!(
            growth_ranges(145, 192).collect::<Vec<_>>(),
            [145..160, 160..176, 176..192]
        );
        assert!(growth_ranges(192, 204).eq(std::iter::once(192..204)));
        assert_eq!(
            growth_ranges(145, 204).collect::<Vec<_>>(),
            [145..160, 160..176, 176..192, 192..204]
        );
        let mut capacity = 95;
        let mut ranges: Vec<_> = (0..95)
            .step_by(CHUNK_SLOTS)
            .map(|start| start..(start + CHUNK_SLOTS).min(95))
            .collect();
        for target in [96, 117, 128, 181, 192, 511] {
            ranges.extend(growth_ranges(capacity, target));
            capacity = target;
        }
        assert_eq!(ranges.len(), 35);
        assert!(ranges.len() <= MAX_CHUNKS);
        let mut end = 0;
        for range in ranges {
            assert_eq!(range.start, end);
            assert!(range.start < range.end);
            assert!(range.end <= (range.start / CHUNK_SLOTS + 1) * CHUNK_SLOTS);
            end = range.end;
        }
        assert_eq!(end, 511);
    }

    #[test]
    fn sixteen_slot_growth_and_whole_prefix_release_keep_descriptors_bounded() {
        fn grow(ranges: &mut Vec<Range<usize>>, target: usize) {
            let mut capacity = ranges.last().unwrap().end;
            while capacity < target {
                let next = ((capacity / 16 + 1) * 16).min(target);
                ranges.extend(growth_ranges(capacity, next));
                capacity = next;
            }
        }

        fn check(ranges: &[Range<usize>], capacity: usize) {
            assert!(ranges.len() <= MAX_CHUNKS);
            let mut end = 0;
            for range in ranges {
                assert_eq!(range.start, end);
                assert!(range.start < range.end);
                assert!(range.end <= (range.start / CHUNK_SLOTS + 1) * CHUNK_SLOTS);
                end = range.end;
            }
            assert_eq!(end, capacity);
        }

        let mut maximum = 0;
        for initial in 64..=MAX_BOUNDED_SLOTS {
            let startup: Vec<_> = (0..initial)
                .step_by(CHUNK_SLOTS)
                .map(|start| start..(start + CHUNK_SLOTS).min(initial))
                .collect();
            for ceiling in initial..=MAX_BOUNDED_SLOTS {
                let mut ranges = startup.clone();
                grow(&mut ranges, ceiling);
                check(&ranges, ceiling);
                maximum = maximum.max(ranges.len());
                for target in (64..ceiling).step_by(CHUNK_SLOTS) {
                    let keep = ranges.iter().position(|range| range.end == target).unwrap() + 1;
                    let prefix = ranges[..keep].to_vec();
                    let mut cycled = prefix.clone();
                    grow(&mut cycled, ceiling);
                    check(&cycled, ceiling);
                    maximum = maximum.max(cycled.len());
                    let restored = cycled.clone();
                    cycled.truncate(keep);
                    assert_eq!(cycled, prefix);
                    grow(&mut cycled, ceiling);
                    assert_eq!(cycled, restored);
                }
            }
        }
        // A partial startup tail can split one additional 16-slot interval.
        assert_eq!(maximum, 33);
    }

    #[test]
    #[ignore = "requires exclusive Metal ownership"]
    fn growth_preparation_rolls_back_and_retains_prefix_tail_and_exact_quote() -> Result<()> {
        let metal = Metal::new()?;
        for codec in [Codec::Iq2K, Codec::Iq3K] {
            let geometries = [Geometry::new(codec, 768, 16)?; 3];
            for initial in [65, 102, 178] {
                let mut chunks = FixedChunks::new(&metal, geometries, initial)?;
                let (keep, _) = chunks.prepare_suffix(64)?;
                chunks.check_quiescent()?;
                chunks.commit_suffix(keep, 64);
                let prefix = chunks.buffers().next().unwrap().clone();
                prefix.write_bytes(0, &vec![0x5a; prefix.byte_len()])?;
                let mut prefix_bytes = vec![0; prefix.byte_len()];
                prefix.read_bytes(0, &mut prefix_bytes)?;
                let mut original_rows = vec![0; chunks.addresses.byte_len()];
                chunks.addresses.read_bytes(0, &mut original_rows)?;
                let original_address = chunks
                    .addresses
                    .gpu_address(0, chunks.addresses.byte_len())?;
                let quote = chunks.quote_growth(initial)?;
                let requested = metal.memory_stats()?.live_requested_buffer_bytes;
                let allocations = growth_ranges(64, initial).count() + 1;
                for failure in 0..allocations {
                    let fault = metal.test_fail_allocation_after(failure)?;
                    assert!(chunks.prepare_growth(&metal, initial).is_err());
                    drop(fault);
                    assert_eq!(chunks.capacity(), 64);
                    assert_eq!(chunks.count(), 64 / CHUNK_SLOTS);
                    assert_eq!(metal.memory_stats()?.live_requested_buffer_bytes, requested);
                    let mut actual = vec![0; original_rows.len()];
                    chunks.addresses.read_bytes(0, &mut actual)?;
                    assert_eq!(actual, original_rows);
                    assert_eq!(
                        chunks.addresses.gpu_address(0, actual.len())?,
                        original_address
                    );
                }
                {
                    let prepared = chunks.prepare_growth(&metal, initial)?;
                    assert_eq!(
                        metal.memory_stats()?.live_requested_buffer_bytes - requested,
                        quote.preparation_requested_bytes
                    );
                    assert_eq!(
                        prepared.quote.added_requested_bytes,
                        quote.added_requested_bytes
                    );
                }
                assert_eq!(metal.memory_stats()?.live_requested_buffer_bytes, requested);
                let prepared = chunks.prepare_growth(&metal, initial)?;
                let fault = metal.test_fail_allocation_after(0)?;
                chunks.commit_growth(prepared);
                drop(fault);
                assert_eq!(
                    metal.memory_stats()?.live_requested_buffer_bytes - requested,
                    quote.added_requested_bytes
                );
                assert_eq!(chunks.capacity(), initial);
                assert_eq!(chunks.count(), initial.div_ceil(CHUNK_SLOTS));
                assert_eq!(chunks.range(chunks.count() - 1)?.end, initial);
                assert!(prefix.shares_allocation(chunks.buffers().next().unwrap()));
                let mut actual = vec![0; prefix_bytes.len()];
                prefix.read_bytes(0, &mut actual)?;
                assert_eq!(actual, prefix_bytes);
                let mut rows = vec![0; original_rows.len()];
                chunks.addresses.read_bytes(0, &mut rows)?;
                assert_eq!(&rows[..64 * 18 * 8], &original_rows[..64 * 18 * 8]);
                assert_ne!(
                    chunks.addresses.gpu_address(0, rows.len())?,
                    original_address
                );
                for slot in 64..initial {
                    let (chunk, local) = chunks.location(slot)?;
                    for (projection, kind) in Projection::ALL.into_iter().enumerate() {
                        let bindings = chunks.projection(chunk, kind)?.stream_bindings();
                        for (stream, bytes) in IqkGemv::slot_stream_bytes(geometries[projection])?
                            .into_iter()
                            .enumerate()
                        {
                            let at = (slot * 18 + projection * 6 + stream) * 8;
                            assert_eq!(
                                u64::from_le_bytes(rows[at..at + 8].try_into()?),
                                bindings[stream].buffer.gpu_address(
                                    bindings[stream].offset + local * bytes,
                                    bytes.max(1)
                                )?
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires exclusive Metal ownership"]
    fn appended_growth_preserves_every_stream_and_rolls_back_all_candidates() -> Result<()> {
        let metal = Metal::new()?;
        for codec in [Codec::Iq2K, Codec::Iq3K] {
            let geometries = [Geometry::new(codec, 768, 16)?; 3];
            let mut chunks = FixedChunks::new(&metal, geometries, 95)?;
            for (index, chunk) in chunks.chunks.iter().enumerate() {
                let bytes: Vec<_> = (0..chunk.buffer.byte_len())
                    .map(|offset| (offset.wrapping_mul(31) + index * 7) as u8)
                    .collect();
                chunk.buffer.write_bytes(0, &bytes)?;
            }
            let read_slots = |chunks: &FixedChunks, slots: usize| -> Result<Vec<Vec<u8>>> {
                (0..slots)
                    .map(|slot| {
                        let (chunk, local) = chunks.location(slot)?;
                        let mut row = Vec::new();
                        for kind in Projection::ALL {
                            let projection = projection_index(kind);
                            let bindings = chunks.projection(chunk, kind)?.stream_bindings();
                            for (stream, bytes) in
                                IqkGemv::slot_stream_bytes(geometries[projection])?
                                    .into_iter()
                                    .enumerate()
                            {
                                let start = row.len();
                                row.resize(start + bytes, 0);
                                bindings[stream].buffer.read_bytes(
                                    bindings[stream].offset + local * bytes,
                                    &mut row[start..],
                                )?;
                            }
                        }
                        Ok(row)
                    })
                    .collect()
            };
            for target in [96, 117, 128, 145, 160, 176, 192, 204, 511] {
                let old_capacity = chunks.capacity();
                let retained = read_slots(&chunks, old_capacity)?;
                let old_count = chunks.count();
                let prefix_addresses: Vec<_> = chunks
                    .chunks
                    .iter()
                    .map(|chunk| chunk.buffer.gpu_address(0, chunk.buffer.byte_len()))
                    .collect::<Result<_>>()?;
                let mut old_addresses = vec![0; chunks.addresses.byte_len()];
                chunks.addresses.read_bytes(0, &mut old_addresses)?;
                let address_owner = chunks.addresses.gpu_address(0, old_addresses.len())?;
                let requested = metal.memory_stats()?.live_requested_buffer_bytes;
                let quote = chunks.quote_growth(target)?;
                let (_, per_slot) = slab_layout(geometries, 1)?;
                assert_eq!(
                    quote.added_requested_bytes,
                    (target - old_capacity) as u64 * per_slot as u64
                );
                assert_eq!(
                    quote.preparation_requested_bytes,
                    (target - old_capacity) as u64 * per_slot as u64
                        + (MAX_BOUNDED_SLOTS * 18 * 8) as u64
                );
                let added_chunks = growth_ranges(old_capacity, target).count();
                let allocations = added_chunks + 1;
                for failure in 0..allocations {
                    let fault = metal.test_fail_allocation_after(failure)?;
                    assert!(chunks.prepare_growth(&metal, target).is_err());
                    drop(fault);
                    assert_eq!(chunks.capacity(), old_capacity);
                    assert_eq!(read_slots(&chunks, old_capacity)?, retained);
                    assert_eq!(metal.memory_stats()?.live_requested_buffer_bytes, requested);
                    let mut actual = vec![0; old_addresses.len()];
                    chunks.addresses.read_bytes(0, &mut actual)?;
                    assert_eq!(actual, old_addresses);
                    assert_eq!(
                        chunks.addresses.gpu_address(0, actual.len())?,
                        address_owner
                    );
                }
                {
                    let prepared = chunks.prepare_growth(&metal, target)?;
                    assert_eq!(
                        metal.memory_stats()?.live_requested_buffer_bytes - requested,
                        prepared.quote.preparation_requested_bytes
                    );
                }
                assert_eq!(read_slots(&chunks, old_capacity)?, retained);
                assert_eq!(metal.memory_stats()?.live_requested_buffer_bytes, requested);
                let prepared = chunks.prepare_growth(&metal, target)?;
                let fault = metal.test_fail_allocation_after(0)?;
                chunks.commit_growth(prepared);
                drop(fault);
                assert_eq!(chunks.capacity(), target);
                assert_eq!(chunks.count(), old_count + added_chunks);
                assert_eq!(chunks.range(chunks.count() - 1)?.end, target);
                assert_eq!(read_slots(&chunks, old_capacity)?, retained);
                assert_eq!(
                    metal.memory_stats()?.live_requested_buffer_bytes - requested,
                    quote.added_requested_bytes
                );
                for (chunk, address) in chunks.chunks.iter().zip(prefix_addresses) {
                    assert_eq!(
                        chunk.buffer.gpu_address(0, chunk.buffer.byte_len())?,
                        address
                    );
                }
                let mut rows = vec![0; chunks.addresses.byte_len()];
                chunks.addresses.read_bytes(0, &mut rows)?;
                assert_eq!(
                    &rows[..old_capacity * 18 * 8],
                    &old_addresses[..old_capacity * 18 * 8]
                );
                assert_ne!(chunks.addresses.gpu_address(0, rows.len())?, address_owner);
                for slot in 0..target {
                    let (chunk, local) = chunks.location(slot)?;
                    for (projection, kind) in Projection::ALL.into_iter().enumerate() {
                        let bindings = chunks.projection(chunk, kind)?.stream_bindings();
                        for (stream, bytes) in IqkGemv::slot_stream_bytes(geometries[projection])?
                            .into_iter()
                            .enumerate()
                        {
                            let at = (slot * 18 + projection * 6 + stream) * 8;
                            assert_eq!(
                                u64::from_le_bytes(rows[at..at + 8].try_into()?),
                                bindings[stream].buffer.gpu_address(
                                    bindings[stream].offset + local * bytes,
                                    bytes.max(1)
                                )?
                            );
                        }
                    }
                }
            }
            assert!(chunks.quote_growth(512).is_err());
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires exclusive Metal ownership"]
    fn segment_limit_fails_before_allocation_and_shrink_regrowth_stays_bounded() -> Result<()> {
        let metal = Metal::new()?;
        let geometries = [Geometry::new(Codec::Iq2K, 768, 16)?; 3];
        let mut chunks = FixedChunks::new(&metal, geometries, 64)?;
        let limit_capacity = 64 + MAX_CHUNKS - chunks.count();
        for target in 65..=limit_capacity {
            let prepared = chunks.prepare_growth(&metal, target)?;
            chunks.commit_growth(prepared);
        }
        assert_eq!(chunks.count(), MAX_CHUNKS);
        assert!(chunks.chunks.capacity() >= MAX_CHUNKS);
        let requested = metal.memory_stats()?.live_requested_buffer_bytes;
        let address = chunks
            .addresses
            .gpu_address(0, chunks.addresses.byte_len())?;
        let mut rows = vec![0; chunks.addresses.byte_len()];
        chunks.addresses.read_bytes(0, &mut rows)?;
        let error = chunks
            .quote_growth(limit_capacity + 1)
            .err()
            .context("quote unexpectedly succeeded")?;
        assert!(error.to_string().contains("segment"));
        let fault = metal.test_fail_allocation_after(0)?;
        let error = chunks
            .prepare_growth(&metal, limit_capacity + 1)
            .err()
            .context("growth unexpectedly succeeded")?;
        assert!(error.to_string().contains("segment"));
        drop(fault);
        assert_eq!(chunks.capacity(), limit_capacity);
        assert_eq!(chunks.count(), MAX_CHUNKS);
        assert_eq!(metal.memory_stats()?.live_requested_buffer_bytes, requested);
        let mut actual = vec![0; rows.len()];
        chunks.addresses.read_bytes(0, &mut actual)?;
        assert_eq!(actual, rows);
        assert_eq!(chunks.addresses.gpu_address(0, rows.len())?, address);
        let (keep, _) = chunks.prepare_suffix(64)?;
        chunks.commit_suffix(keep, 64);
        for _ in 0..16 {
            for target in [145, 192, 204] {
                let prepared = chunks.prepare_growth(&metal, target)?;
                chunks.commit_growth(prepared);
            }
            assert_eq!(chunks.count(), 14);
            assert_eq!(chunks.range(8)?, 128..144);
            assert_eq!(chunks.range(9)?, 144..145);
            assert_eq!(chunks.range(10)?, 145..160);
            assert_eq!(chunks.range(13)?, 192..204);
            let (keep, _) = chunks.prepare_suffix(192)?;
            assert_eq!(keep, 13);
            chunks.commit_suffix(keep, 192);
            let (keep, _) = chunks.prepare_suffix(64)?;
            assert_eq!(keep, 64 / CHUNK_SLOTS);
            chunks.commit_suffix(keep, 64);
            assert_eq!(chunks.count(), 64 / CHUNK_SLOTS);
        }
        Ok(())
    }

    #[test]
    fn fixed_layout_has_exact_tails_and_disjoint_projection_spans() -> Result<()> {
        for codec in [Codec::Iq2K, Codec::Iq3K] {
            let geometries = [
                Geometry::new(codec, 2560, 640)?,
                Geometry::new(codec, 2560, 640)?,
                Geometry::new(codec, 768, 2560)?,
            ];
            let per_slot = geometries
                .into_iter()
                .map(|geometry| {
                    Ok(IqkGemv::slot_stream_bytes(geometry)?
                        .into_iter()
                        .sum::<usize>())
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .sum::<usize>();
            for capacity in [10, 65, 102, 178, 511] {
                let mut total = 0;
                let mut ranges = Vec::new();
                for start in (0..capacity).step_by(CHUNK_SLOTS) {
                    let slots = (capacity - start).min(CHUNK_SLOTS);
                    let (offsets, bytes) = slab_layout(geometries, slots)?;
                    assert_eq!(bytes, slots * per_slot);
                    let mut end = 0;
                    for (projection, geometry) in geometries.into_iter().enumerate() {
                        for (stream, extent) in IqkGemv::slot_stream_bytes(geometry)?
                            .into_iter()
                            .enumerate()
                        {
                            if extent == 0 {
                                assert_eq!(offsets[projection][stream], 0);
                                continue;
                            }
                            assert_eq!(offsets[projection][stream], end);
                            assert!(end.is_multiple_of(16));
                            end += slots * extent;
                        }
                    }
                    assert_eq!(end, bytes);
                    total += bytes;
                    ranges.push(start..start + slots);
                }
                assert_eq!(total, capacity * per_slot);
                assert_eq!(ranges.len(), capacity.div_ceil(CHUNK_SLOTS));
                assert_eq!(ranges.last().unwrap().end, capacity);
                assert!(ranges.windows(2).all(|pair| pair[0].end == pair[1].start));
            }
        }
        assert!(slab_layout([Geometry::new(Codec::Iq2K, 768, 16)?; 3], 0).is_err());
        assert!(slab_layout([Geometry::new(Codec::Iq2K, 768, 16)?; 3], 65).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires exclusive Metal ownership"]
    fn chunk_slab_offsets_writes_table_and_unsubmitted_guards() -> Result<()> {
        let metal = Metal::new()?;
        for codec in [Codec::Iq2K, Codec::Iq3K] {
            let geometries = [Geometry::new(codec, 768, 16)?; 3];
            for capacity in [65, 102, 178] {
                let chunks = FixedChunks::new(&metal, geometries, capacity)?;
                let mut expected = chunks
                    .chunks
                    .iter()
                    .map(|chunk| {
                        let bytes = vec![0xa5; chunk.buffer.byte_len()];
                        chunk.buffer.write_bytes(0, &bytes)?;
                        Ok(bytes)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let gate = metal.shared_event()?.point(1)?;
                for (kind, slot) in [
                    (Projection::Gate, 63),
                    (Projection::Up, 64),
                    (Projection::Down, capacity - 1),
                ] {
                    let (chunk, local) = chunks.location(slot)?;
                    let projection = chunks.projection(chunk, kind)?;
                    let geometry = geometries[projection_index(kind)];
                    let row: Vec<_> = (0..16 * geometry.row_bytes())
                        .map(|index| {
                            (index.wrapping_mul(31) + slot * 7 + projection_index(kind)) as u8
                        })
                        .collect();
                    match kind {
                        Projection::Gate => projection.write_slot(local, &row)?,
                        // SAFETY: No work is encoded; this exact slot has no reader.
                        Projection::Up => unsafe {
                            projection.write_slot_disjoint(local, &row)?;
                        },
                        // SAFETY: The unopened event excludes every future reader.
                        Projection::Down => unsafe {
                            projection.write_slot_gated(local, &row, &gate)?;
                        },
                    }
                    let mut split = SplitStreams::new(geometry);
                    split.append_rows(&row)?;
                    let payloads = split.stream_slices();
                    let bindings = projection.stream_bindings();
                    for (stream, payload) in payloads.into_iter().enumerate() {
                        if payload.is_empty() {
                            continue;
                        }
                        let start = bindings[stream].offset + local * payload.len();
                        expected[chunk][start..start + payload.len()].copy_from_slice(payload);
                    }
                    assert!(
                        projection
                            .preflight_slot(chunks.range(chunk)?.len(), &row)
                            .is_err()
                    );
                }
                gate.release();
                let projection = chunks.projection(0, Projection::Gate)?;
                let row = vec![0; 16 * geometries[0].row_bytes()];
                // SAFETY: A consumed gate must fail before any byte mutation.
                assert!(unsafe { projection.write_slot_gated(0, &row, &gate) }.is_err());
                let mut batch = metal.batch()?;
                let owners: Vec<_> = chunks.buffers().collect();
                batch.use_indirect_reads(&owners)?;
                // Even encoded-but-unsubmitted indirect ownership reserves every
                // projection alias of the common slab against ordinary writes.
                assert!(projection.write_slot(0, &row).is_err());
                assert!(
                    chunks
                        .projection(0, Projection::Down)?
                        .check_writable()
                        .is_err()
                );
                drop(batch);
                chunks.check_writable()?;
                for (chunk, expected) in chunks.chunks.iter().zip(expected) {
                    let mut actual = vec![0; chunk.buffer.byte_len()];
                    chunk.buffer.read_bytes(0, &mut actual)?;
                    assert_eq!(actual, expected);
                }
                let mut addresses = vec![0; chunks.addresses.byte_len()];
                chunks.addresses.read_bytes(0, &mut addresses)?;
                for slot in 0..capacity {
                    let (chunk, local) = chunks.location(slot)?;
                    for (projection, kind) in Projection::ALL.into_iter().enumerate() {
                        let bindings = chunks.projection(chunk, kind)?.stream_bindings();
                        for (stream, bytes) in IqkGemv::slot_stream_bytes(geometries[projection])?
                            .into_iter()
                            .enumerate()
                        {
                            let index = (slot * 18 + projection * 6 + stream) * 8;
                            let address =
                                u64::from_le_bytes(addresses[index..index + 8].try_into()?);
                            assert_eq!(
                                address,
                                bindings[stream].buffer.gpu_address(
                                    bindings[stream].offset + local * bytes,
                                    bytes.max(1)
                                )?
                            );
                        }
                    }
                }
                assert!(chunks.location(capacity).is_err());
                assert!(chunks.range(chunks.count()).is_err());
            }
        }
        Ok(())
    }
}
