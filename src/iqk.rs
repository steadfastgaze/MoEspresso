//! IQ2_K and IQ3_K relayout streams and fixed-width decode GEMV geometry.
//!
//! The package stores one row's streams consecutively; the Metal kernels
//! consume each stream consecutively across all rows of an expert projection.

use anyhow::{Context, Result, ensure};
use half::f16;

pub const METAL_SOURCE: &str = include_str!("../kernels/metal/iqk_gemv.metal");
pub const ROWS_PER_THREADGROUP: usize = 16;
const WEIGHTS_PER_LANE: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Codec {
    Iq2K,
    Iq3K,
}

impl Codec {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "iq2_k" => Ok(Self::Iq2K),
            "iq3_k" => Ok(Self::Iq3K),
            _ => anyhow::bail!("unsupported IQ_K GEMV codec {name}"),
        }
    }

    const fn bytes_per_block(self) -> usize {
        match self {
            Self::Iq2K => 76,
            Self::Iq3K => 110,
        }
    }

    pub fn value_table_fp16(self) -> Vec<u8> {
        const IQ2: [i16; 8] = [-31, -13, 1, 17, -26, -8, 6, 22];
        const IQ3: [i16; 16] = [
            -63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51,
        ];
        let table: &[i16] = match self {
            Self::Iq2K => &IQ2,
            Self::Iq3K => &IQ3,
        };
        table
            .iter()
            .flat_map(|&value| f16::from_f32(f32::from(value)).to_bits().to_le_bytes())
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Geometry {
    codec: Codec,
    in_features: usize,
    out_features: usize,
}

impl Geometry {
    pub fn new(codec: Codec, in_features: usize, out_features: usize) -> Result<Self> {
        ensure!(
            matches!(in_features, 768 | 2560),
            "IQ_K GEMV requires stored width 768 or 2560"
        );
        ensure!(
            out_features > 0 && out_features.is_multiple_of(ROWS_PER_THREADGROUP),
            "IQ_K GEMV output rows must be a positive multiple of 16"
        );
        ensure!(
            out_features <= u32::MAX as usize,
            "IQ_K output width exceeds Metal dimensions"
        );
        Ok(Self {
            codec,
            in_features,
            out_features,
        })
    }

    pub fn from_projection(layout: &crate::package::ProjectionLayout) -> Result<Self> {
        Self::new(
            Codec::parse(&layout.codec)?,
            layout.stored_in_features,
            layout.out_features,
        )
    }

    pub fn row_bytes(self) -> usize {
        self.in_features / 256 * self.codec.bytes_per_block()
    }

    pub fn codec(self) -> Codec {
        self.codec
    }
    pub fn in_features(self) -> usize {
        self.in_features
    }
    pub fn out_features(self) -> usize {
        self.out_features
    }

    pub fn entrypoint(self) -> &'static str {
        match (self.codec, self.in_features) {
            (Codec::Iq2K, 2560) => "iq2_k_2560",
            (Codec::Iq2K, 768) => "iq2_k_768",
            (Codec::Iq3K, 2560) => "iq3_k_2560",
            (Codec::Iq3K, 768) => "iq3_k_768",
            _ => unreachable!("validated IQ_K width"),
        }
    }

    pub fn threads_per_threadgroup(self) -> usize {
        self.in_features / WEIGHTS_PER_LANE
    }

    fn stream_lengths(self) -> StreamLengths {
        let n = self.in_features;
        StreamLengths {
            qs: n / 4,
            qh: if self.codec == Codec::Iq3K { n / 8 } else { 0 },
            scl: n / 32,
            sch: if self.codec == Codec::Iq3K {
                n / 128
            } else {
                0
            },
            sex: n / 128,
            dv: n / 256 * 2,
        }
    }
}

#[derive(Clone, Copy)]
struct StreamLengths {
    qs: usize,
    qh: usize,
    scl: usize,
    sch: usize,
    sex: usize,
    dv: usize,
}

#[derive(Clone, Copy, Debug)]
struct StreamCopySpan {
    row_offset: usize,
    row_bytes: usize,
    slot_bytes: usize,
}

/// Validated row spans for copying one relayout projection into stream-major slots.
#[derive(Clone, Copy, Debug)]
struct RelayoutSlotCopyPlan {
    row_bytes: usize,
    rows: usize,
    slots: usize,
    spans: [StreamCopySpan; 6],
}

impl RelayoutSlotCopyPlan {
    fn new(geometry: Geometry, slots: usize) -> Result<Self> {
        ensure!(slots > 0, "IQ_K copy plan requires physical slots");
        let lengths = geometry.stream_lengths();
        let widths = [
            lengths.qs,
            lengths.qh,
            lengths.scl,
            lengths.sch,
            lengths.sex,
            lengths.dv,
        ];
        let mut spans = [StreamCopySpan {
            row_offset: 0,
            row_bytes: 0,
            slot_bytes: 0,
        }; 6];
        let mut offset = 0usize;
        for (index, width) in widths.into_iter().enumerate() {
            spans[index] = StreamCopySpan {
                row_offset: offset,
                row_bytes: width,
                slot_bytes: geometry
                    .out_features
                    .checked_mul(width)
                    .context("IQ_K stream slot byte count overflow")?,
            };
            offset = offset
                .checked_add(width)
                .context("IQ_K stream span overflow")?;
        }
        ensure!(
            offset == geometry.row_bytes(),
            "IQ_K stream spans do not fill one row"
        );
        Ok(Self {
            row_bytes: offset,
            rows: geometry.out_features,
            slots,
            spans,
        })
    }

    fn validate_destinations(&self, bytes: [Option<usize>; 6]) -> Result<()> {
        for (span, size) in self.spans.iter().zip(bytes) {
            let expected = self
                .slots
                .checked_mul(span.slot_bytes)
                .context("IQ_K stream allocation size overflow")?;
            ensure!(
                if span.row_bytes == 0 {
                    size.is_none()
                } else {
                    size == Some(expected)
                },
                "IQ_K stream storage differs from relayout copy plan"
            );
        }
        Ok(())
    }

    fn slot_offsets(&self, slot: usize, source_len: usize) -> Result<[Option<usize>; 6]> {
        ensure!(slot < self.slots, "IQ_K slot out of range");
        ensure!(
            source_len
                == self
                    .rows
                    .checked_mul(self.row_bytes)
                    .context("IQ_K projection byte count overflow")?,
            "IQ_K projection payload has invalid size"
        );
        let mut offsets = [None; 6];
        for (offset, span) in offsets.iter_mut().zip(self.spans) {
            if span.row_bytes != 0 {
                let start = slot
                    .checked_mul(span.slot_bytes)
                    .context("IQ_K slot offset overflow")?;
                ensure!(
                    start.checked_add(span.slot_bytes).is_some_and(|end| self
                        .slots
                        .checked_mul(span.slot_bytes)
                        .is_some_and(|length| end <= length)),
                    "IQ_K slot span exceeds stream storage"
                );
                *offset = Some(start);
            }
        }
        Ok(offsets)
    }
}

/// Per-expert projection streams in Metal's row-major stream order.
#[derive(Debug)]
pub struct SplitStreams {
    geometry: Geometry,
    experts: usize,
    qs: Vec<u8>,
    qh: Vec<u8>,
    scl: Vec<u8>,
    sch: Vec<u8>,
    sex: Vec<u8>,
    dv: Vec<u8>,
}

impl SplitStreams {
    #[cfg(test)]
    pub(crate) fn stream_slices(&self) -> [&[u8]; 6] {
        [
            &self.qs, &self.qh, &self.scl, &self.sch, &self.sex, &self.dv,
        ]
    }

    pub fn experts(&self) -> usize {
        self.experts
    }
    pub fn geometry(&self) -> Geometry {
        self.geometry
    }

    pub fn new(geometry: Geometry) -> Self {
        Self {
            geometry,
            experts: 0,
            qs: Vec::new(),
            qh: Vec::new(),
            scl: Vec::new(),
            sch: Vec::new(),
            sex: Vec::new(),
            dv: Vec::new(),
        }
    }

    pub fn from_rows(geometry: Geometry, rows: &[u8]) -> Result<Self> {
        let mut streams = Self::new(geometry);
        streams.append_rows(rows)?;
        Ok(streams)
    }

    pub fn append_projection_row(
        &mut self,
        layout: &crate::package::ProjectionLayout,
        expert_row: &[u8],
    ) -> Result<()> {
        ensure!(
            self.geometry == Geometry::from_projection(layout)?,
            "IQ_K projection geometry differs from stream geometry"
        );
        self.append_rows(layout.bytes(expert_row)?)
    }

    /// Append exactly one expert's `[out_features, bytes_per_row]` payload.
    pub fn append_rows(&mut self, rows: &[u8]) -> Result<()> {
        let g = self.geometry;
        let row_bytes = g.row_bytes();
        let expected = g
            .out_features
            .checked_mul(row_bytes)
            .context("IQ_K projection size overflow")?;
        ensure!(
            rows.len() == expected,
            "IQ_K projection has {} bytes; expected {expected}",
            rows.len()
        );
        ensure!(self.experts < u32::MAX as usize, "too many IQ_K experts");
        let len = g.stream_lengths();
        ensure!(
            len.qs + len.qh + len.scl + len.sch + len.sex + len.dv == row_bytes,
            "IQ_K stream lengths do not fill the stored row"
        );
        for row in rows.chunks_exact(row_bytes) {
            let mut at = 0;
            for (dst, size) in [
                (&mut self.qs, len.qs),
                (&mut self.qh, len.qh),
                (&mut self.scl, len.scl),
                (&mut self.sch, len.sch),
                (&mut self.sex, len.sex),
                (&mut self.dv, len.dv),
            ] {
                dst.extend_from_slice(&row[at..at + size]);
                at += size;
            }
        }
        self.experts += 1;
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        ensure!(self.experts > 0, "IQ_K GEMV needs at least one expert");
        let g = Geometry::new(
            self.geometry.codec,
            self.geometry.in_features,
            self.geometry.out_features,
        )?;
        let per_row = g.stream_lengths();
        let rows = self
            .experts
            .checked_mul(g.out_features)
            .context("IQ_K stream size overflow")?;
        for (actual, one) in [
            (self.qs.len(), per_row.qs),
            (self.qh.len(), per_row.qh),
            (self.scl.len(), per_row.scl),
            (self.sch.len(), per_row.sch),
            (self.sex.len(), per_row.sex),
            (self.dv.len(), per_row.dv),
        ] {
            ensure!(
                actual == rows.checked_mul(one).context("IQ_K stream size overflow")?,
                "IQ_K stream byte count differs from projection geometry"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GemvDescriptor {
    pub geometry: Geometry,
    pub pairs: usize,
    pub pairs_per_token: usize,
    pub group_count: usize,
    pub grid_threads: usize,
    pub group_threads: usize,
    pub dims: [u32; 2],
}

impl GemvDescriptor {
    pub fn new(geometry: Geometry, pairs: usize, pairs_per_token: usize) -> Result<Self> {
        ensure!(
            pairs > 0 && pairs_per_token > 0 && pairs.is_multiple_of(pairs_per_token),
            "IQ_K pairs must contain whole tokens with positive fanout"
        );
        ensure!(
            pairs_per_token <= u32::MAX as usize,
            "IQ_K fanout exceeds Metal dimensions"
        );
        let group_threads = geometry.threads_per_threadgroup();
        let group_count = pairs
            .checked_mul(geometry.out_features / ROWS_PER_THREADGROUP)
            .context("IQ_K grid size overflow")?;
        ensure!(
            group_count <= u32::MAX as usize,
            "IQ_K group grid exceeds Metal uint index"
        );
        let grid_threads = group_count
            .checked_mul(group_threads)
            .context("IQ_K grid size overflow")?;
        Ok(Self {
            geometry,
            pairs,
            pairs_per_token,
            group_count,
            grid_threads,
            group_threads,
            dims: [pairs_per_token as u32, geometry.out_features as u32],
        })
    }

    pub fn output_bytes(self) -> Result<usize> {
        self.pairs
            .checked_mul(self.geometry.out_features)
            .and_then(|n| n.checked_mul(2))
            .context("IQ_K output size overflow")
    }

    pub fn input_bytes(self) -> Result<usize> {
        (self.pairs / self.pairs_per_token)
            .checked_mul(self.geometry.in_features)
            .and_then(|n| n.checked_mul(2))
            .context("IQ_K input size overflow")
    }
}

#[cfg(target_os = "macos")]
mod gpu {
    use super::{GemvDescriptor, Geometry, METAL_SOURCE, RelayoutSlotCopyPlan, SplitStreams};
    use crate::metal::{Batch, Binding, Buffer, Metal, Pipeline};
    use anyhow::{Context, Result, ensure};

    struct GpuStreams {
        qs: Buffer,
        qh: Option<Buffer>,
        scl: Buffer,
        sch: Option<Buffer>,
        sex: Buffer,
        dv: Buffer,
        table: Buffer,
    }

    pub struct IqkGemv {
        geometry: Geometry,
        experts: usize,
        pipeline: Pipeline,
        streams: GpuStreams,
        copy_plan: RelayoutSlotCopyPlan,
        stream_offsets: [usize; 6],
        stream_bytes: [Option<usize>; 6],
    }

    fn uploaded(metal: &Metal, bytes: &[u8]) -> Result<Buffer> {
        let buffer = metal.shared_buffer(bytes.len())?;
        buffer.write_bytes(0, bytes)?;
        Ok(buffer)
    }

    impl IqkGemv {
        /// Read-only stream bindings retained by the consuming command batch.
        /// Absent IQ2 high-bit streams bind existing storage and must not be read.
        pub(crate) fn stream_bindings(&self) -> [Binding<'_>; 7] {
            let buffers = [
                &self.streams.qs,
                self.streams.qh.as_ref().unwrap_or(&self.streams.qs),
                &self.streams.scl,
                self.streams.sch.as_ref().unwrap_or(&self.streams.scl),
                &self.streams.sex,
                &self.streams.dv,
                &self.streams.table,
            ];
            std::array::from_fn(|index| Binding {
                buffer: buffers[index],
                offset: match index {
                    1 if self.streams.qh.is_none() => self.stream_offsets[0],
                    3 if self.streams.sch.is_none() => self.stream_offsets[2],
                    0..6 => self.stream_offsets[index],
                    _ => 0,
                },
            })
        }

        pub fn empty_slots(metal: &Metal, geometry: Geometry, slots: usize) -> Result<Self> {
            ensure!(
                slots > 0 && slots <= u32::MAX as usize,
                "invalid IQ_K slot count"
            );
            let mut split = SplitStreams::new(geometry);
            let row = vec![
                0;
                geometry
                    .out_features
                    .checked_mul(geometry.row_bytes())
                    .context("IQ_K slot size overflow")?
            ];
            for _ in 0..slots {
                split.append_rows(&row)?;
            }
            Self::new(metal, split)
        }

        pub fn new(metal: &Metal, split: SplitStreams) -> Result<Self> {
            split.validate()?;
            let geometry = split.geometry;
            let copy_plan = RelayoutSlotCopyPlan::new(geometry, split.experts)?;
            let pipeline = metal.compile_msl(METAL_SOURCE, geometry.entrypoint())?;
            let streams = GpuStreams {
                qs: uploaded(metal, &split.qs)?,
                qh: if split.qh.is_empty() {
                    None
                } else {
                    Some(uploaded(metal, &split.qh)?)
                },
                scl: uploaded(metal, &split.scl)?,
                sch: if split.sch.is_empty() {
                    None
                } else {
                    Some(uploaded(metal, &split.sch)?)
                },
                sex: uploaded(metal, &split.sex)?,
                dv: uploaded(metal, &split.dv)?,
                table: uploaded(metal, &geometry.codec.value_table_fp16())?,
            };
            let stream_bytes = [
                Some(streams.qs.byte_len()),
                streams.qh.as_ref().map(Buffer::byte_len),
                Some(streams.scl.byte_len()),
                streams.sch.as_ref().map(Buffer::byte_len),
                Some(streams.sex.byte_len()),
                Some(streams.dv.byte_len()),
            ];
            copy_plan.validate_destinations(stream_bytes)?;
            Ok(Self {
                geometry,
                experts: split.experts,
                pipeline,
                streams,
                copy_plan,
                stream_offsets: [0; 6],
                stream_bytes,
            })
        }

        pub(crate) fn slot_stream_bytes(geometry: Geometry) -> Result<[usize; 6]> {
            Ok(RelayoutSlotCopyPlan::new(geometry, 1)?
                .spans
                .map(|span| span.slot_bytes))
        }

        /// View exact stream spans in one independently owned expert slab.
        pub(crate) fn slab_view(
            metal: &Metal,
            geometry: Geometry,
            slots: usize,
            slab: &Buffer,
            offsets: [usize; 6],
            table: &Buffer,
        ) -> Result<Self> {
            let copy_plan = RelayoutSlotCopyPlan::new(geometry, slots)?;
            let mut stream_bytes = [None; 6];
            let mut spans = Vec::new();
            for (index, span) in copy_plan.spans.iter().enumerate() {
                if span.row_bytes == 0 {
                    ensure!(
                        offsets[index] == 0,
                        "absent IQ_K stream requires a zero offset"
                    );
                    continue;
                }
                let bytes = slots
                    .checked_mul(span.slot_bytes)
                    .context("slab stream extent overflow")?;
                let end = offsets[index]
                    .checked_add(bytes)
                    .context("slab stream end overflow")?;
                ensure!(
                    offsets[index].is_multiple_of(16) && end <= slab.byte_len(),
                    "IQ_K slab stream is misaligned or out of bounds"
                );
                stream_bytes[index] = Some(bytes);
                spans.push(offsets[index]..end);
            }
            spans.sort_by_key(|span| span.start);
            ensure!(
                spans.windows(2).all(|pair| pair[0].end <= pair[1].start),
                "IQ_K slab streams overlap"
            );
            copy_plan.validate_destinations(stream_bytes)?;
            ensure!(
                table.byte_len() == geometry.codec.value_table_fp16().len(),
                "IQ_K slab value table differs from codec"
            );
            Ok(Self {
                geometry,
                experts: slots,
                pipeline: metal.compile_msl(METAL_SOURCE, geometry.entrypoint())?,
                streams: GpuStreams {
                    qs: slab.clone(),
                    qh: stream_bytes[1].map(|_| slab.clone()),
                    scl: slab.clone(),
                    sch: stream_bytes[3].map(|_| slab.clone()),
                    sex: slab.clone(),
                    dv: slab.clone(),
                    table: table.clone(),
                },
                copy_plan,
                stream_offsets: offsets,
                stream_bytes,
            })
        }

        pub(crate) fn check_writable(&self) -> Result<()> {
            let mut empty = [];
            for buffer in [
                Some(&self.streams.qs),
                self.streams.qh.as_ref(),
                Some(&self.streams.scl),
                self.streams.sch.as_ref(),
                Some(&self.streams.sex),
                Some(&self.streams.dv),
            ]
            .into_iter()
            .flatten()
            {
                buffer.read_bytes(0, &mut empty)?;
            }
            Ok(())
        }

        /// Replace one physical slot. A caller must invalidate its logical mapping first.
        ///
        /// # Safety
        /// All readers of this pool wait behind this unopened publication
        /// point; preceding readers have completed and been retired.
        pub(crate) unsafe fn write_slot_gated(
            &self,
            slot: usize,
            rows: &[u8],
            event: &crate::metal::EventPoint,
        ) -> Result<()> {
            let offsets = self.preflight_slot(slot, rows)?;
            let buffers = [
                Some(&self.streams.qs),
                self.streams.qh.as_ref(),
                Some(&self.streams.scl),
                self.streams.sch.as_ref(),
                Some(&self.streams.sex),
                Some(&self.streams.dv),
            ];
            for (index, ((buffer, span), offset)) in buffers
                .into_iter()
                .zip(self.copy_plan.spans)
                .zip(offsets)
                .enumerate()
            {
                if let (Some(buffer), Some(offset)) = (buffer, offset) {
                    // SAFETY: This exact stream belongs to the gated pool.
                    let lease = unsafe { crate::metal::PublicationLease::new(buffer, event)? };
                    lease.write_rows(
                        self.stream_offsets[index]
                            .checked_add(offset)
                            .context("slab slot offset overflow")?,
                        rows,
                        self.copy_plan.rows,
                        self.copy_plan.row_bytes,
                        span.row_offset,
                        span.row_bytes,
                    )?;
                }
            }
            Ok(())
        }

        /// Replace one physical slot. A caller must invalidate its logical mapping first.
        pub(crate) fn write_slot(&self, slot: usize, rows: &[u8]) -> Result<()> {
            let offsets = self.preflight_slot(slot, rows)?;
            let buffers = [
                Some(&self.streams.qs),
                self.streams.qh.as_ref(),
                Some(&self.streams.scl),
                self.streams.sch.as_ref(),
                Some(&self.streams.sex),
                Some(&self.streams.dv),
            ];
            // Check every stream before the first destination is changed.
            self.check_writable()?;
            for (index, ((buffer, span), offset)) in buffers
                .into_iter()
                .zip(self.copy_plan.spans)
                .zip(offsets)
                .enumerate()
            {
                if let (Some(buffer), Some(offset)) = (buffer, offset) {
                    buffer.write_strided_rows(
                        self.stream_offsets[index]
                            .checked_add(offset)
                            .context("slab slot offset overflow")?,
                        rows,
                        self.copy_plan.rows,
                        self.copy_plan.row_bytes,
                        span.row_offset,
                        span.row_bytes,
                    )?;
                }
            }
            Ok(())
        }

        /// Check every real stream before the pool invalidates a victim.
        pub(crate) fn preflight_slot(
            &self,
            slot: usize,
            rows: &[u8],
        ) -> Result<[Option<usize>; 6]> {
            self.copy_plan.validate_destinations(self.stream_bytes)?;
            self.copy_plan.slot_offsets(slot, rows.len())
        }

        /// Copy one unpublished physical slot without excluding other slots.
        ///
        /// # Safety
        /// The pool scope must exclude this slot from every encoded/submitted
        /// reader and host alias. Each real stream has one contiguous span per
        /// slot. IQ2 placeholder bindings alias qs/scl; absent qh/sch streams
        /// are neither copied here nor read by IQ2 shaders.
        pub(crate) unsafe fn write_slot_disjoint(&self, slot: usize, rows: &[u8]) -> Result<()> {
            let offsets = self.preflight_slot(slot, rows)?;
            let buffers = [
                Some(&self.streams.qs),
                self.streams.qh.as_ref(),
                Some(&self.streams.scl),
                self.streams.sch.as_ref(),
                Some(&self.streams.sex),
                Some(&self.streams.dv),
            ];
            for (index, ((buffer, span), offset)) in buffers
                .into_iter()
                .zip(self.copy_plan.spans)
                .zip(offsets)
                .enumerate()
            {
                if let (Some(buffer), Some(offset)) = (buffer, offset) {
                    // SAFETY: This is exactly the caller-excluded slot span.
                    unsafe {
                        buffer.write_strided_rows_disjoint(
                            self.stream_offsets[index]
                                .checked_add(offset)
                                .context("slab slot offset overflow")?,
                            rows,
                            self.copy_plan.rows,
                            self.copy_plan.row_bytes,
                            span.row_offset,
                            span.row_bytes,
                        )?;
                    }
                }
            }
            Ok(())
        }

        /// Encode after the caller has written checked selectors and dimensions.
        ///
        /// # Safety
        /// The caller must write `selectors` and `dims` into the corresponding
        /// buffers before the first encode in this batch, then leave them
        /// unchanged until completion. Other buffer bounds are checked here.
        #[allow(clippy::too_many_arguments)] // Explicit MSL buffer bindings are the kernel ABI.
        pub(crate) unsafe fn encode(
            &self,
            batch: &mut Batch,
            d: GemvDescriptor,
            selectors: &[u32],
            x: &Buffer,
            sel: &Buffer,
            dims: &Buffer,
            out: &Buffer,
        ) -> Result<()> {
            ensure!(
                d.geometry == self.geometry && selectors.len() == d.pairs,
                "IQ_K dispatch geometry differs from loaded projection"
            );
            ensure!(
                selectors.iter().all(|&e| (e as usize) < self.experts),
                "IQ_K selector refers to an unloaded expert"
            );
            // SAFETY: Checked host selectors match the device values by the
            // caller's contract, including the published-slot requirement.
            unsafe { self.encode_device_selectors(batch, d, x, sel, dims, out) }
        }

        /// Encode selectors produced by an earlier operation in this batch.
        ///
        /// # Safety
        /// Every device selector must address a published physical slot below
        /// this projection's slot count. Dimensions must equal `d.dims`, and
        /// all producers must precede this dispatch without concurrent writes.
        pub(crate) unsafe fn encode_device_selectors(
            &self,
            batch: &mut Batch,
            d: GemvDescriptor,
            x: &Buffer,
            sel: &Buffer,
            dims: &Buffer,
            out: &Buffer,
        ) -> Result<()> {
            ensure!(
                d.geometry == self.geometry,
                "IQ_K dispatch geometry differs from loaded projection"
            );
            ensure!(
                x.byte_len() >= d.input_bytes()?
                    && out.byte_len() >= d.output_bytes()?
                    && sel.byte_len() >= d.pairs * 4
                    && dims.byte_len() >= 8,
                "IQ_K dispatch buffer is shorter than its geometry"
            );
            let mut bindings = vec![
                Binding {
                    buffer: x,
                    offset: 0,
                },
                Binding {
                    buffer: &self.streams.qs,
                    offset: self.stream_offsets[0],
                },
            ];
            if let Some(qh) = &self.streams.qh {
                bindings.push(Binding {
                    buffer: qh,
                    offset: self.stream_offsets[1],
                });
            }
            bindings.push(Binding {
                buffer: &self.streams.scl,
                offset: self.stream_offsets[2],
            });
            if let Some(sch) = &self.streams.sch {
                bindings.push(Binding {
                    buffer: sch,
                    offset: self.stream_offsets[3],
                });
            }
            bindings.extend_from_slice(&[
                Binding {
                    buffer: &self.streams.sex,
                    offset: self.stream_offsets[4],
                },
                Binding {
                    buffer: &self.streams.dv,
                    offset: self.stream_offsets[5],
                },
                Binding {
                    buffer: &self.streams.table,
                    offset: 0,
                },
                Binding {
                    buffer: sel,
                    offset: 0,
                },
                Binding {
                    buffer: dims,
                    offset: 0,
                },
                Binding {
                    buffer: out,
                    offset: 0,
                },
            ]);
            // SAFETY: `new` validated every stream's exact size; descriptor and
            // selectors are checked above. The caller owns selector/dims bytes.
            unsafe { batch.dispatch_1d(&self.pipeline, &bindings, d.grid_threads, d.group_threads) }
        }

        /// Run the pair-order GEMV. Inputs and output are exact fp16 byte arrays.
        pub fn run(
            &self,
            metal: &Metal,
            x_fp16: &[u8],
            selected: &[u32],
            pairs_per_token: usize,
        ) -> Result<Vec<u8>> {
            let d = GemvDescriptor::new(self.geometry, selected.len(), pairs_per_token)?;
            ensure!(
                x_fp16.len() == d.input_bytes()?,
                "IQ_K input fp16 byte count differs from token geometry"
            );
            ensure!(
                selected.iter().all(|&e| (e as usize) < self.experts),
                "IQ_K selector refers to an unloaded expert"
            );
            let x = uploaded(metal, x_fp16)?;
            let selector_bytes: Vec<u8> = selected.iter().flat_map(|e| e.to_le_bytes()).collect();
            let sel = uploaded(metal, &selector_bytes)?;
            let dims_bytes: Vec<u8> = d.dims.iter().flat_map(|v| v.to_le_bytes()).collect();
            let dims = uploaded(metal, &dims_bytes)?;
            let out = metal.shared_buffer(d.output_bytes()?)?;
            let mut bindings = vec![
                Binding {
                    buffer: &x,
                    offset: 0,
                },
                Binding {
                    buffer: &self.streams.qs,
                    offset: self.stream_offsets[0],
                },
            ];
            if let Some(qh) = &self.streams.qh {
                bindings.push(Binding {
                    buffer: qh,
                    offset: self.stream_offsets[1],
                });
            }
            bindings.push(Binding {
                buffer: &self.streams.scl,
                offset: self.stream_offsets[2],
            });
            if let Some(sch) = &self.streams.sch {
                bindings.push(Binding {
                    buffer: sch,
                    offset: self.stream_offsets[3],
                });
            }
            bindings.extend_from_slice(&[
                Binding {
                    buffer: &self.streams.sex,
                    offset: self.stream_offsets[4],
                },
                Binding {
                    buffer: &self.streams.dv,
                    offset: self.stream_offsets[5],
                },
                Binding {
                    buffer: &self.streams.table,
                    offset: 0,
                },
                Binding {
                    buffer: &sel,
                    offset: 0,
                },
                Binding {
                    buffer: &dims,
                    offset: 0,
                },
                Binding {
                    buffer: &out,
                    offset: 0,
                },
            ]);
            let mut batch: Batch = metal.batch()?;
            // SAFETY: Geometry and selector checks above bound every kernel read;
            // the stream byte counts come from validated row slices, and all
            // bindings follow the fixed IQ2_K or IQ3_K MSL signature.
            unsafe {
                batch.dispatch_1d(&self.pipeline, &bindings, d.grid_threads, d.group_threads)?;
            }
            batch.submit().wait().context("IQ_K GEMV command")?;
            let mut result = vec![0; d.output_bytes()?];
            out.read_bytes(0, &mut result)?;
            Ok(result)
        }
    }
}

#[cfg(target_os = "macos")]
pub use gpu::IqkGemv;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host_copy::copy_strided_rows;

    #[test]
    fn stream_split_preserves_each_byte_and_expert_order() {
        for codec in [Codec::Iq2K, Codec::Iq3K] {
            for width in [768, 2560] {
                let g = Geometry::new(codec, width, 16).unwrap();
                let n = g.stream_lengths();
                let mut split = SplitStreams::new(g);
                for expert in 0..2u8 {
                    let mut rows = Vec::new();
                    for row in 0..16u8 {
                        for (stream, count) in [n.qs, n.qh, n.scl, n.sch, n.sex, n.dv]
                            .into_iter()
                            .enumerate()
                        {
                            rows.extend(std::iter::repeat_n(
                                expert * 32 + row + stream as u8,
                                count,
                            ));
                        }
                    }
                    split.append_rows(&rows).unwrap();
                }
                assert_eq!(split.experts, 2);
                for (stream, size, bytes) in [
                    (0, n.qs, &split.qs),
                    (1, n.qh, &split.qh),
                    (2, n.scl, &split.scl),
                    (3, n.sch, &split.sch),
                    (4, n.sex, &split.sex),
                    (5, n.dv, &split.dv),
                ] {
                    assert_eq!(bytes.len(), 32 * size);
                    for (row, chunk) in bytes.chunks(size.max(1)).enumerate() {
                        if size > 0 {
                            assert!(
                                chunk.iter().all(
                                    |&b| b == (row / 16) as u8 * 32 + (row % 16) as u8 + stream
                                )
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn direct_slot_copy_matches_split_streams_for_all_released_geometries() {
        for codec in [Codec::Iq2K, Codec::Iq3K] {
            for width in [768, 2560] {
                for out in [16, 640, 768, 2560] {
                    let geometry = Geometry::new(codec, width, out).unwrap();
                    let plan = RelayoutSlotCopyPlan::new(geometry, 3).unwrap();
                    let source = |seed: u32| -> Vec<u8> {
                        let mut state = seed ^ ((width as u32) << 4) ^ out as u32;
                        (0..out * geometry.row_bytes())
                            .map(|_| {
                                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                                (state >> 24) as u8
                            })
                            .collect()
                    };
                    let first = source(11);
                    let replaced = source(73);
                    let last = source(191);
                    assert_ne!(&first[..256], &first[256..512]);
                    let empty = vec![0; out * geometry.row_bytes()];
                    let mut streams: [Option<Vec<u8>>; 6] = std::array::from_fn(|index| {
                        let span = plan.spans[index];
                        (span.row_bytes > 0).then(|| vec![0; 3 * span.slot_bytes])
                    });
                    plan.validate_destinations(
                        streams
                            .each_ref()
                            .map(|stream| stream.as_ref().map(Vec::len)),
                    )
                    .unwrap();
                    for (slot, payload) in [(2, &first), (0, &replaced), (2, &last)] {
                        let offsets = plan.slot_offsets(slot, payload.len()).unwrap();
                        for (index, offset) in offsets.into_iter().enumerate() {
                            if let Some(offset) = offset {
                                let span = plan.spans[index];
                                let destination = streams[index].as_mut().unwrap();
                                copy_strided_rows(
                                    &mut destination[offset..offset + span.slot_bytes],
                                    payload,
                                    out,
                                    plan.row_bytes,
                                    span.row_offset,
                                    span.row_bytes,
                                )
                                .unwrap();
                            }
                        }
                    }
                    let mut reference = SplitStreams::new(geometry);
                    for payload in [&replaced, &empty, &last] {
                        reference.append_rows(payload).unwrap();
                    }
                    let expected = [
                        Some(&reference.qs),
                        (!reference.qh.is_empty()).then_some(&reference.qh),
                        Some(&reference.scl),
                        (!reference.sch.is_empty()).then_some(&reference.sch),
                        Some(&reference.sex),
                        Some(&reference.dv),
                    ];
                    for (actual, expected) in streams.iter().zip(expected) {
                        assert_eq!(
                            actual.as_ref(),
                            expected,
                            "direct IQ_K slot copy differed for {codec:?} width {width}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn direct_slot_copy_rejects_invalid_geometry_before_writing() {
        let geometry = Geometry::new(Codec::Iq3K, 768, 16).unwrap();
        let plan = RelayoutSlotCopyPlan::new(geometry, 3).unwrap();
        let source = vec![9u8; 16 * geometry.row_bytes()];
        assert!(plan.slot_offsets(3, source.len()).is_err());
        assert!(plan.slot_offsets(1, source.len() - 1).is_err());
        assert!(plan.slot_offsets(1, source.len() + 1).is_err());
        let mut lengths = plan
            .spans
            .map(|span| (span.row_bytes > 0).then_some(span.slot_bytes * plan.slots));
        lengths[0] = Some(lengths[0].unwrap() - 1);
        assert!(plan.validate_destinations(lengths).is_err());
        let span = plan.spans[0];
        let mut destination = vec![0xa5u8; span.slot_bytes];
        for (rows, stride, offset, width) in [
            (0, plan.row_bytes, span.row_offset, span.row_bytes),
            (16, plan.row_bytes, span.row_offset, 0),
            (16, plan.row_bytes - 1, span.row_offset, span.row_bytes),
            (16, plan.row_bytes, plan.row_bytes, span.row_bytes),
            (16, plan.row_bytes, span.row_offset, span.row_bytes + 1),
            (17, plan.row_bytes, span.row_offset, span.row_bytes),
            (16, plan.row_bytes, usize::MAX, span.row_bytes),
            (usize::MAX, plan.row_bytes, span.row_offset, span.row_bytes),
        ] {
            assert!(
                copy_strided_rows(&mut destination, &source, rows, stride, offset, width).is_err()
            );
            assert!(destination.iter().all(|&byte| byte == 0xa5));
        }
        let overflow = RelayoutSlotCopyPlan::new(geometry, usize::MAX).unwrap();
        assert!(overflow.slot_offsets(usize::MAX - 1, source.len()).is_err());
        assert!(overflow.validate_destinations([Some(0); 6]).is_err());
        assert!(destination.iter().all(|&byte| byte == 0xa5));
    }

    #[test]
    fn launch_geometry_and_guard() {
        assert!(Geometry::new(Codec::Iq2K, 2048, 16).is_err());
        assert!(Geometry::new(Codec::Iq3K, 768, 15).is_err());
        let small = Geometry::new(Codec::Iq3K, 768, 32).unwrap();
        let d = GemvDescriptor::new(small, 4, 2).unwrap();
        assert_eq!(
            (d.group_count, d.grid_threads, d.group_threads),
            (8, 192, 24)
        );
        assert_eq!(
            (d.input_bytes().unwrap(), d.output_bytes().unwrap()),
            (3072, 256)
        );
        let large = Geometry::new(Codec::Iq2K, 2560, 16).unwrap();
        assert_eq!(large.row_bytes(), 760);
        assert_eq!(large.threads_per_threadgroup(), 80);
        assert!(GemvDescriptor::new(large, 3, 2).is_err());
        assert!(GemvDescriptor::new(large, u32::MAX as usize + 1, 1).is_err());
    }

    #[test]
    fn fp16_tables_keep_exact_integers() {
        assert_eq!(f16::from_f32(1.0).to_bits(), 0x3c00);
        assert_eq!(f16::from_f32(-8.0).to_bits(), 0xc800);
        assert_eq!(f16::from_f32(32.0).to_bits(), 0x5000);
        assert_eq!(Codec::Iq2K.value_table_fp16().len(), 16);
        assert_eq!(Codec::Iq3K.value_table_fp16().len(), 32);
    }
}
