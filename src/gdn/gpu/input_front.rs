use super::*;
use crate::kquant::DeviceMatrix;

pub(super) const WIDTHS: [usize; 4] = [CONV, VALUE, HEADS, HEADS];

pub(super) struct StackedFront {
    pub(super) matrices: [DeviceMatrix; 4],
    pipeline: Pipeline,
}

impl StackedFront {
    pub(super) fn new(metal: &Metal, projections: [&DenseMatvec; 4]) -> Result<Self> {
        let matrices = projections.map(DenseMatvec::shared_matrix);
        for (matrix, rows) in matrices.iter().zip(WIDTHS) {
            matrix.check(WeightFormat::Q6K, [rows, HIDDEN])?;
        }
        let source = format!(
            "{}\n{}",
            crate::kquant::METAL_SOURCE,
            include_str!("../../../kernels/metal/gdn_input_front.metal")
        );
        let pipeline = metal.compile_msl_strict(&source, "qwen4_gdn_input_front_q6_k")?;
        ensure!(
            pipeline.max_threads_per_threadgroup() >= 64,
            "GDN input front requires a 64-thread Metal group"
        );
        Ok(Self { matrices, pipeline })
    }

    pub(super) fn encode(
        &self,
        batch: &mut Batch,
        hidden: &Buffer,
        outputs: [&Buffer; 4],
    ) -> Result<()> {
        ensure!(
            hidden.byte_len() == HIDDEN * 2,
            "GDN input front hidden width differs"
        );
        for (index, (output, rows)) in outputs.iter().zip(WIDTHS).enumerate() {
            ensure!(
                output.byte_len() >= rows * 2,
                "GDN input front output {index} is too short"
            );
            ensure!(
                !output.shares_allocation(hidden)
                    && outputs[..index]
                        .iter()
                        .all(|other| !output.shares_allocation(other))
                    && self
                        .matrices
                        .iter()
                        .all(|matrix| !output.shares_allocation(matrix.binding().buffer)),
                "GDN input front output {index} aliases an input or output"
            );
        }
        let bindings = [
            self.matrices[0].binding(),
            self.matrices[1].binding(),
            self.matrices[2].binding(),
            self.matrices[3].binding(),
            Binding {
                buffer: hidden,
                offset: 0,
            },
            Binding {
                buffer: outputs[0],
                offset: 0,
            },
            Binding {
                buffer: outputs[1],
                offset: 0,
            },
            Binding {
                buffer: outputs[2],
                offset: 0,
            },
            Binding {
                buffer: outputs[3],
                offset: 0,
            },
        ];
        // SAFETY: Validated Q6 shapes and disjoint BF16 extents cover the
        // fixed grid. Batch checks device identity and retains all bindings.
        unsafe { batch.dispatch_1d(&self.pipeline, &bindings, 2060 * 64, 64)? };
        Ok(())
    }
}

pub(super) fn preferred_for_device(name: &str) -> bool {
    name == "Apple M3 Max"
}

#[cfg(test)]
mod tests {
    use super::preferred_for_device;

    #[test]
    fn device_selection_keeps_unqualified_devices_on_separate_projections() {
        assert!(preferred_for_device("Apple M3 Max"));
        for name in [
            "Apple M1 Max",
            "Apple M2 Max",
            "Apple M3",
            "Apple M3 Pro",
            "Apple M4 Max",
            "",
            "Unknown GPU",
        ] {
            assert!(!preferred_for_device(name), "{name}");
        }
    }
}
