//! Explicit row capacity for the shared prefill workspace.

use anyhow::{Result, ensure};

pub const DEFAULT_ROWS: usize = 512;
pub const AUTOMATIC_ROWS: usize = 2048;
/// Largest explicit GPU workspace and physical ANE projection width.
pub const MAX_ROWS: usize = 8192;

/// Select one physical ANE shape from the initial GPU workspace. Later
/// workspace replacement does not change the retained program or cache scope.
pub fn ane_physical_rows(workspace_rows: usize) -> Result<usize> {
    validate_capacity(workspace_rows)?;
    Ok(if workspace_rows == 8192 { 8192 } else { 4096 })
}

/// Admit a wider workspace at a fixed expert count only when its complete
/// estimate, including serving reserves, fits the available budget.
pub fn admitted_rows(wide_component_bytes: u64, budget_bytes: u64) -> usize {
    if wide_component_bytes <= budget_bytes {
        AUTOMATIC_ROWS
    } else {
        DEFAULT_ROWS
    }
}

pub fn validate_capacity(rows: usize) -> Result<()> {
    ensure!(
        matches!(rows, DEFAULT_ROWS | AUTOMATIC_ROWS | 4096 | 8192),
        "prefill capacity must be 512, 2048, 4096 or 8192"
    );
    Ok(())
}

/// The first completed cold-prefill chunk crossing the configured stride.
pub fn first_checkpoint_frontier(stride: usize, capacity: usize) -> Option<usize> {
    if stride == 0 || validate_capacity(capacity).is_err() {
        return None;
    }
    ((stride - 1) / capacity + 1).checked_mul(capacity)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ane_physical_shape_is_selected_from_initial_workspace() {
        for rows in [512, 2048, 4096] {
            assert_eq!(ane_physical_rows(rows).unwrap(), 4096);
        }
        assert_eq!(ane_physical_rows(8192).unwrap(), 8192);
        for rows in [0, 4097, 8191, 8193, usize::MAX] {
            assert!(ane_physical_rows(rows).is_err());
        }
    }

    #[test]
    fn explicit_8192_preserves_ordinary_admission_defaults() {
        assert_eq!(DEFAULT_ROWS, 512);
        assert_eq!(AUTOMATIC_ROWS, 2048);
        assert_eq!(admitted_rows(1, 1), 2048);
        assert_eq!(admitted_rows(2, 1), 512);
        validate_capacity(8192).unwrap();
        assert_eq!(MAX_ROWS, 8192);
        for rows in [4097, 8191, 8193] {
            assert!(validate_capacity(rows).is_err());
        }
        assert_eq!(first_checkpoint_frontier(4096, 8192), Some(8192));
    }
    #[test]
    fn explicit_8192_row_indices_buffers_and_dispatches_fit_metal_uint() {
        let rows = 8192usize;
        let pairs = rows * 10;
        for elements in [
            rows * 12288,
            rows * 10240,
            rows * 6144,
            rows * 2560,
            rows * 2051,
            pairs * 2560,
            pairs * 640,
        ] {
            assert!(elements * 2 < u32::MAX as usize);
        }
        assert_eq!(pairs, 81920);
        assert_eq!(pairs * 2560 * 2, 419_430_400);
        assert_eq!(
            crate::routing::RoutingGeometry::new(rows)
                .unwrap()
                .grid_threads(),
            1_048_576
        );
        for tile in [8usize, 16, 32] {
            let maximum_tiles = 512 + pairs.div_ceil(tile);
            assert!(maximum_tiles * 3 * 4 <= (512 + pairs.div_ceil(8)) * 3 * 4);
            assert!(maximum_tiles * 80 * 128 < u32::MAX as usize);
            let counts = [pairs, 0, 0, 0];
            assert!(counts.into_iter().map(|n| n.div_ceil(tile)).sum::<usize>() <= maximum_tiles);
        }
        let tiles: Vec<_> = (0..rows).step_by(64).collect();
        assert_eq!(tiles.len(), 128);
        assert_eq!(tiles.last().copied().unwrap() + 64, rows);
        assert_eq!(rows * 2051 * 4, 67_207_168);
        assert_eq!(rows * 12288 * 2, 201_326_592);
        for context in [8192usize, 131072, 262144] {
            assert!(context * 1024 < u32::MAX as usize);
            assert!(context < i32::MAX as usize);
        }
        assert_eq!(
            crate::gdn_batch::next_frontier(262144 - 8192, rows).unwrap(),
            262144
        );
        assert!(crate::gdn_batch::next_frontier(u64::MAX, rows).is_err());
    }
    #[test]
    fn fixed_capacity_wider_admission_requires_the_complete_estimate() {
        let estimate = 24 << 30;
        assert_eq!(admitted_rows(estimate, estimate), 2048);
        assert_eq!(admitted_rows(estimate, estimate - 1), 512);
        assert_eq!(admitted_rows(u64::MAX, 32 << 30), 512);
    }
    #[test]
    fn explicit_capacity_supports_4096_without_changing_automatic_admission() {
        for rows in [512, 2048, 4096] {
            validate_capacity(rows).unwrap();
        }
        for rows in [0, 513, 2049, 4097, usize::MAX] {
            assert!(validate_capacity(rows).is_err());
        }
        assert_eq!(admitted_rows(1, 1), 2048);
        assert_eq!(first_checkpoint_frontier(2048, 4096), Some(4096));
    }
    #[test]
    fn checkpoint_frontier_rounds_up_without_changing_chunks() {
        assert_eq!(first_checkpoint_frontier(1024, 512), Some(1024));
        assert_eq!(first_checkpoint_frontier(1024, 2048), Some(2048));
        assert_eq!(first_checkpoint_frontier(1536, 2048), Some(2048));
        assert_eq!(first_checkpoint_frontier(0, 2048), None);
        assert_eq!(first_checkpoint_frontier(usize::MAX, 2048), None);
    }
}
