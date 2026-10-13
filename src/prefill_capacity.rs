//! Explicit row capacity for the shared prefill workspace.

use anyhow::{Result, ensure};

pub const DEFAULT_ROWS: usize = 512;
pub const AUTOMATIC_ROWS: usize = 2048;
pub const MAX_ROWS: usize = 4096;
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
        matches!(rows, DEFAULT_ROWS | AUTOMATIC_ROWS | MAX_ROWS),
        "prefill capacity must be 512, 2048 or 4096"
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
