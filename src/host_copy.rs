//! Checked strided host bytes used by stream-major weight storage.

use anyhow::{Context, Result, ensure};

/// Copy one byte span from each source row into contiguous destination rows.
/// Validation completes before the first destination byte is changed.
pub(crate) fn copy_strided_rows(
    destination: &mut [u8],
    source: &[u8],
    rows: usize,
    source_row_bytes: usize,
    source_offset: usize,
    stream_row_bytes: usize,
) -> Result<()> {
    ensure!(rows > 0 && stream_row_bytes > 0, "strided copy is empty");
    ensure!(
        source_offset
            .checked_add(stream_row_bytes)
            .is_some_and(|end| end <= source_row_bytes),
        "strided span exceeds source row"
    );
    ensure!(
        source.len()
            == rows
                .checked_mul(source_row_bytes)
                .context("strided source byte count overflow")?,
        "strided source size differs"
    );
    ensure!(
        destination.len()
            == rows
                .checked_mul(stream_row_bytes)
                .context("strided destination byte count overflow")?,
        "strided destination size differs"
    );
    for (target, row) in destination
        .chunks_exact_mut(stream_row_bytes)
        .zip(source.chunks_exact(source_row_bytes))
    {
        target.copy_from_slice(&row[source_offset..source_offset + stream_row_bytes]);
    }
    Ok(())
}
