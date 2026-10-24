//! Native model execution, package access and Metal kernels.

#[cfg(target_os = "macos")]
pub(crate) mod adaptive_memory;

#[cfg(target_os = "macos")]
pub mod ane;

#[cfg(target_os = "macos")]
mod decode_handoff;
pub mod decoder;
pub mod dense_batch;
#[cfg(target_os = "macos")]
pub(crate) mod device_status;
pub mod embedding;
#[cfg(target_os = "macos")]
pub mod engine;
pub mod experts;
pub mod gdn;
pub mod gdn_batch;
#[cfg(target_os = "macos")]
pub mod generation;
pub mod hc;
pub mod hc_batch;
mod host_copy;
#[cfg(target_os = "macos")]
pub mod host_memory;
pub mod iqk;
pub mod kquant;
pub mod kvarn;
pub mod kvarn_pending;
pub mod memory;
#[cfg(target_os = "macos")]
pub mod metal;
pub mod mlp;
pub mod mlp_batch;
pub mod model;
pub mod package;
pub mod ple;
pub mod ple_batch;
pub mod prefill_capacity;
pub mod qsa_attention;
pub mod qsa_batch_attention;
pub mod qsa_batch_projection;
pub mod qsa_index;
pub mod qsa_projection;
pub mod routing;
pub mod sampling;
#[cfg(target_os = "macos")]
pub mod server;
pub mod tokenizer;
