//! Soul chess engine.
//!
//! - `core`: Board representation, move generation, and static rules.
//! - `engine`: Search, evaluation, and time management.
//! - `protocols`: UCI and XBoard communication.
//! - `tools`: Benchmarking, perft, and dataset utilities.
//! - `weave`: Hardware-specific SIMD intrinsics.
//! - `hugepages`: Huge-page-backed allocation for the transposition table.
//! - `numa`: NUMA topology detection and thread binding.

#![feature(adt_const_params)]
#![feature(stdarch_const_x86)]
#![feature(stmt_expr_attributes)]
#![feature(const_trait_impl)]
#![feature(const_cmp)]
#![feature(const_convert)]
#![feature(const_ops)]
#![feature(likely_unlikely)]
#![feature(sync_nonpoison)]
#![feature(nonpoison_mutex)]
// Allowed because we inherently pass large parameter structs
#![allow(clippy::too_many_arguments)]
// Allowed because domain acronyms
#![allow(clippy::upper_case_acronyms)]

/// `0.6.13-dev-a1b2c3d`, or `0.6.13-a1b2c3d` on a release tag.
pub const VERSION: &str = env!("SOUL_VERSION");

pub mod cli;
pub mod color;
pub mod core;
pub mod engine;
pub mod hugepages;
pub mod numa;
pub mod protocols;
pub mod tools;
pub mod weave;
