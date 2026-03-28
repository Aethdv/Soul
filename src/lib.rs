//! Soul chess engine.
//!
//! # Architecture
//! - `core`: Board representation, move generation, and static rules.
//! - `engine`: Search, evaluation, and time management.
//! - `protocols`: UCI and XBoard communication.
//! - `tools`: Benchmarking, perft, and dataset utilities.
//! - `weave`: Hardware-specific SIMD intrinsics.

#![feature(adt_const_params)]
#![feature(stdarch_const_x86)]
#![feature(stmt_expr_attributes)]
#![feature(custom_inner_attributes)]
#![feature(sync_unsafe_cell)]
#![feature(const_range)]
#![feature(const_trait_impl)]
#![feature(const_convert)]
#![feature(const_ops)]
#![feature(likely_unlikely)]
#![feature(hint_prefetch)]
#![feature(slice_swap_unchecked)]
#![feature(thread_local)]
// Allowed because we inherently pass large parameter structs
#![allow(clippy::too_many_arguments)]
// Allowed because domain acronyms
#![allow(clippy::upper_case_acronyms)]

pub mod cli;
pub mod core;
pub mod engine;
pub mod protocols;
pub mod tools;
pub mod weave;
