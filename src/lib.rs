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
// A search function takes the node's state as arguments, and there is a lot of it.
#![allow(clippy::too_many_arguments)]
// Domain acronyms keep their capitals.
#![allow(clippy::upper_case_acronyms)]

/// `0.6.13-dev-a1b2c3d`, or `0.6.13-a1b2c3d` on a release tag.
pub const VERSION: &str = env!("SOUL_VERSION");

pub mod cli;
pub mod color;
pub mod core;
pub mod engine;
mod hugepages;
mod numa;
pub mod protocols;
pub mod tools;
mod weave;
