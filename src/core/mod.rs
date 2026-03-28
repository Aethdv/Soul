//! Core chess domain logic.
//!
//! Board representation, move representation, PSQT tables,
//! Zobrist hashing, phase calculation, and spatial primitives.

#[macro_use]
pub mod macros;
pub mod board;
pub mod defs;
pub mod error;
pub mod moves;
pub mod phase;
pub mod primitives;
pub mod psqt;
pub mod util;
pub mod zobrist;
