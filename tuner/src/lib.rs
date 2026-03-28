#![feature(custom_inner_attributes)]
// Mathematical implementations heavily rely on standard linear algebra notation (x, y, z, w, m).
#![allow(clippy::many_single_char_names)]
// Optimizer states (c_mu, c_sigma, p_c, p_sigma) precisely match CMA-ES literature.
#![allow(clippy::similar_names)]

pub mod core;
pub mod evaltune;
pub mod searchtune;
