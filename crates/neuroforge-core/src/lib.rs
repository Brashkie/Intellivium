//! NeuroForge core — motor de deep learning en Rust puro (sin C/C++).
//!
//! Tres piezas:
//! - `tape`: autograd reverse-mode sobre una cinta.
//! - `nn`:   capas densas + modelo secuencial + train/predict.
//! - `rng`:  RNG mínimo para inicialización.

pub mod nn;
pub mod rng;
pub mod tape;

pub use nn::{Activation, Dense, Model};
pub use rng::Rng;
pub use tape::Tape;
