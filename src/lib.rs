#![recursion_limit = "256"]

pub mod batcher;
pub mod image_classifier;
// pub use mnist_dataset::burn_interop::{MnistBatch, MnistBatcher};

#[cfg(test)]
mod test;