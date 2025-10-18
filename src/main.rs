use burn_book::image_classifier::{ImageClassifierConfiguration, training::TrainingConfig};
use mnist_dataset::DigitClass;

use std::path::Path;

use anyhow::Error;
use burn::{
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice},
    optim::AdamConfig,
};

type Backend = Wgpu<f32, i32>;

fn main() -> Result<(), Error> {
    let device = WgpuDevice::default();
    let artifact_path = Path::new("./artifacts/guide");
    burn_book::image_classifier::training::train::<Autodiff<Backend>>(
        artifact_path,
        TrainingConfig::new(
            ImageClassifierConfiguration::new(DigitClass::COUNT, 512),
            AdamConfig::new(),
        ),
        device,
    )
}
