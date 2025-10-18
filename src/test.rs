use crate::image_classifier::{self, ImageClassifierConfiguration, training::TrainingConfig};

use std::path::Path;

use burn::{
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice},
    data::dataset::Dataset,
    optim::AdamConfig,
};

type Backend = Wgpu<f32, i32>;

#[test]
fn train() {
    let device = WgpuDevice::default();
    let artifact_path = Path::new("./artifacts/");
    image_classifier::training::train::<Autodiff<Backend>>(
        artifact_path,
        TrainingConfig::new(
            ImageClassifierConfiguration::new(10, 512),
            AdamConfig::new(),
        ),
        device.clone(),
    )
    .unwrap();
}

#[test]
fn infer() {
    let device = WgpuDevice::default();
    let artifact_path = Path::new("./artifacts/");
    image_classifier::inference::infer::<Backend>(
        artifact_path,
        device.clone(),
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    )
    .expect("Must run train test first");
}
