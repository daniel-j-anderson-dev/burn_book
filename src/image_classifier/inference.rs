use crate::{batcher::MnistBatcher, image_classifier::training::TrainingConfig};

use core::error::Error;
use std::path::Path;

use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
    record::{CompactRecorder, Recorder},
};
pub fn infer<B: Backend>(
    artifact_path: impl AsRef<Path>,
    device: B::Device,
    item: MnistItem,
) -> Result<(), Box<dyn Error>> {
    let artifact_path = artifact_path.as_ref();

    let config = TrainingConfig::load(artifact_path.join("training_configuration.json"))?;
    let record = CompactRecorder::new().load(artifact_path.join("model"), &device)?;

    let model = config
        .image_classifier
        .init::<B>(&device)
        .load_record(record);

    let label = item.label;
    let batch = MnistBatcher.batch(vec![item], &device);
    let output = model.forward(batch.images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("predicted: {}, expected: {}", label, predicted);

    Ok(())
}
