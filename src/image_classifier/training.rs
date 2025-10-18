use crate::{batcher::*, image_classifier::*};
use anyhow::Error;

use std::{fs, io, path::Path};

use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    nn::loss::CrossEntropyLossConfig,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
        metric::{AccuracyMetric, LossMetric},
    },
};

impl<B: Backend> ImageClassifier<B> {
    /// returns the output of forward and the [CrossEntropyLoss]
    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        // send the image through our network
        let output = self.forward(images);

        // calculate the cross entropy loss of this output
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for ImageClassifier<B> {
    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let classification_output = self.forward_classification(batch.images, batch.labels);
        let gradients = classification_output.loss.backward();
        TrainOutput::new(self, gradients, classification_output)
    }
}
impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for ImageClassifier<B> {
    fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.labels)
    }
}
#[derive(Config)]
pub struct TrainingConfig {
    pub image_classifier: ImageClassifierConfiguration,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub number_of_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub number_of_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_or_clear_directory(artifact_path: impl AsRef<Path>) -> Result<(), io::Error> {
    let artifact_path = artifact_path.as_ref();
    let _ = fs::remove_dir_all(artifact_path);
    fs::create_dir_all(artifact_path)?;
    Ok(())
}

pub fn train<B: AutodiffBackend>(
    artifact_path: impl AsRef<Path>,
    training_configuration: TrainingConfig,
    device: B::Device,
) -> Result<(), Error> {
    let artifact_path = artifact_path.as_ref();
    create_or_clear_directory(artifact_path)?;
    training_configuration.save(artifact_path.join("training_configuration.json"))?;

    B::seed(training_configuration.seed);

    let dataloader_train = DataLoaderBuilder::new(MnistBatcher)
        .batch_size(training_configuration.batch_size)
        .shuffle(training_configuration.seed)
        .num_workers(training_configuration.number_of_workers)
        .build(MnistDataset::train());

    let dataloader_test = DataLoaderBuilder::new(MnistBatcher)
        .batch_size(training_configuration.batch_size)
        .shuffle(training_configuration.seed)
        .num_workers(training_configuration.number_of_workers)
        .build(MnistDataset::test());

    let learner = LearnerBuilder::<B, _, _, _, _, _>::new(artifact_path)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(training_configuration.number_of_epochs)
        .summary()
        .build(
            training_configuration.image_classifier.init(&device),
            training_configuration.optimizer.init(),
            training_configuration.learning_rate,
        );

    let trained_model = learner.fit(dataloader_train, dataloader_test);

    trained_model.save_file(artifact_path.join("/model"), &CompactRecorder::new())?;

    Ok(())
}
