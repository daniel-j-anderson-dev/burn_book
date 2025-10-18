pub mod training;
pub mod inference;

use burn::{
    nn::{
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    prelude::*,
};

#[derive(Debug, Config)]
pub struct ImageClassifierConfiguration {
    class_count: usize,
    hidden_layer_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}
impl ImageClassifierConfiguration {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ImageClassifier<B> {
        ImageClassifier {
            convolutional_layers: [
                Conv2dConfig::new([1, 8], [3, 3]).init(device),
                Conv2dConfig::new([8, 16], [3, 3]).init(device),
            ],
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: Relu::new(),
            linear_layers: [
                LinearConfig::new(16 * 8 * 8, self.hidden_layer_size).init(device),
                LinearConfig::new(self.hidden_layer_size, self.class_count).init(device),
            ],
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

/// Our goal will be to create a basic convolutional neural network used for image classification.
/// - We will keep the model simple by using
///   - two convolution layers followed
///   - by two linear layers,
///   - some pooling
///   - use dropout to improve training performance
///   - and some ReLU activations
#[derive(Debug, Module)]
pub struct ImageClassifier<B: Backend> {
    convolutional_layers: [Conv2d<B>; 2],
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    linear_layers: [Linear<B>; 2],
    activation: Relu,
}
impl<B: Backend> ImageClassifier<B> {
    /// # Shape
    /// - `images`: `[batch_size, row_count, column_count]`
    /// - return value: `[batch_size, class_count]`
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, row_count, column_count] = images.dims();

        let x = images.reshape([batch_size, 1, row_count, column_count]);

        let x = self.convolutional_layers[0].forward(x);
        let x = self.dropout.forward(x);
        let x = self.convolutional_layers[1].forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x);
        let x = x.reshape([batch_size, 16 * 8 * 8]);
        let x = self.linear_layers[0].forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear_layers[1].forward(x)
    }
}
