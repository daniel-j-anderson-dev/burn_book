use burn_book::image_classifier::ImageClassifierConfiguration;

use burn::backend::Wgpu;

type Backend = Wgpu<f32, i32>;
type Device = <Backend as burn::prelude::Backend>::Device;

const DIGIT_CLASS_COUNT: usize = 10;
const MNIST_HIDDEN_LAYER_SIZE: usize = 512;

fn main() {
    let device = Device::default();

    let mnist_classifier =
        ImageClassifierConfiguration::new(DIGIT_CLASS_COUNT, MNIST_HIDDEN_LAYER_SIZE)
            .init::<Backend>(&device);

    println!("{}", mnist_classifier)
}
