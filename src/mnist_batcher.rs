use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
};

#[derive(Debug, Clone)]
pub struct MnistBatcher {}

#[derive(Debug, Clone)]
pub struct MnistBatch<B: Backend> {
    /// Shape: [batch_size, image_height, image_width]
    pub images: Tensor<B, 3, Float>,
    /// Shape: [batch_size]
    pub targets: Tensor<B, 1, Int>,
}

/// Normalize: scale between [0,1] and make the mean=0 and std=1
/// values mean=0.1307,std=0.3081 are from the PyTorch MNIST example
/// https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
fn normalize<B: Backend, const N: usize>(tensor: Tensor<B, N>) -> Tensor<B, N> {
    ((tensor / 255) - 0.1307) / 0.3081
}

impl<B: Backend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBatch<B> {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> MnistBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image).convert::<B::FloatElem>()) // convert the float type to match the current in use by `B`
            .map(|data| normalize(Tensor::<B, 2>::from_data(data, device).reshape([1, 28, 28]))) // shape [batch_size, image_height, image_width]
            .collect();

        let targets = items
            .iter()
            .map(|item| [(item.label as i64).elem::<B::IntElem>()])
            .map(|data| Tensor::<B, 1, Int>::from_data(data, device))
            .collect();

        MnistBatch {
            images: Tensor::cat(images, 0),
            targets: Tensor::cat(targets, 0),
        }
    }
}
