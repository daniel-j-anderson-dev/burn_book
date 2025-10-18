use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
    tensor,
};

/// - The `0th` index of each [Tensor]: indexes which image/label in this batch
/// - The `1th` index of each `images`: indexes which row of pixels in this image
/// - The `2th` index of each `images`: indexes which pixel in this row
#[derive(Debug, Clone)]
pub struct MnistBatch<B: Backend> {
    /// Shape: `[batch_size, image_height, image_width]`
    pub images: Tensor<B, 3>,

    /// Shape: `[batch_size]`
    pub labels: Tensor<B, 1, Int>,
}

pub struct MnistBatcher;
impl<B: Backend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &<B as Backend>::Device) -> MnistBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 2>::from_data(data, device))
            .map(|tensor| ((tensor / 255) - 0.1307) - 0.3081)
            .map(|tensor| tensor.reshape([1, 28, 28]))
            .collect();

        let labels = items
            .iter()
            .map(|item| [(item.label as i64).elem::<B::IntElem>()])
            .map(|data| Tensor::<B, 1, Int>::from_data(data, device))
            .collect();

        MnistBatch {
            images: Tensor::cat(images, 0),
            labels: Tensor::cat(labels, 0),
        }
    }
}
