use crate::image_classifier::ImageClassifier;

use burn::{prelude::*, train::ClassificationOutput};

impl<B: Backend> ImageClassifier<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        
        todo!()
    }
}
