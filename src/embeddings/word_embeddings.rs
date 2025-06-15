use std::collections::HashMap;

use candle_core::{ Device, DType, Result, Tensor };
use candle_nn::{ embedding, Embedding, Module, VarBuilder };

pub struct WordEmbeddings {
    #[allow(dead_code)]
    pub d_model: usize,

    #[allow(dead_code)]
    pub vocab_size: usize,
    pub word_embedding: Embedding,
}

impl WordEmbeddings {
    pub fn new(vocab_size: usize, d_model: usize, device: &Device) -> Result<Self> {
        let mapping = HashMap::from([
            (
                String::from("weight"),
                Tensor::randn(0_f32, 1.0_f32, (vocab_size, d_model), device)?
            )
        ]);
        let builder = VarBuilder::from_tensors(mapping, DType::F32, device);
        let word_embedding = embedding(vocab_size, d_model, builder)?;

        Ok(
            WordEmbeddings {
                d_model,
                vocab_size,
                word_embedding,
            }
        )
    }

    #[allow(dead_code)]
    pub fn forwad(&self, indices: &[u32], device: &Device) -> Result<Tensor> {
        let tensor = Tensor::from_slice(indices, (indices.len(),), device)?;
        self.word_embedding.forward(&tensor)
    }
}