use candle_core::{Device, Result, Tensor};
use candle_nn::{Dropout};

#[allow(dead_code)]
pub struct PositionalEmbeddings {
    positional_embeddings: Tensor,
    seq_len: usize,
    d_model: usize,
    dropout: Dropout,
}

#[allow(dead_code)]
impl PositionalEmbeddings {
    pub fn new(seq_len: usize, d_model: usize, dropout: Dropout, device: &Device) -> Result<Self> {
        // Shape: (seq_len,)
        let positions = Tensor::arange(0_f32, seq_len as f32, device)?;

        // Shape: (d_model,)
        let denomiators = (
            (
                Tensor::arange_step(
                    0_f32,
                    d_model as f32,
                    2_f32,
                    device
                )?
                * (-10000.0_f64.ln() / d_model as f64)
            )?
        ).exp()?;

        // Shape: (seq_len, 1)
        let positions = positions.unsqueeze(1)?;

        // Shape: (1, d_model)
        let denomiators = denomiators.unsqueeze(0)?;
        
        // Shape: (seq_len, d_model)
        let temp = (positions.matmul(&denomiators))?;

        let even_embeddings = temp.sin()?;
        let odd_embeddings = temp.cos()?;
        let even_col_0 = even_embeddings.get_on_dim(1, 0)?;
        let odd_col_0 = odd_embeddings.get_on_dim(1, 0)?;

        let mut positional_embeddings = Tensor::cat(&[&even_col_0, &odd_col_0], 0)?;
        for col in 1..d_model {
            let even_col = even_embeddings.get_on_dim(1, col)?;
            let odd_col = odd_embeddings.get_on_dim(1, col)?;
            positional_embeddings = Tensor::cat(&[&positional_embeddings, &even_col], 0)?;
            positional_embeddings = Tensor::cat(&[&positional_embeddings, &odd_col], 0)?;
        }

        // Shape: (1, seq_len, d_model)
        positional_embeddings = positional_embeddings
            .reshape((d_model, seq_len))?
            .transpose(0, 1)?
            .unsqueeze(0)?;

        Ok(
            PositionalEmbeddings {
                positional_embeddings,
                seq_len,
                d_model,
                dropout
            }
        )
    }
}