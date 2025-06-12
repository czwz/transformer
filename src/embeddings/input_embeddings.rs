use candle::{Device, Result, Tensor};
use candle_nn::{Dropout};

pub struct PosEmbeddings {
    pos_embeddings: Tensor,
    seq_len: usize,
    d_model: usize,
    dropout: Dropout,
}

impl PosEmbeddings {
    pub fn new(seq_len: usize, d_model: usize, dropout: Dropout, device: &Device) -> Result<Self> {
        let pos = Tensor::arange(0_f32, seq_len as f32, device)?;
        let denom = ((Tensor::arange_step(0_f32, d_model as f32, 2_f32, device)?
            * (-1000000.0_f64.ln() / d_model as f64))?)
            .exp()?;
        let pos = pos.unsqueeze(1)?;
        let denom = denom.unsqueeze(0)?;
        let tmp = (pos.matmul(&denom))?;
        let even_embeds = tmp.sin()?;
        let odd_embeds = tmp.cos()?;
    }
}