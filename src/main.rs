use candle_core::Device;
use tokenizers::tokenizer::Tokenizer;

mod embeddings;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let tokenizer = Tokenizer::from_file("./wordlevel-wiki.json")?;
    let encoding = tokenizer.encode(
        ("Hello, world", "I am raven"), 
        true
    )?;

    println!("tok: {:?}", encoding.get_tokens());
    println!("ids: {:?}", encoding.get_ids());

    let e = embeddings::postional_embeddings::PositionalEmbeddings::new(
        512,
        768,
        candle_nn::Dropout::new(0.1),
        &Device::Cpu,
    ).unwrap();

    println!("Positional Embeddings: {:?}", e.positional_embeddings);

    Ok(())
}
