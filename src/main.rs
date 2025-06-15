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

    let pe = embeddings::postional_embeddings::PositionalEmbeddings::new(
        512,
        768,
        candle_nn::Dropout::new(0.1),
        &Device::Cpu,
    )?;

    println!("Positional Embeddings: {:?}", pe.positional_embeddings);

    let we = embeddings::word_embeddings::WordEmbeddings::new(
        tokenizer.get_vocab_size(true),
        768,
        &Device::Cpu,
    )?;

    println!("Word Embeddings: {:?}", we.word_embedding);

    Ok(())
}
