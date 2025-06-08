mod tokenizer;

fn main() {
    if let Err(e) = tokenizer::tokentrainers::word_level_tokenizer() {
        eprintln!("Error: {}", e);
    }
}
