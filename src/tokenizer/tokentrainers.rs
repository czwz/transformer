use tokenizers::{
    models::{
        wordlevel::{WordLevel, WordLevelTrainer},
        TrainerWrapper
    },
    normalizers::{
        utils::Sequence as NormalizeSequence, Lowercase, StripAccents, NFD },
        pre_tokenizers::{ digits::Digits, sequence::Sequence, whitespace::Whitespace },
        processors::template::TemplateProcessing,
        tokenizer::Tokenizer, AddedToken
};


#[allow(dead_code)]
pub fn word_level_tokenizer() -> Result<(), tokenizers::Error> {
    let mut tokenzier = Tokenizer::new(WordLevel::default());
    let normalizer = NormalizeSequence::new(vec![NFD.into(), StripAccents.into(), Lowercase.into()]);
    tokenzier.with_normalizer(Some(normalizer));

    let pre_tokenizer = Sequence::new(vec![Whitespace.into(), Digits::default().into()]);
    tokenzier.with_pre_tokenizer(Some(pre_tokenizer));

    let mut trainer: TrainerWrapper = WordLevelTrainer::builder()
        .vocab_size(30_522)
        .special_tokens(vec![
            AddedToken::from("[UNK]", true),
            AddedToken::from("[CLS]", true),
            AddedToken::from("[SEP]", true),
            AddedToken::from("[PAD]", true),
            AddedToken::from("[MASK]", true),
        ])
        .build()?
        .into();

    let files = vec![
        "./src/tokenizer/wikitext-103-raw/wiki.train.raw".into(),
        "./src/tokenizer/wikitext-103-raw/wiki.test.raw".into(),
        "./src/tokenizer/wikitext-103-raw/wiki.valid.raw".into()
    ];
    
    let _ = tokenzier.train_from_files(&mut trainer, files);
    tokenzier.with_post_processor(
        Some(
            TemplateProcessing::builder()
                .try_single("[CLS] $A [SEP]")
                .unwrap()
                .try_pair("[CLS] $A [SEP] $B:1 [SEP]:1")
                .unwrap()
                .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
                .build()
                .unwrap()
        ),
    );
    tokenzier.save("./wordlevel-wiki.json", false)?;

    let encoding = tokenzier.encode(("Hello World", "I'm raven."), true)?;
    println!("tok: {:?}", encoding.get_tokens());

    Ok(())
}