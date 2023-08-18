#[derive(Eq, PartialEq, Hash)]
pub enum Chat {
    Gpt3dot5TurboToken4k,
    Gpt3dot5TurboToken16k,
    Gpt4Token8k,
    Gpt4Token32k,
}

#[derive(Eq, PartialEq, Hash)]
pub enum Embed {
    TextEmbeddingAda002,
}
