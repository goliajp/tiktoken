mod bpe;
mod encoding;
mod models;
mod price;

use bpe::CoreBpe;
use encoding::Encoding;

pub fn count_text(chat_model_name: &str, text: &str) -> isize {
    let enc = Encoding::get_by_chat_model(chat_model_name).expect("get encoding failed");
    let bpe = CoreBpe::new(
        enc.merging_ranks,
        enc.special_tokens,
        enc.dict.get_regex_pattern(),
    )
    .expect("get bpe failed");
    let tokens = bpe.encode_native(text).0;
    tokens.len() as isize
}

pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
}

pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub name: Option<String>,
}

pub fn count_request(request: &ChatRequest) -> isize {
    let enc = Encoding::get_by_chat_model(&request.model).expect("get encoding failed");
    let bpe = CoreBpe::new(
        enc.merging_ranks,
        enc.special_tokens,
        enc.dict.get_regex_pattern(),
    )
    .expect("get bpe failed");
    let per_message: isize = 3;
    let per_name: isize = 1;
    let per_request: isize = 3;
    let mut count = per_request;
    for message in &request.messages {
        count += per_message;
        count += bpe.encode_native(message.role.as_str()).0.len() as isize;
        count += bpe.encode_native(message.content.as_str()).0.len() as isize;
        if let Some(name) = &message.name {
            count += per_name;
            count += bpe.encode_native(name).0.len() as isize
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = count_text("gpt-4", "hello, openai");
        assert_eq!(result, 2);
    }
}
