use super::models::{Chat, Embed};
use lazy_static::lazy_static;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;

// map

lazy_static! {
    pub static ref CHAT_PRICE_INPUT: HashMap<Chat, Decimal> = {
        let mut map = HashMap::new();
        map.insert(Chat::Gpt3dot5TurboToken4k, dec!(0.0015));
        map.insert(Chat::Gpt3dot5TurboToken16k, dec!(0.003));
        map.insert(Chat::Gpt4Token8k, dec!(0.03));
        map.insert(Chat::Gpt4Token32k, dec!(0.06));
        map
    };
    pub static ref CHAT_PRICE_OUTPUT: HashMap<Chat, Decimal> = {
        let mut map = HashMap::new();
        map.insert(Chat::Gpt3dot5TurboToken4k, dec!(0.002));
        map.insert(Chat::Gpt3dot5TurboToken16k, dec!(0.004));
        map.insert(Chat::Gpt4Token8k, dec!(0.06));
        map.insert(Chat::Gpt4Token32k, dec!(0.12));
        map
    };
    pub static ref EMBED_PRICE: HashMap<Embed, Decimal> = {
        let mut map = HashMap::new();
        map.insert(Embed::TextEmbeddingAda002, dec!(0.0001));
        map
    };
}

// chat

impl Chat {
    pub fn get_input_price(&self, tokens: isize) -> Decimal {
        let count = Decimal::from(tokens) / Decimal::from(1000);
        let unit_price = CHAT_PRICE_INPUT.get(self).unwrap();
        count * unit_price
    }
    pub fn get_output_price(&self, tokens: isize) -> Decimal {
        let count = Decimal::from(tokens) / Decimal::from(1000);
        let unit_price = CHAT_PRICE_OUTPUT.get(self).unwrap();
        count * unit_price
    }
}

pub fn get_chat_price(model: Chat, input_tokens: isize, output_tokens: isize) -> Decimal {
    let input_price = model.get_input_price(input_tokens);
    let output_price = model.get_output_price(output_tokens);
    input_price + output_price
}

// embed

impl Embed {
    pub fn get_price(&self, tokens: isize) -> Decimal {
        let count = Decimal::from(tokens) / Decimal::from(1000);
        let unit_price = EMBED_PRICE.get(self).unwrap();
        count * unit_price
    }
}

pub fn get_embed_price(model: Embed, tokens: isize) -> Decimal {
    model.get_price(tokens)
}
