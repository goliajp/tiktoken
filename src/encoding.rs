use anyhow::{anyhow, Error};
use base64::Engine;
use lazy_static::lazy_static;
use maplit::hashmap;
use std::collections::HashMap;

// enums

#[derive(Eq, PartialEq, Hash)]
pub enum SpecialToken {
    EndOfText,
    FimPrefix,
    FimMiddle,
    FimSuffix,
    EndOfPrompt,
}

impl SpecialToken {
    pub fn to_string(&self) -> String {
        match self {
            Self::EndOfText => String::from("<|endoftext|>"),
            Self::FimPrefix => String::from("<|fim_prefix|>"),
            Self::FimMiddle => String::from("<|fim_middle|>"),
            Self::FimSuffix => String::from("<|fim_suffix|>"),
            Self::EndOfPrompt => String::from("<|endofprompt|>"),
        }
    }
}

pub enum Dict {
    Cl100kBase,
    P50kBase,
    P50kEdit,
    R50kBase,
}

impl Dict {
    pub fn to_string(&self) -> String {
        match self {
            Self::Cl100kBase => String::from("cl100k_base"),
            Self::P50kBase => String::from("p50k_base"),
            Self::P50kEdit => String::from("p50k_edit"),
            Self::R50kBase => String::from("r50k_base"),
        }
    }

    pub fn get_file(&self) -> &[u8] {
        match self {
            Self::Cl100kBase => include_bytes!("encodings/cl100k_base.tiktoken"),
            Self::P50kBase => include_bytes!("encodings/p50k_base.tiktoken"),
            Self::P50kEdit => include_bytes!("encodings/p50k_base.tiktoken"), // same to p50k_base
            Self::R50kBase => include_bytes!("encodings/r50k_base.tiktoken"),
        }
    }

    pub fn get_regex_pattern(&self) -> String {
        match self {
            Self::Cl100kBase => String::from(
                r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            ),
            _ => String::from(
                r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            ),
        }
    }
}

#[derive(Eq, PartialEq, Hash)]
pub enum ChatModelPrefix {
    Gpt3dot5,
    Gpt4,
}

impl ChatModelPrefix {
    pub fn to_string(&self) -> String {
        match self {
            Self::Gpt4 => String::from("gpt-4"),
            Self::Gpt3dot5 => String::from("gpt-3.5"),
        }
    }

    pub fn get_by_name(chat_model_name: &str) -> Result<Self, Error> {
        match chat_model_name.to_lowercase() {
            _ if chat_model_name.starts_with(Self::Gpt3dot5.to_string().as_str()) => {
                Ok(Self::Gpt3dot5)
            }
            _ if chat_model_name.starts_with(Self::Gpt4.to_string().as_str()) => Ok(Self::Gpt4),
            _ => Err(anyhow!("no prefix for model {}", chat_model_name)),
        }
    }
}

// mapping

lazy_static! {
    pub static ref CHAT_MODEL_PREFIX_TO_DICT: HashMap<ChatModelPrefix, Dict> = {
        let mut map = HashMap::new();
        map.insert(ChatModelPrefix::Gpt3dot5, Dict::Cl100kBase);
        map.insert(ChatModelPrefix::Gpt4, Dict::Cl100kBase);
        map
    };
}

// encoding

pub struct Encoding {
    pub dict: Dict,
    pub merging_ranks: HashMap<String, isize>,
    pub special_tokens: HashMap<String, isize>,
    pub explicit_vocab_size: isize,
}

impl Encoding {
    pub fn get_by_dict(dict: &Dict) -> Result<Self, Error> {
        match dict {
            Dict::Cl100kBase => cl100k_base(),
            Dict::P50kBase => p50k_base(),
            Dict::P50kEdit => p50k_edit(),
            Dict::R50kBase => r50k_base(),
        }
    }

    pub fn get_by_chat_model(chat_model_name: &str) -> Result<Self, Error> {
        let model_prefix = ChatModelPrefix::get_by_name(chat_model_name)?;
        match CHAT_MODEL_PREFIX_TO_DICT.get(&model_prefix) {
            Some(dict) => Encoding::get_by_dict(dict),
            None => Err(anyhow!(
                "no encoding for model prefix {}",
                model_prefix.to_string()
            )),
        }
    }
}

fn cl100k_base() -> Result<Encoding, Error> {
    let dict_data = Dict::Cl100kBase.get_file();
    let merging_ranks = parse_dict_data(dict_data)?;
    let special_tokens = hashmap! {
        SpecialToken::EndOfText.to_string() => 100257,
        SpecialToken::FimPrefix.to_string() => 100258,
        SpecialToken::FimMiddle.to_string() => 100259,
        SpecialToken::FimSuffix.to_string() => 100260,
        SpecialToken::EndOfPrompt.to_string() => 100276,
    };

    Ok(Encoding {
        dict: Dict::Cl100kBase,
        merging_ranks,
        special_tokens,
        explicit_vocab_size: 0,
    })
}

fn p50k_base() -> Result<Encoding, Error> {
    let dict_data = Dict::P50kBase.get_file();
    let merging_ranks = parse_dict_data(dict_data)?;
    let special_tokens = hashmap! {
        SpecialToken::EndOfText.to_string() => 50256,
    };

    Ok(Encoding {
        dict: Dict::P50kBase,
        merging_ranks,
        special_tokens,
        explicit_vocab_size: 50281,
    })
}

fn p50k_edit() -> Result<Encoding, Error> {
    let dict_data = Dict::P50kEdit.get_file();
    let merging_ranks = parse_dict_data(dict_data)?;
    let special_tokens = hashmap! {
        SpecialToken::EndOfText.to_string() => 50256,
        SpecialToken::FimPrefix.to_string() => 50281,
        SpecialToken::FimMiddle.to_string() => 50282,
        SpecialToken::FimSuffix.to_string() => 50283,
    };

    Ok(Encoding {
        dict: Dict::P50kEdit,
        merging_ranks,
        special_tokens,
        explicit_vocab_size: 0,
    })
}

fn r50k_base() -> Result<Encoding, Error> {
    let dict_data = Dict::R50kBase.get_file();
    let merging_ranks = parse_dict_data(dict_data)?;
    let special_tokens = hashmap! {
        SpecialToken::EndOfText.to_string() => 50256,
    };

    Ok(Encoding {
        dict: Dict::R50kBase,
        merging_ranks,
        special_tokens,
        explicit_vocab_size: 50257,
    })
}

fn parse_dict_data(contents: &[u8]) -> Result<HashMap<String, isize>, Error> {
    let mut bpe_ranks = HashMap::new();
    let engine = base64::engine::general_purpose::STANDARD;
    unsafe {
        let content_str = String::from_utf8_unchecked(Vec::from(contents));
        for line in content_str.lines() {
            if line.trim() == "" {
                continue;
            }
            let parts: Vec<&str> = line.split(" ").collect();
            if parts.len() != 2 {
                return Err(anyhow!("unexpected line format: {}", line));
            }
            let token_bytes = engine.decode(parts[0])?;

            let token = String::from_utf8_unchecked(token_bytes);
            let rank: isize = parts[1].parse()?;
            bpe_ranks.insert(token, rank);
        }
    }
    Ok(bpe_ranks)
}
