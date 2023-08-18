use anyhow::Error;
use pcre2::bytes::{Regex, RegexBuilder};
use std::collections::HashMap;

pub struct CoreBpe {
    encoder: HashMap<String, isize>,
    decoder: HashMap<isize, String>,
    special_tokens_encoder: HashMap<String, isize>,
    special_tokens_decoder: HashMap<isize, String>,
    tl_regex: Regex,
    tl_special_regex: Regex,
    sorted_token_bytes: Vec<Vec<u8>>,
}

impl CoreBpe {
    pub fn new(
        encoder: HashMap<String, isize>,
        special_tokens_encoder: HashMap<String, isize>,
        pattern: String,
    ) -> Result<Self, Error> {
        // build regex
        let tl_regex = RegexBuilder::new()
            .jit_if_available(true)
            .ucp(true)
            .utf(true)
            .build(pattern.as_str())?;
        let special_regex_strings: Vec<String> = special_tokens_encoder
            .iter()
            .map(|token| pcre2::escape(token.0).to_string())
            .collect();
        let special_regex_pattern = special_regex_strings.join("|");
        let tl_special_regex = Regex::new(&special_regex_pattern)?;

        // create decoder
        let mut decoder = HashMap::new();
        for (key, value) in encoder.clone().into_iter() {
            decoder.insert(value, key);
        }

        // create special tokens decoder
        let mut special_tokens_decoder = HashMap::new();
        for (key, value) in special_tokens_encoder.clone().into_iter() {
            special_tokens_decoder.insert(value, key);
        }

        // sort token bytes
        let mut sorted_token_bytes: Vec<Vec<u8>> =
            encoder.keys().map(|s| s.as_bytes().to_vec()).collect();
        sorted_token_bytes.sort_by(|a, b| a.cmp(b));

        Ok(Self {
            encoder,
            special_tokens_encoder,
            decoder,
            special_tokens_decoder,
            tl_regex,
            tl_special_regex,
            sorted_token_bytes,
        })
    }

    pub fn encode_native(&self, text: &str) -> (Vec<isize>, usize) {
        let mut result = vec![];
        let mut last_piece_token_len = 0;
        let mut start = 0;
        let text_chars: Vec<char> = text.chars().collect();

        // encoding
        loop {
            let mut next_special = None;
            let mut start_find = start;
            loop {
                // find next special token
                let temp = cut_chars(&text_chars, start_find, text_chars.len());
                next_special = find_regex_to_string_index(&temp, &self.tl_special_regex);
                if next_special.is_none() {
                    break;
                }
                start_find += next_special.unwrap().1;
            }

            // confirm text slice end
            let mut end = text_chars.len();
            if !next_special.is_none() {
                end = start + next_special.unwrap().0;
            }

            // handle text
            let temp = cut_chars(&text_chars, start, end);
            let store = find_regex_to_all_string_match_index(&temp, &self.tl_regex);
            for matched in store {
                let piece = cut_chars(&text_chars, start + matched.0, start + matched.1);
                if let Some(&token) = self.encoder.get(&piece) {
                    last_piece_token_len = 1;
                    result.push(token);
                    continue;
                }

                // encode byte pair
                let tokens = byte_pair_encode(piece.as_bytes(), &self.encoder);
                last_piece_token_len = tokens.len();
                result.extend(tokens);
            }

            // handle next special token
            if let Some(matched) = next_special {
                let temp = cut_chars(&text_chars, start + matched.0, start + matched.1);
                if let Some(&token) = self.special_tokens_encoder.get(temp.as_str()) {
                    result.push(token);
                    start += matched.1;
                    last_piece_token_len = 0;
                }
            } else {
                break;
            }
        }
        (result, last_piece_token_len)
    }
}

fn cut_chars(chars: &Vec<char>, start: usize, end: usize) -> String {
    let start = start.min(chars.len());
    let end = end.min(chars.len());
    chars[start..end].iter().collect()
}

fn find_regex_to_string_index(text: &str, regex: &Regex) -> Option<(usize, usize)> {
    if let Some(matched) = regex.find(text.as_bytes()).unwrap() {
        let matched_str = std::str::from_utf8(matched.as_bytes()).unwrap();
        let start = text[..matched.start()].chars().count();
        let end = start + matched_str.chars().count();
        Some((start, end))
    } else {
        None
    }
}

fn find_regex_to_all_string_match_index(text: &str, regex: &Regex) -> Vec<(usize, usize)> {
    regex
        .find_iter(text.as_bytes())
        .map(|matched| {
            let matched = matched.unwrap();
            let matched_str = std::str::from_utf8(matched.as_bytes()).unwrap();
            let start = text[..matched.start()].chars().count();
            let end = start + matched_str.chars().count();
            (start, end)
        })
        .collect()
}

fn byte_pair_encode(piece: &[u8], ranks: &HashMap<String, isize>) -> Vec<isize> {
    if piece.len() == 1 {
        unsafe {
            let key = String::from_utf8_unchecked(Vec::from(piece));
            let value = *ranks.get(&key).unwrap();
            return vec![value];
        }
    }
    byte_pair_merge(piece, ranks, |start, end| -> isize {
        let slice = piece[start..end].to_vec();
        unsafe {
            let key = String::from_utf8_unchecked(slice);
            ranks.get(&key).cloned().unwrap_or(0)
        }
    })
}

fn byte_pair_merge<T, F>(piece: &[u8], ranks: &HashMap<String, isize>, f: F) -> Vec<T>
where
    F: Fn(usize, usize) -> T,
{
    let mut parts: Vec<[usize; 2]> = (0..piece.len() + 1).map(|i| [i, usize::MAX]).collect();
    let get_rank = |start_idx: usize, skip: usize, parts: &Vec<[usize; 2]>| -> isize {
        if start_idx + skip + 2 < parts.len() {
            let b = &piece[parts[start_idx][0]..parts[start_idx + skip + 2][0]];
            unsafe {
                let key = String::from_utf8_unchecked(b.to_vec());
                if let Some(&rank) = ranks.get(&key) {
                    return rank;
                }
            }
        }
        -1
    };
    for i in 0..parts.len() - 2 {
        let rank = get_rank(i, 0, &parts);
        if rank >= 0 {
            parts[i][1] = rank as usize;
        }
    }
    while parts.len() > 1 {
        if let Some(min_idx) = (0..parts.len() - 1).min_by_key(|&i| parts[i][1]) {
            let rank = get_rank(min_idx, 1, &parts);
            if rank >= 0 {
                parts[min_idx][1] = rank as usize;
            }
            if min_idx > 0 {
                let rk = get_rank(min_idx - 1, 1, &parts);
                if rk >= 0 {
                    parts[min_idx - 1][1] = rk as usize;
                }
            }
            parts.remove(min_idx + 1);
        } else {
            break;
        }
    }
    (0..parts.len() - 1)
        .map(|i| f(parts[i][0], parts[i + 1][0]))
        .collect()
}
