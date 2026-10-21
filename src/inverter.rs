//! Inverting a source into units: the one thing that differs between a full-text index, a label index, a geo index
//! and an audio fingerprint index.

use unicode_general_category::{get_general_category, GeneralCategory as G};

/// Turns a source of type `S` into units, each at a position (positions order units for phrases; an inverter
/// without order may give every unit position 0).
pub trait Inverter<S: ?Sized> {
    type Unit;

    fn invert(&self, source: &S, unit: impl FnMut(Self::Unit, u32));

    /// The units, in order.
    fn units(&self, source: &S) -> Vec<(Self::Unit, u32)> {
        let mut out = Vec::new();
        self.invert(source, |u, p| out.push((u, p)));
        out
    }
}

/// Text into words: runs of Unicode letters (L*) and decimal digits (Nd), lower-cased a character at a time, at most
/// `max_bytes` bytes (longer words keep their first bytes). No stemming, no stop words.
#[derive(Clone, Copy, Debug)]
pub struct Words {
    pub max_bytes: usize,
}

impl Default for Words {
    fn default() -> Self {
        Words { max_bytes: 32 }
    }
}

#[inline]
fn is_letter_or_digit(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphanumeric();
    }
    matches!(get_general_category(c), G::UppercaseLetter | G::LowercaseLetter | G::TitlecaseLetter | G::ModifierLetter | G::OtherLetter | G::DecimalNumber)
}

/// One character lower-cased, never expanding (as .NET's ToLowerInvariant).
#[inline]
fn lower(c: char) -> char {
    if c.is_ascii() {
        return c.to_ascii_lowercase();
    }
    let mut lower = c.to_lowercase();
    match (lower.next(), lower.next()) {
        (Some(l), None) => l,
        _ => c,
    }
}

impl Words {
    /// Calls `word(text, position)` for each word, without allocating per word.
    pub fn each(&self, text: &str, mut word: impl FnMut(&str, u32)) {
        let mut current = String::with_capacity(self.max_bytes);
        let mut position = 0u32;
        let mut in_word = false;
        for c in text.chars() {
            if is_letter_or_digit(c) {
                in_word = true;
                let lower = lower(c);
                if current.len() + lower.len_utf8() <= self.max_bytes {
                    current.push(lower);
                }
            } else if in_word {
                word(&current, position);
                position += 1;
                current.clear();
                in_word = false;
            }
        }
        if in_word {
            word(&current, position);
        }
    }
}

impl Inverter<str> for Words {
    type Unit = String;

    fn invert(&self, source: &str, mut unit: impl FnMut(String, u32)) {
        self.each(source, |w, p| unit(w.to_string(), p));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words() {
        let words = |t: &str| Words::default().units(t).into_iter().map(|u| u.0).collect::<Vec<_>>();
        assert_eq!(words("The White-Whale's 2nd ÉTÉ!"), vec!["the", "white", "whale", "s", "2nd", "été"]);
        assert_eq!(words(&"x".repeat(40))[0].len(), 32);
        // other numbers, letter numbers and marks split words, as .NET's IsLetterOrDigit
        assert_eq!(words("x² café\u{301} Ⅻ"), vec!["x", "café"]);
    }
}
