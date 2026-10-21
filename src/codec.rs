//! How source keys and units are stored: an order-preserving encoding to bytes. If `a < b` then
//! `encode(a) < encode(b)` byte-wise, so the dictionary's order is the units' order and ranges of units are ranges of
//! bytes.
//!
//! A value encodes either last in a key (`encode`, which may run to the end of the bytes) or nested before other
//! values (`encode_nested`, which must be self-delimiting). Tuples encode every element but the last nested, so a
//! tuple's encoding begins with its leading elements' — which is what makes [`crate::Term::Prefix`] mean "these
//! leading elements, and a last element starting with this".
//!
//! Provided: byte strings and strings (nested: zero bytes escaped as 00 FF, terminated by 00 00), unsigned and signed
//! integers (big-endian, sign bit flipped), `bool`, `char`, `f64` (IEEE bits, negatives inverted; NaN sorts last),
//! `()`, and tuples of up to four. Implement `Codec` for a struct by delegating to a tuple of its fields.

pub trait Codec: Sized {
    /// Appends the encoding, as the last value in a key.
    fn encode(&self, out: &mut Vec<u8>);
    /// Decodes a value encoded last (all of `bytes`).
    fn decode(bytes: &[u8]) -> Self;
    /// Appends a self-delimiting encoding, ordered as `encode`, for a value followed by others.
    fn encode_nested(&self, out: &mut Vec<u8>) {
        self.encode(out)
    }
    /// Decodes a nested value starting at `at`, moving `at` past it.
    fn decode_nested(bytes: &[u8], at: &mut usize) -> Self;

    /// The value's encoding, as a new vector.
    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode(&mut out);
        out
    }
}

fn encode_escaped(bytes: &[u8], out: &mut Vec<u8>) {
    for &b in bytes {
        out.push(b);
        if b == 0 {
            out.push(0xFF);
        }
    }
    out.extend_from_slice(&[0, 0]);
}

fn decode_escaped(bytes: &[u8], at: &mut usize) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let b = bytes[*at];
        *at += 1;
        if b != 0 {
            out.push(b);
            continue;
        }
        let next = bytes[*at];
        *at += 1;
        if next == 0 {
            return out;
        }
        out.push(0);
    }
}

impl Codec for Vec<u8> {
    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(self);
    }
    fn decode(bytes: &[u8]) -> Self {
        bytes.to_vec()
    }
    fn encode_nested(&self, out: &mut Vec<u8>) {
        encode_escaped(self, out)
    }
    fn decode_nested(bytes: &[u8], at: &mut usize) -> Self {
        decode_escaped(bytes, at)
    }
}

impl Codec for String {
    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(self.as_bytes());
    }
    fn decode(bytes: &[u8]) -> Self {
        String::from_utf8_lossy(bytes).into_owned()
    }
    fn encode_nested(&self, out: &mut Vec<u8>) {
        encode_escaped(self.as_bytes(), out)
    }
    fn decode_nested(bytes: &[u8], at: &mut usize) -> Self {
        String::from_utf8(decode_escaped(bytes, at)).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
    }
}

macro_rules! unsigned {
    ($($t:ty),*) => {$(
        impl Codec for $t {
            fn encode(&self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_be_bytes());
            }
            fn decode(bytes: &[u8]) -> Self {
                <$t>::from_be_bytes(bytes[..size_of::<$t>()].try_into().unwrap())
            }
            fn decode_nested(bytes: &[u8], at: &mut usize) -> Self {
                let value = Self::decode(&bytes[*at..]);
                *at += size_of::<$t>();
                value
            }
        }
    )*};
}
unsigned!(u8, u16, u32, u64, u128);

macro_rules! signed {
    ($($t:ty => $u:ty),*) => {$(
        impl Codec for $t {
            fn encode(&self, out: &mut Vec<u8>) {
                ((*self as $u) ^ (1 << (<$u>::BITS - 1))).encode(out)
            }
            fn decode(bytes: &[u8]) -> Self {
                (<$u>::decode(bytes) ^ (1 << (<$u>::BITS - 1))) as $t
            }
            fn decode_nested(bytes: &[u8], at: &mut usize) -> Self {
                (<$u>::decode_nested(bytes, at) ^ (1 << (<$u>::BITS - 1))) as $t
            }
        }
    )*};
}
signed!(i8 => u8, i16 => u16, i32 => u32, i64 => u64, i128 => u128);

impl Codec for bool {
    fn encode(&self, out: &mut Vec<u8>) {
        out.push(*self as u8);
    }
    fn decode(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }
    fn decode_nested(bytes: &[u8], at: &mut usize) -> Self {
        *at += 1;
        bytes[*at - 1] != 0
    }
}

impl Codec for char {
    fn encode(&self, out: &mut Vec<u8>) {
        (*self as u32).encode(out)
    }
    fn decode(bytes: &[u8]) -> Self {
        char::from_u32(u32::decode(bytes)).unwrap_or(char::REPLACEMENT_CHARACTER)
    }
    fn decode_nested(bytes: &[u8], at: &mut usize) -> Self {
        char::from_u32(u32::decode_nested(bytes, at)).unwrap_or(char::REPLACEMENT_CHARACTER)
    }
}

/// Totally ordered: -inf < negatives < -0 < +0 < positives < +inf < NaN.
impl Codec for f64 {
    fn encode(&self, out: &mut Vec<u8>) {
        let bits = self.to_bits();
        let ordered = if bits >> 63 == 1 { !bits } else { bits | (1 << 63) };
        ordered.encode(out)
    }
    fn decode(bytes: &[u8]) -> Self {
        let ordered = u64::decode(bytes);
        f64::from_bits(if ordered >> 63 == 1 { ordered & !(1 << 63) } else { !ordered })
    }
    fn decode_nested(bytes: &[u8], at: &mut usize) -> Self {
        let value = Self::decode(&bytes[*at..]);
        *at += 8;
        value
    }
}

impl Codec for () {
    fn encode(&self, _out: &mut Vec<u8>) {}
    fn decode(_bytes: &[u8]) -> Self {}
    fn decode_nested(_bytes: &[u8], _at: &mut usize) -> Self {}
}

#[allow(non_snake_case, unused_mut)]
mod tuples {
    use super::Codec;

    macro_rules! tuple {
        ($($name:ident $index:tt),* ; $last_name:ident $last:tt) => {
            impl<$($name: Codec,)* $last_name: Codec> Codec for ($($name,)* $last_name,) {
                fn encode(&self, out: &mut Vec<u8>) {
                    $(self.$index.encode_nested(out);)*
                    self.$last.encode(out);
                }
                fn decode(bytes: &[u8]) -> Self {
                    let mut at = 0;
                    $(let $name = $name::decode_nested(bytes, &mut at);)*
                    ($($name,)* $last_name::decode(&bytes[at..]),)
                }
                fn encode_nested(&self, out: &mut Vec<u8>) {
                    $(self.$index.encode_nested(out);)*
                    self.$last.encode_nested(out);
                }
                fn decode_nested(bytes: &[u8], at: &mut usize) -> Self {
                    $(let $name = $name::decode_nested(bytes, at);)*
                    ($($name,)* $last_name::decode_nested(bytes, at),)
                }
            }
        };
    }
    tuple!(; A 0);
    tuple!(A 0; B 1);
    tuple!(A 0, B 1; C 2);
    tuple!(A 0, B 1, C 2; D 3);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ordered<T: Codec + PartialOrd + std::fmt::Debug + Clone>(mut values: Vec<T>) {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for pair in values.windows(2) {
            let (a, b) = (pair[0].to_bytes(), pair[1].to_bytes());
            assert!(a <= b, "{:?} < {:?} but {a:?} > {b:?}", pair[0], pair[1]);
        }
        for v in &values {
            let bytes = v.to_bytes();
            assert_eq!(&T::decode(&bytes), v);
            let mut nested = Vec::new();
            v.encode_nested(&mut nested);
            nested.push(0x42);
            let mut at = 0;
            assert_eq!(&T::decode_nested(&nested, &mut at), v);
            assert_eq!(at, nested.len() - 1);
        }
    }

    #[test]
    fn encodings_keep_order_and_round_trip() {
        ordered(vec![0u64, 1, 255, 256, u64::MAX]);
        ordered(vec![i64::MIN, -300, -1, 0, 1, 300, i64::MAX]);
        ordered(vec![i32::MIN, -1, 0, 7, i32::MAX]);
        ordered(vec![f64::NEG_INFINITY, -2.5, -0.0, 0.5, 3.0, f64::INFINITY]);
        ordered(vec![String::new(), "a".into(), "a\0".into(), "a\0b".into(), "ab".into(), "b".into(), "é".into()]);
        ordered(vec![
            (String::from("a"), 5u32),
            (String::from("a"), 70000),
            (String::from("a\0"), 0),
            (String::from("ab"), 0),
        ]);
        ordered(vec![(String::from("text"), String::from("")), (String::from("text"), String::from("whale")), (String::from("title"), String::from("a"))]);
        ordered(vec![(1u8, -4i16, String::from("x"), 2.5f64), (1, -4, String::from("x"), 3.0), (1, 2, String::new(), 0.0)]);
        ordered(vec!['a', 'z', '€']);
    }

    #[test]
    fn a_tuple_encoding_begins_with_its_leading_elements() {
        let whale = (String::from("text"), String::from("whale")).to_bytes();
        let prefix = (String::from("text"), String::from("wha")).to_bytes();
        assert!(whale.starts_with(&prefix));
        let other_field = (String::from("textual"), String::from("whale")).to_bytes();
        assert!(!other_field.starts_with(&prefix));
    }
}
