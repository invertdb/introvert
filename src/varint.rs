//! Variable-length unsigned integers: 7 bits a byte, high bit set on all but the last.

#[inline]
pub fn write(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push(value as u8 | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

#[inline]
pub fn read(bytes: &[u8], at: &mut usize) -> u64 {
    let mut value = 0u64;
    let mut shift = 0;
    loop {
        let b = bytes[*at];
        *at += 1;
        value |= ((b & 0x7F) as u64) << shift;
        if b < 0x80 {
            return value;
        }
        shift += 7;
    }
}
