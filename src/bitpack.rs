//! Fixed-width bit packing of blocks of values (frame of reference): every value of a block in the bits the
//! largest needs, little-endian, so a block decodes in one tight loop. Readers may read up to 8 bytes past a block:
//! records end with [`PADDING`] bytes.

pub const PADDING: usize = 8;

#[inline]
pub fn width(max: u64) -> u32 {
    64 - max.leading_zeros()
}

#[inline]
pub fn bytes(count: usize, width: u32) -> usize {
    (count * width as usize).div_ceil(8)
}

pub fn pack(values: &[u64], width: u32, out: &mut Vec<u8>) {
    let start = out.len();
    out.resize(start + bytes(values.len(), width), 0);
    if width == 0 {
        return;
    }
    let mut bit = 0usize;
    for &value in values {
        let mut written = 0u32;
        while written < width {
            let index = start + (bit + written as usize) / 8;
            let shift = ((bit + written as usize) % 8) as u32;
            let take = (8 - shift).min(width - written);
            out[index] |= (((value >> written) & ((1u64 << take) - 1)) << shift) as u8;
            written += take;
        }
        bit += width as usize;
    }
}

/// Unpacks `out.len()` values of `width` bits (at most 57) from `src` (which must have 8 readable bytes past the data).
#[inline]
pub fn unpack_u32(src: &[u8], width: u32, out: &mut [u32]) {
    if width == 0 {
        out.fill(0);
        return;
    }
    let mask = (1u64 << width) - 1;
    let mut bit = 0usize;
    for v in out.iter_mut() {
        let word = u64::from_le_bytes(src[bit / 8..bit / 8 + 8].try_into().unwrap());
        *v = ((word >> (bit % 8)) & mask) as u32;
        bit += width as usize;
    }
}

#[inline]
pub fn unpack_u64(src: &[u8], width: u32, out: &mut [u64]) {
    if width == 0 {
        out.fill(0);
        return;
    }
    let mask = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
    let mut bit = 0usize;
    for v in out.iter_mut() {
        let word = u64::from_le_bytes(src[bit / 8..bit / 8 + 8].try_into().unwrap());
        *v = (word >> (bit % 8)) & mask;
        bit += width as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        for w in [0u32, 1, 3, 7, 13, 31, 32, 40, 57] {
            let max = if w == 0 { 0 } else { (1u64 << w) - 1 };
            let values: Vec<u64> = (0..128u64).map(|i| (i * 2654435761) & max).collect();
            assert_eq!(width(values.iter().copied().max().unwrap()) <= w, true);
            let mut packed = Vec::new();
            pack(&values, w, &mut packed);
            packed.extend_from_slice(&[0; PADDING]);
            let mut out = vec![0u64; 128];
            unpack_u64(&packed, w, &mut out);
            assert_eq!(out, values, "width {w}");
        }
    }
}
