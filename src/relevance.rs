//! How well a source matches a unit, from how often the unit occurs in it and how many sources contain the unit.
//!
//! Top-k pruning bounds scores from postings maxima, so [`Relevance::score`] must not decrease as the unit frequency
//! grows, nor increase as the source length grows.

pub trait Relevance: Send + Sync {
    /// What matching the unit is worth, given that `source_frequency` of `source_count` sources contain it.
    fn weight(&self, source_frequency: u64, source_count: u64) -> f64;

    /// A source's score for a unit of the given weight occurring `unit_frequency` times in it (lengths in units).
    fn score(&self, weight: f64, unit_frequency: u32, source_length: u32, average_length: f64) -> f64;

    /// As [`Bm25`], if this is BM25 with exact lengths: lets the searcher score it inline.
    fn as_bm25(&self) -> Option<&Bm25> {
        None
    }
}

/// Okapi BM25 (as Lucene's BM25Similarity): frequency saturates by `k1`, normalized by length by `b`.
#[derive(Clone, Copy, Debug)]
pub struct Bm25 {
    pub k1: f64,
    pub b: f64,
    /// Lengths rounded as Lucene stores them (a one-byte norm), as Elasticsearch scores.
    pub quantize_lengths: bool,
}

impl Default for Bm25 {
    fn default() -> Self {
        Bm25 { k1: 1.2, b: 0.75, quantize_lengths: false }
    }
}

impl Bm25 {
    /// A length as Lucene's SmallFloat.intToByte4 then byte4ToInt keep it: exact to 24, then 4 significant bits.
    pub fn quantize(length: u32) -> u32 {
        const FREE: u32 = 24;
        if length < FREE {
            return length;
        }
        let i = (length - FREE) as u64;
        let bits = 64 - i.leading_zeros();
        if bits < 4 {
            return FREE + i as u32;
        }
        let shift = bits - 4;
        FREE + ((((i >> shift) & 0x07) | 0x08) << shift) as u32
    }

    #[inline]
    pub fn norm(&self, length: u32, average_length: f64) -> f64 {
        self.k1 * (1.0 - self.b + self.b * length as f64 / average_length)
    }
}

impl Relevance for Bm25 {
    fn weight(&self, source_frequency: u64, source_count: u64) -> f64 {
        (1.0 + (source_count as f64 - source_frequency as f64 + 0.5) / (source_frequency as f64 + 0.5)).ln()
    }

    #[inline]
    fn score(&self, weight: f64, unit_frequency: u32, source_length: u32, average_length: f64) -> f64 {
        let length = if self.quantize_lengths { Self::quantize(source_length) } else { source_length };
        let f = unit_frequency as f64;
        weight * f / (f + self.norm(length, average_length))
    }

    fn as_bm25(&self) -> Option<&Bm25> {
        if self.quantize_lengths { None } else { Some(self) }
    }
}

/// Unit frequency × inverse source frequency: weight ln(1 + N / n), score weight × (1 + ln f). Ignores length.
#[derive(Clone, Copy, Debug, Default)]
pub struct TfIdf;

impl Relevance for TfIdf {
    fn weight(&self, source_frequency: u64, source_count: u64) -> f64 {
        (1.0 + source_count as f64 / source_frequency as f64).ln()
    }

    fn score(&self, weight: f64, unit_frequency: u32, _source_length: u32, _average_length: f64) -> f64 {
        weight * (1.0 + (unit_frequency as f64).ln())
    }
}
