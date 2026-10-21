//! How well a source matches a unit, from how often the unit occurs in it and how many sources contain the unit.
//!
//! A searcher is generic over its relevance, so scoring compiles inline. A source's length enters through
//! [`Relevance::Norm`], computed once per source and shared by every unit scored against it.
//!
//! Top-k pruning bounds scores from postings maxima, so [`Relevance::score`] must not decrease as the unit frequency
//! grows, nor increase as the source length grows.

pub trait Relevance: Send + Sync {
    /// What a source's length contributes to its scores.
    type Norm: Copy;

    /// What matching the unit is worth, given that `source_frequency` of `source_count` sources contain it.
    fn weight(&self, source_frequency: u64, source_count: u64) -> f64;

    /// A source's norm, from its length in units and the average length.
    fn norm(&self, source_length: u32, average_length: f64) -> Self::Norm;

    /// A source's score for a unit of the given weight occurring `unit_frequency` times in it.
    fn score(&self, weight: f64, unit_frequency: u32, norm: Self::Norm) -> f64;
}

/// Okapi BM25 (Robertson et al., 1994, "Okapi at TREC-3"; Robertson and Zaragoza, 2009, "The Probabilistic
/// Relevance Framework: BM25 and Beyond"): a unit's frequency saturates by `k1`, and a source's length normalizes by
/// `b` (0: not at all; 1: fully, against the average length). The weight is the non-negative inverse source
/// frequency ln(1 + (N − n + 0.5) / (n + 0.5)).
#[derive(Clone, Copy, Debug)]
pub struct Bm25 {
    pub k1: f64,
    pub b: f64,
}

impl Default for Bm25 {
    fn default() -> Self {
        Bm25 { k1: 1.2, b: 0.75 }
    }
}

impl Relevance for Bm25 {
    type Norm = f64;

    fn weight(&self, source_frequency: u64, source_count: u64) -> f64 {
        (1.0 + (source_count as f64 - source_frequency as f64 + 0.5) / (source_frequency as f64 + 0.5)).ln()
    }

    #[inline]
    fn norm(&self, source_length: u32, average_length: f64) -> f64 {
        self.k1 * (1.0 - self.b + self.b * source_length as f64 / average_length)
    }

    #[inline]
    fn score(&self, weight: f64, unit_frequency: u32, norm: f64) -> f64 {
        let f = unit_frequency as f64;
        weight * f / (f + norm)
    }
}

/// Unit frequency × inverse source frequency (Spärck Jones, 1972; Salton and Buckley, 1988): weight ln(1 + N / n),
/// score weight × (1 + ln f). Ignores length.
#[derive(Clone, Copy, Debug, Default)]
pub struct TfIdf;

impl Relevance for TfIdf {
    type Norm = ();

    fn weight(&self, source_frequency: u64, source_count: u64) -> f64 {
        (1.0 + source_count as f64 / source_frequency as f64).ln()
    }

    #[inline]
    fn norm(&self, _source_length: u32, _average_length: f64) {}

    #[inline]
    fn score(&self, weight: f64, unit_frequency: u32, _norm: ()) -> f64 {
        weight * (1.0 + (unit_frequency as f64).ln())
    }
}
