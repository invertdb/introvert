//! introvert: a generic inverted index. Any source type inverted into any unit type: the words of a text, the
//! labels of a time series, the cells of a place, the landmarks of a song.
//!
//! An [`Index<K, U>`] holds sources, each under a key `K`, over units `U`, both stored by an order-preserving
//! [`Codec`], so the dictionary's order is the units' order: a [`Match`] covers a unit, the units beginning with a
//! prefix (e.g. a tuple's leading elements and the start of its last), or a range of units (e.g. times, cells,
//! numbers). An [`Inverter<S>`] turns a source of type `S` into units; [`Words`] is the text one. A
//! [`Searcher<K, U, R>`] ranks by a [`Relevance`] `R`, compiled in (BM25, TF-IDF, or your own).
//!
//! The dictionary is a [b-positive](https://github.com/invertdb/b-positive) tree from unit to its postings, whose
//! snapshots make readers independent of the writer. Postings are bit-packed 128-entry blocks behind a block index
//! for skipping, with impacts that bound scores; searches rank the top k with MaxScore over them.
//!
//! Theory: inverted files as surveyed by Zobel and Moffat (2006); relevance by BM25 (Robertson et al., 1994;
//! Robertson and Zaragoza, 2009) or TF-IDF (Spärck Jones, 1972; Salton and Buckley, 1988); postings compressed by
//! frame of reference (Goldstein, Ramakrishnan and Shaft, 1998); top-k by MaxScore (Turtle and Flood, 1995) with
//! pre-computed impacts (Anh and Moffat, 2006) and block maxima (Ding and Suel, 2011); units ordered by normalized
//! keys (Graefe, 2011). Full references are in the README.

pub mod bitpack;
pub mod codec;
pub mod index;
pub mod inverter;
pub mod postings;
pub mod relevance;
pub mod search;
pub mod sources;
pub mod sourceset;
pub mod varint;

pub use codec::Codec;
pub use index::{Index, Options, Snapshot, UnitPostings};
pub use inverter::{Inverter, Words};
pub use relevance::{Bm25, Relevance, TfIdf};
pub use search::{Clause, Match, SearchResult, SearchResults, Searcher};
pub use sourceset::SourceSet;
