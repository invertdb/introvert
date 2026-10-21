//! introvert: a generic inverted index. Any source can be inverted into any units: words of a document, labels of
//! a time series, cells of a place, landmarks of a song. Sources and units are byte strings here (units in byte
//! order, so a prefix is a range); typed layers encode their own.
//!
//! The dictionary is a [b-positive](https://github.com/invertdb/b-positive) tree from unit to its postings, whose
//! snapshots make readers independent of the writer. Postings are bit-packed 128-entry blocks behind a block index
//! for skipping, with impacts that bound scores; [`search::Searcher`] ranks the top k with MaxScore over them.

pub mod bitpack;
pub mod index;
pub mod postings;
pub mod relevance;
pub mod search;
pub mod sources;
pub mod sourceset;
pub mod varint;

pub use index::{Index, Options, Snapshot, UnitPostings};
pub use relevance::{Bm25, Relevance, TfIdf};
pub use search::{Clause, Hit, Results, Searcher};
pub use sourceset::SourceSet;
