# introvert

A generic inverted index: any source type inverted into any unit type, ranked with BM25 or TF-IDF. Examples: the
words of a text, the labels of a time series, the cells of a place, the landmarks of a song. The engine of
invertdb, on [b-positive](https://github.com/invertdb/b-positive).

```rust
use introvert::{Bm25, Clause, Index, Match, Searcher, Words};

// sources keyed by any K, inverted into any units U (each with an order-preserving Codec)
let index: Index<u32, String> = Index::open("idx", Default::default())?;
index.invert(&1, "the white whale", &Words::default());   // an Inverter<S> turns a source S into units
index.commit()?;

let snapshot = index.snapshot();                           // any thread; unaffected by later commits
let bm25 = Bm25::default();
let searcher = Searcher::new(&snapshot, &bm25);            // generic over the relevance: scoring compiles inline
let top = searcher.search(&Clause::should(["whale".to_string()]), 10).top;    // SearchResult<u32>: top[0].key == 1
let wh = searcher.search(&Clause::should([Match::Prefix("wh".to_string())]), 10);
```

## Generics

**Everything is typed:**
- `Index<K, U>`, `Snapshot<K, U>`, `Clause<U>`, `Match<U>`, `SearchResults<K>`, and
  `Searcher<K, U, R>` with `R: Relevance`.
- An `Inverter<S>` turns a source of any type `S` into units.
- A source's key is any `K: Codec` and a unit any `U: Codec`. Bytes are only the storage format.

**`Codec` is order-preserving** (normalized keys; Graefe, 2011), so the dictionary's order is the
units' order:
- A `Match::Range` of typed units is a range scan. Examples: temperatures across zero, times,
  geo cells.
- A `Match::Prefix` of a tuple (or a struct encoded as one) stays within its leading elements,
  e.g. `("title", "whal")`.
- Provided codecs: strings, byte vectors, all integer widths (signed sorted correctly), `f64`
  (totally ordered), `bool`, `char`, `()`, and tuples of up to four. A struct implements `Codec`
  by delegating to its members.

**`Relevance` is a type parameter**, not a trait object. A source's length enters once per
candidate through `Relevance::Norm`, so BM25's per-candidate norm is shared across units by
monomorphization, with no special case.

## Vocabulary

| | |
|---|---|
| **source** | What is inverted (a text, a series, a place, a song). It has a key `K`, an ordinal (its position in the index) and a length in units. |
| **unit** | What a source is inverted into, at positions. |
| **inverter** | Turns a source into units. |
| **postings** | A unit's sources, with frequencies and positions. |
| **match** | What a query covers: a unit, a prefix, a range, or any of a set of units (or of their postings, already looked up). |
| **clause** | Must, should and must-not matches and sub-clauses, or a phrase of units. |
| **relevance** | Weight (per unit) and score (per unit in a source). |

## Design

Each technique is cited where it is used and listed under [References](#references).

- **Dictionary:** a b-positive tree from each unit's encoding to its postings head and count.
  Its snapshots keep readers independent of the writer, with no draining or generation swapping.
- **Postings:** append-only records chained per unit.
  - Records over 128 entries bit-pack 128-entry blocks, frame-of-reference coded (Goldstein et al.,
    1998), behind a fixed-width block index (last source, data offset, highest frequency, shortest
    source), which cursors binary-search to skip.
  - Each record carries Pareto (frequency, length) impacts that bound scores (after Anh and Moffat,
    2006).
  - Positions live in their own file, decoded only for phrases.
- **Sources:** an ordinal per source with its key and length, in shared chunks; a snapshot copies
  nothing.
- **Commits are atomic:** data files are written first, then the dictionary commits with the
  lengths that are now valid. A crash anywhere leaves the previous commit, and anything past it is
  discarded on open.
- **Search:**
  - Boolean clauses: must, should, must-not, phrases, sub-clauses, evaluated smallest first with
    early exit on bitsets. A unit much larger than what's left is intersected, or excluded, by
    advancing its cursor to each remaining source, so its block index skips what isn't needed.
  - Top-k for plain disjunctions by MaxScore (Turtle and Flood, 1995), with window skipping on
    block maxima (Ding and Suel, 2011). It is exactly equal to scoring every match (tested).
  - Relevance: BM25 (Robertson et al., 1994; Robertson and Zaragoza, 2009) and TF-IDF (Spärck Jones,
    1972; Salton and Buckley, 1988).
  - Prefix and range matches expand to the units they cover, up to a limit.
  - Alignment: the sources in which the most of a query's units occur at one offset from their
    query positions, and that offset (the voting scheme of Wang, 2003). It's a phrase match that
    tolerates missing units.

## Tests

`tests/index.rs`:
- **Postings:** checked against exactly what was added (decode, source sets, cursor `advance`
  to random targets, positions).
- **Pruning:** pruned equals exhaustive on 1,350 random queries (BM25 at two tunings, TF-IDF;
  top 1, 10, 100).
- **Queries:** booleans, phrases, prefixes and ranges.
- **Typed indexes:** `(String, u16)` keys over a struct unit (signed ranges across zero), and
  `(String, String)` tuple units (prefixes within the first element).
- **Codecs:** order preserved and round trips for every provided codec.
- **Recovery:** snapshot isolation, and recovery from a crash mid-commit.

## References

- Anh, V. N., and Moffat, A. (2006). Pruned query evaluation using pre-computed impacts. *SIGIR 2006*.
- Ding, S., and Suel, T. (2011). Faster top-k document retrieval using block-max indexes. *SIGIR 2011*.
- Goldstein, J., Ramakrishnan, R., and Shaft, U. (1998). Compressing relations and indexes. *ICDE 1998*.
- Graefe, G. (2011). Modern B-tree techniques. *Foundations and Trends in Databases* 3(4).
- Robertson, S. E., Walker, S., Jones, S., Hancock-Beaulieu, M., and Gatford, M. (1994). Okapi at TREC-3.
  *TREC-3*.
- Robertson, S., and Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond.
  *Foundations and Trends in Information Retrieval* 3(4).
- Salton, G., and Buckley, C. (1988). Term-weighting approaches in automatic text retrieval.
  *Information Processing & Management* 24(5).
- Spärck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval.
  *Journal of Documentation* 28(1).
- Wang, A. (2003). An industrial-strength audio search algorithm. *ISMIR 2003*.
- Turtle, H., and Flood, J. (1995). Query evaluation: strategies and optimizations.
  *Information Processing & Management* 31(6).
- Zobel, J., and Moffat, A. (2006). Inverted files for text search engines. *ACM Computing Surveys* 38(2).

Repos are checked out side by side (`../b-positive`).
