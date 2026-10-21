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

**`Codec` is order-preserving**, so the dictionary's order is the units' order:
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
| **match** | What a query covers: a unit, a prefix, or a range of units. |
| **clause** | Must, should and must-not matches and sub-clauses, or a phrase of units. |
| **relevance** | Weight (per unit) and score (per unit in a source). |

## Design

- **Dictionary:** a b-positive tree from each unit's encoding to its postings head and count.
  Its snapshots keep readers independent of the writer, with no draining or generation swapping.
- **Postings:** append-only records chained per unit.
  - Records over 128 entries bit-pack 128-entry blocks behind a fixed-width block index (last
    source, data offset, highest frequency, shortest source), which cursors binary-search to skip.
  - Each record carries Pareto (frequency, length) impacts that bound scores.
  - Positions live in their own file, decoded only for phrases.
- **Sources:** an ordinal per source with its key and length, in shared chunks; a snapshot copies
  nothing.
- **Commits are atomic:** data files are written first, then the dictionary commits with the
  lengths that are now valid. A crash anywhere leaves the previous commit, and anything past it is
  discarded on open.
- **Search:**
  - Boolean clauses: must, should, must-not, phrases, sub-clauses, evaluated smallest first with
    early exit on bitsets.
  - Top-k for plain disjunctions by MaxScore, with window skipping on block maxima and inline
    BM25. It is exactly equal to scoring every match (tested).
  - Prefix and range matches expand to the units they cover, up to a limit.

## Tests

`tests/index.rs`:
- **Postings:** checked against exactly what was added (decode, source sets, cursor `advance`
  to random targets, positions).
- **Pruning:** pruned equals exhaustive on 1,350 random queries (BM25, quantized BM25, TF-IDF;
  top 1, 10, 100).
- **Queries:** booleans, phrases, prefixes and ranges.
- **Typed indexes:** `(String, u16)` keys over a struct unit (signed ranges across zero), and
  `(String, String)` tuple units (prefixes within the first element).
- **Codecs:** order preserved and round trips for every provided codec.
- **Recovery:** snapshot isolation, and recovery from a crash mid-commit.

Repos are checked out side by side (`../b-positive`).
