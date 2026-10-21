# introvert

A generic inverted index: any source inverted into any units (the words of a document, the labels of a time
series, the cells of a place, the landmarks of a song), ranked with BM25 or TF-IDF. The engine of invertdb,
reimplemented from bsharptree's `simpleindex`, on [b-positive](https://github.com/invertdb/b-positive).

```rust
use introvert::{Bm25, Clause, Index, Searcher, Term, Words};

// any source type S, any unit type U (each with an order-preserving Codec)
let index: Index<u32, String> = Index::open("idx", Default::default())?;
index.invert(&1, "the white whale", &Words::default());   // an Inverter<D> turns a D into units
index.commit()?;

let snapshot = index.snapshot();                           // any thread; unaffected by later commits
let bm25 = Bm25::default();
let searcher = Searcher::new(&snapshot, &bm25);            // generic over the relevance: scoring compiles inline
let hits = searcher.search(&Clause::should(["whale".to_string()]), 10).top;    // Hit<u32>: hits[0].source == 1
let wh = searcher.search(&Clause::should([Term::Prefix("wh".to_string())]), 10);
```

## Generics

**Everything is typed:**
- `Index<S, U>`, `Snapshot<S, U>`, `Clause<U>`, `Term<U>`, `Hit<S>`, and `Searcher<S, U, R>` with
  `R: Relevance`.
- An `Inverter<D>` turns a document of any type into units.
- A source key is any `S: Codec` and a unit any `U: Codec`. Bytes are only the storage format.

**`Codec` is order-preserving**, so the dictionary's order is the units' order:
- A `Term::Range` of typed units is a range scan. Examples: temperatures across zero, times,
  geo cells.
- A `Term::Prefix` of a tuple (or a struct encoded as one) stays within its leading elements, e.g.
  `(field, "whal")`.
- Provided codecs: strings, byte vectors, all integer widths (signed sorted correctly), `f64`
  (totally ordered), `bool`, `char`, `()`, and tuples of up to four. A struct implements `Codec`
  by delegating to its fields.

**`Relevance` is a type parameter**, not a trait object. A source's length enters once per
candidate through `Relevance::Norm`, so BM25's per-candidate norm is shared across units by
monomorphization, with no special case.

## Design

- **Dictionary:** a b-positive tree from each unit's encoding to its postings head and count. Its snapshots keep readers independent of the writer, with no draining
  or generation swapping.
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
  - Prefix and range terms expand to the units they cover, up to a limit.

## Tests

`tests/index.rs`:
- **Postings:** checked against exactly what was added (decode, source sets, cursor `advance`
  to random targets, positions).
- **Pruning:** pruned equals exhaustive on 1,350 random queries (BM25, quantized BM25, TF-IDF;
  top 1, 10, 100).
- **Queries:** booleans, phrases, prefixes and ranges.
- **Typed indexes:** `(String, u16)` keys over a struct unit (signed ranges across zero), and
  `(field, term)` tuple units (prefixes within a field).
- **Codecs:** order preserved and round trips for every provided codec.
- **Recovery:** snapshot isolation, and recovery from a crash mid-commit.

Repos are checked out side by side (`../b-positive`).
