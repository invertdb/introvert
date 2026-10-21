# introvert

A generic inverted index: any source inverted into any units (the words of a document, the labels of a time
series, the cells of a place, the landmarks of a song), ranked with BM25 or TF-IDF. The engine of invertdb,
reimplemented from bsharptree's `simpleindex`, on [b-positive](https://github.com/invertdb/b-positive).

```rust
let index = introvert::Index::open("idx", Default::default())?;
index.add(b"doc1", [(&b"white"[..], 0), (&b"whale"[..], 1)]);
index.commit()?;
let snapshot = index.snapshot();                         // any thread; unaffected by later commits
let bm25 = introvert::Bm25::default();
let results = introvert::Searcher::new(&snapshot, &bm25).search(&introvert::Clause::should([b"whale".to_vec()]), 10);
```

## Design

- **Dictionary:** a b-positive tree from unit (bytes, in byte order: a prefix is a range) to its
  postings head and count. Its snapshots keep readers independent of the writer, with no draining
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
  - Unit expansion is pluggable (e.g. `prefix*`).

## Tests

`tests/index.rs`:
- **Postings:** checked against exactly what was added (decode, source sets, cursor `advance`
  to random targets, positions).
- **Pruning:** pruned equals exhaustive on 1,350 random queries (BM25, quantized BM25, TF-IDF;
  top 1, 10, 100).
- **Queries:** booleans and phrases.
- **Recovery:** snapshot isolation, and recovery from a crash mid-commit.

Repos are checked out side by side (`../b-positive`).
