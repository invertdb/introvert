use std::collections::BTreeMap;

use introvert::relevance::Relevance;
use introvert::{Bm25, Clause, Index, Options, Searcher, SourceSet, TfIdf};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

struct Built {
    _dir: tempfile::TempDir,
    index: Index,
    /// unit -> (source, frequency), in source order
    expected: BTreeMap<Vec<u8>, Vec<(u32, u32)>>,
    vocabulary: Vec<Vec<u8>>,
}

/// 20,000 sources of 5..80 units from a skewed vocabulary, committed every few thousand (so chains of records).
fn build(seed: u64) -> Built {
    let dir = tempfile::tempdir().unwrap();
    let index = Index::open(dir.path(), Options::default()).unwrap();
    let mut rng = StdRng::seed_from_u64(seed);
    let vocabulary: Vec<Vec<u8>> = (0..3000).map(|i| format!("u{i}").into_bytes()).collect();
    let mut expected: BTreeMap<Vec<u8>, Vec<(u32, u32)>> = BTreeMap::new();
    for s in 0..20_000u32 {
        let length = rng.random_range(5..80);
        let units: Vec<&[u8]> = (0..length).map(|_| vocabulary[(3000.0 * rng.random::<f64>().powi(3)) as usize].as_slice()).collect();
        let ordinal = index.add(format!("doc{s}").as_bytes(), units.iter().enumerate().map(|(i, u)| (*u, i as u32)));
        assert_eq!(ordinal, s);
        let mut counts: BTreeMap<&[u8], u32> = BTreeMap::new();
        for u in &units {
            *counts.entry(u).or_default() += 1;
        }
        for (u, c) in counts {
            expected.entry(u.to_vec()).or_default().push((ordinal, c));
        }
        if s % 3500 == 3499 {
            index.commit().unwrap();
        }
    }
    index.commit().unwrap();
    Built { _dir: dir, index, expected, vocabulary }
}

#[test]
fn postings_hold_exactly_what_was_added() {
    let built = build(7);
    let snapshot = built.index.snapshot();
    assert_eq!(snapshot.source_count(), 20_000);
    let mut rng = StdRng::seed_from_u64(11);
    for unit in ["u0", "u1", "u2", "u5", "u50", "u500", "u2999"] {
        let added = &built.expected[unit.as_bytes()];
        let postings = snapshot.unit(unit.as_bytes()).unwrap();
        assert_eq!(postings.count, added.len() as u64, "{unit}");
        let decoded = snapshot.postings.decode(postings.head);
        assert_eq!(decoded.iter().map(|e| (e.0, e.1)).collect::<Vec<_>>(), *added, "{unit}: decode");

        let mut set = SourceSet::new(20_000);
        snapshot.postings.add_sources_to(postings.head, &mut set);
        assert_eq!(set.iter().collect::<Vec<_>>(), added.iter().map(|a| a.0).collect::<Vec<_>>(), "{unit}: sources");

        // advance to increasing targets: the first entry at or after each
        let mut cursor = snapshot.postings.cursor(postings.head);
        let mut target = 0u32;
        loop {
            target += rng.random_range(1..600);
            let i = added.partition_point(|a| a.0 < target);
            let found = cursor.advance(target);
            if i == added.len() {
                assert!(!found, "{unit}: past the end at {target}");
                break;
            }
            assert!(found);
            assert_eq!((cursor.doc, cursor.frequency), added[i], "{unit}: advancing to {target}");
        }

        // positions point at where the unit is
        let mut cursor = snapshot.postings.cursor(postings.head);
        let mut positions = Vec::new();
        for _ in 0..5 {
            assert!(cursor.next());
            let offset = cursor.positions_offset();
            snapshot.positions(offset, cursor.frequency, &mut positions);
            assert_eq!(positions.len() as u32, cursor.frequency);
        }
    }
}

#[test]
fn pruned_searches_return_what_scoring_everything_returns() {
    let built = build(3);
    let snapshot = built.index.snapshot();
    let relevances: Vec<Box<dyn Relevance>> =
        vec![Box::new(Bm25::default()), Box::new(Bm25 { quantize_lengths: true, ..Bm25::default() }), Box::new(TfIdf)];
    let mut rng = StdRng::seed_from_u64(5);
    for relevance in &relevances {
        let searcher = Searcher::new(&snapshot, relevance.as_ref());
        for top in [1usize, 10, 100] {
            let mut pruned = 0;
            for q in 0..150 {
                let terms = rng.random_range(1..7);
                let units: Vec<Vec<u8>> = (0..terms)
                    .map(|t| if rng.random_range(0..4) == 0 { format!("missing{t}").into_bytes() } else { built.vocabulary[(3000.0 * rng.random::<f64>().powi(2)) as usize].clone() })
                    .collect();
                let clause = Clause::should(units);
                let expected = searcher.search_all(&clause, top);
                let actual = searcher.search(&clause, top);
                if !actual.exact {
                    pruned += 1;
                }
                assert_eq!(actual.top, expected.top, "query {q} (top {top}): {:?}", clause.should.iter().map(|u| String::from_utf8_lossy(u)).collect::<Vec<_>>());
                assert!(actual.count <= expected.count);
            }
            assert!(pruned > 100, "plain disjunctions take the pruned path");
        }
    }
}

#[test]
fn boolean_clauses_and_phrases() {
    let dir = tempfile::tempdir().unwrap();
    let index = Index::open(dir.path(), Options::default()).unwrap();
    let docs = ["the white whale swims", "a whale of a time", "white sand and a white whale", "the sea"];
    for (i, d) in docs.iter().enumerate() {
        index.add(format!("{i}").as_bytes(), d.split(' ').enumerate().map(|(p, w)| (w.as_bytes(), p as u32)));
    }
    index.commit().unwrap();
    let snapshot = index.snapshot();
    let bm25 = Bm25::default();
    let searcher = Searcher::new(&snapshot, &bm25);
    let keys = |c: &Clause| {
        let mut k: Vec<u32> = searcher.search(c, 10).top.iter().map(|h| h.source).collect();
        k.sort();
        k
    };
    assert_eq!(keys(&Clause::phrase([b"white".to_vec(), b"whale".to_vec()])), vec![0, 2]);
    assert_eq!(keys(&Clause::phrase([b"a".to_vec(), b"white".to_vec()])), vec![2]);
    assert_eq!(keys(&Clause::must([b"whale".to_vec()])), vec![0, 1, 2]);
    let mut c = Clause::must([b"whale".to_vec()]);
    c.must_not = vec![b"white".to_vec()];
    assert_eq!(keys(&c), vec![1]);
    let mut c = Clause::default();
    c.must_not = vec![b"whale".to_vec()];
    assert_eq!(keys(&c), vec![3]);
}

#[test]
fn snapshots_and_crash_recovery() {
    let dir = tempfile::tempdir().unwrap();
    {
        let index = Index::open(dir.path(), Options::default()).unwrap();
        index.add(b"a", [(&b"x"[..], 0)]);
        index.commit().unwrap();
        let before = index.snapshot();
        index.add(b"b", [(&b"x"[..], 0), (&b"y"[..], 1)]);
        assert_eq!(index.snapshot().source_count(), 1, "uncommitted sources aren't visible");
        index.commit().unwrap();
        assert_eq!(before.source_count(), 1, "a snapshot keeps its commit");
        assert_eq!(before.unit(b"x").unwrap().count, 1);
        assert_eq!(index.snapshot().unit(b"x").unwrap().count, 2);
        // added but never committed: lost in the "crash" below
        index.add(b"c", [(&b"z"[..], 0)]);
    }
    // a crash mid-commit: bytes past the committed lengths
    use std::io::Write;
    std::fs::OpenOptions::new().append(true).open(dir.path().join("postings.dat")).unwrap().write_all(&[0xFF; 100]).unwrap();
    std::fs::OpenOptions::new().append(true).open(dir.path().join("sources.dat")).unwrap().write_all(&[0x05, b'j', b'u', b'n', b'k']).unwrap();
    let index = Index::open(dir.path(), Options::default()).unwrap();
    let snapshot = index.snapshot();
    assert_eq!(snapshot.source_count(), 2);
    assert_eq!(snapshot.sources.key(1), b"b");
    assert_eq!(snapshot.unit(b"z"), None);
    index.add(b"d", [(&b"x"[..], 0)]);
    index.commit().unwrap();
    assert_eq!(index.snapshot().unit(b"x").unwrap().count, 3);
    assert_eq!(index.snapshot().postings.decode(index.snapshot().unit(b"x").unwrap().head).iter().map(|e| e.0).collect::<Vec<_>>(), vec![0, 1, 2]);
}
