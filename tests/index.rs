use std::collections::BTreeMap;

use std::ops::Bound;

use introvert::{Bm25, Clause, Codec, Index, Options, Relevance, Searcher, Snapshot, SourceSet, Match, TfIdf, Words};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

struct Built {
    _dir: tempfile::TempDir,
    index: Index<String, String>,
    /// unit -> (source, frequency), in source order
    expected: BTreeMap<String, Vec<(u32, u32)>>,
    vocabulary: Vec<String>,
}

/// 20,000 sources of 5..80 units from a skewed vocabulary, committed every few thousand (so chains of records).
fn build(seed: u64) -> Built {
    let dir = tempfile::tempdir().unwrap();
    let index = Index::open(dir.path(), Options::default()).unwrap();
    let mut rng = StdRng::seed_from_u64(seed);
    let vocabulary: Vec<String> = (0..3000).map(|i| format!("u{i}")).collect();
    let mut expected: BTreeMap<String, Vec<(u32, u32)>> = BTreeMap::new();
    for s in 0..20_000u32 {
        let length = rng.random_range(5..80);
        let units: Vec<&String> = (0..length).map(|_| &vocabulary[(3000.0 * rng.random::<f64>().powi(3)) as usize]).collect();
        let ordinal = index.add(&format!("source{s}"), units.iter().enumerate().map(|(i, u)| ((*u).clone(), i as u32)));
        assert_eq!(ordinal, s);
        let mut counts: BTreeMap<&String, u32> = BTreeMap::new();
        for u in &units {
            *counts.entry(u).or_default() += 1;
        }
        for (u, c) in counts {
            expected.entry(u.clone()).or_default().push((ordinal, c));
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
        let added = &built.expected[unit];
        let postings = snapshot.unit(&unit.to_string()).unwrap();
        assert_eq!(postings.count, added.len() as u64, "{unit}");
        let decoded = snapshot.postings.decode(postings.head);
        assert_eq!(snapshot.postings.first_source(postings.head), Some(added[0].0), "{unit}: first source");
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
            assert_eq!((cursor.source, cursor.frequency), added[i], "{unit}: advancing to {target}");
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

fn pruned_equals_exhaustive<R: Relevance>(built: &Built, snapshot: &Snapshot<String, String>, relevance: &R, rng: &mut StdRng) {
    let searcher = Searcher::new(snapshot, relevance);
    for top in [1usize, 10, 100] {
        let mut pruned = 0;
        for q in 0..150 {
            let count = rng.random_range(1..7);
            let units: Vec<String> = (0..count)
                .map(|t| if rng.random_range(0..4) == 0 { format!("missing{t}") } else { built.vocabulary[(3000.0 * rng.random::<f64>().powi(2)) as usize].clone() })
                .collect();
            let clause = Clause::should(units);
            let expected = searcher.search_all(&clause, top);
            let actual = searcher.search(&clause, top);
            if !actual.count_is_exact {
                pruned += 1;
            }
            assert_eq!(actual.top, expected.top, "query {q} (top {top}): {:?}", clause.should);
            assert!(actual.count <= expected.count);
        }
        assert!(pruned > 100, "plain disjunctions take the pruned path");
    }
}

#[test]
fn pruned_searches_return_what_scoring_everything_returns() {
    let built = build(3);
    let snapshot = built.index.snapshot();
    let mut rng = StdRng::seed_from_u64(5);
    pruned_equals_exhaustive(&built, &snapshot, &Bm25::default(), &mut rng);
    pruned_equals_exhaustive(&built, &snapshot, &Bm25 { k1: 2.0, b: 0.3 }, &mut rng);
    pruned_equals_exhaustive(&built, &snapshot, &TfIdf, &mut rng);
}

#[test]
fn boolean_clauses_and_phrases() {
    let dir = tempfile::tempdir().unwrap();
    let index: Index<u32, String> = Index::open(dir.path(), Options::default()).unwrap();
    let texts = ["the white whale swims", "a whale of a time", "white sand and a white whale", "the sea"];
    for (i, text) in texts.iter().enumerate() {
        index.invert(&(i as u32 * 10), *text, &Words::default());
    }
    index.commit().unwrap();
    let snapshot = index.snapshot();
    let bm25 = Bm25::default();
    let searcher = Searcher::new(&snapshot, &bm25);
    let keys = |c: &Clause<String>| {
        let mut k: Vec<u32> = searcher.search(c, 10).top.iter().map(|r| r.key).collect();
        k.sort();
        k
    };
    let s = |w: &str| w.to_string();
    assert_eq!(keys(&Clause::phrase([s("white"), s("whale")])), vec![0, 20]);
    assert_eq!(keys(&Clause::phrase([s("a"), s("white")])), vec![20]);
    assert_eq!(keys(&Clause::must([s("whale")])), vec![0, 10, 20]);
    let mut c = Clause::must([s("whale")]);
    c.must_not = vec![Match::Unit(s("white"))];
    assert_eq!(keys(&c), vec![10]);
    let mut c = Clause::default();
    c.must_not = vec![Match::Unit(s("whale"))];
    assert_eq!(keys(&c), vec![30]);
    assert_eq!(keys(&Clause::should([Match::Prefix(s("wh"))])), vec![0, 10, 20]);
    assert_eq!(keys(&Clause::must([Match::Any(vec![s("swims"), s("sea"), s("absent")])])), vec![0, 30]);
    assert_eq!(keys(&Clause::must([Match::Range(Bound::Included(s("sa")), Bound::Excluded(s("t")))])), vec![0, 20, 30]);
}

/// A unit of a typed index: a measurement's name and a bucketed value, ordered by name then value.
#[derive(Clone, Debug, PartialEq)]
struct Reading {
    metric: String,
    bucket: i64,
}

impl Codec for Reading {
    fn encode(&self, out: &mut Vec<u8>) {
        (self.metric.clone(), self.bucket).encode(out)
    }
    fn decode(bytes: &[u8]) -> Self {
        let (metric, bucket) = <(String, i64)>::decode(bytes);
        Reading { metric, bucket }
    }
    fn encode_nested(&self, out: &mut Vec<u8>) {
        (self.metric.clone(), self.bucket).encode_nested(out)
    }
    fn decode_nested(bytes: &[u8], at: &mut usize) -> Self {
        let (metric, bucket) = <(String, i64)>::decode_nested(bytes, at);
        Reading { metric, bucket }
    }
}

#[test]
fn typed_units_and_ranges() {
    // sources keyed by (station, day); units are readings: ranges of typed units are ranges of values
    let dir = tempfile::tempdir().unwrap();
    let index: Index<(String, u16), Reading> = Index::open(dir.path(), Options { positions: false, ..Options::default() }).unwrap();
    let r = |metric: &str, bucket: i64| Reading { metric: metric.into(), bucket };
    index.add(&("oslo".into(), 1), [(r("temp", -12), 0), (r("wind", 4), 0)]);
    index.add(&("oslo".into(), 2), [(r("temp", -3), 0), (r("wind", 9), 0)]);
    index.add(&("lima".into(), 1), [(r("temp", 19), 0), (r("wind", 2), 0)]);
    index.add(&("lima".into(), 2), [(r("temp", 22), 0), (r("tempest", 1), 0)]);
    index.commit().unwrap();
    let snapshot = index.snapshot();
    let bm25 = Bm25::default();
    let searcher = Searcher::new(&snapshot, &bm25);
    let sources = |c: Clause<Reading>| {
        let mut s: Vec<(String, u16)> = searcher.search(&c, 10).top.into_iter().map(|r| r.key).collect();
        s.sort();
        s
    };
    // below freezing, across the sign boundary
    let freezing = Match::Range(Bound::Included(r("temp", i64::MIN)), Bound::Excluded(r("temp", 0)));
    assert_eq!(sources(Clause::must([freezing])), vec![("oslo".into(), 1), ("oslo".into(), 2)]);
    let warm = Match::Range(Bound::Included(r("temp", 15)), Bound::Included(r("temp", i64::MAX)));
    let windy = Match::Range(Bound::Included(r("wind", 5)), Bound::Included(r("wind", i64::MAX)));
    assert_eq!(sources(Clause { must: vec![warm], must_not: vec![windy], ..Default::default() }), vec![("lima".into(), 1), ("lima".into(), 2)]);
    // ranges are over whole units: unbounded, this one runs past "temp" into "tempest" and "wind" (oslo 1: wind 4)
    let past = Match::Range(Bound::Included(r("temp", 15)), Bound::Unbounded);
    assert_eq!(sources(Clause::must([past])), vec![("lima".into(), 1), ("lima".into(), 2), ("oslo".into(), 1), ("oslo".into(), 2)]);
    // the units in order: by metric, then by value (negatives first)
    let temps: Vec<i64> = snapshot.units(r("temp", i64::MIN)..r("temp", i64::MAX)).map(|(u, _)| u.bucket).collect();
    assert_eq!(temps, vec![-12, -3, 19, 22]);
    assert_eq!(snapshot.key(2), ("lima".to_string(), 1));
}

#[test]
fn prefixes_of_tuple_units_stay_within_leading_elements() {
    let dir = tempfile::tempdir().unwrap();
    let index: Index<String, (String, String)> = Index::open(dir.path(), Options::default()).unwrap();
    let words = Words::default();
    for (id, title, text) in [("a", "Whale Song", "the sea"), ("b", "Sea Shanties", "a whale of a time"), ("c", "Whaling", "harpoons")] {
        let mut units = Vec::new();
        let mut position = 0;
        for (part, value) in [("title", title), ("text", text)] {
            words.each(value, |w, p| units.push(((part.to_string(), w.to_string()), position + p)));
            position += 100;
        }
        index.add(&id.to_string(), units);
    }
    index.commit().unwrap();
    let snapshot = index.snapshot();
    let bm25 = Bm25::default();
    let searcher = Searcher::new(&snapshot, &bm25);
    let ids = |c: Clause<(String, String)>| {
        let mut s: Vec<String> = searcher.search(&c, 10).top.into_iter().map(|r| r.key).collect();
        s.sort();
        s
    };
    let t = |f: &str, w: &str| (f.to_string(), w.to_string());
    assert_eq!(ids(Clause::should([Match::Prefix(t("title", "whal"))])), vec!["a", "c"]);
    assert_eq!(ids(Clause::should([Match::Prefix(t("text", "whal"))])), vec!["b"]);
    assert_eq!(ids(Clause::phrase([t("text", "a"), t("text", "whale")])), vec!["b"]);
    let titles: Vec<String> = snapshot.units_with_prefix(&t("title", "")).map(|(u, _)| u.1).collect();
    assert_eq!(titles, vec!["sea", "shanties", "song", "whale", "whaling"]);
}

#[test]
fn snapshots_and_crash_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let x = || "x".to_string();
    {
        let index: Index<String, String> = Index::open(dir.path(), Options::default()).unwrap();
        index.add(&"a".into(), [(x(), 0)]);
        index.commit().unwrap();
        let before = index.snapshot();
        index.add(&"b".into(), [(x(), 0), ("y".into(), 1)]);
        assert_eq!(index.snapshot().source_count(), 1, "uncommitted sources aren't visible");
        index.commit().unwrap();
        assert_eq!(before.source_count(), 1, "a snapshot keeps its commit");
        assert_eq!(before.unit(&x()).unwrap().count, 1);
        assert_eq!(index.snapshot().unit(&x()).unwrap().count, 2);
        // added but never committed: lost in the "crash" below
        index.add(&"c".into(), [("z".into(), 0)]);
    }
    // a crash mid-commit: bytes past the committed lengths
    use std::io::Write;
    std::fs::OpenOptions::new().append(true).open(dir.path().join("postings.dat")).unwrap().write_all(&[0xFF; 100]).unwrap();
    std::fs::OpenOptions::new().append(true).open(dir.path().join("sources.dat")).unwrap().write_all(&[0x05, b'j', b'u', b'n', b'k']).unwrap();
    let index: Index<String, String> = Index::open(dir.path(), Options::default()).unwrap();
    let snapshot = index.snapshot();
    assert_eq!(snapshot.source_count(), 2);
    assert_eq!(snapshot.key(1), "b");
    assert_eq!(snapshot.unit(&"z".into()), None);
    index.add(&"d".into(), [(x(), 0)]);
    index.commit().unwrap();
    let snapshot = index.snapshot();
    assert_eq!(snapshot.unit(&x()).unwrap().count, 3);
    assert_eq!(snapshot.postings.decode(snapshot.unit(&x()).unwrap().head).iter().map(|e| e.0).collect::<Vec<_>>(), vec![0, 1, 2]);
    // the empty string is a unit like any other
    index.add(&"e".into(), [(String::new(), 0)]);
    index.commit().unwrap();
    assert_eq!(index.snapshot().unit(&String::new()).unwrap().count, 1);
}

#[test]
fn alignment_finds_where_a_query_lines_up() {
    // sources are sequences of numbered units; a query is a stretch of one of them, with units missing and noise
    let dir = tempfile::tempdir().unwrap();
    let index: Index<u32, u32> = Index::open(dir.path(), Options::default()).unwrap();
    let mut rng = StdRng::seed_from_u64(8);
    let sequences: Vec<Vec<u32>> = (0..200).map(|_| (0..500).map(|_| rng.random_range(0..5000)).collect()).collect();
    for (key, sequence) in sequences.iter().enumerate() {
        index.add(&(key as u32 + 100), sequence.iter().enumerate().map(|(p, &u)| (u, p as u32)));
    }
    index.commit().unwrap();
    let snapshot = index.snapshot();
    let relevance = TfIdf;
    let searcher = Searcher::new(&snapshot, &relevance);
    for (source, start) in [(7usize, 120usize), (150, 0), (42, 400)] {
        // 60 units from position `start`, every third dropped, plus 20 random units, at query positions from 1000
        let mut query: Vec<(u32, u32)> = (0..60).filter(|i| i % 3 != 0).map(|i| (sequences[source][start + i], 1000 + i as u32)).collect();
        query.extend((0..20).map(|i| (rng.random_range(0..5000), 1000 + i)));
        let found = searcher.align(&query, 3, 1, 100, 0);
        assert_eq!(found[0].key, source as u32 + 100);
        assert_eq!(found[0].offset, start as i64 - 1000);
        assert!(found[0].count >= 40 && (found.len() < 2 || found[1].count < 10), "{found:?}");
    }
}
