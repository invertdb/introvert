use std::collections::BTreeMap;

use std::ops::Bound;

use introvert::{Bm25, Clause, Overlap, Codec, Index, Options, Relevance, Searcher, Snapshot, SourceSet, Match, TfIdf, Words, MAX_UNIT_BYTES};
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
        let decoded = snapshot.postings.decode(&postings);
        assert_eq!(snapshot.postings.first_source(&postings), Some(added[0].0), "{unit}: first source");
        assert_eq!(decoded.iter().map(|e| (e.0, e.1)).collect::<Vec<_>>(), *added, "{unit}: decode");

        let mut set = SourceSet::new(20_000);
        snapshot.postings.add_sources_to(&postings, &mut set);
        assert_eq!(set.iter().collect::<Vec<_>>(), added.iter().map(|a| a.0).collect::<Vec<_>>(), "{unit}: sources");

        // advance to increasing targets: the first entry at or after each
        let mut cursor = snapshot.postings.cursor(&postings);
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
        let mut cursor = snapshot.postings.cursor(&postings);
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
fn rare_units_stay_inline_then_move_to_records() {
    // units in a handful of sources up to a few dozen, added over several commits: inline values, inline ones merged
    // with new entries, inline ones moving to a record, then chains; under every combination of options
    for (frequencies, positions) in [(true, true), (true, false), (false, false)] {
        let dir = tempfile::tempdir().unwrap();
        let options = Options { frequencies, positions, ..Options::default() };
        let mut index: Index<u32, String> = Index::open(dir.path(), options).unwrap();
        let mut rng = StdRng::seed_from_u64(21);
        // unit -> (source, its positions), in source order
        let mut expected: BTreeMap<String, Vec<(u32, Vec<u32>)>> = BTreeMap::new();
        let mut source = 0u32;
        let check = |index: &Index<u32, String>, expected: &BTreeMap<String, Vec<(u32, Vec<u32>)>>, round: usize| {
            let snapshot = index.snapshot();
            let mut found = Vec::new();
            for (unit, added) in expected {
                let context = format!("{unit} ({frequencies}, {positions}) round {round}");
                let postings = snapshot.unit(unit).unwrap();
                assert_eq!(postings.count, added.len() as u64, "{context}");
                assert_eq!(snapshot.postings.first_source(&postings), Some(added[0].0), "{context}");
                let frequency = |p: &Vec<u32>| if frequencies { p.len() as u32 } else { 1 };
                let decoded: Vec<(u32, u32)> = snapshot.postings.decode(&postings).iter().map(|e| (e.0, e.1)).collect();
                assert_eq!(decoded, added.iter().map(|(s, p)| (*s, frequency(p))).collect::<Vec<_>>(), "{context}");
                let mut cursor = snapshot.postings.cursor(&postings);
                for (s, p) in added {
                    assert!(cursor.next() && cursor.source == *s, "{context}");
                    if positions {
                        snapshot.positions(cursor.positions_offset(), cursor.frequency, &mut found);
                        assert_eq!(&found, p, "{context}: positions in {s}");
                    }
                }
                assert!(!cursor.next(), "{context}: past the end");
                let mut cursor = snapshot.postings.cursor(&postings);
                let target = added[added.len() / 2].0;
                assert!(cursor.advance(target) && cursor.source == target, "{context}: advance");
                let mut impacts = 0;
                snapshot.postings.impacts(&postings, |s| snapshot.source_length(s), |_, _| impacts += 1);
                assert!(impacts >= 1, "{context}: impacts");
            }
            // a search ranks the same either way (pruned path vs all)
            let searcher = Searcher::new(&snapshot, &TfIdf);
            let clause = Clause::should(expected.keys().take(5).cloned());
            assert_eq!(searcher.search(&clause, 10).top, searcher.search_all(&clause, 10).top);
        };
        for round in 0..6 {
            for _ in 0..200 {
                // unit r{n} turns up in about n sources a round (r0 in none, until it's a rare one-off)
                let mut units: Vec<(String, u32)> = Vec::new();
                for n in 0..40u32 {
                    if rng.random_range(0..200) < n.max(1) {
                        let unit = format!("r{n}");
                        for _ in 0..rng.random_range(1..4) {
                            units.push((unit.clone(), units.len() as u32));
                        }
                    }
                }
                units.push(("filler".to_string(), units.len() as u32));
                for (unit, position) in &units {
                    let entries = expected.entry(unit.clone()).or_default();
                    match entries.last_mut() {
                        Some((s, p)) if *s == source => p.push(*position),
                        _ => entries.push((source, vec![*position])),
                    }
                }
                assert_eq!(index.add(&source, units), source);
                source += 1;
            }
            index.commit().unwrap();
            check(&index, &expected, round);
        }
        drop(index);
        index = Index::open(dir.path(), options).unwrap();
        check(&index, &expected, 99);
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
    pruned_equals_exhaustive(&built, &snapshot, &Overlap, &mut rng);
}

#[test]
fn overlap_scores_the_size_of_the_intersection() {
    // sources as sets; a query's top sources are those sharing the most of its units
    let dir = tempfile::tempdir().unwrap();
    let index: Index<u32, u32> = Index::open(dir.path(), Options { frequencies: false, positions: false, ..Options::default() }).unwrap();
    let mut rng = StdRng::seed_from_u64(4);
    let sets: Vec<std::collections::BTreeSet<u32>> = (0..500).map(|_| (0..rng.random_range(5..200)).map(|_| rng.random_range(0..2000)).collect()).collect();
    for (key, set) in sets.iter().enumerate() {
        index.add(&(key as u32), set.iter().map(|&u| (u, 0)));
    }
    index.commit().unwrap();
    let snapshot = index.snapshot();
    let searcher = Searcher::new(&snapshot, &Overlap);
    for q in [3usize, 77, 401] {
        let found = searcher.search(&Clause::should(sets[q].iter().copied()), 10);
        let mut expected: Vec<(usize, u32)> = sets.iter().enumerate().map(|(s, set)| (sets[q].intersection(set).count(), s as u32)).collect();
        expected.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let got: Vec<(usize, u32)> = found.top.iter().map(|r| (r.score as usize, r.key)).collect();
        assert_eq!(got, expected[..10].to_vec(), "query {q}");
        assert_eq!(got[0], (sets[q].len(), q as u32), "a set overlaps itself most");
    }
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
    assert_eq!(snapshot.postings.decode(&snapshot.unit(&x()).unwrap()).iter().map(|e| e.0).collect::<Vec<_>>(), vec![0, 1, 2]);
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

#[test]
fn alignment_matches_voting_by_brute_force() {
    // align against a reference that votes every (query, source) occurrence pair from decoded postings
    let dir = tempfile::tempdir().unwrap();
    let index: Index<u32, u32> = Index::open(dir.path(), Options::default()).unwrap();
    let mut rng = StdRng::seed_from_u64(31);
    for key in 0..300u32 {
        let length = rng.random_range(20..400);
        // a small alphabet so units repeat within and across sources
        index.add(&key, (0..length).map(|p| (rng.random_range(0..800u32), p)));
        if key % 70 == 69 {
            index.commit().unwrap();
        }
    }
    index.commit().unwrap();
    let snapshot = index.snapshot();
    let relevance = TfIdf;
    let searcher = Searcher::new(&snapshot, &relevance);
    let reference = |query: &[(u32, u32)], tolerance: i64, depth: usize, max_source_frequency: u64| {
        let mut by_unit: BTreeMap<u32, Vec<i64>> = BTreeMap::new();
        for &(u, p) in query {
            by_unit.entry(u).or_default().push(p as i64);
        }
        // source -> (shared, votes)
        let mut sources: BTreeMap<u32, (u64, Vec<i64>)> = BTreeMap::new();
        let mut positions = Vec::new();
        for (u, qs) in &by_unit {
            let Some(postings) = snapshot.unit(u) else { continue };
            if max_source_frequency > 0 && postings.count > max_source_frequency {
                continue;
            }
            for (source, frequency, offset) in snapshot.postings.decode(&postings) {
                let entry = sources.entry(source).or_default();
                entry.0 += frequency as u64 * qs.len() as u64;
                snapshot.positions(offset, frequency, &mut positions);
                for &p in &positions {
                    entry.1.extend(qs.iter().map(|q| p as i64 - q));
                }
            }
        }
        let mut ranked: Vec<(u32, u64, Vec<i64>)> = sources.into_iter().map(|(s, (shared, votes))| (s, shared, votes)).collect();
        if depth > 0 {
            ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            ranked.truncate(depth);
        }
        let mut best: Vec<(u32, u64, u64, i64)> = ranked
            .into_iter()
            .map(|(s, shared, mut votes)| {
                votes.sort();
                // the lowest offset whose window holds the most votes
                let (count, offset) = votes.iter().fold((0, 0), |(c, o), &v| {
                    let n = votes.iter().filter(|&&w| (w - v).abs() <= tolerance).count();
                    if n > c { (n, v) } else { (c, o) }
                });
                (s, shared, count as u64, offset)
            })
            .collect();
        best.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));
        best
    };
    for q in 0..60 {
        let length = rng.random_range(1..80);
        let query: Vec<(u32, u32)> = (0..length).map(|i| (rng.random_range(0..800u32), 500 + i * rng.random_range(1..3))).collect();
        let (tolerance, depth, max_source_frequency) = [(0, 0, 0), (3, 10, 0), (20, 50, 60), (1, 1, 0)][q % 4];
        let found = searcher.align(&query, 1000, tolerance, depth, max_source_frequency);
        let expected = reference(&query, tolerance, depth, max_source_frequency);
        let found: Vec<(u32, u64, u64, i64)> = found.iter().map(|a| (a.source, a.matches, a.count, a.offset)).collect();
        assert_eq!(found, expected, "query {q}");
        assert!(searcher.align(&query, 3, tolerance, depth, max_source_frequency).iter().all(|a| a.key == a.source));
    }
}

#[test]
fn intersecting_a_small_set_with_a_large_one() {
    // the larger unit's sources are kept by advancing its cursor to each of the smaller set's
    let built = build(12);
    let snapshot = built.index.snapshot();
    let relevance = TfIdf;
    let searcher = Searcher::new(&snapshot, &relevance);
    for (rare, common) in [("u2999", "u0"), ("u1500", "u1"), ("u40", "u0"), ("u2999", "u2998")] {
        let have = |u: &str| built.expected.get(u).map(|e| e.iter().map(|x| x.0).collect::<std::collections::BTreeSet<u32>>()).unwrap_or_default();
        let expected: Vec<u32> = have(rare).intersection(&have(common)).copied().collect();
        let found: Vec<u32> = searcher.matching(&Clause::must([rare.to_string(), common.to_string()])).iter().collect();
        assert_eq!(found, expected, "{rare} and {common}");
        let found: Vec<u32> = searcher.matching(&Clause::must([Match::Unit(rare.to_string()), Match::Any(vec![common.to_string(), "u3".to_string()])])).iter().collect();
        let either: std::collections::BTreeSet<u32> = have(common).union(&have("u3")).copied().collect();
        assert_eq!(found, have(rare).intersection(&either).copied().collect::<Vec<_>>(), "{rare} and any of {common}, u3");
        // and excluding the common one: its cursor removes sources from the rare one's
        let mut c = Clause::must([rare.to_string()]);
        c.must_not = vec![Match::Unit(common.to_string())];
        let found: Vec<u32> = searcher.matching(&c).iter().collect();
        assert_eq!(found, have(rare).difference(&have(common)).copied().collect::<Vec<_>>(), "{rare} without {common}");
    }
}

#[test]
fn units_too_long_for_the_dictionary_are_left_out_and_the_commit_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let index: Index<String, String> = Index::open(dir.path(), Options::default()).unwrap();
    let longest = "a".repeat(MAX_UNIT_BYTES);
    let too_long = "b".repeat(MAX_UNIT_BYTES + 1);
    let units = [("short".to_string(), 0), (too_long.clone(), 1), (longest.clone(), 2), (too_long.clone(), 3)];
    index.add(&"s0".to_string(), units);
    index.add(&"s1".to_string(), [("short".to_string(), 0)]);
    index.commit().unwrap();
    assert_eq!(index.dropped_units(), 2);
    let snapshot = index.snapshot();
    assert_eq!(snapshot.source_length(0), 2, "the units kept");
    assert_eq!(snapshot.unit(&"short".to_string()).unwrap().count, 2);
    assert_eq!(snapshot.unit(&longest).unwrap().count, 1);
    assert!(snapshot.unit(&too_long).is_none());
}

#[test]
fn a_compacted_index_answers_the_same() {
    let dir = tempfile::tempdir().unwrap();
    let store = larry::Store::open(dir.path()).unwrap();
    let vocabulary: Vec<String> = (0..5000).map(|i| format!("unit{i}")).collect();
    {
        let index: Index<String, String> = Index::open_in(&store, Options::default()).unwrap();
        for s in 0..20_000u32 {
            let units = (0..6).map(|j| (vocabulary[((s * 31 + j * 977) % 5000) as usize].clone(), j));
            index.add(&format!("s{s}"), units);
            if s % 2000 == 1999 {
                index.commit().unwrap();
            }
        }
        index.commit().unwrap();
    }
    let before: Vec<(String, u64)> = {
        let index: Index<String, String> = Index::open_in(&store, Options::default()).unwrap();
        index.snapshot().units(..).map(|(u, p)| (u, p.count)).collect()
    };
    let size = |s: &larry::Store| std::fs::metadata(s.path("units.db").unwrap()).unwrap().len();
    let old = size(&store);
    Index::<String, String>::compact_in(&store).unwrap();
    assert!(size(&store) < old, "compacted {} from {old}", size(&store));
    let index: Index<String, String> = Index::open_in(&store, Options::default()).unwrap();
    let snapshot = index.snapshot();
    assert_eq!(snapshot.units(..).map(|(u, p)| (u, p.count)).collect::<Vec<_>>(), before);
    assert_eq!(snapshot.source_count(), 20_000);
    let found = snapshot.unit(&vocabulary[42]).unwrap();
    assert_eq!(snapshot.postings.decode(&found).len() as u64, found.count);
}

#[test]
fn excluded_sources_are_left_out_before_the_top_is_chosen() {
    let built = build(11);
    let snapshot = built.index.snapshot();
    let mut rng = StdRng::seed_from_u64(12);
    // every third source and a run of 2,000, as a collection's deleted sources
    let mut excluded = SourceSet::new(snapshot.source_count());
    for s in (0..snapshot.source_count()).filter(|s| s % 3 == 0 || (5000..7000).contains(s)) {
        excluded.insert(s);
    }
    let relevance = Bm25::default();
    let all = Searcher::new(&snapshot, &relevance);
    let live = Searcher::new(&snapshot, &relevance).without(&excluded);
    for top in [1usize, 10, 100] {
        for q in 0..100 {
            let count = rng.random_range(1..6);
            let units: Vec<String> = (0..count).map(|_| built.vocabulary[(3000.0 * rng.random::<f64>().powi(2)) as usize].clone()).collect();
            let clause = Clause::should(units);
            // the reference: everything scored, the excluded removed, then the top taken
            let every = all.search_all(&clause, snapshot.source_count() as usize);
            let expected: Vec<_> = every.top.iter().filter(|r| !excluded.contains(r.source)).take(top).map(|r| (r.source, r.score)).collect();
            let exhaustive = live.search_all(&clause, top);
            let pruned = live.search(&clause, top);
            assert_eq!(exhaustive.top.iter().map(|r| (r.source, r.score)).collect::<Vec<_>>(), expected, "query {q}, top {top}");
            assert_eq!(pruned.top.iter().map(|r| (r.source, r.score)).collect::<Vec<_>>(), expected, "query {q}, top {top} (pruned)");
            assert_eq!(exhaustive.count as usize, every.top.iter().filter(|r| !excluded.contains(r.source)).count());
            assert!(live.matching(&clause).iter().all(|s| !excluded.contains(s)));
        }
    }
    // an empty set excludes nothing
    let none = SourceSet::new(snapshot.source_count());
    let clause = Clause::should(built.vocabulary[..3].to_vec());
    assert_eq!(Searcher::new(&snapshot, &relevance).without(&none).search(&clause, 10).top.len(), all.search(&clause, 10).top.len());
}

#[test]
fn alignment_leaves_out_excluded_sources() {
    let dir = tempfile::tempdir().unwrap();
    let index: Index<u32, u32> = Index::open(dir.path(), Options::default()).unwrap();
    let mut rng = StdRng::seed_from_u64(8);
    let sequences: Vec<Vec<u32>> = (0..200).map(|_| (0..500).map(|_| rng.random_range(0..5000)).collect()).collect();
    for (key, sequence) in sequences.iter().enumerate() {
        index.add(&(key as u32), sequence.iter().enumerate().map(|(p, &u)| (u, p as u32)));
    }
    // source 200 is a copy of source 7
    index.add(&200, sequences[7].iter().enumerate().map(|(p, &u)| (u, p as u32)));
    index.commit().unwrap();
    let snapshot = index.snapshot();
    let relevance = TfIdf;
    let query: Vec<(u32, u32)> = (0..60).map(|i| (sequences[7][100 + i], i as u32)).collect();
    let found = Searcher::new(&snapshot, &relevance).align(&query, 2, 1, 100, 0);
    assert_eq!(found.iter().map(|a| a.key).collect::<Vec<_>>(), [7, 200]);
    let mut excluded = SourceSet::new(snapshot.source_count());
    excluded.insert(7);
    let found = Searcher::new(&snapshot, &relevance).without(&excluded).align(&query, 2, 1, 100, 0);
    assert_eq!(found[0].key, 200);
    assert!(found.len() < 2 || found[1].count < 10, "{found:?}");
}
