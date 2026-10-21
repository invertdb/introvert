//! Queries: boolean clauses over units, matched and ranked.
//!
//! A [`Clause`] requires (must), prefers (should) or excludes (must not) units and sub-clauses, and may be a phrase.
//! [`Searcher::search`] ranks the top k: a plain disjunction of units takes the pruned (MaxScore) path, anything
//! else scores every match. Both return the same results, summed in the same order.

use std::cell::RefCell;
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

use crate::index::{Snapshot, UnitPostings};
use crate::postings::{Cursor, NO_MORE};
use crate::relevance::{Bm25, Relevance};
use crate::sourceset::SourceSet;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Clause {
    pub must: Vec<Vec<u8>>,
    pub should: Vec<Vec<u8>>,
    pub must_not: Vec<Vec<u8>>,
    /// Units that must occur at consecutive positions (with two or more).
    pub phrase: Vec<Vec<u8>>,
    pub must_clauses: Vec<Clause>,
    pub should_clauses: Vec<Clause>,
    pub must_not_clauses: Vec<Clause>,
}

impl Clause {
    pub fn should(units: impl IntoIterator<Item = Vec<u8>>) -> Clause {
        Clause { should: units.into_iter().collect(), ..Default::default() }
    }

    pub fn must(units: impl IntoIterator<Item = Vec<u8>>) -> Clause {
        Clause { must: units.into_iter().collect(), ..Default::default() }
    }

    pub fn phrase(units: impl IntoIterator<Item = Vec<u8>>) -> Clause {
        let phrase: Vec<Vec<u8>> = units.into_iter().collect();
        Clause { must: phrase.clone(), phrase, ..Default::default() }
    }

    fn is_plain_disjunction(&self) -> bool {
        !self.should.is_empty()
            && self.must.is_empty()
            && self.must_not.is_empty()
            && self.phrase.len() <= 1
            && self.must_clauses.is_empty()
            && self.should_clauses.is_empty()
            && self.must_not_clauses.is_empty()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub source: u32,
    pub score: f64,
    /// Occurrences of the query's positive units and phrases in the source.
    pub hits: u32,
}

#[derive(Clone, Debug)]
pub struct Results {
    /// Matching sources: exact, or (when `exact` is false) the ones scored, a lower bound.
    pub count: u64,
    pub exact: bool,
    pub top: Vec<Hit>,
}

/// The index units a query unit stands for.
pub type Expander = dyn Fn(&Snapshot, &[u8]) -> Vec<UnitPostings> + Send + Sync;

fn exact(snapshot: &Snapshot, unit: &[u8]) -> Vec<UnitPostings> {
    snapshot.unit(unit).into_iter().collect()
}

pub struct Searcher<'s> {
    snapshot: &'s Snapshot,
    relevance: &'s dyn Relevance,
    expand: &'s Expander,
    bounds: RefCell<HashMap<u64, f64>>,
}

/// A heap entry, ordered so the heap's top is the worst kept: lowest score, then highest source.
#[derive(Clone, Copy)]
struct Kept(Hit);

impl PartialEq for Kept {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Kept {}
impl PartialOrd for Kept {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Kept {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap: "greater" = worse = lower score, then higher source
        other.0.score.partial_cmp(&self.0.score).unwrap_or(Ordering::Equal).then(self.0.source.cmp(&other.0.source))
    }
}

/// Keeps the best `top`; sources arrive in increasing order, so a later one displaces the worst only if it scores
/// strictly higher.
struct TopK {
    top: usize,
    heap: BinaryHeap<Kept>,
}

impl TopK {
    fn new(top: usize) -> TopK {
        TopK { top, heap: BinaryHeap::with_capacity(top + 1) }
    }

    fn full(&self) -> bool {
        self.heap.len() == self.top
    }

    fn threshold(&self) -> f64 {
        if self.full() { self.heap.peek().unwrap().0.score } else { f64::NEG_INFINITY }
    }

    fn offer(&mut self, hit: Hit) {
        if self.heap.len() < self.top {
            self.heap.push(Kept(hit));
        } else if self.top > 0 && hit.score > self.heap.peek().unwrap().0.score {
            self.heap.pop();
            self.heap.push(Kept(hit));
        }
    }

    fn into_sorted(self) -> Vec<Hit> {
        let mut hits: Vec<Hit> = self.heap.into_iter().map(|k| k.0).collect();
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal).then(a.source.cmp(&b.source)));
        hits
    }
}

/// A query unit's postings, merged across what it expands to, decoded on first use.
struct UnitData {
    parts: Vec<UnitPostings>,
    estimate: u64,
    entries: Option<Vec<(u32, u32, u64)>>,
}

struct Query<'q, 's> {
    searcher: &'q Searcher<'s>,
    units: HashMap<Vec<u8>, UnitData>,
    phrase_frequencies: HashMap<*const Clause, HashMap<u32, u32>>,
}

impl<'s> Searcher<'s> {
    pub fn new(snapshot: &'s Snapshot, relevance: &'s dyn Relevance) -> Searcher<'s> {
        Searcher { snapshot, relevance, expand: &exact, bounds: RefCell::new(HashMap::new()) }
    }

    /// How query units are found: by default exactly; e.g. prefix queries expand to every unit with a prefix.
    pub fn with_expander(mut self, expand: &'s Expander) -> Searcher<'s> {
        self.expand = expand;
        self
    }

    pub fn snapshot(&self) -> &Snapshot {
        self.snapshot
    }

    /// The sources matching a clause, unranked.
    pub fn matching(&self, clause: &Clause) -> SourceSet {
        Query::new(self).evaluate(clause)
    }

    /// The best `top` sources for a clause, best first.
    pub fn search(&self, clause: &Clause, top: usize) -> Results {
        if top > 0 && clause.is_plain_disjunction() {
            let mut units: Vec<&[u8]> = Vec::new();
            for unit in &clause.should {
                if !units.contains(&unit.as_slice()) {
                    units.push(unit);
                }
            }
            if let Some(results) = self.search_disjunction(&units, top) {
                return results;
            }
        }
        self.search_all(clause, top)
    }

    /// Scores every matching source; counts them exactly.
    pub fn search_all(&self, clause: &Clause, top: usize) -> Results {
        let mut query = Query::new(self);
        let matches = query.evaluate(clause);
        let n = self.snapshot.source_count() as usize;
        let mut scores = vec![0.0f64; n];
        let mut hits = vec![0u32; n];
        query.score(clause, &matches, &mut scores, &mut hits);
        let mut best = TopK::new(top);
        let mut count = 0;
        for source in matches.iter() {
            count += 1;
            best.offer(Hit { source, score: scores[source as usize], hits: hits[source as usize] });
        }
        Results { count, exact: true, top: best.into_sorted() }
    }

    fn bound(&self, unit: UnitPostings, weight: f64) -> f64 {
        if let Some(&b) = self.bounds.borrow().get(&unit.head) {
            return b;
        }
        let sources = &self.snapshot.sources;
        let average = sources.average_length();
        let mut highest = 0.0f64;
        self.snapshot.postings.impacts(unit.head, |s| sources.length(s), |f, l| {
            highest = highest.max(self.relevance.score(weight, f, l, average));
        });
        self.bounds.borrow_mut().insert(unit.head, highest);
        highest
    }

    /// Top-k for a disjunction of units by MaxScore (Turtle and Flood). Units ordered by the most any source can
    /// score for them; once the k-th best reaches the sum of the smallest bounds, those units can't put a source in
    /// the top alone: candidates come only from the others ("essential"), and the rest are advanced to each
    /// candidate only while it can still make it. Windows where the essential units' block maxima can't beat the
    /// threshold are skipped undecoded. None when a unit expands to several index units.
    fn search_disjunction(&self, units: &[&[u8]], top: usize) -> Option<Results> {
        struct Term<'a> {
            order: usize,
            weight: f64,
            bound: f64,
            cursor: Cursor<'a>,
            known: bool,
            highest: u32,
            shortest: u32,
        }
        let sources = &self.snapshot.sources;
        let average = sources.average_length();
        let n_sources = self.snapshot.source_count() as u64;
        let mut terms: Vec<Term> = Vec::new();
        for (order, unit) in units.iter().enumerate() {
            let parts = (self.expand)(self.snapshot, unit);
            if parts.len() > 1 {
                return None;
            }
            let Some(&part) = parts.first() else { continue };
            let weight = self.relevance.weight(part.count, n_sources);
            let bound = self.bound(part, weight);
            let mut cursor = self.snapshot.postings.cursor(part.head);
            cursor.next();
            terms.push(Term { order, weight, bound, cursor, known: false, highest: 0, shortest: 0 });
        }
        if terms.is_empty() {
            return Some(Results { count: 0, exact: false, top: Vec::new() });
        }
        terms.sort_by(|a, b| a.bound.partial_cmp(&b.bound).unwrap());
        let n = terms.len();
        let prefix: Vec<f64> = terms.iter().scan(0.0, |sum, t| {
            *sum += t.bound;
            Some(*sum)
        }).collect();

        let bm25: Option<&Bm25> = self.relevance.as_bm25();
        let mut best = TopK::new(top);
        let mut essential = 0;
        let mut contributions = vec![0.0f64; units.len()];
        let mut frequencies = vec![0u32; units.len()];
        let mut orders = Vec::with_capacity(units.len());
        let mut scored = 0u64;
        let mut window_end = 0u32;
        loop {
            let threshold = best.threshold();
            if best.full() {
                while essential < n && prefix[essential] <= threshold {
                    essential += 1;
                }
            }
            if essential == n {
                break;
            }
            let doc = terms[essential..].iter().map(|t| t.cursor.doc).min().unwrap();
            if doc == NO_MORE {
                break;
            }

            if best.full() && doc >= window_end {
                // a window to the nearest end of an essential unit's block: each can add at most its block's
                // maximum (or its overall bound where blocks aren't indexed, or nothing if its next entry is past
                // the window); with the others' bounds, if that can't beat the threshold, skip the window
                window_end = u32::MAX;
                for t in terms[essential..].iter_mut() {
                    match t.cursor.block_at(doc) {
                        Some((last, highest, shortest)) => {
                            t.known = true;
                            t.highest = highest;
                            t.shortest = shortest;
                            if highest > 0 {
                                window_end = window_end.min(last.saturating_add(1));
                            }
                        }
                        None => t.known = false,
                    }
                }
                if window_end == u32::MAX {
                    window_end = doc + 1;
                }
                let mut bound = if essential > 0 { prefix[essential - 1] } else { 0.0 };
                for t in &terms[essential..] {
                    if bound > threshold {
                        break;
                    }
                    if t.cursor.doc >= window_end || (t.known && t.highest == 0) {
                        continue;
                    }
                    bound += if t.known { t.bound.min(self.relevance.score(t.weight, t.highest, t.shortest, average)) } else { t.bound };
                }
                if bound <= threshold {
                    for t in terms[essential..].iter_mut() {
                        t.cursor.advance(window_end);
                    }
                    continue;
                }
            }

            let length = sources.length(doc);
            let norm = bm25.map(|b| b.norm(length, average)).unwrap_or(0.0);
            let score_of = |t: &Term, f: u32| match bm25 {
                Some(_) => t.weight * f as f64 / (f as f64 + norm),
                None => self.relevance.score(t.weight, f, length, average),
            };
            let mut upper = if essential > 0 { prefix[essential - 1] } else { 0.0 };
            orders.clear();
            for t in terms[essential..].iter_mut() {
                if t.cursor.doc != doc {
                    continue;
                }
                let f = t.cursor.frequency;
                let c = score_of(t, f);
                contributions[t.order] = c;
                frequencies[t.order] = f;
                orders.push(t.order);
                upper += c;
                t.cursor.next();
            }
            for i in (0..essential).rev() {
                if best.full() && upper <= threshold {
                    break;
                }
                let t = &mut terms[i];
                upper -= t.bound;
                if !t.cursor.advance(doc) || t.cursor.doc != doc {
                    continue;
                }
                let f = t.cursor.frequency;
                let c = score_of(t, f);
                contributions[t.order] = c;
                frequencies[t.order] = f;
                orders.push(t.order);
                upper += c;
            }
            // the exact score: summed in unit order, as search_all adds them
            orders.sort_unstable();
            let mut score = 0.0;
            let mut hits = 0;
            for &u in &orders {
                score += contributions[u];
                hits += frequencies[u];
            }
            scored += 1;
            best.offer(Hit { source: doc, score, hits });
        }
        Some(Results { count: scored, exact: false, top: best.into_sorted() })
    }
}

impl<'q, 's> Query<'q, 's> {
    fn new(searcher: &'q Searcher<'s>) -> Self {
        Query { searcher, units: HashMap::new(), phrase_frequencies: HashMap::new() }
    }

    fn data(&mut self, unit: &[u8]) -> &mut UnitData {
        if !self.units.contains_key(unit) {
            let parts = (self.searcher.expand)(self.searcher.snapshot, unit);
            let estimate = parts.iter().map(|p| p.count).sum();
            self.units.insert(unit.to_vec(), UnitData { parts, estimate, entries: None });
        }
        self.units.get_mut(unit).unwrap()
    }

    /// The unit's entries (source, frequency, positions offset), merged across its parts by source.
    fn entries(&mut self, unit: &[u8]) -> &Vec<(u32, u32, u64)> {
        let postings = self.searcher.snapshot.postings.clone();
        let data = self.data(unit);
        if data.entries.is_none() {
            let mut entries: Vec<(u32, u32, u64)> = Vec::new();
            for part in &data.parts {
                entries.extend(postings.decode(part.head));
            }
            if data.parts.len() > 1 {
                entries.sort_by_key(|e| e.0);
                entries.dedup_by(|b, a| {
                    if a.0 == b.0 {
                        a.1 += b.1;
                        true
                    } else {
                        false
                    }
                });
            }
            data.entries = Some(entries);
        }
        data.entries.as_ref().unwrap()
    }

    fn set(&mut self, unit: &[u8]) -> SourceSet {
        let mut set = SourceSet::new(self.searcher.snapshot.source_count());
        for e in self.entries(unit).iter() {
            set.insert(e.0);
        }
        set
    }

    fn count(&mut self, unit: &[u8]) -> u64 {
        if self.data(unit).parts.len() <= 1 {
            return self.data(unit).estimate;
        }
        self.entries(unit).len() as u64
    }

    fn evaluate(&mut self, clause: &Clause) -> SourceSet {
        let n = self.searcher.snapshot.source_count();
        let mut result: SourceSet;
        if !clause.must.is_empty() || !clause.must_clauses.is_empty() {
            // smallest first, and stop once nothing can match
            let mut musts: Vec<&Vec<u8>> = clause.must.iter().collect();
            musts.sort_by_key(|u| self.data(u).estimate);
            if musts.first().is_some_and(|u| self.data(u).estimate == 0) {
                return SourceSet::new(n);
            }
            let mut acc: Option<SourceSet> = None;
            for unit in musts {
                let set = self.set(unit);
                match &mut acc {
                    None => acc = Some(set),
                    Some(a) => a.and(&set),
                }
                if acc.as_ref().unwrap().is_empty() {
                    return acc.unwrap();
                }
            }
            for sub in &clause.must_clauses {
                let set = self.evaluate(sub);
                match &mut acc {
                    None => acc = Some(set),
                    Some(a) => a.and(&set),
                }
                if acc.as_ref().unwrap().is_empty() {
                    return acc.unwrap();
                }
            }
            result = acc.unwrap();
        } else if !clause.should.is_empty() || !clause.should_clauses.is_empty() {
            result = SourceSet::new(n);
            for unit in &clause.should {
                let set = self.set(unit);
                result.or(&set);
            }
            for sub in &clause.should_clauses {
                let set = self.evaluate(sub);
                result.or(&set);
            }
        } else {
            result = SourceSet::all(n);
        }
        for unit in &clause.must_not {
            let set = self.set(unit);
            result.and_not(&set);
        }
        for sub in &clause.must_not_clauses {
            let set = self.evaluate(sub);
            result.and_not(&set);
        }
        if clause.phrase.len() > 1 {
            let mut frequencies = HashMap::new();
            let candidates: Vec<u32> = result.iter().collect();
            for source in candidates {
                let f = self.phrase_frequency(&clause.phrase, source);
                if f == 0 {
                    result.remove(source);
                } else {
                    frequencies.insert(source, f);
                }
            }
            self.phrase_frequencies.insert(clause as *const Clause, frequencies);
        }
        result
    }

    /// Occurrences of the phrase's units at consecutive positions in a source.
    fn phrase_frequency(&mut self, phrase: &[Vec<u8>], source: u32) -> u32 {
        let mut starts: Option<Vec<u32>> = None;
        let mut positions = Vec::new();
        for (i, unit) in phrase.iter().enumerate() {
            let entries = self.entries(unit);
            let Ok(at) = entries.binary_search_by_key(&source, |e| e.0) else { return 0 };
            let (_, frequency, offset) = entries[at];
            self.searcher.snapshot.positions(offset, frequency, &mut positions);
            let shifted: Vec<u32> = positions.iter().filter(|&&p| p >= i as u32).map(|&p| p - i as u32).collect();
            starts = Some(match starts {
                None => shifted,
                Some(s) => s.into_iter().filter(|p| shifted.binary_search(p).is_ok()).collect(),
            });
            if starts.as_ref().unwrap().is_empty() {
                return 0;
            }
        }
        starts.map_or(0, |s| s.len() as u32)
    }

    fn positive_units(clause: &Clause, out: &mut Vec<Vec<u8>>) {
        if clause.phrase.len() <= 1 {
            out.extend(clause.must.iter().chain(&clause.should).cloned());
        }
        for sub in clause.must_clauses.iter().chain(&clause.should_clauses) {
            Self::positive_units(sub, out);
        }
    }

    fn positive_phrases<'c>(clause: &'c Clause, out: &mut Vec<&'c Clause>) {
        if clause.phrase.len() > 1 {
            out.push(clause);
        }
        for sub in clause.must_clauses.iter().chain(&clause.should_clauses) {
            Self::positive_phrases(sub, out);
        }
    }

    fn score(&mut self, clause: &Clause, matches: &SourceSet, scores: &mut [f64], hits: &mut [u32]) {
        let relevance = self.searcher.relevance;
        let sources = &self.searcher.snapshot.sources;
        let average = sources.average_length();
        let n = self.searcher.snapshot.source_count() as u64;
        let mut units = Vec::new();
        Self::positive_units(clause, &mut units);
        let mut seen = Vec::new();
        for unit in units {
            if seen.contains(&unit) {
                continue;
            }
            let count = self.count(&unit);
            if count == 0 {
                seen.push(unit);
                continue;
            }
            let weight = relevance.weight(count, n);
            for &(source, frequency, _) in self.entries(&unit).iter() {
                if matches.contains(source) {
                    scores[source as usize] += relevance.score(weight, frequency, sources.length(source), average);
                    hits[source as usize] += frequency;
                }
            }
            seen.push(unit);
        }
        let mut phrases = Vec::new();
        Self::positive_phrases(clause, &mut phrases);
        for phrase in phrases {
            let Some(frequencies) = self.phrase_frequencies.get(&(phrase as *const Clause)).cloned() else { continue };
            let weight: f64 = phrase.phrase.iter().map(|u| {
                let c = self.count(u);
                relevance.weight(c, n)
            }).sum();
            for (source, f) in frequencies {
                if matches.contains(source) {
                    scores[source as usize] += relevance.score(weight, f, sources.length(source), average);
                    hits[source as usize] += f;
                }
            }
        }
    }
}
