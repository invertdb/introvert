//! Queries: boolean clauses over units, matched and ranked.
//!
//! A [`Clause`] requires (must), prefers (should) or excludes (must not) terms and sub-clauses, and may be a phrase.
//! A [`Term`] is a unit, or every unit with a prefix or in a range (units are ordered by their encoding, which
//! preserves their order). [`Searcher::search`] ranks the top k: a disjunction of single units takes the pruned
//! (MaxScore) path, anything else scores every match. Both return the same results, summed in the same order.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::ops::Bound;

use crate::codec::Codec;
use crate::index::{Snapshot, UnitPostings};
use crate::postings::{Cursor, NO_MORE};
use crate::relevance::Relevance;
use crate::sourceset::SourceSet;

/// What a query matches units by.
#[derive(Clone, Debug, PartialEq)]
pub enum Term<U> {
    /// This unit.
    Unit(U),
    /// Every unit whose encoding begins with this one's (see [`Snapshot::units_with_prefix`]).
    Prefix(U),
    /// Every unit in the range.
    Range(Bound<U>, Bound<U>),
}

impl<U> From<U> for Term<U> {
    fn from(unit: U) -> Self {
        Term::Unit(unit)
    }
}

impl<U: Codec> Term<U> {
    /// Identifies the term (by its kind and encodings), for caching what it matches.
    fn key(&self) -> Vec<u8> {
        let mut key = Vec::new();
        let bound = |key: &mut Vec<u8>, b: &Bound<U>| match b {
            Bound::Included(u) => {
                key.push(0);
                u.encode_nested(key);
            }
            Bound::Excluded(u) => {
                key.push(1);
                u.encode_nested(key);
            }
            Bound::Unbounded => key.push(2),
        };
        match self {
            Term::Unit(u) => {
                key.push(0);
                u.encode(&mut key);
            }
            Term::Prefix(u) => {
                key.push(1);
                u.encode(&mut key);
            }
            Term::Range(start, end) => {
                key.push(2);
                bound(&mut key, start);
                bound(&mut key, end);
            }
        }
        key
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Clause<U> {
    pub must: Vec<Term<U>>,
    pub should: Vec<Term<U>>,
    pub must_not: Vec<Term<U>>,
    /// Units that must occur at consecutive positions (with two or more).
    pub phrase: Vec<U>,
    pub must_clauses: Vec<Clause<U>>,
    pub should_clauses: Vec<Clause<U>>,
    pub must_not_clauses: Vec<Clause<U>>,
}

impl<U> Default for Clause<U> {
    fn default() -> Self {
        Clause {
            must: Vec::new(),
            should: Vec::new(),
            must_not: Vec::new(),
            phrase: Vec::new(),
            must_clauses: Vec::new(),
            should_clauses: Vec::new(),
            must_not_clauses: Vec::new(),
        }
    }
}

impl<U: Clone> Clause<U> {
    pub fn should(terms: impl IntoIterator<Item = impl Into<Term<U>>>) -> Self {
        Clause { should: terms.into_iter().map(Into::into).collect(), ..Default::default() }
    }

    pub fn must(terms: impl IntoIterator<Item = impl Into<Term<U>>>) -> Self {
        Clause { must: terms.into_iter().map(Into::into).collect(), ..Default::default() }
    }

    pub fn phrase(units: impl IntoIterator<Item = U>) -> Self {
        let phrase: Vec<U> = units.into_iter().collect();
        Clause { must: phrase.iter().cloned().map(Term::Unit).collect(), phrase, ..Default::default() }
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

#[derive(Clone, Debug, PartialEq)]
pub struct Hit<S> {
    /// The source's ordinal in the index.
    pub ordinal: u32,
    /// The source's key.
    pub source: S,
    pub score: f64,
    /// Occurrences of the query's positive units and phrases in the source.
    pub hits: u32,
}

#[derive(Clone, Debug)]
pub struct Results<S> {
    /// Matching sources: exact, or (when `exact` is false) the ones scored, a lower bound.
    pub count: u64,
    pub exact: bool,
    pub top: Vec<Hit<S>>,
}

pub const DEFAULT_MAX_EXPANSION: usize = 1024;

/// Searches a snapshot of an index of sources `S` over units `U`, ranking by relevance `R`.
pub struct Searcher<'s, S, U, R> {
    snapshot: &'s Snapshot<S, U>,
    relevance: &'s R,
    max_expansion: usize,
    bounds: RefCell<HashMap<u64, f64>>,
}

/// A scored source, before its key is decoded.
#[derive(Clone, Copy)]
struct Scored {
    ordinal: u32,
    score: f64,
    hits: u32,
}

/// A heap entry, ordered so the heap's top is the worst kept: lowest score, then highest ordinal.
#[derive(Clone, Copy)]
struct Kept(Scored);

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
        // BinaryHeap is a max-heap: "greater" = worse = lower score, then higher ordinal
        other.0.score.partial_cmp(&self.0.score).unwrap_or(Ordering::Equal).then(self.0.ordinal.cmp(&other.0.ordinal))
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

    fn offer(&mut self, scored: Scored) {
        if self.heap.len() < self.top {
            self.heap.push(Kept(scored));
        } else if self.top > 0 && scored.score > self.heap.peek().unwrap().0.score {
            self.heap.pop();
            self.heap.push(Kept(scored));
        }
    }

    fn into_hits<S: Codec, U: Codec>(self, snapshot: &Snapshot<S, U>) -> Vec<Hit<S>> {
        let mut scored: Vec<Scored> = self.heap.into_iter().map(|k| k.0).collect();
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal).then(a.ordinal.cmp(&b.ordinal)));
        scored.into_iter().map(|s| Hit { ordinal: s.ordinal, source: snapshot.source(s.ordinal), score: s.score, hits: s.hits }).collect()
    }
}

/// A term's postings, merged across the units it expands to, decoded on first use.
struct TermData {
    parts: Vec<UnitPostings>,
    estimate: u64,
    entries: Option<Vec<(u32, u32, u64)>>,
}

struct Query<'q, 's, S, U, R> {
    searcher: &'q Searcher<'s, S, U, R>,
    terms: HashMap<Vec<u8>, TermData>,
    phrase_frequencies: HashMap<*const Clause<U>, HashMap<u32, u32>>,
}

impl<'s, S: Codec, U: Codec + Clone, R: Relevance> Searcher<'s, S, U, R> {
    pub fn new(snapshot: &'s Snapshot<S, U>, relevance: &'s R) -> Self {
        Searcher { snapshot, relevance, max_expansion: DEFAULT_MAX_EXPANSION, bounds: RefCell::new(HashMap::new()) }
    }

    /// The most units a prefix or range term expands to (the first in order).
    pub fn with_max_expansion(mut self, max_expansion: usize) -> Self {
        self.max_expansion = max_expansion;
        self
    }

    pub fn snapshot(&self) -> &Snapshot<S, U> {
        self.snapshot
    }

    /// The units a term matches, with their postings.
    pub fn expand(&self, term: &Term<U>) -> Vec<UnitPostings> {
        match term {
            Term::Unit(u) => self.snapshot.unit(u).into_iter().collect(),
            Term::Prefix(u) => self.snapshot.encoded_prefix(u).take(self.max_expansion).map(|(_, p)| p).collect(),
            Term::Range(start, end) => self.snapshot.encoded_units((start.as_ref(), end.as_ref())).take(self.max_expansion).map(|(_, p)| p).collect(),
        }
    }

    /// The sources matching a clause, unranked.
    pub fn matching(&self, clause: &Clause<U>) -> SourceSet {
        Query::new(self).evaluate(clause)
    }

    /// The best `top` sources for a clause, best first.
    pub fn search(&self, clause: &Clause<U>, top: usize) -> Results<S> {
        if top > 0 && clause.is_plain_disjunction() {
            let mut keys: Vec<Vec<u8>> = Vec::new();
            let mut terms: Vec<&Term<U>> = Vec::new();
            for term in &clause.should {
                let key = term.key();
                if !keys.contains(&key) {
                    keys.push(key);
                    terms.push(term);
                }
            }
            if let Some(results) = self.search_disjunction(&terms, top) {
                return results;
            }
        }
        self.search_all(clause, top)
    }

    /// Scores every matching source; counts them exactly.
    pub fn search_all(&self, clause: &Clause<U>, top: usize) -> Results<S> {
        let mut query = Query::new(self);
        let matches = query.evaluate(clause);
        let n = self.snapshot.source_count() as usize;
        let mut scores = vec![0.0f64; n];
        let mut hits = vec![0u32; n];
        query.score(clause, &matches, &mut scores, &mut hits);
        let mut best = TopK::new(top);
        let mut count = 0;
        for ordinal in matches.iter() {
            count += 1;
            best.offer(Scored { ordinal, score: scores[ordinal as usize], hits: hits[ordinal as usize] });
        }
        Results { count, exact: true, top: best.into_hits(self.snapshot) }
    }

    fn bound(&self, unit: UnitPostings, weight: f64) -> f64 {
        if let Some(&b) = self.bounds.borrow().get(&unit.head) {
            return b;
        }
        let snapshot = self.snapshot;
        let average = snapshot.average_length();
        let mut highest = 0.0f64;
        snapshot.postings.impacts(unit.head, |s| snapshot.source_length(s), |f, l| {
            highest = highest.max(self.relevance.score(weight, f, self.relevance.norm(l, average)));
        });
        self.bounds.borrow_mut().insert(unit.head, highest);
        highest
    }

    /// Top-k for a disjunction of terms by MaxScore (Turtle and Flood). Terms ordered by the most any source can
    /// score for them; once the k-th best reaches the sum of the smallest bounds, those terms can't put a source in
    /// the top alone: candidates come only from the others ("essential"), and the rest are advanced to each
    /// candidate only while it can still make it. Windows where the essential terms' block maxima can't beat the
    /// threshold are skipped undecoded. None when a term expands to several units.
    fn search_disjunction(&self, query_terms: &[&Term<U>], top: usize) -> Option<Results<S>> {
        struct Cursored<'a> {
            order: usize,
            weight: f64,
            bound: f64,
            cursor: Cursor<'a>,
            known: bool,
            highest: u32,
            shortest: u32,
        }
        let snapshot = self.snapshot;
        let relevance = self.relevance;
        let average = snapshot.average_length();
        let n_sources = snapshot.source_count() as u64;
        let mut terms: Vec<Cursored> = Vec::new();
        for (order, term) in query_terms.iter().enumerate() {
            let parts = self.expand(term);
            if parts.len() > 1 {
                return None;
            }
            let Some(&part) = parts.first() else { continue };
            let weight = relevance.weight(part.count, n_sources);
            let bound = self.bound(part, weight);
            let mut cursor = snapshot.postings.cursor(part.head);
            cursor.next();
            terms.push(Cursored { order, weight, bound, cursor, known: false, highest: 0, shortest: 0 });
        }
        if terms.is_empty() {
            return Some(Results { count: 0, exact: false, top: Vec::new() });
        }
        terms.sort_by(|a, b| a.bound.partial_cmp(&b.bound).unwrap());
        let n = terms.len();
        let prefix: Vec<f64> = terms
            .iter()
            .scan(0.0, |sum, t| {
                *sum += t.bound;
                Some(*sum)
            })
            .collect();

        let mut best = TopK::new(top);
        let mut essential = 0;
        let mut contributions = vec![0.0f64; query_terms.len()];
        let mut frequencies = vec![0u32; query_terms.len()];
        let mut orders = Vec::with_capacity(query_terms.len());
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
                // a window to the nearest end of an essential term's block: each can add at most its block's
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
                    bound += if t.known { t.bound.min(relevance.score(t.weight, t.highest, relevance.norm(t.shortest, average))) } else { t.bound };
                }
                if bound <= threshold {
                    for t in terms[essential..].iter_mut() {
                        t.cursor.advance(window_end);
                    }
                    continue;
                }
            }

            let norm = relevance.norm(snapshot.source_length(doc), average);
            let mut upper = if essential > 0 { prefix[essential - 1] } else { 0.0 };
            orders.clear();
            for t in terms[essential..].iter_mut() {
                if t.cursor.doc != doc {
                    continue;
                }
                let f = t.cursor.frequency;
                let c = relevance.score(t.weight, f, norm);
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
                let c = relevance.score(t.weight, f, norm);
                contributions[t.order] = c;
                frequencies[t.order] = f;
                orders.push(t.order);
                upper += c;
            }
            // the exact score: summed in term order, as search_all adds them
            orders.sort_unstable();
            let mut score = 0.0;
            let mut hits = 0;
            for &u in &orders {
                score += contributions[u];
                hits += frequencies[u];
            }
            scored += 1;
            best.offer(Scored { ordinal: doc, score, hits });
        }
        Some(Results { count: scored, exact: false, top: best.into_hits(snapshot) })
    }
}

impl<'q, 's, S: Codec, U: Codec + Clone, R: Relevance> Query<'q, 's, S, U, R> {
    fn new(searcher: &'q Searcher<'s, S, U, R>) -> Self {
        Query { searcher, terms: HashMap::new(), phrase_frequencies: HashMap::new() }
    }

    fn data(&mut self, term: &Term<U>) -> &mut TermData {
        let key = term.key();
        if !self.terms.contains_key(&key) {
            let parts = self.searcher.expand(term);
            let estimate = parts.iter().map(|p| p.count).sum();
            self.terms.insert(key.clone(), TermData { parts, estimate, entries: None });
        }
        self.terms.get_mut(&key).unwrap()
    }

    /// The term's entries (source, frequency, positions offset), merged across its units by source.
    fn entries(&mut self, term: &Term<U>) -> &Vec<(u32, u32, u64)> {
        let postings = self.searcher.snapshot.postings.clone();
        let data = self.data(term);
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

    fn set(&mut self, term: &Term<U>) -> SourceSet {
        let mut set = SourceSet::new(self.searcher.snapshot.source_count());
        for e in self.entries(term).iter() {
            set.insert(e.0);
        }
        set
    }

    fn count(&mut self, term: &Term<U>) -> u64 {
        if self.data(term).parts.len() <= 1 {
            return self.data(term).estimate;
        }
        self.entries(term).len() as u64
    }

    fn evaluate(&mut self, clause: &Clause<U>) -> SourceSet {
        let n = self.searcher.snapshot.source_count();
        let mut result: SourceSet;
        if !clause.must.is_empty() || !clause.must_clauses.is_empty() {
            // smallest first, and stop once nothing can match
            let mut musts: Vec<&Term<U>> = clause.must.iter().collect();
            musts.sort_by_key(|t| self.data(t).estimate);
            if musts.first().is_some_and(|t| self.data(t).estimate == 0) {
                return SourceSet::new(n);
            }
            let mut acc: Option<SourceSet> = None;
            for term in musts {
                let set = self.set(term);
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
            for term in &clause.should {
                let set = self.set(term);
                result.or(&set);
            }
            for sub in &clause.should_clauses {
                let set = self.evaluate(sub);
                result.or(&set);
            }
        } else {
            result = SourceSet::all(n);
        }
        for term in &clause.must_not {
            let set = self.set(term);
            result.and_not(&set);
        }
        for sub in &clause.must_not_clauses {
            let set = self.evaluate(sub);
            result.and_not(&set);
        }
        if clause.phrase.len() > 1 {
            let phrase: Vec<Term<U>> = clause.phrase.iter().cloned().map(Term::Unit).collect();
            let mut frequencies = HashMap::new();
            let candidates: Vec<u32> = result.iter().collect();
            for source in candidates {
                let f = self.phrase_frequency(&phrase, source);
                if f == 0 {
                    result.remove(source);
                } else {
                    frequencies.insert(source, f);
                }
            }
            self.phrase_frequencies.insert(clause as *const Clause<U>, frequencies);
        }
        result
    }

    /// Occurrences of the phrase's units at consecutive positions in a source.
    fn phrase_frequency(&mut self, phrase: &[Term<U>], source: u32) -> u32 {
        let mut starts: Option<Vec<u32>> = None;
        let mut positions = Vec::new();
        for (i, term) in phrase.iter().enumerate() {
            let entries = self.entries(term);
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

    fn positive_terms<'c>(clause: &'c Clause<U>, out: &mut Vec<&'c Term<U>>) {
        if clause.phrase.len() <= 1 {
            out.extend(clause.must.iter().chain(&clause.should));
        }
        for sub in clause.must_clauses.iter().chain(&clause.should_clauses) {
            Self::positive_terms(sub, out);
        }
    }

    fn positive_phrases<'c>(clause: &'c Clause<U>, out: &mut Vec<&'c Clause<U>>) {
        if clause.phrase.len() > 1 {
            out.push(clause);
        }
        for sub in clause.must_clauses.iter().chain(&clause.should_clauses) {
            Self::positive_phrases(sub, out);
        }
    }

    fn score(&mut self, clause: &Clause<U>, matches: &SourceSet, scores: &mut [f64], hits: &mut [u32]) {
        let relevance = self.searcher.relevance;
        let snapshot = self.searcher.snapshot;
        let average = snapshot.average_length();
        let n = snapshot.source_count() as u64;
        let mut terms = Vec::new();
        Self::positive_terms(clause, &mut terms);
        let mut seen: Vec<Vec<u8>> = Vec::new();
        for term in terms {
            let key = term.key();
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            let count = self.count(term);
            if count == 0 {
                continue;
            }
            let weight = relevance.weight(count, n);
            for &(source, frequency, _) in self.entries(term).iter() {
                if matches.contains(source) {
                    let norm = relevance.norm(snapshot.source_length(source), average);
                    scores[source as usize] += relevance.score(weight, frequency, norm);
                    hits[source as usize] += frequency;
                }
            }
        }
        let mut phrases = Vec::new();
        Self::positive_phrases(clause, &mut phrases);
        for phrase in phrases {
            let Some(frequencies) = self.phrase_frequencies.get(&(phrase as *const Clause<U>)).cloned() else { continue };
            let mut weight = 0.0;
            for unit in &phrase.phrase {
                let c = self.count(&Term::Unit(unit.clone()));
                weight += relevance.weight(c, n);
            }
            for (source, f) in frequencies {
                if matches.contains(source) {
                    let norm = relevance.norm(snapshot.source_length(source), average);
                    scores[source as usize] += relevance.score(weight, f, norm);
                    hits[source as usize] += f;
                }
            }
        }
    }
}
