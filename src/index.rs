//! An index of sources keyed by `K` inverted into units of type `U`: a dictionary from unit to its postings (a
//! b-positive tree, keyed by the units' order-preserving encoding, so the dictionary's order is the units' order),
//! the postings and positions, and the sources. One writer adds sources and commits; any number of snapshots read.
//!
//! Storage: the dictionary is a b-positive tree; postings, positions and sources are append-only blobs in a larry
//! store. A commit appends to the blobs first and commits the dictionary last, recording the blobs' committed
//! lengths, so a crash anywhere leaves the previous commit (and opening drops whatever lies past it).

use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::{Bound, RangeBounds};
use std::io;
use std::path::Path;
use std::sync::{Mutex, RwLock};

use b_positive::{Db, Durability, ReadTxn};
use larry::{Blob, Store, View};

use crate::codec::Codec;
use crate::inverter::Inverter;
use crate::postings::{self, Buffer, Postings, NO_RECORD};
use crate::sources::Sources;
use crate::varint;

/// The dictionary's key for what the last commit recorded; every unit's key is UNIT_TAG then its encoding.
const META_KEY: &[u8] = &[0];
const UNIT_TAG: u8 = 1;

#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Record how often a unit occurs in a source (else every frequency reads 1).
    pub frequencies: bool,
    /// Record units' positions in sources (for phrases).
    pub positions: bool,
    pub durability: Durability,
}

impl Default for Options {
    fn default() -> Self {
        Options { frequencies: true, positions: true, durability: Durability::Process }
    }
}

#[derive(Clone, Copy, Default)]
struct Meta {
    sources: u32,
    postings_length: u64,
    positions_length: u64,
    sources_length: u64,
}

impl Meta {
    fn encode(&self) -> [u8; 28] {
        let mut out = [0u8; 28];
        out[0..4].copy_from_slice(&self.sources.to_le_bytes());
        out[4..12].copy_from_slice(&self.postings_length.to_le_bytes());
        out[12..20].copy_from_slice(&self.positions_length.to_le_bytes());
        out[20..28].copy_from_slice(&self.sources_length.to_le_bytes());
        out
    }

    fn decode(bytes: &[u8]) -> Meta {
        let u64_at = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
        Meta { sources: u32::from_le_bytes(bytes[0..4].try_into().unwrap()), postings_length: u64_at(4), positions_length: u64_at(12), sources_length: u64_at(20) }
    }
}

struct Writing {
    buffers: HashMap<Vec<u8>, Buffer>,
    buffered: usize,
    positions: Vec<u8>,       // positions written since the last commit
    sources_bytes: Vec<u8>,   // source records written since the last commit
    sources: Sources,
    meta: Meta,
}

struct Published {
    postings: View,
    positions: Option<View>,
    sources: Sources,
}

/// An inverted index of sources keyed by `K` over units `U` (any types with an order-preserving [`Codec`]).
pub struct Index<K, U> {
    dict: Db,
    postings_blob: Blob,
    positions_blob: Option<Blob>,
    sources_blob: Blob,
    options: Options,
    writing: Mutex<Writing>,
    published: RwLock<Published>,
    types: PhantomData<fn() -> (K, U)>,
}

impl<K: Codec, U: Codec> Index<K, U> {
    /// Opens (or creates) an index in a directory of its own.
    pub fn open(directory: impl AsRef<Path>, options: Options) -> io::Result<Self> {
        Self::open_in(&Store::open(directory)?, options)
    }

    /// Opens (or creates) an index in a store of its own.
    pub fn open_in(store: &Store, options: Options) -> io::Result<Self> {
        let dict = Db::open(store.path("units.db")?, b_positive::Options { durability: options.durability })?;
        let postings_blob = store.blob("postings.dat")?;
        let meta = dict.begin_read().get(META_KEY).map(|m| Meta::decode(&m)).unwrap_or_default();

        // a new index records its options in the postings header; an existing one keeps what it was created with
        let mut options = options;
        if meta.postings_length == 0 {
            postings_blob.truncate(0)?;
            postings_blob.append(&postings::blob_header(options.frequencies, options.positions))?;
        } else {
            let mut header = [0u8; postings::HEADER];
            postings_blob.read_at(0, &mut header)?;
            if &header[0..4] != b"IVPT" || header[4] != postings::VERSION {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "not an introvert postings blob (or another version)"));
            }
            options.frequencies = header[5] & postings::FLAG_FREQUENCIES != 0;
            options.positions = header[5] & postings::FLAG_POSITIONS != 0;
            // anything past the last commit never committed (a crash): drop it
            postings_blob.truncate(meta.postings_length)?;
        }
        let postings_length = postings_blob.len();
        let positions_blob = if options.positions {
            let blob = store.blob("positions.dat")?;
            blob.truncate(meta.positions_length)?;
            Some(blob)
        } else {
            None
        };
        let sources_blob = store.blob("sources.dat")?;
        sources_blob.truncate(meta.sources_length)?;

        // the sources, from their blob
        let mut sources = Sources::default();
        let bytes = sources_blob.read_all()?;
        let mut at = 0;
        while at < bytes.len() {
            let key_length = varint::read(&bytes, &mut at) as usize;
            let key = &bytes[at..at + key_length];
            at += key_length;
            let length = varint::read(&bytes, &mut at) as u32;
            sources.push(key, length);
        }
        sources.truncate(meta.sources);

        let meta = Meta { postings_length, positions_length: meta.positions_length, sources_length: meta.sources_length, sources: sources.count() };
        let published = Published { postings: postings_blob.view()?, positions: positions_blob.as_ref().map(Blob::view).transpose()?, sources: sources.clone() };
        Ok(Index {
            dict,
            postings_blob,
            positions_blob,
            sources_blob,
            options,
            writing: Mutex::new(Writing { buffers: HashMap::new(), buffered: 0, positions: Vec::new(), sources_bytes: Vec::new(), sources, meta }),
            published: RwLock::new(published),
            types: PhantomData,
        })
    }

    pub fn options(&self) -> Options {
        self.options
    }

    /// Inverts a source with an inverter and adds it under its key (see [`Index::add`]).
    pub fn invert<S: ?Sized, I: Inverter<S, Unit = U>>(&self, key: &K, source: &S, inverter: &I) -> u32 {
        let mut units = Vec::new();
        inverter.invert(source, |u, p| units.push((u, p)));
        self.add(key, units)
    }

    /// Adds a source: its key and its units, with their positions. Returns the source's ordinal. Visible to snapshots taken
    /// after the next [`Index::commit`].
    pub fn add(&self, key: &K, units: impl IntoIterator<Item = (U, u32)>) -> u32 {
        // every unit's dictionary key in one buffer, then grouped by key
        let mut encoded = Vec::new();
        let mut spans = Vec::new();
        for (unit, position) in units {
            let start = encoded.len();
            encoded.push(UNIT_TAG);
            unit.encode(&mut encoded);
            spans.push((start, encoded.len(), position));
        }
        let mut grouped: HashMap<&[u8], Vec<u32>> = HashMap::with_capacity(spans.len());
        for &(start, end, position) in &spans {
            grouped.entry(&encoded[start..end]).or_default().push(position);
        }
        let length = spans.len() as u32;
        let key = key.to_bytes();
        let key = key.as_slice();
        let mut w = self.writing.lock().unwrap();
        let ordinal = w.sources.push(key, length);
        varint::write(&mut w.sources_bytes, key.len() as u64);
        w.sources_bytes.extend_from_slice(key);
        varint::write(&mut w.sources_bytes, length as u64);

        for (unit, mut positions) in grouped {
            positions.sort_unstable();
            let mut offset = 0;
            if self.options.positions {
                offset = w.meta.positions_length + w.positions.len() as u64;
                let mut previous = 0;
                for &p in &positions {
                    varint::write(&mut w.positions, (p - previous) as u64);
                    previous = p;
                }
            }
            let frequency = if self.options.frequencies { positions.len() as u32 } else { 1 };
            let buffer = match w.buffers.get_mut(unit) {
                Some(b) => b,
                None => w.buffers.entry(unit.to_vec()).or_default(),
            };
            buffer.push(ordinal, frequency, offset, length);
            w.buffered += 16;
        }
        ordinal
    }

    /// Bytes buffered since the last commit (postings, positions and sources).
    pub fn buffered_bytes(&self) -> usize {
        let w = self.writing.lock().unwrap();
        w.buffered + w.positions.len() + w.sources_bytes.len()
    }

    /// Makes everything added since the last commit durable and visible to new snapshots. The blobs are appended
    /// first and the dictionary (with the lengths that are now valid) commits last: a crash anywhere leaves the
    /// previous commit.
    pub fn commit(&self) -> io::Result<()> {
        let mut w = self.writing.lock().unwrap();
        let mut units: Vec<(Vec<u8>, Buffer)> = w.buffers.drain().collect();
        units.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let mut txn = self.dict.begin_write();
        let mut records = Vec::with_capacity(units.iter().map(|(_, b)| b.sources.len() * 3).sum::<usize>());
        for (unit, buffer) in &units {
            let (previous, count) = match txn.get(unit) {
                Some(v) => (u64::from_le_bytes(v[0..8].try_into().unwrap()), u64::from_le_bytes(v[8..16].try_into().unwrap())),
                None => (NO_RECORD, 0),
            };
            let head = w.meta.postings_length + records.len() as u64;
            postings::encode_record(previous, buffer, self.options.frequencies, self.options.positions, &mut records);
            let mut value = [0u8; 16];
            value[0..8].copy_from_slice(&head.to_le_bytes());
            value[8..16].copy_from_slice(&(count + buffer.sources.len() as u64).to_le_bytes());
            txn.put(unit, &value)?;
        }

        debug_assert_eq!(self.postings_blob.len(), w.meta.postings_length);
        self.postings_blob.append(&records)?;
        w.meta.postings_length += records.len() as u64;
        if let Some(blob) = &self.positions_blob {
            blob.append(&w.positions)?;
            w.meta.positions_length += w.positions.len() as u64;
        }
        self.sources_blob.append(&w.sources_bytes)?;
        w.meta.sources_length += w.sources_bytes.len() as u64;
        if self.options.durability == Durability::Full {
            self.postings_blob.sync()?;
            if let Some(blob) = &self.positions_blob {
                blob.sync()?;
            }
            self.sources_blob.sync()?;
        }
        w.meta.sources = w.sources.count();
        txn.put(META_KEY, &w.meta.encode())?;
        txn.commit()?;

        w.positions.clear();
        w.sources_bytes.clear();
        w.buffered = 0;
        let published = Published {
            postings: self.postings_blob.view()?,
            positions: self.positions_blob.as_ref().map(Blob::view).transpose()?,
            sources: w.sources.clone(),
        };
        *self.published.write().unwrap() = published;
        Ok(())
    }

    /// What the last commit holds; cheap, and usable from any thread.
    pub fn snapshot(&self) -> Snapshot<K, U> {
        let published = self.published.read().unwrap();
        Snapshot {
            dict: self.dict.begin_read(),
            postings: Postings { map: published.postings.clone(), frequencies: self.options.frequencies, positions: self.options.positions },
            positions: published.positions.clone(),
            sources: published.sources.clone(),
            types: PhantomData,
        }
    }
}

/// A unit's postings: where its chain starts, and how many sources it lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitPostings {
    pub head: u64,
    pub count: u64,
}

impl UnitPostings {
    fn decode(value: &[u8]) -> UnitPostings {
        UnitPostings { head: u64::from_le_bytes(value[0..8].try_into().unwrap()), count: u64::from_le_bytes(value[8..16].try_into().unwrap()) }
    }
}

/// The index as of a commit.
pub struct Snapshot<K, U> {
    dict: ReadTxn,
    pub postings: Postings,
    positions: Option<View>,
    pub(crate) sources: Sources,
    types: PhantomData<fn() -> (K, U)>,
}

impl<K: Codec, U: Codec> Snapshot<K, U> {
    pub fn source_count(&self) -> u32 {
        self.sources.count()
    }

    /// A source's key.
    pub fn key(&self, source: u32) -> K {
        K::decode(self.sources.key(source))
    }

    /// A source's length in units.
    pub fn source_length(&self, source: u32) -> u32 {
        self.sources.length(source)
    }

    pub fn average_length(&self) -> f64 {
        self.sources.average_length()
    }

    pub fn unit(&self, unit: &U) -> Option<UnitPostings> {
        let mut key = vec![UNIT_TAG];
        unit.encode(&mut key);
        self.dict.get(&key).map(|v| UnitPostings::decode(&v))
    }

    /// Units in a range, in order, with their postings.
    pub fn units(&self, range: impl RangeBounds<U>) -> impl Iterator<Item = (U, UnitPostings)> + '_ {
        self.encoded_units(range).map(|(k, p)| (U::decode(&k[1..]), p))
    }

    pub(crate) fn encoded_units(&self, range: impl RangeBounds<U>) -> impl Iterator<Item = (&[u8], UnitPostings)> + '_ {
        let key = |u: &U| {
            let mut key = vec![UNIT_TAG];
            u.encode(&mut key);
            key
        };
        let (start, skip) = match range.start_bound() {
            Bound::Included(u) => (key(u), None),
            Bound::Excluded(u) => (key(u), Some(key(u))),
            Bound::Unbounded => (vec![UNIT_TAG], None),
        };
        let end = match range.end_bound() {
            Bound::Included(u) => Bound::Included(key(u)),
            Bound::Excluded(u) => Bound::Excluded(key(u)),
            Bound::Unbounded => Bound::Unbounded,
        };
        self.encoded_range(start, end).filter(move |(k, _)| skip.as_deref() != Some(*k))
    }

    /// Units whose encoding begins with `prefix`'s: for strings, those starting with it; for a tuple, those with the
    /// same leading elements and a last element beginning with `prefix`'s.
    pub fn units_with_prefix(&self, prefix: &U) -> impl Iterator<Item = (U, UnitPostings)> + '_ {
        self.encoded_prefix(prefix).map(|(k, p)| (U::decode(&k[1..]), p))
    }

    pub(crate) fn encoded_prefix(&self, prefix: &U) -> impl Iterator<Item = (&[u8], UnitPostings)> + '_ {
        let mut start = vec![UNIT_TAG];
        prefix.encode(&mut start);
        let bound = start.clone();
        self.dict.range_from(&start).take_while(move |(k, _)| k.starts_with(&bound)).map(|(k, v)| (k, UnitPostings::decode(&v)))
    }

    pub(crate) fn encoded_range(&self, start: Vec<u8>, end: Bound<Vec<u8>>) -> impl Iterator<Item = (&[u8], UnitPostings)> + '_ {
        self.dict
            .range_from(&start)
            .take_while(move |(k, _)| match &end {
                Bound::Included(e) => *k <= e.as_slice(),
                Bound::Excluded(e) => *k < e.as_slice(),
                Bound::Unbounded => k.first() == Some(&UNIT_TAG),
            })
            .map(|(k, v)| (k, UnitPostings::decode(&v)))
    }

    /// Decodes the `frequency` positions stored at `offset`.
    pub fn positions(&self, offset: u64, frequency: u32, out: &mut Vec<u32>) {
        out.clear();
        let Some(map) = &self.positions else { return };
        let mut at = offset as usize;
        let mut position = 0u32;
        for _ in 0..frequency {
            position += varint::read(map, &mut at) as u32;
            out.push(position);
        }
    }
}
