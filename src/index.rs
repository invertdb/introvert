//! An index: a dictionary from unit to its postings (a b-positive tree), the postings and positions files, and the
//! sources. One writer adds sources and commits; any number of snapshots read.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use b_positive::{Db, Durability, ReadTxn};
use memmap2::Mmap;

use crate::postings::{self, Buffer, Postings, NO_RECORD};
use crate::sources::Sources;
use crate::varint;

/// The dictionary's reserved key (units are never empty): what the last commit recorded.
const META_KEY: &[u8] = b"";

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
    postings: Arc<Mmap>,
    positions: Option<Arc<Mmap>>,
    sources: Sources,
}

/// An inverted index of byte-string sources (keys) over byte-string units.
pub struct Index {
    dict: Db,
    postings_file: File,
    positions_file: Option<File>,
    sources_file: File,
    options: Options,
    writing: Mutex<Writing>,
    published: RwLock<Published>,
}

fn open_append(path: &PathBuf) -> io::Result<File> {
    OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path)
}

impl Index {
    pub fn open(directory: impl AsRef<Path>, options: Options) -> io::Result<Index> {
        let directory = directory.as_ref();
        std::fs::create_dir_all(directory)?;
        let dict = Db::open(directory.join("units.db"), b_positive::Options { durability: options.durability })?;
        let postings_file = open_append(&directory.join("postings.dat"))?;
        let meta = dict.begin_read().get(META_KEY).map(|m| Meta::decode(&m)).unwrap_or_default();

        // a new index records its options in the postings header; an existing one keeps what it was created with
        let mut options = options;
        if meta.postings_length == 0 {
            postings_file.set_len(0)?;
            postings_file.write_all_at(&postings::file_header(options.frequencies, options.positions), 0)?;
        } else {
            let mut header = [0u8; postings::HEADER];
            postings_file.read_exact_at(&mut header, 0)?;
            if &header[0..4] != b"IVPT" || header[4] != postings::VERSION {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "not an introvert postings file (or another version)"));
            }
            options.frequencies = header[5] & postings::FLAG_FREQUENCIES != 0;
            options.positions = header[5] & postings::FLAG_POSITIONS != 0;
            // anything past the last commit never committed (a crash): drop it
            postings_file.set_len(meta.postings_length)?;
        }
        let postings_length = postings_file.metadata()?.len();
        let positions_file = if options.positions {
            let file = open_append(&directory.join("positions.dat"))?;
            file.set_len(meta.positions_length)?;
            Some(file)
        } else {
            None
        };
        let sources_file = open_append(&directory.join("sources.dat"))?;
        sources_file.set_len(meta.sources_length)?;

        // the sources, from their file
        let mut sources = Sources::default();
        let mut bytes = vec![0u8; meta.sources_length as usize];
        sources_file.read_exact_at(&mut bytes, 0)?;
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
        let published = Published { postings: map(&postings_file)?, positions: positions_file.as_ref().map(map).transpose()?, sources: sources.clone() };
        Ok(Index {
            dict,
            postings_file,
            positions_file,
            sources_file,
            options,
            writing: Mutex::new(Writing { buffers: HashMap::new(), buffered: 0, positions: Vec::new(), sources_bytes: Vec::new(), sources, meta }),
            published: RwLock::new(published),
        })
    }

    pub fn options(&self) -> Options {
        self.options
    }

    /// Adds a source: its key and its units in order, with their positions. Returns its ordinal. Visible to
    /// snapshots taken after the next [`Index::commit`].
    pub fn add<'u>(&self, key: &[u8], units: impl IntoIterator<Item = (&'u [u8], u32)>) -> u32 {
        let mut grouped: HashMap<&[u8], Vec<u32>> = HashMap::new();
        let mut length = 0u32;
        for (unit, position) in units {
            assert!(!unit.is_empty(), "units must not be empty");
            grouped.entry(unit).or_default().push(position);
            length += 1;
        }
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

    /// Makes everything added since the last commit durable and visible to new snapshots. The data files are
    /// written first and the dictionary (with the lengths that are now valid) commits last: a crash anywhere
    /// leaves the previous commit.
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

        self.postings_file.write_all_at(&records, w.meta.postings_length)?;
        w.meta.postings_length += records.len() as u64;
        if let Some(file) = &self.positions_file {
            file.write_all_at(&w.positions, w.meta.positions_length)?;
            w.meta.positions_length += w.positions.len() as u64;
        }
        self.sources_file.write_all_at(&w.sources_bytes, w.meta.sources_length)?;
        w.meta.sources_length += w.sources_bytes.len() as u64;
        if self.options.durability == Durability::Full {
            self.postings_file.sync_data()?;
            if let Some(file) = &self.positions_file {
                file.sync_data()?;
            }
            self.sources_file.sync_data()?;
        }
        w.meta.sources = w.sources.count();
        txn.put(META_KEY, &w.meta.encode())?;
        txn.commit()?;

        w.positions.clear();
        w.sources_bytes.clear();
        w.buffered = 0;
        let published = Published {
            postings: map(&self.postings_file)?,
            positions: self.positions_file.as_ref().map(map).transpose()?,
            sources: w.sources.clone(),
        };
        *self.published.write().unwrap() = published;
        Ok(())
    }

    /// What the last commit holds; cheap, and usable from any thread.
    pub fn snapshot(&self) -> Snapshot {
        let published = self.published.read().unwrap();
        Snapshot {
            dict: self.dict.begin_read(),
            postings: Postings { map: published.postings.clone(), frequencies: self.options.frequencies, positions: self.options.positions },
            positions: published.positions.clone(),
            sources: published.sources.clone(),
        }
    }
}

fn map(file: &File) -> io::Result<Arc<Mmap>> {
    // a zero-length file can't be mapped; postings always have their header, other files may be empty
    if file.metadata()?.len() == 0 {
        return Ok(Arc::new(memmap2::MmapOptions::new().len(0).map_anon()?.make_read_only()?));
    }
    Ok(Arc::new(unsafe { Mmap::map(file)? }))
}

/// A unit's postings: where its chain starts, and how many sources it lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitPostings {
    pub head: u64,
    pub count: u64,
}

/// The index as of a commit.
pub struct Snapshot {
    dict: ReadTxn,
    pub postings: Postings,
    positions: Option<Arc<Mmap>>,
    pub sources: Sources,
}

impl Snapshot {
    pub fn source_count(&self) -> u32 {
        self.sources.count()
    }

    pub fn unit(&self, unit: &[u8]) -> Option<UnitPostings> {
        if unit.is_empty() {
            return None;
        }
        self.dict.get(unit).map(|v| UnitPostings { head: u64::from_le_bytes(v[0..8].try_into().unwrap()), count: u64::from_le_bytes(v[8..16].try_into().unwrap()) })
    }

    /// Units starting with `prefix`, in byte order, with their postings.
    pub fn units_with_prefix<'a>(&'a self, prefix: &'a [u8]) -> impl Iterator<Item = (&'a [u8], UnitPostings)> + 'a {
        self.dict.range_from(prefix).take_while(move |(k, _)| k.starts_with(prefix)).filter(|(k, _)| !k.is_empty()).map(|(k, v)| {
            (k, UnitPostings { head: u64::from_le_bytes(v[0..8].try_into().unwrap()), count: u64::from_le_bytes(v[8..16].try_into().unwrap()) })
        })
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
