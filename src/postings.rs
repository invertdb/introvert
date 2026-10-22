//! Postings: which sources a unit occurs in, how often, and where its positions are.
//!
//! The postings blob is append-only: an 8-byte header ("IVPT", version, flags), then records. Each record holds a
//! unit's entries from one flush and points at the unit's previous record (a chain, newest first):
//!
//! ```text
//! previous record offset u64 | entry count varint | entries
//! ```
//!
//! A record of at most [`BLOCK`] entries holds them as varints (source delta, frequency, positions-offset delta; deltas
//! start from 0). A larger one is packed:
//!
//! ```text
//! first source varint | first positions offset varint | impact count varint | (frequency, length) varints
//! block index: per full block of BLOCK entries: last source u32 | data offset u32 | highest frequency u32 |
//!              shortest source u32 | last positions offset u64
//! data length u32 | blocks: widths (source, frequency, positions) u8 each, then the bit-packed source deltas,
//!                   frequencies − 1 and positions-offset deltas
//! remaining entries as varints (deltas from the last block's last entry) | padding
//! ```
//!
//! A unit's dictionary value ([`UnitPostings`]) holds its count and its newest record, or, for a unit whose entries
//! fit [`INLINE`] bytes and that has no record yet, the entries themselves (as a small record holds them): a rare unit
//! then costs no record and no second lookup.
//!
//! ```text
//! chain    count << 1 varint     | newest record offset varint
//! inline   count << 1 | 1 varint | entries
//! ```
//!
//! Impacts are (frequency, source length) pairs no entry of the record beats on both, bounding what any of its sources
//! can score. Frequencies and positions are optional per index (flags).
//!
//! Blocks are frame-of-reference coded (Goldstein, Ramakrishnan and Shaft, 1998): deltas in the bits the largest
//! needs. Impacts follow Anh and Moffat (2006), here as a Pareto set rather than quantized; per-block maxima in the
//! block index follow the block-max indexes of Ding and Suel (2011).



use crate::bitpack;
use crate::sourceset::SourceSet;
use crate::varint;

pub const BLOCK: usize = 128;
pub const HEADER: usize = 8;
pub const MAX_IMPACTS: usize = 8;
pub const NO_RECORD: u64 = u64::MAX;
pub const VERSION: u8 = 2;
/// Most bytes of entries a dictionary value holds inline (what fits [`UnitPostings`] in 32 bytes).
pub const INLINE: usize = 23;
const CHAIN: u8 = u8::MAX; // UnitPostings::length when its data holds the newest record's offset
pub const FLAG_FREQUENCIES: u8 = 1;
pub const FLAG_POSITIONS: u8 = 2;

pub fn blob_header(frequencies: bool, positions: bool) -> [u8; HEADER] {
    let flags = if frequencies { FLAG_FREQUENCIES } else { 0 } | if positions { FLAG_POSITIONS } else { 0 };
    [b'I', b'V', b'P', b'T', VERSION, flags, 0, 0]
}

/// A unit's postings: how many sources list it, and where its chain of records starts or its entries inline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UnitPostings {
    pub count: u64,
    data: [u8; INLINE], // the entries (`length` bytes), or the newest record's offset (length CHAIN)
    length: u8,
}

impl UnitPostings {
    pub(crate) fn decode(value: &[u8]) -> UnitPostings {
        let mut at = 0;
        let tagged = varint::read(value, &mut at);
        let mut unit = UnitPostings { count: tagged >> 1, data: [0; INLINE], length: CHAIN };
        if tagged & 1 == 1 {
            let entries = &value[at..];
            unit.length = entries.len() as u8;
            unit.data[..entries.len()].copy_from_slice(entries);
        } else {
            unit.data[..8].copy_from_slice(&varint::read(value, &mut at).to_le_bytes());
        }
        unit
    }

    /// The newest record of the unit's chain (NO_RECORD if its entries are inline).
    pub(crate) fn head(&self) -> u64 {
        if self.length == CHAIN { u64::from_le_bytes(self.data[..8].try_into().unwrap()) } else { NO_RECORD }
    }

    /// The entries, if the dictionary value holds them.
    pub(crate) fn inline(&self) -> Option<&[u8]> {
        (self.length != CHAIN).then(|| &self.data[..self.length as usize])
    }

    /// Identifies the postings (equal keys, equal postings), for caching what they match.
    /// A chain is its head; inline entries are their bytes (which determine the count).
    #[inline]
    pub(crate) fn key(&self, out: &mut Vec<u8>) {
        out.push(self.length);
        out.extend_from_slice(&self.data[..if self.length == CHAIN { 8 } else { self.length as usize }]);
    }
}

/// Encodes a unit's new entries (`buffer`, after `prior` older ones whose newest record is `previous`) and returns its
/// dictionary value: the entries inline if the unit has no record and they fit, else a record appended to `records`,
/// which starts at offset `base` in the blob. `scratch` is reused between calls.
#[allow(clippy::too_many_arguments)]
pub fn encode_unit(previous: u64, prior: u64, buffer: &Buffer, frequencies: bool, positions: bool, base: u64, records: &mut Vec<u8>, scratch: &mut Vec<u8>) -> Vec<u8> {
    let count = prior + buffer.sources.len() as u64;
    let mut value = Vec::with_capacity(12);
    if previous == NO_RECORD && buffer.sources.len() <= INLINE {
        scratch.clear();
        encode_entries(buffer, frequencies, positions, scratch);
        if scratch.len() <= INLINE {
            varint::write(&mut value, count << 1 | 1);
            value.extend_from_slice(scratch);
            return value;
        }
    }
    varint::write(&mut value, count << 1);
    varint::write(&mut value, base + records.len() as u64);
    encode_record(previous, buffer, frequencies, positions, records);
    value
}

/// A unit's inline entries followed by new ones (`buffer`, of later sources) as one buffer; `length` gives a source's
/// length, for the impacts.
pub fn merge_inline(unit: &UnitPostings, buffer: &Buffer, frequencies: bool, positions: bool, length: impl Fn(u32) -> u32) -> Buffer {
    let mut merged = Buffer::default();
    small_entries(unit.inline().unwrap(), 0, unit.count as usize, frequencies, positions, |s, f, o| merged.push(s, f, o, length(s)));
    for i in 0..buffer.sources.len() {
        merged.push(buffer.sources[i], buffer.frequencies[i], buffer.positions[i], length(buffer.sources[i]));
    }
    merged
}

/// Reads `count` small-record entries from `at`: (source, frequency, positions offset) each.
fn small_entries(bytes: &[u8], mut at: usize, count: usize, frequencies: bool, positions: bool, mut entry: impl FnMut(u32, u32, u64)) {
    let (mut source, mut offset) = (0u32, 0u64);
    for _ in 0..count {
        source += varint::read(bytes, &mut at) as u32;
        let f = if frequencies { varint::read(bytes, &mut at) as u32 } else { 1 };
        if positions {
            offset += varint::read(bytes, &mut at);
        }
        entry(source, f, offset);
    }
}

fn index_entry(positions: bool) -> usize {
    if positions { 24 } else { 16 }
}

/// A unit's postings from one flush, as the writer buffers them.
#[derive(Default, Clone, Debug)]
pub struct Buffer {
    pub sources: Vec<u32>,
    pub frequencies: Vec<u32>,
    pub positions: Vec<u64>,
    pub impacts: Vec<(u32, u32)>,
    pub block_maxima: Vec<(u32, u32)>,
    block_highest: u32,
    block_shortest: u32,
}

impl Buffer {
    pub fn push(&mut self, source: u32, frequency: u32, positions_offset: u64, length: u32) {
        if self.sources.is_empty() {
            self.block_shortest = u32::MAX;
        }
        self.sources.push(source);
        self.frequencies.push(frequency);
        self.positions.push(positions_offset);
        self.add_impact(frequency, length);
        self.block_highest = self.block_highest.max(frequency);
        self.block_shortest = self.block_shortest.min(length);
        if self.sources.len() % BLOCK == 0 {
            self.block_maxima.push((self.block_highest, self.block_shortest));
            self.block_highest = 0;
            self.block_shortest = u32::MAX;
        }
    }

    /// Keeps (frequency, length) pairs no entry beats on both; past MAX_IMPACTS the nearest two merge into one that
    /// beats both, which still bounds every entry.
    fn add_impact(&mut self, frequency: u32, length: u32) {
        let impacts = &mut self.impacts;
        if impacts.iter().any(|&(f, l)| f >= frequency && l <= length) {
            return;
        }
        impacts.retain(|&(f, l)| !(frequency >= f && length <= l));
        impacts.push((frequency, length));
        if impacts.len() <= MAX_IMPACTS {
            return;
        }
        impacts.sort_unstable();
        let merge = (0..impacts.len() - 1).min_by_key(|&i| impacts[i + 1].0 - impacts[i].0).unwrap();
        impacts[merge] = (impacts[merge + 1].0, impacts[merge].1);
        impacts.remove(merge + 1);
    }

    /// Bytes the buffer holds (for flushing by size).
    pub fn bytes(&self) -> usize {
        self.sources.len() * 16 + self.impacts.len() * 8
    }
}

/// A small record's entries (at most [`BLOCK`]): varints, each a delta from the entry before (the first from 0).
pub fn encode_entries(buffer: &Buffer, frequencies: bool, positions: bool, out: &mut Vec<u8>) {
    let (mut source, mut offset) = (0u32, 0u64);
    for i in 0..buffer.sources.len() {
        varint::write(out, (buffer.sources[i] - source) as u64);
        if frequencies {
            varint::write(out, buffer.frequencies[i] as u64);
        }
        if positions {
            varint::write(out, buffer.positions[i] - offset);
        }
        source = buffer.sources[i];
        offset = buffer.positions[i];
    }
}

/// Encodes a record (everything from the previous-record offset on).
pub fn encode_record(previous: u64, buffer: &Buffer, frequencies: bool, positions: bool, out: &mut Vec<u8>) {
    let count = buffer.sources.len();
    out.extend_from_slice(&previous.to_le_bytes());
    varint::write(out, count as u64);
    let (sources, freqs, offsets) = (&buffer.sources, &buffer.frequencies, &buffer.positions);

    let write_varints = |out: &mut Vec<u8>, from: usize, mut source: u32, mut offset: u64| {
        for i in from..count {
            varint::write(out, (sources[i] - source) as u64);
            if frequencies {
                varint::write(out, freqs[i] as u64);
            }
            if positions {
                varint::write(out, offsets[i] - offset);
            }
            source = sources[i];
            offset = offsets[i];
        }
    };
    if count <= BLOCK {
        encode_entries(buffer, frequencies, positions, out);
        return;
    }

    let blocks = count / BLOCK;
    varint::write(out, sources[0] as u64);
    if positions {
        varint::write(out, offsets[0]);
    }
    varint::write(out, buffer.impacts.len() as u64);
    for &(f, l) in &buffer.impacts {
        varint::write(out, f as u64);
        varint::write(out, l as u64);
    }

    let mut index = Vec::with_capacity(blocks * index_entry(positions));
    let mut data = Vec::with_capacity(count * 2);
    let mut values = [0u64; BLOCK];
    for b in 0..blocks {
        let first = b * BLOCK;
        // the first block's deltas start from one before its first source (wrapping, as decoding does, for source 0)
        let base_source = if b == 0 { sources[0].wrapping_sub(1) } else { sources[first - 1] };
        let base_offset = if b == 0 { offsets[0] } else { offsets[first - 1] };
        index.extend_from_slice(&sources[first + BLOCK - 1].to_le_bytes());
        index.extend_from_slice(&(data.len() as u32).to_le_bytes());
        index.extend_from_slice(&buffer.block_maxima[b].0.to_le_bytes());
        index.extend_from_slice(&buffer.block_maxima[b].1.to_le_bytes());
        if positions {
            index.extend_from_slice(&offsets[first + BLOCK - 1].to_le_bytes());
        }

        for i in 0..BLOCK {
            values[i] = sources[first + i].wrapping_sub(if i == 0 { base_source } else { sources[first + i - 1] }) as u64;
        }
        let source_width = bitpack::width(values.iter().copied().max().unwrap());
        let mut freq_values = [0u64; BLOCK];
        let mut offset_values = [0u64; BLOCK];
        for i in 0..BLOCK {
            freq_values[i] = (freqs[first + i] - 1) as u64;
            offset_values[i] = offsets[first + i] - if i == 0 { base_offset } else { offsets[first + i - 1] };
        }
        let freq_width = bitpack::width(freq_values.iter().copied().max().unwrap());
        let offset_width = bitpack::width(offset_values.iter().copied().max().unwrap());
        data.push(source_width as u8);
        if frequencies {
            data.push(freq_width as u8);
        }
        if positions {
            data.push(offset_width as u8);
        }
        bitpack::pack(&values, source_width, &mut data);
        if frequencies {
            bitpack::pack(&freq_values, freq_width, &mut data);
        }
        if positions {
            bitpack::pack(&offset_values, offset_width, &mut data);
        }
    }
    out.extend_from_slice(&index);
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&data);
    let last = blocks * BLOCK - 1;
    write_varints(out, blocks * BLOCK, sources[last], offsets[last]);
    out.extend_from_slice(&[0; bitpack::PADDING]);
}

/// The postings blob, through a view: reading records and cursors over chains.
#[derive(Clone)]
pub struct Postings {
    pub(crate) map: larry::View,
    pub frequencies: bool,
    pub positions: bool,
}

/// A packed record's layout, from just after its entry count.
#[derive(Clone, Copy, Default)]
struct Packed {
    count: usize,
    blocks: usize,
    entry: usize,
    widths: usize,
    first_source: u32,
    first_positions: u64,
    index: usize,
    data: usize,
    tail: usize,
}

impl Postings {
    fn bytes(&self) -> &[u8] {
        &self.map
    }

    pub(crate) fn previous(&self, record: u64) -> u64 {
        u64::from_le_bytes(self.bytes()[record as usize..record as usize + 8].try_into().unwrap())
    }

    /// The records of a unit's chain, oldest first (no allocation for the common chain of one record).
    fn chain(&self, unit: &UnitPostings) -> Records {
        if unit.inline().is_some() {
            return Records::Inline;
        }
        let head = unit.head();
        if self.previous(head) == NO_RECORD {
            return Records::One(head);
        }
        let mut records = Vec::new();
        let mut record = head;
        while record != NO_RECORD {
            records.push(record);
            record = self.previous(record);
        }
        records.reverse();
        Records::Many(records)
    }

    fn packed(&self, mut at: usize, count: usize) -> Packed {
        let bytes = self.bytes();
        let first_source = varint::read(bytes, &mut at) as u32;
        let first_positions = if self.positions { varint::read(bytes, &mut at) } else { 0 };
        let impacts = varint::read(bytes, &mut at) as usize;
        for _ in 0..2 * impacts {
            varint::read(bytes, &mut at);
        }
        let blocks = count / BLOCK;
        let entry = index_entry(self.positions);
        let index = at;
        let data_length = u32::from_le_bytes(bytes[index + blocks * entry..index + blocks * entry + 4].try_into().unwrap()) as usize;
        let data = index + blocks * entry + 4;
        Packed {
            count,
            blocks,
            entry,
            widths: 1 + self.frequencies as usize + self.positions as usize,
            first_source,
            first_positions,
            index,
            data,
            tail: data + data_length,
        }
    }

    /// (frequency, source length) pairs that together bound every entry of the chain: a packed record's impacts, or
    /// each entry of a smaller one (`length` gives a source's length).
    pub fn impacts(&self, unit: &UnitPostings, length: impl Fn(u32) -> u32, mut impact: impl FnMut(u32, u32)) {
        if let Some(entries) = unit.inline() {
            small_entries(entries, 0, unit.count as usize, self.frequencies, self.positions, |s, f, _| impact(f, length(s)));
            return;
        }
        let bytes = self.bytes();
        let mut record = unit.head();
        while record != NO_RECORD {
            let mut at = record as usize + 8;
            let count = varint::read(bytes, &mut at) as usize;
            if count > BLOCK {
                varint::read(bytes, &mut at);
                if self.positions {
                    varint::read(bytes, &mut at);
                }
                for _ in 0..varint::read(bytes, &mut at) {
                    let f = varint::read(bytes, &mut at) as u32;
                    impact(f, varint::read(bytes, &mut at) as u32);
                }
            } else {
                small_entries(bytes, at, count, self.frequencies, self.positions, |s, f, _| impact(f, length(s)));
            }
            record = self.previous(record);
        }
    }

    /// The unit's first (smallest) source, from its oldest record's header alone: in either layout, the first
    /// value after the entry count (a packed record's first source, or a small record's first delta from 0).
    pub fn first_source(&self, unit: &UnitPostings) -> Option<u32> {
        if let Some(entries) = unit.inline() {
            return Some(varint::read(entries, &mut 0) as u32);
        }
        let oldest = self.chain(unit).first()?;
        let mut at = oldest as usize + 8;
        if varint::read(self.bytes(), &mut at) == 0 {
            return None;
        }
        Some(varint::read(self.bytes(), &mut at) as u32)
    }

    /// Adds every source of the unit's postings to a set.
    pub fn add_sources_to(&self, unit: &UnitPostings, set: &mut SourceSet) {
        if let Some(entries) = unit.inline() {
            // read in place: a cursor would cost more to set up than these few entries do to read
            small_entries(entries, 0, unit.count as usize, self.frequencies, self.positions, |s, _, _| set.insert(s));
            return;
        }
        let mut cursor = self.cursor(unit);
        while cursor.next() {
            set.insert(cursor.source);
        }
    }

    /// Every entry of the unit's postings: (source, frequency, positions offset).
    pub fn decode(&self, unit: &UnitPostings) -> Vec<(u32, u32, u64)> {
        let mut out = Vec::new();
        if let Some(entries) = unit.inline() {
            out.reserve(unit.count as usize);
            small_entries(entries, 0, unit.count as usize, self.frequencies, self.positions, |s, f, o| out.push((s, f, o)));
            return out;
        }
        let mut cursor = self.cursor(unit);
        while cursor.next() {
            out.push((cursor.source, cursor.frequency, cursor.positions_offset()));
        }
        out
    }

    /// A cursor over the unit's postings (which it borrows: inline entries are read where they are).
    pub fn cursor<'a>(&'a self, unit: &'a UnitPostings) -> Cursor<'a> {
        Cursor {
            postings: self,
            records: self.chain(unit),
            inline: unit.inline().unwrap_or(&[]),
            inline_count: unit.count as usize,
            varints: self.bytes(),
            record: usize::MAX,
            packed: false,
            view: Packed::default(),
            block: 0,
            entry: 0,
            sources: [0; BLOCK],
            freqs: [0; BLOCK],
            offsets: [0; BLOCK],
            offsets_decoded: false,
            at: 0,
            remaining: 0,
            running_offset: 0,
            source: NOT_STARTED,
            frequency: 0,
        }
    }
}

/// A chain's records, oldest first (or the unit's entries, inline: as one record).
enum Records {
    One(u64),
    Many(Vec<u64>),
    Inline,
}

impl Records {
    fn len(&self) -> usize {
        match self {
            Records::One(_) | Records::Inline => 1,
            Records::Many(r) => r.len(),
        }
    }

    fn get(&self, i: usize) -> u64 {
        match self {
            Records::One(r) => *r,
            Records::Many(r) => r[i],
            Records::Inline => unreachable!("inline entries have no record"),
        }
    }

    fn first(&self) -> Option<u64> {
        match self {
            Records::One(r) => Some(*r),
            Records::Many(r) => r.first().copied(),
            Records::Inline => None,
        }
    }
}

pub const NOT_STARTED: u32 = u32::MAX - 1;
const BEFORE: usize = usize::MAX; // a block's entry index before its first entry
pub const NO_MORE: u32 = u32::MAX;

/// Reads a chain in source order. In packed records it decodes a block at a time and [`Cursor::advance`]
/// binary-searches the block index.
pub struct Cursor<'a> {
    postings: &'a Postings,
    records: Records,
    inline: &'a [u8],
    inline_count: usize,
    varints: &'a [u8], // where the current record's varints are: the blob, or the inline entries
    record: usize,
    packed: bool,
    view: Packed,
    block: usize,   // current block of a packed record (== blocks: in its varint tail)
    entry: usize,   // index in the block (BEFORE: before its first)
    sources: [u32; BLOCK],
    freqs: [u32; BLOCK],
    offsets: [u64; BLOCK],
    offsets_decoded: bool,
    at: usize,       // next varint entry
    remaining: usize,
    running_offset: u64,
    /// The current source: NOT_STARTED, a source, or NO_MORE.
    pub source: u32,
    pub frequency: u32,
}

impl<'a> Cursor<'a> {
    #[inline]
    fn index_u32(&self, block: usize, offset: usize) -> u32 {
        let at = self.view.index + block * self.view.entry + offset;
        u32::from_le_bytes(self.postings.bytes()[at..at + 4].try_into().unwrap())
    }

    #[inline]
    fn block_last(&self, block: usize) -> u32 {
        self.index_u32(block, 0)
    }

    fn block_positions_last(&self, block: usize) -> u64 {
        let at = self.view.index + block * self.view.entry + 16;
        u64::from_le_bytes(self.postings.bytes()[at..at + 8].try_into().unwrap())
    }

    /// Where the current entry's positions are (decoded on first use in a block).
    pub fn positions_offset(&mut self) -> u64 {
        if !self.packed || self.block >= self.view.blocks {
            return self.running_offset;
        }
        if !self.offsets_decoded {
            self.decode_offsets();
        }
        self.offsets[self.entry]
    }

    /// Moves to the next entry; false when there are none.
    pub fn next(&mut self) -> bool {
        loop {
            if self.packed && self.block < self.view.blocks {
                self.entry = self.entry.wrapping_add(1);
                if self.entry < BLOCK {
                    self.source = self.sources[self.entry];
                    self.frequency = self.freqs[self.entry];
                    return true;
                }
                if self.block + 1 < self.view.blocks {
                    self.load_block(self.block + 1);
                    self.entry = 0;
                    self.source = self.sources[0];
                    self.frequency = self.freqs[0];
                    return true;
                }
                self.enter_tail();
            }
            if self.remaining > 0 {
                self.read_varint();
                return true;
            }
            if !self.enter_next_record() {
                return false;
            }
        }
    }

    /// Moves to the first entry at or after `target` (no move if already there); false when there is none.
    pub fn advance(&mut self, target: u32) -> bool {
        if self.source != NOT_STARTED && self.source >= target {
            return self.source != NO_MORE;
        }
        loop {
            if self.packed && self.block < self.view.blocks {
                if self.sources[BLOCK - 1] >= target {
                    let mut i = self.entry.wrapping_add(1);
                    while self.sources[i] < target {
                        i += 1;
                    }
                    self.entry = i;
                    self.source = self.sources[i];
                    self.frequency = self.freqs[i];
                    return true;
                }
                // the first later block whose last source reaches the target
                let (mut low, mut high) = (self.block + 1, self.view.blocks);
                while low < high {
                    let mid = (low + high) / 2;
                    if self.block_last(mid) < target { low = mid + 1 } else { high = mid }
                }
                if low < self.view.blocks {
                    self.load_block(low);
                    let mut i = 0;
                    while self.sources[i] < target {
                        i += 1;
                    }
                    self.entry = i;
                    self.source = self.sources[i];
                    self.frequency = self.freqs[i];
                    return true;
                }
                self.enter_tail();
            }
            while self.remaining > 0 {
                self.read_varint();
                if self.source >= target {
                    return true;
                }
            }
            if !self.enter_next_record() {
                return false;
            }
        }
    }

    /// The block holding this cursor's entries from `target` on (not before the current entry): its last source,
    /// highest frequency and shortest source. None when not known (outside a packed record's blocks). Moves nothing.
    pub fn block_at(&self, target: u32) -> Option<(u32, u32, u32)> {
        if self.source == NO_MORE {
            return Some((NO_MORE, 0, 0));
        }
        if !self.packed || self.block >= self.view.blocks {
            return None;
        }
        let mut block = self.block;
        if self.sources[BLOCK - 1] < target {
            let (mut low, mut high) = (self.block + 1, self.view.blocks);
            while low < high {
                let mid = (low + high) / 2;
                if self.block_last(mid) < target { low = mid + 1 } else { high = mid }
            }
            if low >= self.view.blocks {
                return None;
            }
            block = low;
        }
        Some((self.block_last(block), self.index_u32(block, 8), self.index_u32(block, 12)))
    }

    fn load_block(&mut self, block: usize) {
        self.block = block;
        let bytes = self.postings.bytes();
        let data = self.view.data + self.index_u32(block, 4) as usize;
        let source_width = bytes[data] as u32;
        let freq_width = if self.postings.frequencies { bytes[data + 1] as u32 } else { 0 };
        let packed = data + self.view.widths;
        bitpack::unpack_u32(&bytes[packed..], source_width, &mut self.sources);
        let mut source = if block == 0 { self.view.first_source.wrapping_sub(1) } else { self.block_last(block - 1) };
        for d in self.sources.iter_mut() {
            source = source.wrapping_add(*d);
            *d = source;
        }
        if self.postings.frequencies {
            bitpack::unpack_u32(&bytes[packed + bitpack::bytes(BLOCK, source_width)..], freq_width, &mut self.freqs);
            for f in self.freqs.iter_mut() {
                *f += 1;
            }
        } else {
            self.freqs.fill(1);
        }
        self.offsets_decoded = false;
    }

    fn decode_offsets(&mut self) {
        let bytes = self.postings.bytes();
        let data = self.view.data + self.index_u32(self.block, 4) as usize;
        let source_width = bytes[data] as u32;
        let freq_width = if self.postings.frequencies { bytes[data + 1] as u32 } else { 0 };
        let offset_width = bytes[data + self.view.widths - 1] as u32;
        let packed = data + self.view.widths + bitpack::bytes(BLOCK, source_width) + if self.postings.frequencies { bitpack::bytes(BLOCK, freq_width) } else { 0 };
        bitpack::unpack_u64(&bytes[packed..], offset_width, &mut self.offsets);
        let mut position = if self.block == 0 { self.view.first_positions } else { self.block_positions_last(self.block - 1) };
        for o in self.offsets.iter_mut() {
            position += *o;
            *o = position;
        }
        self.offsets_decoded = true;
    }

    fn enter_tail(&mut self) {
        let last = self.view.blocks - 1;
        self.source = self.block_last(last);
        self.running_offset = if self.postings.positions { self.block_positions_last(last) } else { 0 };
        self.at = self.view.tail;
        self.remaining = self.view.count - self.view.blocks * BLOCK;
        self.block = self.view.blocks;
    }

    #[inline]
    fn read_varint(&mut self) {
        let bytes = self.varints;
        let base = if self.source == NOT_STARTED { 0 } else { self.source };
        self.source = base + varint::read(bytes, &mut self.at) as u32;
        self.frequency = if self.postings.frequencies { varint::read(bytes, &mut self.at) as u32 } else { 1 };
        if self.postings.positions {
            self.running_offset += varint::read(bytes, &mut self.at);
        }
        self.remaining -= 1;
    }

    fn enter_next_record(&mut self) -> bool {
        self.record = self.record.wrapping_add(1);
        if self.record >= self.records.len() {
            self.source = NO_MORE;
            return false;
        }
        self.source = NOT_STARTED; // deltas restart in each record
        self.running_offset = 0;
        if let Records::Inline = self.records {
            self.varints = self.inline;
            self.packed = false;
            self.at = 0;
            self.remaining = self.inline_count;
            return true;
        }
        let mut at = self.records.get(self.record) as usize + 8;
        let count = varint::read(self.postings.bytes(), &mut at) as usize;
        self.packed = count > BLOCK;
        if self.packed {
            self.view = self.postings.packed(at, count);
            self.load_block(0);
            self.entry = BEFORE;
            self.remaining = 0;
        } else {
            self.at = at;
            self.remaining = count;
        }
        true
    }
}
