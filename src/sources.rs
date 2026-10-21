//! The sources: each gets an ordinal (0, 1, 2, ...) in the order added; postings refer to sources by ordinal. For
//! each, its key and length in units. Held in memory in chunks of 65,536, so a snapshot shares every full chunk
//! and copies at most the last when the writer appends after it.

use std::sync::Arc;

pub const CHUNK_BITS: u32 = 16;
pub const CHUNK: usize = 1 << CHUNK_BITS;

#[derive(Clone, Default)]
pub struct Chunk {
    lengths: Vec<u32>,
    key_ends: Vec<u32>,
    keys: Vec<u8>,
}

#[derive(Clone, Default)]
pub struct Sources {
    chunks: Vec<Arc<Chunk>>,
    count: u32,
    total_length: u64,
}

impl Sources {
    pub fn count(&self) -> u32 {
        self.count
    }

    pub fn total_length(&self) -> u64 {
        self.total_length
    }

    pub fn average_length(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.total_length as f64 / self.count as f64 }
    }

    #[inline]
    pub fn length(&self, ordinal: u32) -> u32 {
        self.chunks[(ordinal >> CHUNK_BITS) as usize].lengths[ordinal as usize & (CHUNK - 1)]
    }

    pub fn key(&self, ordinal: u32) -> &[u8] {
        let chunk = &self.chunks[(ordinal >> CHUNK_BITS) as usize];
        let i = ordinal as usize & (CHUNK - 1);
        let start = if i == 0 { 0 } else { chunk.key_ends[i - 1] as usize };
        &chunk.keys[start..chunk.key_ends[i] as usize]
    }

    /// Adds a source; returns its ordinal.
    pub fn push(&mut self, key: &[u8], length: u32) -> u32 {
        if self.count as usize % CHUNK == 0 {
            self.chunks.push(Arc::new(Chunk::default()));
        }
        let chunk = Arc::make_mut(self.chunks.last_mut().unwrap());
        chunk.keys.extend_from_slice(key);
        chunk.key_ends.push(chunk.keys.len() as u32);
        chunk.lengths.push(length);
        self.total_length += length as u64;
        self.count += 1;
        self.count - 1
    }

    /// The first `count` sources (after a crash, the ones the last commit recorded).
    pub fn truncate(&mut self, count: u32) {
        while self.count > count {
            let chunk = Arc::make_mut(self.chunks.last_mut().unwrap());
            self.total_length -= chunk.lengths.pop().unwrap() as u64;
            chunk.key_ends.pop();
            chunk.keys.truncate(chunk.key_ends.last().copied().unwrap_or(0) as usize);
            if chunk.lengths.is_empty() {
                self.chunks.pop();
            }
            self.count -= 1;
        }
    }
}
