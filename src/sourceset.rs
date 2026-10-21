//! A set of source ordinals: a bitset whose 64-bit words are allocated lazily, in blocks.

const BLOCK_WORDS: usize = 1024; // 65,536 sources a block

#[derive(Clone, Debug)]
pub struct SourceSet {
    capacity: u32,
    blocks: Vec<Option<Box<[u64; BLOCK_WORDS]>>>,
}

impl SourceSet {
    pub fn new(capacity: u32) -> SourceSet {
        SourceSet { capacity, blocks: vec![None; (capacity as usize).div_ceil(BLOCK_WORDS * 64)] }
    }

    pub fn all(capacity: u32) -> SourceSet {
        let mut set = SourceSet::new(capacity);
        for source in 0..capacity {
            set.insert(source);
        }
        set
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    #[inline]
    pub fn insert(&mut self, source: u32) {
        let word = source as usize / 64;
        let block = self.blocks[word / BLOCK_WORDS].get_or_insert_with(|| Box::new([0; BLOCK_WORDS]));
        block[word % BLOCK_WORDS] |= 1 << (source % 64);
    }

    #[inline]
    pub fn remove(&mut self, source: u32) {
        let word = source as usize / 64;
        if let Some(block) = &mut self.blocks[word / BLOCK_WORDS] {
            block[word % BLOCK_WORDS] &= !(1 << (source % 64));
        }
    }

    #[inline]
    pub fn contains(&self, source: u32) -> bool {
        let word = source as usize / 64;
        match &self.blocks[word / BLOCK_WORDS] {
            Some(block) => block[word % BLOCK_WORDS] & (1 << (source % 64)) != 0,
            None => false,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.iter().flatten().all(|b| b.iter().all(|&w| w == 0))
    }

    pub fn len(&self) -> usize {
        self.blocks.iter().flatten().map(|b| b.iter().map(|w| w.count_ones() as usize).sum::<usize>()).sum()
    }

    pub fn and(&mut self, other: &SourceSet) {
        for (mine, theirs) in self.blocks.iter_mut().zip(&other.blocks) {
            match (mine.as_mut(), theirs) {
                (Some(a), Some(b)) => a.iter_mut().zip(b.iter()).for_each(|(x, y)| *x &= y),
                (Some(_), None) => *mine = None,
                _ => {}
            }
        }
    }

    pub fn or(&mut self, other: &SourceSet) {
        for (mine, theirs) in self.blocks.iter_mut().zip(&other.blocks) {
            if let Some(b) = theirs {
                let a = mine.get_or_insert_with(|| Box::new([0; BLOCK_WORDS]));
                a.iter_mut().zip(b.iter()).for_each(|(x, y)| *x |= y);
            }
        }
    }

    pub fn and_not(&mut self, other: &SourceSet) {
        for (mine, theirs) in self.blocks.iter_mut().zip(&other.blocks) {
            if let (Some(a), Some(b)) = (mine.as_mut(), theirs) {
                a.iter_mut().zip(b.iter()).for_each(|(x, y)| *x &= !y);
            }
        }
    }

    /// The members, ascending.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.blocks.iter().enumerate().filter_map(|(b, block)| block.as_ref().map(|block| (b, block))).flat_map(|(b, block)| {
            block.iter().enumerate().flat_map(move |(w, &word)| {
                let base = ((b * BLOCK_WORDS + w) * 64) as u32;
                BitIter(word).map(move |bit| base + bit)
            })
        })
    }
}

struct BitIter(u64);

impl Iterator for BitIter {
    type Item = u32;
    #[inline]
    fn next(&mut self) -> Option<u32> {
        if self.0 == 0 {
            return None;
        }
        let bit = self.0.trailing_zeros();
        self.0 &= self.0 - 1;
        Some(bit)
    }
}
