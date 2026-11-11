//! The embedding cache: the image encoder runs once per image (keyed by a hash of the proxy's
//! pixels); every later prompt on the same image only runs the decoder. A small LRU, bounded by
//! entry count (a SAM 3 embedding with its head features is tens of MB).

use std::collections::VecDeque;
use std::hash::{Hash, Hasher};

/// Produces an embedding from an RGB proxy. The SAM engine implements it; tests use a fake.
pub trait Encoder {
    type Embedding;
    fn encode(&mut self, rgb: &[u8], w: usize, h: usize) -> Result<Self::Embedding, String>;
}

/// Content key of an RGB proxy (dimensions + pixels).
pub fn content_key(rgb: &[u8], w: usize, h: usize) -> u64 {
    let mut s = std::collections::hash_map::DefaultHasher::new();
    (w, h).hash(&mut s);
    rgb.hash(&mut s);
    s.finish()
}

/// Least-recently-used cache of embeddings.
pub struct EmbeddingCache<E> {
    capacity: usize,
    /// Most recent last.
    entries: VecDeque<(u64, E)>,
    pub hits: u64,
    pub misses: u64,
}

impl<E> EmbeddingCache<E> {
    pub fn new(capacity: usize) -> Self {
        Self { capacity: capacity.max(1), entries: VecDeque::new(), hits: 0, misses: 0 }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// The embedding for `key`, encoding it with `encoder` on a miss. Returns it and whether
    /// it was a hit.
    pub fn get_or_encode<En: Encoder<Embedding = E>>(&mut self, encoder: &mut En, rgb: &[u8], w: usize, h: usize) -> Result<(&mut E, bool), String> {
        let key = content_key(rgb, w, h);
        let hit = if let Some(pos) = self.entries.iter().position(|(k, _)| *k == key) {
            if let Some(e) = self.entries.remove(pos) {
                self.entries.push_back(e);
            }
            self.hits += 1;
            true
        } else {
            let emb = encoder.encode(rgb, w, h)?;
            while self.entries.len() >= self.capacity {
                self.entries.pop_front();
            }
            self.entries.push_back((key, emb));
            self.misses += 1;
            false
        };
        match self.entries.back_mut() {
            Some((_, e)) => Ok((e, hit)),
            None => Err("embedding cache is empty after insert".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake {
        calls: usize,
    }
    impl Encoder for Fake {
        type Embedding = (usize, u8);
        fn encode(&mut self, rgb: &[u8], _w: usize, _h: usize) -> Result<(usize, u8), String> {
            self.calls += 1;
            Ok((self.calls, rgb.first().copied().unwrap_or(0)))
        }
    }

    #[test]
    fn encoder_runs_once_per_image_and_lru_evicts() {
        let mut enc = Fake { calls: 0 };
        let mut c = EmbeddingCache::new(2);
        let a = [1u8; 12];
        let b = [2u8; 12];
        let d = [3u8; 12];
        assert!(!c.get_or_encode(&mut enc, &a, 2, 2).unwrap().1);
        assert!(c.get_or_encode(&mut enc, &a, 2, 2).unwrap().1);
        assert_eq!(enc.calls, 1);
        // same pixels, other shape: another image
        assert!(!c.get_or_encode(&mut enc, &a, 4, 1).unwrap().1);
        assert!(!c.get_or_encode(&mut enc, &b, 2, 2).unwrap().1);
        assert_eq!(c.len(), 2);
        // a (2×2) was evicted by (4×1) and b; touch b, add d: (4×1) goes
        assert!(c.get_or_encode(&mut enc, &b, 2, 2).unwrap().1);
        assert!(!c.get_or_encode(&mut enc, &d, 2, 2).unwrap().1);
        assert!(c.get_or_encode(&mut enc, &b, 2, 2).unwrap().1);
        assert!(!c.get_or_encode(&mut enc, &a, 4, 1).unwrap().1);
        assert_eq!((c.hits, c.misses), (3, 5));
        assert_eq!(enc.calls, 5);
    }

    #[test]
    fn encoder_errors_are_not_cached() {
        struct Failing;
        impl Encoder for Failing {
            type Embedding = ();
            fn encode(&mut self, _: &[u8], _: usize, _: usize) -> Result<(), String> {
                Err("boom".into())
            }
        }
        let mut c = EmbeddingCache::new(2);
        assert!(c.get_or_encode(&mut Failing, &[0; 3], 1, 1).is_err());
        assert!(c.is_empty());
    }
}
