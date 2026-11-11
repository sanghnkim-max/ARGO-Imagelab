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

/// Side of the fingerprint grid ([`signature`]).
const SIG_GRID: usize = 16;
/// A fingerprint matches when its mean absolute difference is at most this (0..255)…
const SIG_MEAN_TOLERANCE: f64 = 1.5;
/// …and no block differs by more than this.
const SIG_MAX_TOLERANCE: u8 = 12;

/// Coarse fingerprint of an RGB proxy: `SIG_GRID²` block means per channel, plus the size.
pub fn signature(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut sig = Vec::with_capacity(SIG_GRID * SIG_GRID * 3 + 8);
    sig.extend_from_slice(&(w as u32).to_le_bytes());
    sig.extend_from_slice(&(h as u32).to_le_bytes());
    if w == 0 || h == 0 || rgb.len() < w * h * 3 {
        return sig;
    }
    for by in 0..SIG_GRID {
        let (y0, y1) = (by * h / SIG_GRID, ((by + 1) * h / SIG_GRID).max(by * h / SIG_GRID + 1).min(h));
        for bx in 0..SIG_GRID {
            let (x0, x1) = (bx * w / SIG_GRID, ((bx + 1) * w / SIG_GRID).max(bx * w / SIG_GRID + 1).min(w));
            let mut acc = [0u64; 3];
            let mut n = 0u64;
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = (y * w + x) * 3;
                    for (c, a) in acc.iter_mut().enumerate() {
                        *a += u64::from(rgb.get(i + c).copied().unwrap_or(0));
                    }
                    n += 1;
                }
            }
            for a in acc {
                sig.push((a / n.max(1)) as u8);
            }
        }
    }
    sig
}

/// Whether two fingerprints describe the same image (same size, block means within tolerance).
pub fn similar(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() || a.len() < 8 || a.get(..8) != b.get(..8) {
        return false;
    }
    let (mut sum, mut max) = (0u64, 0u8);
    for (x, y) in a.iter().zip(b).skip(8) {
        let d = x.abs_diff(*y);
        sum += u64::from(d);
        max = max.max(d);
    }
    let n = (a.len() - 8).max(1) as f64;
    max <= SIG_MAX_TOLERANCE && (sum as f64 / n) <= SIG_MEAN_TOLERANCE
}

/// Least-recently-used cache of embeddings.
pub struct EmbeddingCache<E> {
    capacity: usize,
    /// Most recent last: (exact key, fingerprint, embedding).
    entries: VecDeque<(u64, Vec<u8>, E)>,
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

    /// The embedding for these exact pixels, encoding on a miss. Returns it and whether it hit.
    pub fn get_or_encode<En: Encoder<Embedding = E>>(&mut self, encoder: &mut En, rgb: &[u8], w: usize, h: usize) -> Result<(&mut E, bool), String> {
        let key = content_key(rgb, w, h);
        self.lookup_or_encode(|k, _| k == key, key, Vec::new(), encoder, rgb, w, h)
    }

    /// The embedding for an image that *looks the same* ([`signature`] within [`similar`]
    /// tolerance), encoding on a miss: robust to render jitter and selection-only changes.
    pub fn get_or_encode_similar<En: Encoder<Embedding = E>>(&mut self, encoder: &mut En, rgb: &[u8], w: usize, h: usize) -> Result<(&mut E, bool), String> {
        let sig = signature(rgb, w, h);
        let key = content_key(rgb, w, h);
        let probe = sig.clone();
        self.lookup_or_encode(move |k, s| k == key || similar(s, &probe), key, sig, encoder, rgb, w, h)
    }

    #[allow(clippy::too_many_arguments)]
    fn lookup_or_encode<En: Encoder<Embedding = E>>(
        &mut self,
        matches: impl Fn(u64, &[u8]) -> bool,
        key: u64,
        sig: Vec<u8>,
        encoder: &mut En,
        rgb: &[u8],
        w: usize,
        h: usize,
    ) -> Result<(&mut E, bool), String> {
        let hit = if let Some(pos) = self.entries.iter().position(|(k, s, _)| matches(*k, s)) {
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
            self.entries.push_back((key, sig, emb));
            self.misses += 1;
            false
        };
        match self.entries.back_mut() {
            Some((_, _, e)) => Ok((e, hit)),
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
    fn similar_images_share_an_embedding() {
        let mut enc = Fake { calls: 0 };
        let mut c = EmbeddingCache::new(4);
        let (w, h) = (64, 48);
        let base: Vec<u8> = (0..w * h * 3).map(|i| ((i * 7) % 200) as u8).collect();
        // Render jitter: a few values off by one.
        let mut jitter = base.clone();
        for i in (0..jitter.len()).step_by(97) {
            jitter[i] = jitter[i].saturating_add(1);
        }
        // A real edit: everything brighter.
        let edited: Vec<u8> = base.iter().map(|v| v.saturating_add(40)).collect();
        assert!(!c.get_or_encode_similar(&mut enc, &base, w, h).unwrap().1);
        assert!(c.get_or_encode_similar(&mut enc, &jitter, w, h).unwrap().1, "jitter reuses the embedding");
        assert!(!c.get_or_encode_similar(&mut enc, &edited, w, h).unwrap().1, "an edit encodes again");
        assert_eq!(enc.calls, 2);
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
