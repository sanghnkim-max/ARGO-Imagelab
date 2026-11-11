//! # imagelab-segment
//!
//! Select + segment areas: PhotoCraft selections and SAM 3 masks (point, box, text concept) with cached embeddings and model management. See docs/INTERFACES.md.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use imagelab_api::{Module, Registry};

/// Registers this crate's areas and functions.
pub struct SegmentModule;

impl Module for SegmentModule {
    fn register(&self, registry: &mut Registry) {
        registry.area("select", "Cheap PhotoCraft selections: subject, sky, object, color range, magic wand");
        registry.area("segment", "SAM 3 masks from points, boxes or text concepts; model status and download");
    }
}
