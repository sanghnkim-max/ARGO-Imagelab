//! # imagelab-library
//!
//! Library + analyze areas: photo library (import, query, albums, ratings, flags, tags, metadata) and analysis (quality scores, duplicates, histogram, faces, contact sheet). See docs/INTERFACES.md.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use imagelab_api::{Module, Registry};

/// Registers this crate's areas and functions.
pub struct LibraryModule;

impl Module for LibraryModule {
    fn register(&self, registry: &mut Registry) {
        registry.area("library", "Photos in the workspace: import, query, albums, ratings, flags, tags, metadata");
        registry.area("analyze", "Measure photos: quality scores, near-duplicates, histograms, faces, contact sheets");
    }
}
