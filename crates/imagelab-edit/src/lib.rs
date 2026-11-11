//! # imagelab-edit
//!
//! Doc + edit + batch areas: the PhotoCraft document, allowlisted engine commands, masked adjustments, recipes, batch apply and export presets. See docs/INTERFACES.md.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use imagelab_api::{Module, Registry};

/// Registers this crate's areas and functions.
pub struct EditModule;

impl Module for EditModule {
    fn register(&self, registry: &mut Registry) {
        registry.area("doc", "The open PhotoCraft document: open, inspect, preview, save, history");
        registry.area("edit", "Edit the document: engine commands, masked adjustment layers, recipes");
        registry.area("batch", "Apply a recipe or program to many photos; export presets");
    }
}
