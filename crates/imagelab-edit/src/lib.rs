//! # imagelab-edit
//!
//! Doc + edit + batch areas (docs/INTERFACES.md §2):
//!
//! * `doc` — the UI's PhotoCraft session: open (sidecar-aware), new, list, select, inspect,
//!   preview (to an `img:N` handle), save (to the `.pcraft` sidecar), export, close, undo/redo.
//! * `edit` — allowlisted engine commands (generated from PhotoCraft's registry, see [`allow`]),
//!   atomic step lists, adjustment layers, auto fixes, straighten, crop, recipes.
//! * `batch` — recipes/steps over many photos in throwaway sessions, with progress events,
//!   cancel and export presets.
//!
//! Originals under `library/` are never written; edits live in `edits/<id>.pcraft`, recipes in
//! `recipes/<name>.json`, exports default to `exports/`.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod allow;
pub mod batch;
pub mod doc;
pub mod edit;
pub mod fonts;
pub mod ops;
pub mod paths;

#[cfg(test)]
mod tests;

use imagelab_api::{Module, Registry};

/// Registers this crate's areas and functions.
pub struct EditModule;

impl Module for EditModule {
    fn register(&self, registry: &mut Registry) {
        registry.area("doc", "The open PhotoCraft document: open, inspect, preview, save, history");
        registry.area("edit", "Edit the document: engine commands, masked adjustment layers, recipes");
        registry.area("batch", "Apply a recipe or program to many photos; export presets");
        fonts::register_area(registry);
        doc::register(registry);
        edit::register(registry);
        batch::register(registry);
    }
}
