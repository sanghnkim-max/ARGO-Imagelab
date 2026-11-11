//! # imagelab-cv
//!
//! Cv area: OpenCV-style image operations on handles, pure Rust. See docs/INTERFACES.md.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use imagelab_api::{Module, Registry};

/// Registers this crate's areas and functions.
pub struct CvModule;

impl Module for CvModule {
    fn register(&self, registry: &mut Registry) {
        registry.area("cv", "OpenCV-style image ops on handles (img:N, mask:N): resize, blur, canny, threshold, contours, masks…");
    }
}
