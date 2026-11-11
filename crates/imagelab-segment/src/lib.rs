//! # imagelab-segment
//!
//! The `select` and `segment` areas (docs/INTERFACES.md §6, "Segmentation, used wisely"):
//!
//! * `select.*` — PhotoCraft's own selections on the open document (subject, sky, object,
//!   colour range, magic wand, refine edge, all/none/inverse), no model, and the bridges
//!   `select.toMask` (selection → `mask:N`) and `select.fromMask` (`mask:N` → selection, one
//!   undoable step), so masks from any area feed adjustment layers and filters.
//! * `segment.*` — SAM 3 (LightCraft's pure-Rust candle port, feature `sam3`) from points, a
//!   box, a text concept or the subject; encoder once per image on a ≤1008 px proxy, embeddings
//!   cached by content (LRU), model loaded lazily and dropped when idle; PhotoCraft fallback
//!   when the model is missing or fails. Model status, download (confirmed), install.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod cache;
pub mod input;
pub mod mask;
#[cfg(feature = "mobile-sam")]
pub mod mobile;
#[cfg(feature = "sam3")]
mod sam;
pub mod segment;
pub mod select;

use std::sync::Arc;

use imagelab_api::{Module, Registry};

#[cfg(feature = "sam3")]
use sam::Sam;

/// Registers this crate's areas and functions. Each registration owns its SAM state (loaded
/// model, embedding cache, download), shared by that registry's handlers.
pub struct SegmentModule;

/// State shared by the handlers of one registration.
#[derive(Default)]
pub struct State {
    sam: Sam,
    mobile: Mobile,
}

impl Module for SegmentModule {
    fn register(&self, registry: &mut Registry) {
        registry.area("select", "Cheap PhotoCraft selections on the open document (subject, sky, object, color range, magic wand, refine edge) and selection ↔ mask:N bridges");
        registry.area("segment", "Masks from points, boxes, text concepts or the subject: SAM 3 (desktop), MobileSAM (phone, 40 MB), PhotoCraft fallback; model status and install");
        select::register(registry);
        segment::register(registry, Arc::new(State::default()));
    }
}

/// Built without `sam3`: every model call reports it and the document falls back to PhotoCraft.
#[cfg(not(feature = "sam3"))]
mod nosam {
    use std::path::Path;

    use imagelab_api::{ApiError, ApiResult, Ctx};
    use serde_json::{Value, json};

    use crate::input::Input;
    use crate::segment::{Click, Pred};

    type Instance = (Pred, Option<[f32; 4]>);

    const WHY: &str = "this build has no SAM 3 (feature `sam3` off)";

    #[derive(Default)]
    pub struct Sam;

    impl Sam {
        pub fn clicks(&self, _: &Path, _: &Input, _: &[Click]) -> Result<(Pred, bool), String> {
            Err(WHY.into())
        }
        pub fn text(&self, _: &Path, _: &Input, _: &str, _: f32, _: usize) -> Result<(Vec<Instance>, bool), String> {
            Err(WHY.into())
        }
        pub fn subject(&self, _: &Path, _: &Input) -> Result<(Option<Pred>, bool), String> {
            Err(WHY.into())
        }
        pub fn status(&self, _: &Path) -> Value {
            json!({"sam3": "unavailable", "reason": WHY, "loaded": false, "cachedEmbeddings": 0})
        }
        pub fn ensure(&self, _: &Ctx, _: bool) -> ApiResult<Value> {
            Err(ApiError::unavailable(WHY))
        }
        pub fn install(&self, _: &Ctx, _: &Path) -> ApiResult<Value> {
            Err(ApiError::unavailable(WHY))
        }
        pub fn cancel_download(&self) -> bool {
            false
        }
        pub fn unload(&self) -> bool {
            false
        }
    }
}
#[cfg(not(feature = "sam3"))]
use nosam::Sam;

#[cfg(feature = "mobile-sam")]
use mobile::MobileSam as Mobile;

/// Built without `mobile-sam`: the engine is never available.
#[cfg(not(feature = "mobile-sam"))]
mod nomobile {
    use std::path::Path;

    use imagelab_api::{ApiError, ApiResult, Ctx};
    use serde_json::{Value, json};

    use crate::input::Input;
    use crate::segment::{Click, Pred};

    const WHY: &str = "this build has no MobileSAM (feature `mobile-sam` off)";

    #[derive(Default)]
    pub struct Mobile;

    impl Mobile {
        pub fn installed(_: &Path) -> bool {
            false
        }
        pub fn segment(&self, _: &Path, _: &Input, _: &[Click], _: Option<[f32; 4]>) -> Result<(Pred, bool), String> {
            Err(WHY.into())
        }
        pub fn prepare(&self, _: &Path, _: &Input) -> Result<bool, String> {
            Err(WHY.into())
        }
        pub fn status(&self, _: &Path) -> Value {
            json!({"state": "unavailable", "reason": WHY})
        }
        pub fn install(&self, _: &Ctx, _: &Path) -> ApiResult<Value> {
            Err(ApiError::unavailable(WHY))
        }
        pub fn unload(&self) -> bool {
            false
        }
    }
}
#[cfg(not(feature = "mobile-sam"))]
use nomobile::Mobile;
