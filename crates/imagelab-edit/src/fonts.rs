//! The `fonts` area: fonts the workspace adds to PhotoCraft's text engine (Google Fonts and
//! any other TTF/OTF), so text layers — from the app or an agent — can use them by family.
//!
//! Network stays out of the core: the app downloads a font (Google Fonts through Fontsource's
//! catalog and the google/fonts repository, with the platform's TLS) into the workspace, then
//! `fonts.install` checks it is a font, files it under `fonts/` and registers it with PhotoCraft's
//! shared text engine (`photocraft_text::shared`). [`register_all`] re-registers every installed
//! font when the core starts, so a document reopened later finds its families.

use std::path::{Path, PathBuf};

use imagelab_api::{ApiError, ApiResult, Ctx, Effects, Registry};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::paths;

/// Folder (in the workspace) that holds installed fonts and their manifest.
pub const FONTS_DIR: &str = "fonts";
const MANIFEST: &str = "manifest.json";
/// Largest font file accepted (full CJK fonts are a few MB to ~20 MB).
const MAX_FONT_BYTES: u64 = 40 << 20;
const MAX_FONTS: usize = 200;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Manifest {
    fonts: Vec<Installed>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Installed {
    /// File name under `fonts/`.
    file: String,
    /// Families the file provides (from the font itself), best first.
    families: Vec<String>,
    /// Where it came from, as the installer said (e.g. "google").
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    license: Option<String>,
}

fn dir(root: &Path) -> PathBuf {
    root.join(FONTS_DIR)
}

fn read_manifest(root: &Path) -> Manifest {
    std::fs::read(dir(root).join(MANIFEST)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

fn write_manifest(root: &Path, m: &Manifest) -> ApiResult<()> {
    let d = dir(root);
    std::fs::create_dir_all(&d).map_err(|e| ApiError::new("disk", e.to_string()))?;
    let tmp = d.join(format!("{MANIFEST}.tmp"));
    let bytes = serde_json::to_vec_pretty(m).map_err(|e| ApiError::internal(e.to_string()))?;
    std::fs::write(&tmp, bytes).map_err(|e| ApiError::new("disk", e.to_string()))?;
    std::fs::rename(&tmp, d.join(MANIFEST)).map_err(|e| ApiError::new("disk", e.to_string()))
}

/// TrueType, OpenType (CFF), Apple 'true' or a collection.
fn looks_like_font(bytes: &[u8]) -> bool {
    matches!(bytes.get(..4), Some([0, 1, 0, 0]) | Some(b"OTTO") | Some(b"true") | Some(b"ttcf"))
}

fn register(bytes: Vec<u8>) -> Vec<String> {
    let engine = photocraft_text::shared();
    let mut e = engine.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    e.register_font_data(bytes)
}

/// Register every installed font with PhotoCraft's text engine (call once at startup).
/// Returns how many files registered.
pub fn register_all(root: &Path) -> usize {
    let m = read_manifest(root);
    let mut n = 0;
    for f in m.fonts.iter().take(MAX_FONTS) {
        let p = dir(root).join(&f.file);
        if let Ok(bytes) = std::fs::read(&p)
            && looks_like_font(&bytes)
            && !register(bytes).is_empty()
        {
            n += 1;
        }
    }
    n
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InstallParams {
    /// Workspace-relative font file (TTF/OTF/TTC), e.g. `downloads/DoHyeon-Regular.ttf`.
    path: String,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    license: Option<String>,
    /// Move the file into `fonts/` (default) instead of copying it.
    #[serde(default = "yes")]
    r#move: bool,
}

fn yes() -> bool {
    true
}

fn install(ctx: &Ctx, p: InstallParams) -> ApiResult<Value> {
    let rel = paths::relative(&p.path)?;
    let src = ctx.shared.root.join(&rel);
    // No symlink escapes: the file must stay inside the workspace.
    if let (Ok(c), Ok(r)) = (src.canonicalize(), ctx.shared.root.canonicalize())
        && !c.starts_with(&r)
    {
        return Err(ApiError::bad_params(format!("{} leads outside the workspace", p.path)));
    }
    let meta = std::fs::metadata(&src).map_err(|e| ApiError::not_found(format!("{}: {e}", p.path)))?;
    if !meta.is_file() || meta.len() > MAX_FONT_BYTES {
        return Err(ApiError::bad_params(format!("{} is not a font file under {} MB", p.path, MAX_FONT_BYTES >> 20)));
    }
    let bytes = std::fs::read(&src).map_err(|e| ApiError::new("disk", e.to_string()))?;
    if !looks_like_font(&bytes) {
        return Err(ApiError::bad_params(format!("{} is not a TrueType/OpenType font", p.path)));
    }
    let families = register(bytes.clone());
    if families.is_empty() {
        return Err(ApiError::bad_params(format!("{}: the font could not be read", p.path)));
    }
    let root = &ctx.shared.root;
    let mut m = read_manifest(root);
    if m.fonts.len() >= MAX_FONTS {
        return Err(ApiError::too_large(format!("at most {MAX_FONTS} fonts")));
    }
    let name = src.file_name().and_then(|n| n.to_str()).unwrap_or("font.ttf").replace(['/', '\\'], "_");
    let d = dir(root);
    std::fs::create_dir_all(&d).map_err(|e| ApiError::new("disk", e.to_string()))?;
    let dst = d.join(&name);
    if dst != src {
        let moved = p.r#move && std::fs::rename(&src, &dst).is_ok();
        if !moved {
            std::fs::write(&dst, &bytes).map_err(|e| ApiError::new("disk", e.to_string()))?;
        }
    }
    m.fonts.retain(|f| f.file != name);
    m.fonts.push(Installed { file: name.clone(), families: families.clone(), source: p.source, license: p.license });
    write_manifest(root, &m)?;
    Ok(json!({"families": families, "file": format!("{FONTS_DIR}/{name}")}))
}

fn list(ctx: &Ctx) -> ApiResult<Value> {
    let m = read_manifest(&ctx.shared.root);
    let installed: Vec<Value> = m.fonts.iter().map(|f| json!({"families": f.families, "file": format!("{FONTS_DIR}/{}", f.file), "source": f.source, "license": f.license})).collect();
    // Every family the text engine knows (system + bundled + installed), via PhotoCraft.
    let all = ctx.with_session(|h| Ok(h.command_run("type.fonts", json!({})).unwrap_or(Value::Null)))?;
    Ok(json!({"installed": installed, "available": all}))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Empty {}

pub fn register_area(r: &mut Registry) {
    r.area("fonts", "Fonts for text layers: installed families (Google Fonts the app downloaded, system, bundled); install a TTF/OTF");
    r.function("fonts", "list", "Installed and available font families")
        .description("Returns {installed: [{families, file, source, license}], available: PhotoCraft's type.fonts listing}. Use a family name as `font` in type.create / type.setStyle.")
        .params(json!({"type": "object", "properties": {}}))
        .effects(Effects::ReadOnly)
        .typed(|ctx, _: Empty| list(ctx))
        .register();
    r.function("fonts", "install", "Install a font file from the workspace (TTF/OTF/TTC) for text layers")
        .description("`path`: workspace-relative font file (the app downloads Google Fonts into downloads/). It is checked, moved into fonts/ and registered with the text engine; returns {families, file}. Installed fonts re-register when the app starts.")
        .params(json!({"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}, "source": {"type": "string"}, "license": {"type": "string"}, "move": {"type": "boolean"}}}))
        .effects(Effects::Revertible)
        .typed(install)
        .register();
}
