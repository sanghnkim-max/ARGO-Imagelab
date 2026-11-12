//! Area `edit`: allowlisted engine commands, adjustment layers, auto fixes, geometry, recipes.

use std::path::Path;

use imagelab_api::{ApiError, ApiResult, Ctx, Effects, Registry};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::doc::{mutate, resolve_open};
use crate::ops::{self, AdjustParams, AutoParams, CropParams, RunParams, Step, StraightenParams, MAX_STEPS};
use crate::{allow, paths};

const MASK_PATTERN: &str = "Masks: adjustment layers and filters apply to the active selection, so to edit a region make it the selection first: `await lab.select.fromMask({mask}); await lab.edit.adjust({...})` (or a PhotoCraft selection such as select.subject / select.sky via edit.run). Deselect with edit.run({id:'select.deselect'}).";

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct CommandsParams {
    #[serde(default)]
    filter: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
}

fn commands(p: CommandsParams) -> Value {
    let filtered = p.filter.as_deref().is_some_and(|f| !f.trim().is_empty());
    let all = allow::entries(p.filter.as_deref());
    // No filter and no limit: an index of the groups (a few hundred tokens), not hundreds of
    // full rows; the caller then searches one group with `filter`.
    if !filtered && p.limit.is_none() {
        // area → second segment → count, e.g. image → {adjustments: 41, autoTone: 1}.
        let mut groups: std::collections::BTreeMap<String, std::collections::BTreeMap<String, usize>> = std::collections::BTreeMap::new();
        for e in &all {
            let Some(id) = e.get("id").and_then(Value::as_str) else { continue };
            let mut parts = id.split('.');
            let area = parts.next().unwrap_or_default().to_string();
            let sub = parts.next().unwrap_or_default().to_string();
            *groups.entry(area).or_default().entry(sub).or_default() += 1;
        }
        let index: serde_json::Map<String, Value> = groups
            .into_iter()
            .map(|(area, subs)| {
                let names: Vec<String> = subs.into_iter().map(|(sub, n)| if n > 1 { format!("{sub}({n})") } else { sub }).collect();
                (area, Value::String(names.join(" ")))
            })
            .collect();
        return json!({
            "total": all.len(),
            "groups": index,
            "hint": "Groups by area (name(count)). Search with filter (\"image.adjustments\", \"filter.blur\", or a word like \"curves\", \"vibrance\") for the ids and their params."
        });
    }
    let limit = p.limit.unwrap_or(40).clamp(1, 1000);
    json!({"total": all.len(), "allowlisted": allow::len(), "commands": all.into_iter().take(limit).collect::<Vec<_>>()})
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StepsParams {
    steps: Vec<Step>,
    #[serde(default = "yes")]
    atomic: bool,
}

fn yes() -> bool {
    true
}

// ---------------------------------------------------------------- recipes

/// A stored recipe: `recipes/<name>.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Recipe {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub steps: Vec<Step>,
}

/// Largest recipe file read.
const MAX_RECIPE_BYTES: u64 = 1 << 20;
/// Most recipes listed.
const MAX_RECIPES_LISTED: usize = 500;

fn recipe_path(root: &Path, name: &str) -> ApiResult<std::path::PathBuf> {
    let name = paths::recipe_name(name)?;
    Ok(root.join(paths::RECIPES_DIR).join(format!("{name}.json")))
}

/// Read a recipe by name.
pub fn load_recipe(root: &Path, name: &str) -> ApiResult<Recipe> {
    let path = recipe_path(root, name)?;
    let meta = std::fs::metadata(&path).map_err(|_| ApiError::not_found(format!("no recipe `{name}`; see edit.listRecipes")))?;
    if !meta.is_file() || meta.len() > MAX_RECIPE_BYTES {
        return Err(ApiError::too_large(format!("recipe `{name}` is not a file under {MAX_RECIPE_BYTES} bytes")));
    }
    let bytes = std::fs::read(&path).map_err(|e| ApiError::internal(format!("read recipe: {e}")))?;
    serde_json::from_slice(&bytes).map_err(|e| ApiError::new("corrupt", format!("recipe `{name}` is not valid: {e}")))
}

fn validate_steps(steps: &[Step]) -> ApiResult<()> {
    if steps.is_empty() || steps.len() > MAX_STEPS {
        return Err(ApiError::bad_params(format!("a recipe has 1..{MAX_STEPS} steps")));
    }
    for (i, s) in steps.iter().enumerate() {
        ops::validate_step(s).map_err(|e| ApiError::new(&e.code, format!("step {i}: {}", e.message)))?;
    }
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SaveRecipeParams {
    name: String,
    steps: Vec<Step>,
    #[serde(default)]
    description: Option<String>,
    /// Replace an existing recipe of that name.
    #[serde(default)]
    replace: bool,
}

fn save_recipe(ctx: &Ctx, p: SaveRecipeParams) -> ApiResult<Value> {
    let name = paths::recipe_name(&p.name)?;
    validate_steps(&p.steps)?;
    let path = recipe_path(&ctx.shared.root, &name)?;
    if path.exists() && !p.replace {
        return Err(ApiError::new("exists", format!("recipe `{name}` exists; pass replace: true or pick another name")));
    }
    std::fs::create_dir_all(ctx.shared.root.join(paths::RECIPES_DIR)).map_err(|e| ApiError::internal(format!("create recipes/: {e}")))?;
    let recipe = Recipe { name: name.clone(), description: p.description.map(|d| d.chars().take(2000).collect()), steps: p.steps };
    let bytes = serde_json::to_vec_pretty(&recipe).map_err(|e| ApiError::internal(e.to_string()))?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &bytes).and_then(|()| std::fs::rename(&tmp, &path)).map_err(|e| ApiError::internal(format!("write recipe: {e}")))?;
    Ok(json!({"name": name, "path": format!("{}/{name}.json", paths::RECIPES_DIR), "steps": recipe.steps.len()}))
}

fn list_recipes(ctx: &Ctx) -> ApiResult<Value> {
    let dir = ctx.shared.root.join(paths::RECIPES_DIR);
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        let mut names: Vec<String> = entries
            .filter_map(Result::ok)
            .filter_map(|e| e.file_name().to_str().and_then(|n| n.strip_suffix(".json")).map(str::to_string))
            .filter(|n| paths::recipe_name(n).is_ok())
            .collect();
        names.sort();
        for name in names.into_iter().take(MAX_RECIPES_LISTED) {
            match load_recipe(&ctx.shared.root, &name) {
                Ok(r) => out.push(json!({"name": r.name, "description": r.description, "steps": r.steps.len()})),
                Err(e) => out.push(json!({"name": name, "error": e.message})),
            }
        }
    }
    Ok(json!({"recipes": out}))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NameParams {
    name: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ApplyRecipeParams {
    name: String,
    /// Open this photo (its sidecar if any) first; otherwise the active document.
    #[serde(default)]
    photo: Option<String>,
}

fn apply_recipe(ctx: &Ctx, p: ApplyRecipeParams) -> ApiResult<Value> {
    let recipe = load_recipe(&ctx.shared.root, &p.name)?;
    let open = p.photo.as_deref().map(|photo| resolve_open(&ctx.shared.root, photo, false)).transpose()?;
    mutate(ctx, |h| {
        if let Some((file, _)) = &open {
            h.open(Path::new(file))?;
        }
        let results = ops::apply_steps(h, &recipe.steps, true)?;
        Ok(json!({"recipe": recipe.name, "applied": results.len(), "opened": open.as_ref().map(|o| o.0.clone())}))
    })
}

fn delete_recipe(ctx: &Ctx, p: NameParams) -> ApiResult<Value> {
    let path = recipe_path(&ctx.shared.root, &p.name)?;
    if !path.is_file() {
        return Err(ApiError::not_found(format!("no recipe `{}`", p.name)));
    }
    std::fs::remove_file(&path).map_err(|e| ApiError::internal(format!("delete recipe: {e}")))?;
    Ok(json!({"deleted": p.name}))
}

const STEP_SCHEMA: &str = r#"{"type":"object","description":"{id, params}: an allowlisted engine command; or {fn, params} with fn one of edit.run, edit.adjust, edit.auto, edit.straighten, edit.cropTo","properties":{"id":{"type":"string"},"fn":{"type":"string"},"params":{"type":"object"}}}"#;

fn step_schema() -> Value {
    serde_json::from_str(STEP_SCHEMA).unwrap_or_else(|_| json!({"type": "object"}))
}

pub fn register(r: &mut Registry) {
    r.function("edit", "commands", "Allowlisted PhotoCraft engine commands with their param docs")
        .description("Searches the engine commands edit.run may run (id, label or keywords contain `filter`; default limit 40), with each command's params. With neither filter nor limit: an index of the command groups (id prefix, count, example labels) to search next. The list is generated from PhotoCraft's registry: image.*, layer.*, filter.*, select.*, paint.* and in-document edit.* (undo/redo, fill, stroke, transform, content-aware); never file, export, preferences, plugin, script, print, view or tool-state commands.")
        .params(json!({"type":"object","properties":{"filter":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":1000}}}))
        .typed(|_: &Ctx, p: CommandsParams| Ok(commands(p)))
        .register();
    r.function("edit", "run", "Run one allowlisted PhotoCraft engine command on the active document")
        .description(&format!("{{id, params}}, e.g. {{id:'filter.blur.gaussianBlur', params:{{radius: 12}}}}. Ids outside the allowlist (file.*, preferences, plugins…) are refused with code `not_allowed`. {MASK_PATTERN}"))
        .params(json!({"type":"object","required":["id"],"properties":{"id":{"type":"string"},"params":{"type":"object"}}}))
        .effects(Effects::Revertible)
        .typed(|ctx: &Ctx, p: RunParams| mutate(ctx, |h| ops::run_allowed(h, &p.id, p.params)))
        .register();
    r.function("edit", "steps", "Run several steps; atomic (default) undoes them all if one fails")
        .description("{steps:[{id, params} | {fn, params}], atomic?: true}. At most 200 steps. edit.undo/redo cannot be steps. Returns each step's result.")
        .params(json!({"type":"object","required":["steps"],"properties":{"steps":{"type":"array","maxItems":MAX_STEPS,"items":step_schema()},"atomic":{"type":"boolean","default":true}}}))
        .effects(Effects::Revertible)
        .typed(|ctx: &Ctx, p: StepsParams| mutate(ctx, |h| Ok(json!({"results": ops::apply_steps(h, &p.steps, p.atomic)?}))))
        .register();
    r.function("edit", "adjust", "Add a non-destructive adjustment layer (masked by the active selection)")
        .description(&format!(
            "{{kind, params, name?, opacity? (0..1), mode?}}. kind: brightnessContrast, levels, curves, exposure, vibrance, hueSaturation, colorBalance, blackWhite, photoFilter, channelMixer, invert, posterize, threshold, gradientMap, selectiveColor, colorLookup (built-in luts only). params as layer.newAdjustmentLayer.<kind> takes them (see edit.commands({{filter:'newAdjustmentLayer'}})), e.g. {{kind:'exposure', params:{{exposure: 0.5}}}}, {{kind:'photoFilter', params:{{filter:'warming85', density: 30}}}}. Returns {{layer}}. {MASK_PATTERN}"
        ))
        .params(json!({"type":"object","required":["kind"],"properties":{"kind":{"type":"string"},"params":{"type":"object"},"name":{"type":"string"},"opacity":{"type":"number","minimum":0,"maximum":1},"mode":{"type":"string"}}}))
        .effects(Effects::Revertible)
        .typed(|ctx: &Ctx, p: AdjustParams| mutate(ctx, |h| ops::adjust(h, &p)))
        .register();
    r.function("edit", "auto", "Auto Tone / Contrast / Color on the active pixel layer")
        .description("{tone?, contrast?, color?}: with none set, Auto Tone. Works on the active pixel layer (select it first if an adjustment layer is active).")
        .params(json!({"type":"object","properties":{"tone":{"type":"boolean"},"contrast":{"type":"boolean"},"color":{"type":"boolean"}}}))
        .effects(Effects::Revertible)
        .typed(|ctx: &Ctx, p: AutoParams| mutate(ctx, |h| ops::auto(h, &p)))
        .register();
    r.function("edit", "straighten", "Rotate the canvas by `angle` degrees (clockwise +) and crop away the empty corners")
        .description("{angle: -45..45, crop?: true}. Measure the horizon first (cv.houghLines) and pass the negated tilt. The crop is the largest rectangle inside the rotated image.")
        .params(json!({"type":"object","required":["angle"],"properties":{"angle":{"type":"number","minimum":-45,"maximum":45},"crop":{"type":"boolean","default":true}}}))
        .effects(Effects::Revertible)
        .typed(|ctx: &Ctx, p: StraightenParams| mutate(ctx, |h| ops::straighten(h, &p)))
        .register();
    r.function("edit", "cropTo", "Crop to {x,y,w,h}, or to an aspect (\"4:5\", \"1:1\", \"16:9\") around the centre or the subject")
        .description("{x, y, w, h} in document px, or {aspect, anchor?: center|subject}. anchor subject centres on PhotoCraft's Select Subject (falls back to the centre).")
        .params(json!({"type":"object","properties":{"x":{"type":"integer"},"y":{"type":"integer"},"w":{"type":"integer"},"h":{"type":"integer"},"aspect":{"type":"string"},"anchor":{"type":"string","enum":["center","subject"]}}}))
        .effects(Effects::Revertible)
        .typed(|ctx: &Ctx, p: CropParams| mutate(ctx, |h| ops::crop_to(h, &p)))
        .register();
    r.function("edit", "looks", "One-slider looks and templates: ids, groups, ranges, styles")
        .description("The catalog edit.look takes: Essentials (enhance, sky, structure, details, denoise, bw, vignette), Landscape (goldenHour, foliage, atmosphere, skyReplace), Creative (dramatic, glow, matte, toning, grain), Portrait (faceLight, skin, eyes, redEye, bokeh) and templates that combine them.")
        .typed(|_: &Ctx, _: Value| Ok(crate::looks::catalog()))
        .register();
    r.function("edit", "look", "Apply a one-slider look or template as one history step")
        .description("{look, amount?, sky?, style?}. amount is the look's range (see edit.looks; default its default). enhance measures the photo and corrects exposure, shadows, highlights, contrast and colour; sky (0..100) with enhance also enhances the sky. style: toning colours, skyReplace sky, or the template id (look: 'template'). Sky looks need a sky (Select Sky) and portrait looks a face (image.detectFaces); when there is none the result says skipped and nothing changes. Returns {applied, steps, skipped}. Global looks apply inside the active selection; region looks (sky, faces, background) replace it and deselect.")
        .params(json!({"type":"object","required":["look"],"properties":{"look":{"type":"string"},"amount":{"type":"number"},"sky":{"type":"number","minimum":0,"maximum":100},"style":{"type":"string"}}}))
        .effects(Effects::Revertible)
        .typed(|ctx: &Ctx, p: crate::looks::LookParams| mutate(ctx, |h| crate::looks::apply(h, &p)))
        .register();
    r.function("edit", "lookSuggest", "The photo's scene (portrait, landscape, general) and templates that suit it")
        .description("{scene, faces, sky (share of the canvas), brightness (median), templates:[{id,label}]}. Leaves the document as it was.")
        .typed(|ctx: &Ctx, _: Value| ctx.with_session(crate::looks::suggest))
        .register();
    r.function("edit", "faceLiquify", "Face-aware liquify: eyes, nose, mouth, smile, face width, chin, forehead")
        .description("{eyes?, nose?, mouth?, smile?, faceWidth?, chin?, forehead?: -100..100 (0 = unchanged), face?: index (default every face)}. Strokes are placed and sized from image.detectFaces (box + eyes, nose, mouth corners). One Liquify history step. {applied:false, skipped} when no face is found.")
        .params(json!({"type":"object","properties":{"face":{"type":"integer","minimum":0},"eyes":{"type":"number"},"nose":{"type":"number"},"mouth":{"type":"number"},"smile":{"type":"number"},"faceWidth":{"type":"number"},"chin":{"type":"number"},"forehead":{"type":"number"}}}))
        .effects(Effects::Revertible)
        .typed(|ctx: &Ctx, p: crate::face::FaceParams| mutate(ctx, |h| crate::face::apply(h, &p)))
        .register();
    r.function("edit", "preview", "Preview an edit on a small proxy of the photo (the document is not changed)")
        .description("{begin?, maxSide? (256..2048, default 1024), step?: {fn: 'edit.look' | 'edit.faceLiquify' | 'edit.run', params}}. begin snapshots the active document (flattened) as the proxy; later calls reuse it and replace the previous preview. Returns {image: img:N, width, height, ms, result}. For sliders: preview while dragging, then apply with the real function. Pixel-unit params (blur radius) are not rescaled for the proxy.")
        .params(json!({"type":"object","properties":{"begin":{"type":"boolean"},"maxSide":{"type":"integer","minimum":256,"maximum":2048},"step":{"type":"object","properties":{"fn":{"type":"string"},"params":{"type":"object"}},"required":["fn"]}}}))
        .typed(crate::preview::preview)
        .register();
    r.function("edit", "saveRecipe", "Store steps as a named recipe (recipes/<name>.json)")
        .description("{name, steps, description?, replace?}. Steps as edit.steps takes them; validated against the allowlist when saved. Replay with edit.applyRecipe or batch.apply.")
        .params(json!({"type":"object","required":["name","steps"],"properties":{"name":{"type":"string"},"steps":{"type":"array","maxItems":MAX_STEPS,"items":step_schema()},"description":{"type":"string"},"replace":{"type":"boolean"}}}))
        .effects(Effects::Revertible)
        .typed(save_recipe)
        .register();
    r.function("edit", "listRecipes", "Stored recipes {recipes:[{name, description, steps}]}")
        .typed(|ctx: &Ctx, _: Value| list_recipes(ctx))
        .register();
    r.function("edit", "getRecipe", "One recipe with its steps")
        .params(json!({"type":"object","required":["name"],"properties":{"name":{"type":"string"}}}))
        .typed(|ctx: &Ctx, p: NameParams| load_recipe(&ctx.shared.root, &p.name))
        .register();
    r.function("edit", "applyRecipe", "Apply a recipe to the active document, or open `photo` first (atomic)")
        .params(json!({"type":"object","required":["name"],"properties":{"name":{"type":"string"},"photo":{"type":"string"}}}))
        .effects(Effects::Revertible)
        .typed(apply_recipe)
        .register();
    r.function("edit", "deleteRecipe", "Delete a stored recipe")
        .params(json!({"type":"object","required":["name"],"properties":{"name":{"type":"string"}}}))
        .effects(Effects::Destructive)
        .typed(delete_recipe)
        .register();
}
