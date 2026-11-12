//! The engine commands `edit.run` may run: generated from PhotoCraft's own registry
//! (`Headless::command_list`) at first use, so the list cannot drift from what runs.
//!
//! Rule: a command is allowed when its id starts with an editing prefix (`image.`, `layer.`,
//! `filter.`, `select.`, `paint.`, or one of the in-document `edit.*` families), contains none of
//! the words that mean "touches a file, the app's settings or its preset libraries", and
//! PhotoCraft's own automation gate (`authorize_engine_command`) accepts it with empty params.
//! Everything else (`file.*`, `edit.preferences.*`, `plugin.*`, `actions.*` scripts, print,
//! `view.*`, `tools.*`, `jobs.*`, `history.*` panels, smart objects and video layers that read
//! files, exports) is refused.

use std::collections::BTreeSet;
use std::sync::OnceLock;

use imagelab_api::Headless;
use serde_json::{Value, json};

/// Prefixes whose commands edit the open document.
// `type.` = text layers (create, edit, style) in the open document; `type.fonts` only lists fonts.
const PREFIXES: &[&str] = &["image.", "layer.", "filter.", "select.", "paint.", "type."];

/// `edit.*` families that edit the open document in memory (the rest of `edit.*` is settings,
/// preset libraries, purges, colour settings, keyboard shortcuts…).
const EDIT_FAMILIES: &[&str] = &[
    "edit.undo",
    "edit.redo",
    "edit.fill",
    "edit.stroke",
    "edit.clear",
    "edit.transform",
    "edit.contentAwareFill",
    "edit.contentAwareScale",
    "edit.deleteAndFillSelection",
    "edit.puppetWarp",
    "edit.perspectiveWarp",
    "edit.fade",
    "edit.cut",
    "edit.copy",
    "edit.paste",
    "edit.toggleLastState",
];

/// Words (lower case, matched anywhere in the id) that mean a command reaches outside the
/// document: files, exports, settings, preset libraries, scripts, printing.
const DENY_WORDS: &[&str] = &[
    "preferences",
    "plugin",
    "script",
    "export",
    "import",
    "save",
    "print",
    "relink",
    "linked",
    "replacecontents",
    "footage",
    "fromfile",
    "reloadframe",
    "presets",
    "keyboardshortcuts",
    "purge",
    "colorsettings",
    "installedprofiles",
    "copytodocument",
    "contours.load",
    "definebrushpreset",
    "definecustomshape",
    "definepattern",
];

/// Does the rule allow `id`? (Membership in the generated list is checked by [`is_allowed`].)
pub fn rule(id: &str) -> bool {
    let family = |f: &&str| id == *f || id.strip_prefix(*f).is_some_and(|rest| rest.starts_with('.') || rest.chars().next().is_some_and(char::is_uppercase));
    let prefixed = PREFIXES.iter().any(|p| id.starts_with(p)) || EDIT_FAMILIES.iter().any(family);
    if !prefixed {
        return false;
    }
    let lower = id.to_ascii_lowercase();
    if DENY_WORDS.iter().any(|w| lower.contains(w)) {
        return false;
    }
    photocraft_automation::workspace::authorize_engine_command(id, &json!({})).is_ok()
}

struct Allowlist {
    ids: BTreeSet<String>,
    /// The command list entries of allowed ids, `{id, label, menu, params}`.
    entries: Vec<Value>,
}

fn list() -> &'static Allowlist {
    static LIST: OnceLock<Allowlist> = OnceLock::new();
    LIST.get_or_init(|| {
        let all = Headless::new().command_list();
        let mut ids = BTreeSet::new();
        let mut entries = Vec::new();
        for entry in all.as_array().into_iter().flatten() {
            let Some(id) = entry.get("id").and_then(Value::as_str) else { continue };
            if rule(id) && ids.insert(id.to_string()) {
                entries.push(json!({
                    "id": id,
                    "label": entry.get("label").cloned().unwrap_or(Value::Null),
                    "menu": entry.get("menu").cloned().unwrap_or(Value::Null),
                    "params": entry.get("params").cloned().unwrap_or(Value::Null),
                    "keywords": entry.get("keywords").cloned().unwrap_or(Value::Null),
                }));
            }
        }
        Allowlist { ids, entries }
    })
}

/// Is `id` an allowlisted engine command?
pub fn is_allowed(id: &str) -> bool {
    list().ids.contains(id)
}

/// How many commands are allowlisted.
pub fn len() -> usize {
    list().ids.len()
}

/// Allowlisted entries whose id, label or keywords contain `filter` (case-insensitive);
/// `a|b` matches either.
pub fn entries(filter: Option<&str>) -> Vec<&'static Value> {
    let needles: Vec<String> = filter.map(|f| f.split('|').map(|n| n.trim().to_lowercase()).filter(|n| !n.is_empty()).collect()).unwrap_or_default();
    list()
        .entries
        .iter()
        .filter(|e| {
            needles.is_empty()
                || ["id", "label", "keywords"].iter().any(|k| e.get(*k).and_then(Value::as_str).is_some_and(|v| {
                    let v = v.to_lowercase();
                    needles.iter().any(|n| v.contains(n))
                }))
        })
        .collect()
}

/// Allowlisted ids starting with `prefix`.
pub fn ids_with_prefix(prefix: &str) -> Vec<&'static str> {
    list().ids.iter().filter(|id| id.starts_with(prefix)).map(String::as_str).collect()
}

/// Why `id` is refused, with a hint.
pub fn refusal(id: &str) -> String {
    let known = Headless::new().command_list().as_array().is_some_and(|a| a.iter().any(|e| e.get("id").and_then(Value::as_str) == Some(id)));
    if known {
        format!(
            "`{id}` is not an editing command ImageLab lets agents run (files, settings, presets, scripts and UI state are off limits). Use doc.open / doc.save / doc.export for files; see edit.commands for the {} allowed commands",
            len()
        )
    } else {
        format!("`{id}` is not a PhotoCraft command; search edit.commands({{filter}}) for the right id")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_allows_editing_and_refuses_files() {
        for ok in ["filter.blur.gaussianBlur", "image.imageSize", "layer.newAdjustmentLayer.curves", "edit.undo", "edit.transform.flipHorizontal", "select.all", "paint.stroke"] {
            assert!(is_allowed(ok), "{ok} should be allowed");
        }
        for bad in ["file.open", "file.save", "edit.preferences.general", "plugin.run", "layer.exportAs", "layer.smartObjects.relinkToFile", "edit.keyboardShortcuts", "view.zoomIn", "edit.purge.all"] {
            assert!(!is_allowed(bad), "{bad} should be refused");
        }
        assert!(len() > 200, "allowlist has {}", len());
        eprintln!("allowlist: {} commands", len());
    }
}
