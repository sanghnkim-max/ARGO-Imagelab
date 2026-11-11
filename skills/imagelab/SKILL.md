---
name: imagelab
description: Edit photos in ImageLab (PhotoCraft's engine on Android) over MCP — inspect, run engine commands, render previews, export. Use when the imagelab MCP server is connected.
---

# Editing with ImageLab

ImageLab serves PhotoCraft's MCP tools over the document open on the phone. Whatever you do
shows up on the person's screen.

1. **Look first.** `session_list`, then `doc_inspect` (layers, history, selection). See the image
   with `doc_render_preview` and a small `max_side` (512–1024).
2. **Find commands.** `command_list` (filter by text, e.g. `blur`, `adjustment`). Each entry has
   an `id` and a compact `params` description with ranges and defaults.
3. **Edit.** `command_run {id, params}` or `command_batch` for several steps. Prefer adjustment
   layers (`layer.newAdjustmentLayer.brightnessContrast`, `.hueSaturation`, `.curves`, `.levels`,
   `.vibrance`, `.colorBalance`, `.blackWhite`, `.colorLookup`) over destructive
   `image.adjustments.*`. Filters: `filter.blur.gaussianBlur`, `filter.sharpen.unsharpMask`,
   `filter.noise.reduceNoise`. Undo with `edit.undo`.
4. **Experiment safely.** `image.duplicate` first; edit the copy.
5. **Files.** Relative paths in the app workspace: `imports/` holds photos the person opened
   (don't overwrite), save to `exports/` with `doc_save {path:"exports/name.jpg"}` (format by
   extension; `.pcraft` keeps layers).
6. **Check** one preview at the end and summarize what changed.
