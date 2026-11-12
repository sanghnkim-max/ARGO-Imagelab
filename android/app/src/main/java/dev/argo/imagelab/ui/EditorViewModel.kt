package dev.argo.imagelab.ui

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import dev.argo.imagelab.core.Lab
import android.net.Uri
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.geometry.Offset
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import dev.argo.imagelab.ImageLabApp
import dev.argo.imagelab.core.Fn
import dev.argo.imagelab.core.Gallery
import dev.argo.imagelab.core.Imports
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import java.io.File
import kotlin.math.roundToInt

/** How a tap or drag on the canvas selects (Select tool). */
enum class SelectMode(val label: String) { Object("Object"), Box("Box"), Rect("Rectangle"), Ellipse("Ellipse"), Lasso("Lasso"), Wand("Magic wand"), Color("Color") }

/** How a retouch brush uses the canvas. */
enum class BrushUse { Stroke, Source, Drag }

/**
 * Retouch brushes: a drag on the canvas is one stroke of the chosen engine tool. [BrushUse.Source]
 * brushes copy from a source point (tap to set it); [BrushUse.Drag] tools move the selection's
 * content by the drag.
 */
enum class RetouchBrush(val label: String, val command: String, val liquify: String? = null, val use: BrushUse = BrushUse.Stroke, val group: String = "Repair") {
    Heal("Heal", "paint.spotHealing"),
    Remove("Remove", "paint.remove"),
    Clone("Clone", "paint.cloneStamp", use = BrushUse.Source),
    HealFrom("Heal from", "paint.healingBrush", use = BrushUse.Source),
    Patch("Patch", "paint.patch", use = BrushUse.Drag),
    Move("Move", "paint.contentAwareMove", use = BrushUse.Drag),
    Dodge("Lighten", "paint.dodge", group = "Tone"),
    Burn("Darken", "paint.burn", group = "Tone"),
    Saturate("Saturate", "paint.sponge", group = "Tone"),
    Desaturate("Desaturate", "paint.sponge", group = "Tone"),
    Soften("Blur", "paint.blur", group = "Tone"),
    Crisp("Sharpen", "paint.sharpen", group = "Tone"),
    Smudge("Smudge", "paint.smudge", group = "Tone"),
    Warp("Push", "filter.liquify", "forwardWarp"),
    Bloat("Bloat", "filter.liquify", "bloat"),
    Pucker("Pinch", "filter.liquify", "pucker"),
    Restore("Restore", "filter.liquify", "reconstruct"),
}

/** How a new selection combines with the current one. */
enum class Combine(val label: String) { New("New"), Add("Add"), Subtract("Subtract") }

enum class Tool(val label: String) {
    Enhance("Enhance"), Layers("Layers"), Select("Select"), Adjust("Adjust"), Filters("Filters"), Effects("Effects"),
    Retouch("Retouch"), Paint("Paint"), Text("Text"), Transform("Transform"), Crop("Crop & size"), History("History"),
}

/** What a Paint drag does. */
enum class PaintMode(val label: String) { Brush("Brush"), Mixer("Mixer"), Gradient("Gradient"), Bucket("Bucket") }

/** On-canvas handles: four corners of a quad (perspective) or puppet pins. */
enum class HandleMode { None, Quad, Pins }

/** Export settings: format, quality and the longest side (null = full size). */
data class ExportSpec(val format: String = "jpg", val quality: Int = 92, val maxSide: Int? = null)

/**
 * The current selection: a registry mask handle (`mask:N`, from `segment.*`/`select.*`) with
 * its tinted overlay, or a PhotoCraft selection living in the document (no handle).
 */
data class Selection(val label: String, val mask: String?, val overlay: Bitmap?)

/**
 * Editor view state (tool, sheet, zoom/pan, selection) — kept across rotation and fold/unfold.
 * Every edit goes through [dev.argo.imagelab.core.LabApi.call] with registry ids ([Fn]),
 * the same path an agent step takes.
 */
class EditorViewModel(application: Application) : AndroidViewModel(application) {
    private val app = application as ImageLabApp
    val lab = app.lab

    var tool by mutableStateOf<Tool?>(null)
    var layerStripOpen by mutableStateOf(true)
    var scale by mutableFloatStateOf(1f)
    var offset by mutableStateOf(Offset.Zero)
    var selection by mutableStateOf<Selection?>(null); private set
    var selecting by mutableStateOf(false); private set

    /** Live adjustment: the photo follows the Adjust sliders (masked by the selection's handle). */
    val live = LiveAdjust(lab, viewModelScope) { selection?.mask }

    /** Live straighten: the photo rotates with the Crop sheet's slider. */
    val straightener = LiveStraighten(lab, viewModelScope)

    /** Last stylus pressure on the canvas (0..1): plumbed for the coming retouch brushes. */
    var stylusPressure by mutableFloatStateOf(0f)

    /**
     * Per-document state (selection mask, SAM prompt, clone source, handles, text spot, paint
     * layer) belongs to one document at one size: when the document or its size changes (another
     * photo, a crop or rotate from anywhere — undo, history, the agent), it is dropped.
     */
    private var docKey: String? = null

    init {
        viewModelScope.launch {
            lab.state.collect { st ->
                val key = "${st.documents}:${st.path ?: st.name}:${st.width}x${st.height}"
                if (docKey != null && key != docKey) resetDocumentState()
                docKey = key
            }
        }
    }

    private fun resetDocumentState() {
        selection = null
        samPoints.clear()
        cloneSource = null
        cloneOffset = null
        settingSource = false
        handles.clear()
        handleMode = HandleMode.None
        textAt = null
        paintLayerMade = false
        maskEnsuredFor = null
        before = null
        beforeFor = null
        // SAM's cached encodings belong to the previous photo (or size).
        viewModelScope.launch { lab.fn("segment.forgetImages", "{}", refreshAfter = false) }
    }

    /** Serialises selection round trips (fromMask → run, run → toMask → deselect), so two quick
     *  taps or a tap during a masked edit never read or clear each other's document selection. */
    private val selectionLock = kotlinx.coroutines.sync.Mutex()

    /** Any catalog item (filters, effects, light & color, size, layer properties), live. */
    val command = dev.argo.imagelab.ui.menu.LiveCommand(lab, viewModelScope) { selection?.mask }

    fun pickTool(t: Tool) {
        // Leaving a tool keeps what its open item did.
        if (command.item != null) command.apply()
        tool = if (tool == t) null else t
        if (tool == Tool.Select) onSelectOpened()
        if (tool == Tool.Text) refreshFonts()
        if (tool == Tool.Enhance) suggest()
        if (tool == Tool.History) refreshHistory()
        if (tool != Tool.Retouch) retouchBrush = null
        if (tool != Tool.Transform) handleMode = HandleMode.None
    }

    // ── enhance ──────────────────────────────────────────────────────────────────────────

    /** The photo's scene ("portrait", "landscape", "general") and the templates that suit it. */
    var scene by mutableStateOf<String?>(null); private set
    var suggested by mutableStateOf<List<String>>(emptyList()); private set
    private var suggestedFor: String? = null

    private fun suggest() = viewModelScope.launch {
        val st = lab.state.value
        val key = "${st.name}:${st.width}x${st.height}"
        if (key == suggestedFor) return@launch
        val r = lab.fn(Fn.EDIT_LOOK_SUGGEST, "{}", refreshAfter = false).getOrNull() as? JsonObject ?: return@launch
        scene = (r["scene"] as? JsonPrimitive)?.content
        suggested = (r["templates"] as? JsonArray)?.mapNotNull { ((it as? JsonObject)?.get("id") as? JsonPrimitive)?.content } ?: emptyList()
        suggestedFor = key
    }

    // ── retouch brushes ──────────────────────────────────────────────────────────────────

    /** The active retouch brush (null: one finger pans as usual). */
    var retouchBrush by mutableStateOf<RetouchBrush?>(null)
    var retouchSize by mutableFloatStateOf(80f)
    var retouchStrength by mutableFloatStateOf(0.5f)

    /** One retouch stroke in document pixels: [x, y, pressure]. */
    fun retouchStroke(points: List<Triple<Float, Float, Float>>) {
        val brush = retouchBrush ?: return
        if (points.isEmpty()) return
        when (brush.use) {
            BrushUse.Source -> if (settingSource || cloneSource == null) {
                cloneSource = Offset(points[0].first, points[0].second)
                cloneOffset = null
                settingSource = false
                lab.showMessage("Source set: now paint where it should go.")
                return
            }
            BrushUse.Drag -> return dragSelection(brush, points.first(), points.last())
            BrushUse.Stroke -> {}
        }
        val pts = JsonArray(points.map { (x, y, p) -> JsonArray(listOf(JsonPrimitive(x.roundToInt()), JsonPrimitive(y.roundToInt()), JsonPrimitive(p.coerceIn(0.05f, 1f).toDouble()))) })
        val strength = (retouchStrength * 100).roundToInt().coerceIn(1, 100)
        val params = buildJsonObject {
            if (brush.liquify != null) {
                putJsonArray("strokes") {
                    add(buildJsonObject {
                        put("tool", brush.liquify); put("size", retouchSize.roundToInt()); put("pressure", strength); put("rate", 60)
                        put("points", pts)
                    })
                }
            } else {
                put("points", pts); put("size", retouchSize.roundToInt())
                when (brush) {
                    RetouchBrush.Heal -> { put("hardness", 50); put("type", "contentAware") }
                    RetouchBrush.Remove -> put("hardness", 50)
                    RetouchBrush.Clone, RetouchBrush.HealFrom -> {
                        put("hardness", 50); put("aligned", true); put("opacity", strength)
                        // Aligned: the first stroke fixes the offset (source − first point); every
                        // later stroke keeps it, even one started before the first returns.
                        val src = cloneSource
                        val off = cloneOffset ?: src?.let { Offset(it.x - points[0].first, it.y - points[0].second) }?.also { cloneOffset = it }
                        if (off != null) putJsonArray("offset") { add(JsonPrimitive(off.x.roundToInt())); add(JsonPrimitive(off.y.roundToInt())) }
                    }
                    RetouchBrush.Saturate, RetouchBrush.Desaturate -> { put("mode", if (brush == RetouchBrush.Saturate) "saturate" else "desaturate"); put("flow", strength); put("hardness", 0) }
                    RetouchBrush.Soften, RetouchBrush.Crisp, RetouchBrush.Smudge -> { put("strength", strength); put("hardness", 0) }
                    else -> { put("exposure", strength); put("range", "midtones"); put("hardness", 0) }
                }
            }
        }
        viewModelScope.launch {
            lab.markInteractive()
            lab.fn(Fn.EDIT_RUN, buildJsonObject { put("id", brush.command); put("params", params) }.toString())
        }
    }

    /** Clone / Heal from: where pixels are copied from (document px); the next tap sets it while [settingSource]. */
    var cloneSource by mutableStateOf<Offset?>(null)
    private var cloneOffset: Offset? = null
    var settingSource by mutableStateOf(false)

    /** Patch / Move: drag the selection's content from [a] to [b] (content-aware). */
    private fun dragSelection(brush: RetouchBrush, a: Triple<Float, Float, Float>, b: Triple<Float, Float, Float>) {
        val sel = selection ?: return lab.showMessage("Select the area first (Select tool), then drag it here.")
        val dx = (b.first - a.first).roundToInt()
        val dy = (b.second - a.second).roundToInt()
        if (dx == 0 && dy == 0) return lab.showMessage("Drag from the selection to where it should go.")
        viewModelScope.launch {
            lab.markInteractive()
            val params = if (brush == RetouchBrush.Patch) {
                // Patch: fill the selection with the content found where it was dragged to.
                """{"offset":[$dx,$dy],"contentAware":true}"""
            } else {
                """{"offset":[$dx,$dy],"mode":"move"}"""
            }
            val r = editNow(brush.command, params, sel.mask)
            if (r.isFailure) { editNow("select.deselect", "{}", null); return@launch }
            // The engine moved (or used up) the selection; the mask handle no longer matches it.
            selection = null
            editNow("select.deselect", "{}", null)
            if (r.isSuccess) lab.showMessage(if (brush == RetouchBrush.Patch) "Patched from the dragged-to area." else "Moved; the gap was filled.")
        }
    }

    // ── segmentation model (MobileSAM) ───────────────────────────────────────────────────

    /** "ready" | "missing" | "downloading" | "unavailable" | null (not checked yet). */
    var segModel by mutableStateOf<String?>(null); private set
    var segProgress by mutableFloatStateOf(0f); private set
    /** True while the photo is being encoded ahead of taps. */
    var segPreparing by mutableStateOf(false); private set
    private var segInfo: JsonObject? = null

    /** Opening Select: check the model and, if installed, encode the photo now so taps only decode. */
    fun onSelectOpened() = viewModelScope.launch {
        refreshSegModel()
        if (segModel == "ready") prepareSegmentation()
    }

    private suspend fun refreshSegModel() {
        val models = lab.fn("segment.models", "{}", refreshAfter = false).getOrNull() as? JsonObject
        segInfo = models?.get("mobilesam") as? JsonObject
        if (segModel != "downloading") segModel = (segInfo?.get("state") as? JsonPrimitive)?.contentOrNull ?: "unavailable"
    }

    private suspend fun prepareSegmentation() {
        if (lab.state.value.documents == 0) return
        segPreparing = true
        lab.fn("segment.prepare", "{}", refreshAfter = false)
        segPreparing = false
    }

    /** Download MobileSAM (public, Apache-2.0) with the platform's TLS, then install it (size + SHA-256 checked by the core). */
    fun downloadSegmentModel() = viewModelScope.launch {
        if (segModel == "downloading") return@launch
        if (segInfo == null) refreshSegModel()
        val info = segInfo ?: return@launch lab.showMessage("The segmentation model is not available in this build.")
        val url = (info["url"] as? JsonPrimitive)?.contentOrNull ?: return@launch
        val file = (info["file"] as? JsonPrimitive)?.contentOrNull ?: "mobile_sam.safetensors"
        val rel = "downloads/$file"
        val dest = File(app.workspace, rel)
        segModel = "downloading"
        segProgress = 0f
        val ok = withContext(Dispatchers.IO) {
            runCatching {
                dest.parentFile?.mkdirs()
                val client = okhttp3.OkHttpClient()
                client.newCall(okhttp3.Request.Builder().url(url).build()).execute().use { resp ->
                    check(resp.isSuccessful) { "HTTP ${resp.code}" }
                    val body = resp.body
                    val total = body.contentLength().takeIf { it > 0 } ?: 1L
                    body.byteStream().use { input ->
                        dest.outputStream().use { out ->
                            val buf = ByteArray(1 shl 16)
                            var done = 0L
                            while (true) {
                                val n = input.read(buf)
                                if (n < 0) break
                                out.write(buf, 0, n)
                                done += n
                                segProgress = (done.toFloat() / total).coerceIn(0f, 1f)
                            }
                        }
                    }
                }
            }.onFailure { t -> lab.showMessage("Download failed: ${t.message}") }.isSuccess
        }
        if (ok) {
            val installed = lab.fn("segment.installModel", buildJsonObject { put("model", "mobilesam"); put("path", rel) }.toString(), refreshAfter = false)
            withContext(Dispatchers.IO) { dest.delete() }
            segModel = if (installed.isSuccess) "ready" else "missing"
            if (installed.isSuccess) prepareSegmentation()
        } else {
            withContext(Dispatchers.IO) { dest.delete() }
            segModel = "missing"
        }
    }

    fun resetView() {
        scale = 1f
        offset = Offset.Zero
    }

    // ── editing ──────────────────────────────────────────────────────────────────────────

    /** One engine command through `edit.run`, masked by the current selection's handle when [masked]. */
    fun edit(id: String, params: String = "{}", masked: Boolean = false, mask: String? = null) {
        viewModelScope.launch { editNow(id, params, if (masked) selection?.mask else mask) }
    }

    private suspend fun editNow(id: String, params: String, mask: String?): Result<JsonElement?> = selectionLock.withLock { editUnlocked(id, params, mask) }

    private suspend fun editUnlocked(id: String, params: String, mask: String?): Result<JsonElement?> {
        // A mask becomes the document selection first; PhotoCraft's filters and new adjustment
        // layers then apply to it (the selection becomes the adjustment layer's mask).
        if (mask != null) {
            val loaded = lab.fn(Fn.SELECT_FROM_MASK, buildJsonObject { put("mask", mask) }.toString(), refreshAfter = false)
            if (loaded.isFailure) return loaded
        }
        val body = buildJsonObject {
            put("id", id)
            put("params", lab.parse(params) ?: JsonObject(emptyMap()))
        }
        return lab.fn(Fn.EDIT_RUN, body.toString())
    }

    fun undo() = viewModelScope.launch { lab.call { it.undo() } }
    fun redo() = viewModelScope.launch { lab.call { it.redo() } }

    fun newCanvas() = viewModelScope.launch {
        lab.call { it.newDocument("""{"width":1080,"height":1350,"background":"white","name":"Untitled"}""") }
    }

    fun addPhotoLayer(uri: Uri) = viewModelScope.launch {
        val picked = Imports.read(app, uri) ?: return@launch lab.showMessage("Cannot read the picked image.")
        val rel = Imports.store(app.workspace, picked.first, picked.second)
        // Decode into a handle, then place it as a new layer of the open document.
        val read = lab.fn(Fn.CV_READ, buildJsonObject { put("path", rel) }.toString(), refreshAfter = false).getOrNull()
        val img = ((read as? JsonObject)?.get("img") as? kotlinx.serialization.json.JsonPrimitive)?.content
            ?: return@launch lab.showMessage("Cannot decode the picked image.")
        lab.fn(Fn.CV_TO_LAYER, buildJsonObject { put("img", img); put("name", picked.first) }.toString())
    }

    fun addText() {
        val s = lab.state.value
        val w = s.width.coerceAtLeast(100)
        val h = s.height.coerceAtLeast(100)
        edit("type.create", """{"x":${w / 10},"y":${h / 2},"text":"Text","size":${(h / 12).coerceAtLeast(12)}}""")
    }

    fun addSolidFill() = edit("layer.newFillLayer.solidColor", """{"color":"#3D7BF5"}""", masked = true)
    fun addGradientFill() = edit("layer.newFillLayer.gradient", """{"from":"#000000","to":"#ffffff","angle":90}""", masked = true)

    /** Save into exports/ with [exportSpec] (format, quality, size) and hand a copy to the gallery (Pictures/ImageLab). */
    fun export() = viewModelScope.launch {
        exportOpen = false
        val spec = exportSpec
        val stamp = System.currentTimeMillis()
        val ext = spec.format
        val rel = "exports/imagelab-$stamp.$ext"
        val body = buildJsonObject {
            put("path", rel); put("format", ext)
            if (ext == "jpg" || ext == "webp") put("quality", spec.quality)
            spec.maxSide?.let { put("maxSide", it) }
        }
        val saved = lab.fn(Fn.DOC_EXPORT, body.toString(), refreshAfter = false)
        if (saved.isFailure) return@launch
        val copied = withContext(Dispatchers.IO) {
            runCatching { Gallery.save(app, File(app.workspace, rel), "imagelab-$stamp.$ext") }
        }
        lab.showMessage(if (copied.isSuccess) "Exported to Pictures/ImageLab" else "Saved to $rel")
    }

    // ── selection (docs/INTERFACES.md §6, §10) ───────────────────────────────────────────

    fun selectSubject() = select("Subject", Fn.SELECT_SUBJECT, "{}")
    fun selectSky() = select("Sky", Fn.SELECT_SKY, "{}")

    // ── selection modes ──────────────────────────────────────────────────────────────────

    var selectMode by mutableStateOf(SelectMode.Object)
    var combine by mutableStateOf(Combine.New)

    /** SAM's detail pass (on by default): re-segment a full-resolution crop around the object
     *  (sharper edges on large photos, ~0.2 s more on a phone GPU). */
    var detail by mutableStateOf(true)

    /** " · detail ×3.2" when the detail pass ran, else "". */
    private fun detailNote(result: JsonElement?): String {
        val d = (result as? JsonObject)?.get("detail") as? JsonObject ?: return ""
        val refined = (d["refined"] as? JsonPrimitive)?.booleanOrNull == true
        if (!refined) {
            // On by default, so a skip for a large object (nothing to gain) stays quiet; say so
            // only when the detailed look disagreed or failed.
            val gainOnly = d["detail"] != null && d["agreement"] == null
            if (!gainOnly) (d["reason"] as? JsonPrimitive)?.contentOrNull?.let { lab.showMessage("Fine detail kept the first mask: $it") }
            return ""
        }
        val gain = (d["detail"] as? JsonPrimitive)?.doubleOrNull ?: return " · detail"
        return " · detail ×${"%.1f".format(gain)}"
    }

    /** The SAM prompt behind the current Object selection: [x, y, label] in document pixels. */
    private val samPoints = mutableListOf<Triple<Int, Int, Int>>()

    /** Canvas tap (document pixels). Object mode: a SAM point; New starts over, Add/Subtract
     *  add an include/exclude point to the same prompt (only the decoder runs again). */
    fun canvasTap(x: Float, y: Float) {
        if (selectMode == SelectMode.Wand || selectMode == SelectMode.Color) return colorTap(x.roundToInt(), y.roundToInt())
        if (selectMode != SelectMode.Object) return
        // Add / Subtract onto a selection that is not this SAM prompt (a rectangle, the subject,
        // a wand area): segment the tapped object alone and combine its mask with the selection.
        if (combine != Combine.New && samPoints.isEmpty() && selection?.mask != null) {
            val seq = ++tapSeq
            viewModelScope.launch {
                selecting = true
                val params = buildJsonObject {
                    putJsonArray("points") { add(JsonArray(listOf(JsonPrimitive(x.roundToInt()), JsonPrimitive(y.roundToInt()), JsonPrimitive(1)))) }
                    if (detail) put("refine", true)
                }
                val r = lab.fn(Fn.SEGMENT_POINT, params.toString(), refreshAfter = false).getOrNull()
                if (seq == tapSeq) combineWith(r, "Object")
                selecting = false
            }
            return
        }
        val p = Triple(x.roundToInt(), y.roundToInt(), if (combine == Combine.Subtract) 0 else 1)
        when (combine) {
            Combine.New -> { samPoints.clear(); samPoints += p }
            Combine.Add -> samPoints += p
            Combine.Subtract -> {
                if (samPoints.none { it.third == 1 }) return lab.showMessage("Tap the object first, then subtract parts of it.")
                samPoints += p
            }
        }
        val pts = samPoints.toList()
        val label = if (pts.size == 1) "Object" else "Object (${pts.count { it.third == 1 }}+ ${pts.count { it.third == 0 }}−)"
        viewModelScope.launch {
            selecting = true
            val params = buildJsonObject {
                putJsonArray("points") { pts.forEach { (px, py, l) -> add(JsonArray(listOf(JsonPrimitive(px), JsonPrimitive(py), JsonPrimitive(l)))) } }
                if (detail) put("refine", true)
            }
            val seq = ++tapSeq
            lab.fn(Fn.SEGMENT_POINT, params.toString(), refreshAfter = false).onSuccess { result -> if (seq == tapSeq) setSelection(label + detailNote(result), result) }
            selecting = false
        }
    }

    /** Newest selection tap; an older tap's result arriving late is dropped. */
    private var tapSeq = 0

    /** Canvas drag (document pixels): a box (SAM), rectangle, ellipse or freehand lasso. */
    fun canvasDrag(path: List<androidx.compose.ui.geometry.Offset>) {
        if (path.size < 2) return
        val xs = path.map { it.x }
        val ys = path.map { it.y }
        val x = xs.min().roundToInt()
        val y = ys.min().roundToInt()
        val w = (xs.max() - xs.min()).roundToInt().coerceAtLeast(1)
        val h = (ys.max() - ys.min()).roundToInt().coerceAtLeast(1)
        val mode = selectMode
        viewModelScope.launch {
            selecting = true
            var boxDetail = ""
            val fresh: JsonElement? = when (mode) {
                SelectMode.Object, SelectMode.Box ->
                    lab.fn(
                        Fn.SEGMENT_BOX,
                        buildJsonObject {
                            putJsonArray("box") { listOf(x, y, w, h).forEach { add(JsonPrimitive(it)) } }
                            if (detail) put("refine", true)
                        }.toString(),
                        refreshAfter = false,
                    ).getOrNull().also { boxDetail = detailNote(it) }
                SelectMode.Rect, SelectMode.Ellipse -> photocraftShape(
                    "select.rect",
                    buildJsonObject { put("x", x); put("y", y); put("width", w); put("height", h); put("ellipse", mode == SelectMode.Ellipse) },
                )
                SelectMode.Lasso -> photocraftShape(
                    "select.lasso",
                    buildJsonObject { putJsonArray("points") { path.forEach { add(JsonArray(listOf(JsonPrimitive(it.x.roundToInt()), JsonPrimitive(it.y.roundToInt())))) } } },
                )
                SelectMode.Wand, SelectMode.Color -> null
            }
            combineWith(fresh, mode.label + boxDetail)
            selecting = false
        }
    }

    /** A new mask from a selection tool, combined with the current one per [combine]. */
    private suspend fun combineWith(fresh: JsonElement?, what: String) {
        val newMask = fresh?.let(::maskHandleIn) ?: return
        samPoints.clear()
        val old = selection?.mask
        val combined = if (old != null && combine != Combine.New) {
            val op = if (combine == Combine.Add) "union" else "subtract"
            lab.fn("cv.combineMasks", buildJsonObject { put("a", old); put("b", newMask); put("op", op) }.toString(), refreshAfter = false).getOrNull()
        } else fresh
        val label = if (combine == Combine.New || old == null) what else "${selection?.label} ${if (combine == Combine.Add) "+" else "−"} $what"
        setSelection(label, combined)
    }

    /** Magic wand / color range tolerance (0..255 wand, fuzziness for color). */
    var tolerance by mutableFloatStateOf(32f)

    /** Magic wand (touching pixels of a similar color) or color range (that color everywhere). */
    private fun colorTap(x: Int, y: Int) {
        val mode = selectMode
        viewModelScope.launch {
            selecting = true
            val fresh = if (mode == SelectMode.Wand) {
                photocraftShape("select.magicWand", buildJsonObject { put("x", x); put("y", y); put("tolerance", tolerance.roundToInt()); put("sampleAllLayers", true) })
            } else {
                photocraftShape("select.colorRange", buildJsonObject { putJsonArray("points") { add(JsonArray(listOf(JsonPrimitive(x), JsonPrimitive(y)))) }; put("fuzziness", tolerance.roundToInt().coerceAtMost(200)) })
            }
            combineWith(fresh, mode.label)
            selecting = false
        }
    }

    /** The in-focus area (PhotoCraft Focus Area). */
    fun selectFocus() = viewModelScope.launch {
        selecting = true
        combineWith(photocraftShape("select.focusArea", buildJsonObject { put("range", 0.5) }), "In focus")
        selecting = false
    }

    fun selectAll() = viewModelScope.launch {
        combine = Combine.New
        combineWith(photocraftShape("select.all", JsonObject(emptyMap())), "All")
    }

    fun invertSelection() = viewModelScope.launch {
        val sel = selection ?: return@launch
        val mask = sel.mask ?: return@launch edit("select.inverse")
        val inv = lab.fn(Fn.CV_INVERT_MASK, buildJsonObject { put("mask", mask) }.toString(), refreshAfter = false).getOrNull()
        setSelection("${sel.label} (inverted)", inv)
    }

    /** `select.modify.<op>` (feather, expand, contract, smooth, border) or `select.similar` on the current selection. */
    fun modifySelection(op: String, amount: Float) = viewModelScope.launch {
        val sel = selection ?: return@launch
        selecting = true
        val (id, params) = if (op == "similar") "select.similar" to buildJsonObject { put("tolerance", amount.roundToInt()); put("sampleAllLayers", true) }
        else "select.modify.$op" to buildJsonObject { put("radius", amount.roundToInt().coerceAtLeast(1)) }
        val r = photocraftShape(id, params, loadMask = sel.mask)
        val what = mapOf("feather" to "feathered", "expand" to "expanded", "contract" to "contracted", "smooth" to "smoothed", "border" to "border", "similar" to "+ similar")[op] ?: op
        if (r != null) setSelection("${sel.label.substringBefore(" · ")} · $what", r)
        selecting = false
    }

    /** Cut the subject out: a layer mask from Select Subject (the Background becomes a layer). */
    fun removeBackground() = edit("layer.removeBackground")

    /** Run a PhotoCraft selection command, then read the selection back as a mask handle. The
     *  document selection is dropped again: the handle is the selection from here on (edits
     *  load it back), so a stale document selection never limits a later edit. */
    private suspend fun photocraftShape(id: String, params: JsonObject, loadMask: String? = null): JsonElement? = selectionLock.withLock {
        // [loadMask]: start from this selection (refining it).
        if (loadMask != null && lab.fn(Fn.SELECT_FROM_MASK, buildJsonObject { put("mask", loadMask) }.toString(), refreshAfter = false).isFailure) return@withLock null
        if (lab.fn(Fn.EDIT_RUN, buildJsonObject { put("id", id); put("params", params) }.toString(), refreshAfter = false).isFailure) return@withLock null
        val mask = lab.fn("select.toMask", "{}", refreshAfter = false).getOrNull()
        lab.fn(Fn.EDIT_RUN, """{"id":"select.deselect","params":{}}""", refreshAfter = false)
        mask
    }

    private suspend fun setSelection(label: String, result: JsonElement?) {
        val handle = result?.let(::maskHandleIn) ?: return
        val engine = ((result as? JsonObject)?.get("engine") as? JsonPrimitive)?.contentOrNull
        selection = Selection(if (engine != null) "$label · $engine" else label, handle, renderOverlay(handle))
    }

    fun selectConcept(text: String) {
        if (text.isBlank()) return
        select("“${text.trim()}”", Fn.SEGMENT_CONCEPT, buildJsonObject { put("prompt", text.trim()) }.toString())
    }

    private fun select(label: String, id: String, params: String) {
        viewModelScope.launch {
            selecting = true
            val r = lab.fn(id, params)
            r.onSuccess { result ->
                val handle = result?.let(::maskHandleIn)
                val overlay = handle?.let { renderOverlay(it) }
                val engine = ((result as? JsonObject)?.get("engine") as? JsonPrimitive)?.contentOrNull
                selection = Selection(if (engine != null) "$label · $engine" else label, handle, overlay)
            }
            selecting = false
        }
    }

    fun clearSelection() {
        val had = selection
        selection = null
        samPoints.clear()
        // Also drop any document selection (a masked edit loads the mask as one).
        if (had != null) edit("select.deselect")
    }

    fun adjustInside() {
        tool = Tool.Adjust
    }

    fun blurBackground() = viewModelScope.launch {
        val sel = selection ?: return@launch lab.showMessage("Select the subject first.")
        if (sel.mask != null) {
            val inverted = lab.fn(Fn.CV_INVERT_MASK, buildJsonObject { put("mask", sel.mask) }.toString(), refreshAfter = false)
                .getOrNull()?.let(::maskHandleIn) ?: return@launch
            editNow("filter.blur.gaussianBlur", """{"radius":12}""", inverted)
        } else {
            // A PhotoCraft selection: invert, blur, invert back.
            if (editNow("select.inverse", "{}", null).isFailure) return@launch
            editNow("filter.blur.gaussianBlur", """{"radius":12}""", null)
            editNow("select.inverse", "{}", null)
        }
    }

    fun removeSelected() {
        if (selection == null) return lab.showMessage("Select what to remove first.")
        edit("edit.contentAwareFill", "{}", masked = true)
    }

    fun cutToLayer() {
        if (selection == null) return lab.showMessage("Select something first.")
        edit("layer.new.layerViaCut", "{}", masked = true)
    }

    fun denoise() = viewModelScope.launch {
        val sel = selection?.mask
        if (sel == null) editNow("filter.noise.reduceNoise", """{"strength":6}""", null)
        else editNow("filter.noise.reduceNoise", """{"strength":6}""", sel)
    }

    // ── crop ─────────────────────────────────────────────────────────────────────────────

    /** Centered crop to [aspect] (w/h); null = keep the document's own proportions. */
    fun cropTo(aspect: Float?) {
        if (aspect == null) return
        // The engine centres the crop on the document's current size (never a stale one).
        val ratio = when {
            kotlin.math.abs(aspect - 1f) < 0.01f -> "1:1"
            else -> { val w = (aspect * 1000).roundToInt(); "$w:1000" }
        }
        viewModelScope.launch { lab.fn("edit.cropTo", """{"aspect":"$ratio"}""") }
    }

    fun straighten(angle: Float) {
        val s = lab.state.value
        if (s.width <= 0 || angle == 0f) return
        edit("image.crop", """{"x":0,"y":0,"width":${s.width},"height":${s.height},"angle":$angle,"contentAware":false}""")
    }

    // ── paint ────────────────────────────────────────────────────────────────────────────

    var paintColor by mutableStateOf("#ffffff")
    var paintSize by mutableFloatStateOf(24f)
    var paintOpacity by mutableFloatStateOf(1f)
    var paintHardness by mutableFloatStateOf(0.8f)
    var eraser by mutableStateOf(false)
    /** Paint on a new "Paint" layer (made on the first stroke) instead of the photo's pixels. */
    var paintOnNewLayer by mutableStateOf(true)
    private var paintLayerMade = false

    var paintMode by mutableStateOf(PaintMode.Brush)
    /** Paint on the active layer's mask (white reveals, black hides) instead of pixels. */
    var paintOnMask by mutableStateOf(false)
    var maskReveal by mutableStateOf(false)
    var gradientTo by mutableStateOf("transparent")
    var gradientStyle by mutableStateOf("linear")
    var bucketTolerance by mutableFloatStateOf(32f)
    var mixerWet by mutableFloatStateOf(50f)
    var mixerMix by mutableFloatStateOf(50f)
    var symmetry by mutableStateOf<String?>(null)
        private set

    fun setSymmetryMode(mode: String?) {
        symmetry = mode
        if (mode == null) edit("paint.symmetryDisable") else edit("paint.setSymmetry", """{"mode":"$mode"}""")
    }

    private fun maskColor() = if (maskReveal) "#ffffff" else "#000000"

    /** A mask to paint on: the active layer's, added (white) first when it has none. */
    private suspend fun ensureMask(): Boolean {
        val st = lab.state.value
        val active = st.layers.firstOrNull { it.id == st.activeLayer }
        // `layers` lags a stroke by up to a second: remember the layer a mask was added to, so a
        // second quick stroke never adds (and wipes) it again.
        if (active?.hasMask == true || (active?.id != null && active.id == maskEnsuredFor)) return true
        maskEnsuredFor = active?.id
        val ok = lab.fn(Fn.EDIT_RUN, """{"id":"layer.layerMask.revealAll","params":{}}""", refreshAfter = false).isSuccess
        if (!ok) maskEnsuredFor = null
        return ok
    }

    private var maskEnsuredFor: Long? = null

    /** One stroke in document pixels: [x, y, pressure]. */
    fun paintStroke(points: List<Triple<Float, Float, Float>>) {
        if (points.isEmpty()) return
        viewModelScope.launch {
            if (paintOnMask) {
                if (!ensureMask()) return@launch
            } else if (paintOnNewLayer && !eraser && !paintLayerMade && paintMode != PaintMode.Bucket) {
                paintLayerMade = true // before suspending: a second quick stroke must not make another layer
                val made = lab.fn(Fn.EDIT_RUN, buildJsonObject { put("id", "layer.new.layer"); put("params", buildJsonObject { put("name", "Paint") }) }.toString(), refreshAfter = false)
                if (made.isFailure) { paintLayerMade = false; return@launch }
            }
            val color = if (paintOnMask) maskColor() else paintColor
            val (a, b) = points.first() to points.last()
            val (id, params) = when (paintMode) {
                PaintMode.Gradient -> "paint.gradient" to buildJsonObject {
                    putJsonArray("from") { add(JsonPrimitive(a.first.roundToInt())); add(JsonPrimitive(a.second.roundToInt())) }
                    putJsonArray("to") { add(JsonPrimitive(b.first.roundToInt())); add(JsonPrimitive(b.second.roundToInt())) }
                    put("style", gradientStyle)
                    val end = if (paintOnMask) (if (maskReveal) "#000000" else "#ffffff") else if (gradientTo == "transparent") color else gradientTo
                    putJsonArray("stops") { add(JsonArray(listOf(JsonPrimitive(0), JsonPrimitive(color)))); add(JsonArray(listOf(JsonPrimitive(1), JsonPrimitive(end)))) }
                    if (!paintOnMask && gradientTo == "transparent") putJsonArray("transparency") { add(JsonArray(listOf(JsonPrimitive(0), JsonPrimitive(100)))); add(JsonArray(listOf(JsonPrimitive(1), JsonPrimitive(0)))) }
                    put("opacity", (paintOpacity * 100).roundToInt().coerceIn(1, 100))
                }
                PaintMode.Bucket -> "paint.bucket" to buildJsonObject {
                    put("x", a.first.roundToInt()); put("y", a.second.roundToInt())
                    put("tolerance", bucketTolerance.roundToInt()); put("color", color)
                    put("opacity", (paintOpacity * 100).roundToInt().coerceIn(1, 100))
                    put("sampleAllLayers", true)
                }
                PaintMode.Mixer -> "paint.mixerBrush" to buildJsonObject {
                    putJsonArray("points") { points.forEach { (x, y, p) -> add(JsonArray(listOf(JsonPrimitive(x.roundToInt()), JsonPrimitive(y.roundToInt()), JsonPrimitive(p.coerceIn(0.05f, 1f).toDouble())))) } }
                    put("size", paintSize.toDouble()); put("wet", mixerWet.roundToInt()); put("mix", mixerMix.roundToInt()); put("color", color)
                    put("sampleAllLayers", true)
                }
                PaintMode.Brush -> "paint.stroke" to buildJsonObject {
                    putJsonArray("points") { points.forEach { (x, y, p) -> add(JsonArray(listOf(JsonPrimitive(x.roundToInt()), JsonPrimitive(y.roundToInt()), JsonPrimitive(p.coerceIn(0.05f, 1f).toDouble())))) } }
                    put("size", paintSize.toDouble())
                    put("opacity", paintOpacity.toDouble())
                    put("hardness", paintHardness.toDouble())
                    put("color", color)
                    if (eraser && !paintOnMask) put("erase", true)
                }
            }
            val withTarget = if (paintOnMask) JsonObject(params + ("target" to JsonPrimitive("mask"))) else params
            lab.markInteractive()
            lab.fn(Fn.EDIT_RUN, buildJsonObject { put("id", id); put("params", withTarget) }.toString())
        }
    }

    /** The next stroke starts a new Paint layer. */
    fun newPaintLayer() {
        paintLayerMade = false
    }

    // ── history & snapshots ──────────────────────────────────────────────────────────────

    /** `doc.history`: `states` (past, current, redo), `current`, `snapshots`. */
    var history by mutableStateOf<JsonObject?>(null); private set

    fun refreshHistory() = viewModelScope.launch {
        history = lab.fn(Fn.DOC_HISTORY, "{}", refreshAfter = false).getOrNull() as? JsonObject
    }

    fun historyGo(state: Int) = viewModelScope.launch {
        if (command.item != null) command.apply()
        history = lab.fn(Fn.DOC_HISTORY_GO, """{"state":$state}""").getOrNull() as? JsonObject ?: history
    }

    fun snapshot(name: String) = viewModelScope.launch {
        val body = buildJsonObject { if (name.isNotBlank()) put("name", name.trim()) }
        val r = lab.fn(Fn.DOC_SNAPSHOT, body.toString(), refreshAfter = false)
        history = r.getOrNull() as? JsonObject ?: history
        if (r.isSuccess) lab.showMessage("Snapshot kept.")
    }

    fun restoreSnapshot(id: JsonElement) = viewModelScope.launch {
        history = lab.fn(Fn.DOC_RESTORE_SNAPSHOT, buildJsonObject { put("snapshot", id) }.toString()).getOrNull() as? JsonObject ?: history
    }

    fun deleteSnapshot(id: JsonElement) = viewModelScope.launch {
        history = lab.fn(Fn.DOC_DELETE_SNAPSHOT, buildJsonObject { put("snapshot", id) }.toString(), refreshAfter = false).getOrNull() as? JsonObject ?: history
    }

    // ── before / after ───────────────────────────────────────────────────────────────────

    /** The photo as opened, shown while the compare button is held. */
    var before by mutableStateOf<Bitmap?>(null); private set
    var comparing by mutableStateOf(false); private set
    private var beforeFor: String? = null

    fun compare(on: Boolean) {
        comparing = on
        if (!on) return
        val key = docKey
        if (before != null && beforeFor == key) return
        before = null // never show another document's original while this one loads
        viewModelScope.launch {
            val r = lab.fn(Fn.DOC_ORIGINAL, """{"maxSide":${Lab.PREVIEW_SIDE}}""", refreshAfter = false).getOrNull() as? JsonObject ?: return@launch
            val handle = (r["image"] as? JsonPrimitive)?.content ?: return@launch
            before = lab.handleBitmap(handle)
            beforeFor = key
        }
    }

    // ── transform handles (perspective crop / straighten, puppet warp) ───────────────────

    var handleMode by mutableStateOf(HandleMode.None)
    /** Handles in document px: (where it started, where it is now). */
    val handles = androidx.compose.runtime.mutableStateListOf<Pair<Offset, Offset>>()

    fun startQuad() {
        val st = lab.state.value
        val (w, h) = st.width.toFloat() to st.height.toFloat()
        handles.clear()
        listOf(Offset(w * 0.1f, h * 0.1f), Offset(w * 0.9f, h * 0.1f), Offset(w * 0.9f, h * 0.9f), Offset(w * 0.1f, h * 0.9f)).forEach { handles += it to it }
        handleMode = HandleMode.Quad
    }

    fun startPins() {
        handles.clear()
        handleMode = HandleMode.Pins
    }

    fun cancelHandles() {
        handles.clear()
        handleMode = HandleMode.None
    }

    private fun pt(o: Offset) = JsonArray(listOf(JsonPrimitive(o.x.roundToInt()), JsonPrimitive(o.y.roundToInt())))

    /** Crop to the quad, squared up (`image.perspectiveCrop`). */
    fun applyPerspectiveCrop() {
        if (handles.size != 4) return
        val corners = JsonArray(handles.map { pt(it.second) })
        viewModelScope.launch {
            if (editNow("image.perspectiveCrop", buildJsonObject { put("corners", corners) }.toString(), null).isSuccess) cancelHandles()
        }
    }

    /** Straighten what the quad outlines (a building, a page) into a rectangle, keeping the canvas (`edit.perspectiveWarp`). */
    fun applyPerspectiveStraighten() {
        if (handles.size != 4) return
        val q = handles.map { it.second }
        val left = (q[0].x + q[3].x) / 2; val right = (q[1].x + q[2].x) / 2
        val top = (q[0].y + q[1].y) / 2; val bottom = (q[2].y + q[3].y) / 2
        val dst = listOf(Offset(left, top), Offset(right, top), Offset(right, bottom), Offset(left, bottom))
        val body = buildJsonObject {
            putJsonArray("planes") { add(buildJsonObject { put("src", JsonArray(q.map(::pt))); put("dst", JsonArray(dst.map(::pt))) }) }
        }
        viewModelScope.launch {
            if (!movable()) return@launch
            if (editNow("edit.perspectiveWarp", body.toString(), null).isSuccess) cancelHandles()
        }
    }

    /** Puppet warp the active layer by its pins (pins left in place hold that part still). */
    fun applyPuppet() {
        if (handles.size < 2 || handles.none { it.first != it.second }) return lab.showMessage("Add pins, then drag one or more of them.")
        val body = buildJsonObject {
            putJsonArray("pins") { handles.forEach { (src, dst) -> add(buildJsonObject { put("src", pt(src)); put("dst", pt(dst)) }) } }
        }
        viewModelScope.launch {
            if (!movable()) return@launch
            if (editNow("edit.puppetWarp", body.toString(), null).isSuccess) cancelHandles()
        }
    }

    /** The active layer can move (not the locked Background): turns the Background into a layer first. */
    suspend fun movable(): Boolean {
        val st = lab.state.value
        val active = st.layers.firstOrNull { it.id == st.activeLayer }
        if (active?.name == "Background" && active.kind.equals("Pixel", true)) {
            return lab.fn(Fn.EDIT_RUN, """{"id":"layer.new.layerFromBackground","params":{}}""", refreshAfter = false).isSuccess
        }
        return true
    }

    fun makeMovable() = viewModelScope.launch { if (movable()) lab.requestRefresh() }

    // ── colour sampling (Replace color) ──────────────────────────────────────────────────

    /** True while a canvas tap samples a colour for the open item (Replace color). */
    val sampling get() = tool == Tool.Adjust && command.item?.command == "image.adjustments.replaceColor"

    fun sampleAt(x: Float, y: Float) {
        command.set("points", "json:[[${x.roundToInt()},${y.roundToInt()}]]")
    }

    // ── export ───────────────────────────────────────────────────────────────────────────

    var exportOpen by mutableStateOf(false)
    var exportSpec by mutableStateOf(ExportSpec())

    // ── text ─────────────────────────────────────────────────────────────────────────────

    var textContent by mutableStateOf("Text")
    var textSize by mutableFloatStateOf(72f)
    var textColor by mutableStateOf("#ffffff")
    /** Font family for new text (null: PhotoCraft's default). */
    var textFont by mutableStateOf<String?>(null)
    /** Families installed in this workspace (Google Fonts downloaded so far). */
    var installedFonts by mutableStateOf<List<String>>(emptyList()); private set
    var fontResults by mutableStateOf<List<dev.argo.imagelab.core.FontFamily>>(emptyList()); private set
    /** The family being downloaded, if any. */
    var fontDownloading by mutableStateOf<String?>(null); private set

    fun refreshFonts() = viewModelScope.launch {
        val r = lab.fn("fonts.list", "{}", refreshAfter = false).getOrNull() as? JsonObject ?: return@launch
        installedFonts = (r["installed"] as? JsonArray).orEmpty().flatMap { f ->
            ((f as? JsonObject)?.get("families") as? JsonArray).orEmpty().mapNotNull { (it as? JsonPrimitive)?.contentOrNull }.take(1)
        }.distinct()
    }

    fun searchFonts(query: String) = viewModelScope.launch {
        fontResults = withContext(Dispatchers.IO) { dev.argo.imagelab.core.GoogleFonts.search(query) }
        if (fontResults.isEmpty()) lab.showMessage("No Google Fonts match “$query” (or no connection).")
    }

    /** Use [family]; download and install it from Google Fonts first when needed. */
    fun pickFont(family: String?) = viewModelScope.launch {
        if (family == null || family in installedFonts) {
            textFont = family
            return@launch
        }
        fontDownloading = family
        val got = withContext(Dispatchers.IO) { dev.argo.imagelab.core.GoogleFonts.download(app.workspace, family) }
        got.onSuccess { (path, license) ->
            val r = lab.fn("fonts.install", buildJsonObject { put("path", path); put("source", "google"); put("license", license) }.toString(), refreshAfter = false)
            if (r.isSuccess) {
                val fam = ((r.getOrNull() as? JsonObject)?.get("families") as? JsonArray)?.firstOrNull()?.let { (it as? JsonPrimitive)?.contentOrNull } ?: family
                installedFonts = (installedFonts + fam).distinct()
                textFont = fam
            }
        }.onFailure { lab.showMessage("Font download failed: ${it.message}") }
        fontDownloading = null
    }

    /** Where the next text goes (document pixels); null: centred. */
    var textAt by mutableStateOf<androidx.compose.ui.geometry.Offset?>(null)

    fun placeText() {
        val s = lab.state.value
        val at = textAt ?: androidx.compose.ui.geometry.Offset(s.width / 2f, s.height / 2f)
        val params = buildJsonObject {
            put("x", at.x.roundToInt()); put("y", at.y.roundToInt())
            put("text", textContent.ifBlank { "Text" })
            put("size", textSize.toDouble()); put("color", textColor)
            textFont?.let { put("font", it) }
        }
        viewModelScope.launch { lab.fn(Fn.EDIT_RUN, buildJsonObject { put("id", "type.create"); put("params", params) }.toString()) }
    }

    /** Change the active text layer's words, size and colour. */
    fun updateText() = viewModelScope.launch {
        lab.fn(Fn.EDIT_RUN, buildJsonObject { put("id", "type.edit"); put("params", buildJsonObject { put("text", textContent) }) }.toString(), refreshAfter = false)
        lab.fn(Fn.EDIT_RUN, buildJsonObject { put("id", "type.setStyle"); put("params", buildJsonObject { put("size", textSize.toDouble()); put("color", textColor); textFont?.let { put("font", it) } }) }.toString())
    }

    // ── layers ───────────────────────────────────────────────────────────────────────────

    fun setLayerVisible(id: Long?, visible: Boolean) {
        if (id != null) edit("layer.setProps", buildJsonObject { put("layer", id); put("visible", visible) }.toString())
    }

    fun renameLayer(name: String) {
        if (name.isNotBlank()) edit("layer.renameLayer", buildJsonObject { put("name", name.trim()) }.toString())
    }

    /** The active layer's selection becomes its mask (or a full mask without a selection). */
    fun maskFromSelection() = viewModelScope.launch {
        val sel = selection
        if (sel?.mask != null) {
            lab.fn(Fn.SELECT_FROM_MASK, buildJsonObject { put("mask", sel.mask) }.toString(), refreshAfter = false)
            editNow("layer.layerMask.revealSelection", "{}", null)
        } else editNow("layer.layerMask.revealAll", "{}", null)
    }


    fun selectLayer(id: Long?) {
        if (id != null) edit("layer.select", """{"layer":$id}""")
    }

    // ── helpers ──────────────────────────────────────────────────────────────────────────

    private fun maskHandleIn(e: JsonElement): String? = when (e) {
        is JsonPrimitive -> e.contentOrNull?.takeIf { e.isString && it.startsWith("mask:") }
        is JsonObject -> (e["mask"]?.let(::maskHandleIn)) ?: e.values.firstNotNullOfOrNull(::maskHandleIn)
        is JsonArray -> e.firstNotNullOfOrNull(::maskHandleIn)
    }

    /** The mask as a translucent accent overlay (coverage → alpha). */
    private suspend fun renderOverlay(handle: String): Bitmap? {
        val png = lab.io { it.renderHandle(handle, OVERLAY_SIDE.toUInt()) }.getOrElse {
            lab.showMessage("Mask preview unavailable: ${dev.argo.imagelab.core.Lab.describe(it)}")
            return null
        }
        return withContext(Dispatchers.Default) {
            runCatching {
                val src = BitmapFactory.decodeByteArray(png, 0, png.size) ?: return@runCatching null
                val px = IntArray(src.width * src.height)
                src.getPixels(px, 0, src.width, 0, 0, src.width, src.height)
                for (i in px.indices) {
                    val c = px[i]
                    val coverage = ((c shr 16) and 0xFF) * ((c ushr 24) and 0xFF) / 255
                    val a = coverage * 115 / 255
                    px[i] = (a shl 24) or 0x3D7BF5
                }
                Bitmap.createBitmap(px, src.width, src.height, Bitmap.Config.ARGB_8888)
            }.getOrNull()
        }
    }

    companion object {
        private const val OVERLAY_SIDE = 1024
    }
}
