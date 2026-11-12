package dev.argo.imagelab.ui

import android.app.Application
import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.provider.MediaStore
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.geometry.Offset
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import dev.argo.imagelab.ImageLabApp
import dev.argo.imagelab.core.Fn
import dev.argo.imagelab.core.Imports
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
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
enum class SelectMode(val label: String) { Object("Object"), Box("Box"), Rect("Rectangle"), Ellipse("Ellipse"), Lasso("Lasso") }

/** How a new selection combines with the current one. */
enum class Combine(val label: String) { New("New"), Add("Add"), Subtract("Subtract") }

enum class Tool(val label: String) {
    Layers("Layers"), Select("Select"), Adjust("Adjust"), Filters("Filters"), Effects("Effects"),
    Retouch("Retouch"), Paint("Paint"), Text("Text"), Crop("Crop & size"),
}

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

    /** Any catalog item (filters, effects, light & color, size, layer properties), live. */
    val command = dev.argo.imagelab.ui.menu.LiveCommand(lab, viewModelScope) { selection?.mask }

    fun pickTool(t: Tool) {
        // Leaving a tool keeps what its open item did.
        if (command.item != null) command.apply()
        tool = if (tool == t) null else t
        if (tool == Tool.Select) onSelectOpened()
        if (tool == Tool.Text) refreshFonts()
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

    private suspend fun editNow(id: String, params: String, mask: String?): Result<JsonElement?> {
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

    /** Save at full size into exports/ and hand a copy to the gallery (Pictures/ImageLab). */
    fun export() = viewModelScope.launch {
        val stamp = System.currentTimeMillis()
        val rel = "exports/imagelab-$stamp.jpg"
        val saved = lab.call(refreshAfter = false) { it.save(rel, null) }
        if (saved.isFailure) return@launch
        val copied = withContext(Dispatchers.IO) {
            runCatching {
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, "imagelab-$stamp.jpg")
                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/ImageLab")
                }
                val resolver = app.contentResolver
                val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: error("no gallery")
                resolver.openOutputStream(uri)?.use { out -> File(app.workspace, rel).inputStream().use { it.copyTo(out) } }
            }
        }
        lab.showMessage(if (copied.isSuccess) "Exported to Pictures/ImageLab" else "Saved to $rel")
    }

    // ── selection (docs/INTERFACES.md §6, §10) ───────────────────────────────────────────

    fun selectSubject() = select("Subject", Fn.SELECT_SUBJECT, "{}")
    fun selectSky() = select("Sky", Fn.SELECT_SKY, "{}")

    // ── selection modes ──────────────────────────────────────────────────────────────────

    var selectMode by mutableStateOf(SelectMode.Object)
    var combine by mutableStateOf(Combine.New)

    /** SAM's detail pass: re-segment a full-resolution crop around the object (sharper edges on
     *  large photos, ~0.2 s more on a phone GPU). */
    var detail by mutableStateOf(false)

    /** " · detail ×3.2" when the detail pass ran, else "". */
    private fun detailNote(result: JsonElement?): String {
        val d = (result as? JsonObject)?.get("detail") as? JsonObject ?: return ""
        val refined = (d["refined"] as? JsonPrimitive)?.booleanOrNull == true
        if (!refined) {
            (d["reason"] as? JsonPrimitive)?.contentOrNull?.let { lab.showMessage("Fine detail skipped: $it") }
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
        if (selectMode != SelectMode.Object) return
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
            lab.fn(Fn.SEGMENT_POINT, params.toString(), refreshAfter = false).onSuccess { result -> setSelection(label + detailNote(result), result) }
            selecting = false
        }
    }

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
            }
            val newMask = fresh?.let(::maskHandleIn)
            if (newMask != null) {
                samPoints.clear()
                val old = selection?.mask
                val combined = if (old != null && combine != Combine.New) {
                    val op = if (combine == Combine.Add) "union" else "subtract"
                    lab.fn("cv.combineMasks", buildJsonObject { put("a", old); put("b", newMask); put("op", op) }.toString(), refreshAfter = false).getOrNull()
                } else fresh
                val label = when {
                    combine == Combine.New || old == null -> mode.label + boxDetail
                    else -> "${selection?.label} ${if (combine == Combine.Add) "+" else "−"} ${mode.label}$boxDetail"
                }
                setSelection(label, combined)
            }
            selecting = false
        }
    }

    /** Run a PhotoCraft selection command, then read the selection back as a mask handle. */
    private suspend fun photocraftShape(id: String, params: JsonObject): JsonElement? {
        if (lab.fn(Fn.EDIT_RUN, buildJsonObject { put("id", id); put("params", params) }.toString(), refreshAfter = false).isFailure) return null
        return lab.fn("select.toMask", "{}", refreshAfter = false).getOrNull()
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
        if (had != null && had.mask == null) edit("select.deselect")
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
        val s = lab.state.value
        if (s.width <= 0 || s.height <= 0 || aspect == null) return
        val docAspect = s.width.toFloat() / s.height
        val (w, h) = if (docAspect > aspect) (s.height * aspect).roundToInt() to s.height else s.width to (s.width / aspect).roundToInt()
        edit("image.crop", """{"x":${(s.width - w) / 2},"y":${(s.height - h) / 2},"width":$w,"height":$h,"deleteCroppedPixels":true}""")
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

    /** One stroke in document pixels: [x, y, pressure]. */
    fun paintStroke(points: List<Triple<Float, Float, Float>>) {
        if (points.isEmpty()) return
        viewModelScope.launch {
            if (paintOnNewLayer && !eraser && !paintLayerMade) {
                lab.fn(Fn.EDIT_RUN, buildJsonObject { put("id", "layer.new.layer"); put("params", buildJsonObject { put("name", "Paint") }) }.toString(), refreshAfter = false)
                paintLayerMade = true
            }
            val params = buildJsonObject {
                putJsonArray("points") { points.forEach { (x, y, p) -> add(JsonArray(listOf(JsonPrimitive(x.roundToInt()), JsonPrimitive(y.roundToInt()), JsonPrimitive(p.coerceIn(0.05f, 1f).toDouble())))) } }
                put("size", paintSize.toDouble())
                put("opacity", paintOpacity.toDouble())
                put("hardness", paintHardness.toDouble())
                put("color", paintColor)
                if (eraser) put("erase", true)
            }
            lab.markInteractive()
            lab.fn(Fn.EDIT_RUN, buildJsonObject { put("id", "paint.stroke"); put("params", params) }.toString())
        }
    }

    /** The next stroke starts a new Paint layer. */
    fun newPaintLayer() {
        paintLayerMade = false
    }

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
