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
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import java.io.File
import kotlin.math.roundToInt

enum class Tool(val label: String) { Layers("Layers"), Select("Select"), Adjust("Adjust"), Retouch("Retouch"), Crop("Crop") }

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

    /** Last stylus pressure on the canvas (0..1): plumbed for the coming retouch brushes. */
    var stylusPressure by mutableFloatStateOf(0f)

    fun pickTool(t: Tool) {
        tool = if (tool == t) null else t
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

    fun selectPoint(x: Float, y: Float) = select("Point", Fn.SEGMENT_POINT, buildJsonObject {
        putJsonArray("points") { add(buildJsonObject { put("x", x.roundToInt()); put("y", y.roundToInt()); put("label", 1) }) }
    }.toString())

    fun selectBox(x0: Float, y0: Float, x1: Float, y1: Float) = select("Box", Fn.SEGMENT_BOX, buildJsonObject {
        putJsonArray("box") { listOf(minOf(x0, x1), minOf(y0, y1), maxOf(x0, x1), maxOf(y0, y1)).forEach { add(JsonPrimitive(it.roundToInt())) } }
    }.toString())

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
                selection = Selection(label, handle, overlay)
            }
            selecting = false
        }
    }

    fun clearSelection() {
        val had = selection
        selection = null
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

    // ── layers ───────────────────────────────────────────────────────────────────────────

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
