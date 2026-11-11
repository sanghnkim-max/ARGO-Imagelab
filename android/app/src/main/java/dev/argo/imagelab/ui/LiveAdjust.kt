package dev.argo.imagelab.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.ColorMatrix
import dev.argo.imagelab.core.Fn
import dev.argo.imagelab.core.Lab
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.longOrNull
import kotlinx.serialization.json.put
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * One slider of the Adjust sheet: a PhotoCraft adjustment-layer kind, how a slider value maps
 * to its params, and a colour-matrix approximation of a *change* in that value (drawn on the
 * GPU over the last engine frame, so the photo moves with the finger before the engine renders).
 */
data class AdjustSpec(
    val label: String,
    val kind: String,
    val range: ClosedFloatingPointRange<Float>,
    val params: (Float) -> JsonObject,
    val matrix: (rendered: Float, target: Float) -> ColorMatrix?,
)

private fun offsetMatrix(t: Float) = ColorMatrix(floatArrayOf(1f, 0f, 0f, 0f, t, 0f, 1f, 0f, 0f, t, 0f, 0f, 1f, 0f, t, 0f, 0f, 0f, 1f, 0f))
private fun scaleAround(s: Float, pivot: Float = 128f) = (pivot * (1f - s)).let { o -> ColorMatrix(floatArrayOf(s, 0f, 0f, 0f, o, 0f, s, 0f, 0f, o, 0f, 0f, s, 0f, o, 0f, 0f, 0f, 1f, 0f)) }
private fun saturation(s: Float) = ColorMatrix().apply { setToSaturation(s.coerceIn(0f, 4f)) }

/** Luminance-preserving hue rotation (degrees). */
private fun hueRotation(deg: Float): ColorMatrix {
    val a = Math.toRadians(deg.toDouble())
    val c = cos(a).toFloat()
    val s = sin(a).toFloat()
    val (lr, lg, lb) = Triple(0.213f, 0.715f, 0.072f)
    return ColorMatrix(
        floatArrayOf(
            lr + c * (1 - lr) + s * (-lr), lg + c * (-lg) + s * (-lg), lb + c * (-lb) + s * (1 - lb), 0f, 0f,
            lr + c * (-lr) + s * 0.143f, lg + c * (1 - lg) + s * 0.140f, lb + c * (-lb) + s * (-0.283f), 0f, 0f,
            lr + c * (-lr) + s * (-(1 - lr)), lg + c * (-lg) + s * lg, lb + c * (1 - lb) + s * lb, 0f, 0f,
            0f, 0f, 0f, 1f, 0f,
        ),
    )
}

private fun json(vararg kv: Pair<String, Number>) = buildJsonObject { kv.forEach { (k, v) -> put(k, v) } }

/** The Adjust sheet's sliders (adjustment layers: non-destructive, maskable). */
val adjustSpecs = listOf(
    AdjustSpec("Brightness", "brightnessContrast", -150f..150f, { json("brightness" to it.roundToInt()) }, { r, t -> offsetMatrix((t - r) * 0.45f) }),
    AdjustSpec("Contrast", "brightnessContrast", -50f..100f, { json("contrast" to it.roundToInt()) }, { r, t -> scaleAround((1f + t / 100f) / (1f + r / 100f)) }),
    AdjustSpec("Exposure", "exposure", -5f..5f, { json("exposure" to (it * 100).roundToInt() / 100.0) }, { r, t -> scaleAround(2f.pow((t - r) / 2.2f), 0f) }),
    AdjustSpec("Vibrance", "vibrance", -100f..100f, { json("vibrance" to it.roundToInt()) }, { r, t -> saturation((1f + t / 200f) / (1f + r / 200f)) }),
    AdjustSpec("Saturation", "hueSaturation", -100f..100f, { json("saturation" to it.roundToInt()) }, { r, t -> saturation((1f + t / 100f) / (1f + r / 100f)) }),
    AdjustSpec("Hue", "hueSaturation", -180f..180f, { json("hue" to it.roundToInt()) }, { r, t -> hueRotation(t - r) }),
)

/**
 * Live adjustment: the photo follows the slider.
 *
 * * Every value change sets [matrix] at once: the GPU draws the change since the last engine
 *   frame over it (60 fps, approximate).
 * * The engine follows as fast as it can: the newest value only (older ones are skipped), on
 *   the real adjustment layer, then an interactive render (1024 px RGBA) replaces the frame and
 *   the matrix shrinks to what is still pending.
 * * One drag is one undo step: the first drag on a slider creates the layer, later drags update
 *   it with `layer.setAdjustment`; inside a drag each update undoes the previous one.
 * * "New layer" (or switching slider) starts the next adjustment on a fresh layer.
 */
class LiveAdjust(private val lab: Lab, scope: CoroutineScope, private val maskForNewLayer: () -> String?) {
    /** Colour matrix for the not-yet-rendered part of the change, or null. */
    var matrix by mutableStateOf<ColorMatrix?>(null); private set

    private sealed interface Msg {
        data class Value(val spec: AdjustSpec, val value: Float) : Msg
        data object Release : Msg
    }

    private val inbox = Channel<Msg>(Channel.UNLIMITED)

    // Engine-side state (touched only by the worker coroutine).
    private var layer: Long? = null
    private var layerSpec: AdjustSpec? = null
    private var stepInDrag = false
    private var createdInDrag = false

    // UI-side state.
    @Volatile private var rendered = 0f
    @Volatile private var target = 0f
    @Volatile private var activeSpec: AdjustSpec? = null

    init {
        scope.launch {
            while (true) {
                var msg = inbox.receive()
                // Coalesce: keep only the newest value; a Release ends the drag after it.
                var release = msg is Msg.Release
                while (true) {
                    val next = inbox.tryReceive().getOrNull() ?: break
                    if (next is Msg.Release) release = true else msg = next
                }
                if (msg is Msg.Value) apply(msg.spec, msg.value)
                if (release) {
                    stepInDrag = false
                    createdInDrag = false
                }
            }
        }
    }

    /** The slider moved. */
    fun drag(spec: AdjustSpec, value: Float) {
        if (activeSpec != spec) {
            activeSpec = spec
            rendered = 0f
            inbox.trySend(Msg.Release)
        }
        target = value
        matrix = spec.matrix(rendered, value)
        lab.markInteractive()
        inbox.trySend(Msg.Value(spec, value))
    }

    /** The finger lifted: commit this drag as one undo step. */
    fun release() {
        inbox.trySend(Msg.Release)
    }

    /** The next drag starts a new adjustment layer. */
    fun newLayer() {
        inbox.trySend(Msg.Release)
        activeSpec = null
        rendered = 0f
        matrix = null
        layer = null
        layerSpec = null
    }

    private suspend fun apply(spec: AdjustSpec, value: Float) {
        if (layerSpec != spec) {
            layer = null
            layerSpec = spec
        }
        if (stepInDrag) lab.io { it.undo() }
        val ok = if (layer == null || createdInDrag) {
            val mask = if (layer == null) maskForNewLayer() else null
            if (mask != null) lab.io { it.call(Fn.SELECT_FROM_MASK, buildJsonObject { put("mask", mask) }.toString(), "ui") }
            val r = lab.io { it.call(Fn.EDIT_ADJUST, buildJsonObject { put("kind", spec.kind); put("params", spec.params(value)) }.toString(), "ui") }
            r.getOrNull()?.let { text -> ((lab.parse(text) as? JsonObject)?.get("layer") as? JsonPrimitive)?.longOrNull }?.let { layer = it }
            createdInDrag = true
            r.isSuccess
        } else {
            val params = buildJsonObject {
                put("layer", layer ?: 0L)
                spec.params(value).forEach { (k, v) -> put(k, v) }
            }
            lab.io { it.call(Fn.EDIT_RUN, buildJsonObject { put("id", "layer.setAdjustment"); put("params", params) }.toString(), "ui") }.isSuccess
        }
        stepInDrag = ok
        if (!ok) return
        lab.markInteractive()
        if (lab.renderInteractive()) {
            rendered = value
            val t = target
            matrix = if (t == value) null else spec.matrix(value, t)
        }
    }
}

/**
 * Live straighten: the photo rotates with the Crop sheet's slider.
 *
 * * Every value change rotates the displayed frame at once on the GPU ([rotation]), zoomed
 *   ([zoom]) just enough to hide the corners — what `edit.straighten`'s crop keeps.
 * * The engine follows with the newest angle (`edit.straighten {angle, crop: true}`: rotate +
 *   crop to the largest inscribed rectangle, two undo steps), replacing its previous attempt in
 *   the same drag, then an interactive render replaces the frame.
 * * Dragging again re-adjusts the same straighten (its steps are undone first) as long as
 *   nothing else edited the document in between; [done] starts a new one.
 */
class LiveStraighten(private val lab: Lab, scope: CoroutineScope) {
    /** Degrees (clockwise) the displayed frame is rotated beyond what the engine rendered. */
    var rotation by mutableStateOf(0f); private set
    /** Extra scale that hides the rotated frame's corners. */
    var zoom by mutableStateOf(1f); private set
    /** True while straightening (the canvas draws a grid). */
    var active by mutableStateOf(false); private set

    private sealed interface Msg {
        data class Angle(val angle: Float) : Msg
        data object Release : Msg
    }

    private val inbox = Channel<Msg>(Channel.UNLIMITED)

    // Engine-side (worker only): steps our current straighten made, and the document revision
    // right after them (another edit since → start a new straighten instead of replacing).
    private var steps = 0
    private var revisionAfter: Long? = null

    @Volatile private var rendered = 0f
    @Volatile private var target = 0f

    init {
        scope.launch {
            while (true) {
                var msg = inbox.receive()
                var release = msg is Msg.Release
                while (true) {
                    val next = inbox.tryReceive().getOrNull() ?: break
                    if (next is Msg.Release) release = true else msg = next
                }
                if (msg is Msg.Angle) apply(msg.angle)
                if (release && inbox.isEmpty) {
                    active = false
                    lab.requestRefresh()
                }
            }
        }
    }

    fun drag(angle: Float) {
        active = true
        target = angle
        setDelta(angle - rendered)
        lab.markInteractive()
        inbox.trySend(Msg.Angle(angle))
    }

    fun release() {
        inbox.trySend(Msg.Release)
    }

    /** Keep the current straighten; the next drag starts a new one from 0°. */
    fun done() {
        inbox.trySend(Msg.Release)
        steps = 0
        revisionAfter = null
        rendered = 0f
        target = 0f
        setDelta(0f)
    }

    private fun setDelta(d: Float) {
        rotation = d
        val bmp = lab.state.value.preview
        val ratio = if (bmp != null && bmp.width > 0 && bmp.height > 0) maxOf(bmp.width, bmp.height).toFloat() / minOf(bmp.width, bmp.height) else 1f
        val r = Math.toRadians(kotlin.math.abs(d).toDouble())
        zoom = (cos(r) + ratio * sin(r)).toFloat().coerceAtLeast(1f)
    }

    private suspend fun revision(): Long? = lab.io { api ->
        val info = lab.parse(api.sessionInfo()) as? JsonObject
        val docs = info?.get("documents") as? kotlinx.serialization.json.JsonArray
        val active = (info?.get("active") as? JsonPrimitive)?.content?.toIntOrNull() ?: 0
        ((docs?.getOrNull(active) as? JsonObject)?.get("revision") as? JsonPrimitive)?.longOrNull
    }.getOrNull()

    private suspend fun apply(angle: Float) {
        // Replace our previous straighten only if it is still the last thing that happened.
        if (steps > 0 && revisionAfter != null && revision() == revisionAfter) {
            lab.io { it.call("doc.undo", buildJsonObject { put("steps", steps) }.toString(), "ui") }
        }
        steps = 0
        val r = lab.io { it.call(Fn.EDIT_STRAIGHTEN, buildJsonObject { put("angle", angle); put("crop", true) }.toString(), "ui") }
        val result = r.getOrNull()?.let { lab.parse(it) as? JsonObject }
        val rotated = (result?.get("rotated") as? JsonPrimitive)?.content == "true"
        val cropped = (result?.get("cropped") as? JsonPrimitive)?.content == "true"
        steps = (if (rotated) 1 else 0) + (if (cropped) 1 else 0)
        revisionAfter = revision()
        if (r.isFailure) return
        lab.markInteractive()
        if (lab.renderInteractive()) {
            rendered = angle
            setDelta(target - angle)
        }
    }
}
