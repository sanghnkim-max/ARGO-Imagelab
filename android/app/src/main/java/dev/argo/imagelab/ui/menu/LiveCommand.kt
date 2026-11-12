package dev.argo.imagelab.ui.menu

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import dev.argo.imagelab.core.Fn
import dev.argo.imagelab.core.Lab
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.longOrNull
import kotlinx.serialization.json.put
import kotlin.math.roundToInt

/**
 * Runs any [MenuItem] live: opening it previews the defaults, every control change re-runs the
 * command with the newest values (older ones skipped) replacing the previous attempt, and an
 * interactive render shows it. [apply] keeps the result as one undo step; [cancel] undoes it.
 * Non-live items run once on [apply].
 *
 * The previous attempt is undone only while it is still the document's last change (revision
 * check), so an edit from elsewhere (the agent, MCP) is never undone by mistake.
 *
 * Fast preview: for scale-free items (looks, face shape, Camera Raw, relight) with no selection,
 * a moving slider runs on a ≤ [PROXY_SIDE] proxy of the photo (`edit.preview`, a session of its
 * own, the document untouched) and that frame is shown; the full-resolution run happens when
 * the slider is released (and for discrete changes). A frame is shown only if it is newer than
 * the one on screen, so a late full-size render never covers a newer proxy frame or vice versa.
 */
class LiveCommand(private val lab: Lab, scope: CoroutineScope, private val maskForNew: () -> String?) {
    var item by mutableStateOf<MenuItem?>(null); private set
    val values = mutableStateMapOf<String, Any>()
    /** True while the engine is working on the newest values. */
    var busy by mutableStateOf(false); private set
    /** True once a preview step is in the document for this item. */
    var applied by mutableStateOf(false); private set

    private sealed interface Msg {
        data class Run(val item: MenuItem, val values: Map<String, Any>, val version: Long) : Msg
        data class Begin(val item: MenuItem, val values: Map<String, Any>, val version: Long) : Msg
        data class End(val keep: Boolean) : Msg
    }

    private data class Preview(val item: MenuItem, val values: Map<String, Any>, val version: Long)

    /** The current item previews on the proxy while its sliders move. */
    private var proxy = false
    /** Bumped on every control change; frames carry the version they show. */
    private var version = 0L
    @Volatile private var shownVersion = -1L
    /** The item whose proxy snapshot is ready (`edit.preview {begin}` done). */
    @Volatile private var begun: MenuItem? = null
    private val previews = Channel<Preview>(Channel.CONFLATED)

    private val inbox = Channel<Msg>(Channel.UNLIMITED)
    private var revisionAfter: Long? = null
    private var lastSkipped: String? = null

    init {
        scope.launch {
            while (true) {
                // Everything queued, in order, with each stretch of Runs reduced to its newest.
                val batch = mutableListOf(inbox.receive())
                while (true) batch += inbox.tryReceive().getOrNull() ?: break
                batch.forEachIndexed { i, msg ->
                    when (msg) {
                        is Msg.Run -> if (batch.getOrNull(i + 1) !is Msg.Run) run(msg)
                        is Msg.Begin -> begin(msg)
                        is Msg.End -> end(msg.keep)
                    }
                }
            }
        }
        scope.launch {
            for (pv in previews) if (begun === pv.item) preview(pv)
        }
    }

    /** Scale-free items whose look on a small proxy matches the full-size result. */
    private fun proxyable(it: MenuItem) = it.live && it.params.isNotEmpty() &&
        (it.kind == Kind.Look || it.kind == Kind.Face || it.kind == Kind.AdjustmentLayer || (it.kind == Kind.Pixels && it.command in PROXY_COMMANDS))

    /** The photo on screen (histograms for Curves). */
    fun previewBitmap() = lab.state.value.preview

    /** Start an item (keeping whatever the previous one did). */
    fun open(next: MenuItem) {
        if (item != null) apply()
        item = next
        values.clear()
        next.params.forEach { p ->
            values[p.key] = next.initial[p.key] ?: when (p) {
                is Slider -> p.default
                is Choice -> p.default
                is Toggle -> p.default
                is ColorParam -> p.default
                is CurveParam -> CurveSet()
            }
        }
        proxy = proxyable(next) && maskForNew() == null
        if (proxy) {
            version++
            busy = true
            inbox.trySend(Msg.Begin(next, values.toMap(), version))
            inbox.trySend(Msg.Run(next, values.toMap(), version))
        } else if (next.live && next.params.isNotEmpty()) {
            send()
        }
    }

    /** A discrete change (a choice, a switch, a colour): preview and apply at once. */
    fun set(key: String, value: Any) {
        values[key] = value
        if (item?.live != true) return
        if (proxy) {
            sendPreview()
            release()
        } else {
            send()
        }
    }

    /** A slider moving: proxy frames for proxy items, full runs otherwise. */
    fun drag(key: String, value: Any) {
        values[key] = value
        if (item?.live != true) return
        if (proxy) sendPreview() else send()
    }

    /** The slider was let go: run at full resolution (proxy items). */
    fun release() {
        val it = item ?: return
        if (!proxy || !it.live) return
        busy = true
        inbox.trySend(Msg.Run(it, values.toMap(), version))
    }

    private fun sendPreview() {
        val it = item ?: return
        version++
        previews.trySend(Preview(it, values.toMap(), version))
    }

    /** Keep the result (or, for a non-live item, run it now). */
    fun apply() {
        val it = item ?: return
        if (!it.live || !applied) inbox.trySend(Msg.Run(it, values.toMap(), version))
        inbox.trySend(Msg.End(keep = true))
        item = null
    }

    /** Undo the preview and close. */
    fun cancel() {
        inbox.trySend(Msg.End(keep = false))
        item = null
    }

    private fun send() {
        val it = item ?: return
        busy = true
        lab.markInteractive()
        version++
        inbox.trySend(Msg.Run(it, values.toMap(), version))
    }

    /** `edit.preview`'s step for [item] at [values]. */
    private fun step(item: MenuItem, values: Map<String, Any>): JsonObject {
        val p = params(item, values)
        return when (item.kind) {
            Kind.Look -> buildJsonObject { put("fn", Fn.EDIT_LOOK); put("params", buildJsonObject { put("look", item.command); p.forEach { (k, v) -> put(k, v) } }) }
            Kind.Face -> buildJsonObject { put("fn", Fn.EDIT_FACE_LIQUIFY); put("params", p) }
            Kind.AdjustmentLayer -> buildJsonObject { put("fn", Fn.EDIT_RUN); put("params", buildJsonObject { put("id", "layer.newAdjustmentLayer.${item.command}"); put("params", p) }) }
            else -> buildJsonObject { put("fn", Fn.EDIT_RUN); put("params", buildJsonObject { put("id", item.command); put("params", p) }) }
        }
    }

    /** Show a proxy frame if nothing newer is on screen. */
    private suspend fun showFrame(result: JsonObject?, v: Long) {
        val image = (result?.get("image") as? JsonPrimitive)?.content ?: return
        if (v <= shownVersion) return
        if (lab.showHandle(image)) shownVersion = v
    }

    private suspend fun begin(msg: Msg.Begin) {
        val body = buildJsonObject { put("begin", true); put("maxSide", PROXY_SIDE); put("step", step(msg.item, msg.values)) }
        val r = lab.io { it.call(Fn.EDIT_PREVIEW, body.toString(), "ui") }
        if (r.isFailure) return
        begun = msg.item
        showFrame(lab.parse(r.getOrThrow()) as? JsonObject, msg.version)
        // A slider moved while the snapshot was being made: preview its newest values too.
        if (item === msg.item && version > msg.version) previews.trySend(Preview(msg.item, values.toMap(), version))
    }

    private suspend fun preview(pv: Preview) {
        val body = buildJsonObject { put("maxSide", PROXY_SIDE); put("step", step(pv.item, pv.values)) }
        val r = lab.io { it.call(Fn.EDIT_PREVIEW, body.toString(), "ui") }
        showFrame(r.getOrNull()?.let { lab.parse(it) } as? JsonObject, pv.version)
    }

    /**
     * The command's params for [values]. Keys may address part of a structured param, so plain
     * sliders can drive it: `hslSat[2]` is index 2 of the array `hslSat` (missing indices are 0),
     * `gradeShadows.hue` is field `hue` of the object `gradeShadows`. A [Choice] value starting
     * with `json:` is sent as that JSON (gradient stops, curves).
     */
    private fun params(item: MenuItem, values: Map<String, Any>): JsonObject {
        val flat = linkedMapOf<String, JsonElement>()
        item.fixed.forEach { (k, v) -> flat[k] = element(v) }
        item.params.forEach { p ->
            val v = values[p.key] ?: return@forEach
            if (v is CurveSet) { flat.putAll(v.json()); return@forEach }
            flat[p.key] = if (v is Float) {
                val s = p as? Slider
                val shown = if (s?.integer == true) v.roundToInt().toFloat() else v
                val sent = shown * (s?.sendScale ?: 1f)
                if (s?.integer == true && s.sendScale == 1f) JsonPrimitive(sent.roundToInt()) else JsonPrimitive(sent.toDouble())
            } else element(v)
        }
        // Values set for keys that are not controls (a sampled colour's `points`).
        val keys = item.params.map { it.key }.toSet()
        values.forEach { (k, v) -> if (k !in keys) flat[k] = element(v) }
        return nest(flat)
    }

    private fun element(v: Any): JsonElement = when (v) {
        is JsonElement -> v
        is Boolean -> JsonPrimitive(v)
        is Int -> JsonPrimitive(v)
        is Long -> JsonPrimitive(v)
        is Float -> JsonPrimitive(v.toDouble())
        is Double -> JsonPrimitive(v)
        is String -> if (v.startsWith("json:")) lab.parse(v.removePrefix("json:")) ?: JsonPrimitive(v) else JsonPrimitive(v)
        else -> JsonPrimitive(v.toString())
    }

    private fun nest(flat: Map<String, JsonElement>): JsonObject {
        val out = linkedMapOf<String, Any>()
        flat.forEach { (k, v) ->
            val idx = INDEXED.matchEntire(k)
            val dot = k.indexOf('.')
            when {
                idx != null -> {
                    @Suppress("UNCHECKED_CAST")
                    val arr = out.getOrPut(idx.groupValues[1]) { mutableMapOf<Int, JsonElement>() } as? MutableMap<Int, JsonElement> ?: return@forEach
                    arr[idx.groupValues[2].toInt()] = v
                }
                dot > 0 -> {
                    @Suppress("UNCHECKED_CAST")
                    val obj = out.getOrPut(k.substring(0, dot)) { linkedMapOf<String, JsonElement>() } as? MutableMap<String, JsonElement> ?: return@forEach
                    obj[k.substring(dot + 1)] = v
                }
                else -> out[k] = v
            }
        }
        return JsonObject(out.mapValues { (_, v) ->
            when {
                v is JsonElement -> v
                (v as Map<*, *>).keys.firstOrNull() is Int -> {
                    @Suppress("UNCHECKED_CAST") val m = v as Map<Int, JsonElement>
                    JsonArray((0..(m.keys.maxOrNull() ?: -1)).map { m[it] ?: JsonPrimitive(0) })
                }
                else -> {
                    @Suppress("UNCHECKED_CAST") JsonObject(v as Map<String, JsonElement>)
                }
            }
        })
    }

    private suspend fun revision(): Long? = lab.io { api ->
        val info = lab.parse(api.sessionInfo()) as? JsonObject
        val docs = info?.get("documents") as? JsonArray
        val active = (info?.get("active") as? JsonPrimitive)?.content?.toIntOrNull() ?: 0
        ((docs?.getOrNull(active) as? JsonObject)?.get("revision") as? JsonPrimitive)?.longOrNull
    }.getOrNull()

    private suspend fun run(msg: Msg.Run) {
        val item = msg.item
        if (applied && revisionAfter != null && revision() == revisionAfter) {
            lab.io { it.call("doc.undo", "{}", "ui") }
        } else if (!applied && item.kind != Kind.AdjustmentEdit) {
            // First run of this item: a selection becomes the mask (filters apply inside it,
            // adjustment layers take it as their layer mask).
            maskForNew()?.let { mask -> lab.io { it.call(Fn.SELECT_FROM_MASK, buildJsonObject { put("mask", mask) }.toString(), "ui") } }
        }
        val p = params(item, msg.values)
        val r = when (item.kind) {
            Kind.AdjustmentLayer -> lab.io { it.call(Fn.EDIT_ADJUST, buildJsonObject { put("kind", item.command); put("params", p) }.toString(), "ui") }
            Kind.Look -> lab.io { it.call(Fn.EDIT_LOOK, buildJsonObject { put("look", item.command); p.forEach { (k, v) -> put(k, v) } }.toString(), "ui") }
            Kind.Face -> lab.io { it.call(Fn.EDIT_FACE_LIQUIFY, p.toString(), "ui") }
            Kind.AdjustmentEdit -> lab.io { it.call(Fn.EDIT_RUN, buildJsonObject { put("id", "layer.setAdjustment"); put("params", p) }.toString(), "ui") }
            else -> lab.io { it.call(Fn.EDIT_RUN, buildJsonObject { put("id", item.command); put("params", p) }.toString(), "ui") }
        }
        r.exceptionOrNull()?.let { lab.showMessage(Lab.describe(it)) }
        // Looks and face tools that found nothing to work on (no sky, no face) change nothing:
        // say why, and keep no preview step (so the next run never undoes an earlier edit).
        val result = r.getOrNull()?.let { lab.parse(it) } as? JsonObject
        val nothing = (result?.get("applied") as? JsonPrimitive)?.content == "false"
        if (nothing) {
            val why = (result?.get("skipped") as? JsonArray)?.mapNotNull { (it as? JsonPrimitive)?.content }?.distinct()?.joinToString()
            if (!why.isNullOrBlank() && why != lastSkipped) lab.showMessage("${item.label}: $why")
            lastSkipped = why
        } else {
            lastSkipped = null
        }
        applied = r.isSuccess && !nothing
        revisionAfter = if (r.isSuccess) revision() else null
        if (r.isSuccess && msg.version >= shownVersion) {
            shownVersion = msg.version
            lab.markInteractive()
            lab.renderInteractive()
        }
        busy = false
    }

    private suspend fun end(keep: Boolean) {
        if (!keep && applied && revisionAfter != null && revision() == revisionAfter) {
            lab.io { it.call("doc.undo", "{}", "ui") }
        }
        applied = false
        revisionAfter = null
        begun = null
        shownVersion = -1L
        busy = false
        lab.requestRefresh()
    }

    companion object {
        /** Longest side of the proxy previews follow while a slider moves. */
        const val PROXY_SIDE = 768
        /** Pixel commands without pixel-unit parameters, so a proxy previews them faithfully. */
        val PROXY_COMMANDS = setOf("filter.cameraRaw", "filter.render.relight", "filter.lensCorrection")
        private val INDEXED = Regex("""^(\w+)\[(\d+)]$""")
    }
}
