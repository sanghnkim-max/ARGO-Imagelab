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
 */
class LiveCommand(private val lab: Lab, scope: CoroutineScope, private val maskForNew: () -> String?) {
    var item by mutableStateOf<MenuItem?>(null); private set
    val values = mutableStateMapOf<String, Any>()
    /** True while the engine is working on the newest values. */
    var busy by mutableStateOf(false); private set
    /** True once a preview step is in the document for this item. */
    var applied by mutableStateOf(false); private set

    private sealed interface Msg {
        data class Run(val item: MenuItem, val values: Map<String, Any>) : Msg
        data class End(val keep: Boolean) : Msg
    }

    private val inbox = Channel<Msg>(Channel.UNLIMITED)
    private var revisionAfter: Long? = null
    private var lastSkipped: String? = null

    init {
        scope.launch {
            while (true) {
                var msg = inbox.receive()
                // Keep only the newest Run before an End.
                while (msg is Msg.Run) {
                    val next = inbox.tryReceive().getOrNull() ?: break
                    if (next is Msg.End) {
                        run(msg)
                        msg = next
                        break
                    }
                    msg = next
                }
                when (msg) {
                    is Msg.Run -> run(msg)
                    is Msg.End -> end(msg.keep)
                }
            }
        }
    }

    /** Start an item (keeping whatever the previous one did). */
    fun open(next: MenuItem) {
        if (item != null) apply()
        item = next
        values.clear()
        next.params.forEach { p ->
            values[p.key] = when (p) {
                is Slider -> p.default
                is Choice -> p.default
                is Toggle -> p.default
                is ColorParam -> p.default
            }
        }
        if (next.live && next.params.isNotEmpty()) send()
    }

    fun set(key: String, value: Any) {
        values[key] = value
        if (item?.live == true) send()
    }

    /** Keep the result (or, for a non-live item, run it now). */
    fun apply() {
        val it = item ?: return
        if (!it.live || !applied) inbox.trySend(Msg.Run(it, values.toMap()))
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
        inbox.trySend(Msg.Run(it, values.toMap()))
    }

    private fun params(item: MenuItem, values: Map<String, Any>): JsonObject = buildJsonObject {
        item.fixed.forEach { (k, v) -> putAny(k, v) }
        item.params.forEach { p ->
            when (val v = values[p.key]) {
                is Float -> {
                    val s = p as? Slider
                    val shown = if (s?.integer == true) v.roundToInt().toFloat() else v
                    val sent = shown * (s?.sendScale ?: 1f)
                    if (s?.integer == true && s.sendScale == 1f) put(p.key, sent.roundToInt()) else put(p.key, sent.toDouble())
                }
                null -> {}
                else -> putAny(p.key, v)
            }
        }
    }

    private fun kotlinx.serialization.json.JsonObjectBuilder.putAny(k: String, v: Any) {
        when (v) {
            is Boolean -> put(k, v)
            is Int -> put(k, v)
            is Long -> put(k, v)
            is Float -> put(k, v.toDouble())
            is Double -> put(k, v)
            else -> put(k, v.toString())
        }
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
        } else if (!applied) {
            // First run of this item: a selection becomes the mask (filters apply inside it,
            // adjustment layers take it as their layer mask).
            maskForNew()?.let { mask -> lab.io { it.call(Fn.SELECT_FROM_MASK, buildJsonObject { put("mask", mask) }.toString(), "ui") } }
        }
        val p = params(item, msg.values)
        val r = when (item.kind) {
            Kind.AdjustmentLayer -> lab.io { it.call(Fn.EDIT_ADJUST, buildJsonObject { put("kind", item.command); put("params", p) }.toString(), "ui") }
            Kind.Look -> lab.io { it.call(Fn.EDIT_LOOK, buildJsonObject { put("look", item.command); p.forEach { (k, v) -> put(k, v) } }.toString(), "ui") }
            Kind.Face -> lab.io { it.call(Fn.EDIT_FACE_LIQUIFY, p.toString(), "ui") }
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
        if (r.isSuccess) {
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
        busy = false
        lab.requestRefresh()
    }
}
