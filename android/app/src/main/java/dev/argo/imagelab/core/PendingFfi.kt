// ─────────────────────────────────────────────────────────────────────────────────────────────
// PENDING FFI — delete this file when imagelab-ffi exports the §11 calls.
//
// docs/INTERFACES.md §11 adds call/help/areasSummary/runScript/cancelScripts/setEventListener/
// renderHandle/startMcpHttp(port, token, allowEdits) and the ScriptResult/ScriptImage records
// and the EventListener callback interface to the generated bindings. Until they exist, this
// file supplies stand-ins with the same names and shapes:
//
//   * the records and the callback interface, declared here in the bindings' package;
//   * the calls, as extension functions on the generated [ImageLab]. Kotlin resolves a member
//     before an extension, so once uniffi generates the real members the stand-ins go unused,
//     and deleting this file is the whole swap. Nothing outside this file names them.
//
// What the stand-ins do (nothing here is meant to survive the merge):
//   * call(): a few registry ids mapped onto existing calls (doc.inspect/open/undo/redo,
//     edit.run without a mask, select.subject/sky as PhotoCraft commands) and a file-listing
//     library (imports/ + library/, ratings/flags in memory); everything else throws
//     "not linked yet".
//   * runScript(): returns ScriptResult(error = "script runtime not linked yet").
//   * setEventListener(): a 1 s poll of sessionInfo() that emits docChanged — the only place
//     the old revision poll still lives.
//   * renderHandle(): throws (no handle arena yet). startMcpHttp(…, allowEdits) ignores the flag.
// ─────────────────────────────────────────────────────────────────────────────────────────────
package dev.argo.imagelab.core

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.put
import java.io.File
import java.util.concurrent.ConcurrentHashMap

data class ScriptImage(val mime: String, val bytes: ByteArray) {
    override fun equals(other: Any?) = other is ScriptImage && mime == other.mime && bytes.contentEquals(other.bytes)
    override fun hashCode() = 31 * mime.hashCode() + bytes.contentHashCode()
}

data class ScriptResult(val valueJson: String, val logs: List<String>, val images: List<ScriptImage>, val error: String?)

interface EventListener {
    fun onEvent(eventJson: String)
}

private object Pending {
    val json = Json { ignoreUnknownKeys = true }
    @Volatile var listener: EventListener? = null
    @Volatile var poller: Thread? = null
    val ratings = ConcurrentHashMap<String, Int>()
    val flags = ConcurrentHashMap<String, String>()
    val imageExt = setOf("jpg", "jpeg", "png", "webp", "heic", "heif", "tif", "tiff", "psd", "pcraft", "dng", "cr2", "nef", "arw")

    fun notLinked(id: String): Nothing =
        throw ImageLabException.Failed("`$id` is not available yet: the function registry is not linked into this build")

    fun emit(json: String) = runCatching { listener?.onEvent(json) }
}

fun ImageLab.call(id: String, paramsJson: String, source: String): String {
    val p = runCatching { Pending.json.parseToJsonElement(paramsJson.ifBlank { "{}" }) as? JsonObject }.getOrNull()
        ?: throw ImageLabException.BadRequest("params must be a JSON object")
    fun str(k: String) = (p[k] as? JsonPrimitive)?.contentOrNull
    fun list(k: String) = (p[k] as? JsonArray).orEmpty().mapNotNull { (it as? JsonPrimitive)?.contentOrNull }
    return when (id) {
        Fn.DOC_INSPECT -> inspect()
        Fn.DOC_OPEN -> open(str("path") ?: str("photo") ?: throw ImageLabException.BadRequest("doc.open needs path"))
        "doc.undo" -> undo()
        "doc.redo" -> redo()
        Fn.EDIT_RUN -> {
            if (p["mask"] != null) Pending.notLinked("edit.run with a mask")
            run(str("id") ?: throw ImageLabException.BadRequest("edit.run needs id"), (p["params"] ?: JsonObject(emptyMap())).toString())
        }
        Fn.SELECT_SUBJECT, Fn.SELECT_SKY -> run(id, paramsJson.ifBlank { "{}" })
        Fn.LIBRARY_QUERY -> pendingLibrary(workspaceRoot(), (p["limit"] as? JsonPrimitive)?.intOrNull ?: 500).toString()
        Fn.LIBRARY_IMPORT -> {
            val ids = list("paths")
            Pending.emit(buildJsonObject { put("type", "libraryChanged"); put("photos", JsonArray(ids.map(::JsonPrimitive))); put("source", source) }.toString())
            JsonArray(ids.map(::JsonPrimitive)).toString()
        }
        Fn.LIBRARY_SET_RATING -> {
            val rating = (p["rating"] as? JsonPrimitive)?.intOrNull ?: 0
            list("photos").forEach { Pending.ratings[it] = rating }
            Pending.emit(buildJsonObject { put("type", "libraryChanged"); put("photos", p["photos"] ?: JsonArray(emptyList())); put("source", source) }.toString())
            "{}"
        }
        Fn.LIBRARY_SET_FLAG -> {
            val flag = str("flag") ?: "none"
            list("photos").forEach { if (flag == "none") Pending.flags.remove(it) else Pending.flags[it] = flag }
            Pending.emit(buildJsonObject { put("type", "libraryChanged"); put("photos", p["photos"] ?: JsonArray(emptyList())); put("source", source) }.toString())
            "{}"
        }
        else -> Pending.notLinked(id)
    }
}

/** Photos = image files under imports/ and library/, id = relative path. */
private fun pendingLibrary(root: String, limit: Int): JsonArray {
    val base = File(root)
    val files = listOf("library", "imports").flatMap { dir ->
        File(base, dir).walkTopDown().maxDepth(3).filter { it.isFile && it.extension.lowercase() in Pending.imageExt }.toList()
    }.sortedByDescending { it.lastModified() }.take(limit.coerceIn(1, 5000))
    return buildJsonArray {
        files.forEach { f ->
            val rel = f.relativeTo(base).path
            add(buildJsonObject {
                put("id", rel)
                put("path", rel)
                put("name", f.name)
                put("rating", Pending.ratings[rel] ?: 0)
                put("flag", Pending.flags[rel] ?: "none")
                put("takenAt", f.lastModified())
            })
        }
    }
}

fun ImageLab.help(area: String?): String =
    if (area == null) areasSummary() else "help($area): the function registry is not linked into this build yet."

@Suppress("UnusedReceiverParameter")
fun ImageLab.areasSummary(): String = """
    library — photos in the workspace: import, query, albums, ratings, flags, tags, metadata
    analyze — quality scores, near-duplicates, histograms, faces, contact sheets
    doc — open/close/inspect/preview/save the PhotoCraft document; history
    edit — PhotoCraft engine commands, masked adjustments, recipes
    batch — apply a recipe/program to many photos; export presets
    select — PhotoCraft selections (subject, sky, object, color range)
    segment — SAM masks from a point, a box or a text concept
    cv — OpenCV-style image ops on handles
""".trimIndent()

@Suppress("UnusedReceiverParameter")
fun ImageLab.runScript(code: String, readOnly: Boolean, timeoutMs: ULong): ScriptResult =
    ScriptResult(valueJson = "null", logs = emptyList(), images = emptyList(), error = "script runtime not linked yet")

@Suppress("UnusedReceiverParameter")
fun ImageLab.cancelScripts() {}

/** The revision poll, kept only here until the core pushes events. */
fun ImageLab.setEventListener(listener: EventListener) {
    Pending.listener = listener
    if (Pending.poller != null) return
    val core = this
    Pending.poller = Thread({
        var last: String? = null
        while (true) {
            try {
                Thread.sleep(1000)
            } catch (_: InterruptedException) {
                return@Thread
            }
            val info = runCatching { core.sessionInfo() }.getOrNull() ?: continue
            val key = Regex("\"revision\":\\d+|\"active\":\\d+|\"index\":\\d+|\"name\":\"[^\"]*\"").findAll(info).joinToString { it.value }
            if (last != null && key != last) {
                val obj = runCatching { Pending.json.parseToJsonElement(info) as? JsonObject }.getOrNull()
                val active = (obj?.get("active") as? JsonPrimitive)?.intOrNull ?: 0
                val revision = Regex("\"revision\":(\\d+)").find(info)?.groupValues?.get(1)?.toLongOrNull() ?: 0L
                Pending.emit("""{"type":"docChanged","doc":$active,"revision":$revision,"source":"ui"}""")
            }
            last = key
        }
    }, "imagelab-pending-events").apply { isDaemon = true; start() }
}

@Suppress("UnusedReceiverParameter")
fun ImageLab.renderHandle(handle: String, maxSide: UInt): ByteArray =
    throw ImageLabException.Failed("cannot render $handle: handles are not linked into this build yet")

fun ImageLab.startMcpHttp(port: UShort, token: String, allowEdits: Boolean): UShort = startMcpHttp(port, token)
