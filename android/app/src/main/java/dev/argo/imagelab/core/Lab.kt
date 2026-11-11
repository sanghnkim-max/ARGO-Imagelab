package dev.argo.imagelab.core

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.async
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull

/** One layer as `doc.inspect` describes it (top first). */
data class LayerInfo(val id: Long?, val name: String, val kind: String, val visible: Boolean, val hasMask: Boolean)

/** What the editor shows. `version` changes whenever the document may have changed. */
data class EditorState(
    val preview: Bitmap? = null,
    val documents: Int = 0,
    val name: String? = null,
    val width: Int = 0,
    val height: Int = 0,
    val layers: List<LayerInfo> = emptyList(),
    val activeLayer: Long? = null,
    val busy: Boolean = false,
    val message: String? = null,
    val version: Long = 0,
)

/**
 * The app's handle on the Rust core, through [LabApi].
 *
 * Every core call runs on [Dispatchers.IO]: the core blocks (filters, renders). The preview
 * follows the core's events (§5): a `docChanged` from any source — the UI, the in-app agent,
 * the CLI console, an external MCP client — schedules one debounced re-render.
 */
class Lab(createApi: () -> LabApi, private val scope: CoroutineScope) {
    /** The core is made on IO (opening the workspace and probing the GPU takes seconds), never on main. */
    private val apiDeferred = scope.async(Dispatchers.IO) { createApi() }

    private val _state = MutableStateFlow(EditorState())
    val state: StateFlow<EditorState> = _state.asStateFlow()
    private val _events = MutableSharedFlow<LabEvent>(extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST)
    val events: SharedFlow<LabEvent> = _events.asSharedFlow()
    private val renderLock = Mutex()
    private val refreshRequests = Channel<Unit>(Channel.CONFLATED)
    val json = Json { ignoreUnknownKeys = true }

    init {
        scope.launch(Dispatchers.IO) {
            runCatching {
                apiDeferred.await().setEventListener(object : EventListener {
                    override fun onEvent(eventJson: String) {
                        val event = LabEvent.parse(eventJson) ?: return
                        _events.tryEmit(event)
                        if (event is LabEvent.DocChanged) refreshRequests.trySend(Unit)
                    }
                })
            }.onFailure { t -> _state.update { it.copy(message = "Events unavailable: ${describe(t)}") } }
        }
        // Debounce: a burst of edits (a batch, an agent program) renders once.
        scope.launch {
            for (unit in refreshRequests) {
                delay(REFRESH_DEBOUNCE_MS)
                refresh()
            }
        }
    }

    /** Ask for a re-render soon (debounced). */
    fun requestRefresh() {
        refreshRequests.trySend(Unit)
    }

    /** Run [block] on the core off the main thread, without touching the busy flag or message. */
    suspend fun <T> io(block: (LabApi) -> T): Result<T> = withContext(Dispatchers.IO) { runCatching { block(apiDeferred.await()) } }

    /** Run [block] on the core with a busy flag; a failure becomes the editor's message. */
    suspend fun <T> call(refreshAfter: Boolean = true, block: (LabApi) -> T): Result<T> {
        _state.update { it.copy(busy = true) }
        val result = io(block)
        _state.update { it.copy(busy = false, message = result.exceptionOrNull()?.let(::describe) ?: it.message) }
        if (refreshAfter && result.isSuccess) requestRefresh()
        return result
    }

    /** A registry call from the UI: `id` with JSON params; the parsed result or a message. */
    suspend fun fn(id: String, params: String = "{}", refreshAfter: Boolean = true): Result<JsonElement?> =
        call(refreshAfter) { parse(it.call(id, params, "ui")) }

    /** Re-read the session and re-render the preview. */
    suspend fun refresh() = renderLock.withLock {
        withContext(Dispatchers.IO) {
            val api = runCatching { apiDeferred.await() }.getOrElse { t ->
                _state.update { it.copy(message = "The engine failed to start: ${describe(t)}") }
                return@withContext
            }
            val info = runCatching { json.parseToJsonElement(api.sessionInfo()) }.getOrNull()
            val docs = (info as? JsonObject)?.get("documents") as? JsonArray
            if (docs.isNullOrEmpty()) {
                _state.update { it.copy(preview = null, documents = 0, name = null, width = 0, height = 0, layers = emptyList(), version = it.version + 1) }
                return@withContext
            }
            val active = (info["active"]?.jsonPrimitive?.intOrNull ?: 0).coerceIn(0, docs.size - 1)
            val doc = docs[active] as? JsonObject
            val png = runCatching { api.renderPng(PREVIEW_SIDE.toUInt()) }
            val bitmap = png.getOrNull()?.let { BitmapFactory.decodeByteArray(it, 0, it.size) }
            val detail = (runCatching { api.call(Fn.DOC_INSPECT, "{}", "ui") }.getOrNull() ?: runCatching { api.inspect() }.getOrNull())
                ?.let(::parse) as? JsonObject
            _state.update {
                it.copy(
                    preview = bitmap ?: it.preview,
                    documents = docs.size,
                    name = doc?.string("name"),
                    width = doc?.int("width") ?: 0,
                    height = doc?.int("height") ?: 0,
                    layers = detail?.let(::layersOf) ?: it.layers,
                    activeLayer = (detail?.get("activeLayer") as? JsonPrimitive)?.longOrNull,
                    message = png.exceptionOrNull()?.let(::describe) ?: it.message,
                    version = it.version + 1,
                )
            }
        }
    }

    fun clearMessage() = _state.update { it.copy(message = null) }
    fun showMessage(text: String) = _state.update { it.copy(message = text) }

    fun parse(text: String): JsonElement? = runCatching { json.parseToJsonElement(text) }.getOrNull()

    private fun layersOf(detail: JsonObject): List<LayerInfo> =
        (detail["layers"] as? JsonArray).orEmpty().mapNotNull { it as? JsonObject }.map { l ->
            LayerInfo(
                id = (l["id"] as? JsonPrimitive)?.longOrNull,
                name = l.string("name") ?: "Layer",
                kind = l.string("kind") ?: "pixel",
                visible = (l["visible"] as? JsonPrimitive)?.booleanOrNull ?: true,
                hasMask = (l["hasMask"] as? JsonPrimitive)?.booleanOrNull ?: false,
            )
        }

    private fun JsonObject.string(key: String) = (this[key] as? JsonPrimitive)?.contentOrNull
    private fun JsonObject.int(key: String): Int? =
        (this[key] as? JsonPrimitive)?.intOrNull ?: ((this["size"] as? JsonObject)?.get(key) as? JsonPrimitive)?.intOrNull

    companion object {
        /** Longest side of the on-screen preview. Exports always render at full size. */
        const val PREVIEW_SIDE = 2048
        private const val REFRESH_DEBOUNCE_MS = 120L

        fun describe(t: Throwable): String = when (t) {
            is ImageLabException.BadRequest -> t.reason
            is ImageLabException.Failed -> t.reason
            else -> t.message ?: t.toString()
        }
    }
}
