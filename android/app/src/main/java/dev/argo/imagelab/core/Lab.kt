package dev.argo.imagelab.core

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonPrimitive

/** What the editor shows. `version` changes whenever the document may have changed. */
data class EditorState(
    val preview: Bitmap? = null,
    val documents: Int = 0,
    val name: String? = null,
    val width: Int = 0,
    val height: Int = 0,
    val busy: Boolean = false,
    val message: String? = null,
    val version: Long = 0,
)

/**
 * The app's handle on the Rust core (`imagelab-ffi`'s generated [ImageLab]).
 *
 * Every call runs on [Dispatchers.IO]: the core blocks (filters, renders). After anything
 * that may edit the document, [refresh] re-renders the preview, so edits made by the UI, the
 * in-app agent, the CLI console or an external MCP client all show up the same way.
 */
class Lab(val core: ImageLab) {
    private val _state = MutableStateFlow(EditorState())
    val state: StateFlow<EditorState> = _state.asStateFlow()
    private val renderLock = Mutex()
    private val json = Json { ignoreUnknownKeys = true }

    /** Run [block] on the core off the main thread, with a busy flag and errors as messages. */
    suspend fun <T> call(refreshAfter: Boolean = true, block: (ImageLab) -> T): Result<T> {
        _state.update { it.copy(busy = true) }
        val result = withContext(Dispatchers.IO) { runCatching { block(core) } }
        _state.update { it.copy(busy = false, message = result.exceptionOrNull()?.let(::describe)) }
        if (refreshAfter) refresh()
        return result
    }

    /** Re-read the session and re-render the preview. */
    suspend fun refresh() = renderLock.withLock {
        withContext(Dispatchers.IO) {
            val info = runCatching { json.parseToJsonElement(core.sessionInfo()) }.getOrNull()
            val docs = (info as? JsonObject)?.get("documents") as? JsonArray
            if (docs.isNullOrEmpty()) {
                _state.update { it.copy(preview = null, documents = 0, name = null, width = 0, height = 0, version = it.version + 1) }
                return@withContext
            }
            val active = (info["active"]?.jsonPrimitive?.intOrNull ?: 0).coerceIn(0, docs.size - 1)
            val doc = docs[active] as? JsonObject
            val png = runCatching { core.renderPng(PREVIEW_SIDE.toUInt()) }
            val bitmap = png.getOrNull()?.let { BitmapFactory.decodeByteArray(it, 0, it.size) }
            _state.update {
                it.copy(
                    preview = bitmap ?: it.preview,
                    documents = docs.size,
                    name = doc?.string("name"),
                    width = doc?.int("width") ?: 0,
                    height = doc?.int("height") ?: 0,
                    message = png.exceptionOrNull()?.let(::describe) ?: it.message,
                    version = it.version + 1,
                )
            }
        }
    }

    fun clearMessage() = _state.update { it.copy(message = null) }

    fun parse(text: String): JsonElement? = runCatching { json.parseToJsonElement(text) }.getOrNull()

    private fun JsonObject.string(key: String) = (this[key] as? kotlinx.serialization.json.JsonPrimitive)?.content
    private fun JsonObject.int(key: String): Int? =
        (this[key] as? kotlinx.serialization.json.JsonPrimitive)?.intOrNull
            ?: ((this["size"] as? JsonObject)?.get(key) as? kotlinx.serialization.json.JsonPrimitive)?.intOrNull

    companion object {
        /** Longest side of the on-screen preview. Exports always render at full size. */
        const val PREVIEW_SIDE = 2048

        fun describe(t: Throwable): String = when (t) {
            is ImageLabException.BadRequest -> t.reason
            is ImageLabException.Failed -> t.reason
            else -> t.message ?: t.toString()
        }
    }
}
