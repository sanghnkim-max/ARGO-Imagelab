package dev.argo.imagelab.ui

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.LruCache
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import dev.argo.imagelab.ImageLabApp
import dev.argo.imagelab.core.Fn
import dev.argo.imagelab.core.Imports
import dev.argo.imagelab.core.Lab
import dev.argo.imagelab.core.LabEvent
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import java.io.File

/** One photo as `library.query` returns it (fields read leniently). */
data class Photo(val id: String, val path: String?, val name: String, val rating: Int, val flag: String)

/** A batch job's progress card, from jobProgress / jobFinished events. */
data class JobCard(val job: String, val done: Long, val total: Long, val message: String, val finished: Boolean = false, val ok: Boolean = true)

class LibraryViewModel(application: Application) : AndroidViewModel(application) {
    private val app = application as ImageLabApp
    private val lab = app.lab

    var photos by mutableStateOf<List<Photo>>(emptyList()); private set
    var selected by mutableStateOf<Set<String>>(emptySet()); private set
    var focused by mutableStateOf<String?>(null)
    var loading by mutableStateOf(false); private set
    var unavailable by mutableStateOf<String?>(null); private set
    var job by mutableStateOf<JobCard?>(null); private set

    private val thumbs = object : LruCache<String, Bitmap>(48 * 1024 * 1024) {
        override fun sizeOf(key: String, value: Bitmap) = value.byteCount
    }

    val selecting get() = selected.isNotEmpty()

    init {
        load()
        viewModelScope.launch {
            lab.events.collect { e ->
                when (e) {
                    is LabEvent.LibraryChanged -> load()
                    is LabEvent.JobProgress -> job = JobCard(e.job, e.done, e.total, e.message)
                    is LabEvent.JobFinished -> {
                        job = (job ?: JobCard(e.job, 0, 0, "")).copy(finished = true, ok = e.ok, message = e.summary)
                        load()
                    }
                    else -> Unit
                }
            }
        }
    }

    fun load() {
        viewModelScope.launch {
            loading = true
            val r = lab.io { it.call(Fn.LIBRARY_QUERY, """{"limit":1000}""", "ui") }
            loading = false
            r.onSuccess { text ->
                unavailable = null
                photos = parsePhotos(text)
                selected = selected.filter { id -> photos.any { it.id == id } }.toSet()
            }.onFailure { unavailable = Lab.describe(it) }
        }
    }

    private fun parsePhotos(text: String): List<Photo> {
        val root = lab.parse(text)
        val arr = (root as? JsonArray) ?: ((root as? JsonObject)?.get("photos") as? JsonArray) ?: return emptyList()
        return arr.mapNotNull { e ->
            when (e) {
                is JsonPrimitive -> e.contentOrNull?.let { Photo(it, it, it.substringAfterLast('/'), 0, "none") }
                is JsonObject -> {
                    fun s(k: String) = (e[k] as? JsonPrimitive)?.contentOrNull
                    val id = s("id") ?: return@mapNotNull null
                    val path = s("path") ?: s("file") ?: id
                    Photo(id, path, s("name") ?: path?.substringAfterLast('/') ?: id, (e["rating"] as? JsonPrimitive)?.intOrNull ?: 0, s("flag") ?: "none")
                }
                else -> null
            }
        }
    }

    fun toggle(id: String) {
        selected = if (id in selected) selected - id else selected + id
    }

    fun clearSelection() {
        selected = emptySet()
    }

    fun selectAll() {
        selected = photos.map { it.id }.toSet()
    }

    fun rate(rating: Int) = mutate(Fn.LIBRARY_SET_RATING) { put("rating", rating) }
    fun flag(flag: String) = mutate(Fn.LIBRARY_SET_FLAG) { put("flag", flag) }

    private fun mutate(id: String, extra: kotlinx.serialization.json.JsonObjectBuilder.() -> Unit) {
        val ids = selected.ifEmpty { setOfNotNull(focused) }
        if (ids.isEmpty()) return
        viewModelScope.launch {
            val params = buildJsonObject {
                putJsonArray("photos") { ids.forEach { add(JsonPrimitive(it)) } }
                extra()
            }
            lab.fn(id, params.toString(), refreshAfter = false).onSuccess { load() }
        }
    }

    fun batchExport() {
        val ids = selected.ifEmpty { setOfNotNull(focused) }
        if (ids.isEmpty()) return
        viewModelScope.launch {
            val params = buildJsonObject {
                putJsonArray("photos") { ids.forEach { add(JsonPrimitive(it)) } }
                put("preset", "jpeg-2048")
                put("dir", "exports")
            }
            job = JobCard("export", 0, ids.size.toLong(), "Starting export…")
            lab.fn(Fn.BATCH_EXPORT, params.toString(), refreshAfter = false)
                .onFailure { job = job?.copy(finished = true, ok = false, message = Lab.describe(it)) }
        }
    }

    fun dismissJob() {
        job = null
    }

    /** System picker → copies into imports/ → `library.import`. */
    fun import(uris: List<Uri>) {
        if (uris.isEmpty()) return
        viewModelScope.launch {
            val paths = uris.mapNotNull { uri ->
                val picked = Imports.read(app, uri) ?: return@mapNotNull null
                runCatching { Imports.store(app.workspace, picked.first, picked.second) }.getOrNull()
            }
            if (paths.isEmpty()) {
                lab.showMessage("Nothing could be imported.")
                return@launch
            }
            val params = buildJsonObject { putJsonArray("paths") { paths.forEach { add(JsonPrimitive(it)) } } }
            lab.fn(Fn.LIBRARY_IMPORT, params.toString(), refreshAfter = false)
            load()
        }
    }

    /** Open a photo in the editor (its file, through the session). */
    fun open(photo: Photo, then: () -> Unit) {
        viewModelScope.launch {
            val opened = if (photo.path != null) lab.call { it.open(photo.path) }
            else lab.call { it.call(Fn.DOC_OPEN, buildJsonObject { put("photo", photo.id) }.toString(), "ui") }
            if (opened.isSuccess) then()
        }
    }

    /** A small decoded thumbnail of the photo's file, or null (RAW/HEIC the platform can't read). */
    suspend fun thumbnail(photo: Photo, side: Int = 384): Bitmap? {
        val path = photo.path ?: return null
        thumbs.get(path)?.let { return it }
        return withContext(Dispatchers.IO) {
            runCatching {
                val file = File(app.workspace, path)
                val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                BitmapFactory.decodeFile(file.path, bounds)
                var sample = 1
                while (minOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= side) sample *= 2
                BitmapFactory.decodeFile(file.path, BitmapFactory.Options().apply { inSampleSize = sample })
            }.getOrNull()?.also { thumbs.put(path, it) }
        }
    }
}
