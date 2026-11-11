package dev.argo.imagelab.core

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.floatOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.longOrNull

/** Core events (docs/INTERFACES.md §5), parsed from `EventListener.onEvent(eventJson)`. */
sealed interface LabEvent {
    data class DocChanged(val doc: Int, val revision: Long, val source: String) : LabEvent
    data class LibraryChanged(val photos: List<String>, val source: String) : LabEvent
    data class JobProgress(val job: String, val done: Long, val total: Long, val message: String) : LabEvent
    data class JobFinished(val job: String, val ok: Boolean, val summary: String) : LabEvent
    data class ModelStatus(val model: String, val state: String, val progress: Float) : LabEvent

    companion object {
        private val json = Json { ignoreUnknownKeys = true }

        /** Null for anything unknown or malformed: a newer core may send more kinds. */
        fun parse(text: String): LabEvent? {
            val o = runCatching { json.parseToJsonElement(text) as? JsonObject }.getOrNull() ?: return null
            fun s(k: String) = (o[k] as? JsonPrimitive)?.contentOrNull
            fun l(k: String) = (o[k] as? JsonPrimitive)?.longOrNull
            return when (s("type")?.lowercase()) {
                "docchanged" -> DocChanged((o["doc"] as? JsonPrimitive)?.intOrNull ?: 0, l("revision") ?: 0, s("source") ?: "ui")
                "librarychanged" -> LibraryChanged(
                    (o["photos"] as? JsonArray).orEmpty().mapNotNull { (it as? JsonPrimitive)?.contentOrNull },
                    s("source") ?: "ui",
                )
                "jobprogress" -> JobProgress(s("job").orEmpty(), l("done") ?: 0, l("total") ?: 0, s("message").orEmpty())
                "jobfinished" -> JobFinished(s("job").orEmpty(), (o["ok"] as? JsonPrimitive)?.booleanOrNull ?: false, s("summary").orEmpty())
                "modelstatus" -> ModelStatus(s("model").orEmpty(), s("state").orEmpty(), (o["progress"] as? JsonPrimitive)?.floatOrNull ?: 0f)
                else -> null
            }
        }
    }
}
