package dev.argo.imagelab.core

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/** Copies picked images into the workspace (`imports/`), with sanitized, unique names. */
object Imports {
    private const val MAX_BYTES = 512L * 1024 * 1024

    /** Read [uri] fully: (display name, bytes), or null if it cannot be read. */
    suspend fun read(context: Context, uri: Uri): Pair<String, ByteArray>? = withContext(Dispatchers.IO) {
        runCatching {
            val resolver = context.contentResolver
            val name = resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE), null, null, null)?.use { c ->
                if (c.moveToFirst()) {
                    val size = if (c.isNull(1)) 0L else c.getLong(1)
                    require(size <= MAX_BYTES) { "the picked file is too large" }
                    c.getString(0)
                } else null
            } ?: "photo.jpg"
            val bytes = resolver.openInputStream(uri)?.use { it.readBytes() } ?: return@runCatching null
            name to bytes
        }.getOrNull()
    }

    /** Write into `<workspace>/imports/`; returns the workspace-relative path. */
    suspend fun store(workspace: File, name: String, bytes: ByteArray): String = withContext(Dispatchers.IO) {
        val dir = File(workspace, "imports").apply { mkdirs() }
        val clean = sanitize(name)
        val base = clean.substringBeforeLast('.', clean)
        val ext = clean.substringAfterLast('.', "jpg")
        var file = File(dir, "$base.$ext")
        var n = 1
        while (file.exists()) file = File(dir, "$base-${n++}.$ext")
        file.writeBytes(bytes)
        "imports/${file.name}"
    }

    fun sanitize(name: String): String {
        val leaf = name.substringAfterLast('/').substringAfterLast('\\')
        val cleaned = leaf.replace(Regex("[^A-Za-z0-9._ -]"), "_").trim().trimStart('.')
        return cleaned.take(96).ifEmpty { "photo.jpg" }
    }
}
