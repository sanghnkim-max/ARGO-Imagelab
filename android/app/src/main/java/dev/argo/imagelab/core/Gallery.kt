package dev.argo.imagelab.core

import android.content.ContentUris
import android.content.ContentValues
import android.content.Context
import android.provider.MediaStore
import java.io.File

/** Copies of exports in the phone's gallery (Pictures/ImageLab). */
object Gallery {
    /** Where an export landed in the gallery: its MediaStore id (what ARGO's photos tools take) and URI. */
    data class Saved(val mediaId: Long, val uri: String)

    fun mimeOf(ext: String) = when (ext.lowercase()) {
        "png" -> "image/png"
        "webp" -> "image/webp"
        "tif", "tiff" -> "image/tiff"
        else -> "image/jpeg"
    }

    /** Copy [file] into Pictures/ImageLab as [displayName]. Blocking: call off the main thread. */
    fun save(context: Context, file: File, displayName: String): Saved {
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, displayName)
            put(MediaStore.Images.Media.MIME_TYPE, mimeOf(displayName.substringAfterLast('.', "jpg")))
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/ImageLab")
        }
        val resolver = context.contentResolver
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: error("the gallery refused a new photo")
        try {
            val out = resolver.openOutputStream(uri) ?: error("the gallery refused the file")
            out.use { o -> file.inputStream().use { it.copyTo(o) } }
        } catch (e: Exception) {
            runCatching { resolver.delete(uri, null, null) }
            throw e
        }
        return Saved(ContentUris.parseId(uri), uri.toString())
    }
}
