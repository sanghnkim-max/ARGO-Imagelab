package dev.argo.imagelab.core

import android.content.ContentUris
import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.provider.MediaStore
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileNotFoundException
import kotlin.math.max
import kotlin.math.roundToInt

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

/**
 * The phone's gallery for programs that take `{gallery: id}` photos (docs/INTERFACES.md §3):
 * one photo per call, read-only, decoded already downscaled so a program over dozens of photos
 * never holds more than one small bitmap. The core calls it off the main thread.
 */
class MediaStoreGallery(private val context: Context) : GalleryProvider {
    override fun readPhoto(id: Long, maxSide: UInt): GalleryReply = try {
        read(id, maxSide.toInt())
    } catch (_: SecurityException) {
        failure("permission", "ImageLab has no access to the phone's photos; the person can allow it in ImageLab's settings")
    } catch (_: FileNotFoundException) {
        failure("not_found", "gallery photo $id does not exist (it may have been deleted)")
    } catch (e: OutOfMemoryError) {
        failure("too_large", "gallery photo $id is too large to read")
    } catch (e: Exception) {
        failure("failed", "gallery photo $id could not be read: ${e.message ?: e.javaClass.simpleName}")
    }

    private fun failure(code: String, message: String) =
        GalleryReply(code, message, "", 0u, 0u, ByteArray(0))

    private class Row(val name: String, val width: Int, val height: Int, val orientation: Int)

    private fun row(uri: android.net.Uri, id: Long): Row {
        val columns = arrayOf(
            MediaStore.Images.Media.DISPLAY_NAME, MediaStore.Images.Media.WIDTH,
            MediaStore.Images.Media.HEIGHT, MediaStore.Images.Media.ORIENTATION,
        )
        context.contentResolver.query(uri, columns, null, null, null)?.use { c ->
            if (c.moveToFirst()) return Row(c.getString(0) ?: "photo-$id.jpg", c.getInt(1), c.getInt(2), c.getInt(3))
        }
        throw FileNotFoundException("no gallery photo $id")
    }

    private fun read(id: Long, maxSide: Int): GalleryReply {
        val uri = ContentUris.withAppendedId(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, id)
        val row = row(uri, id)
        return if (maxSide <= 0) original(uri, id, row) else scaled(uri, row, maxSide)
    }

    /** The file's own bytes (an edit or export works on a copy), at most [MAX_ORIGINAL_BYTES]. */
    private fun original(uri: android.net.Uri, id: Long, row: Row): GalleryReply {
        val out = ByteArrayOutputStream()
        val input = context.contentResolver.openInputStream(uri) ?: throw FileNotFoundException("no gallery photo $id")
        input.use { stream ->
            val buf = ByteArray(1 shl 16)
            while (true) {
                val n = stream.read(buf)
                if (n < 0) break
                if (out.size() + n > MAX_ORIGINAL_BYTES) return failure("too_large", "gallery photo $id is larger than ${MAX_ORIGINAL_BYTES shr 20} MB")
                out.write(buf, 0, n)
            }
        }
        var (w, h) = row.width to row.height
        if (w <= 0 || h <= 0) {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            val bytes = out.toByteArray()
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
            w = bounds.outWidth; h = bounds.outHeight
        }
        if (row.orientation == 90 || row.orientation == 270) w = h.also { h = w }
        return GalleryReply("", "", row.name, w.coerceAtLeast(0).toUInt(), h.coerceAtLeast(0).toUInt(), out.toByteArray())
    }

    /** Decoded at a power-of-two reduction, then fitted to [maxSide]; upright (ImageDecoder applies EXIF). */
    private fun scaled(uri: android.net.Uri, row: Row, maxSide: Int): GalleryReply {
        var longSide = max(row.width, row.height)
        val source = ImageDecoder.createSource(context.contentResolver, uri)
        val decoded = ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            longSide = max(longSide, max(info.size.width, info.size.height))
            var sample = 1
            while (longSide / (sample * 2) >= maxSide) sample *= 2
            if (sample > 1) decoder.setTargetSampleSize(sample)
        }
        val fitted = run {
            val long = max(decoded.width, decoded.height)
            if (long <= maxSide) decoded else {
                val f = maxSide.toFloat() / long
                Bitmap.createScaledBitmap(decoded, max(1, (decoded.width * f).roundToInt()), max(1, (decoded.height * f).roundToInt()), true)
            }
        }
        try {
            val out = ByteArrayOutputStream()
            fitted.compress(Bitmap.CompressFormat.JPEG, 90, out)
            // The original's upright size: its long side, in the decoded image's proportions.
            val (w, h) = if (decoded.width >= decoded.height) {
                longSide to (longSide.toFloat() * decoded.height / decoded.width).roundToInt()
            } else {
                (longSide.toFloat() * decoded.width / decoded.height).roundToInt() to longSide
            }
            return GalleryReply("", "", row.name, w.toUInt(), h.toUInt(), out.toByteArray())
        } finally {
            if (fitted !== decoded) fitted.recycle()
            decoded.recycle()
        }
    }

    private companion object {
        const val MAX_ORIGINAL_BYTES = 64 shl 20
    }
}
