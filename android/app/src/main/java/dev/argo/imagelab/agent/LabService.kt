package dev.argo.imagelab.agent

import android.app.Service
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Binder
import android.os.IBinder
import android.os.ParcelFileDescriptor
import android.os.Process
import android.util.Log
import dev.argo.imagelab.BuildConfig
import dev.argo.imagelab.ILabService
import dev.argo.imagelab.ImageLabApp
import dev.argo.imagelab.core.Gallery
import dev.argo.imagelab.core.Lab
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.put
import java.io.FileInputStream
import java.util.concurrent.Executors

/**
 * The companion wire: the ARGO app's agent drives ImageLab through [ILabService]
 * (docs/INTERFACES.md §12). Exported, and answered only for a caller whose package is on
 * [ALLOWED] and is signed by one of its pinned certificates: on Android any app can bind an
 * exported service, so the caller's signature, checked on every call, is the access control.
 *
 * The tools are ImageLab's own MCP tools (`lab_read`, `lab_script`, `lab_help`) on the in-process
 * server, the same ones the in-app agent and the HTTP server serve, so an edit ARGO makes is one
 * undoable history step the person sees live in the editor.
 */
class LabService : Service() {
    private val app get() = application as ImageLabApp

    /** Long work (programs run up to 30 minutes) never holds a Binder thread. */
    private val workers = Executors.newCachedThreadPool { r -> Thread(r, "imagelab-companion").apply { isDaemon = true } }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onDestroy() {
        workers.shutdown()
        super.onDestroy()
    }

    private val binder = object : ILabService.Stub() {
        override fun contractVersion(): Int = CONTRACT_VERSION

        override fun call(op: String?, argsJson: String?): ParcelFileDescriptor {
            val refusal = refusal()
            val (read, write) = ParcelFileDescriptor.createPipe()
            workers.execute {
                val reply = if (refusal != null) error(refusal) else runCatching { answer(op.orEmpty(), argsJson.orEmpty()) }
                    .getOrElse { error(Lab.describe(it)) }
                ParcelFileDescriptor.AutoCloseOutputStream(write).use { it.write(reply.toByteArray(Charsets.UTF_8)) }
            }
            return read
        }

        override fun importPhoto(name: String?, photo: ParcelFileDescriptor?): String {
            refusal()?.let { return error(it) }
            if (photo == null) return error("no photo was given")
            return runCatching {
                val bytes = ParcelFileDescriptor.AutoCloseInputStream(photo).use { input ->
                    val out = java.io.ByteArrayOutputStream()
                    val buf = ByteArray(1 shl 16)
                    var total = 0L
                    while (true) {
                        val n = input.read(buf)
                        if (n < 0) break
                        total += n
                        if (total > MAX_PHOTO_BYTES) throw IllegalArgumentException("the photo is larger than ${MAX_PHOTO_BYTES shr 20} MB")
                        out.write(buf, 0, n)
                    }
                    out.toByteArray()
                }
                val fileName = name?.substringAfterLast('/')?.takeIf { it.isNotBlank() } ?: "photo.jpg"
                val opened = runBlocking { app.lab.io { it.importBytes(fileName, bytes) } }.getOrThrow()
                app.lab.requestRefresh()
                ok(Json.parseToJsonElement(opened))
            }.getOrElse { error(Lab.describe(it)) }
        }

        override fun cancel() {
            if (refusal() != null) return
            workers.execute { runBlocking { app.lab.io { it.cancelScripts() } } }
        }
    }

    private fun answer(op: String, argsJson: String): String {
        val args = argsJson.takeIf { it.isNotBlank() }?.let { Json.parseToJsonElement(it) as? JsonObject } ?: JsonObject(emptyMap())
        return when (op) {
            "status" -> {
                val info = runBlocking { app.lab.io { it.sessionInfo() } }.getOrNull()
                ok(buildJsonObject {
                    put("contract", CONTRACT_VERSION)
                    put("version", BuildConfig.VERSION_NAME)
                    put("document", info?.let { runCatching { Json.parseToJsonElement(it) }.getOrNull() } ?: JsonPrimitive(null as String?))
                })
            }
            "tools/list" -> mcp("tools/list", "{}")
            "tools/call" -> {
                val name = (args["name"] as? JsonPrimitive)?.content ?: return error("tools/call needs `name`")
                if (name !in TOOLS) return error("unknown tool `$name`; ImageLab serves ${TOOLS.joinToString()}")
                val result = mcp("tools/call", args.toString())
                // An edit changes the document under the editor: re-render it.
                if (name == "lab_script") app.lab.requestRefresh()
                result
            }
            "export" -> export(args)
            else -> error("unknown op `$op`")
        }
    }

    /**
     * Save the open document (`format` jpg | png | webp, `quality` 1–100, `maxSide`) into
     * exports/ and copy it to the gallery (Pictures/ImageLab): `{path, mediaId, uri}`, the
     * mediaId being what ARGO's photo tools take.
     */
    private fun export(args: JsonObject): String {
        val format = ((args["format"] as? JsonPrimitive)?.content ?: "jpg").lowercase().let { if (it == "jpeg") "jpg" else it }
        if (format !in setOf("jpg", "png", "webp")) return error("format must be jpg, png or webp")
        val stamp = System.currentTimeMillis()
        val rel = "exports/imagelab-$stamp.$format"
        val body = buildJsonObject {
            put("path", rel)
            put("format", format)
            (args["quality"] as? JsonPrimitive)?.content?.toIntOrNull()?.let { put("quality", it.coerceIn(1, 100)) }
            (args["maxSide"] as? JsonPrimitive)?.content?.toIntOrNull()?.let { put("maxSide", it.coerceIn(16, 16384)) }
        }
        runBlocking { app.lab.io { it.call("doc.export", body.toString(), "agent") } }.getOrThrow()
        val saved = Gallery.save(app, java.io.File(app.workspace, rel), "imagelab-$stamp.$format")
        return ok(buildJsonObject {
            put("path", rel)
            put("mediaId", saved.mediaId)
            put("uri", saved.uri)
        })
    }

    private fun mcp(method: String, paramsJson: String): String {
        val result = runBlocking { app.lab.io { it.labMcpRequest(method, paramsJson) } }.getOrThrow()
        return ok(Json.parseToJsonElement(result))
    }

    /** Why the current Binder caller is refused, or null when it is ARGO. */
    private fun refusal(): String? {
        val uid = Binder.getCallingUid()
        if (uid == Process.myUid()) return null
        val packages = packageManager.getPackagesForUid(uid).orEmpty()
        val allowed = packages.firstOrNull { pkg ->
            ALLOWED[pkg]?.any { digest -> packageManager.hasSigningCertificate(pkg, digest, PackageManager.CERT_INPUT_SHA256) } == true
        }
        if (allowed == null) {
            Log.w(TAG, "refused uid $uid (${packages.joinToString()})")
            return "this caller is not allowed to drive ImageLab"
        }
        return null
    }

    companion object {
        private const val TAG = "ImageLabCompanion"
        const val CONTRACT_VERSION = 1
        private const val MAX_PHOTO_BYTES = 200L shl 20
        private val TOOLS = setOf("lab_read", "lab_script", "lab_help")

        /**
         * Who may drive ImageLab: package → SHA-256 digests of its signing certificate. ARGO's
         * debug key (`apps/argo-android/debug.keystore` in ARGOv2, the same one ARGO Browser
         * carries); a release ARGO adds its release certificate here.
         */
        private val ALLOWED: Map<String, List<ByteArray>> = mapOf(
            "com.argo.android" to listOf(hex("4eecf98c331b09ebb54559b60ab544856e21c76f1239fdf9e3916bbf60f63c6a")),
        )

        private fun hex(s: String) = ByteArray(s.length / 2) { i -> s.substring(i * 2, i * 2 + 2).toInt(16).toByte() }

        private fun ok(value: JsonElement) = buildJsonObject { put("ok", value) }.toString()
        private fun error(message: String) = buildJsonObject { put("error", message) }.toString()
    }
}
