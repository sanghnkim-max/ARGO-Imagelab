package dev.argo.imagelab.agent

import android.content.ContentUris
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.provider.MediaStore
import android.util.Base64
import dev.argo.imagelab.BuildConfig
import dev.argo.imagelab.ImageLabApp
import dev.argo.imagelab.core.Gallery
import dev.argo.imagelab.core.Lab
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.longOrNull
import kotlinx.serialization.json.put
import java.io.ByteArrayOutputStream

/**
 * ImageLab as an MCP server for the ARGO app (docs/INTERFACES.md §12): one JSON-RPC message in,
 * one out, carried over [LabService]'s `mcp` op. ARGO mounts it under the server key `imagelab`,
 * so its tools reach the model as `imagelab__open`, `imagelab__read`, `imagelab__edit`,
 * `imagelab__help` and `imagelab__save`.
 *
 * The three MCP server features:
 * - **Tools** (`tools/list`, `tools/call`): `open`, `read`, `edit`, `help`, `save`, and
 *   `overwrite`, the one destructive tool (a client asks before each call).
 * - **Resources** (`resources/list`, `resources/templates/list`, `resources/read`): the open
 *   document (`imagelab://document`), its history, a preview image, and the function reference
 *   (`imagelab://reference`, one area at `imagelab://reference/{area}`).
 * - **Prompts** (`prompts/list`, `prompts/get`): `enhance`, `sky` and `caption`, the common edits
 *   as ready requests.
 *
 * Also `initialize`, `ping`, and the notifications `notifications/initialized` and
 * `notifications/cancelled` (which stops the running programs).
 *
 * `read` and `edit` are ImageLab's own `lab_read` and `lab_script` (the same tools the in-app
 * agent and the HTTP server serve), so every edit is one undoable history step the person sees
 * live in the editor. Images come back as standard `image` content blocks, re-encoded as JPEG
 * unless they carry transparency.
 */
class CompanionMcp(private val app: ImageLabApp) {
    /** The response to [message] (JSON-RPC text), or null for a notification. */
    fun handle(message: String): String? {
        val request = runCatching { Json.parseToJsonElement(message) as? JsonObject }.getOrNull()
            ?: return error(JsonNull, PARSE_ERROR, "not a JSON-RPC message")
        val method = (request["method"] as? JsonPrimitive)?.contentOrNull
        val id = request["id"]
        val params = request["params"] as? JsonObject ?: JsonObject(emptyMap())
        if (id == null || id is JsonNull) {
            // A notification: nothing answers it.
            if (method == "notifications/cancelled") runBlocking { app.lab.io { it.cancelScripts() } }
            return null
        }
        return runCatching {
            when (method) {
                "initialize" -> result(id, initialize(params))
                "ping" -> result(id, JsonObject(emptyMap()))
                "tools/list" -> result(id, buildJsonObject { put("tools", TOOLS) })
                "tools/call" -> result(id, call(params))
                "resources/list" -> result(id, buildJsonObject { put("resources", RESOURCES) })
                "resources/templates/list" -> result(id, buildJsonObject { put("resourceTemplates", RESOURCE_TEMPLATES) })
                "resources/read" -> readResource(id, params)
                "prompts/list" -> result(id, buildJsonObject { put("prompts", PROMPTS) })
                "prompts/get" -> prompt(id, params)
                null -> error(id, INVALID_REQUEST, "no method")
                else -> error(id, METHOD_NOT_FOUND, "ImageLab does not serve `$method`")
            }
        }.getOrElse { error(id, INTERNAL_ERROR, Lab.describe(it)) }
    }

    private fun initialize(params: JsonObject): JsonObject {
        val asked = (params["protocolVersion"] as? JsonPrimitive)?.contentOrNull
        return buildJsonObject {
            put("protocolVersion", if (asked in PROTOCOL_VERSIONS) asked else PROTOCOL_VERSIONS.first())
            put("capabilities", buildJsonObject {
                put("tools", JsonObject(emptyMap()))
                put("resources", JsonObject(emptyMap()))
                put("prompts", JsonObject(emptyMap()))
            })
            put("serverInfo", buildJsonObject {
                put("name", "imagelab")
                put("title", "ImageLab")
                put("version", BuildConfig.VERSION_NAME)
            })
            put("instructions", INSTRUCTIONS)
        }
    }

    /** A `resources/read`: the resource's contents, or `-32002` for a URI ImageLab does not serve. */
    private fun readResource(id: JsonElement, params: JsonObject): String {
        val uri = params.text("uri") ?: return error(id, INVALID_PARAMS, "resources/read needs `uri`")
        val contents: JsonObject = when {
            uri == DOCUMENT -> jsonContents(uri, labCall("doc.inspect"))
            uri == HISTORY -> jsonContents(uri, labCall("doc.history"))
            uri == PREVIEW -> previewContents(uri) ?: return error(id, RESOURCE_NOT_FOUND, "no photo is open in ImageLab, so there is no preview")
            uri == REFERENCE || uri.startsWith("$REFERENCE/") -> {
                val area = uri.removePrefix(REFERENCE).removePrefix("/").takeIf { it.isNotBlank() }
                val help = lab("lab_help", buildJsonObject { area?.let { put("area", it) } })
                if ((help["isError"] as? JsonPrimitive)?.contentOrNull == "true") {
                    return error(id, RESOURCE_NOT_FOUND, "ImageLab has no reference for `$area`")
                }
                textContents(uri, "text/markdown", blocksText(help))
            }
            else -> return error(id, RESOURCE_NOT_FOUND, "ImageLab serves no resource `$uri`")
        }
        return result(id, buildJsonObject { put("contents", buildJsonArray { add(contents) }) })
    }

    /** ImageLab function [fn] with no parameters, read-only: its JSON answer, or `{"error"}` when it failed. */
    private fun labCall(fn: String): String =
        runBlocking { app.lab.io { it.call(fn, "{}", "agent") } }
            .getOrElse { buildJsonObject { put("error", Lab.describe(it)) }.toString() }

    /** The open document as a JPEG, through a read-only program; null when nothing is open. */
    private fun previewContents(uri: String): JsonObject? {
        val shown = program("lab_read", buildJsonObject {
            put("code", "return {image: (await lab.doc.preview({maxSide: 1024})).image}")
            put("description", "Preview for ARGO")
        })
        if ((shown["isError"] as? JsonPrimitive)?.contentOrNull == "true") return null
        val image = (shown["content"] as? JsonArray)?.firstOrNull { (it as? JsonObject)?.text("type") == "image" } as? JsonObject ?: return null
        return buildJsonObject {
            put("uri", uri)
            put("mimeType", image.text("mimeType") ?: "image/jpeg")
            put("blob", image.text("data") ?: return null)
        }
    }

    /** A `prompts/get`: the prompt's messages, its arguments filled in. */
    private fun prompt(id: JsonElement, params: JsonObject): String {
        val name = params.text("name") ?: return error(id, INVALID_PARAMS, "prompts/get needs `name`")
        val args = params["arguments"] as? JsonObject ?: JsonObject(emptyMap())
        fun arg(key: String) = args.text(key)
        val photo = arg("photo")?.let { "Open gallery photo $it in ImageLab with `open`. " } ?: "Work on the photo open in ImageLab (open one with `open` if none is). "
        val (description, text) = when (name) {
            "enhance" -> "A natural overall enhancement" to
                photo + "Look at it first with `read` (a preview and its histogram), then make a natural overall " +
                "enhancement with `edit`: exposure, white balance, contrast and saturation, nothing heavy-handed. " +
                (arg("style")?.let { "The look the person wants: $it. " } ?: "") +
                "Look at the result before you say it is done, then keep it with `save`."
            "sky" -> "Change only the sky" to
                photo + "Change only the sky: make it ${arg("look") ?: "a deeper, richer blue"}. Make the sky the " +
                "selection with select.sky (edgeFill on, so no halo is left along the skyline), edit it with `edit`, " +
                "then select.none. Check the skyline closely in a preview before saying it is done, then `save`."
            "caption" -> {
                val words = arg("text") ?: return error(id, INVALID_PARAMS, "the caption prompt needs `text`")
                "Add a caption" to
                    photo + "Add the caption \"$words\" ${arg("position")?.let { "at the $it" } ?: "near the bottom"}, " +
                    "in a clean, readable type that suits the photo (lab.help('fonts') lists them), with enough contrast " +
                    "against what is behind it. Look at a preview, then `save`."
            }
            else -> return error(id, INVALID_PARAMS, "ImageLab has no prompt `$name`")
        }
        return result(id, buildJsonObject {
            put("description", description)
            put("messages", buildJsonArray {
                add(buildJsonObject {
                    put("role", "user")
                    put("content", buildJsonObject {
                        put("type", "text")
                        put("text", text)
                    })
                })
            })
        })
    }

    /** A `tools/call`: the `CallToolResult`, `isError` for a refusal the model can act on. */
    private fun call(params: JsonObject): JsonObject {
        val name = (params["name"] as? JsonPrimitive)?.contentOrNull ?: return failed("tools/call needs `name`")
        val args = params["arguments"] as? JsonObject ?: JsonObject(emptyMap())
        return when (name) {
            "read" -> program("lab_read", args)
            "edit" -> program("lab_script", args).also { app.lab.requestRefresh() }
            "help" -> lab("lab_help", buildJsonObject { args.text("area")?.let { put("area", it) } })
            "open" -> open(args)
            "save" -> save(args)
            "overwrite" -> overwrite(args)
            else -> failed("unknown tool `$name`; ImageLab serves ${TOOL_NAMES.joinToString()}")
        }
    }

    private fun program(labTool: String, args: JsonObject): JsonObject {
        val code = args.text("code") ?: return failed("`code` is required: the body of an async JavaScript function")
        return lab(labTool, buildJsonObject {
            put("code", code)
            put("description", args.text("description") ?: "ARGO's program")
            (args["timeoutMs"] as? JsonPrimitive)?.longOrNull?.let { put("timeoutMs", it.coerceIn(1_000L, MAX_PROGRAM_MS)) }
        })
    }

    /** ImageLab's own MCP tool [labTool], its images re-encoded for the trip. */
    private fun lab(labTool: String, arguments: JsonObject): JsonObject {
        val request = buildJsonObject {
            put("name", labTool)
            put("arguments", arguments)
        }
        val answer = runBlocking { app.lab.io { it.labMcpRequest("tools/call", request.toString()) } }.getOrThrow()
        val result = Json.parseToJsonElement(answer) as? JsonObject ?: return failed("ImageLab's tool answered no result")
        val blocks = result["content"] as? JsonArray ?: return result
        return JsonObject(result + ("content" to JsonArray(blocks.map { block -> (block as? JsonObject)?.let(::jpeg) ?: block })))
    }

    /** Open gallery photo `photo` (a MediaStore id) as the document. */
    private fun open(args: JsonObject): JsonObject {
        val id = (args["photo"] as? JsonPrimitive)?.longOrNull ?: return failed("`photo` is required: a photo id from the gallery")
        val resolver = app.contentResolver
        val uri = ContentUris.withAppendedId(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, id)
        val name = runCatching {
            resolver.query(uri, arrayOf(MediaStore.Images.Media.DISPLAY_NAME), null, null, null)?.use { c ->
                if (c.moveToFirst()) c.getString(0) else null
            }
        }.getOrNull() ?: "photo-$id.jpg"
        val bytes = try {
            resolver.openInputStream(uri)?.use { input -> readCapped(input) }
        } catch (_: SecurityException) {
            return failed(NO_PHOTO_ACCESS)
        } catch (_: java.io.FileNotFoundException) {
            null
        } ?: return failed("photo $id could not be read: it may have been deleted")
        val opened = runBlocking { app.lab.io { it.importBytes(name, bytes) } }.getOrThrow()
        app.lab.requestRefresh()
        val document = Json.parseToJsonElement(opened)
        return ok(
            "Photo $id ($name) is open in ImageLab: `read` and `edit` work on it now. The gallery original is unchanged.",
            buildJsonObject {
                put("opened", true)
                put("photo", id)
                put("document", document)
            },
        )
    }

    /** Export the document to the gallery (Pictures/ImageLab): a new photo, the original untouched. */
    private fun save(args: JsonObject): JsonObject {
        val format = (args.text("format") ?: "jpg").lowercase().let { if (it == "jpeg") "jpg" else it }
        if (format !in setOf("jpg", "png", "webp")) return failed("format must be jpg, png or webp")
        // Always a new file: a name already taken gets the next number.
        val stamp = System.currentTimeMillis()
        var fileName = "imagelab-$stamp.$format"
        var n = 1
        while (java.io.File(app.workspace, "exports/$fileName").exists()) fileName = "imagelab-$stamp-${n++}.$format"
        val rel = "exports/$fileName"
        val body = buildJsonObject {
            put("path", rel)
            put("format", format)
            (args["quality"] as? JsonPrimitive)?.longOrNull?.let { put("quality", it.coerceIn(1, 100)) }
            (args["maxSide"] as? JsonPrimitive)?.longOrNull?.let { put("maxSide", it.coerceIn(16, 16384)) }
        }
        runBlocking { app.lab.io { it.call("doc.export", body.toString(), "agent") } }.getOrThrow()
        val saved = Gallery.save(app, java.io.File(app.workspace, rel), fileName)
        return ok(
            "Saved as a new gallery photo (id ${saved.mediaId}, Pictures/ImageLab/$fileName). The original is unchanged.",
            buildJsonObject {
                put("saved", true)
                put("photo", saved.mediaId)
                put("uri", saved.uri)
                put("album", "Pictures/ImageLab")
                // The saved photo as a photo result: ARGO shows it to the person with the reply.
                put("photos", buildJsonArray {
                    add(buildJsonObject {
                        put("id", saved.mediaId)
                        put("name", "Pictures/ImageLab / $fileName")
                    })
                })
            },
        )
    }

    /**
     * Replace an existing file in ImageLab's workspace with the open document: the one write that
     * destroys something, so it is its own tool, marked destructive, which the client asks the
     * person about before every call. Programs are refused it (`needs_approval`).
     */
    private fun overwrite(args: JsonObject): JsonObject {
        val path = args.text("path") ?: return failed("`path` is required: the existing file to replace, e.g. exports/imagelab-1.jpg")
        if (!java.io.File(app.workspace, path).exists()) return failed("`$path` does not exist; to keep a result, use `save`, which makes a new photo")
        val body = buildJsonObject {
            put("path", path)
            args.text("format")?.let { put("format", it) }
            (args["quality"] as? JsonPrimitive)?.longOrNull?.let { put("quality", it.coerceIn(1, 100)) }
            (args["maxSide"] as? JsonPrimitive)?.longOrNull?.let { put("maxSide", it.coerceIn(16, 16384)) }
        }
        runBlocking { app.lab.io { it.call("doc.exportOverwrite", body.toString(), "agent") } }.getOrThrow()
        return ok("Replaced $path with the open document.", buildJsonObject {
            put("replaced", true)
            put("path", path)
        })
    }

    private fun readCapped(input: java.io.InputStream): ByteArray {
        val out = ByteArrayOutputStream()
        val buf = ByteArray(1 shl 16)
        var total = 0L
        while (true) {
            val n = input.read(buf)
            if (n < 0) break
            total += n
            require(total <= MAX_PHOTO_BYTES) { "the photo is larger than ${MAX_PHOTO_BYTES shr 20} MB" }
            out.write(buf, 0, n)
        }
        return out.toByteArray()
    }

    /**
     * An image block as JPEG: the client keeps it, and a photo-sized PNG is several megabytes.
     * A mask or cut-out with any transparent pixel stays PNG: every pixel's alpha is checked
     * ([fullyOpaque]), since one missed would be lost for good. Anything that fails stays as sent.
     */
    private fun jpeg(block: JsonObject): JsonObject {
        if (block.text("type") != "image" || block.text("mimeType") == "image/jpeg") return block
        return runCatching {
            val bytes = Base64.decode(block.text("data") ?: return block, Base64.DEFAULT)
            val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: return block
            try {
                if (bitmap.hasAlpha() && !fullyOpaque(bitmap)) return block
                val out = ByteArrayOutputStream()
                bitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out)
                buildJsonObject {
                    put("type", "image")
                    put("mimeType", "image/jpeg")
                    put("data", Base64.encodeToString(out.toByteArray(), Base64.NO_WRAP))
                }
            } finally {
                bitmap.recycle()
            }
        }.getOrDefault(block)
    }

    /** Whether every pixel of [bitmap] is opaque: read a row at a time, stopping at the first that is not. */
    private fun fullyOpaque(bitmap: Bitmap): Boolean {
        val row = IntArray(bitmap.width)
        for (y in 0 until bitmap.height) {
            bitmap.getPixels(row, 0, bitmap.width, 0, y, bitmap.width, 1)
            for (pixel in row) if ((pixel ushr 24) != 0xFF) return false
        }
        return true
    }

    companion object {
        /** Newest first: what ImageLab answers when the client asks for one it does not know. */
        private val PROTOCOL_VERSIONS = listOf("2025-06-18", "2025-03-26", "2024-11-05")

        private const val PARSE_ERROR = -32700
        private const val INVALID_REQUEST = -32600
        private const val METHOD_NOT_FOUND = -32601
        private const val INVALID_PARAMS = -32602
        private const val RESOURCE_NOT_FOUND = -32002

        private const val DOCUMENT = "imagelab://document"
        private const val HISTORY = "imagelab://document/history"
        private const val PREVIEW = "imagelab://document/preview"
        private const val REFERENCE = "imagelab://reference"

        private val RESOURCES: JsonArray = buildJsonArray {
            add(resource(DOCUMENT, "document", "The open document", "The photo open in ImageLab: size, layers, selection, what it was opened from.", "application/json"))
            add(resource(HISTORY, "history", "Edit history", "The open document's undoable steps, newest last.", "application/json"))
            add(resource(PREVIEW, "preview", "Preview", "The open document as it looks now, at most 1024 px a side.", "image/jpeg"))
            add(resource(REFERENCE, "reference", "Function reference", "Every area of ImageLab's program API and its functions.", "text/markdown"))
        }

        private val RESOURCE_TEMPLATES: JsonArray = buildJsonArray {
            add(buildJsonObject {
                put("uriTemplate", "$REFERENCE/{area}")
                put("name", "reference-area")
                put("title", "Function reference for one area")
                put("description", "One area (\"edit\", \"select\") or one function (\"select.fromMask\"): descriptions, parameter schemas and effects.")
                put("mimeType", "text/markdown")
            })
        }

        private val PROMPTS: JsonArray = buildJsonArray {
            add(promptOf("enhance", "Enhance a photo", "A natural overall enhancement: exposure, white balance, contrast, saturation.",
                "photo" to "A gallery photo id; the open photo when left out.", "style" to "The look wanted, e.g. \"warm and soft\"."))
            add(promptOf("sky", "Change the sky", "Change only the sky, with no halo along the skyline.",
                "photo" to "A gallery photo id; the open photo when left out.", "look" to "What the sky should look like; a deeper blue by default."))
            add(promptOf("caption", "Add a caption", "Add a line of text to the photo in a readable type.",
                "text" to "!The caption.", "photo" to "A gallery photo id; the open photo when left out.", "position" to "Where, e.g. \"bottom\" or \"top left\"."))
        }

        private fun resource(uri: String, name: String, title: String, description: String, mime: String) = buildJsonObject {
            put("uri", uri)
            put("name", name)
            put("title", title)
            put("description", description)
            put("mimeType", mime)
        }

        /** A prompt's listing; an argument description starting with `!` is a required argument. */
        private fun promptOf(name: String, title: String, description: String, vararg arguments: Pair<String, String>) = buildJsonObject {
            put("name", name)
            put("title", title)
            put("description", description)
            put("arguments", buildJsonArray {
                for ((arg, about) in arguments) add(buildJsonObject {
                    put("name", arg)
                    put("description", about.removePrefix("!"))
                    put("required", about.startsWith("!"))
                })
            })
        }

        private fun jsonContents(uri: String, json: String) = textContents(uri, "application/json", json)

        private fun textContents(uri: String, mime: String, text: String) = buildJsonObject {
            put("uri", uri)
            put("mimeType", mime)
            put("text", text)
        }

        /** The text blocks of a `CallToolResult`, joined. */
        private fun blocksText(result: JsonObject): String =
            (result["content"] as? JsonArray).orEmpty()
                .mapNotNull { (it as? JsonObject)?.takeIf { b -> b.text("type") == "text" }?.text("text") }
                .joinToString("\n")
        private const val INTERNAL_ERROR = -32603

        private const val MAX_PROGRAM_MS = 590_000L
        private const val MAX_PHOTO_BYTES = 200L shl 20
        private const val JPEG_QUALITY = 88

        const val NO_PHOTO_ACCESS =
            "ImageLab has no access to the phone's photos. The person can open ImageLab and allow photo access, then try again."

        private const val INSTRUCTIONS =
            "ImageLab is the phone's photo editor. Open a gallery photo with `open`, look at it with `read` (return a " +
                "preview image), change it with `edit`, and keep the result with `save`, which makes a new gallery photo. " +
                "The gallery original is never changed, and every edit is one undoable step in ImageLab."

        private const val PROGRAM_RULES =
            "The program is the body of an async JavaScript function: use `await` and `return` the result (JSON). " +
                "Call ImageLab as `await lab.<area>.<fn>({...})`; `lab.help('<area>')` (or the `help` tool) gives an area's " +
                "functions with parameter schemas. Areas: doc (open document: open, inspect, preview, history, undo, export), " +
                "edit (PhotoCraft engine commands via edit.run {id, params}, looks, recipes; edit.commands() indexes the command " +
                "groups and edit.commands({filter: 'curves'}) gives ids with their params), " +
                "select (subject, sky, object, colorRange, magicWand, fromMask, none), segment (SAM masks by point/box/text), " +
                "cv (OpenCV-style ops on image handles), analyze (quality, histogram, contact sheets), library, batch, fonts. " +
                "Images and masks are handles (\"img:N\", \"mask:N\"), never bytes; any `{image: \"img:N\"}` in the returned " +
                "value comes back to you as an image (e.g. `return {image: (await lab.doc.preview({maxSide: 1024})).image}`). " +
                "To edit part of a photo, make the region the selection (select.subject, or select.fromMask with a mask) " +
                "before edit.run, then select.none. A failed call throws LabCallError {code, message}. No fetch, timers or files."

        private val PROGRAM_SCHEMA = """
            {"type":"object","properties":{
              "code":{"type":"string","description":"The program body (async JavaScript)."},
              "description":{"type":"string","description":"One short line for the person: what this program does."},
              "timeoutMs":{"type":"integer","minimum":1000,"description":"Wall-time budget; default 120000."}
            },"required":["code","description"],"additionalProperties":false}
        """

        private val TOOLS: JsonArray = buildJsonArray {
            add(tool(
                "open", "Open a gallery photo",
                "Open a photo from the phone's gallery in ImageLab, so `read` and `edit` work on it. `photo`: the photo's " +
                    "gallery (MediaStore) id, as the phone's photo search gives it. ImageLab edits its own copy; the gallery " +
                    "original stays as it is.",
                """{"type":"object","properties":{"photo":{"type":"integer","description":"The photo's gallery id."}},"required":["photo"],"additionalProperties":false}""",
                readOnly = false, idempotent = true,
            ))
            add(tool(
                "read", "Look at the photo",
                "Look at and measure the photo open in ImageLab without changing anything: render a preview to see it, " +
                    "inspect layers and history, histograms, quality scores, find objects and masks. Open a gallery photo " +
                    "first with `open`. Any function that changes something throws LabCallError{code:\"read_only\"}. $PROGRAM_RULES",
                PROGRAM_SCHEMA, readOnly = true, idempotent = true,
            ))
            add(tool(
                "edit", "Edit the photo",
                "Edit the photo open in ImageLab: tone and color, curves, filters, looks, retouching, crops and transforms, " +
                    "selections and masks, text and layers, using its PhotoCraft engine. Each command is one undoable step " +
                    "the person sees live in ImageLab; the gallery original is never changed. Open a gallery photo first with " +
                    "`open` (or create a canvas with lab.doc.new), look at the result (return a preview image) before saying " +
                    "it is done, and keep it with `save` when the person wants it. $PROGRAM_RULES",
                PROGRAM_SCHEMA, readOnly = false, idempotent = false,
            ))
            add(tool(
                "help", "Function reference",
                "ImageLab's function reference: with no area, every area and its function names; with an area (\"edit\") " +
                    "or one function (\"select.fromMask\"), full descriptions, parameter schemas and effects. For PhotoCraft " +
                    "engine command ids and their parameters, call lab.edit.commands({filter}) in `read`.",
                """{"type":"object","properties":{"area":{"type":"string","description":"An area (\"edit\") or a function (\"select.fromMask\"); omit for all areas."}},"additionalProperties":false}""",
                readOnly = true, idempotent = true,
            ))
            add(tool(
                "save", "Save to the gallery",
                "Save the photo open in ImageLab, with its edits, as a new photo in the gallery (album Pictures/ImageLab). " +
                    "Answers the new photo's gallery id. The original stays as it was.",
                """{"type":"object","properties":{
                  "format":{"type":"string","enum":["jpg","png","webp"],"description":"Default jpg."},
                  "quality":{"type":"integer","minimum":1,"maximum":100,"description":"jpg/webp quality; default 92."},
                  "maxSide":{"type":"integer","minimum":16,"description":"Shrink so the long side is at most this many pixels."}
                },"additionalProperties":false}""",
                readOnly = false, idempotent = false,
            ))
            add(tool(
                "overwrite", "Replace an existing file",
                "Replace an existing file in ImageLab's workspace (exports/, edits/) with the open photo. It cannot be " +
                    "undone, so use it only when the person asked to replace that file; to keep a result, use `save`, " +
                    "which always makes a new photo. Programs may not overwrite (LabCallError `needs_approval`).",
                """{"type":"object","properties":{
                  "path":{"type":"string","description":"The existing file, e.g. exports/imagelab-1.jpg."},
                  "format":{"type":"string","enum":["jpg","png","webp"]},
                  "quality":{"type":"integer","minimum":1,"maximum":100},
                  "maxSide":{"type":"integer","minimum":16}
                },"required":["path"],"additionalProperties":false}""",
                readOnly = false, idempotent = true, destructive = true,
            ))
        }

        private val TOOL_NAMES = TOOLS.map { ((it as JsonObject)["name"] as JsonPrimitive).content }

        private fun tool(name: String, title: String, description: String, schema: String, readOnly: Boolean, idempotent: Boolean, destructive: Boolean = false) =
            buildJsonObject {
                put("name", name)
                put("title", title)
                put("description", description)
                put("inputSchema", Json.parseToJsonElement(schema.trimIndent()))
                put("annotations", buildJsonObject {
                    put("title", title)
                    put("readOnlyHint", readOnly)
                    // Edits are undoable in ImageLab and saving makes a new photo: nothing is lost,
                    // except by `overwrite`, which says so.
                    put("destructiveHint", destructive)
                    put("idempotentHint", idempotent)
                    // Everything stays on the phone, in ImageLab and the gallery.
                    put("openWorldHint", false)
                })
            }

        private fun ok(text: String, structured: JsonObject) = buildJsonObject {
            put("content", buildJsonArray {
                add(buildJsonObject {
                    put("type", "text")
                    put("text", text)
                })
            })
            put("structuredContent", structured)
            put("isError", false)
        }

        private fun failed(message: String) = buildJsonObject {
            put("content", buildJsonArray {
                add(buildJsonObject {
                    put("type", "text")
                    put("text", message)
                })
            })
            put("isError", true)
        }

        private fun result(id: JsonElement, value: JsonElement) = buildJsonObject {
            put("jsonrpc", "2.0")
            put("id", id)
            put("result", value)
        }.toString()

        private fun error(id: JsonElement, code: Int, message: String) = buildJsonObject {
            put("jsonrpc", "2.0")
            put("id", id)
            put("error", buildJsonObject {
                put("code", code)
                put("message", message)
            })
        }.toString()

        private fun JsonObject.text(key: String): String? = (this[key] as? JsonPrimitive)?.contentOrNull?.takeIf { it.isNotBlank() }
    }
}
