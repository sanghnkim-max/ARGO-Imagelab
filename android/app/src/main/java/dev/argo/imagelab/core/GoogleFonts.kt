package dev.argo.imagelab.core

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.net.URLEncoder
import java.util.concurrent.TimeUnit

/** One Google Fonts family from the catalog. */
data class FontFamily(val id: String, val family: String, val category: String, val subsets: List<String>, val license: String)

/**
 * Google Fonts for text layers (open-licensed: OFL / Apache / UFL), the way ARGO-Workspace gets
 * them: the catalog from Fontsource's open API (no key), the font file from the google/fonts
 * repository (the complete file — Fontsource splits CJK fonts into ~100 chunks), else
 * Fontsource's single-subset TTF. The app downloads with the platform's TLS; the core only
 * checks and registers the file (`fonts.install`).
 */
object GoogleFonts {
    private val http = OkHttpClient.Builder().callTimeout(120, TimeUnit.SECONDS).build()
    private val json = Json { ignoreUnknownKeys = true }

    /** A short starter list (Latin and Korean display/body faces). */
    val popular = listOf(
        "Roboto", "Inter", "Montserrat", "Playfair Display", "Oswald", "Bebas Neue", "Lobster", "Pacifico",
        "Dancing Script", "Noto Sans KR", "Do Hyeon", "Black Han Sans", "Jua", "Nanum Pen Script",
    )

    private fun get(url: String): String? = runCatching {
        http.newCall(Request.Builder().url(url).build()).execute().use { r -> if (r.isSuccessful) r.body.string() else null }
    }.getOrNull()

    private fun JsonObject.str(k: String) = (this[k] as? JsonPrimitive)?.contentOrNull

    private fun parse(o: JsonObject) = FontFamily(
        id = o.str("id").orEmpty(),
        family = o.str("family").orEmpty(),
        category = o.str("category").orEmpty(),
        subsets = (o["subsets"] as? JsonArray).orEmpty().mapNotNull { (it as? JsonPrimitive)?.contentOrNull },
        license = o.str("license").orEmpty(),
    )

    /** Google families whose name contains [query] (blocking; call off the main thread). */
    fun search(query: String, limit: Int = 30): List<FontFamily> {
        val body = get("https://api.fontsource.org/v1/fonts?type=google") ?: return emptyList()
        val all = (runCatching { json.parseToJsonElement(body) }.getOrNull() as? JsonArray).orEmpty().mapNotNull { (it as? JsonObject)?.let(::parse) }
        val q = query.trim().lowercase()
        return all.filter { q.isEmpty() || it.family.lowercase().contains(q) }
            .sortedBy { if (it.family.lowercase().startsWith(q)) 0 else 1 }
            .take(limit)
    }

    private fun lookup(family: String): FontFamily? {
        val body = get("https://api.fontsource.org/v1/fonts?family=" + URLEncoder.encode(family, "UTF-8")) ?: return null
        return (runCatching { json.parseToJsonElement(body) }.getOrNull() as? JsonArray)?.firstNotNullOfOrNull { (it as? JsonObject)?.let(::parse) }
    }

    /** The regular face's file in google/fonts (`ofl|apache|ufl/<dir>/METADATA.pb`), as a raw URL. */
    private fun repoFile(family: String): String? {
        val dir = family.lowercase().replace(Regex("[^a-z0-9]"), "")
        for (lic in listOf("ofl", "apache", "ufl")) {
            val base = "https://raw.githubusercontent.com/google/fonts/main/$lic/$dir"
            val meta = get("$base/METADATA.pb") ?: continue
            val faces = Regex("""fonts \{(.*?)\}""", RegexOption.DOT_MATCHES_ALL).findAll(meta).map { it.groupValues[1] }.toList()
            fun field(face: String, k: String) = Regex("""$k: "?([^"\n]+)"?""").find(face)?.groupValues?.get(1)?.trim()
            val pick = faces.firstOrNull { field(it, "style") == "normal" && field(it, "weight") == "400" }
                ?: faces.firstOrNull { field(it, "style") == "normal" }
                ?: faces.firstOrNull()
            val file = pick?.let { field(it, "filename") } ?: continue
            return "$base/" + file.replace("[", "%5B").replace("]", "%5D")
        }
        return null
    }

    /** Fontsource's TTF for weight 400 normal: the Korean subset for Korean fonts, else the default subset. */
    private fun fontsourceFile(f: FontFamily): String? {
        val body = get("https://api.fontsource.org/v1/fonts/${f.id}") ?: return null
        val o = runCatching { json.parseToJsonElement(body) }.getOrNull() as? JsonObject ?: return null
        val normal = ((o["variants"] as? JsonObject)?.get("400") as? JsonObject)?.get("normal") as? JsonObject ?: return null
        val subset = when {
            "korean" in f.subsets && normal.containsKey("korean") -> "korean"
            else -> o.str("defSubset") ?: "latin"
        }
        return (((normal[subset] as? JsonObject)?.get("url") as? JsonObject)?.get("ttf") as? JsonPrimitive)?.contentOrNull
    }

    /**
     * Download [family] into `<workspace>/downloads/` (blocking). Returns the workspace-relative
     * path and the licence, or an error message.
     */
    fun download(workspace: File, family: String): Result<Pair<String, String>> = runCatching {
        val info = lookup(family) ?: error("$family is not in the Google Fonts catalog")
        val url = repoFile(family) ?: fontsourceFile(info) ?: error("no downloadable file for $family")
        val name = url.substringAfterLast('/').replace("%5B", "[").replace("%5D", "]").replace(Regex("[^A-Za-z0-9._\\[\\]-]"), "_")
        val dest = File(workspace, "downloads/$name").apply { parentFile?.mkdirs() }
        http.newCall(Request.Builder().url(url).build()).execute().use { r ->
            check(r.isSuccessful) { "HTTP ${r.code} for $family" }
            dest.outputStream().use { out -> r.body.byteStream().copyTo(out) }
        }
        "downloads/$name" to info.license
    }
}
