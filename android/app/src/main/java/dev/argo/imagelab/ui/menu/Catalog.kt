package dev.argo.imagelab.ui.menu

/**
 * The editor's menus as data: each entry is one PhotoCraft engine command with the parameters a
 * person can change. [CommandSheet] renders any group and [LiveCommand] previews it live, so a
 * new menu item is one line here, not new UI code. Parameter names and ranges are PhotoCraft's
 * (`imagelab commands --json`).
 */

/** One control. `key` is the command's parameter name. */
sealed interface Param {
    val key: String
    val label: String
}

/** A slider; `integer` rounds the value before it is sent. */
data class Slider(
    override val key: String,
    override val label: String,
    val min: Float,
    val max: Float,
    val default: Float,
    val integer: Boolean = true,
    /** Multiplies the shown value before sending (e.g. 0.01 for a % shown, 0..1 sent). */
    val sendScale: Float = 1f,
    /** Shown only while the value of key `first` is `second` (one set of sliders per choice). */
    val showWhen: Pair<String, String>? = null,
) : Param

/** A choice among PhotoCraft's values: (value, label). */
data class Choice(override val key: String, override val label: String, val options: List<Pair<String, String>>, val default: String) : Param

data class Toggle(override val key: String, override val label: String, val default: Boolean) : Param

/** A colour (`#rrggbb`) from a small palette. */
data class ColorParam(override val key: String, override val label: String, val default: String) : Param

/** How an entry reaches the engine. */
enum class Kind {
    /** A pixel command on the active layer (`edit.run`): filters, destructive adjustments. */
    Pixels,
    /** A new adjustment layer (`edit.adjust {kind}`): non-destructive, maskable. */
    AdjustmentLayer,
    /** A layer style effect on the active layer (`edit.run layer.layerStyle.*`). */
    LayerStyle,
    /** A document command applied once (size, canvas); not previewed. */
    Document,
    /** A one-slider look (`edit.look {look, amount, sky?, style?}`): one history step. */
    Look,
    /** Face-aware liquify (`edit.faceLiquify`): sliders placed from the detected faces. */
    Face,
    /** Change an existing adjustment layer (`layer.setAdjustment {layer, …}`); `fixed` holds the layer. */
    AdjustmentEdit,
}

data class MenuItem(
    val label: String,
    /** Engine command id; for [Kind.AdjustmentLayer] the adjustment kind (e.g. `levels`). */
    val command: String,
    val kind: Kind,
    val params: List<Param> = emptyList(),
    /** Fixed params sent with every call. */
    val fixed: Map<String, Any> = emptyMap(),
    /** Preview while values change (false: only on Apply — slow or document-changing commands). */
    val live: Boolean = true,
    /** Starting values by param key (re-editing); missing keys start at their defaults. */
    val initial: Map<String, Any> = emptyMap(),
)

data class MenuGroup(val label: String, val items: List<MenuItem>)

private val blends = listOf(
    "normal" to "Normal", "multiply" to "Multiply", "screen" to "Screen", "overlay" to "Overlay",
    "softLight" to "Soft light", "hardLight" to "Hard light", "colorDodge" to "Color dodge",
    "colorBurn" to "Color burn", "darken" to "Darken", "lighten" to "Lighten", "difference" to "Difference",
    "hue" to "Hue", "saturation" to "Saturation", "color" to "Color", "luminosity" to "Luminosity",
)

/** Blend modes as `layer.setProps` names them. */
val layerBlendModes = listOf(
    "Normal", "Dissolve", "Darken", "Multiply", "Color Burn", "Linear Burn", "Lighten", "Screen", "Color Dodge",
    "Linear Dodge (Add)", "Overlay", "Soft Light", "Hard Light", "Vivid Light", "Linear Light", "Pin Light",
    "Difference", "Exclusion", "Subtract", "Divide", "Hue", "Saturation", "Color", "Luminosity",
)

private val tonings = listOf("tealOrange" to "Teal & orange", "warmFade" to "Warm fade", "coolBlue" to "Cool blue", "cinematic" to "Cinematic", "sepia" to "Sepia", "rose" to "Rose")
private val skies = listOf("blue" to "Clear blue", "golden" to "Golden", "sunset" to "Sunset", "dusk" to "Dusk", "storm" to "Storm")

/** A look with one Amount slider. */
private fun look(label: String, id: String, default: Float, min: Float = 0f, extra: List<Param> = emptyList()) =
    MenuItem(label, id, Kind.Look, listOf(Slider("amount", "Amount", min, 100f, default)) + extra)

/** A template: several looks in one, with a Strength slider. */
private fun template(label: String, id: String) =
    MenuItem(label, "template", Kind.Look, listOf(Slider("amount", "Strength", 0f, 100f, 80f)), fixed = mapOf("style" to id))

private val selectiveRanges = listOf("reds" to "Reds", "yellows" to "Yellows", "greens" to "Greens", "cyans" to "Cyans", "blues" to "Blues", "magentas" to "Magentas", "whites" to "Whites", "neutrals" to "Neutrals", "blacks" to "Blacks")

private val bands = listOf("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")

/** Eight HSL band sliders for one of Camera Raw's `hslHue` / `hslSat` / `hslLum` arrays. */
private fun hsl(label: String, key: String, range: Float) =
    MenuItem(label, "filter.cameraRaw", Kind.Pixels, bands.mapIndexed { i, b -> Slider("$key[$i]", b, -range, range, 0f) })

private val textures = listOf("canvas" to "Canvas", "brick" to "Brick", "burlap" to "Burlap", "sandstone" to "Sandstone")

/** A Filter Gallery filter: `filter.gallery.<id>`. */
private fun gallery(label: String, id: String, vararg params: Param) = MenuItem(label, "filter.gallery.$id", Kind.Pixels, params.toList())

/** Gradient map presets as `json:` stops (sent as JSON). */
private val gradients = listOf(
    "json:[[0,\"#000000\"],[1,\"#ffffff\"]]" to "Black → white",
    "json:[[0,\"#1b1035\"],[0.5,\"#c2416b\"],[1,\"#ffd27f\"]]" to "Sunset",
    "json:[[0,\"#0b2a3c\"],[0.55,\"#4f8a8b\"],[1,\"#f4e9cd\"]]" to "Teal cream",
    "json:[[0,\"#2b1a0e\"],[0.6,\"#a9744f\"],[1,\"#f3e3c3\"]]" to "Sepia",
    "json:[[0,\"#10002b\"],[0.5,\"#7b2cbf\"],[1,\"#e0aaff\"]]" to "Violet",
    "json:[[0,\"#001219\"],[0.35,\"#0a9396\"],[0.7,\"#ee9b00\"],[1,\"#ffffff\"]]" to "Duotone pop",
)

object Catalog {
    /**
     * An adjustment layer's catalog item turned into an editor for that layer: same controls,
     * starting at the layer's current settings (`doc.inspect` `adjustment.params`), applied with
     * `layer.setAdjustment`. Null when the kind has no item here.
     */
    fun editorFor(layer: Long, adjustment: kotlinx.serialization.json.JsonObject): MenuItem? {
        val kind = (adjustment["kind"] as? kotlinx.serialization.json.JsonPrimitive)?.content ?: return null
        val params = adjustment["params"] as? kotlinx.serialization.json.JsonObject ?: kotlinx.serialization.json.JsonObject(emptyMap())
        val found = (adjustmentLayers.items + light.items).firstOrNull { it.kind == Kind.AdjustmentLayer && it.command == kind } ?: return null
        // Strength is the layer's opacity, not an adjustment setting: Layers › Opacity changes it.
        val item = found.copy(params = found.params.filter { it.key != "opacity" })
        val initial = mutableMapOf<String, Any>()
        item.params.forEach { p -> initialValue(p, params)?.let { initial[p.key] = it } }
        return item.copy(label = "Edit ${item.label}", kind = Kind.AdjustmentEdit, fixed = item.fixed + ("layer" to layer), initial = initial)
    }

    private fun lookup(params: kotlinx.serialization.json.JsonObject, key: String): kotlinx.serialization.json.JsonElement? {
        Regex("""^(\w+)\[(\d+)]$""").matchEntire(key)?.let { m ->
            return (params[m.groupValues[1]] as? kotlinx.serialization.json.JsonArray)?.getOrNull(m.groupValues[2].toInt())
        }
        val dot = key.indexOf('.')
        if (dot > 0) return (params[key.substring(0, dot)] as? kotlinx.serialization.json.JsonObject)?.get(key.substring(dot + 1))
        return params[key]
    }

    private fun initialValue(p: Param, params: kotlinx.serialization.json.JsonObject): Any? {
        if (p is CurveParam) {
            fun curve(ch: String): Curve? = (params[ch] as? kotlinx.serialization.json.JsonArray)?.mapNotNull { pt ->
                val a = pt as? kotlinx.serialization.json.JsonArray ?: return@mapNotNull null
                val i = (a.getOrNull(0) as? kotlinx.serialization.json.JsonPrimitive)?.content?.toDoubleOrNull()?.let { kotlin.math.round(it).toInt().coerceIn(0, 255) } ?: return@mapNotNull null
                val o = (a.getOrNull(1) as? kotlinx.serialization.json.JsonPrimitive)?.content?.toDoubleOrNull()?.let { kotlin.math.round(it).toInt().coerceIn(0, 255) } ?: return@mapNotNull null
                i to o
            }?.sortedBy { it.first }?.distinctBy { it.first }?.takeIf { it.size >= 2 }
            return CurveSet(CurveSet.CHANNELS.associate { (ch, _) -> ch to (curve(ch) ?: LINEAR) })
        }
        val v = lookup(params, p.key) ?: return null
        val prim = v as? kotlinx.serialization.json.JsonPrimitive
        return when (p) {
            is Slider -> prim?.content?.toFloatOrNull()?.let { (it / p.sendScale).coerceIn(p.min, p.max) }
            is Toggle -> prim?.content?.toBooleanStrictOrNull()
            is ColorParam -> prim?.content
            is Choice -> {
                val s = if (prim != null && prim.isString) prim.content else null
                p.options.firstOrNull { (value, _) ->
                    value == s || (value.startsWith("json:") && runCatching { jsonEq(kotlinx.serialization.json.Json.parseToJsonElement(value.removePrefix("json:")), v) }.getOrDefault(false))
                }?.first
            }
            is CurveParam -> null
        }
    }

    /** JSON equal up to number formatting (0 vs 0.0, f32 rounding) and hex case. */
    private fun jsonEq(a: kotlinx.serialization.json.JsonElement, b: kotlinx.serialization.json.JsonElement): Boolean = when {
        a is kotlinx.serialization.json.JsonArray && b is kotlinx.serialization.json.JsonArray -> a.size == b.size && a.indices.all { jsonEq(a[it], b[it]) }
        a is kotlinx.serialization.json.JsonPrimitive && b is kotlinx.serialization.json.JsonPrimitive -> {
            val x = a.content.toDoubleOrNull(); val y = b.content.toDoubleOrNull()
            if (x != null && y != null) kotlin.math.abs(x - y) < 1e-3 else a.content.equals(b.content, ignoreCase = true)
        }
        else -> a == b
    }

    /** Warp the active layer to a shape (`edit.transform.warp`); on a text layer it is Warp Text. */
    val warp = MenuItem(
        "Warp", "edit.transform.warp", Kind.Pixels,
        listOf(
            Choice("style", "Shape", listOf("arc" to "Arc", "arch" to "Arch", "bulge" to "Bulge", "flag" to "Flag", "wave" to "Wave", "fish" to "Fish", "rise" to "Rise", "fisheye" to "Fisheye", "inflate" to "Inflate", "squeeze" to "Squeeze", "twist" to "Twist", "shellUpper" to "Shell"), "arc"),
            Slider("bend", "Bend %", -100f, 100f, 30f),
            Slider("hDistort", "Horizontal %", -100f, 100f, 0f),
            Slider("vDistort", "Vertical %", -100f, 100f, 0f),
        ),
    )

    /** Stretch or squeeze the photo keeping what matters in shape (`edit.contentAwareScale`). */
    val contentAwareScale = MenuItem(
        "Content-aware scale", "edit.contentAwareScale", Kind.Pixels,
        listOf(Slider("scaleX", "Width %", 30f, 150f, 80f), Slider("scaleY", "Height %", 30f, 150f, 100f), Slider("amount", "Content-aware %", 0f, 100f, 100f), Toggle("protectSkinTones", "Protect skin", true)),
        live = false,
    )

    /** Replace a colour: tap the photo to sample it (`points`), then shift its hue, saturation, lightness. */
    val replaceColor = MenuItem(
        "Replace color", "image.adjustments.replaceColor", Kind.Pixels,
        listOf(
            Slider("hue", "Hue", -180f, 180f, 40f),
            Slider("saturation", "Saturation", -100f, 100f, 0f),
            Slider("lightness", "Lightness", -100f, 100f, 0f),
            Slider("fuzziness", "Fuzziness", 0f, 200f, 40f),
        ),
    )

    /** Curves: a non-destructive Curves adjustment layer edited on a draggable curve. */
    val curves = MenuItem("Curves", "curves", Kind.AdjustmentLayer, listOf(CurveParam()))

    /** Lens correction: geometry, chromatic aberration and lens vignetting (scale-free, so it previews on the proxy). */
    val lensCorrection = MenuItem(
        "Lens correction", "filter.lensCorrection", Kind.Pixels,
        listOf(
            Slider("distortion", "Distortion", -100f, 100f, 0f),
            Slider("vertical", "Vertical", -100f, 100f, 0f),
            Slider("horizontal", "Horizontal", -100f, 100f, 0f),
            Slider("angle", "Rotate", -45f, 45f, 0f, integer = false),
            Slider("scale", "Scale %", 50f, 150f, 100f),
            Slider("redCyan", "Red/cyan fringe", -100f, 100f, 0f),
            Slider("blueYellow", "Blue/yellow fringe", -100f, 100f, 0f),
            Slider("vignetteAmount", "Lens vignette", -100f, 100f, 0f),
        ),
        fixed = mapOf("profile" to "none", "edge" to "edgeExtension"),
    )

    /** Template ids as `edit.lookSuggest` names them → items, for the "For this photo" row. */
    val templates = listOf(
        "auto" to "Auto", "airy" to "Bright & airy", "crisp" to "Crisp landscape", "golden" to "Golden", "moody" to "Moody",
        "portrait" to "Portrait glow", "film" to "Film", "tealOrange" to "Teal & orange", "noir" to "Noir",
    ).map { (id, label) -> template(label, id) }

    fun templateFor(id: String): MenuItem? = templates.firstOrNull { it.fixed["style"] == id }

    /** The Enhance tool: easy, photo-aware tools (PhotoCraft commands composed by `edit.look`). */
    val enhance = listOf(
        MenuGroup("Templates", templates),
        MenuGroup(
            "Essentials",
            listOf(
                look("Enhance", "enhance", 60f, extra = listOf(Slider("sky", "Sky", 0f, 100f, 0f))),
                look("Sky", "sky", 60f),
                look("Structure", "structure", 40f, min = -100f),
                look("Details", "details", 40f),
                look("Denoise", "denoise", 40f),
                look("B&W", "bw", 100f),
                look("Vignette", "vignette", 35f, min = -100f),
            ),
        ),
        MenuGroup(
            "Landscape",
            listOf(
                look("Golden hour", "goldenHour", 50f),
                look("Foliage", "foliage", 50f),
                look("Atmosphere", "atmosphere", 35f, min = -100f),
                look("Sky replacement", "skyReplace", 85f, extra = listOf(Choice("style", "Sky", skies, "blue"))),
                MenuItem(
                    "Relight", "filter.render.relight", Kind.Pixels,
                    listOf(Slider("intensity", "Light", 0f, 100f, 40f), Slider("angle", "Direction", -180f, 180f, 45f), Slider("elevation", "Height", 0f, 90f, 40f), Slider("warmth", "Warmth", -100f, 100f, 0f), Slider("ambient", "Ambient", 0f, 100f, 55f)),
                ),
            ),
        ),
        MenuGroup(
            "Creative",
            listOf(
                look("Dramatic", "dramatic", 50f),
                look("Glow", "glow", 40f),
                look("Matte", "matte", 40f),
                look("Toning", "toning", 50f, extra = listOf(Choice("style", "Colors", tonings, "tealOrange"))),
                look("Film grain", "grain", 35f),
            ),
        ),
        MenuGroup(
            "Portrait",
            listOf(
                look("Smooth skin", "skin", 40f),
                look("Face light", "faceLight", 50f),
                look("Eyes", "eyes", 50f),
                look("Red eye", "redEye", 60f),
                look("Background blur", "bokeh", 50f),
                MenuItem(
                    "Face shape", "faceLiquify", Kind.Face,
                    listOf(
                        Slider("eyes", "Eye size", -100f, 100f, 0f),
                        Slider("faceWidth", "Face width", -100f, 100f, 0f),
                        Slider("chin", "Chin", -100f, 100f, 0f),
                        Slider("forehead", "Forehead", -100f, 100f, 0f),
                        Slider("nose", "Nose", -100f, 100f, 0f),
                        Slider("mouth", "Mouth", -100f, 100f, 0f),
                        Slider("smile", "Smile", -100f, 100f, 0f),
                    ),
                ),
            ),
        ),
    )

    /** Light & color: one Camera Raw-style pass (PhotoCraft's `filter.cameraRaw`). */
    val light = MenuGroup(
        "Light & color",
        listOf(
            MenuItem(
                "Light", "filter.cameraRaw", Kind.Pixels,
                listOf(
                    Slider("exposure", "Exposure", -5f, 5f, 0f, integer = false),
                    Slider("contrast", "Contrast", -100f, 100f, 0f),
                    Slider("highlights", "Highlights", -100f, 100f, 0f),
                    Slider("shadows", "Shadows", -100f, 100f, 0f),
                    Slider("whites", "Whites", -100f, 100f, 0f),
                    Slider("blacks", "Blacks", -100f, 100f, 0f),
                ),
            ),
            curves,
            MenuItem(
                "Color", "filter.cameraRaw", Kind.Pixels,
                listOf(
                    Slider("temperature", "Temperature", -100f, 100f, 0f),
                    Slider("tint", "Tint", -100f, 100f, 0f),
                    Slider("vibrance", "Vibrance", -100f, 100f, 0f),
                    Slider("saturation", "Saturation", -100f, 100f, 0f),
                ),
            ),
            MenuItem(
                "Effects", "filter.cameraRaw", Kind.Pixels,
                listOf(
                    Slider("texture", "Texture", -100f, 100f, 0f),
                    Slider("clarity", "Clarity", -100f, 100f, 0f),
                    Slider("dehaze", "Dehaze", -100f, 100f, 0f),
                ),
            ),
            MenuItem(
                "Tone curve", "filter.cameraRaw", Kind.Pixels,
                listOf(
                    Slider("curveHighlights", "Highlights", -100f, 100f, 0f),
                    Slider("curveLights", "Lights", -100f, 100f, 0f),
                    Slider("curveDarks", "Darks", -100f, 100f, 0f),
                    Slider("curveShadows", "Shadows", -100f, 100f, 0f),
                ),
            ),
            hsl("HSL hue", "hslHue", 100f),
            hsl("HSL saturation", "hslSat", 100f),
            hsl("HSL luminance", "hslLum", 100f),
            MenuItem(
                "Color grading", "filter.cameraRaw", Kind.Pixels,
                listOf(
                    Slider("gradeShadows.hue", "Shadows hue", 0f, 360f, 220f),
                    Slider("gradeShadows.sat", "Shadows sat.", 0f, 100f, 0f),
                    Slider("gradeMidtones.hue", "Midtones hue", 0f, 360f, 30f),
                    Slider("gradeMidtones.sat", "Midtones sat.", 0f, 100f, 0f),
                    Slider("gradeHighlights.hue", "Highlights hue", 0f, 360f, 45f),
                    Slider("gradeHighlights.sat", "Highlights sat.", 0f, 100f, 0f),
                    Slider("gradeBlending", "Blending", 0f, 100f, 50f),
                    Slider("gradeBalance", "Balance", -100f, 100f, 0f),
                ),
            ),
            MenuItem(
                "Detail", "filter.cameraRaw", Kind.Pixels,
                listOf(
                    Slider("sharpenAmount", "Sharpen", 0f, 150f, 0f),
                    Slider("sharpenRadius", "Radius", 0.5f, 3f, 1f, integer = false),
                    Slider("sharpenMasking", "Masking", 0f, 100f, 0f),
                    Slider("noiseLuminance", "Noise", 0f, 100f, 0f),
                    Slider("noiseColor", "Color noise", 0f, 100f, 0f),
                ),
            ),
            MenuItem(
                "Vignette & grain", "filter.cameraRaw", Kind.Pixels,
                listOf(
                    Slider("vignetteAmount", "Vignette", -100f, 100f, 0f),
                    Slider("vignetteMidpoint", "Midpoint", 0f, 100f, 50f),
                    Slider("vignetteRoundness", "Roundness", -100f, 100f, 0f),
                    Slider("vignetteFeather", "Feather", 0f, 100f, 50f),
                    Slider("grainAmount", "Grain", 0f, 100f, 0f),
                    Slider("grainSize", "Grain size", 0f, 100f, 25f),
                    Slider("grainRoughness", "Roughness", 0f, 100f, 50f),
                ),
            ),
            MenuItem(
                "Shadows/Highlights", "image.adjustments.shadowsHighlights", Kind.Pixels,
                listOf(
                    Slider("shadowAmount", "Shadows", 0f, 100f, 35f),
                    Slider("highlightAmount", "Highlights", 0f, 100f, 0f),
                    Slider("shadowTone", "Shadow tone", 0f, 100f, 50f),
                    Slider("highlightTone", "Highlight tone", 0f, 100f, 50f),
                    Slider("color", "Color", -100f, 100f, 20f),
                    Slider("midtone", "Midtone", -100f, 100f, 0f),
                ),
            ),
            MenuItem(
                "HDR toning", "image.adjustments.hdrToning", Kind.Pixels,
                listOf(
                    Slider("strength", "Strength", 0.1f, 4f, 0.5f, integer = false),
                    Slider("radius", "Radius", 1f, 500f, 30f),
                    Slider("detail", "Detail", -100f, 300f, 30f),
                    Slider("exposure", "Exposure", -5f, 5f, 0f, integer = false),
                    Slider("shadow", "Shadow", -100f, 100f, 0f),
                    Slider("highlight", "Highlight", -100f, 100f, 0f),
                    Slider("vibrance", "Vibrance", -100f, 100f, 0f),
                    Slider("saturation", "Saturation", -100f, 100f, 20f),
                ),
            ),
        ),
    )

    /** Adjustment layers beyond the Adjust sheet's quick sliders. */
    val adjustmentLayers = MenuGroup(
        "Adjustment layers",
        listOf(
            curves,
            MenuItem(
                "Levels", "levels", Kind.AdjustmentLayer,
                listOf(
                    Slider("inBlack", "Input black", 0f, 253f, 0f),
                    Slider("gamma", "Midtones", 0.1f, 3f, 1f, integer = false),
                    Slider("inWhite", "Input white", 2f, 255f, 255f),
                    Slider("outBlack", "Output black", 0f, 255f, 0f),
                    Slider("outWhite", "Output white", 0f, 255f, 255f),
                ),
            ),
            MenuItem(
                "Hue/Saturation", "hueSaturation", Kind.AdjustmentLayer,
                listOf(
                    Slider("hue", "Hue", -180f, 180f, 0f),
                    Slider("saturation", "Saturation", -100f, 100f, 0f),
                    Slider("lightness", "Lightness", -100f, 100f, 0f),
                    Toggle("colorize", "Colorize", false),
                ),
            ),
            MenuItem("Vibrance", "vibrance", Kind.AdjustmentLayer, listOf(Slider("vibrance", "Vibrance", -100f, 100f, 25f), Slider("saturation", "Saturation", -100f, 100f, 0f))),
            MenuItem(
                "Color balance", "colorBalance", Kind.AdjustmentLayer,
                listOf("shadows" to "Shadows", "midtones" to "Midtones", "highlights" to "Highlights").flatMap { (k, l) ->
                    listOf(Slider("$k[0]", "$l: cyan–red", -100f, 100f, 0f), Slider("$k[1]", "$l: magenta–green", -100f, 100f, 0f), Slider("$k[2]", "$l: yellow–blue", -100f, 100f, 0f))
                } + Toggle("preserveLuminosity", "Preserve luminosity", true),
            ),
            MenuItem(
                "Selective color", "selectiveColor", Kind.AdjustmentLayer,
                // One set of CMYK sliders per range (`reds: [c, m, y, k]` …): switching ranges
                // keeps what the others were set to.
                listOf<Param>(Choice("_colors", "Colors", selectiveRanges, "reds")) +
                    selectiveRanges.flatMap { (r, _) -> listOf("Cyan", "Magenta", "Yellow", "Black").mapIndexed { i, l -> Slider("$r[$i]", l, -100f, 100f, 0f, showWhen = "_colors" to r) } } +
                    Choice("method", "Method", listOf("relative" to "Relative", "absolute" to "Absolute"), "relative"),
            ),
            MenuItem(
                "Channel mixer", "channelMixer", Kind.AdjustmentLayer,
                listOf("red" to "Red", "green" to "Green", "blue" to "Blue").flatMapIndexed { o, (k, l) ->
                    listOf("R", "G", "B").mapIndexed { i, src -> Slider("$k[$i]", "$l ← $src %", -200f, 200f, if (i == o) 100f else 0f) }
                },
                fixed = mapOf("red[3]" to 0, "green[3]" to 0, "blue[3]" to 0),
            ),
            MenuItem(
                "Gradient map", "gradientMap", Kind.AdjustmentLayer,
                listOf(Choice("stops", "Gradient", gradients, gradients.first().first), Toggle("reverse", "Reverse", false), Slider("opacity", "Strength %", 0f, 100f, 100f, sendScale = 0.01f, integer = false)),
            ),
            MenuItem(
                "Photo filter", "photoFilter", Kind.AdjustmentLayer,
                listOf(
                    Choice(
                        "filter", "Filter",
                        listOf("warming85" to "Warming 85", "warming81" to "Warming 81", "cooling80" to "Cooling 80", "cooling82" to "Cooling 82", "sepia" to "Sepia", "underwater" to "Underwater", "deepBlue" to "Deep blue", "deepEmerald" to "Deep emerald"),
                        "warming85",
                    ),
                    Slider("density", "Density", 0f, 100f, 25f),
                    Toggle("preserveLuminosity", "Preserve luminosity", true),
                ),
            ),
            MenuItem(
                "Look", "colorLookup", Kind.AdjustmentLayer,
                listOf(
                    Choice(
                        "lut", "Look",
                        listOf("warm" to "Warm", "cool" to "Cool", "tealOrange" to "Teal & orange", "bleachBypass" to "Bleach bypass", "fadedFilm" to "Faded film", "dayForNight" to "Day for night", "monoContrast" to "Mono contrast", "crossProcess" to "Cross process"),
                        "tealOrange",
                    ),
                    Slider("opacity", "Strength %", 0f, 100f, 100f, sendScale = 0.01f, integer = false),
                ),
            ),
            MenuItem(
                "Black & white", "blackWhite", Kind.AdjustmentLayer,
                listOf(
                    Slider("reds", "Reds", -200f, 300f, 40f),
                    Slider("yellows", "Yellows", -200f, 300f, 60f),
                    Slider("greens", "Greens", -200f, 300f, 40f),
                    Slider("cyans", "Cyans", -200f, 300f, 60f),
                    Slider("blues", "Blues", -200f, 300f, 20f),
                    Slider("magentas", "Magentas", -200f, 300f, 80f),
                ),
            ),
            MenuItem("Posterize", "posterize", Kind.AdjustmentLayer, listOf(Slider("levels", "Levels", 2f, 32f, 4f))),
            MenuItem("Threshold", "threshold", Kind.AdjustmentLayer, listOf(Slider("level", "Level", 1f, 255f, 128f))),
            MenuItem("Invert", "invert", Kind.AdjustmentLayer),
        ),
    )

    val filters = listOf(
        MenuGroup(
            "Blur",
            listOf(
                MenuItem("Gaussian", "filter.blur.gaussianBlur", Kind.Pixels, listOf(Slider("radius", "Radius", 0.1f, 100f, 4f, integer = false))),
                MenuItem("Motion", "filter.blur.motionBlur", Kind.Pixels, listOf(Slider("angle", "Angle", -180f, 180f, 0f), Slider("distance", "Distance", 1f, 300f, 20f))),
                MenuItem(
                    "Radial", "filter.blur.radialBlur", Kind.Pixels,
                    listOf(Slider("amount", "Amount", 1f, 100f, 10f), Choice("method", "Method", listOf("spin" to "Spin", "zoom" to "Zoom"), "zoom")),
                    fixed = mapOf("quality" to "draft"),
                ),
                MenuItem(
                    "Tilt-shift", "filter.blurGallery.tiltShift", Kind.Pixels,
                    listOf(Slider("blur", "Blur", 0f, 100f, 15f), Slider("centerY", "Position", 0f, 1f, 0.5f, integer = false), Slider("focus", "Focus band", 0f, 0.5f, 0.1f, integer = false), Slider("angle", "Angle", -90f, 90f, 0f)),
                ),
                MenuItem(
                    "Iris", "filter.blurGallery.irisBlur", Kind.Pixels,
                    listOf(Slider("blur", "Blur", 0f, 100f, 15f), Slider("radiusX", "Width", 0.05f, 1f, 0.35f, integer = false), Slider("radiusY", "Height", 0.05f, 1f, 0.25f, integer = false), Slider("feather", "Feather", 0f, 0.99f, 0.5f, integer = false)),
                ),
                MenuItem("Surface", "filter.blur.surfaceBlur", Kind.Pixels, listOf(Slider("radius", "Radius", 1f, 100f, 5f), Slider("threshold", "Threshold", 2f, 255f, 15f))),
                MenuItem("Lens blur", "filter.blur.lensBlur", Kind.Pixels, listOf(Slider("radius", "Radius", 0f, 100f, 15f), Choice("shape", "Iris", listOf("hexagon" to "Hexagon", "pentagon" to "Pentagon", "octagon" to "Octagon", "triangle" to "Triangle", "square" to "Square"), "hexagon"), Slider("brightness", "Bokeh brightness", 0f, 100f, 0f))),
                MenuItem("Spin", "filter.blurGallery.spinBlur", Kind.Pixels, listOf(Slider("blurAngle", "Angle", 0f, 360f, 15f), Slider("centerX", "X", 0f, 1f, 0.5f, integer = false), Slider("centerY", "Y", 0f, 1f, 0.5f, integer = false), Slider("radiusX", "Width", 0.01f, 1f, 0.3f, integer = false), Slider("radiusY", "Height", 0.01f, 1f, 0.3f, integer = false))),
                MenuItem("Field", "filter.blurGallery.fieldBlur", Kind.Pixels, listOf(Slider("blur", "Blur", 0f, 200f, 15f))),
                MenuItem("Smart", "filter.blur.smartBlur", Kind.Pixels, listOf(Slider("radius", "Radius", 0.1f, 100f, 3f, integer = false), Slider("threshold", "Threshold", 0.1f, 100f, 25f, integer = false), Choice("mode", "Mode", listOf("normal" to "Normal", "edgeOnly" to "Edges only", "overlayEdge" to "Overlay edges"), "normal")), fixed = mapOf("quality" to "medium")),
                MenuItem("Box", "filter.blur.boxBlur", Kind.Pixels, listOf(Slider("radius", "Radius", 1f, 200f, 8f))),
            ),
        ),
        MenuGroup(
            "Sharpen",
            listOf(
                MenuItem("Unsharp mask", "filter.sharpen.unsharpMask", Kind.Pixels, listOf(Slider("amount", "Amount", 1f, 300f, 80f), Slider("radius", "Radius", 0.1f, 10f, 1.2f, integer = false), Slider("threshold", "Threshold", 0f, 50f, 2f))),
                MenuItem("Smart sharpen", "filter.sharpen.smartSharpen", Kind.Pixels, listOf(Slider("amount", "Amount", 1f, 300f, 100f), Slider("radius", "Radius", 0.1f, 10f, 1f, integer = false), Slider("reduceNoise", "Reduce noise", 0f, 100f, 10f))),
                MenuItem("High pass", "filter.other.highPass", Kind.Pixels, listOf(Slider("radius", "Radius", 0.1f, 50f, 10f, integer = false))),
            ),
        ),
        MenuGroup(
            "Noise",
            listOf(
                MenuItem("Reduce noise", "filter.noise.reduceNoise", Kind.Pixels, listOf(Slider("strength", "Strength", 0f, 10f, 6f), Slider("preserveDetails", "Preserve details", 0f, 100f, 60f), Slider("reduceColorNoise", "Color noise", 0f, 100f, 45f))),
                MenuItem("Add noise", "filter.noise.addNoise", Kind.Pixels, listOf(Slider("amount", "Amount", 0.1f, 50f, 8f, integer = false), Toggle("monochromatic", "Monochrome", true))),
                MenuItem("Dust & scratches", "filter.noise.dustAndScratches", Kind.Pixels, listOf(Slider("radius", "Radius", 1f, 20f, 2f), Slider("threshold", "Threshold", 0f, 100f, 10f))),
                MenuItem("Median", "filter.noise.median", Kind.Pixels, listOf(Slider("radius", "Radius", 1f, 20f, 2f))),
            ),
        ),
        MenuGroup("Lens", listOf(lensCorrection)),
        MenuGroup(
            "Painting",
            listOf(
                gallery("Watercolor", "watercolor", Slider("brushDetail", "Brush detail", 1f, 14f, 9f), Slider("shadowIntensity", "Shadows", 0f, 10f, 1f), Slider("texture", "Texture", 1f, 3f, 1f)),
                gallery("Dry brush", "dryBrush", Slider("brushSize", "Brush size", 0f, 10f, 2f), Slider("brushDetail", "Brush detail", 0f, 10f, 8f), Slider("texture", "Texture", 1f, 3f, 1f)),
                gallery("Palette knife", "paletteKnife", Slider("strokeSize", "Stroke size", 1f, 50f, 25f), Slider("strokeDetail", "Detail", 1f, 3f, 3f), Slider("softness", "Softness", 0f, 10f, 0f)),
                gallery("Fresco", "fresco", Slider("brushSize", "Brush size", 0f, 10f, 2f), Slider("brushDetail", "Brush detail", 0f, 10f, 8f), Slider("texture", "Texture", 1f, 3f, 1f)),
                gallery("Underpainting", "underpainting", Slider("brushSize", "Brush size", 0f, 40f, 6f), Slider("textureCoverage", "Coverage", 0f, 40f, 16f), Choice("texture", "Texture", textures, "canvas")),
                gallery("Rough pastels", "roughPastels", Slider("strokeLength", "Stroke length", 0f, 40f, 6f), Slider("strokeDetail", "Detail", 1f, 20f, 4f), Choice("texture", "Texture", textures, "canvas")),
                gallery("Sponge", "sponge", Slider("brushSize", "Brush size", 0f, 10f, 2f), Slider("definition", "Definition", 0f, 25f, 12f), Slider("smoothness", "Smoothness", 1f, 15f, 5f)),
                gallery("Sumi-e", "sumiE", Slider("strokeWidth", "Stroke width", 3f, 15f, 10f), Slider("strokePressure", "Pressure", 0f, 15f, 2f), Slider("contrast", "Contrast", 0f, 40f, 16f)),
                gallery("Cutout", "cutout", Slider("numberOfLevels", "Levels", 2f, 8f, 4f), Slider("edgeSimplicity", "Simplicity", 0f, 10f, 4f), Slider("edgeFidelity", "Fidelity", 1f, 3f, 2f)),
                gallery("Poster edges", "posterEdges", Slider("edgeThickness", "Edge thickness", 0f, 10f, 2f), Slider("edgeIntensity", "Edge intensity", 0f, 10f, 1f), Slider("posterization", "Posterize", 0f, 6f, 2f)),
                gallery("Plastic wrap", "plasticWrap", Slider("highlightStrength", "Highlights", 0f, 20f, 15f), Slider("detail", "Detail", 1f, 15f, 9f), Slider("smoothness", "Smoothness", 1f, 15f, 7f)),
            ),
        ),
        MenuGroup(
            "Sketch",
            listOf(
                gallery("Colored pencil", "coloredPencil", Slider("pencilWidth", "Pencil width", 1f, 24f, 4f), Slider("strokePressure", "Pressure", 0f, 15f, 8f), Slider("paperBrightness", "Paper", 0f, 50f, 25f)),
                gallery("Ink outlines", "inkOutlines", Slider("strokeLength", "Stroke length", 1f, 50f, 4f), Slider("darkIntensity", "Darks", 0f, 50f, 20f), Slider("lightIntensity", "Lights", 0f, 50f, 10f)),
                gallery("Charcoal", "charcoal", Slider("charcoalThickness", "Thickness", 1f, 7f, 1f), Slider("detail", "Detail", 0f, 5f, 5f), Slider("lightDarkBalance", "Balance", 0f, 100f, 50f)),
                gallery("Graphic pen", "graphicPen", Slider("strokeLength", "Stroke length", 1f, 15f, 15f), Slider("lightDarkBalance", "Balance", 0f, 100f, 50f), Choice("strokeDirection", "Direction", listOf("rightDiagonal" to "Right diagonal", "leftDiagonal" to "Left diagonal", "horizontal" to "Horizontal", "vertical" to "Vertical"), "rightDiagonal")),
                gallery("Photocopy", "photocopy", Slider("detail", "Detail", 1f, 24f, 7f), Slider("darkness", "Darkness", 1f, 50f, 8f)),
                gallery("Halftone", "halftonePattern", Slider("size", "Size", 1f, 12f, 1f), Slider("contrast", "Contrast", 0f, 50f, 5f), Choice("patternType", "Pattern", listOf("dot" to "Dots", "circle" to "Circles", "line" to "Lines"), "dot")),
                gallery("Glowing edges", "glowingEdges", Slider("edgeWidth", "Edge width", 1f, 14f, 2f), Slider("edgeBrightness", "Brightness", 0f, 20f, 6f), Slider("smoothness", "Smoothness", 1f, 15f, 5f)),
                gallery("Neon glow", "neonGlow", Slider("glowSize", "Glow size", -24f, 24f, 5f), Slider("glowBrightness", "Brightness", 0f, 50f, 15f)),
                gallery("Chrome", "chrome", Slider("detail", "Detail", 0f, 10f, 4f), Slider("smoothness", "Smoothness", 0f, 10f, 7f)),
            ),
        ),
        MenuGroup(
            "Texture",
            listOf(
                gallery("Texturizer", "texturizer", Choice("texture", "Texture", textures, "canvas"), Slider("scaling", "Scale %", 50f, 200f, 100f), Slider("relief", "Relief", 0f, 50f, 4f)),
                gallery("Craquelure", "craquelure", Slider("crackSpacing", "Spacing", 2f, 100f, 15f), Slider("crackDepth", "Depth", 0f, 10f, 6f), Slider("crackBrightness", "Brightness", 0f, 10f, 9f)),
                gallery("Stained glass", "stainedGlass", Slider("cellSize", "Cell size", 2f, 50f, 10f), Slider("borderThickness", "Border", 1f, 20f, 4f), Slider("lightIntensity", "Light", 0f, 10f, 3f)),
                gallery("Mosaic tiles", "mosaicTiles", Slider("tileSize", "Tile size", 2f, 100f, 12f), Slider("groutWidth", "Grout", 1f, 15f, 3f), Slider("lightenGrout", "Lighten grout", 0f, 10f, 9f)),
                gallery("Glass", "glass", Slider("distortion", "Distortion", 0f, 20f, 5f), Slider("smoothness", "Smoothness", 1f, 15f, 3f), Choice("texture", "Texture", listOf("frosted" to "Frosted", "blocks" to "Blocks", "canvas" to "Canvas", "tinyLens" to "Tiny lens"), "frosted")),
                MenuItem("Picture frame", "filter.render.pictureFrame", Kind.Pixels, listOf(Choice("frame", "Frame", listOf("vineWithLeaves" to "Vine", "roses" to "Roses", "daisies" to "Daisies", "ivy" to "Ivy", "bamboo" to "Bamboo", "waves" to "Waves", "rope" to "Rope", "scallops" to "Scallops", "doubleLine" to "Double line", "dots" to "Dots", "stars" to "Stars", "hearts" to "Hearts", "snowflakes" to "Snowflakes"), "vineWithLeaves"), Slider("margin", "Margin", 0f, 30f, 4f), Slider("size", "Size", 1f, 100f, 50f)), fixed = mapOf("newLayer" to true)),
            ),
        ),
        MenuGroup(
            "Stylize",
            listOf(
                MenuItem("Oil paint", "filter.stylize.oilPaint", Kind.Pixels, listOf(Slider("stylization", "Stylization", 0.1f, 10f, 4f, integer = false), Slider("cleanliness", "Cleanliness", 0f, 10f, 5f, integer = false), Slider("bristleDetail", "Bristle detail", 0f, 10f, 5f, integer = false))),
                MenuItem("Emboss", "filter.stylize.emboss", Kind.Pixels, listOf(Slider("angle", "Angle", -180f, 180f, 135f), Slider("height", "Height", 1f, 20f, 3f), Slider("amount", "Amount", 1f, 500f, 100f))),
                MenuItem("Find edges", "filter.stylize.findEdges", Kind.Pixels),
                MenuItem("Solarize", "filter.stylize.solarize", Kind.Pixels),
                MenuItem("Diffuse", "filter.stylize.diffuse", Kind.Pixels, listOf(Choice("mode", "Mode", listOf("normal" to "Normal", "darkenOnly" to "Darken only", "lightenOnly" to "Lighten only", "anisotropic" to "Anisotropic"), "normal"))),
                MenuItem("Tiles", "filter.stylize.tiles", Kind.Pixels, listOf(Slider("count", "Tiles across", 1f, 99f, 10f), Slider("maxOffset", "Offset %", 1f, 99f, 10f))),
                MenuItem("Extrude", "filter.stylize.extrude", Kind.Pixels, listOf(Choice("type", "Type", listOf("blocks" to "Blocks", "pyramids" to "Pyramids"), "blocks"), Slider("size", "Size", 2f, 255f, 30f), Slider("depth", "Depth", 1f, 255f, 30f))),
                MenuItem("Wind", "filter.stylize.wind", Kind.Pixels, listOf(Choice("method", "Method", listOf("wind" to "Wind", "blast" to "Blast", "stagger" to "Stagger"), "wind"), Choice("direction", "Direction", listOf("fromRight" to "From right", "fromLeft" to "From left"), "fromRight"))),
            ),
        ),
        MenuGroup(
            "Distort",
            listOf(
                MenuItem("Twirl", "filter.distort.twirl", Kind.Pixels, listOf(Slider("angle", "Angle", -999f, 999f, 50f))),
                MenuItem("Pinch", "filter.distort.pinch", Kind.Pixels, listOf(Slider("amount", "Amount", -100f, 100f, 50f))),
                MenuItem("Spherize", "filter.distort.spherize", Kind.Pixels, listOf(Slider("amount", "Amount", -100f, 100f, 60f))),
                MenuItem("Ripple", "filter.distort.ripple", Kind.Pixels, listOf(Slider("amount", "Amount", -999f, 999f, 100f), Choice("size", "Size", listOf("small" to "Small", "medium" to "Medium", "large" to "Large"), "medium"))),
                MenuItem("Zig zag", "filter.distort.zigZag", Kind.Pixels, listOf(Slider("amount", "Amount", -100f, 100f, 10f), Slider("ridges", "Ridges", 0f, 20f, 5f))),
                MenuItem("Wave", "filter.distort.wave", Kind.Pixels, listOf(Slider("generators", "Generators", 1f, 20f, 5f), Slider("wavelengthMax", "Wavelength", 2f, 999f, 120f), Slider("amplitudeMax", "Amplitude", 1f, 999f, 35f), Choice("type", "Type", listOf("sine" to "Sine", "triangle" to "Triangle", "square" to "Square"), "sine"))),
                MenuItem("Polar", "filter.distort.polarCoordinates", Kind.Pixels, listOf(Choice("mode", "Mode", listOf("rectangularToPolar" to "To polar", "polarToRectangular" to "From polar"), "rectangularToPolar"))),
            ),
        ),
        MenuGroup(
            "Pixelate",
            listOf(
                MenuItem("Mosaic", "filter.pixelate.mosaic", Kind.Pixels, listOf(Slider("cellSize", "Cell size", 2f, 100f, 10f))),
                MenuItem("Crystallize", "filter.pixelate.crystallize", Kind.Pixels, listOf(Slider("cellSize", "Cell size", 3f, 100f, 10f))),
                MenuItem("Pointillize", "filter.pixelate.pointillize", Kind.Pixels, listOf(Slider("cellSize", "Cell size", 3f, 60f, 5f))),
                MenuItem("Halftone", "filter.pixelate.colorHalftone", Kind.Pixels, listOf(Slider("maxRadius", "Radius", 4f, 40f, 8f))),
                MenuItem("Facet", "filter.pixelate.facet", Kind.Pixels),
            ),
        ),
        MenuGroup(
            "Render",
            listOf(
                MenuItem("Lens flare", "filter.render.lensFlare", Kind.Pixels, listOf(Slider("brightness", "Brightness", 10f, 300f, 100f), Slider("centerX", "X", 0f, 1f, 0.3f, integer = false), Slider("centerY", "Y", 0f, 1f, 0.3f, integer = false), Choice("lens", "Lens", listOf("zoom" to "50–300 mm zoom", "prime35" to "35 mm", "prime105" to "105 mm", "moviePrime" to "Movie"), "zoom"))),
                MenuItem("Lighting", "filter.render.lightingEffects", Kind.Pixels, listOf(Choice("lightType", "Light", listOf("spot" to "Spot", "point" to "Point", "infinite" to "Infinite"), "spot"), Slider("intensity", "Intensity", -100f, 100f, 75f), Slider("lightX", "X", 0f, 1f, 0.25f, integer = false), Slider("lightY", "Y", 0f, 1f, 0.2f, integer = false))),
                MenuItem("Clouds", "filter.render.clouds", Kind.Pixels, listOf(Slider("seed", "Variation", 0f, 100f, 0f)), live = false),
            ),
        ),
    )

    /** Layer effects on the active layer (needs a layer that is not the locked Background). */
    val styles = MenuGroup(
        "Effects",
        listOf(
            MenuItem("Drop shadow", "layer.layerStyle.dropShadow", Kind.LayerStyle, listOf(ColorParam("color", "Color", "#000000"), Slider("opacity", "Opacity", 0f, 100f, 75f), Slider("angle", "Angle", -180f, 180f, 120f), Slider("distance", "Distance", 0f, 200f, 10f), Slider("size", "Size", 0f, 100f, 10f), Choice("blend", "Blend", blends, "multiply"))),
            MenuItem("Stroke", "layer.layerStyle.stroke", Kind.LayerStyle, listOf(ColorParam("color", "Color", "#ffffff"), Slider("size", "Size", 1f, 50f, 4f), Choice("position", "Position", listOf("outside" to "Outside", "inside" to "Inside", "center" to "Center"), "outside"), Slider("opacity", "Opacity", 0f, 100f, 100f))),
            MenuItem("Outer glow", "layer.layerStyle.outerGlow", Kind.LayerStyle, listOf(ColorParam("color", "Color", "#ffe08a"), Slider("opacity", "Opacity", 0f, 100f, 75f), Slider("size", "Size", 0f, 100f, 20f), Slider("spread", "Spread", 0f, 100f, 0f))),
            MenuItem("Inner glow", "layer.layerStyle.innerGlow", Kind.LayerStyle, listOf(ColorParam("color", "Color", "#ffe08a"), Slider("opacity", "Opacity", 0f, 100f, 75f), Slider("size", "Size", 0f, 100f, 15f))),
            MenuItem("Inner shadow", "layer.layerStyle.innerShadow", Kind.LayerStyle, listOf(ColorParam("color", "Color", "#000000"), Slider("opacity", "Opacity", 0f, 100f, 75f), Slider("distance", "Distance", 0f, 100f, 5f), Slider("size", "Size", 0f, 100f, 5f))),
            MenuItem("Color overlay", "layer.layerStyle.colorOverlay", Kind.LayerStyle, listOf(ColorParam("color", "Color", "#3d7bf5"), Slider("opacity", "Opacity", 0f, 100f, 60f), Choice("blend", "Blend", blends, "normal"))),
            MenuItem("Bevel", "layer.layerStyle.bevelEmboss", Kind.LayerStyle, listOf(Choice("style", "Style", listOf("inner" to "Inner", "outer" to "Outer", "emboss" to "Emboss", "pillow" to "Pillow"), "inner"), Slider("depth", "Depth", 1f, 500f, 100f), Slider("size", "Size", 0f, 100f, 5f))),
        ),
    )

    /** Crop › Geometry: perspective and lens fixes. */
    val geometry = MenuGroup("Geometry", listOf(lensCorrection))

    val document = MenuGroup(
        "Size",
        listOf(
            MenuItem("Image size", "image.imageSize", Kind.Document, listOf(Slider("percent", "Scale %", 10f, 200f, 100f)), fixed = mapOf("resample" to "automatic"), live = false),
            MenuItem("Canvas", "image.canvasSize", Kind.Document, listOf(Slider("width", "Add width %", 0f, 100f, 10f), Slider("height", "Add height %", 0f, 100f, 10f), Choice("anchor", "Anchor", listOf("center" to "Center", "top" to "Top", "bottom" to "Bottom", "left" to "Left", "right" to "Right"), "center")), fixed = mapOf("units" to "percent", "relative" to true), live = false),
            MenuItem("Trim", "image.trim", Kind.Document, listOf(Choice("basedOn", "Based on", listOf("transparent" to "Transparent", "topLeft" to "Top-left color", "bottomRight" to "Bottom-right color"), "topLeft")), live = false),
        ),
    )
}
