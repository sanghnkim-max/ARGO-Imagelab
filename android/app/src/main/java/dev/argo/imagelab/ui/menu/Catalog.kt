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

object Catalog {
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
        ),
    )

    /** Adjustment layers beyond the Adjust sheet's quick sliders. */
    val adjustmentLayers = MenuGroup(
        "Adjustment layers",
        listOf(
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
        MenuGroup(
            "Stylize",
            listOf(
                MenuItem("Oil paint", "filter.stylize.oilPaint", Kind.Pixels, listOf(Slider("stylization", "Stylization", 0.1f, 10f, 4f, integer = false), Slider("cleanliness", "Cleanliness", 0f, 10f, 5f, integer = false), Slider("bristleDetail", "Bristle detail", 0f, 10f, 5f, integer = false))),
                MenuItem("Emboss", "filter.stylize.emboss", Kind.Pixels, listOf(Slider("angle", "Angle", -180f, 180f, 135f), Slider("height", "Height", 1f, 20f, 3f), Slider("amount", "Amount", 1f, 500f, 100f))),
                MenuItem("Find edges", "filter.stylize.findEdges", Kind.Pixels),
                MenuItem("Solarize", "filter.stylize.solarize", Kind.Pixels),
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

    val document = MenuGroup(
        "Size",
        listOf(
            MenuItem("Image size", "image.imageSize", Kind.Document, listOf(Slider("percent", "Scale %", 10f, 200f, 100f)), fixed = mapOf("resample" to "automatic"), live = false),
            MenuItem("Canvas", "image.canvasSize", Kind.Document, listOf(Slider("width", "Add width %", 0f, 100f, 10f), Slider("height", "Add height %", 0f, 100f, 10f), Choice("anchor", "Anchor", listOf("center" to "Center", "top" to "Top", "bottom" to "Bottom", "left" to "Left", "right" to "Right"), "center")), fixed = mapOf("units" to "percent", "relative" to true), live = false),
            MenuItem("Trim", "image.trim", Kind.Document, listOf(Choice("basedOn", "Based on", listOf("transparent" to "Transparent", "topLeft" to "Top-left color", "bottomRight" to "Bottom-right color"), "topLeft")), live = false),
        ),
    )
}
