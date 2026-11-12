package dev.argo.imagelab.ui

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Switch
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import dev.argo.imagelab.core.EditorState
import androidx.compose.material3.CircularProgressIndicator
import dev.argo.imagelab.core.GoogleFonts
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.clickable
import androidx.compose.foundation.border
import androidx.compose.foundation.background
import dev.argo.imagelab.ui.menu.layerBlendModes
import dev.argo.imagelab.ui.menu.Choice
import dev.argo.imagelab.ui.menu.Toggle
import dev.argo.imagelab.ui.menu.Kind
import dev.argo.imagelab.ui.menu.MenuItem
import dev.argo.imagelab.ui.menu.MenuGroup
import dev.argo.imagelab.ui.menu.CommandSheet
import dev.argo.imagelab.ui.menu.Catalog
import dev.argo.imagelab.ui.menu.Slider as MSlider
import androidx.compose.material3.HorizontalDivider
import kotlin.math.roundToInt

/** The contextual sheet for [tool]: above the tool bar (compact) or in the right pane. */
@Composable
fun ToolSheet(tool: Tool, state: EditorState, vm: EditorViewModel, modifier: Modifier) {
    Surface(modifier, color = MaterialTheme.colorScheme.surface) {
        Column(Modifier.padding(horizontal = 12.dp, vertical = 8.dp).verticalScroll(rememberScrollState())) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(tool.label, style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
                SymButton(Sym.close, "Close ${tool.label}", { vm.tool = null })
            }
            if (state.documents == 0) {
                Text("Open a photo first.", color = Color.Gray)
                return@Column
            }
            when (tool) {
                Tool.Enhance -> EnhanceSheet(vm)
                Tool.Layers -> LayersSheet(state, vm)
                Tool.Select -> SelectSheet(vm)
                Tool.Adjust -> {
                    // Replace color starts from the photo's centre until a tap picks the colour.
                    androidx.compose.runtime.LaunchedEffect(vm.sampling) {
                        if (vm.sampling && vm.command.values["points"] == null) vm.sampleAt(state.width / 2f, state.height / 2f)
                    }
                    AdjustSheet(vm)
                    HorizontalDivider(Modifier.padding(vertical = 8.dp))
                    CommandSheet(listOf(Catalog.light, Catalog.adjustmentLayers, colorTools(state)), vm.command, note = if (vm.sampling) "Tap the photo on the colour to replace." else null)
                }
                Tool.Filters -> CommandSheet(Catalog.filters, vm.command, note = if (vm.selection != null) "Filters apply inside the selection." else "Filters change the active layer's pixels; the photo follows the sliders.")
                Tool.Effects -> CommandSheet(listOf(Catalog.styles), vm.command, note = "Effects style the active layer (not the locked Background: duplicate it or use a text/shape layer).")
                Tool.Retouch -> RetouchSheet(vm)
                Tool.Paint -> PaintSheet(vm)
                Tool.Text -> TextSheet(state, vm)
                Tool.Transform -> TransformSheet(state, vm)
                Tool.History -> HistorySheet(state, vm)
                Tool.Crop -> {
                    CropSheet(vm)
                    HorizontalDivider(Modifier.padding(vertical = 8.dp))
                    CommandSheet(listOf(Catalog.geometry, Catalog.document), vm.command)
                }
            }
        }
    }
}

@Composable
private fun ChipRow(content: @Composable () -> Unit) {
    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) { content() }
}

// ── Enhance ──────────────────────────────────────────────────────────────────────────────

@Composable
private fun EnhanceSheet(vm: EditorViewModel) {
    if (vm.suggested.isNotEmpty()) {
        Text("For this photo" + (vm.scene?.let { " · $it" } ?: ""), style = MaterialTheme.typography.labelLarge, color = Accent)
        ChipRow {
            vm.suggested.mapNotNull { Catalog.templateFor(it) }.forEach { item ->
                FilterChip(selected = vm.command.item == item, onClick = { if (vm.command.item != item) vm.command.open(item) }, label = { Text(item.label) }, leadingIcon = { SymIcon(Sym.enhance, null, Modifier.size(16.dp)) })
            }
        }
        HorizontalDivider(Modifier.padding(vertical = 6.dp))
    }
    CommandSheet(
        Catalog.enhance, vm.command,
        note = if (vm.selection != null) "Whole-photo looks apply inside the selection; sky and portrait looks find their own area." else "One slider each; the photo follows it. Sky and portrait tools find the sky and faces themselves.",
    )
}

// ── Layers ───────────────────────────────────────────────────────────────────────────────────

@Composable
private fun LayersSheet(state: EditorState, vm: EditorViewModel) {
    LayerList(state, vm, Modifier.fillMaxWidth().heightIn(max = 200.dp))
    val active = state.layers.firstOrNull { it.id != null && it.id == state.activeLayer }
    ChipRow {
        AssistChip(onClick = { vm.edit("layer.arrange.bringForward") }, label = { Text("Up") }, leadingIcon = { SymIcon(Sym.up, null, Modifier.size(18.dp)) })
        AssistChip(onClick = { vm.edit("layer.arrange.sendBackward") }, label = { Text("Down") }, leadingIcon = { SymIcon(Sym.down, null, Modifier.size(18.dp)) })
        AssistChip(onClick = { vm.edit("layer.duplicate") }, label = { Text("Duplicate") }, leadingIcon = { SymIcon(Sym.copy, null, Modifier.size(18.dp)) })
        AssistChip(onClick = { vm.edit("layer.mergeDown") }, label = { Text("Merge down") }, leadingIcon = { SymIcon(Sym.merge, null, Modifier.size(18.dp)) })
        AssistChip(onClick = { vm.edit("layer.delete") }, label = { Text("Delete") }, leadingIcon = { SymIcon(Sym.delete, null, Modifier.size(18.dp)) })
        AssistChip(onClick = { vm.edit("layer.groupLayers") }, label = { Text("Group") }, leadingIcon = { SymIcon(Sym.layers, null, Modifier.size(18.dp)) })
        AssistChip(onClick = { vm.edit("layer.createClippingMask") }, label = { Text("Clip to below") })
        AssistChip(onClick = { vm.edit("layer.releaseClippingMask") }, label = { Text("Release clip") })
        AssistChip(onClick = { vm.edit("layer.mergeVisible") }, label = { Text("Merge visible") })
        AssistChip(onClick = { vm.edit("layer.flattenImage") }, label = { Text("Flatten") })
    }
    ChipRow {
        AssistChip(onClick = vm::removeBackground, label = { Text("Remove background") }, leadingIcon = { SymIcon(Sym.person, null, Modifier.size(18.dp)) })
        AssistChip(onClick = { vm.edit("layer.matting.defringe", """{"width":2}""") }, label = { Text("Defringe") })
    }
    ChipRow {
        AssistChip(onClick = { vm.maskFromSelection() }, label = { Text(if (vm.selection != null) "Mask from selection" else "Add mask") })
        AssistChip(onClick = { vm.edit("layer.layerMask.hideAll") }, label = { Text("Black mask") })
        AssistChip(onClick = { vm.edit("layer.layerMask.delete") }, label = { Text("Delete mask") })
        AssistChip(onClick = { vm.edit("layer.layerMask.apply") }, label = { Text("Apply mask") })
        AssistChip(onClick = { vm.edit("layer.layerMask.fromTransparency") }, label = { Text("Mask from transparency") })
    }
    // An adjustment layer reopens with its own controls at its current settings.
    // Kept while it is open, so the controls stay as the layer's settings change underneath.
    val editing = vm.command.item?.kind == Kind.AdjustmentEdit
    val editor = remember(active?.id, editing) {
        val adj = active?.adjustment
        if (editing) vm.command.item else if (active?.id != null && adj != null) Catalog.editorFor(active.id, adj) else null
    }
    if (editor != null) {
        Text("Adjustment layer: tap to change its settings", style = MaterialTheme.typography.labelMedium, color = Accent)
        CommandSheet(listOf(MenuGroup("Adjustment", listOf(editor))), vm.command)
    }
    Text("Align the active layer to the canvas (or the selection)", style = MaterialTheme.typography.labelMedium, color = Color.Gray)
    ChipRow {
        listOf("leftEdges" to "Left", "horizontalCenters" to "Center", "rightEdges" to "Right", "topEdges" to "Top", "verticalCenters" to "Middle", "bottomEdges" to "Bottom").forEach { (id, l) ->
            AssistChip(onClick = { vm.edit("layer.align.$id") }, label = { Text(l) })
        }
    }
    if (active != null) {
        var name by remember(active.id) { mutableStateOf(active.name) }
        Row(verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(name, { name = it }, singleLine = true, label = { Text("Name") }, modifier = Modifier.weight(1f))
            TextButton(onClick = { vm.renameLayer(name) }, enabled = name.isNotBlank() && name != active.name) { Text("Rename") }
        }
        // Opacity, fill and blend of the active layer, live through the same engine as filters.
        val id = active.id ?: 0L
        CommandSheet(
            listOf(
                MenuGroup(
                    "Properties",
                    listOf(
                        MenuItem("Opacity", "layer.setProps", Kind.Pixels, listOf(MSlider("opacity", "Opacity %", 0f, 100f, 100f, sendScale = 0.01f, integer = false)), fixed = mapOf("layer" to id)),
                        MenuItem("Fill", "layer.setProps", Kind.Pixels, listOf(MSlider("fill", "Fill %", 0f, 100f, 100f, sendScale = 0.01f, integer = false)), fixed = mapOf("layer" to id)),
                        MenuItem("Blend", "layer.setProps", Kind.Pixels, listOf(Choice("blend", "Blend mode", layerBlendModes.map { it to it }, "Normal")), fixed = mapOf("layer" to id)),
                    ),
                ),
            ),
            vm.command,
        )
    }
}

// ── Paint ────────────────────────────────────────────────────────────────────────────────────

private val paintSwatches = listOf("#ffffff", "#000000", "#e53935", "#fb8c00", "#fdd835", "#43a047", "#3d7bf5", "#8e24aa")

@Composable
private fun PaintSheet(vm: EditorViewModel) {
    ChipRow {
        PaintMode.entries.forEach { m -> FilterChip(selected = vm.paintMode == m, onClick = { vm.paintMode = m }, label = { Text(m.label) }) }
    }
    Text(
        when (vm.paintMode) {
            PaintMode.Brush -> "Draw with one finger (two fingers zoom). Pressure works with a stylus."
            PaintMode.Mixer -> "Paint that blends with the colours already there, like wet paint."
            PaintMode.Gradient -> "Drag across the photo: the gradient runs from where you start to where you lift."
            PaintMode.Bucket -> "Tap an area to fill it (similar colours that touch)."
        },
        style = MaterialTheme.typography.bodySmall, color = Color.Gray,
    )
    // Where paint goes: pixels, or the active layer's mask (white shows the layer, black hides it).
    ChipRow {
        FilterChip(selected = !vm.paintOnMask, onClick = { vm.paintOnMask = false }, label = { Text("On pixels") })
        FilterChip(selected = vm.paintOnMask, onClick = { vm.paintOnMask = true }, label = { Text("On layer mask") })
        if (!vm.paintOnMask && vm.paintMode != PaintMode.Bucket) FilterChip(selected = vm.paintOnNewLayer, onClick = { vm.paintOnNewLayer = !vm.paintOnNewLayer }, label = { Text("On a new layer") })
        if (!vm.paintOnMask) AssistChip(onClick = vm::newPaintLayer, label = { Text("Next layer") })
    }
    if (vm.paintOnMask) {
        Text("Paints the active layer's mask (one is added if it has none) — e.g. an adjustment layer, to limit where it applies.", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
        ChipRow {
            FilterChip(selected = !vm.maskReveal, onClick = { vm.maskReveal = false }, label = { Text("Hide (black)") })
            FilterChip(selected = vm.maskReveal, onClick = { vm.maskReveal = true }, label = { Text("Show (white)") })
        }
    } else {
        if (vm.paintMode == PaintMode.Brush) ChipRow {
            FilterChip(selected = !vm.eraser, onClick = { vm.eraser = false }, label = { Text("Brush") }, leadingIcon = { SymIcon(Sym.brush, null, Modifier.size(18.dp)) })
            FilterChip(selected = vm.eraser, onClick = { vm.eraser = true }, label = { Text("Eraser") }, leadingIcon = { SymIcon(Sym.erase, null, Modifier.size(18.dp)) })
        }
        if (!vm.eraser || vm.paintMode != PaintMode.Brush) Swatches(paintSwatches, vm.paintColor) { vm.paintColor = it }
    }
    when (vm.paintMode) {
        PaintMode.Gradient -> {
            ChipRow {
                listOf("linear" to "Linear", "radial" to "Radial", "reflected" to "Reflected", "angle" to "Angle", "diamond" to "Diamond").forEach { (v, l) ->
                    FilterChip(selected = vm.gradientStyle == v, onClick = { vm.gradientStyle = v }, label = { Text(l) })
                }
            }
            if (!vm.paintOnMask) {
                Text("Ends in", style = MaterialTheme.typography.labelMedium)
                ChipRow {
                    FilterChip(selected = vm.gradientTo == "transparent", onClick = { vm.gradientTo = "transparent" }, label = { Text("Transparent") })
                    paintSwatches.forEach { hex -> FilterChip(selected = vm.gradientTo == hex, onClick = { vm.gradientTo = hex }, label = { Box(Modifier.size(16.dp).background(Color(android.graphics.Color.parseColor(hex)), CircleShape)) }) }
                }
            }
            LabeledSlider("Opacity", vm.paintOpacity, 0.05f..1f, { vm.paintOpacity = it }) { "${(it * 100).roundToInt()}%" }
        }
        PaintMode.Bucket -> {
            LabeledSlider("Tolerance", vm.bucketTolerance, 0f..255f, { vm.bucketTolerance = it }) { it.roundToInt().toString() }
            LabeledSlider("Opacity", vm.paintOpacity, 0.05f..1f, { vm.paintOpacity = it }) { "${(it * 100).roundToInt()}%" }
        }
        PaintMode.Mixer -> {
            LabeledSlider("Size", vm.paintSize, 1f..300f, { vm.paintSize = it }) { "${it.roundToInt()} px" }
            LabeledSlider("Wet", vm.mixerWet, 0f..100f, { vm.mixerWet = it }) { "${it.roundToInt()}%" }
            LabeledSlider("Mix", vm.mixerMix, 0f..100f, { vm.mixerMix = it }) { "${it.roundToInt()}%" }
        }
        PaintMode.Brush -> {
            LabeledSlider("Size", vm.paintSize, 1f..300f, { vm.paintSize = it }) { "${it.roundToInt()} px" }
            LabeledSlider("Opacity", vm.paintOpacity, 0.05f..1f, { vm.paintOpacity = it }) { "${(it * 100).roundToInt()}%" }
            LabeledSlider("Hardness", vm.paintHardness, 0f..1f, { vm.paintHardness = it }) { "${(it * 100).roundToInt()}%" }
        }
    }
    if (vm.paintMode == PaintMode.Brush || vm.paintMode == PaintMode.Mixer) {
        Text("Symmetry", style = MaterialTheme.typography.labelMedium)
        ChipRow {
            listOf<Pair<String?, String>>(null to "Off", "vertical" to "Mirror ↔", "horizontal" to "Mirror ↕", "dual" to "Four ways", "diagonal" to "Diagonal").forEach { (m, l) ->
                FilterChip(selected = vm.symmetry == m, onClick = { vm.setSymmetryMode(m) }, label = { Text(l) })
            }
        }
    }
}

// ── Text ─────────────────────────────────────────────────────────────────────────────────────

@Composable
private fun TextSheet(state: EditorState, vm: EditorViewModel) {
    val activeIsText = state.layers.firstOrNull { it.id == state.activeLayer }?.kind?.let { it.contains("type", true) || it.contains("text", true) } == true
    Text(if (vm.textAt == null) "Tap the photo to place text, or add it in the centre." else "Placing at (${vm.textAt?.x?.roundToInt()}, ${vm.textAt?.y?.roundToInt()}).", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
    OutlinedTextField(vm.textContent, { vm.textContent = it }, label = { Text("Text") }, modifier = Modifier.fillMaxWidth(), maxLines = 3)
    LabeledSlider("Size", vm.textSize, 8f..400f, { vm.textSize = it }) { "${it.roundToInt()} pt" }
    Swatches(paintSwatches, vm.textColor) { vm.textColor = it }
    FontPicker(vm)
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Button(onClick = vm::placeText) { Text("Add text") }
        if (activeIsText) OutlinedButton(onClick = { vm.updateText() }) { Text("Update selected text") }
    }
}

/** Fonts: the default, installed families, a starter list of Google Fonts and a search. Picking
 *  a Google family that is not installed downloads it (once) and installs it in the workspace. */
@Composable
private fun FontPicker(vm: EditorViewModel) {
    var query by rememberSaveable { mutableStateOf("") }
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text("Font: ${vm.textFont ?: "Default"}", style = MaterialTheme.typography.labelLarge, modifier = Modifier.weight(1f))
        vm.fontDownloading?.let {
            CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp)
            Text("  Getting $it…", style = MaterialTheme.typography.bodySmall)
        }
    }
    val families = (listOf<String?>(null) + vm.installedFonts + GoogleFonts.popular + vm.fontResults.map { it.family }).distinct()
    ChipRow {
        families.forEach { f ->
            FilterChip(
                selected = vm.textFont == f,
                onClick = { vm.pickFont(f) },
                label = { Text(f ?: "Default") },
                leadingIcon = if (f != null && f !in vm.installedFonts) ({ SymIcon(Sym.download, null, Modifier.size(16.dp)) }) else null,
            )
        }
    }
    OutlinedTextField(
        query, { query = it },
        placeholder = { Text("Search Google Fonts") },
        singleLine = true,
        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
        keyboardActions = KeyboardActions(onSearch = { vm.searchFonts(query) }),
        modifier = Modifier.fillMaxWidth(),
    )
}

@Composable
private fun Swatches(colors: List<String>, selected: String, onPick: (String) -> Unit) {
    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(vertical = 4.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        colors.forEach { hex ->
            val on = hex.equals(selected, ignoreCase = true)
            Box(
                Modifier.size(30.dp).background(Color(android.graphics.Color.parseColor(hex)), CircleShape)
                    .border(if (on) 3.dp else 1.dp, if (on) Accent else Color.Gray, CircleShape)
                    .clickable { onPick(hex) },
            )
        }
    }
}

@Composable
private fun LabeledSlider(label: String, value: Float, range: ClosedFloatingPointRange<Float>, onChange: (Float) -> Unit, format: (Float) -> String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(label, Modifier.width(72.dp), style = MaterialTheme.typography.bodySmall)
        Slider(value = value, onValueChange = onChange, valueRange = range, modifier = Modifier.weight(1f))
        Text(format(value), Modifier.width(60.dp), fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall)
    }
}

// ── Select ───────────────────────────────────────────────────────────────────────────────────

@Composable
private fun SelectSheet(vm: EditorViewModel) {
    var concept by rememberSaveable { mutableStateOf("") }
    val hint = when (vm.selectMode) {
        SelectMode.Object -> when (vm.combine) {
            Combine.New -> "Tap an object (SAM). Drag for a box."
            Combine.Add -> "Tap to add more of the object (include point)."
            Combine.Subtract -> "Tap a part to leave out (exclude point)."
        }
        SelectMode.Box -> "Drag a box around the object (SAM)."
        SelectMode.Rect -> "Drag a rectangle."
        SelectMode.Ellipse -> "Drag an ellipse."
        SelectMode.Lasso -> "Draw around the area with your finger."
        SelectMode.Wand -> "Tap a color: the touching area of similar color is selected."
        SelectMode.Color -> "Tap a color: every area of that color is selected."
    }
    Text(hint, style = MaterialTheme.typography.bodySmall, color = Color.Gray)
    SegModelRow(vm)
    ChipRow {
        SelectMode.entries.forEach { m -> FilterChip(selected = vm.selectMode == m, onClick = { vm.selectMode = m }, label = { Text(m.label) }) }
    }
    if (vm.selectMode == SelectMode.Wand || vm.selectMode == SelectMode.Color) {
        LabeledSlider("Tolerance", vm.tolerance, 1f..150f, { vm.tolerance = it }) { it.roundToInt().toString() }
    }
    if (vm.selectMode == SelectMode.Object || vm.selectMode == SelectMode.Box) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Fine detail", style = MaterialTheme.typography.bodyMedium)
                Text("Looks again at full resolution around the object: sharper edges on big photos.", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
            }
            Switch(checked = vm.detail, onCheckedChange = { vm.detail = it })
        }
    }
    SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth().padding(vertical = 2.dp)) {
        Combine.entries.forEachIndexed { i, c ->
            SegmentedButton(
                selected = vm.combine == c,
                onClick = { vm.combine = c },
                shape = SegmentedButtonDefaults.itemShape(i, Combine.entries.size),
                label = { Text(when (c) { Combine.New -> "New"; Combine.Add -> "＋ Add"; Combine.Subtract -> "− Subtract" }) },
            )
        }
    }
    ChipRow {
        AssistChip(onClick = vm::selectSubject, label = { Text("Subject") }, leadingIcon = { SymIcon(Sym.person, null, Modifier.size(18.dp)) })
        AssistChip(onClick = vm::selectSky, label = { Text("Sky") }, leadingIcon = { SymIcon(Sym.landscape, null, Modifier.size(18.dp)) })
        AssistChip(onClick = vm::selectFocus, label = { Text("In focus") })
        AssistChip(onClick = vm::selectAll, label = { Text("All") })
        AssistChip(onClick = vm::removeBackground, label = { Text("Remove background") })
    }
    OutlinedTextField(
        concept, { concept = it },
        placeholder = { Text("e.g. the red car") },
        singleLine = true,
        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
        keyboardActions = KeyboardActions(onSearch = { vm.selectConcept(concept) }),
        trailingIcon = { SymButton(Sym.select, "Select by description", { vm.selectConcept(concept) }, enabled = concept.isNotBlank()) },
        modifier = Modifier.fillMaxWidth(),
    )
    val sel = vm.selection
    if (sel != null) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 6.dp)) {
            Text("Selected: ${sel.label}" + if (sel.mask == null) " (document selection)" else "", style = MaterialTheme.typography.labelLarge, modifier = Modifier.weight(1f))
            TextButton(onClick = vm::clearSelection) { Text("Clear") }
        }
        ChipRow {
            AssistChip(onClick = vm::adjustInside, label = { Text("Adjust inside") }, leadingIcon = { SymIcon(Sym.adjust, null, Modifier.size(18.dp)) })
            AssistChip(onClick = { vm.blurBackground() }, label = { Text("Blur background") }, leadingIcon = { SymIcon(Sym.blur, null, Modifier.size(18.dp)) })
            AssistChip(onClick = vm::removeSelected, label = { Text("Remove") }, leadingIcon = { SymIcon(Sym.erase, null, Modifier.size(18.dp)) })
            AssistChip(onClick = vm::cutToLayer, label = { Text("Cut to layer") }, leadingIcon = { SymIcon(Sym.cut, null, Modifier.size(18.dp)) })
        }
        // Refine the selection: invert, or grow / shrink / soften its edge by [edge] pixels.
        var edge by rememberSaveable { mutableFloatStateOf(8f) }
        ChipRow {
            AssistChip(onClick = vm::invertSelection, label = { Text("Invert") })
            AssistChip(onClick = { vm.modifySelection("feather", edge) }, label = { Text("Feather") })
            AssistChip(onClick = { vm.modifySelection("expand", edge) }, label = { Text("Expand") })
            AssistChip(onClick = { vm.modifySelection("contract", edge) }, label = { Text("Contract") })
            AssistChip(onClick = { vm.modifySelection("smooth", edge) }, label = { Text("Smooth") })
            AssistChip(onClick = { vm.modifySelection("border", edge) }, label = { Text("Border") })
            AssistChip(onClick = { vm.modifySelection("similar", vm.tolerance) }, label = { Text("Similar colors") })
        }
        LabeledSlider("Edge", edge, 1f..100f, { edge = it }) { "${it.roundToInt()} px" }
    }
}

/** MobileSAM status: offer the one-time download, show progress, or say it is ready. */
@Composable
private fun SegModelRow(vm: EditorViewModel) {
    when (vm.segModel) {
        "missing" -> Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(vertical = 4.dp)) {
            Text("Precise selection uses MobileSAM (40.6 MB, Apache-2.0).", style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
            OutlinedButton(onClick = { vm.downloadSegmentModel() }) { Text("Download") }
        }
        "downloading" -> Column(Modifier.padding(vertical = 4.dp)) {
            Text("Downloading MobileSAM… ${(vm.segProgress * 100).roundToInt()}%", style = MaterialTheme.typography.bodySmall)
            LinearProgressIndicator(progress = { vm.segProgress }, modifier = Modifier.fillMaxWidth())
        }
        "ready" -> Text(
            if (vm.segPreparing) "MobileSAM: preparing this photo…" else "MobileSAM ready: taps and boxes segment precisely.",
            style = MaterialTheme.typography.bodySmall, color = Accent,
        )
        else -> {}
    }
}

// ── Adjust ───────────────────────────────────────────────────────────────────────────────────

/** A slider-driven adjustment layer: `params(value)` builds its JSON. */
private val quick = listOf(
    "Auto Tone" to "image.autoTone",
    "Auto Contrast" to "image.autoContrast",
    "Auto Color" to "image.autoColor",
    "B&W" to "layer.newAdjustmentLayer.blackWhite",
    "Invert" to "layer.newAdjustmentLayer.invert",
    "Equalize" to "image.adjustments.equalize",
    "Desaturate" to "image.adjustments.desaturate",
)

@Composable
private fun AdjustSheet(vm: EditorViewModel) {
    var selectedIndex by rememberSaveable { mutableStateOf(0) }
    val selected = adjustSpecs[selectedIndex]
    var value by remember(selectedIndex) { mutableFloatStateOf(0f) }
    val masked = vm.selection != null
    if (masked) Text("Applies inside the selection (${vm.selection?.label}).", style = MaterialTheme.typography.bodySmall, color = Accent)
    ChipRow {
        quick.forEach { (label, id) -> AssistChip(onClick = { vm.edit(id, masked = id.startsWith("layer.")) }, label = { Text(label) }) }
    }
    ChipRow {
        adjustSpecs.forEachIndexed { i, a -> FilterChip(selected = i == selectedIndex, onClick = { selectedIndex = i }, label = { Text(a.label) }) }
    }
    Row(verticalAlignment = Alignment.CenterVertically) {
        // The photo follows the slider as it moves (LiveAdjust): GPU preview at once, the engine right behind.
        Slider(
            value = value,
            onValueChange = { value = it; vm.live.drag(selected, it) },
            onValueChangeFinished = { vm.live.release() },
            valueRange = selected.range,
            modifier = Modifier.weight(1f),
        )
        Text(if (selected.range.endInclusive <= 10f) "%.2f".format(value) else value.roundToInt().toString(), Modifier.width(52.dp), fontFamily = FontFamily.Monospace)
    }
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedButton(onClick = { vm.live.newLayer(); value = 0f }) { Text("New ${selected.label} layer") }
        TextButton(onClick = { value = 0f; vm.live.drag(selected, 0f); vm.live.release() }) { Text("Reset") }
    }
}

// ── Retouch ──────────────────────────────────────────────────────────────────────────────────

@Composable
private fun RetouchSheet(vm: EditorViewModel) {
    val b = vm.retouchBrush
    Text(
        when (b?.use) {
            null -> "Pick a brush, then paint on the photo (two fingers zoom). Or use the selection tools below."
            BrushUse.Source -> if (vm.cloneSource == null || vm.settingSource) "Tap where to copy from (the source), then paint where it should go." else "Paint where the copy should go; the source follows your strokes."
            BrushUse.Drag -> if (vm.selection == null) "Select the area first (Select tool), then come back and drag." else "Drag from the selection to " + (if (b == RetouchBrush.Patch) "a clean area to patch it from." else "where it should move; the gap is filled.")
            BrushUse.Stroke -> "Paint over what to fix with one finger; two fingers zoom and pan."
        },
        style = MaterialTheme.typography.bodySmall, color = Color.Gray,
    )
    fun pick(x: RetouchBrush) { vm.retouchBrush = if (vm.retouchBrush == x) null else x; vm.settingSource = false }
    listOf("Repair", "Tone").forEach { g ->
        Text(g, style = MaterialTheme.typography.labelLarge, modifier = Modifier.padding(top = 4.dp))
        ChipRow {
            RetouchBrush.entries.filter { it.liquify == null && it.group == g }.forEach { x ->
                FilterChip(selected = b == x, onClick = { pick(x) }, label = { Text(x.label) })
            }
        }
    }
    Text("Liquify", style = MaterialTheme.typography.labelLarge, modifier = Modifier.padding(top = 4.dp))
    ChipRow {
        RetouchBrush.entries.filter { it.liquify != null }.forEach { x ->
            FilterChip(selected = b == x, onClick = { pick(x) }, label = { Text(x.label) })
        }
        AssistChip(onClick = { vm.tool = Tool.Enhance; vm.command.open(Catalog.enhance.last().items.last()) }, label = { Text("Face shape") }, leadingIcon = { SymIcon(Sym.face, null, Modifier.size(18.dp)) })
    }
    if (b?.use == BrushUse.Source) {
        ChipRow {
            FilterChip(selected = vm.settingSource, onClick = { vm.settingSource = !vm.settingSource }, label = { Text(if (vm.cloneSource == null) "Tap to set source" else "Set a new source") })
        }
    }
    if (b != null && b.use != BrushUse.Drag) {
        LabeledSlider("Size", vm.retouchSize, 5f..600f, { vm.retouchSize = it }) { "${it.roundToInt()} px" }
        if (b != RetouchBrush.Heal && b != RetouchBrush.Remove) {
            LabeledSlider(if (b.use == BrushUse.Source) "Opacity" else "Strength", vm.retouchStrength, 0.05f..1f, { vm.retouchStrength = it }) { "${(it * 100).roundToInt()}%" }
        }
    }
    HorizontalDivider(Modifier.padding(vertical = 6.dp))
    Text("Content-aware tools use the selection; select what to fix first.", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
    ChipRow {
        AssistChip(onClick = { vm.edit("edit.contentAwareFill", masked = true) }, label = { Text("Content-aware fill") }, leadingIcon = { SymIcon(Sym.fill, null, Modifier.size(18.dp)) })
        AssistChip(onClick = vm::removeSelected, label = { Text("Remove") }, leadingIcon = { SymIcon(Sym.erase, null, Modifier.size(18.dp)) })
        AssistChip(onClick = { vm.denoise() }, label = { Text("Denoise") })
        AssistChip(onClick = { vm.edit("filter.noise.dustAndScratches", """{"radius":2,"threshold":10}""", masked = true) }, label = { Text("Dust & scratches") })
    }
    if (vm.stylusPressure > 0f) {
        Text("Stylus pressure ${(vm.stylusPressure * 100).roundToInt()}% shapes each stroke.", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
    }
}

// ── Transform ────────────────────────────────────────────────────────────────────────────────

@Composable
private fun TransformSheet(state: EditorState, vm: EditorViewModel) {
    val active = state.layers.firstOrNull { it.id != null && it.id == state.activeLayer }
    val isBackground = active?.name == "Background" && active.kind.equals("Pixel", true)
    if (isBackground) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("The Background is locked in place. Make it a layer to move, scale or warp it.", style = MaterialTheme.typography.bodySmall, color = Color.Gray, modifier = Modifier.weight(1f))
            TextButton(onClick = vm::makeMovable) { Text("Make movable") }
        }
    }
    ChipRow {
        AssistChip(onClick = { vm.edit("edit.transform.flipHorizontal") }, label = { Text("Flip ↔") })
        AssistChip(onClick = { vm.edit("edit.transform.flipVertical") }, label = { Text("Flip ↕") })
        AssistChip(onClick = { vm.edit("edit.transform.rotate90Ccw") }, label = { Text("Rotate left") })
        AssistChip(onClick = { vm.edit("edit.transform.rotate90Cw") }, label = { Text("Rotate right") })
    }
    // Free transform of the active layer, live; ranges follow the document's size.
    val w = state.width.coerceAtLeast(1).toFloat()
    val h = state.height.coerceAtLeast(1).toFloat()
    val free = remember(state.width, state.height) {
        MenuItem(
            "Move & scale", "edit.transform", Kind.Pixels,
            listOf(
                MSlider("dx", "Move ↔ px", -w, w, 0f),
                MSlider("dy", "Move ↕ px", -h, h, 0f),
                MSlider("scaleX", "Width %", 10f, 300f, 100f),
                MSlider("scaleY", "Height %", 10f, 300f, 100f),
                MSlider("rotate", "Rotate °", -180f, 180f, 0f, integer = false),
            ),
        )
    }
    CommandSheet(listOf(MenuGroup("Shape", listOf(free, Catalog.warp, Catalog.contentAwareScale))), vm.command, note = "Move & scale and Warp change the active layer; Content-aware scale keeps people and key shapes intact.")
    HorizontalDivider(Modifier.padding(vertical = 6.dp))
    Text("Perspective & puppet", style = MaterialTheme.typography.labelLarge)
    when (vm.handleMode) {
        HandleMode.None -> ChipRow {
            AssistChip(onClick = vm::startQuad, label = { Text("Perspective") })
            AssistChip(onClick = vm::startPins, label = { Text("Puppet warp") })
        }
        HandleMode.Quad -> {
            Text("Drag the four corners onto something that should be rectangular (a building, a page, a screen).", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = vm::applyPerspectiveCrop) { Text("Crop to it") }
                OutlinedButton(onClick = vm::applyPerspectiveStraighten) { Text("Straighten") }
                TextButton(onClick = vm::cancelHandles) { Text("Cancel") }
            }
        }
        HandleMode.Pins -> {
            Text("Tap to place pins on the layer (${vm.handles.size}), then drag pins to bend it; pins you leave hold their part still.", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = vm::applyPuppet, enabled = vm.handles.size >= 2) { Text("Apply") }
                TextButton(onClick = { vm.startPins() }) { Text("Clear pins") }
                TextButton(onClick = vm::cancelHandles) { Text("Cancel") }
            }
        }
    }
    HorizontalDivider(Modifier.padding(vertical = 6.dp))
    Text("Align to the canvas", style = MaterialTheme.typography.labelLarge)
    ChipRow {
        listOf("leftEdges" to "Left", "horizontalCenters" to "Center", "rightEdges" to "Right", "topEdges" to "Top", "verticalCenters" to "Middle", "bottomEdges" to "Bottom").forEach { (id, l) ->
            AssistChip(onClick = { vm.edit("layer.align.$id") }, label = { Text(l) })
        }
    }
}

// ── History ──────────────────────────────────────────────────────────────────────────────────

@Composable
private fun HistorySheet(state: EditorState, vm: EditorViewModel) {
    androidx.compose.runtime.LaunchedEffect(state.version) { vm.refreshHistory() }
    val h = vm.history
    val states = (h?.get("states") as? kotlinx.serialization.json.JsonArray).orEmpty().mapNotNull { it as? kotlinx.serialization.json.JsonObject }
    val current = (h?.get("current") as? kotlinx.serialization.json.JsonPrimitive)?.content?.toIntOrNull() ?: -1
    Text("Tap a step to go back to it; later steps stay until you make a new change.", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
    Column(Modifier.fillMaxWidth().heightIn(max = 220.dp).verticalScroll(rememberScrollState())) {
        states.asReversed().forEach { st ->
            val i = (st["state"] as? kotlinx.serialization.json.JsonPrimitive)?.content?.toIntOrNull() ?: return@forEach
            val label = (st["label"] as? kotlinx.serialization.json.JsonPrimitive)?.content ?: "Step"
            val redo = (st["redo"] as? kotlinx.serialization.json.JsonPrimitive)?.content == "true"
            Row(
                Modifier.fillMaxWidth().background(if (i == current) Accent.copy(alpha = 0.25f) else Color.Transparent).clickable(enabled = i != current) { vm.historyGo(i) }.padding(horizontal = 8.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                SymIcon(if (i == current) Sym.check else Sym.history, null, Modifier.size(16.dp), tint = if (redo) Color.Gray else MaterialTheme.colorScheme.onSurface)
                Text("  $label", color = if (redo) Color.Gray else MaterialTheme.colorScheme.onSurface, style = MaterialTheme.typography.bodyMedium)
            }
        }
    }
    HorizontalDivider(Modifier.padding(vertical = 6.dp))
    Text("Snapshots", style = MaterialTheme.typography.labelLarge)
    var name by rememberSaveable { mutableStateOf("") }
    Row(verticalAlignment = Alignment.CenterVertically) {
        OutlinedTextField(name, { name = it }, singleLine = true, placeholder = { Text("Name (optional)") }, modifier = Modifier.weight(1f))
        TextButton(onClick = { vm.snapshot(name); name = "" }) { Text("Keep snapshot") }
    }
    val snaps = (h?.get("snapshots") as? kotlinx.serialization.json.JsonArray).orEmpty().mapNotNull { it as? kotlinx.serialization.json.JsonObject }
    if (snaps.isEmpty()) Text("A snapshot keeps the photo as it is now, to come back to later in this session.", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
    snaps.forEach { sn ->
        val id = sn["id"] ?: return@forEach
        val label = (sn["name"] as? kotlinx.serialization.json.JsonPrimitive)?.content ?: "Snapshot"
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(label, Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
            TextButton(onClick = { vm.restoreSnapshot(id) }) { Text("Restore") }
            SymButton(Sym.delete, "Delete $label", { vm.deleteSnapshot(id) })
        }
    }
}

/** Colour tools (Replace color, Match color); Match color needs another pixel layer to match. */
private fun colorTools(state: EditorState): MenuGroup {
    val others = state.layers.filter { it.id != null && it.id != state.activeLayer && it.kind.equals("Pixel", true) }
    val match = if (others.isEmpty()) null else MenuItem(
        "Match color", "image.adjustments.matchColor", Kind.Pixels,
        listOf(
            Choice("sourceLayer", "Match the colours of", others.map { "json:${it.id}" to it.name }, "json:${others.first().id}"),
            MSlider("luminance", "Luminance", 1f, 200f, 100f),
            MSlider("intensity", "Color intensity", 1f, 200f, 100f),
            MSlider("fade", "Fade", 0f, 100f, 0f),
            Toggle("neutralize", "Neutralize", false),
        ),
    )
    return MenuGroup("Color tools", listOfNotNull(Catalog.replaceColor, match))
}

// ── Crop ─────────────────────────────────────────────────────────────────────────────────────

private val aspects = listOf("1:1" to 1f, "4:5" to 0.8f, "3:2" to 1.5f, "2:3" to 2f / 3f, "16:9" to 16f / 9f, "9:16" to 9f / 16f)

@Composable
private fun CropSheet(vm: EditorViewModel) {
    var angle by rememberSaveable { mutableFloatStateOf(0f) }
    ChipRow {
        aspects.forEach { (label, a) -> AssistChip(onClick = { vm.cropTo(a) }, label = { Text(label) }) }
    }
    ChipRow {
        AssistChip(onClick = { vm.edit("image.imageRotation.90ccw") }, label = { Text("Rotate left") })
        AssistChip(onClick = { vm.edit("image.imageRotation.90cw") }, label = { Text("Rotate right") })
        AssistChip(onClick = { vm.edit("image.imageRotation.180") }, label = { Text("Rotate 180°") })
        AssistChip(onClick = { vm.edit("image.imageRotation.flipCanvasHorizontal") }, label = { Text("Flip") })
        AssistChip(onClick = { vm.edit("image.imageRotation.flipCanvasVertical") }, label = { Text("Flip vertical") })
    }
    Row(verticalAlignment = Alignment.CenterVertically) {
        SymIcon(Sym.straighten, "Straighten", Modifier.size(20.dp))
        // The photo rotates as the slider moves (LiveStraighten); releasing commits the crop.
        Slider(
            value = angle,
            onValueChange = { angle = it; vm.straightener.drag(it) },
            onValueChangeFinished = { vm.straightener.release() },
            valueRange = -15f..15f,
            modifier = Modifier.weight(1f).padding(horizontal = 8.dp),
        )
        Text("%.1f°".format(angle), Modifier.width(52.dp), fontFamily = FontFamily.Monospace)
    }
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Button(onClick = { vm.straightener.done(); angle = 0f }, enabled = angle != 0f) { Text("Done") }
        TextButton(onClick = { angle = 0f; vm.straightener.drag(0f); vm.straightener.release() }) { Text("Reset") }
    }
}
