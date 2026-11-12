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
                Tool.Layers -> LayersSheet(state, vm)
                Tool.Select -> SelectSheet(vm)
                Tool.Adjust -> {
                    AdjustSheet(vm)
                    HorizontalDivider(Modifier.padding(vertical = 8.dp))
                    CommandSheet(listOf(Catalog.light, Catalog.adjustmentLayers), vm.command)
                }
                Tool.Filters -> CommandSheet(Catalog.filters, vm.command, note = if (vm.selection != null) "Filters apply inside the selection." else "Filters change the active layer's pixels; the photo follows the sliders.")
                Tool.Effects -> CommandSheet(listOf(Catalog.styles), vm.command, note = "Effects style the active layer (not the locked Background: duplicate it or use a text/shape layer).")
                Tool.Retouch -> RetouchSheet(vm)
                Tool.Paint -> PaintSheet(vm)
                Tool.Text -> TextSheet(state, vm)
                Tool.Crop -> {
                    CropSheet(vm)
                    HorizontalDivider(Modifier.padding(vertical = 8.dp))
                    CommandSheet(listOf(Catalog.document), vm.command)
                }
            }
        }
    }
}

@Composable
private fun ChipRow(content: @Composable () -> Unit) {
    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) { content() }
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
        AssistChip(onClick = { vm.edit("layer.flattenImage") }, label = { Text("Flatten") })
    }
    ChipRow {
        AssistChip(onClick = { vm.maskFromSelection() }, label = { Text(if (vm.selection != null) "Mask from selection" else "Add mask") })
        AssistChip(onClick = { vm.edit("layer.layerMask.hideAll") }, label = { Text("Black mask") })
        AssistChip(onClick = { vm.edit("layer.layerMask.delete") }, label = { Text("Delete mask") })
        AssistChip(onClick = { vm.edit("layer.layerMask.apply") }, label = { Text("Apply mask") })
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
    Text("Draw on the photo with one finger (two fingers zoom). Pressure works with a stylus.", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
    ChipRow {
        FilterChip(selected = !vm.eraser, onClick = { vm.eraser = false }, label = { Text("Brush") }, leadingIcon = { SymIcon(Sym.brush, null, Modifier.size(18.dp)) })
        FilterChip(selected = vm.eraser, onClick = { vm.eraser = true }, label = { Text("Eraser") }, leadingIcon = { SymIcon(Sym.erase, null, Modifier.size(18.dp)) })
        FilterChip(selected = vm.paintOnNewLayer, onClick = { vm.paintOnNewLayer = !vm.paintOnNewLayer }, label = { Text("On a new layer") })
        AssistChip(onClick = vm::newPaintLayer, label = { Text("Next layer") })
    }
    if (!vm.eraser) Swatches(paintSwatches, vm.paintColor) { vm.paintColor = it }
    LabeledSlider("Size", vm.paintSize, 1f..300f, { vm.paintSize = it }) { "${it.roundToInt()} px" }
    LabeledSlider("Opacity", vm.paintOpacity, 0.05f..1f, { vm.paintOpacity = it }) { "${(it * 100).roundToInt()}%" }
    LabeledSlider("Hardness", vm.paintHardness, 0f..1f, { vm.paintHardness = it }) { "${(it * 100).roundToInt()}%" }
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
    }
    Text(hint, style = MaterialTheme.typography.bodySmall, color = Color.Gray)
    SegModelRow(vm)
    ChipRow {
        SelectMode.entries.forEach { m -> FilterChip(selected = vm.selectMode == m, onClick = { vm.selectMode = m }, label = { Text(m.label) }) }
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
    Text("Content-aware tools use the selection; select what to fix first.", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
    ChipRow {
        AssistChip(onClick = { vm.edit("edit.contentAwareFill", masked = true) }, label = { Text("Content-aware fill") }, leadingIcon = { SymIcon(Sym.fill, null, Modifier.size(18.dp)) })
        AssistChip(onClick = vm::removeSelected, label = { Text("Remove") }, leadingIcon = { SymIcon(Sym.erase, null, Modifier.size(18.dp)) })
        AssistChip(onClick = { vm.denoise() }, label = { Text("Denoise") })
        AssistChip(onClick = { vm.edit("filter.noise.dustAndScratches", """{"radius":2,"threshold":10}""", masked = true) }, label = { Text("Dust & scratches") })
    }
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 4.dp)) {
        OutlinedButton(onClick = {}, enabled = false) { Text("Heal brush (soon)") }
        Text(
            "  stylus pressure ${(vm.stylusPressure * 100).roundToInt()}%",
            style = MaterialTheme.typography.labelSmall,
            color = Color.Gray,
        )
    }
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
        AssistChip(onClick = { vm.edit("image.imageRotation.flipCanvasHorizontal") }, label = { Text("Flip") })
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
