package dev.argo.imagelab.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Slider
import androidx.compose.material3.TextButton
import kotlin.math.roundToInt
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationRail
import androidx.compose.material3.NavigationRailItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.VerticalDivider
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.horizontalScroll
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.PointerEventType
import androidx.compose.ui.input.pointer.PointerType
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.argo.imagelab.ImageLabApp
import dev.argo.imagelab.core.EditorState
import dev.argo.imagelab.core.LayerInfo

/**
 * The editor (docs/INTERFACES.md §10): full-bleed canvas with chrome at the edges.
 *
 *  * Compact: top bar · canvas (+ layer strip on the right edge, floating add) · contextual sheet
 *    above a bottom tool bar; the agent is a sheet (owned by [AppRoot]).
 *  * Medium/expanded: navigation rail (tools) · canvas · right pane (agent, or the tool's options);
 *    expanded adds a layers column.
 *  * Tabletop: canvas on the top half, tools + agent on the bottom half. Book: canvas on one side
 *    of the hinge, the pane on the other.
 */
@Composable
fun EditorScreen(
    app: ImageLabApp,
    layout: WindowLayout,
    agentPane: (@Composable (Modifier) -> Unit)?,
    agentOpen: Boolean,
    onToggleAgent: () -> Unit,
    onHome: () -> Unit,
    onSettings: () -> Unit,
    onOpenUri: (Uri) -> Unit,
) {
    val vm: EditorViewModel = viewModel()
    val state by app.lab.state.collectAsState()
    val openPicker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri -> uri?.let(onOpenUri) }
    val layerPicker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri -> uri?.let(vm::addPhotoLayer) }
    val pickImage = { openPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }
    val pickLayer = { layerPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }

    val topBar: @Composable () -> Unit = {
        EditorTopBar(state, agentOpen, vm, onHome = onHome, onToggleAgent = onToggleAgent, onSettings = onSettings)
    }
    val canvasArea: @Composable (Modifier, Boolean) -> Unit = { m, showStrip ->
        Box(m) {
            EditorCanvas(state, vm, Modifier.fillMaxSize(), onOpen = pickImage, onLibrary = onHome)
            if (state.documents > 0) {
                if (showStrip) LayerStrip(state, vm, Modifier.align(Alignment.CenterEnd))
                AddMenu(vm, onPhotoLayer = pickLayer, modifier = Modifier.align(Alignment.BottomEnd).padding(16.dp))
            }
        }
    }
    val sheet: @Composable (Modifier) -> Unit = { m -> vm.tool?.let { ToolSheet(it, state, vm, m) } }

    if (vm.exportOpen) ExportDialog(vm)
    Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        val posture = layout.posture
        val density = LocalDensity.current
        when {
            posture is Posture.Tabletop -> Column(Modifier.fillMaxSize()) {
                val hingeTop = with(density) { posture.hinge.top.toDp() }
                val hingeHeight = with(density) { posture.hinge.height().toDp() }
                Column(Modifier.fillMaxWidth().height(hingeTop).safeDrawingPadding()) {
                    topBar()
                    canvasArea(Modifier.weight(1f).fillMaxWidth(), false)
                }
                Spacer(Modifier.height(hingeHeight))
                Row(Modifier.weight(1f).fillMaxWidth().safeDrawingPadding()) {
                    Column(Modifier.weight(1f).fillMaxHeight()) {
                        Box(Modifier.weight(1f).fillMaxWidth()) { sheet(Modifier.fillMaxSize()) }
                        ToolBar(vm)
                    }
                    if (agentPane != null) {
                        VerticalDivider()
                        agentPane(Modifier.weight(1f).fillMaxHeight())
                    }
                }
            }
            posture is Posture.Book -> Row(Modifier.fillMaxSize()) {
                val hingeLeft = with(density) { posture.hinge.left.toDp() }
                val hingeWidth = with(density) { posture.hinge.width().toDp() }
                Row(Modifier.width(hingeLeft).fillMaxHeight().safeDrawingPadding()) {
                    ToolRail(vm)
                    Column(Modifier.weight(1f)) {
                        topBar()
                        canvasArea(Modifier.weight(1f).fillMaxWidth(), true)
                    }
                }
                Spacer(Modifier.width(hingeWidth))
                Box(Modifier.weight(1f).fillMaxHeight().safeDrawingPadding()) {
                    if (agentPane != null) agentPane(Modifier.fillMaxSize()) else sheet(Modifier.fillMaxSize())
                }
            }
            layout.compact -> Column(Modifier.fillMaxSize().safeDrawingPadding()) {
                topBar()
                canvasArea(Modifier.weight(1f).fillMaxWidth(), true)
                AnimatedVisibility(vm.tool != null) {
                    sheet(Modifier.fillMaxWidth().heightIn(max = 300.dp))
                }
                ToolBar(vm)
            }
            else -> Column(Modifier.fillMaxSize().safeDrawingPadding()) {
                val paneOpen = agentPane != null || vm.tool != null
                // The layers column only when the canvas keeps room beside the right pane.
                val layersColumn = layout.expanded && state.documents > 0 && (!paneOpen || layout.widthDp >= 1200f)
                val paneWidth = if (layout.expanded) 360.dp else 300.dp
                // The top bar spans the window: beside a rail and a pane the canvas column is too narrow for it.
                topBar()
                Row(Modifier.weight(1f).fillMaxWidth()) {
                    ToolRail(vm)
                    canvasArea(Modifier.weight(1f).fillMaxHeight(), !layersColumn)
                    if (layersColumn) {
                        VerticalDivider()
                        LayersColumn(state, vm, Modifier.width(200.dp).fillMaxHeight())
                    }
                    if (agentPane != null) {
                        VerticalDivider()
                        agentPane(Modifier.width(paneWidth).fillMaxHeight())
                    } else if (vm.tool != null) {
                        VerticalDivider()
                        sheet(Modifier.width(paneWidth).fillMaxHeight())
                    }
                }
            }
        }
    }
}

// ── top bar ──────────────────────────────────────────────────────────────────────────────────

@Composable
private fun EditorTopBar(state: EditorState, agentOpen: Boolean, vm: EditorViewModel, onHome: () -> Unit, onToggleAgent: () -> Unit, onSettings: () -> Unit) {
    val hasDoc = state.documents > 0
    Row(Modifier.fillMaxWidth().padding(horizontal = 4.dp), verticalAlignment = Alignment.CenterVertically) {
        SymButton(Sym.home, "Library", onHome)
        Column(Modifier.weight(1f).padding(horizontal = 4.dp)) {
            Text(state.name ?: "ImageLab", style = MaterialTheme.typography.titleSmall, maxLines = 1, overflow = TextOverflow.Ellipsis)
            if (hasDoc) Text("${state.width}×${state.height}", style = MaterialTheme.typography.labelSmall, color = Color.Gray, maxLines = 1)
        }
        if (state.busy) CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
        if (hasDoc) CompareButton(vm)
        SymButton(Sym.undo, "Undo", { vm.undo() }, enabled = hasDoc)
        SymButton(Sym.redo, "Redo", { vm.redo() }, enabled = hasDoc)
        SymButton(Sym.export, "Export", { vm.exportOpen = true }, enabled = hasDoc)
        SymButton(Sym.agent, if (agentOpen) "Hide agent" else "Agent", onToggleAgent, tint = if (agentOpen) Accent else null)
        SymButton(Sym.settings, "Settings", onSettings)
    }
}

/** Hold to see the photo as it was opened (before), let go for the edit (after). */
@Composable
private fun CompareButton(vm: EditorViewModel) {
    Box(
        Modifier.size(40.dp).pointerInput(Unit) {
            detectTapGestures(onPress = {
                vm.compare(true)
                tryAwaitRelease()
                vm.compare(false)
            })
        },
        contentAlignment = Alignment.Center,
    ) { SymIcon(Sym.compare, "Hold to compare with the original", tint = if (vm.comparing) Accent else MaterialTheme.colorScheme.onSurface) }
}

/** Export: format, quality and size, then save and add to the gallery. */
@Composable
private fun ExportDialog(vm: EditorViewModel) {
    val spec = vm.exportSpec
    AlertDialog(
        onDismissRequest = { vm.exportOpen = false },
        title = { Text("Export") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("Format", style = MaterialTheme.typography.labelLarge)
                Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    listOf("jpg" to "JPEG", "png" to "PNG", "webp" to "WebP", "tif" to "TIFF").forEach { (f, l) ->
                        FilterChip(selected = spec.format == f, onClick = { vm.exportSpec = spec.copy(format = f) }, label = { Text(l) })
                    }
                }
                if (spec.format == "jpg" || spec.format == "webp") {
                    Text("Quality ${spec.quality}", style = MaterialTheme.typography.labelLarge)
                    Slider(value = spec.quality.toFloat(), onValueChange = { vm.exportSpec = spec.copy(quality = it.roundToInt()) }, valueRange = 40f..100f)
                }
                Text("Size (longest side)", style = MaterialTheme.typography.labelLarge)
                Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    listOf<Pair<Int?, String>>(null to "Full", 4096 to "4096", 2048 to "2048", 1080 to "1080").forEach { (m, l) ->
                        FilterChip(selected = spec.maxSide == m, onClick = { vm.exportSpec = spec.copy(maxSide = m) }, label = { Text(l) })
                    }
                }
                Text(if (spec.format == "png" || spec.format == "tif") "Lossless; larger files." else "Saved flattened to Pictures/ImageLab.", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
            }
        },
        confirmButton = { Button(onClick = { vm.export() }) { Text("Export") } },
        dismissButton = { TextButton(onClick = { vm.exportOpen = false }) { Text("Cancel") } },
    )
}

// ── canvas ───────────────────────────────────────────────────────────────────────────────────

/**
 * The preview with pinch-zoom and pan. In Select, one finger selects instead of panning:
 * a tap is a point prompt, a drag a box; two fingers still zoom and pan. Stylus pressure is
 * recorded for brushes; a hovering stylus shows a cursor ring.
 */
@Composable
private fun EditorCanvas(state: EditorState, vm: EditorViewModel, modifier: Modifier, onOpen: () -> Unit, onLibrary: () -> Unit) {
    val preview = state.preview
    val selectMode = vm.tool == Tool.Select
    val retouchMode = vm.tool == Tool.Retouch && vm.retouchBrush != null
    val paintMode = vm.tool == Tool.Paint || retouchMode
    val textMode = vm.tool == Tool.Text
    val handleMode = vm.tool == Tool.Transform && vm.handleMode != HandleMode.None
    val sampleMode = vm.sampling
    // The stroke being painted (screen points) and its pressures.
    var stroke by remember { mutableStateOf<List<Offset>>(emptyList()) }
    var box by remember { mutableStateOf(IntSize.Zero) }
    var draft by remember { mutableStateOf<List<Offset>>(emptyList()) }
    var hover by remember { mutableStateOf<Offset?>(null) }
    val padPx = with(LocalDensity.current) { CANVAS_PAD.toPx() }

    fun toDocOrClamp(p: Offset, clamp: Boolean): Offset? {
        val bmp = preview ?: return null
        val c = Offset(box.width / 2f, box.height / 2f)
        val u = c + (p - c - vm.offset) / vm.scale
        val fit = minOf((box.width - 2 * padPx) / bmp.width, (box.height - 2 * padPx) / bmp.height)
        val dw = bmp.width * fit
        val dh = bmp.height * fit
        val nx = (u.x - (c.x - dw / 2)) / dw
        val ny = (u.y - (c.y - dh / 2)) / dh
        if (!clamp && (nx !in 0f..1f || ny !in 0f..1f)) return null
        val (cx, cy) = nx.coerceIn(0f, 1f) to ny.coerceIn(0f, 1f)
        val docW = if (state.width > 0) state.width else bmp.width
        val docH = if (state.height > 0) state.height else bmp.height
        return Offset(cx * docW, cy * docH)
    }

    fun toDoc(p: Offset): Offset? = toDocOrClamp(p, clamp = false)

    /** Document px → screen (the inverse of [toDocOrClamp]). */
    fun toScreen(d: Offset): Offset? {
        val bmp = preview ?: return null
        val c = Offset(box.width / 2f, box.height / 2f)
        val fit = minOf((box.width - 2 * padPx) / bmp.width, (box.height - 2 * padPx) / bmp.height)
        val dw = bmp.width * fit
        val dh = bmp.height * fit
        val docW = if (state.width > 0) state.width else bmp.width
        val docH = if (state.height > 0) state.height else bmp.height
        val u = Offset(c.x - dw / 2 + d.x / docW * dw, c.y - dh / 2 + d.y / docH * dh)
        return c + (u - c) * vm.scale + vm.offset
    }
    val grabPx = with(LocalDensity.current) { 32.dp.toPx() }

    // A new document starts fitted, not at the last one's zoom and pan.
    LaunchedEffect(state.name, state.width, state.height) { vm.resetView() }

    Box(
        modifier
            // Zoomed or panned, the photo stays inside the canvas: it never covers the top bar,
            // the layer strip's edge or the tool sheet.
            .clipToBounds()
            .background(CanvasBackground)
            .onSizeChanged { box = it }
            .pointerInput(selectMode, paintMode, textMode, retouchMode, handleMode, sampleMode) {
                awaitEachGesture {
                    val down = awaitFirstDown(requireUnconsumed = false)
                    // Handles: drag the nearest one; in Pins, a tap elsewhere adds a pin.
                    if (handleMode) {
                        val idx = vm.handles.indices.minByOrNull { i -> toScreen(vm.handles[i].second)?.let { (it - down.position).getDistance() } ?: Float.MAX_VALUE }
                        val near = idx != null && (toScreen(vm.handles[idx].second)?.let { (it - down.position).getDistance() } ?: Float.MAX_VALUE) <= grabPx
                        var moved = false
                        var multiTouch = false
                        while (true) {
                            val event = awaitPointerEvent()
                            val pressed = event.changes.filter { it.pressed }
                            if (pressed.isEmpty()) break
                            if (pressed.size >= 2) {
                                multiTouch = true
                                vm.scale = (vm.scale * event.calculateZoom()).coerceIn(0.5f, 12f)
                                vm.offset += event.calculatePan()
                            } else if (!multiTouch) {
                                val ch = pressed[0]
                                if ((ch.position - down.position).getDistance() > viewConfiguration.touchSlop) moved = true
                                if (near && idx != null) {
                                    toDocOrClamp(ch.position, clamp = true)?.let { d -> vm.handles[idx] = vm.handles[idx].first.let { src -> if (vm.handleMode == HandleMode.Quad) d to d else src to d } }
                                } else if (moved) {
                                    vm.offset += ch.position - ch.previousPosition
                                }
                            }
                            event.changes.forEach { it.consume() }
                        }
                        if (!near && !moved && !multiTouch && vm.handleMode == HandleMode.Pins) toDoc(down.position)?.let { vm.handles += it to it }
                        return@awaitEachGesture
                    }
                    if (down.type == PointerType.Stylus) vm.stylusPressure = down.pressure
                    val start = down.position
                    val path = mutableListOf(start)
                    val pressures = mutableListOf(if (down.type == PointerType.Stylus) down.pressure else 1f)
                    var multi = false
                    var dragged = false
                    while (true) {
                        val event = awaitPointerEvent()
                        val pressed = event.changes.filter { it.pressed }
                        if (pressed.isEmpty()) break
                        if (pressed.size >= 2) {
                            multi = true
                            draft = emptyList()
                            stroke = emptyList()
                            vm.scale = (vm.scale * event.calculateZoom()).coerceIn(0.5f, 12f)
                            vm.offset += event.calculatePan()
                        } else if (!multi) {
                            val ch = pressed[0]
                            if (ch.type == PointerType.Stylus) vm.stylusPressure = ch.pressure
                            if (paintMode) {
                                if ((ch.position - path.last()).getDistance() > 2f) {
                                    path += ch.position
                                    pressures += if (ch.type == PointerType.Stylus) ch.pressure else 1f
                                    stroke = path.toList()
                                }
                            } else if (selectMode) {
                                if ((ch.position - start).getDistance() > viewConfiguration.touchSlop) dragged = true
                                if (dragged && (ch.position - path.last()).getDistance() > 3f) {
                                    path += ch.position
                                    draft = path.toList()
                                }
                            } else {
                                vm.offset += ch.position - ch.previousPosition
                                if ((ch.position - start).getDistance() > viewConfiguration.touchSlop) dragged = true
                            }
                        }
                        event.changes.forEach { it.consume() }
                    }
                    draft = emptyList()
                    if (paintMode && !multi) {
                        val pts = path.mapIndexedNotNull { i, p -> toDocOrClamp(p, clamp = true)?.let { d -> Triple(d.x, d.y, pressures.getOrElse(i) { 1f }) } }
                        if (retouchMode) vm.retouchStroke(pts) else vm.paintStroke(pts)
                        stroke = emptyList()
                    }
                    if (textMode && !multi) toDoc(start)?.let { vm.textAt = it }
                    if (sampleMode && !multi && !dragged) toDoc(start)?.let { vm.sampleAt(it.x, it.y) }
                    if (selectMode && !multi) {
                        if (dragged) {
                            val pts = if (vm.selectMode == SelectMode.Lasso) path else listOf(path.first(), path.last())
                            vm.canvasDrag(pts.mapNotNull { toDocOrClamp(it, clamp = true) })
                        } else {
                            toDoc(start)?.let { vm.canvasTap(it.x, it.y) }
                        }
                    }
                }
            }
            .pointerInput(Unit) {
                awaitPointerEventScope {
                    while (true) {
                        val e = awaitPointerEvent(PointerEventPass.Initial)
                        val ch = e.changes.firstOrNull() ?: continue
                        hover = when {
                            e.type == PointerEventType.Exit -> null
                            ch.type == PointerType.Stylus && !ch.pressed && (e.type == PointerEventType.Move || e.type == PointerEventType.Enter) -> ch.position
                            ch.pressed -> null
                            else -> hover
                        }
                    }
                }
            },
        contentAlignment = Alignment.Center,
    ) {
        if (preview == null) {
            EmptyCanvas(onOpen = onOpen, onNew = { vm.newCanvas() }, onLibrary = onLibrary)
        } else {
            val transform = Modifier.fillMaxSize().padding(CANVAS_PAD).graphicsLayer {
                scaleX = vm.scale
                scaleY = vm.scale
                translationX = vm.offset.x
                translationY = vm.offset.y
            }
            // While a slider moves, the GPU draws the change the engine has not rendered yet.
            // Straightening rotates the frame inside the photo's own rectangle (clipped), so the
            // drag shows exactly what the crop will keep.
            Box(transform.graphicsLayer { clip = vm.straightener.active; shape = FitRect(preview.width, preview.height) }) {
                val shown = if (vm.comparing) vm.before ?: preview else preview
                Image(
                    bitmap = shown.asImageBitmap(),
                    contentDescription = "Document preview",
                    contentScale = ContentScale.Fit,
                    colorFilter = if (vm.comparing) null else vm.live.matrix?.let { androidx.compose.ui.graphics.ColorFilter.colorMatrix(it) },
                    modifier = Modifier.fillMaxSize().graphicsLayer {
                        rotationZ = vm.straightener.rotation
                        scaleX = vm.straightener.zoom
                        scaleY = vm.straightener.zoom
                    },
                )
            }
            if (!vm.comparing) vm.selection?.overlay?.let { ov ->
                Image(bitmap = ov.asImageBitmap(), contentDescription = "Selection mask", contentScale = ContentScale.Fit, modifier = transform)
            }
        }
        Canvas(Modifier.fillMaxSize()) {
            // The stroke being painted, at its on-screen width.
            val lineTool = (vm.tool == Tool.Paint && (vm.paintMode == PaintMode.Gradient)) || (vm.tool == Tool.Retouch && vm.retouchBrush?.use == BrushUse.Drag)
            if (stroke.size >= 2 && lineTool) {
                drawLine(Accent, stroke.first(), stroke.last(), 3.dp.toPx())
                drawCircle(Accent, 6.dp.toPx(), stroke.first())
                drawCircle(Color.White, 6.dp.toPx(), stroke.last())
            } else if (stroke.size >= 2) {
                val bmp = preview
                val fit = if (bmp != null && bmp.width > 0) minOf((size.width - 2 * padPx) / bmp.width, (size.height - 2 * padPx) / bmp.height) else 1f
                val docW = if (state.width > 0) state.width.toFloat() else (bmp?.width ?: 1).toFloat()
                val screenPerDoc = fit * (bmp?.width ?: 1) / docW * vm.scale
                val color = if (vm.eraser) Color.White.copy(alpha = 0.5f) else Color(android.graphics.Color.parseColor(vm.paintColor)).copy(alpha = vm.paintOpacity)
                drawPath(
                    androidx.compose.ui.graphics.Path().apply {
                        moveTo(stroke[0].x, stroke[0].y)
                        stroke.drop(1).forEach { lineTo(it.x, it.y) }
                    },
                    color,
                    style = Stroke(width = (vm.paintSize * screenPerDoc).coerceAtLeast(1f), cap = androidx.compose.ui.graphics.StrokeCap.Round, join = androidx.compose.ui.graphics.StrokeJoin.Round),
                )
            }
            // The shape being drawn: box/rectangle, ellipse, or the freehand lasso path.
            if (draft.size >= 2) {
                val a = draft.first()
                val b = draft.last()
                val tl = Offset(minOf(a.x, b.x), minOf(a.y, b.y))
                val sz = Size(kotlin.math.abs(b.x - a.x), kotlin.math.abs(b.y - a.y))
                val stroke = Stroke(2.dp.toPx(), pathEffect = androidx.compose.ui.graphics.PathEffect.dashPathEffect(floatArrayOf(10f, 8f)))
                when (vm.selectMode) {
                    SelectMode.Ellipse -> drawOval(Accent, topLeft = tl, size = sz, style = stroke)
                    SelectMode.Lasso -> drawPath(
                        androidx.compose.ui.graphics.Path().apply {
                            moveTo(draft[0].x, draft[0].y)
                            draft.drop(1).forEach { lineTo(it.x, it.y) }
                        },
                        Accent, style = stroke,
                    )
                    SelectMode.Wand, SelectMode.Color -> {}
                    else -> drawRect(Accent, topLeft = tl, size = sz, style = stroke)
                }
            }
            // A rule-of-thirds-and-finer grid while straightening, to line up horizons.
            if (vm.straightener.active) {
                val step = size.minDimension / 8f
                var x = step
                while (x < size.width) { drawLine(Color.White.copy(alpha = 0.25f), Offset(x, 0f), Offset(x, size.height), 1f); x += step }
                var y = step
                while (y < size.height) { drawLine(Color.White.copy(alpha = 0.25f), Offset(0f, y), Offset(size.width, y), 1f); y += step }
            }
            hover?.let { drawCircle(Color.White.copy(alpha = 0.8f), radius = 14.dp.toPx(), center = it, style = Stroke(1.5.dp.toPx())) }
            // Perspective quad / puppet pins.
            if (vm.tool == Tool.Transform && vm.handleMode != HandleMode.None) {
                val pts = vm.handles.mapNotNull { (src, dst) -> toScreen(dst)?.let { d -> toScreen(src) to d } }
                if (vm.handleMode == HandleMode.Quad && pts.size == 4) {
                    val path = androidx.compose.ui.graphics.Path().apply { moveTo(pts[0].second.x, pts[0].second.y); pts.drop(1).forEach { lineTo(it.second.x, it.second.y) }; close() }
                    drawPath(path, Accent, style = Stroke(2.dp.toPx()))
                    // Thirds inside the quad help line up verticals.
                    for (t in listOf(1f / 3, 2f / 3)) {
                        fun lerp(a: Offset, b: Offset) = a + (b - a) * t
                        drawLine(Accent.copy(alpha = 0.5f), lerp(pts[0].second, pts[1].second), lerp(pts[3].second, pts[2].second), 1.dp.toPx())
                        drawLine(Accent.copy(alpha = 0.5f), lerp(pts[0].second, pts[3].second), lerp(pts[1].second, pts[2].second), 1.dp.toPx())
                    }
                }
                pts.forEach { (src, dst) ->
                    if (vm.handleMode == HandleMode.Pins && src != null && (src - dst).getDistance() > 2f) drawLine(Color.White.copy(alpha = 0.7f), src, dst, 2.dp.toPx())
                    drawCircle(Color.Black, 11.dp.toPx(), dst)
                    drawCircle(if (vm.handleMode == HandleMode.Pins) Color(0xFFFFC94D) else Accent, 8.dp.toPx(), dst)
                }
            }
            // Clone / Heal from: the source point.
            if (vm.tool == Tool.Retouch && vm.retouchBrush?.use == BrushUse.Source) vm.cloneSource?.let { src ->
                toScreen(src)?.let { c ->
                    drawCircle(Color.White, 10.dp.toPx(), c, style = Stroke(2.dp.toPx()))
                    drawLine(Color.White, c - Offset(16.dp.toPx(), 0f), c + Offset(16.dp.toPx(), 0f), 1.5.dp.toPx())
                    drawLine(Color.White, c - Offset(0f, 16.dp.toPx()), c + Offset(0f, 16.dp.toPx()), 1.5.dp.toPx())
                }
            }
        }
        if (vm.comparing) Text("Before", color = Color.White, modifier = Modifier.align(Alignment.TopStart).padding(12.dp).background(Color(0x99000000), RoundedCornerShape(6.dp)).padding(horizontal = 8.dp, vertical = 4.dp))
        if (vm.selecting) CircularProgressIndicator(Modifier.align(Alignment.TopCenter).padding(12.dp).size(24.dp), strokeWidth = 2.dp)
    }
}

private val CANVAS_PAD = 8.dp

@Composable
private fun EmptyCanvas(onOpen: () -> Unit, onNew: () -> Unit, onLibrary: () -> Unit) {
    Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(10.dp)) {
        SymIcon(Sym.image, null, Modifier.size(48.dp), tint = Color.Gray)
        Text("Open a photo to start editing.", color = Color.Gray)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = onOpen) { Text("Open photo") }
            OutlinedButton(onClick = onNew) { Text("New canvas") }
        }
        OutlinedButton(onClick = onLibrary) { Text("From the Library") }
    }
}

// ── layers ───────────────────────────────────────────────────────────────────────────────────

/** Collapsible strip of layers on the canvas's right edge (names/kinds from `doc.inspect`). */
@Composable
private fun LayerStrip(state: EditorState, vm: EditorViewModel, modifier: Modifier) {
    Row(modifier.padding(vertical = 72.dp), verticalAlignment = Alignment.CenterVertically) {
        Surface(
            color = Color(0xCC1C1C1E),
            shape = RoundedCornerShape(topStart = 8.dp, bottomStart = 8.dp),
            modifier = Modifier.clickable { vm.layerStripOpen = !vm.layerStripOpen },
        ) {
            SymIcon(if (vm.layerStripOpen) Sym.chevronRight else Sym.chevronLeft, if (vm.layerStripOpen) "Collapse layers" else "Show layers", Modifier.padding(vertical = 10.dp, horizontal = 2.dp))
        }
        AnimatedVisibility(vm.layerStripOpen) {
            Surface(color = Color(0xCC1C1C1E), modifier = Modifier.width(64.dp).heightIn(max = 360.dp)) {
                LazyColumn(Modifier.padding(4.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    items(state.layers) { l ->
                        val active = l.id != null && l.id == state.activeLayer
                        Column(
                            Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(4.dp))
                                .background(if (active) Accent.copy(alpha = 0.35f) else Color(0x33FFFFFF))
                                .clickable { vm.selectLayer(l.id) }
                                .padding(4.dp),
                            horizontalAlignment = Alignment.CenterHorizontally,
                        ) {
                            SymIcon(kindIcon(l), l.kind, Modifier.size(20.dp), tint = if (l.visible) Color.White else Color.Gray)
                            Text(l.name, fontSize = 9.sp, maxLines = 1, overflow = TextOverflow.Ellipsis, color = if (l.visible) Color.White else Color.Gray)
                        }
                    }
                }
            }
        }
    }
}

/** Expanded windows: a full layers column beside the canvas. */
@Composable
private fun LayersColumn(state: EditorState, vm: EditorViewModel, modifier: Modifier) {
    Column(modifier) {
        Text("Layers", style = MaterialTheme.typography.titleSmall, modifier = Modifier.padding(12.dp))
        LayerList(state, vm, Modifier.weight(1f))
    }
}

@Composable
fun LayerList(state: EditorState, vm: EditorViewModel, modifier: Modifier) {
    LazyColumn(modifier) {
        items(state.layers) { l ->
            val active = l.id != null && l.id == state.activeLayer
            Row(
                Modifier.fillMaxWidth().background(if (active) Accent.copy(alpha = 0.25f) else Color.Transparent).clickable { vm.selectLayer(l.id) }.padding(horizontal = 12.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                SymIcon(kindIcon(l), null, Modifier.size(20.dp))
                Column(Modifier.weight(1f).padding(horizontal = 8.dp)) {
                    Text(l.name, maxLines = 1, overflow = TextOverflow.Ellipsis, style = MaterialTheme.typography.bodyMedium)
                    Text(l.kind + if (l.hasMask) " · mask" else "", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
                }
                SymIcon(
                    if (l.visible) Sym.visible else Sym.hidden, if (l.visible) "Hide ${l.name}" else "Show ${l.name}",
                    Modifier.size(32.dp).clickable { vm.setLayerVisible(l.id, !l.visible) }.padding(7.dp), tint = Color.Gray,
                )
            }
        }
    }
}

fun kindIcon(l: LayerInfo): Int = when {
    l.kind.contains("adjust", ignoreCase = true) -> Sym.adjust
    l.kind.contains("text", ignoreCase = true) || l.kind.contains("type", ignoreCase = true) -> Sym.text
    l.kind.contains("fill", ignoreCase = true) -> Sym.fill
    l.kind.contains("group", ignoreCase = true) -> Sym.layers
    else -> Sym.image
}

// ── floating add ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun AddMenu(vm: EditorViewModel, onPhotoLayer: () -> Unit, modifier: Modifier) {
    var open by remember { mutableStateOf(false) }
    Box(modifier) {
        FloatingActionButton(onClick = { open = true }, containerColor = Accent) { SymIcon(Sym.add, "Add layer") }
        DropdownMenu(open, onDismissRequest = { open = false }) {
            DropdownMenuItem(text = { Text("Photo layer") }, leadingIcon = { SymIcon(Sym.addPhoto, null) }, onClick = { open = false; onPhotoLayer() })
            DropdownMenuItem(text = { Text("Text") }, leadingIcon = { SymIcon(Sym.text, null) }, onClick = { open = false; vm.addText() })
            DropdownMenuItem(text = { Text("Solid fill") }, leadingIcon = { SymIcon(Sym.fill, null) }, onClick = { open = false; vm.addSolidFill() })
            DropdownMenuItem(text = { Text("Gradient fill") }, leadingIcon = { SymIcon(Sym.gradient, null) }, onClick = { open = false; vm.addGradientFill() })
            DropdownMenuItem(text = { Text("Adjustment layer") }, leadingIcon = { SymIcon(Sym.adjust, null) }, onClick = { open = false; vm.tool = Tool.Adjust })
            HorizontalDivider()
            DropdownMenuItem(text = { Text("Generate (soon)") }, leadingIcon = { SymIcon(Sym.generate, null) }, enabled = false, onClick = {})
        }
    }
}

// ── tool bar / rail ──────────────────────────────────────────────────────────────────────────

private fun toolIcon(t: Tool) = when (t) {
    Tool.Enhance -> Sym.enhance
    Tool.Layers -> Sym.layers
    Tool.Select -> Sym.select
    Tool.Adjust -> Sym.adjust
    Tool.Filters -> Sym.filters
    Tool.Effects -> Sym.effects
    Tool.Retouch -> Sym.retouch
    Tool.Paint -> Sym.brush
    Tool.Text -> Sym.text
    Tool.Transform -> Sym.transform
    Tool.Crop -> Sym.crop
    Tool.History -> Sym.history
}

/** Nine tools: a horizontally scrolling bar on phones (more than a NavigationBar holds). */
@Composable
private fun ToolBar(vm: EditorViewModel) {
    Surface(color = MaterialTheme.colorScheme.surfaceContainer) {
        Row(
            Modifier.fillMaxWidth().height(72.dp).horizontalScroll(rememberScrollState()).padding(horizontal = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Tool.entries.forEach { t -> ToolButton(t, vm.tool == t) { vm.pickTool(t) } }
        }
    }
}

@Composable
private fun ToolButton(t: Tool, selected: Boolean, onClick: () -> Unit) {
    Column(
        Modifier.width(72.dp).clickable(onClick = onClick).padding(vertical = 6.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Box(
            Modifier.size(width = 52.dp, height = 30.dp).background(if (selected) Accent.copy(alpha = 0.3f) else Color.Transparent, androidx.compose.foundation.shape.RoundedCornerShape(15.dp)),
            contentAlignment = Alignment.Center,
        ) { SymIcon(toolIcon(t), null, tint = if (selected) Accent else MaterialTheme.colorScheme.onSurface) }
        Text(t.label, fontSize = 11.sp, maxLines = 1, color = if (selected) Accent else MaterialTheme.colorScheme.onSurface)
    }
}

@Composable
private fun ToolRail(vm: EditorViewModel) {
    NavigationRail(windowInsets = WindowInsets(0.dp)) {
        Spacer(Modifier.height(8.dp))
        Column(Modifier.verticalScroll(rememberScrollState())) {
        Tool.entries.forEach { t ->
            NavigationRailItem(
                selected = vm.tool == t,
                onClick = { vm.pickTool(t) },
                icon = { SymIcon(toolIcon(t), null) },
                label = { Text(t.label, maxLines = 1) },
            )
        }
        }
    }
}


/** The rectangle a `w×h` bitmap occupies under ContentScale.Fit, centred: the clip while straightening. */
private class FitRect(private val w: Int, private val h: Int) : androidx.compose.ui.graphics.Shape {
    override fun createOutline(size: Size, layoutDirection: androidx.compose.ui.unit.LayoutDirection, density: androidx.compose.ui.unit.Density): androidx.compose.ui.graphics.Outline {
        if (w <= 0 || h <= 0) return androidx.compose.ui.graphics.Outline.Rectangle(androidx.compose.ui.geometry.Rect(Offset.Zero, size))
        val fit = minOf(size.width / w, size.height / h)
        val (dw, dh) = w * fit to h * fit
        val left = (size.width - dw) / 2f
        val top = (size.height - dh) / 2f
        return androidx.compose.ui.graphics.Outline.Rectangle(androidx.compose.ui.geometry.Rect(left, top, left + dw, top + dh))
    }
}
