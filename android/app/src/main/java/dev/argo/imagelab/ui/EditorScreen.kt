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
            else -> Row(Modifier.fillMaxSize().safeDrawingPadding()) {
                ToolRail(vm)
                Column(Modifier.weight(1f).fillMaxHeight()) {
                    topBar()
                    Row(Modifier.weight(1f)) {
                        canvasArea(Modifier.weight(1f).fillMaxHeight(), !layout.expanded)
                        if (layout.expanded && state.documents > 0) {
                            VerticalDivider()
                            LayersColumn(state, vm, Modifier.width(200.dp).fillMaxHeight())
                        }
                    }
                }
                val paneWidth = if (layout.expanded) 380.dp else 320.dp
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

// ── top bar ──────────────────────────────────────────────────────────────────────────────────

@Composable
private fun EditorTopBar(state: EditorState, agentOpen: Boolean, vm: EditorViewModel, onHome: () -> Unit, onToggleAgent: () -> Unit, onSettings: () -> Unit) {
    val hasDoc = state.documents > 0
    Row(Modifier.fillMaxWidth().padding(horizontal = 4.dp), verticalAlignment = Alignment.CenterVertically) {
        SymButton(Sym.home, "Library", onHome)
        Column(Modifier.weight(1f).padding(horizontal = 4.dp)) {
            Text(state.name ?: "ImageLab", style = MaterialTheme.typography.titleSmall, maxLines = 1, overflow = TextOverflow.Ellipsis)
            if (hasDoc) Text("${state.width}×${state.height}", style = MaterialTheme.typography.labelSmall, color = Color.Gray)
        }
        if (state.busy) CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
        SymButton(Sym.undo, "Undo", { vm.undo() }, enabled = hasDoc)
        SymButton(Sym.redo, "Redo", { vm.redo() }, enabled = hasDoc)
        SymButton(Sym.export, "Export", { vm.export() }, enabled = hasDoc)
        SymButton(Sym.agent, if (agentOpen) "Hide agent" else "Agent", onToggleAgent, tint = if (agentOpen) Accent else null)
        SymButton(Sym.settings, "Settings", onSettings)
    }
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
    var box by remember { mutableStateOf(IntSize.Zero) }
    var draft by remember { mutableStateOf<Pair<Offset, Offset>?>(null) }
    var hover by remember { mutableStateOf<Offset?>(null) }
    val padPx = with(LocalDensity.current) { CANVAS_PAD.toPx() }

    fun toDoc(p: Offset): Offset? {
        val bmp = preview ?: return null
        val c = Offset(box.width / 2f, box.height / 2f)
        val u = c + (p - c - vm.offset) / vm.scale
        val fit = minOf((box.width - 2 * padPx) / bmp.width, (box.height - 2 * padPx) / bmp.height)
        val dw = bmp.width * fit
        val dh = bmp.height * fit
        val nx = (u.x - (c.x - dw / 2)) / dw
        val ny = (u.y - (c.y - dh / 2)) / dh
        if (nx !in 0f..1f || ny !in 0f..1f) return null
        val docW = if (state.width > 0) state.width else bmp.width
        val docH = if (state.height > 0) state.height else bmp.height
        return Offset(nx * docW, ny * docH)
    }

    Box(
        modifier
            .background(CanvasBackground)
            .onSizeChanged { box = it }
            .pointerInput(selectMode) {
                awaitEachGesture {
                    val down = awaitFirstDown(requireUnconsumed = false)
                    if (down.type == PointerType.Stylus) vm.stylusPressure = down.pressure
                    val start = down.position
                    var last = start
                    var multi = false
                    var dragged = false
                    while (true) {
                        val event = awaitPointerEvent()
                        val pressed = event.changes.filter { it.pressed }
                        if (pressed.isEmpty()) break
                        if (pressed.size >= 2) {
                            multi = true
                            draft = null
                            vm.scale = (vm.scale * event.calculateZoom()).coerceIn(0.5f, 12f)
                            vm.offset += event.calculatePan()
                        } else if (!multi) {
                            val ch = pressed[0]
                            if (ch.type == PointerType.Stylus) vm.stylusPressure = ch.pressure
                            if (selectMode) {
                                if ((ch.position - start).getDistance() > viewConfiguration.touchSlop) dragged = true
                                if (dragged) draft = start to ch.position
                            } else {
                                vm.offset += ch.position - ch.previousPosition
                            }
                            last = ch.position
                        }
                        event.changes.forEach { it.consume() }
                    }
                    draft = null
                    if (selectMode && !multi) {
                        if (dragged) {
                            val a = toDoc(start)
                            val b = toDoc(last)
                            if (a != null && b != null) vm.selectBox(a.x, a.y, b.x, b.y)
                        } else {
                            toDoc(start)?.let { vm.selectPoint(it.x, it.y) }
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
            Image(bitmap = preview.asImageBitmap(), contentDescription = "Document preview", contentScale = ContentScale.Fit, modifier = transform)
            vm.selection?.overlay?.let { ov ->
                Image(bitmap = ov.asImageBitmap(), contentDescription = "Selection mask", contentScale = ContentScale.Fit, modifier = transform)
            }
        }
        Canvas(Modifier.fillMaxSize()) {
            draft?.let { (a, b) ->
                drawRect(Accent, topLeft = Offset(minOf(a.x, b.x), minOf(a.y, b.y)), size = Size(kotlin.math.abs(b.x - a.x), kotlin.math.abs(b.y - a.y)), style = Stroke(2.dp.toPx()))
            }
            hover?.let { drawCircle(Color.White.copy(alpha = 0.8f), radius = 14.dp.toPx(), center = it, style = Stroke(1.5.dp.toPx())) }
        }
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
                SymIcon(if (l.visible) Sym.visible else Sym.hidden, if (l.visible) "Visible" else "Hidden", Modifier.size(18.dp), tint = Color.Gray)
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
    Tool.Layers -> Sym.layers
    Tool.Select -> Sym.select
    Tool.Adjust -> Sym.adjust
    Tool.Retouch -> Sym.retouch
    Tool.Crop -> Sym.crop
}

@Composable
private fun ToolBar(vm: EditorViewModel) {
    NavigationBar(Modifier.height(72.dp)) {
        Tool.entries.forEach { t ->
            NavigationBarItem(
                selected = vm.tool == t,
                onClick = { vm.pickTool(t) },
                icon = { SymIcon(toolIcon(t), null) },
                label = { Text(t.label, fontSize = 11.sp) },
            )
        }
    }
}

@Composable
private fun ToolRail(vm: EditorViewModel) {
    NavigationRail {
        Spacer(Modifier.height(8.dp))
        Tool.entries.forEach { t ->
            NavigationRailItem(
                selected = vm.tool == t,
                onClick = { vm.pickTool(t) },
                icon = { SymIcon(toolIcon(t), null) },
                label = { Text(t.label) },
            )
        }
    }
}
