package dev.argo.imagelab.ui

import android.content.ContentValues
import android.net.Uri
import android.provider.MediaStore
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.argo.imagelab.ImageLabApp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

enum class Panel(val label: String) { Adjust("Adjust"), Filters("Filters"), Agent("Agent"), Cli("CLI"), Mcp("MCP") }

@Composable
fun EditorScreen(app: ImageLabApp, onPick: (Uri) -> Unit) {
    val lab = app.lab
    val state by lab.state.collectAsState()
    val scope = rememberCoroutineScope()
    val context = LocalContext.current
    var panel by remember { mutableStateOf<Panel?>(Panel.Adjust) }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri -> uri?.let(onPick) }

    // Edits made from outside the app (MCP over HTTP) bump the document's revision; follow them.
    LaunchedEffect(Unit) {
        var last: String? = null
        while (true) {
            delay(1000)
            if (lab.core.mcpHttpPort() == null) continue
            val info = withContext(Dispatchers.IO) { runCatching { lab.core.sessionInfo() }.getOrNull() } ?: continue
            val key = Regex("\"revision\":\\d+|\"active\":\\d+|\"index\":\\d+").findAll(info).joinToString { it.value }
            if (last != null && key != last) lab.refresh()
            last = key
        }
    }

    Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        Column(Modifier.fillMaxSize().safeDrawingPadding()) {
            TopBar(
                title = state.name?.let { "$it  ·  ${state.width}×${state.height}" } ?: "ImageLab",
                busy = state.busy,
                hasDoc = state.documents > 0,
                onOpen = { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                onNew = { scope.launch { lab.call { it.newDocument("""{"width":1080,"height":1350,"background":"white","name":"Untitled"}""") } } },
                onUndo = { scope.launch { lab.call { it.undo() } } },
                onRedo = { scope.launch { lab.call { it.redo() } } },
                onExport = {
                    scope.launch {
                        val stamp = System.currentTimeMillis()
                        val rel = "exports/imagelab-$stamp.jpg"
                        val saved = lab.call(refreshAfter = false) { it.save(rel, null) }
                        if (saved.isSuccess) {
                            // Also hand a copy to the gallery (Pictures/ImageLab).
                            withContext(Dispatchers.IO) {
                                val values = ContentValues().apply {
                                    put(MediaStore.Images.Media.DISPLAY_NAME, "imagelab-$stamp.jpg")
                                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                                    put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/ImageLab")
                                }
                                val resolver = context.contentResolver
                                resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)?.let { uri ->
                                    resolver.openOutputStream(uri)?.use { out -> File(app.workspace, rel).inputStream().use { it.copyTo(out) } }
                                }
                            }
                        }
                    }
                },
            )
            Canvas(state.preview, Modifier.weight(1f).fillMaxWidth())
            state.message?.let { msg ->
                Row(Modifier.fillMaxWidth().background(Color(0xFF5A1E1E)).padding(8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text(msg, Modifier.weight(1f), color = Color.White, fontSize = 13.sp)
                    TextButton(onClick = lab::clearMessage) { Text("Dismiss") }
                }
            }
            panel?.let { p ->
                Box(Modifier.fillMaxWidth().height(if (p == Panel.Agent || p == Panel.Cli) 340.dp else 230.dp).background(MaterialTheme.colorScheme.surface)) {
                    when (p) {
                        Panel.Adjust -> AdjustPanel(lab, scope)
                        Panel.Filters -> FiltersPanel(lab, scope)
                        Panel.Agent -> AgentPanel(app, scope)
                        Panel.Cli -> CliPanel(app, scope)
                        Panel.Mcp -> McpPanel(app)
                    }
                }
            }
            NavigationBar {
                Panel.entries.forEach { p ->
                    NavigationBarItem(
                        selected = panel == p,
                        onClick = { panel = if (panel == p) null else p },
                        icon = { Text(p.label.take(1), fontSize = 18.sp) },
                        label = { Text(p.label) },
                    )
                }
            }
        }
    }
}

@Composable
private fun TopBar(title: String, busy: Boolean, hasDoc: Boolean, onOpen: () -> Unit, onNew: () -> Unit, onUndo: () -> Unit, onRedo: () -> Unit, onExport: () -> Unit) {
    Column(Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 4.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(title, Modifier.weight(1f), style = MaterialTheme.typography.titleMedium, maxLines = 1)
            if (busy) CircularProgressIndicator(Modifier.size(20.dp), strokeWidth = 2.dp)
        }
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Button(onClick = onOpen) { Text("Open") }
            OutlinedButton(onClick = onNew) { Text("New") }
            OutlinedButton(onClick = onUndo, enabled = hasDoc) { Text("↶") }
            OutlinedButton(onClick = onRedo, enabled = hasDoc) { Text("↷") }
            OutlinedButton(onClick = onExport, enabled = hasDoc) { Text("Export") }
        }
    }
}

/** The preview, with pinch-zoom and pan. Double-check exports at full size: this is ≤2048 px. */
@Composable
private fun Canvas(preview: android.graphics.Bitmap?, modifier: Modifier) {
    var scale by remember { mutableFloatStateOf(1f) }
    var offset by remember { mutableStateOf(Offset.Zero) }
    Box(
        modifier
            .background(Color(0xFF0B0B0B))
            .pointerInput(Unit) {
                detectTransformGestures { _, pan, zoom, _ ->
                    scale = (scale * zoom).coerceIn(0.5f, 12f)
                    offset += pan
                }
            },
        contentAlignment = Alignment.Center,
    ) {
        if (preview == null) {
            Text("Open a photo, or start a new document.", color = Color.Gray)
        } else {
            Image(
                bitmap = preview.asImageBitmap(),
                contentDescription = "Document preview",
                contentScale = ContentScale.Fit,
                modifier = Modifier.fillMaxSize().padding(8.dp).graphicsLayer {
                    scaleX = scale
                    scaleY = scale
                    translationX = offset.x
                    translationY = offset.y
                },
            )
        }
    }
}
