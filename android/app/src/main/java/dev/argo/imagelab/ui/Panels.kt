package dev.argo.imagelab.ui

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.argo.imagelab.ImageLabApp
import dev.argo.imagelab.agent.ChatItem
import dev.argo.imagelab.core.Lab
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.roundToInt

/** A slider-driven PhotoCraft command: `params(value)` builds its JSON. */
private data class Adjustment(val label: String, val range: ClosedFloatingPointRange<Float>, val command: String, val params: (Int) -> String)

// Adjustment layers keep each edit live and maskable (PhotoCraft's non-destructive model).
private val adjustments = listOf(
    Adjustment("Brightness", -150f..150f, "layer.newAdjustmentLayer.brightnessContrast") { """{"brightness":$it}""" },
    Adjustment("Contrast", -50f..100f, "layer.newAdjustmentLayer.brightnessContrast") { """{"contrast":$it}""" },
    Adjustment("Exposure ×10", -50f..50f, "layer.newAdjustmentLayer.exposure") { """{"exposure":${it / 10.0}}""" },
    Adjustment("Vibrance", -100f..100f, "layer.newAdjustmentLayer.vibrance") { """{"vibrance":$it}""" },
    Adjustment("Saturation", -100f..100f, "layer.newAdjustmentLayer.hueSaturation") { """{"saturation":$it}""" },
    Adjustment("Hue", -180f..180f, "layer.newAdjustmentLayer.hueSaturation") { """{"hue":$it}""" },
)

private val quick = listOf(
    "Auto Tone" to "image.autoTone",
    "Auto Contrast" to "image.autoContrast",
    "Auto Color" to "image.autoColor",
    "Rotate ⟳" to "image.imageRotation.90cw",
    "Flip ↔" to "image.imageRotation.flipCanvasHorizontal",
    "B&W" to "layer.newAdjustmentLayer.blackWhite",
    "Invert" to "layer.newAdjustmentLayer.invert",
    "Flatten" to "layer.flattenImage",
)

@Composable
fun AdjustPanel(lab: Lab, scope: CoroutineScope) {
    var selected by remember { mutableStateOf(adjustments.first()) }
    var value by remember(selected) { mutableFloatStateOf(0f) }
    Column(Modifier.fillMaxSize().padding(10.dp)) {
        Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            quick.forEach { (label, id) -> AssistChip(onClick = { scope.launch { lab.call { it.run(id, "{}") } } }, label = { Text(label) }) }
        }
        Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            adjustments.forEach { a -> FilterChip(selected = a == selected, onClick = { selected = a }, label = { Text(a.label) }) }
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Slider(value = value, onValueChange = { value = it }, valueRange = selected.range, modifier = Modifier.weight(1f))
            Text(value.roundToInt().toString(), Modifier.width(44.dp), fontFamily = FontFamily.Monospace)
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { scope.launch { lab.call { it.run(selected.command, selected.params(value.roundToInt())) } } }) { Text("Add ${selected.label} layer") }
            TextButton(onClick = { value = 0f }) { Text("Reset") }
        }
    }
}

@Composable
fun FiltersPanel(lab: Lab, scope: CoroutineScope) {
    val filters = listOf(
        "Gaussian Blur 4" to ("filter.blur.gaussianBlur" to """{"radius":4}"""),
        "Unsharp Mask" to ("filter.sharpen.unsharpMask" to """{"amount":80,"radius":1.2,"threshold":2}"""),
        "Reduce Noise" to ("filter.noise.reduceNoise" to """{"strength":6}"""),
        "Median 2" to ("filter.noise.median" to """{"radius":2}"""),
        "Despeckle" to ("filter.noise.despeckle" to "{}"),
        "Add Noise 5%" to ("filter.noise.addNoise" to """{"amount":5}"""),
    )
    Column(Modifier.fillMaxSize().padding(10.dp).verticalScroll(rememberScrollState())) {
        Text("Filters run on the active layer (PhotoCraft's CPU filters, multithreaded).", style = MaterialTheme.typography.bodySmall)
        filters.chunked(2).forEach { pair ->
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                pair.forEach { (label, cmd) ->
                    OutlinedButton(onClick = { scope.launch { lab.call { it.run(cmd.first, cmd.second) } } }, modifier = Modifier.weight(1f)) { Text(label) }
                }
            }
        }
    }
}

@Composable
fun AgentPanel(app: ImageLabApp, scope: CoroutineScope) {
    val agent = app.agent
    val items by agent.items.collectAsState()
    val running by agent.running.collectAsState()
    var input by remember { mutableStateOf("") }
    var showSettings by remember { mutableStateOf(app.settings.apiKey.isBlank()) }
    val listState = rememberLazyListState()
    LaunchedEffect(items.size) { if (items.isNotEmpty()) listState.animateScrollToItem(items.size - 1) }

    Column(Modifier.fillMaxSize().padding(8.dp)) {
        if (showSettings) {
            var key by remember { mutableStateOf(app.settings.apiKey) }
            var model by remember { mutableStateOf(app.settings.model) }
            OutlinedTextField(key, { key = it }, label = { Text("Anthropic API key") }, singleLine = true, visualTransformation = PasswordVisualTransformation(), modifier = Modifier.fillMaxWidth())
            OutlinedTextField(model, { model = it }, label = { Text("Model") }, singleLine = true, modifier = Modifier.fillMaxWidth())
            Row {
                Button(onClick = { app.settings.apiKey = key; app.settings.model = model; showSettings = false }) { Text("Save") }
                TextButton(onClick = { showSettings = false }) { Text("Cancel") }
            }
            return@Column
        }
        LazyColumn(Modifier.weight(1f).fillMaxWidth(), state = listState) {
            items(items) { item ->
                when (item) {
                    is ChatItem.User -> Text("› ${item.text}", color = MaterialTheme.colorScheme.primary, modifier = Modifier.padding(vertical = 3.dp))
                    is ChatItem.Assistant -> SelectionContainer { Text(item.text, modifier = Modifier.padding(vertical = 3.dp)) }
                    is ChatItem.ToolStep -> Text(
                        "${if (item.ok) "✓" else "✗"} ${item.name} ${item.input}\n   ${item.summary}",
                        fontFamily = FontFamily.Monospace, fontSize = 11.sp, color = if (item.ok) Color(0xFF8FBF8F) else Color(0xFFE08080), maxLines = 4,
                    )
                    is ChatItem.Notice -> Text(item.text, color = Color(0xFFE0B060), fontSize = 13.sp)
                }
            }
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(input, { input = it }, placeholder = { Text("Ask the agent to edit…") }, modifier = Modifier.weight(1f), maxLines = 3)
            Button(enabled = !running && input.isNotBlank(), onClick = { val t = input; input = ""; scope.launch { agent.send(t) } }) { Text(if (running) "…" else "Send") }
        }
        Row {
            TextButton(onClick = { showSettings = true }) { Text("Settings") }
            TextButton(onClick = agent::reset, enabled = !running) { Text("New thread") }
        }
    }
}

private const val MAX_CONSOLE_LINES = 1500

/** A console for `photocraft-cli`, run in-process in the workspace folder. */
@Composable
fun CliPanel(app: ImageLabApp, scope: CoroutineScope) {
    val lines = remember { mutableStateListOf("photocraft-cli in ${app.workspace.name}/ — try: commands --filter blur, info imports/<file>, --help") }
    var input by remember { mutableStateOf("") }
    val listState = rememberLazyListState()
    LaunchedEffect(lines.size) { listState.animateScrollToItem(lines.size - 1) }
    Column(Modifier.fillMaxSize().padding(8.dp)) {
        SelectionContainer(Modifier.weight(1f)) {
            LazyColumn(Modifier.fillMaxSize(), state = listState) {
                items(lines) { Text(it, fontFamily = FontFamily.Monospace, fontSize = 11.sp) }
            }
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            OutlinedTextField(input, { input = it }, prefix = { Text("$ ") }, singleLine = true, modifier = Modifier.weight(1f), textStyle = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace))
            Button(enabled = input.isNotBlank(), onClick = {
                val cmd = input
                input = ""
                lines += "$ $cmd"
                scope.launch {
                    val out = withContext(Dispatchers.IO) { app.lab.core.cli(cmd) }
                    // One row per line: a single huge Text can exceed the GPU's texture size and
                    // blank the whole window.
                    for (stream in listOf(out.stdout, out.stderr)) {
                        val all = stream.trimEnd().lines().filter { it.isNotEmpty() }
                        lines += all.take(MAX_CONSOLE_LINES).map { it.take(400) }
                        if (all.size > MAX_CONSOLE_LINES) lines += "… ${all.size - MAX_CONSOLE_LINES} more lines"
                    }
                    lines += "[exit ${out.code}]"
                    app.lab.refresh()
                }
            }) { Text("Run") }
        }
    }
}

/** PhotoCraft's MCP server over loopback HTTP, for agents on a laptop via `adb forward`. */
@Composable
fun McpPanel(app: ImageLabApp) {
    val settings = app.settings
    var running by remember { mutableStateOf(app.lab.core.mcpHttpPort() != null) }
    var token by remember { mutableStateOf(settings.mcpToken) }
    var error by remember { mutableStateOf<String?>(null) }
    val port = settings.mcpPort
    Column(Modifier.fillMaxSize().padding(10.dp).verticalScroll(rememberScrollState())) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("MCP server on 127.0.0.1:$port", Modifier.weight(1f))
            Switch(checked = running, onCheckedChange = { on ->
                error = null
                if (on) {
                    runCatching { app.lab.core.startMcpHttp(port.toUShort(), token) }
                        .onSuccess { running = true }
                        .onFailure { error = Lab.describe(it) }
                } else {
                    app.lab.core.stopMcpHttp()
                    running = false
                }
            })
        }
        error?.let { Text(it, color = Color(0xFFE08080)) }
        Text("Same tools as `photocraft-cli mcp`, on the document open here. From your laptop:", style = MaterialTheme.typography.bodySmall)
        SelectionContainer {
            Text(
                "adb forward tcp:$port tcp:$port\n" +
                    "claude mcp add --transport http imagelab http://localhost:$port/mcp \\\n  --header \"Authorization: Bearer $token\"",
                fontFamily = FontFamily.Monospace, fontSize = 11.sp, modifier = Modifier.padding(vertical = 6.dp),
            )
        }
        Text("CLI on the phone (separate session): adb shell ${app.cliPath} --help", fontFamily = FontFamily.Monospace, fontSize = 10.sp)
        TextButton(onClick = {
            token = settings.rotateMcpToken()
            if (running) runCatching { app.lab.core.startMcpHttp(port.toUShort(), token) }
        }) { Text("Rotate token") }
    }
}
