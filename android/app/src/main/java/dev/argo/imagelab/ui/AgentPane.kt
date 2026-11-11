package dev.argo.imagelab.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.AssistChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.InputChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.argo.imagelab.ImageLabApp
import dev.argo.imagelab.agent.Agent
import dev.argo.imagelab.agent.ChatItem
import kotlinx.coroutines.launch

/**
 * The agent (docs/INTERFACES.md §10): each program it runs is a collapsed "ran a program" row
 * that expands to the code and its output; previews it returned show inline; photos it touched
 * show as chips. Runs in the app scope, so closing the pane or rotating does not stop it.
 */
@Composable
fun AgentPane(app: ImageLabApp, modifier: Modifier, onClose: () -> Unit, onOpenSettings: () -> Unit) {
    val agent = app.agent
    val items by agent.items.collectAsState()
    val running by agent.running.collectAsState()
    val context by agent.context.collectAsState()
    var input by rememberSaveable { mutableStateOf("") }
    val hasKey = remember { app.settings.hasApiKey }
    val listState = rememberLazyListState()
    val uiScope = rememberCoroutineScope()
    LaunchedEffect(items.size) { if (items.isNotEmpty()) listState.animateScrollToItem(items.size - 1) }

    Surface(modifier, color = MaterialTheme.colorScheme.surface) {
        Column(Modifier.imePadding()) {
            Row(Modifier.fillMaxWidth().padding(start = 12.dp, end = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                SymIcon(Sym.agent, null, Modifier.size(20.dp), tint = Accent)
                Text("Agent", style = MaterialTheme.typography.titleMedium, modifier = Modifier.weight(1f).padding(start = 8.dp))
                TextButton(onClick = agent::reset, enabled = !running) { Text("New") }
                SymButton(Sym.close, "Close agent", onClose)
            }
            HorizontalDivider()
            if (!hasKey) {
                Row(Modifier.fillMaxWidth().background(Color(0x332A2A2D)).padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text("Add your Anthropic API key to use the agent.", Modifier.weight(1f), style = MaterialTheme.typography.bodySmall)
                    TextButton(onClick = onOpenSettings) { Text("Settings") }
                }
            }
            LazyColumn(Modifier.weight(1f).fillMaxWidth().padding(horizontal = 12.dp), state = listState, verticalArrangement = Arrangement.spacedBy(6.dp)) {
                if (items.isEmpty()) {
                    item {
                        Text(
                            "Ask for edits (“blur the background”), culls (“pick the best 10 of yesterday”) or batch work. " +
                                "The agent writes small programs against ImageLab and shows each one here.",
                            color = Color.Gray, style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(vertical = 12.dp),
                        )
                    }
                }
                itemsIndexed(items) { _, item -> ChatRow(item) }
            }
            if (context.isNotEmpty()) {
                Row(Modifier.fillMaxWidth().padding(horizontal = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                    InputChip(
                        selected = true,
                        onClick = agent::clearContext,
                        label = { Text("${context.size} photo${if (context.size == 1) "" else "s"} attached") },
                        trailingIcon = { SymIcon(Sym.close, "Remove attached photos", Modifier.size(16.dp)) },
                    )
                }
            }
            Row(Modifier.fillMaxWidth().padding(8.dp), verticalAlignment = Alignment.CenterVertically) {
                OutlinedTextField(input, { input = it }, placeholder = { Text("Ask the agent…") }, modifier = Modifier.weight(1f), maxLines = 4)
                if (running) {
                    SymButton(Sym.stop, "Stop the agent", { uiScope.launch { agent.stop() } })
                } else {
                    SymButton(Sym.send, "Send", {
                        val t = input
                        input = ""
                        app.appScope.launch { agent.send(t) }
                    }, enabled = input.isNotBlank())
                }
            }
        }
    }
}

@Composable
private fun ChatRow(item: ChatItem) {
    when (item) {
        is ChatItem.User -> Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.End) {
            Surface(color = Accent.copy(alpha = 0.25f), shape = RoundedCornerShape(12.dp)) {
                Text(item.text, Modifier.padding(horizontal = 12.dp, vertical = 8.dp))
            }
            if (item.contextPhotos.isNotEmpty()) PhotoChips(item.contextPhotos)
        }
        is ChatItem.Assistant -> SelectionContainer { Text(item.text, style = MaterialTheme.typography.bodyMedium) }
        is ChatItem.ToolStep -> ToolStepRow(item)
        is ChatItem.Notice -> Text(item.text, color = WarnAmber, fontSize = 13.sp)
    }
}

@Composable
private fun ToolStepRow(step: ChatItem.ToolStep) {
    var expanded by rememberSaveable { mutableStateOf(false) }
    val verb = when (step.tool) {
        Agent.READ_TOOL -> "Looked"
        Agent.CLI_TOOL -> "Ran CLI"
        else -> "Ran a program"
    }
    Surface(color = MaterialTheme.colorScheme.surfaceVariant, shape = RoundedCornerShape(8.dp), modifier = Modifier.fillMaxWidth()) {
        Column {
            Row(
                Modifier.fillMaxWidth().clickable { expanded = !expanded }.padding(horizontal = 10.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                SymIcon(if (step.tool == Agent.CLI_TOOL) Sym.terminal else Sym.code, null, Modifier.size(18.dp), tint = if (step.ok) OkGreen else MaterialTheme.colorScheme.error)
                Column(Modifier.weight(1f).padding(horizontal = 8.dp)) {
                    Text(verb, style = MaterialTheme.typography.labelSmall, color = Color.Gray)
                    Text(step.description, style = MaterialTheme.typography.bodySmall, maxLines = 2, overflow = TextOverflow.Ellipsis)
                }
                SymIcon(if (expanded) Sym.expandLess else Sym.expandMore, if (expanded) "Hide program" else "Show program", Modifier.size(20.dp))
            }
            AnimatedVisibility(expanded) {
                SelectionContainer {
                    Column(Modifier.padding(horizontal = 10.dp).padding(bottom = 8.dp)) {
                        Text(
                            step.code, fontFamily = FontFamily.Monospace, fontSize = 11.sp,
                            modifier = Modifier.fillMaxWidth().background(Color(0xFF151517), RoundedCornerShape(4.dp)).padding(8.dp).heightIn(max = 320.dp),
                        )
                        if (step.output.isNotBlank()) {
                            Text("Result", style = MaterialTheme.typography.labelSmall, color = Color.Gray, modifier = Modifier.padding(top = 6.dp))
                            Text(step.output, fontFamily = FontFamily.Monospace, fontSize = 11.sp, maxLines = 40, overflow = TextOverflow.Ellipsis,
                                color = if (step.ok) Color.Unspecified else MaterialTheme.colorScheme.error)
                        }
                    }
                }
            }
            if (step.images.isNotEmpty()) {
                Row(Modifier.horizontalScroll(rememberScrollState()).padding(horizontal = 10.dp).padding(bottom = 8.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    step.images.forEach { b ->
                        Image(b.asImageBitmap(), contentDescription = "Preview the agent looked at", contentScale = ContentScale.Fit,
                            modifier = Modifier.height(160.dp).clip(RoundedCornerShape(6.dp)))
                    }
                }
            }
            if (step.photos.isNotEmpty()) Row(Modifier.padding(horizontal = 10.dp).padding(bottom = 6.dp)) { PhotoChips(step.photos) }
        }
    }
}

@Composable
private fun PhotoChips(ids: List<String>) {
    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
        ids.take(12).forEach { id ->
            AssistChip(onClick = {}, label = { Text(id.substringAfterLast('/'), maxLines = 1, fontSize = 11.sp) }, leadingIcon = { SymIcon(Sym.image, null, Modifier.size(14.dp)) })
        }
        if (ids.size > 12) Text("+${ids.size - 12}", Modifier.align(Alignment.CenterVertically), color = Color.Gray, fontSize = 11.sp)
    }
}
