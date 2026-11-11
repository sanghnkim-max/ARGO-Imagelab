package dev.argo.imagelab.ui

import android.graphics.Bitmap
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.VerticalDivider
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.argo.imagelab.ImageLabApp

/**
 * Home: the photo library (docs/INTERFACES.md §10) — the entry point for cull and batch jobs.
 * Grid of `library.query` results with rating/flag badges; long-press starts multi-select;
 * actions rate, flag, "Ask agent about these" and batch export; a progress card follows batch jobs.
 * Expanded windows show a list/detail pair.
 */
@Composable
fun LibraryScreen(
    app: ImageLabApp,
    layout: WindowLayout,
    agentPane: (@Composable (Modifier) -> Unit)?,
    onOpenEditor: () -> Unit,
    onAskAgent: (List<String>) -> Unit,
    onToggleAgent: () -> Unit,
    onSettings: () -> Unit,
    onEditor: () -> Unit,
) {
    val vm: LibraryViewModel = viewModel()
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia(50)) { uris -> vm.import(uris) }
    val labState by app.lab.state.collectAsState()
    // Re-read on entering the screen (the grid also follows libraryChanged events).
    LaunchedEffect(Unit) { vm.load() }

    Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        Row(Modifier.fillMaxSize().safeDrawingPadding()) {
            Column(Modifier.weight(1f).fillMaxHeight()) {
                LibraryTopBar(
                    vm = vm,
                    hasDocument = labState.documents > 0,
                    onImport = { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                    onAskAgent = { onAskAgent(vm.selected.toList()) },
                    onToggleAgent = onToggleAgent,
                    onSettings = onSettings,
                    onEditor = onEditor,
                )
                vm.job?.let { JobProgressCard(it, onDismiss = vm::dismissJob) }
                Row(Modifier.weight(1f).fillMaxWidth()) {
                    PhotoGrid(
                        vm = vm,
                        twoPane = layout.expanded,
                        modifier = Modifier.weight(1f).fillMaxHeight(),
                        onImport = { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                        onOpen = { vm.open(it, onOpenEditor) },
                    )
                    if (layout.expanded && agentPane == null) {
                        VerticalDivider()
                        PhotoDetail(vm, Modifier.width(360.dp).fillMaxHeight(), onOpen = { vm.open(it, onOpenEditor) }, onAskAgent = { onAskAgent(listOf(it.id)) })
                    }
                }
            }
            if (agentPane != null) {
                VerticalDivider()
                agentPane(Modifier.width(if (layout.expanded) 400.dp else 340.dp).fillMaxHeight())
            }
        }
    }
}

@Composable
private fun LibraryTopBar(
    vm: LibraryViewModel,
    hasDocument: Boolean,
    onImport: () -> Unit,
    onAskAgent: () -> Unit,
    onToggleAgent: () -> Unit,
    onSettings: () -> Unit,
    onEditor: () -> Unit,
) {
    if (!vm.selecting) {
        Row(Modifier.fillMaxWidth().padding(start = 16.dp, end = 4.dp, top = 4.dp, bottom = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            Text("Library", style = MaterialTheme.typography.titleLarge, modifier = Modifier.weight(1f))
            if (hasDocument) SymButton(Sym.image, "Back to the open document", onEditor)
            SymButton(Sym.addPhoto, "Import photos", onImport)
            SymButton(Sym.agent, "Agent", onToggleAgent)
            SymButton(Sym.settings, "Settings", onSettings)
        }
        return
    }
    var rateMenu by remember { mutableStateOf(false) }
    var flagMenu by remember { mutableStateOf(false) }
    Column(Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.surfaceContainerHigh)) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            SymButton(Sym.close, "Clear selection", vm::clearSelection)
            Text("${vm.selected.size} selected", style = MaterialTheme.typography.titleMedium, modifier = Modifier.weight(1f))
            TextButton(onClick = vm::selectAll) { Text("All") }
        }
        Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(horizontal = 8.dp, vertical = 4.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Box {
                AssistChip(onClick = { rateMenu = true }, label = { Text("Rate") }, leadingIcon = { SymIcon(Sym.star, null, Modifier.size(18.dp)) })
                DropdownMenu(rateMenu, onDismissRequest = { rateMenu = false }) {
                    (5 downTo 0).forEach { r ->
                        DropdownMenuItem(text = { Text(if (r == 0) "No rating" else "★".repeat(r)) }, onClick = { rateMenu = false; vm.rate(r) })
                    }
                }
            }
            Box {
                AssistChip(onClick = { flagMenu = true }, label = { Text("Flag") }, leadingIcon = { SymIcon(Sym.flag, null, Modifier.size(18.dp)) })
                DropdownMenu(flagMenu, onDismissRequest = { flagMenu = false }) {
                    DropdownMenuItem(text = { Text("Pick") }, onClick = { flagMenu = false; vm.flag("pick") })
                    DropdownMenuItem(text = { Text("Reject") }, onClick = { flagMenu = false; vm.flag("reject") })
                    DropdownMenuItem(text = { Text("Clear flag") }, onClick = { flagMenu = false; vm.flag("none") })
                }
            }
            AssistChip(onClick = onAskAgent, label = { Text("Ask agent about these") }, leadingIcon = { SymIcon(Sym.agent, null, Modifier.size(18.dp)) })
            AssistChip(onClick = vm::batchExport, label = { Text("Export") }, leadingIcon = { SymIcon(Sym.download, null, Modifier.size(18.dp)) })
        }
    }
}

@Composable
private fun JobProgressCard(job: JobCard, onDismiss: () -> Unit) {
    Card(Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 6.dp)) {
        Column(Modifier.padding(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    when {
                        !job.finished -> "Batch ${job.job}: ${job.done}/${job.total}"
                        job.ok -> "Batch ${job.job} finished"
                        else -> "Batch ${job.job} failed"
                    },
                    style = MaterialTheme.typography.titleSmall,
                    modifier = Modifier.weight(1f),
                )
                if (job.finished) SymButton(Sym.close, "Dismiss", onDismiss)
            }
            if (!job.finished) {
                if (job.total > 0) LinearProgressIndicator(progress = { job.done.toFloat() / job.total }, modifier = Modifier.fillMaxWidth())
                else LinearProgressIndicator(Modifier.fillMaxWidth())
            }
            if (job.message.isNotBlank()) Text(job.message, style = MaterialTheme.typography.bodySmall, maxLines = 2, overflow = TextOverflow.Ellipsis)
        }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun PhotoGrid(vm: LibraryViewModel, twoPane: Boolean, modifier: Modifier, onImport: () -> Unit, onOpen: (Photo) -> Unit) {
    if (vm.photos.isEmpty()) {
        Column(modifier.padding(24.dp), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
            SymIcon(Sym.library, null, Modifier.size(48.dp), tint = Color.Gray)
            Spacer(Modifier.size(12.dp))
            Text(if (vm.loading) "Loading…" else "No photos yet.", style = MaterialTheme.typography.titleMedium)
            vm.unavailable?.let { Text("Library: $it", style = MaterialTheme.typography.bodySmall, color = Color.Gray, modifier = Modifier.padding(top = 4.dp)) }
            Spacer(Modifier.size(16.dp))
            Button(onClick = onImport) { Text("Import photos") }
        }
        return
    }
    LazyVerticalGrid(columns = GridCells.Adaptive(112.dp), modifier = modifier, contentPadding = androidx.compose.foundation.layout.PaddingValues(4.dp)) {
        items(vm.photos, key = { it.id }) { photo ->
            val isSelected = photo.id in vm.selected
            Box(
                Modifier
                    .padding(2.dp)
                    .aspectRatio(1f)
                    .clip(RoundedCornerShape(4.dp))
                    .background(MaterialTheme.colorScheme.surfaceVariant)
                    .then(if (isSelected || (twoPane && vm.focused == photo.id)) Modifier.border(2.dp, Accent, RoundedCornerShape(4.dp)) else Modifier)
                    .semantics { contentDescription = "${photo.name}, ${photo.rating} stars${if (photo.flag != "none") ", ${photo.flag}" else ""}${if (isSelected) ", selected" else ""}" }
                    .combinedClickable(
                        onClick = {
                            when {
                                vm.selecting -> vm.toggle(photo.id)
                                twoPane -> vm.focused = photo.id
                                else -> onOpen(photo)
                            }
                        },
                        onDoubleClick = { if (!vm.selecting) onOpen(photo) },
                        onLongClick = { vm.toggle(photo.id) },
                    ),
            ) {
                Thumbnail(vm, photo, Modifier.fillMaxSize())
                if (photo.rating > 0) {
                    Row(
                        Modifier.align(Alignment.BottomStart).padding(4.dp).background(Color(0x99000000), RoundedCornerShape(4.dp)).padding(horizontal = 4.dp, vertical = 1.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        SymIcon(Sym.starFill, null, Modifier.size(12.dp), tint = WarnAmber)
                        Text("${photo.rating}", fontSize = 11.sp, color = Color.White)
                    }
                }
                when (photo.flag) {
                    "pick" -> SymIcon(Sym.flagFill, null, Modifier.align(Alignment.TopEnd).padding(4.dp).size(18.dp), tint = OkGreen)
                    "reject" -> SymIcon(Sym.close, null, Modifier.align(Alignment.TopEnd).padding(4.dp).size(18.dp), tint = MaterialTheme.colorScheme.error)
                }
                if (isSelected) SymIcon(Sym.checkCircleFill, null, Modifier.align(Alignment.TopStart).padding(4.dp).size(20.dp), tint = Accent)
            }
        }
    }
}

@Composable
private fun Thumbnail(vm: LibraryViewModel, photo: Photo, modifier: Modifier) {
    val bitmap by produceState<Bitmap?>(null, photo.path) { value = vm.thumbnail(photo) }
    val b = bitmap
    if (b != null) {
        Image(b.asImageBitmap(), contentDescription = null, contentScale = ContentScale.Crop, modifier = modifier)
    } else {
        Box(modifier, contentAlignment = Alignment.Center) {
            Text(photo.name, fontSize = 10.sp, color = Color.Gray, maxLines = 2, overflow = TextOverflow.Ellipsis, modifier = Modifier.padding(4.dp))
        }
    }
}

@Composable
private fun PhotoDetail(vm: LibraryViewModel, modifier: Modifier, onOpen: (Photo) -> Unit, onAskAgent: (Photo) -> Unit) {
    val photo = vm.photos.firstOrNull { it.id == vm.focused }
    Column(modifier.padding(12.dp)) {
        if (photo == null) {
            Text("Select a photo to see it here.", color = Color.Gray)
            return@Column
        }
        Thumbnail(vm, photo, Modifier.fillMaxWidth().aspectRatio(1f).clip(RoundedCornerShape(6.dp)))
        Spacer(Modifier.size(8.dp))
        Text(photo.name, style = MaterialTheme.typography.titleMedium, maxLines = 2, overflow = TextOverflow.Ellipsis)
        Text(photo.path ?: photo.id, style = MaterialTheme.typography.bodySmall, color = Color.Gray)
        Row(verticalAlignment = Alignment.CenterVertically) {
            (1..5).forEach { r ->
                SymButton(if (r <= photo.rating) Sym.starFill else Sym.star, "Rate $r", onClick = { vm.focused = photo.id; vm.rate(r) }, tint = if (r <= photo.rating) WarnAmber else null)
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { onOpen(photo) }) { Text("Edit") }
            OutlinedButton(onClick = { onAskAgent(photo) }) { Text("Ask agent") }
        }
    }
}
