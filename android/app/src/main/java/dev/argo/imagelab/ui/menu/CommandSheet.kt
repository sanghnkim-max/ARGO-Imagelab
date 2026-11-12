package dev.argo.imagelab.ui.menu

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import kotlin.math.roundToInt

/** Swatches for colour params (an original small palette). */
private val swatches = listOf("#000000", "#ffffff", "#3d7bf5", "#e53935", "#fb8c00", "#fdd835", "#43a047", "#8e24aa", "#ffe08a", "#6d4c41")

/**
 * Any set of [MenuGroup]s as a sheet: group chips, item chips, then the open item's controls
 * with Apply / Cancel. The photo follows the controls live ([LiveCommand]).
 */
@Composable
fun CommandSheet(groups: List<MenuGroup>, live: LiveCommand, note: String? = null) {
    var groupIndex by rememberSaveable { mutableStateOf(0) }
    val group = groups.getOrElse(groupIndex) { groups.first() }
    if (note != null) Text(note, style = MaterialTheme.typography.bodySmall, color = Color.Gray)
    if (groups.size > 1) {
        Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            groups.forEachIndexed { i, g -> FilterChip(selected = i == groupIndex, onClick = { groupIndex = i }, label = { Text(g.label) }) }
        }
    }
    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        group.items.forEach { item ->
            FilterChip(selected = live.item == item, onClick = { if (live.item != item) live.open(item) }, label = { Text(item.label) })
        }
    }
    val item = live.item
    if (item != null && groups.any { it.items.contains(item) }) {
        if (!group.items.contains(item)) Text(item.label, style = MaterialTheme.typography.labelLarge, modifier = Modifier.padding(top = 4.dp))
        Column(Modifier.padding(top = 4.dp)) {
            item.params.forEach { p -> ParamControl(p, live) }
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = live::apply) { Text(if (item.live) "Apply" else "Apply ${item.label}") }
                TextButton(onClick = live::cancel) { Text("Cancel") }
                if (live.busy) CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
                if (!item.live) Text("Applies when you tap Apply.", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
            }
        }
    }
}

@Composable
private fun ParamControl(p: Param, live: LiveCommand) {
    when (p) {
        is Slider -> {
            p.showWhen?.let { (k, want) -> if (live.values[k] != want) return }
            val v = (live.values[p.key] as? Float) ?: p.default
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(p.label, Modifier.width(96.dp), style = MaterialTheme.typography.bodySmall)
                Slider(value = v, onValueChange = { live.drag(p.key, it) }, onValueChangeFinished = live::release, valueRange = p.min..p.max, modifier = Modifier.weight(1f))
                Text(if (p.integer) v.roundToInt().toString() else "%.2f".format(v), Modifier.width(52.dp), fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall)
            }
        }
        is Choice -> {
            val v = (live.values[p.key] as? String) ?: p.default
            Text(p.label, style = MaterialTheme.typography.bodySmall)
            Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                p.options.forEach { (value, label) -> FilterChip(selected = v == value, onClick = { live.set(p.key, value) }, label = { Text(label) }) }
            }
        }
        is Toggle -> {
            val v = (live.values[p.key] as? Boolean) ?: p.default
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(p.label, Modifier.weight(1f), style = MaterialTheme.typography.bodySmall)
                Switch(checked = v, onCheckedChange = { live.set(p.key, it) })
            }
        }
        is CurveParam -> {
            val v = (live.values[p.key] as? CurveSet) ?: CurveSet()
            // The histogram of the photo as Curves opened, not of each preview.
            val hist = androidx.compose.runtime.remember(live.item) { histograms(live.previewBitmap()) }
            CurveEditor(v, hist, onChange = { live.drag(p.key, it) }, onRelease = live::release, onSet = { live.set(p.key, it) })
        }
        is ColorParam -> {
            val v = (live.values[p.key] as? String) ?: p.default
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.padding(vertical = 4.dp)) {
                Text(p.label, Modifier.width(96.dp), style = MaterialTheme.typography.bodySmall)
                Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    swatches.forEach { hex ->
                        val c = Color(android.graphics.Color.parseColor(hex))
                        Box(
                            Modifier.size(28.dp).background(c, CircleShape)
                                .border(if (hex.equals(v, ignoreCase = true)) 3.dp else 1.dp, if (hex.equals(v, ignoreCase = true)) MaterialTheme.colorScheme.primary else Color.Gray, CircleShape)
                                .clickable { live.set(p.key, hex) },
                        )
                    }
                }
            }
        }
    }
}
