package dev.argo.imagelab.ui.menu

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.size
import androidx.compose.ui.Alignment
import androidx.compose.ui.graphics.drawscope.translate
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChange
import androidx.compose.ui.unit.dp
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonPrimitive
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.roundToInt

/** Space around the plot so the end points (0 and 255) are drawn whole and easy to grab. */
private val INSET = 14.dp

/** One curve: control points (input, output) in 0..255, sorted by input. */
typealias Curve = List<Pair<Int, Int>>

val LINEAR: Curve = listOf(0 to 0, 255 to 255)

/** The four curves of a Curves adjustment: `points` (RGB composite), `red`, `green`, `blue`. */
data class CurveSet(val curves: Map<String, Curve> = CHANNELS.associate { it.first to LINEAR }) {
    operator fun get(ch: String): Curve = curves[ch] ?: LINEAR
    fun with(ch: String, c: Curve) = CurveSet(curves + (ch to c))

    /** The params `layer.newAdjustmentLayer.curves` takes. */
    fun json(): Map<String, JsonElement> = CHANNELS.associate { (ch, _) ->
        ch to JsonArray(this[ch].map { (i, o) -> JsonArray(listOf(JsonPrimitive(i), JsonPrimitive(o))) })
    }

    companion object {
        val CHANNELS = listOf("points" to "RGB", "red" to "Red", "green" to "Green", "blue" to "Blue")
    }
}

/** The Curves editor; its value is a [CurveSet]. */
data class CurveParam(override val key: String = "curves", override val label: String = "Curve") : Param

/** Presets: whole sets (a look), applied at once. */
private val presets: List<Pair<String, CurveSet>> = listOf(
    "Linear" to CurveSet(),
    "Contrast" to CurveSet().with("points", listOf(0 to 0, 64 to 50, 192 to 206, 255 to 255)),
    "Strong" to CurveSet().with("points", listOf(0 to 0, 64 to 38, 192 to 220, 255 to 255)),
    "Matte" to CurveSet().with("points", listOf(0 to 34, 64 to 70, 192 to 200, 255 to 240)),
    "Brighten" to CurveSet().with("points", listOf(0 to 0, 128 to 160, 255 to 255)),
    "Darken" to CurveSet().with("points", listOf(0 to 0, 128 to 100, 255 to 255)),
    "Cross process" to CurveSet()
        .with("red", listOf(0 to 0, 64 to 46, 192 to 214, 255 to 255))
        .with("green", listOf(0 to 0, 64 to 54, 192 to 204, 255 to 255))
        .with("blue", listOf(0 to 30, 255 to 214)),
    "Warm fade" to CurveSet()
        .with("points", listOf(0 to 24, 128 to 132, 255 to 244))
        .with("red", listOf(0 to 8, 128 to 138, 255 to 255))
        .with("blue", listOf(0 to 0, 128 to 116, 255 to 236)),
    "Cool shadows" to CurveSet()
        .with("blue", listOf(0 to 32, 128 to 132, 255 to 250))
        .with("red", listOf(0 to 0, 128 to 124, 255 to 255)),
)

/** Natural cubic spline through [c], sampled at 0..255 and clamped (how the curve is drawn). */
fun sample(c: Curve): IntArray {
    val pts = c.sortedBy { it.first }
    val out = IntArray(256)
    if (pts.size < 2) { for (i in 0..255) out[i] = i; return out }
    val n = pts.size
    val x = DoubleArray(n) { pts[it].first.toDouble() }
    val y = DoubleArray(n) { pts[it].second.toDouble() }
    val m = DoubleArray(n) // second derivatives
    if (n > 2) {
        val a = DoubleArray(n); val b = DoubleArray(n); val cc = DoubleArray(n); val d = DoubleArray(n)
        for (i in 1 until n - 1) {
            a[i] = x[i] - x[i - 1]; b[i] = 2 * (x[i + 1] - x[i - 1]); cc[i] = x[i + 1] - x[i]
            d[i] = 6 * ((y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]))
        }
        for (i in 2 until n - 1) { val w = a[i] / b[i - 1]; b[i] -= w * cc[i - 1]; d[i] -= w * d[i - 1] }
        for (i in n - 2 downTo 1) m[i] = (d[i] - cc[i] * m[i + 1]) / b[i]
    }
    var k = 0
    for (v in 0..255) {
        val xv = v.toDouble()
        val yv = when {
            xv <= x[0] -> y[0]
            xv >= x[n - 1] -> y[n - 1]
            else -> {
                while (xv > x[k + 1]) k++
                val h = x[k + 1] - x[k]
                val t1 = x[k + 1] - xv; val t0 = xv - x[k]
                m[k] * t1 * t1 * t1 / (6 * h) + m[k + 1] * t0 * t0 * t0 / (6 * h) +
                    (y[k] / h - m[k] * h / 6) * t1 + (y[k + 1] / h - m[k + 1] * h / 6) * t0
            }
        }
        out[v] = yv.roundToInt().coerceIn(0, 255)
    }
    return out
}

/** 256-bin histograms (luma, red, green, blue) of a bitmap, sampled; null if it cannot be read. */
fun histograms(bmp: Bitmap?): Map<String, FloatArray>? = runCatching {
    val b = bmp ?: return null
    val step = maxOf(1, kotlin.math.sqrt(b.width.toDouble() * b.height / 80_000).toInt())
    val r = FloatArray(256); val g = FloatArray(256); val bl = FloatArray(256); val l = FloatArray(256)
    val row = IntArray(b.width)
    var y = 0
    while (y < b.height) {
        b.getPixels(row, 0, b.width, 0, y, b.width, 1)
        var x = 0
        while (x < b.width) {
            val p = row[x]
            val rr = (p shr 16) and 255; val gg = (p shr 8) and 255; val bb = p and 255
            r[rr]++; g[gg]++; bl[bb]++; l[(0.299 * rr + 0.587 * gg + 0.114 * bb).toInt()]++
            x += step
        }
        y += step
    }
    mapOf("points" to l, "red" to r, "green" to g, "blue" to bl)
}.getOrNull()

private fun channelColor(ch: String) = when (ch) {
    "red" -> Color(0xFFFF5A5A)
    "green" -> Color(0xFF5ADB6E)
    "blue" -> Color(0xFF5A9BFF)
    else -> Color.White
}

/**
 * A curve you drag: touch near a point to move it, elsewhere to add one on the curve and move
 * it; drag a middle point out of the box to remove it. [onChange] fires while dragging,
 * [onRelease] when the finger lifts, [onSet] for discrete changes (presets, reset).
 */
@Composable
fun CurveEditor(value: CurveSet, histogram: Map<String, FloatArray>?, onChange: (CurveSet) -> Unit, onRelease: () -> Unit, onSet: (CurveSet) -> Unit) {
    var channel by rememberSaveable { mutableStateOf("points") }
    val current by rememberUpdatedState(value)
    Column {
        Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            CurveSet.CHANNELS.forEach { (ch, label) ->
                FilterChip(selected = channel == ch, onClick = { channel = ch }, label = { Text(label, color = if (ch == "points") Color.Unspecified else channelColor(ch)) })
            }
            TextButton(onClick = { onSet(value.with(channel, LINEAR)) }) { Text("Reset channel") }
        }
        Canvas(
            Modifier.padding(vertical = 4.dp).size(240.dp).align(Alignment.CenterHorizontally)
                .background(Color(0xFF1A1A1C))
                .pointerInput(channel) {
                    val grab = 24.dp.toPx()
                    val pad = INSET.toPx()
                    awaitEachGesture {
                        val down = awaitFirstDown()
                        // The plot sits inset in the box so the 0 and 255 end points show whole.
                        val w = size.width - 2 * pad; val h = size.height - 2 * pad
                        fun toCurve(o: Offset) = ((o.x - pad) / w * 255f) to ((1f - (o.y - pad) / h) * 255f)
                        fun toScreen(p: Pair<Int, Int>) = Offset(pad + p.first / 255f * w, pad + (1f - p.second / 255f) * h)
                        var pts = current[channel].sortedBy { it.first }.toMutableList()
                        var idx = pts.indices.minByOrNull { (toScreen(pts[it]) - down.position).getDistance() } ?: return@awaitEachGesture
                        if ((toScreen(pts[idx]) - down.position).getDistance() > grab) {
                            // Not on a point: add one only when the touch is on the line itself;
                            // anywhere else the touch is left alone (the sheet scrolls).
                            if (pts.size >= 19) return@awaitEachGesture
                            val xin = toCurve(down.position).first.roundToInt().coerceIn(1, 254)
                            val onLine = abs(toScreen(xin to sample(pts)[xin]).y - down.position.y) <= grab
                            if (!onLine || pts.any { abs(it.first - xin) < 3 }) return@awaitEachGesture
                            pts.add(xin to sample(pts)[xin])
                            pts.sortBy { it.first }
                            idx = pts.indexOfFirst { it.first == xin }
                            onChange(current.with(channel, pts.toList()))
                        }
                        down.consume()
                        var pos = toScreen(pts[idx])
                        var removing = false
                        while (true) {
                            val ev = awaitPointerEvent()
                            val ch = ev.changes.firstOrNull() ?: break
                            if (!ch.pressed) break
                            pos += ch.positionChange()
                            ch.consume()
                            val middle = idx in 1 until pts.size - 1
                            removing = middle && (pos.y < -grab || pos.y > size.height + grab)
                            val (cx, cy) = toCurve(pos)
                            val lo = if (idx > 0) pts[idx - 1].first + 1 else 0
                            val hi = if (idx < pts.size - 1) pts[idx + 1].first - 1 else 255
                            pts[idx] = cx.roundToInt().coerceIn(lo, hi) to cy.roundToInt().coerceIn(0, 255)
                            val shown = if (removing) pts.filterIndexed { i, _ -> i != idx } else pts.toList()
                            onChange(current.with(channel, shown))
                        }
                        if (removing) pts = pts.filterIndexed { i, _ -> i != idx }.toMutableList()
                        onChange(current.with(channel, pts.toList()))
                        onRelease()
                    }
                },
        ) {
            val pad = INSET.toPx()
            val w = size.width - 2 * pad; val h = size.height - 2 * pad
            translate(pad, pad) {
            // Histogram of the photo as it was when Curves opened (log-scaled so it reads).
            histogram?.get(channel)?.let { bins ->
                val peak = ln(1f + (bins.maxOrNull() ?: 1f))
                val path = Path().apply {
                    moveTo(0f, h)
                    bins.forEachIndexed { i, v -> lineTo(i / 255f * w, h - ln(1f + v) / peak * h * 0.9f) }
                    lineTo(w, h); close()
                }
                drawPath(path, channelColor(channel).copy(alpha = 0.16f))
            }
            val grid = Color(0x33FFFFFF)
            for (q in 1..3) {
                drawLine(grid, Offset(w * q / 4, 0f), Offset(w * q / 4, h))
                drawLine(grid, Offset(0f, h * q / 4), Offset(w, h * q / 4))
            }
            drawLine(Color(0x22FFFFFF), Offset(0f, h), Offset(w, 0f))
            fun curvePath(c: Curve): Path {
                val s = sample(c)
                return Path().apply { s.forEachIndexed { i, v -> val o = Offset(i / 255f * w, (1f - v / 255f) * h); if (i == 0) moveTo(o.x, o.y) else lineTo(o.x, o.y) } }
            }
            // The other channels, faint, when they are not straight.
            CurveSet.CHANNELS.forEach { (ch, _) ->
                if (ch != channel && value[ch] != LINEAR) drawPath(curvePath(value[ch]), channelColor(ch).copy(alpha = 0.35f), style = Stroke(1.5.dp.toPx()))
            }
            val col = channelColor(channel)
            drawPath(curvePath(value[channel]), col, style = Stroke(2.5.dp.toPx()))
            value[channel].forEach { (i, o) ->
                val c = Offset(i / 255f * w, (1f - o / 255f) * h)
                drawCircle(Color.Black, 8.dp.toPx(), c)
                drawCircle(col, 6.dp.toPx(), c)
            }
            drawRect(Color(0x44FFFFFF), Offset.Zero, androidx.compose.ui.geometry.Size(w, h), style = Stroke(1.dp.toPx()))
            }
        }
        Text("Drag a point (the ends too); touch the line to add one; drag a point out of the box to remove it.", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
        Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            presets.forEach { (label, set) -> FilterChip(selected = value == set, onClick = { onSet(set) }, label = { Text(label) }) }
        }
    }
}
