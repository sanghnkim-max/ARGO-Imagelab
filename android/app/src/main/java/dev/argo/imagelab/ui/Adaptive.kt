package dev.argo.imagelab.ui

import android.graphics.Rect
import androidx.compose.runtime.Composable
import androidx.compose.runtime.Immutable
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalWindowInfo
import androidx.window.core.layout.WindowSizeClass
import androidx.window.core.layout.computeWindowSizeClass
import androidx.window.layout.FoldingFeature
import androidx.window.layout.WindowLayoutInfo

/** Material width classes (docs/INTERFACES.md §10), from Jetpack WindowManager's breakpoints. */
enum class WidthClass { Compact, Medium, Expanded }

/** How a foldable is held, from its [FoldingFeature]. */
sealed interface Posture {
    data object Flat : Posture

    /** Half open, horizontal hinge: canvas on top, tools + agent below. [hinge] in window px. */
    data class Tabletop(val hinge: Rect) : Posture

    /** Vertical hinge (separating or half open): canvas on one side, pane on the other. */
    data class Book(val hinge: Rect) : Posture
}

@Immutable
data class WindowLayout(val width: WidthClass, val posture: Posture, val widthDp: Float, val heightDp: Float) {
    val compact get() = width == WidthClass.Compact
    val expanded get() = width == WidthClass.Expanded
}

fun widthClassOf(widthDp: Float, heightDp: Float): WidthClass {
    val size = WindowSizeClass.BREAKPOINTS_V1.computeWindowSizeClass(widthDp, heightDp)
    return when {
        size.isWidthAtLeastBreakpoint(WindowSizeClass.WIDTH_DP_EXPANDED_LOWER_BOUND) -> WidthClass.Expanded
        size.isWidthAtLeastBreakpoint(WindowSizeClass.WIDTH_DP_MEDIUM_LOWER_BOUND) -> WidthClass.Medium
        else -> WidthClass.Compact
    }
}

fun postureOf(info: WindowLayoutInfo?): Posture {
    val fold = info?.displayFeatures?.filterIsInstance<FoldingFeature>()?.firstOrNull() ?: return Posture.Flat
    return when {
        fold.state == FoldingFeature.State.HALF_OPENED && fold.orientation == FoldingFeature.Orientation.HORIZONTAL ->
            Posture.Tabletop(fold.bounds)
        fold.orientation == FoldingFeature.Orientation.VERTICAL && (fold.isSeparating || fold.state == FoldingFeature.State.HALF_OPENED) ->
            Posture.Book(fold.bounds)
        else -> Posture.Flat
    }
}

/** This window's layout now: the window's own size (split screen and folding seen as they are). */
@Composable
fun currentWindowLayout(info: WindowLayoutInfo?): WindowLayout {
    val size = LocalWindowInfo.current.containerSize
    val density = LocalDensity.current
    val (w, h) = if (size.width > 0 && size.height > 0) {
        with(density) { size.width.toDp().value to size.height.toDp().value }
    } else {
        val c = LocalConfiguration.current
        c.screenWidthDp.toFloat() to c.screenHeightDp.toFloat()
    }
    return WindowLayout(widthClassOf(w, h), postureOf(info), w, h)
}
