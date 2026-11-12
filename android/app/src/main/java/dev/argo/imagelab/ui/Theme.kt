package dev.argo.imagelab.ui

import androidx.annotation.DrawableRes
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import dev.argo.imagelab.R

/** A dark editing theme: photos read best on a neutral, near-black surround. */
@Composable
fun ImageLabTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = Accent,
            onPrimary = Color.White,
            secondary = Color(0xFF8AB4F8),
            background = Color(0xFF121212),
            surface = Color(0xFF1C1C1E),
            surfaceVariant = Color(0xFF2A2A2D),
            surfaceContainer = Color(0xFF1C1C1E),
            surfaceContainerHigh = Color(0xFF242427),
            error = Color(0xFFE5807A),
        ),
        content = content,
    )
}

val Accent = Color(0xFF3D7BF5)
val CanvasBackground = Color(0xFF0B0B0B)
val OkGreen = Color(0xFF8FBF8F)
val WarnAmber = Color(0xFFE0B060)

/** Material Symbols Outlined glyphs (android/scripts/material-symbols.sh). */
object Sym {
    @DrawableRes val home = R.drawable.ic_sym_home
    @DrawableRes val undo = R.drawable.ic_sym_undo
    @DrawableRes val redo = R.drawable.ic_sym_redo
    @DrawableRes val export = R.drawable.ic_sym_ios_share
    @DrawableRes val agent = R.drawable.ic_sym_auto_awesome
    @DrawableRes val settings = R.drawable.ic_sym_settings
    @DrawableRes val layers = R.drawable.ic_sym_layers
    @DrawableRes val add = R.drawable.ic_sym_add
    @DrawableRes val select = R.drawable.ic_sym_select
    @DrawableRes val adjust = R.drawable.ic_sym_tune
    @DrawableRes val retouch = R.drawable.ic_sym_healing
    @DrawableRes val crop = R.drawable.ic_sym_crop
    @DrawableRes val brush = R.drawable.ic_sym_brush
    @DrawableRes val filters = R.drawable.ic_sym_filter_vintage
    @DrawableRes val effects = R.drawable.ic_sym_auto_fix_high
    @DrawableRes val resize = R.drawable.ic_sym_photo_size_select_large
    @DrawableRes val up = R.drawable.ic_sym_arrow_upward
    @DrawableRes val down = R.drawable.ic_sym_arrow_downward
    @DrawableRes val copy = R.drawable.ic_sym_content_copy
    @DrawableRes val merge = R.drawable.ic_sym_merge
    @DrawableRes val palette = R.drawable.ic_sym_palette
    @DrawableRes val close = R.drawable.ic_sym_close
    @DrawableRes val send = R.drawable.ic_sym_send
    @DrawableRes val star = R.drawable.ic_sym_star
    @DrawableRes val starFill = R.drawable.ic_sym_star_fill
    @DrawableRes val flag = R.drawable.ic_sym_flag
    @DrawableRes val flagFill = R.drawable.ic_sym_flag_fill
    @DrawableRes val checkCircleFill = R.drawable.ic_sym_check_circle_fill
    @DrawableRes val library = R.drawable.ic_sym_photo_library
    @DrawableRes val text = R.drawable.ic_sym_text_fields
    @DrawableRes val fill = R.drawable.ic_sym_format_color_fill
    @DrawableRes val gradient = R.drawable.ic_sym_gradient
    @DrawableRes val generate = R.drawable.ic_sym_auto_fix_high
    @DrawableRes val chevronLeft = R.drawable.ic_sym_chevron_left
    @DrawableRes val chevronRight = R.drawable.ic_sym_chevron_right
    @DrawableRes val expandMore = R.drawable.ic_sym_expand_more
    @DrawableRes val expandLess = R.drawable.ic_sym_expand_less
    @DrawableRes val code = R.drawable.ic_sym_code
    @DrawableRes val image = R.drawable.ic_sym_image
    @DrawableRes val terminal = R.drawable.ic_sym_terminal
    @DrawableRes val visible = R.drawable.ic_sym_visibility
    @DrawableRes val hidden = R.drawable.ic_sym_visibility_off
    @DrawableRes val delete = R.drawable.ic_sym_delete
    @DrawableRes val addPhoto = R.drawable.ic_sym_add_photo_alternate
    @DrawableRes val straighten = R.drawable.ic_sym_straighten
    @DrawableRes val blur = R.drawable.ic_sym_blur_on
    @DrawableRes val cut = R.drawable.ic_sym_content_cut
    @DrawableRes val erase = R.drawable.ic_sym_ink_eraser
    @DrawableRes val stop = R.drawable.ic_sym_stop
    @DrawableRes val download = R.drawable.ic_sym_download
    @DrawableRes val person = R.drawable.ic_sym_person
    @DrawableRes val landscape = R.drawable.ic_sym_landscape
    @DrawableRes val check = R.drawable.ic_sym_check
    @DrawableRes val grid = R.drawable.ic_sym_grid_view
    @DrawableRes val enhance = R.drawable.ic_sym_magic_button
    @DrawableRes val face = R.drawable.ic_sym_face_retouching_natural
    @DrawableRes val history = R.drawable.ic_sym_history
    @DrawableRes val compare = R.drawable.ic_sym_compare
    @DrawableRes val transform = R.drawable.ic_sym_transform
}

@Composable
fun SymIcon(@DrawableRes res: Int, contentDescription: String?, modifier: Modifier = Modifier, tint: Color? = null) {
    if (tint != null) Icon(painterResource(res), contentDescription, modifier, tint = tint)
    else Icon(painterResource(res), contentDescription, modifier)
}

/** An icon button; [label] is the accessibility label (and the tooltip-free name). */
@Composable
fun SymButton(@DrawableRes res: Int, label: String, onClick: () -> Unit, enabled: Boolean = true, tint: Color? = null) {
    IconButton(onClick = onClick, enabled = enabled) { SymIcon(res, label, tint = tint) }
}
