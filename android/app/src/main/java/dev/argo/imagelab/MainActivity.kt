package dev.argo.imagelab

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.ui.graphics.Color
import androidx.lifecycle.lifecycleScope
import dev.argo.imagelab.ui.EditorScreen
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    private val app get() = application as ImageLabApp

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MaterialTheme(colorScheme = darkColorScheme(primary = Color(0xFF3D7BF5), background = Color(0xFF121212), surface = Color(0xFF1C1C1E))) {
                EditorScreen(app = app, onPick = ::importUri)
            }
        }
        handleIntent(intent)
        lifecycleScope.launch { app.lab.refresh() }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIntent(intent)
    }

    /** "Edit with ImageLab" (ACTION_EDIT) and the share sheet (ACTION_SEND). */
    private fun handleIntent(intent: Intent?) {
        val uri: Uri? = when (intent?.action) {
            Intent.ACTION_EDIT -> intent.data
            Intent.ACTION_SEND -> intent.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
            else -> null
        }
        uri?.let(::importUri)
    }

    /** Copy the picked image into the workspace's imports/ and open it. */
    fun importUri(uri: Uri) {
        lifecycleScope.launch {
            val name = displayName(uri) ?: "photo.jpg"
            val bytes = runCatching { contentResolver.openInputStream(uri)?.use { it.readBytes() } }.getOrNull()
            if (bytes == null) {
                app.lab.call(refreshAfter = false) { error("cannot read the picked image") }
                return@launch
            }
            app.lab.call { it.importBytes(name, bytes) }
        }
    }

    private fun displayName(uri: Uri): String? =
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        }
}
