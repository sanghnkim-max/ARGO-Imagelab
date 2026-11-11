package dev.argo.imagelab

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.window.layout.WindowInfoTracker
import androidx.window.layout.WindowLayoutInfo
import dev.argo.imagelab.core.Imports
import dev.argo.imagelab.ui.AppRoot
import dev.argo.imagelab.ui.AppViewModel
import dev.argo.imagelab.ui.ImageLabTheme
import dev.argo.imagelab.ui.Route
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    private val app get() = application as ImageLabApp
    private val appModel: AppViewModel by viewModels()
    private val windowInfo = mutableStateOf<WindowLayoutInfo?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Light status/nav bar icons: the app is always dark.
        enableEdgeToEdge(SystemBarStyle.dark(android.graphics.Color.TRANSPARENT), SystemBarStyle.dark(android.graphics.Color.TRANSPARENT))
        setContent {
            ImageLabTheme {
                AppRoot(app = app, model = appModel, windowInfo = windowInfo.value, onOpenUri = ::openUri)
            }
        }
        // Folding features (hinge position, posture) for the tabletop and book layouts.
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                WindowInfoTracker.getOrCreate(this@MainActivity).windowLayoutInfo(this@MainActivity).collect { windowInfo.value = it }
            }
        }
        if (savedInstanceState == null) handleIntent(intent)
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
        uri?.let(::openUri)
    }

    /** Copy the image into the workspace's imports/ and open it in the editor. */
    fun openUri(uri: Uri) {
        lifecycleScope.launch {
            val picked = Imports.read(this@MainActivity, uri)
            if (picked == null) {
                app.lab.showMessage("Cannot read the picked image.")
                return@launch
            }
            val opened = app.lab.call { it.importBytes(picked.first, picked.second) }
            if (opened.isSuccess) {
                appModel.go(Route.Editor)
                app.lab.refresh()
            }
        }
    }
}
