package dev.argo.imagelab

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
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

    /** Gallery access, for ARGO's agent opening a photo by id (`CompanionMcp` `open`). */
    private val photoAccess = registerForActivityResult(ActivityResultContracts.RequestPermission()) {}

    /**
     * Ask for gallery access the first time ImageLab opens, so ARGO's agent can open a photo by
     * its id; the editor's own picker needs none. Asked once: a refusal stays the person's.
     */
    private fun askPhotoAccessOnce() {
        val permission = if (Build.VERSION.SDK_INT >= 33) Manifest.permission.READ_MEDIA_IMAGES else Manifest.permission.READ_EXTERNAL_STORAGE
        if (checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED) return
        val prefs = getSharedPreferences("imagelab", MODE_PRIVATE)
        if (prefs.getBoolean(ASKED_PHOTOS, false)) return
        prefs.edit().putBoolean(ASKED_PHOTOS, true).apply()
        photoAccess.launch(permission)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        askPhotoAccessOnce()
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

    private companion object {
        const val ASKED_PHOTOS = "askedPhotoAccess"
    }
}
