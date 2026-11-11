package dev.argo.imagelab

import android.app.Application
import dev.argo.imagelab.agent.Agent
import dev.argo.imagelab.agent.Settings
import dev.argo.imagelab.core.FfiLabApi
import dev.argo.imagelab.core.ImageLab
import dev.argo.imagelab.core.Lab
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import java.io.File

/**
 * Process-wide singletons: one core (one PhotoCraft session) shared by every screen, the
 * in-app agent and the MCP server, as on the desktop. The session lives here, not in an
 * Activity, so fold/unfold and rotation never lose it; screens keep only view state.
 */
class ImageLabApp : Application() {
    /** Outlives every screen: event listening, debounced refreshes, the agent loop. */
    val appScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    val workspace: File by lazy { File(filesDir, "ImageLab").apply { mkdirs() } }
    val settings: Settings by lazy { Settings(this) }
    val lab: Lab by lazy { Lab({ FfiLabApi(ImageLab(workspace.absolutePath, settings.gpu)) }, appScope) }
    val agent: Agent by lazy { Agent(lab, settings) }

    /** Where the bundled `imagelab` CLI lives (extracted with the native libraries). */
    val cliPath: String get() = File(applicationInfo.nativeLibraryDir, "libimagelab_cli.so").absolutePath
}
