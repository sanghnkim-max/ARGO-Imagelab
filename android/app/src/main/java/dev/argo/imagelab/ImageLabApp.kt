package dev.argo.imagelab

import android.app.Application
import dev.argo.imagelab.agent.Agent
import dev.argo.imagelab.agent.Settings
import dev.argo.imagelab.core.ImageLab
import dev.argo.imagelab.core.Lab
import java.io.File

/**
 * Process-wide singletons: one core (one PhotoCraft session) shared by every screen, the
 * in-app agent and the MCP server, as on the desktop.
 */
class ImageLabApp : Application() {
    val workspace: File by lazy { File(filesDir, "ImageLab").apply { mkdirs() } }
    val settings: Settings by lazy { Settings(this) }
    val lab: Lab by lazy { Lab(ImageLab(workspace.absolutePath, settings.gpu)) }
    val agent: Agent by lazy { Agent(lab, settings) }

    /** Where the bundled `imagelab` CLI lives (extracted with the native libraries). */
    val cliPath: String get() = File(applicationInfo.nativeLibraryDir, "libimagelab_cli.so").absolutePath
}
