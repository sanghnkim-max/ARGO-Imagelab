package dev.argo.imagelab.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel

enum class Route { Library, Editor, Settings }

/**
 * App-level view state: which screen, whether the agent is showing. Survives rotation and
 * fold/unfold (a ViewModel); the document itself lives in the core.
 */
class AppViewModel : ViewModel() {
    private val backStack = mutableStateListOf(Route.Library)
    val route: Route get() = backStack.last()
    var agentOpen by mutableStateOf(false)

    fun go(to: Route) {
        if (route == to) return
        backStack.remove(to)
        backStack.add(to)
    }

    /** False when there is nowhere to go back to (the system then closes the app). */
    fun back(): Boolean {
        if (agentOpen) {
            agentOpen = false
            return true
        }
        if (backStack.size <= 1) return false
        backStack.removeAt(backStack.lastIndex)
        return true
    }

    val canGoBack: Boolean get() = agentOpen || backStack.size > 1
}
