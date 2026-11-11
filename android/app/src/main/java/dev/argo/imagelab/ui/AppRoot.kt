package dev.argo.imagelab.ui

import android.net.Uri
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.window.layout.WindowLayoutInfo
import dev.argo.imagelab.ImageLabApp

/**
 * The adaptive shell (docs/INTERFACES.md §10): Library (home), Editor, Settings; the agent as a
 * bottom sheet on compact windows and as a persistent right pane on medium/expanded windows and
 * on a folded-open foldable. Core errors surface here as snackbars.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppRoot(app: ImageLabApp, model: AppViewModel, windowInfo: WindowLayoutInfo?, onOpenUri: (Uri) -> Unit) {
    val layout = currentWindowLayout(windowInfo)
    val snackbar = remember { SnackbarHostState() }
    val labState by app.lab.state.collectAsState()

    LaunchedEffect(labState.message) {
        val msg = labState.message ?: return@LaunchedEffect
        app.lab.clearMessage()
        snackbar.showSnackbar(msg, withDismissAction = true)
    }
    BackHandler(enabled = model.canGoBack) { model.back() }

    val agentAsPane = !layout.compact || layout.posture != Posture.Flat
    val agentPane: (@Composable (Modifier) -> Unit)? = if (agentAsPane && model.agentOpen) {
        { m -> AgentPane(app, m, onClose = { model.agentOpen = false }, onOpenSettings = { model.go(Route.Settings) }) }
    } else null

    Scaffold(
        snackbarHost = { SnackbarHost(snackbar) },
        containerColor = MaterialTheme.colorScheme.background,
        contentWindowInsets = WindowInsets(0.dp),
    ) { pad ->
        Box(Modifier.fillMaxSize().padding(pad)) {
            when (model.route) {
                Route.Library -> LibraryScreen(
                    app = app,
                    layout = layout,
                    agentPane = agentPane,
                    onOpenEditor = { model.go(Route.Editor) },
                    onAskAgent = { ids ->
                        app.agent.attachContext(ids)
                        model.agentOpen = true
                    },
                    onToggleAgent = { model.agentOpen = !model.agentOpen },
                    onSettings = { model.go(Route.Settings) },
                    onEditor = { model.go(Route.Editor) },
                )
                Route.Editor -> EditorScreen(
                    app = app,
                    layout = layout,
                    agentPane = agentPane,
                    agentOpen = model.agentOpen,
                    onToggleAgent = { model.agentOpen = !model.agentOpen },
                    onHome = { model.go(Route.Library) },
                    onSettings = { model.go(Route.Settings) },
                    onOpenUri = onOpenUri,
                )
                Route.Settings -> SettingsScreen(app, onBack = { model.back() })
            }
        }
    }

    if (!agentAsPane && model.agentOpen) {
        ModalBottomSheet(
            onDismissRequest = { model.agentOpen = false },
            sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true),
        ) {
            AgentPane(
                app,
                Modifier.fillMaxHeight(0.88f),
                onClose = { model.agentOpen = false },
                onOpenSettings = {
                    model.agentOpen = false
                    model.go(Route.Settings)
                },
            )
        }
    }
}
