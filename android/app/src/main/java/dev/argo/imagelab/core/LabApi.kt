package dev.argo.imagelab.core

/**
 * Everything the app asks of the Rust core (docs/INTERFACES.md §11), as one interface.
 *
 * Screens, view models and the agent depend on this, never on the generated [ImageLab]
 * directly, so a test can fake it and the FFI can change underneath. Every method blocks:
 * call them from [kotlinx.coroutines.Dispatchers.IO] (see [Lab.io]). Failures are thrown
 * as [ImageLabException] (or any Throwable for "not linked yet"); callers turn them into
 * messages, never crashes.
 */
interface LabApi {
    // ── §11: the registry, scripts, events, handles ──────────────────────────────────────
    /** `Registry::call(ctx, id, params)`. `source` is `ui`, `agent`, `mcp` or `cli`. */
    fun call(id: String, paramsJson: String, source: String = "ui"): String

    /** `Registry::help(area)`: one area's functions with schemas, or every area when null. */
    fun help(area: String?): String

    /** One line per area: what the agent's tool description lists (§3). */
    fun areasSummary(): String

    /** Run a JS program against `lab` (rquickjs, fresh realm). Errors come back in [ScriptResult.error]. */
    fun runScript(code: String, readOnly: Boolean, timeoutMs: ULong): ScriptResult

    /** Abort every running program (the agent's Stop button). */
    fun cancelScripts()

    /** Where core events (§5) go; [EventListener.onEvent] runs off the main thread. */
    fun setEventListener(listener: EventListener)

    /** PNG of an `img:N` / `mask:N` handle, for overlays. */
    fun renderHandle(handle: String, maxSide: UInt): ByteArray

    /** Serve MCP on 127.0.0.1:[port] with a bearer [token]; edit tools only when [allowEdits]. */
    fun startMcpHttp(port: UShort, token: String, allowEdits: Boolean): UShort

    // ── existing calls the app still uses ────────────────────────────────────────────────
    fun workspaceRoot(): String
    fun open(path: String): String
    fun importBytes(name: String, bytes: ByteArray): String
    fun newDocument(paramsJson: String): String
    fun save(path: String, format: String?): String
    fun sessionInfo(): String
    fun inspect(): String
    fun run(id: String, paramsJson: String): String
    fun undo(): String
    fun redo(): String
    fun renderPng(maxSide: UInt): ByteArray
    fun gpuDisabled(): Boolean
    fun stopMcpHttp()
    fun mcpHttpPort(): UShort?
    fun cli(commandLine: String): CliResult
}

/**
 * [LabApi] over the generated uniffi object.
 *
 * The §11 calls resolve to members of [ImageLab] once the core exports them; until then they
 * resolve to the extension stand-ins in `PendingFfi.kt` (Kotlin prefers a member over an
 * extension of the same signature, so deleting that one file is the whole swap).
 */
class FfiLabApi(private val core: ImageLab) : LabApi {
    override fun call(id: String, paramsJson: String, source: String): String = core.call(id, paramsJson, source)
    override fun help(area: String?): String = core.help(area)
    override fun areasSummary(): String = core.areasSummary()
    override fun runScript(code: String, readOnly: Boolean, timeoutMs: ULong): ScriptResult = core.runScript(code, readOnly, timeoutMs)
    override fun cancelScripts() = core.cancelScripts()
    override fun setEventListener(listener: EventListener) = core.setEventListener(listener)
    override fun renderHandle(handle: String, maxSide: UInt): ByteArray = core.renderHandle(handle, maxSide)
    override fun startMcpHttp(port: UShort, token: String, allowEdits: Boolean): UShort = core.startMcpHttp(port, token, allowEdits)

    override fun workspaceRoot(): String = core.workspaceRoot()
    override fun open(path: String): String = core.open(path)
    override fun importBytes(name: String, bytes: ByteArray): String = core.importBytes(name, bytes)
    override fun newDocument(paramsJson: String): String = core.newDocument(paramsJson)
    override fun save(path: String, format: String?): String = core.save(path, format)
    override fun sessionInfo(): String = core.sessionInfo()
    override fun inspect(): String = core.inspect()
    override fun run(id: String, paramsJson: String): String = core.run(id, paramsJson)
    override fun undo(): String = core.undo()
    override fun redo(): String = core.redo()
    override fun renderPng(maxSide: UInt): ByteArray = core.renderPng(maxSide)
    override fun gpuDisabled(): Boolean = core.gpuDisabled()
    override fun stopMcpHttp() = core.stopMcpHttp()
    override fun mcpHttpPort(): UShort? = core.mcpHttpPort()
    override fun cli(commandLine: String): CliResult = core.cli(commandLine)
}
