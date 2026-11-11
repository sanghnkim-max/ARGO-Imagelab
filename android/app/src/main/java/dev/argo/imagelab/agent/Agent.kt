package dev.argo.imagelab.agent

import com.anthropic.client.AnthropicClient
import com.anthropic.client.okhttp.AnthropicOkHttpClient
import com.anthropic.core.JsonValue
import com.anthropic.models.messages.Base64ImageSource
import com.anthropic.models.messages.CacheControlEphemeral
import com.anthropic.models.messages.ContentBlockParam
import com.anthropic.models.messages.ImageBlockParam
import com.anthropic.models.messages.Message
import com.anthropic.models.messages.MessageCreateParams
import com.anthropic.models.messages.MessageParam
import com.anthropic.models.messages.OutputConfig
import com.anthropic.models.messages.StopReason
import com.anthropic.models.messages.TextBlockParam
import com.anthropic.models.messages.Tool
import com.anthropic.models.messages.ToolResultBlockParam
import com.anthropic.models.messages.ToolUseBlock
import dev.argo.imagelab.core.Lab
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.longOrNull
import kotlinx.serialization.json.put

/** One line in the agent panel. */
sealed interface ChatItem {
    data class User(val text: String) : ChatItem
    data class Assistant(val text: String) : ChatItem
    data class ToolStep(val name: String, val input: String, val ok: Boolean, val summary: String) : ChatItem
    data class Notice(val text: String) : ChatItem
}

/**
 * The in-app agent: a manual Claude tool loop (official Anthropic Java SDK) whose tools are
 * PhotoCraft's own MCP tools, listed and called in-process through the core, plus one tool
 * that runs `photocraft-cli` in the workspace. The agent edits the document the person sees;
 * the editor re-renders after each tool step.
 *
 * Tool sources are kept separate on purpose: AI image generation (and later the video app's
 * tools) are meant to join as further MCP servers, not as code here.
 */
class Agent(private val lab: Lab, private val settings: Settings) {
    private val _items = MutableStateFlow<List<ChatItem>>(emptyList())
    val items: StateFlow<List<ChatItem>> = _items.asStateFlow()
    private val _running = MutableStateFlow(false)
    val running: StateFlow<Boolean> = _running.asStateFlow()

    private val json = Json { ignoreUnknownKeys = true }
    private val history = mutableListOf<MessageParam>()
    private var tools: List<Tool>? = null
    private var client: AnthropicClient? = null
    private var clientKey: String? = null

    fun reset() {
        history.clear()
        _items.value = emptyList()
    }

    suspend fun send(text: String) {
        if (_running.value || text.isBlank()) return
        val key = settings.apiKey
        if (key.isBlank()) {
            add(ChatItem.Notice("Add your Anthropic API key in the agent settings first."))
            return
        }
        _running.value = true
        add(ChatItem.User(text))
        try {
            withContext(Dispatchers.IO) { loop(key, text) }
        } catch (t: Throwable) {
            add(ChatItem.Notice("Error: ${t.message ?: t}"))
        } finally {
            _running.value = false
        }
    }

    private fun loop(key: String, text: String) {
        val client = clientFor(key)
        val tools = tools ?: loadTools().also { tools = it }
        history += MessageParam.ofUser(text)
        repeat(MAX_STEPS) {
            val params = MessageCreateParams.builder()
                .model(settings.model)
                .maxTokens(16000L)
                // Opus 5.5: thinking is always on (adaptive); effort is the control. Set it
                // explicitly: this model defaults to medium.
                .outputConfig(OutputConfig.builder().effort(OutputConfig.Effort.HIGH).build())
                .systemOfTextBlockParams(
                    listOf(
                        TextBlockParam.builder().text(SYSTEM_PROMPT)
                            .cacheControl(CacheControlEphemeral.builder().build()).build(),
                    ),
                )
                .tools(tools.map { com.anthropic.models.messages.ToolUnion.ofTool(it) })
                // Auto-cache the growing conversation (tools + system + messages prefix).
                .cacheControl(CacheControlEphemeral.builder().build())
                .messages(history)
                // Server-side refusal fallback: a declined request is retried on a model the
                // API picks by refusal category instead of simply stopping.
                .putAdditionalHeader("anthropic-beta", "server-side-fallback-2026-07-01")
                .putAdditionalBodyProperty("fallbacks", JsonValue.from("default"))
                .build()
            val response: Message = client.messages().create(params)
            history += response.toParam()

            response.content().forEach { block ->
                block.text().ifPresent { t -> if (t.text().isNotBlank()) add(ChatItem.Assistant(t.text())) }
            }
            when (response.stopReason().orElse(null)) {
                StopReason.TOOL_USE -> {
                    val uses = response.content().mapNotNull { it.toolUse().orElse(null) }
                    // All results go back in one user message (keeps parallel tool use working).
                    val results = uses.map { runTool(it) }
                    history += MessageParam.ofUserOfContentBlockParams(results)
                }
                StopReason.REFUSAL -> {
                    add(ChatItem.Notice("The model declined this request."))
                    return
                }
                StopReason.MAX_TOKENS -> {
                    add(ChatItem.Notice("The reply hit the length limit."))
                    return
                }
                else -> return
            }
        }
        add(ChatItem.Notice("Stopped after $MAX_STEPS steps."))
    }

    private fun clientFor(key: String): AnthropicClient {
        val existing = client
        if (existing != null && clientKey == key) return existing
        return AnthropicOkHttpClient.builder().apiKey(key).build().also {
            client = it
            clientKey = key
        }
    }

    // ------------------------------------------------------------------ tools

    /** PhotoCraft's MCP tools (minus bridge-only UI tools), in a stable order, plus the CLI. */
    private fun loadTools(): List<Tool> {
        val listed = json.parseToJsonElement(lab.core.mcpRequest("tools/list", "{}")) as JsonObject
        val mcpTools = (listed["tools"] as? JsonArray).orEmpty().mapNotNull { it as? JsonObject }
            .filter { t -> t.str("name")?.let { it !in BRIDGE_ONLY } == true }
            .sortedBy { it.str("name") }
            .map { t ->
                val schema = t["inputSchema"] as? JsonObject ?: JsonObject(emptyMap())
                tool(t.str("name")!!, t.str("description").orEmpty(), schema)
            }
        val cli = tool(
            CLI_TOOL,
            "Run a photocraft-cli command line in the ImageLab workspace (its own headless session, " +
                "not the open document): e.g. `info imports/a.jpg`, `commands --filter blur`, " +
                "`convert exports/a.pcraft exports/a.png`. Returns exit code, stdout and stderr.",
            buildJsonObject {
                put("type", "object")
                put("properties", buildJsonObject { put("command", buildJsonObject { put("type", "string") }) })
                put("required", JsonArray(listOf(JsonPrimitive("command"))))
            },
        )
        return mcpTools + cli
    }

    private fun tool(name: String, description: String, schema: JsonObject): Tool {
        val properties = (schema["properties"] as? JsonObject).orEmpty()
        val required = (schema["required"] as? JsonArray).orEmpty().mapNotNull { (it as? JsonPrimitive)?.contentOrNull }
        val props = Tool.InputSchema.Properties.builder()
        properties.forEach { (k, v) -> props.putAdditionalProperty(k, JsonValue.from(toJava(v))) }
        val input = Tool.InputSchema.builder().properties(props.build())
        if (required.isNotEmpty()) input.required(required)
        return Tool.builder().name(name).description(description).inputSchema(input.build()).build()
    }

    private fun runTool(use: ToolUseBlock): ContentBlockParam {
        val name = use.name()
        val inputJson = com.anthropic.core.jsonMapper().writeValueAsString(use._input())
        val result = runCatching {
            if (name == CLI_TOOL) runCli(inputJson) else runMcp(name, inputJson)
        }.getOrElse { ToolOutput(listOf(TextPart("error: ${it.message ?: it}")), isError = true) }
        add(ChatItem.ToolStep(name, inputJson.take(300), !result.isError, result.summary()))
        // The tool may have edited the document; let the editor catch up.
        kotlinx.coroutines.runBlocking { lab.refresh() }
        val blocks = result.parts.map { part ->
            when (part) {
                is TextPart -> ToolResultBlockParam.Content.Block.ofText(TextBlockParam.builder().text(part.text.ifEmpty { "(empty)" }).build())
                is ImagePart -> ToolResultBlockParam.Content.Block.ofImage(
                    ImageBlockParam.builder().source(
                        Base64ImageSource.builder().data(part.base64).mediaType(mediaType(part.mime)).build(),
                    ).build(),
                )
            }
        }
        return ContentBlockParam.ofToolResult(
            ToolResultBlockParam.builder().toolUseId(use.id()).contentOfBlocks(blocks).isError(result.isError).build(),
        )
    }

    private fun runMcp(name: String, inputJson: String): ToolOutput {
        val params = buildJsonObject {
            put("name", name)
            put("arguments", json.parseToJsonElement(inputJson))
        }
        val reply = json.parseToJsonElement(lab.core.mcpRequest("tools/call", params.toString())) as JsonObject
        val parts = (reply["content"] as? JsonArray).orEmpty().mapNotNull { c ->
            val o = c as? JsonObject ?: return@mapNotNull null
            when (o.str("type")) {
                "text" -> TextPart(o.str("text").orEmpty().take(MAX_TEXT))
                "image" -> ImagePart(o.str("data").orEmpty(), o.str("mimeType") ?: "image/png")
                else -> null
            }
        }
        val isError = (reply["isError"] as? JsonPrimitive)?.booleanOrNull == true
        return ToolOutput(parts.ifEmpty { listOf(TextPart("ok")) }, isError)
    }

    private fun runCli(inputJson: String): ToolOutput {
        val command = (json.parseToJsonElement(inputJson) as? JsonObject)?.str("command").orEmpty()
        val out = lab.core.cli(command)
        val text = buildString {
            append("exit ${out.code}\n")
            if (out.stdout.isNotEmpty()) append("stdout:\n").append(out.stdout.take(MAX_TEXT)).append('\n')
            if (out.stderr.isNotEmpty()) append("stderr:\n").append(out.stderr.take(MAX_TEXT))
        }
        return ToolOutput(listOf(TextPart(text)), isError = out.code != 0)
    }

    private fun add(item: ChatItem) = _items.update { it + item }

    private sealed interface Part
    private data class TextPart(val text: String) : Part
    private data class ImagePart(val base64: String, val mime: String) : Part
    private data class ToolOutput(val parts: List<Part>, val isError: Boolean) {
        fun summary(): String = parts.joinToString(" ") {
            when (it) {
                is TextPart -> it.text.replace('\n', ' ').take(160)
                is ImagePart -> "[image]"
            }
        }
    }

    private fun mediaType(mime: String) = when (mime) {
        "image/jpeg" -> Base64ImageSource.MediaType.IMAGE_JPEG
        "image/webp" -> Base64ImageSource.MediaType.IMAGE_WEBP
        "image/gif" -> Base64ImageSource.MediaType.IMAGE_GIF
        else -> Base64ImageSource.MediaType.IMAGE_PNG
    }

    private fun JsonObject.str(key: String) = (this[key] as? JsonPrimitive)?.contentOrNull

    /** kotlinx JSON → plain Java values for the SDK's JsonValue. */
    private fun toJava(e: JsonElement): Any? = when (e) {
        is JsonNull -> null
        is JsonObject -> e.mapValues { toJava(it.value) }
        is JsonArray -> e.map { toJava(it) }
        is JsonPrimitive -> when {
            e.isString -> e.content
            e.booleanOrNull != null -> e.booleanOrNull
            e.longOrNull != null -> e.longOrNull
            else -> e.doubleOrNull
        }
    }

    companion object {
        private const val MAX_STEPS = 30
        private const val MAX_TEXT = 60_000
        private const val CLI_TOOL = "photocraft_cli"

        /** Tools that only work against a running desktop app (bridge mode). */
        private val BRIDGE_ONLY = setOf("ui_state", "ui_screenshot", "ui_pointer", "ui_menu_invoke", "ui_set", "control_call")

        private val SYSTEM_PROMPT = """
            You are the editing agent inside ImageLab, an Android photo editor built on PhotoCraft's engine
            (a Photoshop-style layered editor). The person sees the active document on screen while you work.

            How to work:
            - Inspect first: `doc_inspect` for layers/history; `doc_render_preview` (small max_side, e.g. 768) to see the image.
            - Edits are engine commands: `command_list` to find ids and params (filter by text), then `command_run`
              or `command_batch`. Prefer non-destructive edits: adjustment layers (`layer.newAdjustmentLayer.*`) over
              direct pixel adjustments (`image.adjustments.*`), so the person can still change or hide them.
            - Before a risky experiment, duplicate the document (`image.duplicate`) and work on the copy.
            - Files live in the workspace: photos the person opened are under `imports/`; save results to
              `exports/` (`doc_save` with a .png/.jpg/.pcraft path). Never overwrite files in `imports/`.
            - `photocraft_cli` runs the PhotoCraft CLI on files in the workspace (separate session) — useful for
              `info` on a file or batch conversion, not for editing the open document.
            - After editing, look at a preview once to check the result, then say briefly what you changed.
        """.trimIndent()
    }
}
