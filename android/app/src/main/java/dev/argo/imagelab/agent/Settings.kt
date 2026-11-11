package dev.argo.imagelab.agent

import android.content.Context
import java.security.SecureRandom

/**
 * App settings in private SharedPreferences (app-only storage; excluded from backup by
 * `allowBackup=false`). The API key never leaves the device except to api.anthropic.com.
 */
class Settings(context: Context) {
    private val prefs = context.getSharedPreferences("imagelab", Context.MODE_PRIVATE)

    var apiKey: String
        get() = prefs.getString("api_key", "").orEmpty()
        set(v) = prefs.edit().putString("api_key", v.trim()).apply()

    var model: String
        get() = prefs.getString("model", DEFAULT_MODEL) ?: DEFAULT_MODEL
        set(v) = prefs.edit().putString("model", v.trim().ifEmpty { DEFAULT_MODEL }).apply()

    var gpu: Boolean
        get() = prefs.getBoolean("gpu", true)
        set(v) = prefs.edit().putBoolean("gpu", v).apply()

    var mcpPort: Int
        get() = prefs.getInt("mcp_port", 7700)
        set(v) = prefs.edit().putInt("mcp_port", v).apply()

    /** The MCP server's bearer token: random, made once, rotatable from the MCP panel. */
    val mcpToken: String
        get() = prefs.getString("mcp_token", null) ?: rotateMcpToken()

    fun rotateMcpToken(): String {
        val bytes = ByteArray(24).also { SecureRandom().nextBytes(it) }
        val token = bytes.joinToString("") { "%02x".format(it) }
        prefs.edit().putString("mcp_token", token).apply()
        return token
    }

    companion object {
        const val DEFAULT_MODEL = "claude-opus-5-5"
    }
}
