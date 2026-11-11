package dev.argo.imagelab.agent

import android.content.Context
import android.util.Base64
import java.security.SecureRandom

/**
 * App settings in private SharedPreferences (app-only storage; excluded from backup by
 * `allowBackup=false`). The API key is stored sealed with an Android Keystore key
 * ([SecretBox]); a plaintext key from an older build is sealed and removed on first read.
 * The key never leaves the device except to api.anthropic.com.
 *
 * [apiKey] touches the Keystore: read and write it off the main thread.
 */
class Settings(context: Context) {
    private val prefs = context.getSharedPreferences("imagelab", Context.MODE_PRIVATE)

    var apiKey: String
        get() {
            migratePlaintextKey()
            val sealed = prefs.getString(SEALED_KEY, null) ?: return ""
            return runCatching { String(SecretBox.open(Base64.decode(sealed, Base64.NO_WRAP)), Charsets.UTF_8) }.getOrDefault("")
        }
        set(v) {
            val key = v.trim()
            if (key.isEmpty()) {
                prefs.edit().remove(SEALED_KEY).remove(LEGACY_KEY).apply()
                return
            }
            val sealed = Base64.encodeToString(SecretBox.seal(key.toByteArray(Charsets.UTF_8)), Base64.NO_WRAP)
            prefs.edit().putString(SEALED_KEY, sealed).remove(LEGACY_KEY).apply()
        }

    /** Whether a key is stored, without opening it (cheap; main thread is fine). */
    val hasApiKey: Boolean get() = prefs.contains(SEALED_KEY) || !prefs.getString(LEGACY_KEY, null).isNullOrBlank()

    var model: String
        get() = prefs.getString("model", DEFAULT_MODEL) ?: DEFAULT_MODEL
        set(v) = prefs.edit().putString("model", v.trim().ifEmpty { DEFAULT_MODEL }).apply()

    /** Read at startup; a change takes effect on the next launch (the core is made once). */
    var gpu: Boolean
        get() = prefs.getBoolean("gpu", true)
        set(v) = prefs.edit().putBoolean("gpu", v).apply()

    var mcpPort: Int
        get() = prefs.getInt("mcp_port", 7700)
        set(v) = prefs.edit().putInt("mcp_port", v).apply()

    /** MCP clients may edit (otherwise read-only tools only, §8). */
    var mcpAllowEdits: Boolean
        get() = prefs.getBoolean("mcp_allow_edits", false)
        set(v) = prefs.edit().putBoolean("mcp_allow_edits", v).apply()

    /** The MCP server's bearer token: random, made once, rotatable from Settings → Developer. */
    val mcpToken: String
        get() = prefs.getString("mcp_token", null) ?: rotateMcpToken()

    fun rotateMcpToken(): String {
        val bytes = ByteArray(24).also { SecureRandom().nextBytes(it) }
        val token = bytes.joinToString("") { "%02x".format(it) }
        prefs.edit().putString("mcp_token", token).apply()
        return token
    }

    @Synchronized
    private fun migratePlaintextKey() {
        val legacy = prefs.getString(LEGACY_KEY, null) ?: return
        if (legacy.isNotBlank() && !prefs.contains(SEALED_KEY)) {
            runCatching { apiKey = legacy }.onFailure { return } // keep the plaintext if sealing fails
        }
        prefs.edit().remove(LEGACY_KEY).apply()
    }

    companion object {
        const val DEFAULT_MODEL = "claude-opus-5-5"
        private const val LEGACY_KEY = "api_key"
        private const val SEALED_KEY = "api_key_sealed"
    }
}
