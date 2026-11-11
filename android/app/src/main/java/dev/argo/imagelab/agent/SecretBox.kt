package dev.argo.imagelab.agent

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Seals small secrets (the API key) with an AES-256-GCM key in the Android Keystore that never
 * leaves secure hardware (TEE, or StrongBox where the Keystore picks it). Bytes:
 * `[v=1][12-byte IV][ciphertext+tag]`. The Keystore picks every IV.
 *
 * Plaintext is never logged. Key generation is slow the first time: call off the main thread.
 */
object SecretBox {
    private const val KEYSTORE = "AndroidKeyStore"
    private const val ALIAS = "imagelab.secrets.v1"
    private const val VERSION: Byte = 1
    private const val IV_BYTES = 12
    private const val TAG_BITS = 128

    @Synchronized
    private fun key(): SecretKey {
        val store = KeyStore.getInstance(KEYSTORE).apply { load(null) }
        (store.getKey(ALIAS, null) as? SecretKey)?.let { return it }
        val spec = KeyGenParameterSpec.Builder(ALIAS, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
            .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
            .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
            .setKeySize(256)
            .setRandomizedEncryptionRequired(true)
            .setUserAuthenticationRequired(false)
            .build()
        return KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, KEYSTORE).run {
            init(spec)
            generateKey()
        }
    }

    fun seal(plaintext: ByteArray): ByteArray {
        val cipher = Cipher.getInstance("AES/GCM/NoPadding").apply { init(Cipher.ENCRYPT_MODE, key()) }
        val ciphertext = cipher.doFinal(plaintext)
        val iv = cipher.iv
        require(iv.size == IV_BYTES) { "unexpected IV size" }
        return byteArrayOf(VERSION) + iv + ciphertext
    }

    fun open(sealed: ByteArray): ByteArray {
        require(sealed.size > 1 + IV_BYTES && sealed[0] == VERSION) { "not a sealed value" }
        val iv = sealed.copyOfRange(1, 1 + IV_BYTES)
        val cipher = Cipher.getInstance("AES/GCM/NoPadding").apply {
            init(Cipher.DECRYPT_MODE, key(), GCMParameterSpec(TAG_BITS, iv))
        }
        return cipher.doFinal(sealed, 1 + IV_BYTES, sealed.size - 1 - IV_BYTES)
    }
}
