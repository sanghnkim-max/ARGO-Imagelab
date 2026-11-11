// Plugin versions only; the app module holds the configuration.
//
// AGP 9 compiles Kotlin itself with its own pinned Kotlin Gradle Plugin. Pin the classpath to
// the Kotlin version the Compose and serialization compiler plugins are declared at, or the
// plugins load into an older compiler frontend (same fix as ARGO's argo-android).
buildscript {
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:2.4.20")
    }
}

plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.serialization) apply false
    alias(libs.plugins.kotlin.compose) apply false
}
