// ImageLab for Android: a Compose shell over PhotoCraft's engine.
//
// The Rust side (../../crates) builds with cargo-ndk into this module's jniLibs:
//   * libimagelab_ffi.so: the core (imagelab-ffi, uniffi). Kotlin bindings are generated from
//     the debug .so by uniffi-bindgen into build/generated/uniffi.
//   * libimagelab_cli.so: the `imagelab` executable (photocraft-cli), named like a library so
//     the installer extracts it to nativeLibraryDir with the exec bit, for `adb shell`
//     (see README "CLI on the phone"). Android only allows exec of files there.
//
// No `org.jetbrains.kotlin.android` plugin: AGP 9 compiles Kotlin built in.

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "dev.argo.imagelab"
    compileSdk = 37

    defaultConfig {
        applicationId = "dev.argo.imagelab"
        // Vulkan 1.1 is guaranteed for 64-bit devices from Android 10 (API 29), which is what
        // wgpu needs for PhotoCraft's GPU renderer. Below that it falls back to the CPU.
        minSdk = 29
        targetSdk = 37
        versionCode = 1
        versionName = "0.1.0"
        ndk {
            // arm64 phones and the Apple-silicon emulator. Add x86_64 only when a matrix needs it.
            abiFilters += "arm64-v8a"
        }
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    packaging {
        jniLibs {
            // Extract .so files to nativeLibraryDir: libimagelab_cli.so must exist there as a
            // real file to be executed from `adb shell`.
            useLegacyPackaging = true
        }
    }

    // A plain File: AGP 9's SourceSet API refuses a Provider here. preBuild depends on
    // uniffiBindgen, so the directory is filled before anything reads it.
    sourceSets["main"].kotlin.directories.add(layout.buildDirectory.dir("generated/uniffi").get().asFile.path)
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.kotlinx.coroutines.android)
    implementation("${libs.jna.get()}@aar")
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.foundation)
    implementation(libs.compose.material3)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.window)
    implementation(libs.androidx.window.core)
    implementation(libs.okhttp)
    implementation(libs.anthropic.java)
    debugImplementation(libs.compose.ui.tooling)
    testImplementation(libs.junit)
}

// ── Rust ─────────────────────────────────────────────────────────────────────────────────────

val repoRoot: File = rootProject.projectDir.parentFile
val abi = "arm64-v8a"
val ndkTarget = "aarch64-linux-android"
// At or below minSdk: a .so built for an older platform loads on newer ones.
val ndkPlatform = "26"
val ffiLib = "libimagelab_ffi.so"
val uniffiOutDir = layout.buildDirectory.dir("generated/uniffi")

// Cargo decides what is stale; Gradle always asks it (no inputs over the Rust sources).
fun cargoNdkTask(buildType: String, release: Boolean) =
    tasks.register<Exec>("cargoNdk${buildType.replaceFirstChar { it.uppercase() }}") {
        description = "Builds the $buildType Rust libraries and the imagelab CLI with cargo-ndk."
        val jniLibs = layout.projectDirectory.dir("src/$buildType/jniLibs")
        val profileDir = if (release) "release" else "debug"
        workingDir = repoRoot
        commandLine(
            listOf("cargo", "ndk", "-t", abi, "--platform", ndkPlatform, "-o", jniLibs.asFile.path, "build") +
                (if (release) listOf("--release") else emptyList()) +
                listOf("-p", "imagelab-ffi", "-p", "imagelab-cli"),
        )
        doLast {
            // cargo-ndk copies cdylibs only; the CLI executable rides along as a "library".
            val cli = repoRoot.resolve("target/$ndkTarget/$profileDir/imagelab")
            val dest = jniLibs.dir(abi).file("libimagelab_cli.so").asFile
            dest.parentFile.mkdirs()
            cli.copyTo(dest, overwrite = true)
        }
        outputs.upToDateWhen { false }
    }

val cargoNdkDebug = cargoNdkTask("debug", release = false)
val cargoNdkRelease = cargoNdkTask("release", release = true)

val uniffiBindgen by tasks.registering(Exec::class) {
    description = "Generates the Kotlin bindings from the unstripped debug .so."
    dependsOn(cargoNdkDebug)
    workingDir = repoRoot
    val so = repoRoot.resolve("target/$ndkTarget/debug/$ffiLib")
    doFirst { uniffiOutDir.get().asFile.mkdirs() }
    commandLine(
        "cargo", "run", "--locked", "-p", "uniffi-bindgen", "--",
        "generate", "--library", so.path,
        "--language", "kotlin",
        "--no-format",
        "--out-dir", uniffiOutDir.get().asFile.path,
    )
    outputs.dir(uniffiOutDir)
    outputs.upToDateWhen { false }
}

tasks.named("preBuild") { dependsOn(uniffiBindgen) }
tasks.configureEach {
    when (name) {
        "preDebugBuild" -> dependsOn(cargoNdkDebug)
        "preReleaseBuild" -> dependsOn(cargoNdkRelease)
    }
}
