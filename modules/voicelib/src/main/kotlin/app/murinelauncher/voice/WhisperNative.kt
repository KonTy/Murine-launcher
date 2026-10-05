// SPDX-License-Identifier: Apache-2.0
package app.murinelauncher.voice

import java.io.File
import java.io.IOException

/** JNI entry points of libmurine_whisper (see src/main/cpp). */
internal object WhisperNative {
    /** Absolute path of a host build, for the desktop smoke test only. */
    const val LIBRARY_PATH_PROPERTY = "murine.whisper.library"

    private const val LIBRARY = "murine_whisper"
    private const val LIBRARY_DOTPROD = "murine_whisper_dotprod"

    @Volatile
    private var loaded = false

    /**
     * Loads the native library once. A failure (no build for this CPU, a damaged install) is
     * reported as an [IOException], never as an [Error] that would take the launcher down, and is
     * not remembered as success.
     */
    @Throws(IOException::class)
    fun ensureLoaded() {
        if (loaded) return
        synchronized(this) {
            if (loaded) return
            load(System.getProperty(LIBRARY_PATH_PROPERTY), hasArmDotProd(), System::loadLibrary, System::load)
            loaded = true
        }
    }

    /** Picks and loads the right library; [loadLibrary] and [loadPath] are the System calls. */
    @Throws(IOException::class)
    fun load(override: String?, dotProd: Boolean, loadLibrary: (String) -> Unit, loadPath: (String) -> Unit) {
        try {
            when {
                override != null -> loadPath(override)
                dotProd -> try {
                    loadLibrary(LIBRARY_DOTPROD)
                } catch (_: LinkageError) {
                    loadLibrary(LIBRARY)
                }
                else -> loadLibrary(LIBRARY)
            }
        } catch (e: LinkageError) {
            throw IOException("Speech recognition library unavailable", e)
        } catch (e: SecurityException) {
            throw IOException("Speech recognition library unavailable", e)
        }
    }

    /** True on 64-bit ARM CPUs with the ARMv8.2 dot-product and half-precision SIMD extensions. */
    private fun hasArmDotProd(): Boolean {
        if (System.getProperty("os.arch") != "aarch64") return false
        val features = try {
            File("/proc/cpuinfo").useLines { lines ->
                lines.firstOrNull { it.startsWith("Features") }?.substringAfter(':')?.trim()?.split(' ')
            }
        } catch (_: Exception) {
            null
        } ?: return false
        return "asimddp" in features && "asimdhp" in features && "fphp" in features
    }

    @JvmStatic
    external fun nativeOpen(path: String): Long

    @JvmStatic
    external fun nativeTranscribe(
        handle: Long, samples: FloatArray, count: Int, language: String?,
        audioCtx: Int, threads: Int, maxTokens: Int,
    ): ByteArray?

    @JvmStatic
    external fun nativeAbort(handle: Long)

    @JvmStatic
    external fun nativeClose(handle: Long)
}

/** CPU architectures the speech library is built for (see voicelib's build.gradle). */
object VoiceAbis {
    val SUPPORTED: Set<String> = setOf("arm64-v8a", "armeabi-v7a", "x86_64")

    /** Whether one of [deviceAbis] (normally Build.SUPPORTED_ABIS) has a build of the library. */
    fun isSupported(deviceAbis: Array<String>): Boolean = deviceAbis.any { it in SUPPORTED }
}
