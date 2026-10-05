// SPDX-License-Identifier: Apache-2.0
package app.murinelauncher.voice

import java.io.Closeable
import java.io.File
import java.io.IOException

/** A loaded speech-to-text model. Not thread-safe except for [abort] and [close]. */
interface SpeechEngine : Closeable {
    /**
     * Transcribes the first [count] samples (16 kHz mono, -1..1). Blocking.
     * Returns null when aborted or when the model fails.
     */
    fun transcribe(samples: FloatArray, count: Int): String?

    /** Makes a running or upcoming [transcribe] return null as soon as possible. Any thread. */
    fun abort()

    /**
     * Releases the model. Any thread, any number of times: while a transcription is running it is
     * aborted and the native memory is freed as soon as it returns.
     */
    override fun close()
}

fun interface SpeechEngineFactory {
    /** Loads the model. Blocking: call it off the main thread. */
    @Throws(IOException::class)
    fun open(): SpeechEngine
}

/** Settings for one [WhisperModel]. */
data class WhisperOptions(
    val language: String,
    val dynamicAudioContext: Boolean,
    val threads: Int = WhisperTuning.threadCount(),
    val maxTokens: Int = WhisperTuning.MAX_TOKENS,
)

/** whisper.cpp model held in native memory between [open] and [close]. */
class WhisperModel private constructor(
    private var handle: Long,
    private val options: WhisperOptions,
) : SpeechEngine {

    private val lock = Any()
    private var busy = false
    private var aborted = false
    private var closeRequested = false

    override fun transcribe(samples: FloatArray, count: Int): String? {
        val h = synchronized(lock) {
            if (handle == 0L || aborted || busy) return null
            busy = true
            handle
        }
        try {
            val audioCtx = WhisperTuning.audioContext(count, options.dynamicAudioContext)
            val bytes = WhisperNative.nativeTranscribe(
                h, samples, count, options.language, audioCtx, options.threads, options.maxTokens
            ) ?: return null
            return String(bytes, Charsets.UTF_8)
        } catch (_: LinkageError) {
            return null
        } finally {
            val toFree = synchronized(lock) {
                busy = false
                if (closeRequested) handle.also { handle = 0L } else 0L
            }
            if (toFree != 0L) WhisperNative.nativeClose(toFree)
        }
    }

    override fun abort() {
        synchronized(lock) {
            aborted = true
            if (handle != 0L) WhisperNative.nativeAbort(handle)
        }
    }

    override fun close() {
        val toFree = synchronized(lock) {
            if (handle == 0L) return
            aborted = true
            if (busy) {
                // Freed by transcribe() once the native call returns
                closeRequested = true
                WhisperNative.nativeAbort(handle)
                0L
            } else {
                handle.also { handle = 0L }
            }
        }
        WhisperNative.nativeClose(toFree)
    }

    companion object {
        /** Loads [file] into memory. Blocking. */
        @Throws(IOException::class)
        fun open(file: File, options: WhisperOptions): WhisperModel {
            WhisperNative.ensureLoaded()
            val handle = try {
                WhisperNative.nativeOpen(file.absolutePath)
            } catch (e: LinkageError) {
                throw IOException("Speech recognition library unavailable", e)
            }
            if (handle == 0L) throw IOException("Could not load speech model")
            return WhisperModel(handle, options)
        }
    }
}
