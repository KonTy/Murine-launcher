// SPDX-License-Identifier: Apache-2.0
package app.murinelauncher.voice

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test

/**
 * Desktop smoke test of the real JNI library and a real model. Skipped unless pointed at a host
 * build of libmurine_whisper (src/main/cpp builds on Linux too), a model and a 16 kHz mono WAV:
 *
 *   MURINE_WHISPER_LIBRARY=/path/libmurine_whisper.so MURINE_WHISPER_MODEL=/path/model.bin \
 *   MURINE_WHISPER_WAV=/path/speech.wav MURINE_WHISPER_EXPECT="some words" \
 *   gradle :voicelib:testDebugUnitTest --tests '*WhisperNativeHostTest*'
 */
class WhisperNativeHostTest {

    private val library = System.getenv("MURINE_WHISPER_LIBRARY")
    private val model = System.getenv("MURINE_WHISPER_MODEL")?.let(::File)
    private val wav = System.getenv("MURINE_WHISPER_WAV")?.let(::File)

    private fun assumeConfigured() {
        assumeTrue("Native smoke test not configured", library != null && model != null && wav != null)
        System.setProperty(WhisperNative.LIBRARY_PATH_PROPERTY, library!!)
    }

    @Test
    fun transcribesWithTheRealModel() {
        assumeConfigured()
        val header = WhisperFile.validate(model!!)
        val pcm = readWav(wav!!)
        val acft = VoiceModelCatalog.models.any { it.bytes == model.length() }
        val options = WhisperOptions(
            language = WhisperTuning.resolveLanguage(WhisperTuning.LANGUAGE_DEVICE, header.isMultilingual, Locale.ENGLISH),
            dynamicAudioContext = acft,
            threads = System.getenv("MURINE_WHISPER_THREADS")?.toInt() ?: 4,
        )
        val rssBefore = rssKb()
        val t0 = System.nanoTime()
        val engine = WhisperModel.open(model, options)
        val t1 = System.nanoTime()
        val rssLoaded = rssKb()
        val raw = engine.transcribe(pcm, pcm.size)
        val t2 = System.nanoTime()
        val rssPeak = hwmKb()
        engine.close()
        engine.close() // second close is a no-op
        val rssAfter = rssKb()

        val text = TranscriptCleaner.clean(raw.orEmpty())
        println("model=${model.name} audio=${pcm.size / 16_000.0}s threads=${options.threads} " +
            "acft=$acft ctx=${WhisperTuning.audioContext(pcm.size, acft)} " +
            "load=${(t1 - t0) / 1_000_000}ms transcribe=${(t2 - t1) / 1_000_000}ms " +
            "rss: before=${rssBefore / 1024}MB loaded=${rssLoaded / 1024}MB peak=${rssPeak / 1024}MB " +
            "afterClose=${rssAfter / 1024}MB text=\"$text\"")
        val expected = System.getenv("MURINE_WHISPER_EXPECT")
        if (expected != null) {
            assertTrue("Got \"$text\"", text.lowercase().contains(expected.lowercase()))
        } else {
            assertTrue(text.isNotBlank())
        }
    }

    @Test
    fun abortedEngineReturnsNothing() {
        assumeConfigured()
        val engine = WhisperModel.open(model!!, WhisperOptions("en", dynamicAudioContext = true, threads = 2))
        engine.abort()
        assertNull(engine.transcribe(FloatArray(32_000), 32_000))
        engine.close()
        // Using a closed engine is harmless
        assertNull(engine.transcribe(FloatArray(32_000), 32_000))
        engine.abort()
    }

    @Test
    fun abortStopsARunningTranscription() {
        assumeConfigured()
        val engine = WhisperModel.open(model!!, WhisperOptions("en", dynamicAudioContext = false, threads = 1))
        val pcm = readWav(wav!!)
        val aborter = Thread {
            Thread.sleep(50)
            engine.abort()
        }
        val t0 = System.nanoTime()
        aborter.start()
        val result = engine.transcribe(pcm, pcm.size)
        val elapsed = (System.nanoTime() - t0) / 1_000_000
        aborter.join()
        // Closing from another thread mid-run is deferred, never a double free
        engine.close()
        println("aborted after ${elapsed}ms, result=$result")
        assertNull(result)
    }

    @Test
    fun repeatedQueriesDoNotAccumulateMemory() {
        assumeConfigured()
        val pcm = readWav(wav!!)
        val rss = (1..6).map {
            WhisperModel.open(model!!, WhisperOptions("en", dynamicAudioContext = true, threads = 4)).use { engine ->
                engine.transcribe(pcm, pcm.size)
            }
            System.gc()
            rssKb()
        }
        println("rss after each query (MB): ${rss.map { it / 1024 }}")
        // Allocator caches settle after the first rounds; a leak would add a model's worth each time
        assertTrue("RSS grew: $rss", rss.last() - rss[1] < 16 * 1024)
    }

    private fun readWav(file: File): FloatArray = RandomAccessFile(file, "r").use { raf ->
        val bytes = ByteArray(raf.length().toInt()).also { raf.readFully(it) }
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        // Find the "data" chunk
        var pos = 12
        while (pos + 8 <= bytes.size) {
            val id = String(bytes, pos, 4, Charsets.US_ASCII)
            val size = buf.getInt(pos + 4)
            if (id == "data") {
                val samples = size / 2
                return FloatArray(samples) { buf.getShort(pos + 8 + it * 2) / 32768f }
            }
            pos += 8 + size
        }
        error("No data chunk in $file")
    }

    private fun status(key: String): Long = File("/proc/self/status").readLines()
        .firstOrNull { it.startsWith(key) }?.split(Regex("\\s+"))?.get(1)?.toLong() ?: -1

    private fun rssKb() = status("VmRSS:")
    private fun hwmKb() = status("VmHWM:")
}
