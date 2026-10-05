// SPDX-License-Identifier: Apache-2.0
package app.murinelauncher.voice

import kotlin.math.log10
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Energy-based end-of-utterance detector for 16 kHz mono PCM, fed in fixed 20 ms frames.
 *
 * The background level is the quietest stretch of the last couple of seconds (speech always has
 * gaps between words, steady noise does not), so it adapts to a noisy room without mistaking a
 * long sentence for background. A frame is speech when it is clearly louder than that.
 * It is deliberately simple: the model does the recognition, this only decides when to stop.
 */
class EndpointDetector(private val config: Config = Config()) {

    data class Config(
        /** Frames needed above the threshold, in a row, to count as speech onset. */
        val onsetFrames: Int = 3,
        /** Silence after speech that ends the utterance. */
        val trailingSilenceMs: Int = 900,
        /** Minimum speech before trailing silence may end the utterance. */
        val minSpeechMs: Int = 200,
        /** Give up when nobody speaks for this long. */
        val noSpeechTimeoutMs: Int = 6_000,
        /** Hard cap on the utterance length. */
        val maxUtteranceMs: Int = 20_000,
        /** How much louder than the background a frame must be, in dB. */
        val speechMarginDb: Double = 12.0,
        /** Frames quieter than this are never speech, whatever the background. */
        val absoluteFloorDbfs: Double = -55.0,
        /** Assumed background before there is enough audio to measure it (a quiet room). */
        val priorNoiseDbfs: Double = -50.0,
        /** Window the background level is measured over. */
        val noiseWindowMs: Int = 2_000,
        /** Exact digital silence for this long means the microphone is muted or taken. */
        val deadInputMs: Int = 1_000,
    )

    enum class Result { WAITING, SPEECH, END_OF_SPEECH, NO_SPEECH, MAX_LENGTH, DEAD_INPUT }

    private val windowFrames = (config.noiseWindowMs / FRAME_MS).coerceAtLeast(1)
    private val noiseWindow = DoubleArray(windowFrames)
    private var noiseCount = 0
    private val recentPower = DoubleArray(SMOOTHING_FRAMES)

    private var frames = 0
    private var aboveRun = 0
    private var speechFrames = 0
    private var silenceRun = 0
    private var zeroRun = 0

    /** Index of the first frame of speech, or -1. */
    var onsetFrame = -1
        private set

    /** Index of the last frame that counted as speech, or -1. */
    var lastSpeechFrame = -1
        private set

    /** Level of the last frame, 0..1, for the listening indicator. */
    var level = 0f
        private set

    val speechStarted: Boolean get() = onsetFrame >= 0

    /** Feeds one frame of [length] samples (normally [FRAME_SAMPLES]) starting at [offset]. */
    fun feed(samples: ShortArray, offset: Int = 0, length: Int = FRAME_SAMPLES): Result {
        var sum = 0.0
        var allZero = true
        for (i in offset until offset + length) {
            val s = samples[i].toInt()
            if (s != 0) allZero = false
            sum += (s * s).toDouble()
        }
        val power = sum / max(length, 1)
        val db = toDbfs(power)
        level = ((db + 60.0) / 50.0).coerceIn(0.0, 1.0).toFloat()
        val index = frames++

        zeroRun = if (allZero) zeroRun + 1 else 0
        if (!speechStarted && zeroRun * FRAME_MS >= config.deadInputMs) return Result.DEAD_INPUT

        // Background: minimum of a short moving average over the window; digital silence (mic
        // warm-up) says nothing about the room and is skipped.
        recentPower[index % SMOOTHING_FRAMES] = power
        if (!allZero) {
            val smoothed = toDbfs(recentPower.average())
            noiseWindow[noiseCount++ % windowFrames] = smoothed
        }
        var noise = config.priorNoiseDbfs
        if (noiseCount > 0) {
            var min = Double.MAX_VALUE
            for (i in 0 until minOf(noiseCount, windowFrames)) min = minOf(min, noiseWindow[i])
            // Until the window has filled, the prior caps the estimate: someone talking from the
            // very first frame must not be taken for the background
            noise = if (noiseCount < windowFrames) minOf(min, config.priorNoiseDbfs) else min
        }
        val threshold = max(noise + config.speechMarginDb, config.absoluteFloorDbfs)

        if (db > threshold) {
            aboveRun++
            silenceRun = 0
            speechFrames++
            lastSpeechFrame = index
            if (!speechStarted && aboveRun >= config.onsetFrames) onsetFrame = index - aboveRun + 1
        } else {
            aboveRun = 0
            silenceRun++
        }

        val elapsedMs = frames * FRAME_MS
        return when {
            elapsedMs >= config.maxUtteranceMs -> if (speechStarted) Result.MAX_LENGTH else Result.NO_SPEECH
            !speechStarted -> if (elapsedMs >= config.noSpeechTimeoutMs) Result.NO_SPEECH else Result.WAITING
            speechFrames * FRAME_MS >= config.minSpeechMs &&
                silenceRun * FRAME_MS >= config.trailingSilenceMs -> Result.END_OF_SPEECH
            else -> Result.SPEECH
        }
    }

    private fun toDbfs(power: Double): Double {
        val rms = sqrt(power)
        return if (rms < 1.0) -96.0 else 20 * log10(rms / 32768.0)
    }

    companion object {
        const val FRAME_MS = 20
        const val FRAME_SAMPLES = WhisperTuning.SAMPLE_RATE * FRAME_MS / 1000
        private const val SMOOTHING_FRAMES = 3
    }
}
