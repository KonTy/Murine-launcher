// SPDX-License-Identifier: Apache-2.0
package app.murinelauncher.voice

import java.io.File
import java.util.Locale

/** Inference settings derived from the device and the audio. */
object WhisperTuning {
    const val SAMPLE_RATE = 16_000

    /** One encoder position covers 20 ms of audio (10 ms mel hop, stride-2 convolution). */
    private const val SAMPLES_PER_POSITION = 320
    /** Whisper's fixed window: 30 s of audio. */
    const val FULL_AUDIO_CONTEXT = 1500
    /** Never shrink the encoder below 3 s; very short contexts hurt accuracy on single words. */
    private const val MIN_AUDIO_CONTEXT = 150
    /** About 1 s of trailing context past the end of the speech. */
    private const val AUDIO_CONTEXT_HEADROOM = 50

    /**
     * Encoder context for [samples] of audio. For models fine-tuned for dynamic audio context
     * (FUTO's ACFT models) the encoder only processes about as much audio as there is, which is
     * what makes short queries fast; other models get Whisper's default full context (returns 0).
     */
    fun audioContext(samples: Int, dynamic: Boolean): Int {
        if (!dynamic) return 0
        val positions = (samples + SAMPLES_PER_POSITION - 1) / SAMPLES_PER_POSITION
        return (positions + AUDIO_CONTEXT_HEADROOM).coerceIn(MIN_AUDIO_CONTEXT, FULL_AUDIO_CONTEXT)
    }

    /** Upper bound on decoded tokens: a search query, not a dictation. */
    const val MAX_TOKENS = 96

    /**
     * Worker threads for inference: the performance cores only, at most 4. ggml splits each
     * operation evenly across its threads, so one thread on a slow efficiency core holds back the
     * others; and leaving cores free keeps the launcher's UI smooth while transcribing.
     */
    fun threadCount(
        cpuCount: Int = Runtime.getRuntime().availableProcessors(),
        maxFrequencies: List<Long> = readMaxFrequencies(cpuCount),
    ): Int {
        val cpus = cpuCount.coerceAtLeast(1)
        val known = maxFrequencies.filter { it > 0 }
        val lowest = known.minOrNull()
        val performanceCores = if (known.size == cpus && lowest != null && known.any { it > lowest }) {
            // Everything above the slowest cluster
            known.count { it > lowest }
        } else {
            cpus / 2
        }
        return performanceCores.coerceIn(minOf(2, cpus), minOf(4, cpus))
    }

    private fun readMaxFrequencies(cpuCount: Int): List<Long> = (0 until cpuCount).map { cpu ->
        try {
            File("/sys/devices/system/cpu/cpu$cpu/cpufreq/cpuinfo_max_freq").readText().trim().toLong()
        } catch (_: Exception) {
            0L
        }
    }

    /**
     * Languages the multilingual tiny/base/small checkpoints know, as whisper.cpp codes
     * (Cantonese is left out: it only exists in the large-v3 vocabulary).
     */
    val LANGUAGES: List<String> = listOf(
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv",
        "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
        "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr",
        "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
        "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu",
        "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
        "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
    )

    /** Preference value meaning "the device language". */
    const val LANGUAGE_DEVICE = ""
    /** Preference value meaning "let the model detect the language" (unreliable on short audio). */
    const val LANGUAGE_AUTO = "auto"

    /** Whisper code for [locale]'s language, or null if the models do not know it. */
    fun whisperLanguage(locale: Locale): String? {
        val code = when (val lang = locale.language.lowercase(Locale.ROOT)) {
            "iw" -> "he"
            "in" -> "id"
            "ji" -> "yi"
            "jv" -> "jw"
            "nb" -> "no"
            "fil" -> "tl"
            "yue" -> "zh"
            else -> lang
        }
        return code.takeIf { it in LANGUAGES }
    }

    /**
     * The language passed to whisper.cpp. English-only models are always told English; for
     * multilingual models the preference (device language by default) is resolved, falling
     * back to auto-detection when the device language is not supported.
     */
    fun resolveLanguage(preference: String, multilingual: Boolean, locale: Locale = Locale.getDefault()): String =
        when {
            !multilingual -> "en"
            preference == LANGUAGE_AUTO -> LANGUAGE_AUTO
            preference != LANGUAGE_DEVICE && preference in LANGUAGES -> preference
            else -> whisperLanguage(locale) ?: LANGUAGE_AUTO
        }
}
