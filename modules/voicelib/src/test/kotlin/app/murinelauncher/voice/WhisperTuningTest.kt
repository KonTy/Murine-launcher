// SPDX-License-Identifier: Apache-2.0
package app.murinelauncher.voice

import java.util.Locale
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class WhisperTuningTest {

    @Test
    fun audioContextFollowsTheAudioForTunedModels() {
        // 1 s of audio -> 50 positions + 1 s headroom, raised to the 3 s floor
        assertEquals(150, WhisperTuning.audioContext(16_000, dynamic = true))
        // 5 s -> 250 + 50
        assertEquals(300, WhisperTuning.audioContext(80_000, dynamic = true))
        assertEquals(301, WhisperTuning.audioContext(80_001, dynamic = true))
        // Never beyond Whisper's 30 s window
        assertEquals(1500, WhisperTuning.audioContext(16_000 * 40, dynamic = true))
    }

    @Test
    fun otherModelsKeepTheFullContext() {
        assertEquals(0, WhisperTuning.audioContext(16_000, dynamic = false))
    }

    @Test
    fun threadsUsePerformanceCoresOnly() {
        val khz = 1000L
        // 4 big + 4 little
        assertEquals(4, WhisperTuning.threadCount(8, List(4) { 2_400 * khz } + List(4) { 2_000 * khz }))
        // 2 big + 6 little
        assertEquals(2, WhisperTuning.threadCount(8, List(2) { 2_200 * khz } + List(6) { 1_800 * khz }))
        // 1 prime + 4 big + 3 little
        assertEquals(4, WhisperTuning.threadCount(8,
            listOf(3_200 * khz) + List(4) { 2_800 * khz } + List(3) { 2_000 * khz }))
        // Single cluster: half the cores
        assertEquals(4, WhisperTuning.threadCount(8, List(8) { 1_800 * khz }))
        assertEquals(2, WhisperTuning.threadCount(4, List(4) { 1_800 * khz }))
        // Unreadable frequencies
        assertEquals(3, WhisperTuning.threadCount(6, List(6) { 0L }))
        assertEquals(1, WhisperTuning.threadCount(1, emptyList()))
        assertEquals(4, WhisperTuning.threadCount(16, emptyList()))
    }

    @Test
    fun englishModelsAlwaysUseEnglish() {
        assertEquals("en", WhisperTuning.resolveLanguage("it", multilingual = false, Locale.ITALIAN))
        assertEquals("en", WhisperTuning.resolveLanguage(WhisperTuning.LANGUAGE_AUTO, false, Locale.ITALIAN))
    }

    @Test
    fun multilingualModelsDefaultToTheDeviceLanguage() {
        assertEquals("it", WhisperTuning.resolveLanguage(WhisperTuning.LANGUAGE_DEVICE, true, Locale.ITALY))
        assertEquals("de", WhisperTuning.resolveLanguage("de", true, Locale.ITALY))
        assertEquals("auto", WhisperTuning.resolveLanguage(WhisperTuning.LANGUAGE_AUTO, true, Locale.ITALY))
        // Unsupported device language: detect
        assertEquals("auto", WhisperTuning.resolveLanguage("", true, Locale.forLanguageTag("chr")))
        // Unknown stored value: device language
        assertEquals("it", WhisperTuning.resolveLanguage("xx", true, Locale.ITALY))
    }

    @Test
    fun legacyAndRegionalCodesAreMapped() {
        assertEquals("he", WhisperTuning.whisperLanguage(Locale.forLanguageTag("iw")))
        assertEquals("id", WhisperTuning.whisperLanguage(Locale.forLanguageTag("in")))
        assertEquals("no", WhisperTuning.whisperLanguage(Locale.forLanguageTag("nb-NO")))
        assertEquals("tl", WhisperTuning.whisperLanguage(Locale.forLanguageTag("fil")))
        assertEquals("pt", WhisperTuning.whisperLanguage(Locale.forLanguageTag("pt-BR")))
        assertNull(WhisperTuning.whisperLanguage(Locale.forLanguageTag("chr")))
    }
}
